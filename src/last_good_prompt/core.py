"""A bounded prompt/model tournament with exact provider-cost accounting."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
from typing import Any, Callable, Iterable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models?output_modalities=text&sort=intelligence-high-to-low"
OPENROUTER_ZDR_ENDPOINTS_URL = "https://openrouter.ai/api/v1/endpoints/zdr"


class ProviderError(RuntimeError):
    """The model provider could not complete a request."""


class BudgetExceeded(RuntimeError):
    """A new provider call would exceed the customer-approved hard cap."""


@dataclass(frozen=True)
class Turn:
    input: str
    approved_output: str


@dataclass(frozen=True)
class Example:
    input: str
    approved_output: str
    id: str = ""
    turns: tuple[Turn, ...] = ()
    group: str = ""
    difficulty: str = "typical"
    weight: float = 1.0
    critical: bool = False

    def conversation(self) -> tuple[Turn, ...]:
        return self.turns or (Turn(self.input, self.approved_output),)

    @classmethod
    def from_value(cls, value: Example | dict[str, Any], index: int = 0) -> Example:
        if isinstance(value, cls):
            return value
        turns = tuple(
            Turn(str(item.get("input", "")).strip(), str(item.get("approved_output", item.get("expected", ""))).strip())
            for item in value.get("turns", [])
        )
        first = turns[0] if turns else Turn(str(value.get("input", "")).strip(), str(value.get("approved_output", "")).strip())
        return cls(
            first.input,
            turns[-1].approved_output if turns else first.approved_output,
            str(value.get("id", "")).strip() or f"example-{index + 1}",
            turns,
            str(value.get("group", "")).strip(),
            str(value.get("difficulty", "typical")).strip() or "typical",
            float(value.get("weight", 1.0)),
            bool(value.get("critical", False)),
        )


@dataclass(frozen=True)
class Completion:
    content: str
    model: str
    generation_id: str
    cost_usd: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0


@dataclass(frozen=True)
class Judgment:
    passed: bool
    score: float
    reason: str


@dataclass(frozen=True)
class CaseResult:
    example_id: str
    split: str
    prompt_kind: str
    output: str
    approved_output: str
    passed: bool
    score: float
    reason: str
    target_cost_usd: float
    evaluator_cost_usd: float
    target_generation_id: str
    evaluator_generation_id: str
    target_latency_ms: int = 0
    evaluator_latency_ms: int = 0
    group: str = ""
    difficulty: str = "typical"
    weight: float = 1.0
    critical: bool = False


@dataclass
class ModelResult:
    model: str
    selected_prompt: str
    baseline_pass_rate: float
    selected_pass_rate: float
    holdout_pass_rate: float
    baseline_holdout_pass_rate: float
    estimated_production_cost_per_call_usd: float
    estimated_cost_per_successful_call_usd: float
    optimization_spend_usd: float
    passed_quality_floor: bool
    target_latency_p50_ms: int = 0
    target_latency_p90_ms: int = 0
    passed_latency_ceiling: bool = True
    holdout_unique_scenarios: int = 0
    holdout_executions: int = 0
    holdout_execution_pass_rate: float = 0.0
    baseline_holdout_execution_pass_rate: float = 0.0
    holdout_pass_rates_by_difficulty: dict[str, float] = field(default_factory=dict)
    passed_difficulty_floors: bool = True
    prompt_origin: str = "starting_prompt"
    few_shot_example_ids: list[str] = field(default_factory=list)
    few_shot_provenance: list[dict[str, Any]] = field(default_factory=list)
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationResult:
    objective: str
    quality_threshold: float
    exploratory: bool
    winner: ModelResult
    models: list[ModelResult]
    total_provider_spend_usd: float
    warnings: list[str]
    quality_frontier: list[dict[str, Any]]
    diminishing_returns: dict[str, Any]
    regression_suite: dict[str, Any]
    elapsed_seconds: float = 0.0
    comparison_integrity: dict[str, Any] = field(default_factory=dict)
    omitted_configurations: list[dict[str, str]] = field(default_factory=list)
    unavailable_models: list[dict[str, str]] = field(default_factory=list)
    incomplete_models: list[dict[str, str]] = field(default_factory=list)
    skipped_budget_models: list[str] = field(default_factory=list)
    pruned_models: list[str] = field(default_factory=list)
    screening_results: list[dict[str, Any]] = field(default_factory=list)
    winner_scope: str = "Best among every requested target"
    quality_gate_status: str = "QUALIFIED_ROUTE_SELECTED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def save_regression_suite(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.regression_suite, handle, indent=2, ensure_ascii=False)
            handle.write("\n")


@dataclass(frozen=True)
class DraftAnswer:
    task: str
    input: str
    answer: str
    model: str
    provider_cost_usd: float

    def approve(self, example_id: str = "example-1") -> Example:
        return Example(self.input, self.answer, example_id)

    def correct(self, approved_output: str, example_id: str = "example-1") -> Example:
        corrected = str(approved_output).strip()
        if not corrected:
            raise ValueError("A corrected answer cannot be blank.")
        return Example(self.input, corrected, example_id)


class ChatTransport(Protocol):
    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> Completion: ...

    def estimate_cost(
        self, model: str, messages: list[dict[str, str]], *, max_tokens: int
    ) -> float: ...


class OpenRouterTransport:
    """Small dependency-free OpenRouter client with current model-price lookup."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout_seconds: float = 600,
        app_url: str = "https://evalt.dev",
        catalog_ttl_seconds: float = 3600,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        if not api_key and not os.environ.get("OPENROUTER_API_KEY"):
            dotenv_path = Path.cwd() / ".env"
            if dotenv_path.is_file():
                load_dotenv(dotenv_path=dotenv_path, override=False)
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Set OPENROUTER_API_KEY in the environment or in a .env file in the current "
                "working directory, or pass api_key= to Evalt()."
            )
        self.set_timeout_seconds(timeout_seconds)
        self._app_url = app_url
        self._opener = opener
        self._catalog_ttl_seconds = max(0.0, float(catalog_ttl_seconds))
        self._catalog_loaded_at = 0.0
        self._prices: dict[str, tuple[float, float]] | None = None
        self._supported_parameters: dict[str, set[str]] = {}
        self._reasoning: dict[str, dict[str, Any]] = {}
        self._limits: dict[str, tuple[int | None, int | None]] = {}
        self._providers: dict[str, list[str]] = {}
        self._provider_route_names: dict[str, dict[str, str]] = {}
        self._catalog_items: list[dict[str, Any]] = []
        self._preferred_max_latency_seconds: float | None = None
        self._provider_sort = "price"

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    def set_timeout_seconds(self, value: float) -> None:
        resolved = float(value)
        if not 0 < resolved <= 7200:
            raise ValueError("timeout_seconds must be greater than zero and no more than 7200 seconds.")
        self._timeout = resolved

    def set_performance_policy(
        self,
        *,
        preferred_max_latency_seconds: float | None = None,
        provider_sort: str = "price",
    ) -> None:
        """Set current provider-performance preferences for subsequent calls."""
        if provider_sort not in {"price", "latency", "throughput"}:
            raise ValueError("provider_sort must be price, latency, or throughput.")
        if preferred_max_latency_seconds is not None and preferred_max_latency_seconds <= 0:
            raise ValueError("preferred_max_latency_seconds must be positive when provided.")
        self._provider_sort = provider_sort
        self._preferred_max_latency_seconds = (
            float(preferred_max_latency_seconds)
            if preferred_max_latency_seconds is not None else None
        )

    def _request(self, url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "HTTP-Referer": self._app_url,
            "X-Title": "Evalt Python SDK",
        }
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
        started = time.monotonic()
        try:
            with self._opener(request, timeout=self._timeout) as response:
                if not hasattr(response, "read1"):
                    raw = response.read()
                else:
                    chunks: list[bytes] = []
                    while True:
                        if time.monotonic() - started > self._timeout:
                            raise TimeoutError(
                                f"provider response exceeded the {self._timeout:g}s total deadline"
                            )
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                return json.loads(raw.decode("utf-8"))
        except HTTPError as error:
            raw_detail = error.read().decode("utf-8", errors="replace")
            detail = _safe_provider_error_detail(raw_detail)
            provider_error = ProviderError(f"OpenRouter returned HTTP {error.code}: {detail}")
            provider_error.status_code = int(error.code)
            try:
                payload = json.loads(raw_detail)
                provider_error.provider_name = str(
                    ((payload.get("error") or {}).get("metadata") or {}).get("provider_name")
                    or payload.get("provider")
                    or ""
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                provider_error.provider_name = ""
            raise provider_error from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ProviderError(f"OpenRouter request failed: {error}") from error

    def _load_prices(self) -> dict[str, tuple[float, float]]:
        if self._prices is not None and time.monotonic() - self._catalog_loaded_at < self._catalog_ttl_seconds:
            return self._prices
        payload = self._request(OPENROUTER_MODELS_URL)
        endpoint_catalog_available = True
        try:
            endpoint_payload = self._request(OPENROUTER_ZDR_ENDPOINTS_URL)
        except ProviderError:
            endpoint_catalog_available = False
            endpoint_payload = {"data": []}
        endpoint_candidates: dict[str, list[dict[str, Any]]] = {}
        recognized_endpoints = 0
        for endpoint in endpoint_payload.get("data", []):
            model_id = str(endpoint.get("model_id") or "")
            parameters = {str(value) for value in endpoint.get("supported_parameters") or []}
            if model_id and ({"max_tokens", "max_completion_tokens"} & parameters):
                endpoint_candidates.setdefault(model_id, []).append(endpoint)
                recognized_endpoints += 1
        if endpoint_catalog_available and endpoint_payload.get("data") and not recognized_endpoints:
            # Test doubles and legacy gateways may return model-shaped data at the
            # endpoint URL. Do not mistake that malformed catalog for a proof that
            # every ZDR route disappeared.
            endpoint_catalog_available = False
        prices: dict[str, tuple[float, float]] = {}
        catalog_items: list[dict[str, Any]] = []
        self._supported_parameters = {}
        self._reasoning = {}
        self._limits = {}
        self._providers = {}
        self._provider_route_names = {}
        for item in payload.get("data", []):
            pricing = item.get("pricing") or {}
            try:
                model_id = str(item["id"])
                endpoints = endpoint_candidates.get(model_id, [])
                if endpoint_catalog_available and not endpoints:
                    # A model-level listing is not an executable private route. Omit
                    # it before spend instead of launching a guaranteed ZDR 404.
                    continue
                top_provider = item.get("top_provider") or {}

                def endpoint_capacity(value: dict[str, Any]) -> int:
                    raw = value.get("max_completion_tokens") or value.get("context_length") or 0
                    try:
                        return max(0, int(raw))
                    except (TypeError, ValueError):
                        return 0

                def endpoint_price(value: dict[str, Any]) -> float:
                    route = value.get("pricing") or {}
                    return float(route.get("prompt") or "inf") + float(route.get("completion") or "inf")

                model_ceiling_raw = top_provider.get("max_completion_tokens") or item.get("context_length") or 131072
                try:
                    preferred_capacity = min(131072, max(1, int(model_ceiling_raw)))
                except (TypeError, ValueError):
                    preferred_capacity = 131072
                capacity_eligible = [value for value in endpoints if endpoint_capacity(value) >= preferred_capacity]

                def endpoint_contract_score(value: dict[str, Any]) -> int:
                    parameters = {str(item) for item in value.get("supported_parameters") or []}
                    # Structured output prevents correct answers from failing an
                    # exact-JSON task merely because a provider wrapped them in
                    # prose. Adjustable reasoning is the next most useful search
                    # lever. Prefer the strongest executable contract first, then
                    # choose the cheapest route that provides it.
                    return (
                        2 * int(bool({"structured_outputs", "response_format"} & parameters))
                        + int(bool({"reasoning", "reasoning_effort"} & parameters))
                    )

                if capacity_eligible:
                    best_contract = max(endpoint_contract_score(value) for value in capacity_eligible)
                    provider_pool = sorted(
                        (value for value in capacity_eligible if endpoint_contract_score(value) == best_contract),
                        key=endpoint_price,
                    )
                    endpoint = provider_pool[0]
                elif endpoints:
                    maximum_capacity = max(endpoint_capacity(value) for value in endpoints)
                    maximum_capacity_pool = [
                        value for value in endpoints if endpoint_capacity(value) == maximum_capacity
                    ]
                    best_contract = max(endpoint_contract_score(value) for value in maximum_capacity_pool)
                    provider_pool = sorted(
                        (value for value in maximum_capacity_pool if endpoint_contract_score(value) == best_contract),
                        key=endpoint_price,
                    )
                    endpoint = provider_pool[0]
                else:
                    provider_pool = []
                    endpoint = None
                route_pricing = (endpoint or {}).get("pricing") or pricing
                prices[model_id] = (
                    float(route_pricing.get("prompt") or 0),
                    float(route_pricing.get("completion") or 0),
                )
                supported_source = (endpoint or {}).get("supported_parameters") or item.get("supported_parameters") or []
                self._supported_parameters[model_id] = {str(value) for value in supported_source}
                if not endpoint_catalog_available:
                    # Model-level metadata cannot prove that the selected private endpoint
                    # accepts optional reasoning controls, so fail closed on that lever.
                    self._supported_parameters[model_id] -= {"reasoning", "reasoning_effort"}
                self._reasoning[model_id] = dict(item.get("reasoning") or {})
                context_value = (endpoint or {}).get("context_length") or top_provider.get("context_length") or item.get("context_length")
                completion_value = (endpoint or {}).get("max_completion_tokens") or top_provider.get("max_completion_tokens") or context_value
                try:
                    context_limit = int(context_value) if context_value is not None else None
                except (TypeError, ValueError):
                    context_limit = None
                try:
                    completion_limit = int(completion_value) if completion_value is not None else None
                except (TypeError, ValueError):
                    completion_limit = None
                self._limits[model_id] = (context_limit, completion_limit)
                provider_tags = [str(value["tag"]) for value in provider_pool if value.get("tag")]
                if provider_tags:
                    self._providers[model_id] = provider_tags[:3]
                    self._provider_route_names[model_id] = {
                        str(value["tag"]): str(value.get("provider_name") or "")
                        for value in provider_pool[:3] if value.get("tag")
                    }
                intelligence = ((item.get("benchmarks") or {}).get("artificial_analysis") or {}).get("intelligence_index")
                try:
                    intelligence = float(intelligence) if intelligence is not None else None
                except (TypeError, ValueError):
                    intelligence = None
                catalog_items.append({
                    "id": str(item["id"]),
                    "intelligence": intelligence,
                    "blended_price": prices[model_id][0] * 1_000_000 + prices[model_id][1] * 2_000_000,
                    "supported_parameters": sorted(self._supported_parameters[model_id]),
                    "reasoning": item.get("reasoning") or {},
                    "context_length": context_limit,
                    "max_completion_tokens": completion_limit,
                    "private_provider_routes": len(provider_tags),
                })
            except (KeyError, TypeError, ValueError):
                continue
        self._prices = prices
        self._catalog_items = catalog_items
        self._catalog_loaded_at = time.monotonic()
        return prices

    def model_catalog(self) -> list[dict[str, Any]]:
        """Return the current provider catalog used for role shortlisting.

        Intelligence metadata never qualifies a production route by itself; Evalt's
        task-specific holdouts remain the promotion gate.
        """
        self._load_prices()
        return [dict(item) for item in self._catalog_items]

    def configuration_support(self, configuration: str) -> dict[str, Any]:
        """Preflight a model/effort pair against current routed capabilities."""
        explicit_reasoning = "#reasoning=" in configuration
        model, effort = self._split_configuration(configuration)
        prices = self._load_prices()
        if model not in prices:
            return {
                "supported": False,
                "reason": f"OpenRouter did not return a current priced ZDR route for {model!r}.",
            }
        supported = self._supported_parameters.get(model, set())
        reasoning = self._reasoning.get(model, {})
        reasoning_supported = bool({"reasoning", "reasoning_effort"} & supported)
        if explicit_reasoning and effort == "none" and reasoning.get("mandatory"):
            return {
                "supported": False,
                "reason": f"Reasoning is mandatory for {model!r}; the no-reasoning configuration was omitted before spend.",
            }
        if explicit_reasoning and effort != "none" and not reasoning_supported:
            return {
                "supported": False,
                "reason": f"The current ZDR route for {model!r} does not support adjustable reasoning; this configuration was omitted before spend.",
            }
        supported_efforts = {
            str(value) for value in reasoning.get("supported_efforts") or []
        }
        if explicit_reasoning and effort != "none" and supported_efforts and effort not in supported_efforts:
            return {
                "supported": False,
                "reason": f"The current ZDR route for {model!r} does not list reasoning effort {effort!r}; this configuration was omitted before spend.",
            }
        return {
            "supported": True,
            "reason": "Current routed capability metadata accepts this configuration.",
        }

    def estimate_cost(
        self, model: str, messages: list[dict[str, str]], *, max_tokens: int
    ) -> float:
        explicit_reasoning = "#reasoning=" in model
        model, effort = self._split_configuration(model)
        prices = self._load_prices()
        reasoning_metadata = self._reasoning.get(model, {})
        if not explicit_reasoning and reasoning_metadata.get("mandatory"):
            effort = str(reasoning_metadata.get("default_effort") or "medium")
        max_tokens = self._bounded_output_tokens(model, messages, max_tokens, effort)
        if model not in prices:
            raise ProviderError(
                f"OpenRouter did not return current pricing for model {model!r}; "
                "the SDK will not start an unpriced call."
            )
        prompt_price, completion_price = prices[model]
        estimated_prompt_tokens = max(1, len(json.dumps(messages)) // 3)
        return estimated_prompt_tokens * prompt_price + max_tokens * completion_price

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> Completion:
        configuration = model
        explicit_reasoning = "#reasoning=" in model
        model, reasoning_effort = self._split_configuration(model)
        self._load_prices()
        reasoning_metadata = self._reasoning.get(model, {})
        if not explicit_reasoning and reasoning_metadata.get("mandatory"):
            reasoning_effort = str(reasoning_metadata.get("default_effort") or "medium")
        max_tokens = self._bounded_output_tokens(model, messages, max_tokens, reasoning_effort)
        supported = self._supported_parameters.get(model, set())
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "provider": {
                "zdr": True,
                "data_collection": "deny",
                "require_parameters": True,
                "sort": self._provider_sort,
            },
            "usage": {"include": True},
        }
        if self._providers.get(model):
            body["provider"].update({
                "only": self._providers[model],
                "allow_fallbacks": len(self._providers[model]) > 1,
            })
        if self._preferred_max_latency_seconds is not None:
            body["provider"]["preferred_max_latency"] = {
                "p90": self._preferred_max_latency_seconds
            }
        if "max_completion_tokens" in supported:
            body["max_completion_tokens"] = max_tokens
        elif "max_tokens" in supported:
            body["max_tokens"] = max_tokens
        else:
            raise ProviderError(
                f"OpenRouter did not report a bounded output-token parameter for {model!r}."
            )
        if "temperature" in supported:
            body["temperature"] = 0
        reasoning_supported = bool({"reasoning", "reasoning_effort"} & supported)
        if explicit_reasoning and reasoning_effort != "none" and not reasoning_supported:
            raise ProviderError(f"The current ZDR endpoint for {model!r} does not support adjustable reasoning.")
        if explicit_reasoning and reasoning_effort == "none" and reasoning_metadata.get("mandatory"):
            raise ProviderError(f"Reasoning is mandatory for {model!r}; choose low, medium, high, or another model.")
        supported_efforts = {str(value) for value in reasoning_metadata.get("supported_efforts") or []}
        if explicit_reasoning and reasoning_effort != "none" and supported_efforts and reasoning_effort not in supported_efforts:
            raise ProviderError(
                f"The current ZDR endpoint for {model!r} does not list reasoning effort {reasoning_effort!r}."
            )
        if reasoning_supported:
            body["reasoning"] = {"effort": reasoning_effort, "exclude": True}
        if response_schema and ({"structured_outputs", "response_format"} & supported):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "evalt_result",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        started = time.monotonic()
        fallback_spend_usd = 0.0
        while True:
            try:
                payload = self._request(OPENROUTER_CHAT_URL, body)
            except ProviderError as error:
                failed_provider = str(getattr(error, "provider_name", "") or "")
                route_names = self._provider_route_names.get(model, {})
                current_routes = list(body["provider"].get("only") or [])
                remaining_routes = [
                    tag for tag in current_routes
                    if not failed_provider or route_names.get(tag) != failed_provider
                ]
                if (
                    getattr(error, "status_code", None) != 429
                    or not failed_provider
                    or not remaining_routes
                    or remaining_routes == current_routes
                ):
                    raise
                # OpenRouter can surface one provider's 429 even with fallbacks
                # enabled. Retry the same model/configuration against the already
                # preflighted remaining private routes, never against a new model.
                body["provider"]["only"] = remaining_routes
                body["provider"]["allow_fallbacks"] = len(remaining_routes) > 1
                continue
            try:
                choice = payload["choices"][0]
                content = (choice.get("message") or {}).get("content")
                usage = payload.get("usage") or {}
                billed_cost = float(usage.get("cost") or 0)
                if choice.get("finish_reason") == "length":
                    error = ProviderError("OpenRouter reached the response limit before finishing.")
                    error.code = "PROVIDER_TRUNCATED"
                    expanded = self._bounded_output_tokens(model, messages, 131072, reasoning_effort)
                    error.retry_with_more_tokens = expanded > max_tokens
                    error.cost_usd = fallback_spend_usd + billed_cost
                    error.generation_id = str(payload.get("id") or "")
                    raise error
                if content is None or not str(content).strip():
                    # An empty answer is not evidence that more output allowance is
                    # needed. If OpenRouter identifies the serving provider, move to
                    # another already-preflighted compatible route at the same token
                    # ceiling. Preserve the failed route's reported cost.
                    failed_provider = str(payload.get("provider") or "")
                    route_names = self._provider_route_names.get(model, {})
                    current_routes = list(body["provider"].get("only") or [])
                    remaining_routes = [
                        tag for tag in current_routes
                        if not failed_provider
                        or (
                            tag.casefold() != failed_provider.casefold()
                            and route_names.get(tag, "").casefold() != failed_provider.casefold()
                        )
                    ]
                    if failed_provider and remaining_routes and remaining_routes != current_routes:
                        fallback_spend_usd += billed_cost
                        body["provider"]["only"] = remaining_routes
                        body["provider"]["allow_fallbacks"] = len(remaining_routes) > 1
                        continue
                    error = ProviderError("OpenRouter returned an empty completion for this model configuration.")
                    error.code = "PROVIDER_EMPTY"
                    error.retry_with_more_tokens = False
                    error.cost_usd = fallback_spend_usd + billed_cost
                    error.generation_id = str(payload.get("id") or "")
                    raise error
                return Completion(
                    content=str(content),
                    model=configuration,
                    generation_id=str(payload.get("id") or ""),
                    cost_usd=fallback_spend_usd + billed_cost,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    latency_ms=round((time.monotonic() - started) * 1000),
                )
            except (KeyError, IndexError, TypeError, ValueError) as error:
                detail = _safe_provider_error_detail(json.dumps(payload, ensure_ascii=False))
                raise ProviderError(f"OpenRouter returned an invalid completion payload: {detail}") from error

    @staticmethod
    def _split_configuration(configuration: str) -> tuple[str, str]:
        """Decode an auditable model + reasoning-effort candidate identifier."""
        marker = "#reasoning="
        if marker not in configuration:
            return configuration, "none"
        model, effort = configuration.rsplit(marker, 1)
        if effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"Unsupported reasoning effort {effort!r}.")
        return model, effort

    @staticmethod
    def _reasoning_token_ceiling(requested_tokens: int, effort: str) -> int:
        # OpenRouter counts hidden reasoning inside the completion allowance.  The
        # visible answer may be a two-field JSON object while a reasoning model still
        # consumes tens of thousands of tokens before emitting it.  Keep these
        # defaults deliberately generous so a paid evaluation is not thrown away by
        # an allowance tuned to visible output length.
        minimums = {
            "none": 0,
            "minimal": 32768,
            "low": 65536,
            "medium": 98304,
            "high": 131072,
            "xhigh": 131072,
            "max": 131072,
        }
        return max(1, int(requested_tokens), minimums.get(effort, 0))

    def _bounded_output_tokens(
        self,
        model: str,
        messages: list[dict[str, str]],
        requested_tokens: int,
        effort: str,
    ) -> int:
        desired = self._reasoning_token_ceiling(requested_tokens, effort)
        context_limit, completion_limit = self._limits.get(model, (None, None))
        # Use a conservative character/token ratio and reserve framing overhead so
        # Evalt never asks for a full context window on top of a non-empty prompt.
        estimated_prompt_tokens = max(1, math.ceil(len(json.dumps(messages)) / 3) + 128)
        limits = [desired]
        if completion_limit:
            limits.append(max(1, completion_limit))
        if context_limit:
            limits.append(max(1, context_limit - estimated_prompt_tokens))
        return max(1, min(limits))


class _Budget:
    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = float(limit_usd)
        self.spent_usd = 0.0
        self.reserved_usd = 0.0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def authorize(self, estimate_usd: float) -> None:
        estimate = float(estimate_usd)
        with self._condition:
            if estimate < 0:
                raise BudgetExceeded("A provider-cost estimate cannot be negative.")
            # Parallel calls reserve their worst-case completion allowance. A lane
            # that would fit after an in-flight reservation settles must wait, not
            # become a false budget failure. Once no reservation can free enough
            # room, the hard cap still fails closed before another provider call.
            while self.spent_usd + self.reserved_usd + estimate > self.limit_usd + 1e-12:
                if self.reserved_usd <= 1e-12:
                    raise BudgetExceeded(
                        f"The next estimated call would exceed the ${self.limit_usd:.4f} cap."
                    )
                self._condition.wait()
            self.reserved_usd += estimate

    def commit(self, actual_usd: float, reserved_estimate_usd: float = 0.0) -> None:
        actual = max(0.0, float(actual_usd))
        reserved = max(0.0, float(reserved_estimate_usd))
        with self._condition:
            self.reserved_usd = max(0.0, self.reserved_usd - reserved)
            self.spent_usd += actual
            exceeded = self.spent_usd + self.reserved_usd > self.limit_usd + 1e-12
            self._condition.notify_all()
            if exceeded:
                raise BudgetExceeded(
                    "The provider-reported cost exceeded the customer-approved hard cap."
                )

    def release(self, reserved_estimate_usd: float) -> None:
        with self._condition:
            self.reserved_usd = max(0.0, self.reserved_usd - max(0.0, float(reserved_estimate_usd)))
            self._condition.notify_all()


class _BudgetScope:
    """Charge one parallel lane to the shared cap while retaining lane-local spend."""

    def __init__(self, shared: _Budget) -> None:
        self.shared = shared
        self.limit_usd = shared.limit_usd
        self.spent_usd = 0.0

    def authorize(self, estimate_usd: float) -> None:
        self.shared.authorize(estimate_usd)

    def commit(self, actual_usd: float, reserved_estimate_usd: float = 0.0) -> None:
        self.shared.commit(actual_usd, reserved_estimate_usd)
        self.spent_usd += max(0.0, float(actual_usd))

    def release(self, reserved_estimate_usd: float) -> None:
        self.shared.release(reserved_estimate_usd)


class Client:
    """Optimize prompts and compare target models using customer-approved examples."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        transport: ChatTransport | None = None,
    ) -> None:
        self.transport = transport or OpenRouterTransport(api_key)

    def _call(
        self,
        budget: _Budget,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> Completion:
        for output_expansion_attempt in range(2):
            # A simple 2x retry was ineffective for reasoning models: 600 -> 1,200
            # was mapped to the same reasoning floor on both attempts.  A truncation
            # retry now requests a genuinely larger provider ceiling.
            attempt_max_tokens = (
                int(max_tokens)
                if output_expansion_attempt == 0
                else max(int(max_tokens) * 2, 131072)
            )
            estimate = self.transport.estimate_cost(model, messages, max_tokens=attempt_max_tokens)
            budget.authorize(estimate)
            try:
                completion = self.transport.complete(
                    model,
                    messages,
                    max_tokens=attempt_max_tokens,
                    response_schema=response_schema,
                )
            except ProviderError as error:
                billed_cost = float(getattr(error, "cost_usd", 0) or 0)
                if billed_cost:
                    budget.commit(billed_cost, estimate)
                else:
                    budget.release(estimate)
                if output_expansion_attempt == 0 and getattr(error, "code", "") == "PROVIDER_TRUNCATED" and getattr(error, "retry_with_more_tokens", True):
                    continue
                raise
            except Exception:
                budget.release(estimate)
                raise
            budget.commit(completion.cost_usd, estimate)
            return completion
        raise ProviderError("The provider did not return a complete answer after one budgeted expansion.")

    def draft_answer(
        self,
        *,
        task: str,
        input: str,
        model: str = "openai/gpt-5-mini",
        max_cost_usd: float = 0.10,
    ) -> DraftAnswer:
        task_text = str(task).strip()
        input_text = str(input).strip()
        if not task_text or not input_text:
            raise ValueError("Both task and input are required.")
        budget = _Budget(max_cost_usd)
        response = self._call(
            budget,
            model,
            [
                {"role": "system", "content": task_text},
                {"role": "user", "content": input_text},
            ],
            max_tokens=8192,
        )
        return DraftAnswer(task_text, input_text, response.content, response.model, response.cost_usd)

    def optimize(
        self,
        *,
        prompt: str,
        examples: Iterable[Example | dict[str, Any]],
        models: Iterable[str],
        optimizer_model: str = "openai/gpt-5.6-luna",
        evaluator_model: str = "openai/gpt-5.6-luna",
        objective: str = "cheapest_passing",
        quality_threshold: float = 0.95,
        max_optimization_cost_usd: float = 2.00,
        rounds: int = 3,
        optimize_prompt: bool = True,
        minimum_meaningful_quality_gain: float = 0.03,
        allow_few_shot: bool = True,
        max_few_shot_examples: int = 3,
        max_cost_per_run_usd: float | None = None,
        representative_input_chars: int | None = None,
        representative_output_tokens: int | None = None,
        incumbent_model: str | None = None,
        allowed_accuracy_regression: float = 0.0,
        adaptive_search: bool = False,
        holdout_repeats: int = 2,
        evaluator: dict[str, Any] | None = None,
        difficulty_thresholds: dict[str, float] | None = None,
        max_parallel_models: int = 16,
        max_parallel_scenarios: int = 32,
        max_p90_latency_seconds: float | None = None,
        latency_value_usd_per_second: float = 0.0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> OptimizationResult:
        optimization_started = time.monotonic()
        prompt_text = str(prompt).strip()
        model_list = list(dict.fromkeys(str(model).strip() for model in models if str(model).strip()))
        example_list = [Example.from_value(value, index) for index, value in enumerate(examples)]
        self._validate(
            prompt_text,
            example_list,
            model_list,
            quality_threshold,
            max_optimization_cost_usd,
            rounds,
            minimum_meaningful_quality_gain,
        )
        allowed_objectives = {
            "cheapest_passing", "cheapest_at_accuracy", "lowest_cost_at_accuracy",
            "highest_quality", "best_within_cost", "best_within_price", "constrained",
            "match_baseline_at_lowest_cost",
        }
        if objective not in allowed_objectives:
            raise ValueError(f"objective must be one of {sorted(allowed_objectives)}.")
        if max_cost_per_run_usd is not None and max_cost_per_run_usd <= 0:
            raise ValueError("max_cost_per_run_usd must be positive when provided.")
        if not 0 <= allowed_accuracy_regression < 1:
            raise ValueError("allowed_accuracy_regression must be between zero and one.")
        if not 1 <= int(holdout_repeats) <= 5:
            raise ValueError("holdout_repeats must be between one and five.")
        if not 1 <= int(max_parallel_models) <= 32:
            raise ValueError("max_parallel_models must be between one and thirty-two.")
        if not 1 <= int(max_parallel_scenarios) <= 128:
            raise ValueError("max_parallel_scenarios must be between one and one hundred twenty-eight.")
        if max_p90_latency_seconds is not None and max_p90_latency_seconds <= 0:
            raise ValueError("max_p90_latency_seconds must be positive when provided.")
        if latency_value_usd_per_second < 0:
            raise ValueError("latency_value_usd_per_second cannot be negative.")
        evaluator_policy = _validate_evaluator_policy(evaluator)
        difficulty_floor_policy = _validate_difficulty_thresholds(difficulty_thresholds)
        performance_setter = getattr(self.transport, "set_performance_policy", None)
        if callable(performance_setter):
            performance_setter(
                preferred_max_latency_seconds=max_p90_latency_seconds,
                provider_sort="latency" if latency_value_usd_per_second > 0 else "price",
            )
        objective = {
            "lowest_cost_at_accuracy": "cheapest_at_accuracy",
            "best_within_price": "best_within_cost",
        }.get(objective, objective)
        train, dev, holdout = _split_examples(example_list, prompt_text)
        budget = _Budget(max_optimization_cost_usd)
        results = []
        unavailable_models = []
        incomplete_models = []
        skipped_budget_models = []
        pruned_models = []
        screening_results: list[dict[str, Any]] = []
        screening_cases_by_model: dict[str, list[CaseResult]] = {}
        seed_prompt_by_model: dict[str, str] = {}
        seed_few_shot_ids_by_model: dict[str, list[str]] = {}
        prompt_origin_by_model: dict[str, str] = {}
        omitted_configurations = []
        progress_lock = threading.Lock()

        def emit_progress(event: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            # Multiple model lanes finish concurrently. Serialize callbacks so CLI
            # progress remains valid line-delimited JSON and consumer callbacks do
            # not need their own synchronization.
            with progress_lock:
                progress_callback(dict(event))

        catalog_loader = getattr(self.transport, "model_catalog", None)
        catalog_snapshot: list[dict[str, Any]] = []
        if callable(catalog_loader) and model_list:
            catalog_snapshot = list(catalog_loader() or [])
        catalog_intelligence: dict[str, float] = {}
        for catalog_item in catalog_snapshot:
            try:
                intelligence = catalog_item.get("intelligence")
                if intelligence is not None:
                    catalog_intelligence[str(catalog_item["id"])] = float(intelligence)
            except (KeyError, TypeError, ValueError):
                continue
        support_checker = getattr(self.transport, "configuration_support", None)
        if callable(support_checker):
            eligible_models: list[str] = []
            for configuration in model_list:
                support = support_checker(configuration)
                if support.get("supported"):
                    eligible_models.append(configuration)
                    continue
                omitted_configurations.append({
                    "model": configuration,
                    "reason": str(support.get("reason") or "Unsupported by current provider capability metadata."),
                    "stage": "preflight",
                })
                emit_progress({
                    "event": "configuration_omitted",
                    "model": configuration,
                    "reason": omitted_configurations[-1]["reason"],
                    "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
                })
            model_list = eligible_models
        if not model_list:
            reasons = "; ".join(
                f"{item['model']}: {item['reason']}" for item in omitted_configurations
            )
            raise ProviderError(
                "No requested configuration is compatible with current provider capability metadata. "
                + reasons
            )

        broad_models: list[str] = []
        if adaptive_search:
            seen_base_models: set[str] = set()
            hone_models: list[str] = []
            for configuration in model_list:
                base_model = configuration.split("#reasoning=", 1)[0]
                if base_model in seen_base_models:
                    hone_models.append(configuration)
                else:
                    seen_base_models.add(base_model)
                    broad_models.append(configuration)
            model_list = broad_models + hone_models
        def evaluate(model: str) -> ModelResult:
            emit_progress({
                "event": "model_started", "model": model,
                "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
            })
            model_budget = _BudgetScope(budget)
            return self._evaluate_model(
                prompt_text, train, dev, holdout, model, optimizer_model,
                evaluator_model, quality_threshold, model_budget, rounds,
                bool(allow_few_shot and optimize_prompt),
                max_few_shot_examples, representative_input_chars,
                representative_output_tokens, int(holdout_repeats), evaluator_policy,
                difficulty_floor_policy, int(max_parallel_scenarios),
                baseline_dev_cases=screening_cases_by_model.get(model),
                seed_prompt=seed_prompt_by_model.get(model),
                seed_few_shot_ids=seed_few_shot_ids_by_model.get(model),
                prompt_origin=prompt_origin_by_model.get(model, "starting_prompt"),
                optimize_prompt=bool(optimize_prompt),
            )

        def run_batch(configurations: list[str]) -> list[ModelResult]:
            if not configurations:
                return []
            if len(configurations) == 1 or int(max_parallel_models) == 1:
                completed_sequential: list[ModelResult] = []
                for model_index, model in enumerate(configurations):
                    try:
                        item = evaluate(model)
                        completed_sequential.append(item)
                        emit_progress({
                            "event": "model_completed", "model": model,
                            "final_test_pass_rate": item.holdout_pass_rate,
                            "passed_quality_floor": item.passed_quality_floor,
                            "target_latency_p50_ms": item.target_latency_p50_ms,
                            "target_latency_p90_ms": item.target_latency_p90_ms,
                            "optimization_spend_usd": item.optimization_spend_usd,
                            "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
                        })
                    except BudgetExceeded as error:
                        incomplete_models.append({"model": model, "reason": str(error)})
                        emit_progress({"event": "model_incomplete", "model": model, "reason": str(error)})
                        skipped_budget_models.extend(configurations[model_index + 1 :])
                        break
                    except ProviderError as error:
                        unavailable_models.append({"model": model, "reason": str(error)})
                        emit_progress({"event": "model_unavailable", "model": model, "reason": str(error)})
                return completed_sequential
            completed: dict[str, ModelResult] = {}
            with ThreadPoolExecutor(max_workers=min(int(max_parallel_models), len(configurations))) as pool:
                futures = {pool.submit(evaluate, model): model for model in configurations}
                for future in as_completed(futures):
                    model = futures[future]
                    try:
                        item = future.result()
                        completed[model] = item
                        emit_progress({
                            "event": "model_completed", "model": model,
                            "final_test_pass_rate": item.holdout_pass_rate,
                            "passed_quality_floor": item.passed_quality_floor,
                            "target_latency_p50_ms": item.target_latency_p50_ms,
                            "target_latency_p90_ms": item.target_latency_p90_ms,
                            "optimization_spend_usd": item.optimization_spend_usd,
                            "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
                        })
                    except BudgetExceeded as error:
                        incomplete_models.append({"model": model, "reason": str(error)})
                        emit_progress({"event": "model_incomplete", "model": model, "reason": str(error)})
                    except ProviderError as error:
                        unavailable_models.append({"model": model, "reason": str(error)})
                        emit_progress({"event": "model_unavailable", "model": model, "reason": str(error)})
            return [completed[model] for model in configurations if model in completed]

        def screen_model(model: str) -> dict[str, Any]:
            emit_progress({
                "event": "model_screen_started", "model": model,
                "screening_scenarios": len(dev),
                "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
            })
            model_budget = _BudgetScope(budget)
            cases = self._run_cases(
                prompt_text, "screening-baseline", dev, "dev", model,
                evaluator_model, model_budget, evaluator=evaluator_policy,
                max_parallel_scenarios=int(max_parallel_scenarios),
            )
            screening_cases_by_model[model] = cases
            costs = sorted(case.target_cost_usd for case in cases if case.target_cost_usd > 0)
            estimated_cost = (
                costs[max(0, min(len(costs) - 1, math.ceil(len(costs) * 0.90) - 1))]
                if costs else float("inf")
            )
            latencies = [case.target_latency_ms for case in cases if case.target_latency_ms > 0]
            result = {
                "model": model,
                "validation_pass_rate": round(_scenario_pass_rate(cases), 6),
                "validation_scenarios": len(dev),
                "estimated_production_cost_per_call_usd": round(estimated_cost, 10),
                "screening_spend_usd": round(model_budget.spent_usd, 10),
                "target_latency_p90_ms": round(_percentile(latencies, 0.90)) if latencies else 0,
                "status": "SCREENED",
            }
            emit_progress({
                "event": "model_screen_completed", **result,
                "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
            })
            return result

        def run_screening(configurations: list[str]) -> list[dict[str, Any]]:
            completed: dict[str, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=min(int(max_parallel_models), len(configurations))) as pool:
                futures = {pool.submit(screen_model, model): model for model in configurations}
                for future in as_completed(futures):
                    model = futures[future]
                    try:
                        completed[model] = future.result()
                    except BudgetExceeded as error:
                        incomplete_models.append({"model": model, "reason": str(error), "stage": "screening"})
                        emit_progress({"event": "model_incomplete", "model": model, "reason": str(error)})
                    except ProviderError as error:
                        unavailable_models.append({"model": model, "reason": str(error), "stage": "screening"})
                        emit_progress({"event": "model_unavailable", "model": model, "reason": str(error)})
            return [completed[model] for model in configurations if model in completed]

        if adaptive_search:
            sizing_input = representative_input_chars or max(
                1, int(_percentile([len(item.input) for item in example_list], 0.90))
            )
            sizing_output = representative_output_tokens or 64
            sizing_messages = [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": "x" * sizing_input},
            ]
            broad_estimates: dict[str, float] = {}
            for model in broad_models:
                try:
                    broad_estimates[model] = self.transport.estimate_cost(
                        model, sizing_messages, max_tokens=int(sizing_output)
                    )
                except ProviderError:
                    broad_estimates[model] = float("inf")
            ordered_broad = sorted(
                broad_models,
                key=lambda model: (
                    0 if model == incumbent_model else 1,
                    broad_estimates[model],
                    model,
                ),
            )
            results = []
            if len(ordered_broad) > 4 and len(dev) >= 3:
                screening_results.extend(run_screening(ordered_broad))
                if not screening_results:
                    raise ProviderError("No broad model completed the validation screening stage.")
                # Screening is deliberately broad and cheap. Full prompt search plus
                # repeated final testing is deeper, but it must not collapse to a
                # single completed route merely because two initially selected
                # providers fail. Aim for roughly 1.5*sqrt(N) fully settled routes,
                # with a floor of four, and backfill failures from the measured
                # screening frontier until that target is actually complete.
                full_limit = min(
                    len(screening_results),
                    max(4, math.ceil(1.5 * math.sqrt(len(screening_results)))),
                )
                ranked_quality = sorted(
                    screening_results,
                    key=lambda item: (
                        -float(item["validation_pass_rate"]),
                        float(item["estimated_production_cost_per_call_usd"]),
                        str(item["model"]),
                    ),
                )
                measured_frontier = [
                    item for item in screening_results
                    if not any(
                        other["model"] != item["model"]
                        and float(other["validation_pass_rate"]) >= float(item["validation_pass_rate"])
                        and float(other["estimated_production_cost_per_call_usd"])
                        <= float(item["estimated_production_cost_per_call_usd"])
                        and (
                            float(other["validation_pass_rate"]) > float(item["validation_pass_rate"])
                            or float(other["estimated_production_cost_per_call_usd"])
                            < float(item["estimated_production_cost_per_call_usd"])
                        )
                        for other in screening_results
                    )
                ]
                measured_frontier.sort(
                    key=lambda item: (
                        -float(item["validation_pass_rate"]),
                        float(item["estimated_production_cost_per_call_usd"]),
                    )
                )
                full_priority: list[str] = []
                intelligence_anchor_model: str | None = None

                def prioritize(model: str) -> None:
                    if model not in full_priority:
                        full_priority.append(model)

                if incumbent_model and any(item["model"] == incumbent_model for item in screening_results):
                    prioritize(incumbent_model)
                # When every original-prompt screen is below the quality gate,
                # preserve one intelligence anchor from the requested field. A
                # weak prompt can flatten cheap-model scores; the anchor gives the
                # optimizer one capable lane on which to learn a prompt package
                # that can then be propagated back down the cost frontier.
                if max(float(item["validation_pass_rate"]) for item in screening_results) < quality_threshold:
                    anchor_candidates = [
                        item for item in screening_results
                        if item["model"].split("#reasoning=", 1)[0] in catalog_intelligence
                    ]
                    if anchor_candidates:
                        anchor = max(
                            anchor_candidates,
                            key=lambda item: (
                                catalog_intelligence[item["model"].split("#reasoning=", 1)[0]],
                                -float(item["estimated_production_cost_per_call_usd"]),
                            ),
                        )
                        intelligence_anchor_model = str(anchor["model"])
                        prioritize(intelligence_anchor_model)
                for item in measured_frontier:
                    prioritize(str(item["model"]))
                for item in ranked_quality[:3]:
                    prioritize(str(item["model"]))
                prioritize(str(min(
                    screening_results,
                    key=lambda item: (
                        float(item["estimated_production_cost_per_call_usd"]),
                        -float(item["validation_pass_rate"]),
                    ),
                )["model"]))
                for item in ranked_quality:
                    prioritize(str(item["model"]))

                attempted_for_full: list[str] = []
                full_results: list[ModelResult] = []
                # Screening remains maximally parallel, but if every starting-
                # prompt score is weak, protect the one deliberately selected
                # intelligence anchor from shared-budget starvation. Once that
                # lane settles, the rest of the measured frontier runs in
                # parallel and can inherit any prompt package it discovers.
                if intelligence_anchor_model is not None:
                    attempted_for_full.append(intelligence_anchor_model)
                    full_results.extend(run_batch([intelligence_anchor_model]))
                while len(full_results) < full_limit:
                    remaining = [model for model in full_priority if model not in attempted_for_full]
                    if not remaining:
                        break
                    needed = max(1, full_limit - len(full_results))
                    wave = remaining[:needed]
                    attempted_for_full.extend(wave)
                    full_results.extend(run_batch(wave))
                results.extend(full_results)

                # A weak starting prompt can make a capable cheap model look bad in
                # the first screen. Re-test the not-yet-deep models with up to two
                # successful prompt packages learned strictly from training cases.
                # Validation may promote that package; final-test answers remain
                # untouched until the model receives its full evaluation.
                rewrite_packages: list[tuple[str, list[str], str]] = []
                seen_packages: set[tuple[str, tuple[str, ...]]] = set()
                for source in sorted(
                    full_results,
                    key=lambda item: (
                        0 if item.passed_quality_floor else 1,
                        -item.selected_pass_rate,
                        item.estimated_production_cost_per_call_usd,
                    ),
                ):
                    package_key = (
                        source.selected_prompt,
                        tuple(source.few_shot_example_ids),
                    )
                    if (
                        package_key in seen_packages
                        or (
                            source.selected_prompt == prompt_text
                            and not source.few_shot_example_ids
                        )
                    ):
                        continue
                    seen_packages.add(package_key)
                    rewrite_packages.append(
                        (
                            source.selected_prompt,
                            list(source.few_shot_example_ids),
                            source.model,
                        )
                    )
                    if len(rewrite_packages) >= 2:
                        break

                propagation_candidates = [
                    str(item["model"]) for item in screening_results
                    if item["model"] not in attempted_for_full
                ]
                propagation_by_model: dict[str, dict[str, Any]] = {}

                def screen_propagated(model: str) -> dict[str, Any]:
                    emit_progress({
                        "event": "prompt_propagation_started",
                        "model": model,
                        "candidate_prompt_packages": len(rewrite_packages),
                        "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
                    })
                    model_budget = _BudgetScope(budget)
                    best: dict[str, Any] | None = None
                    for propagated_prompt, few_shot_ids, source_model in rewrite_packages:
                        cases = self._run_cases(
                            propagated_prompt,
                            "propagated-screen",
                            dev,
                            "dev",
                            model,
                            evaluator_model,
                            model_budget,
                            train,
                            few_shot_ids,
                            evaluator=evaluator_policy,
                            max_parallel_scenarios=int(max_parallel_scenarios),
                        )
                        candidate = {
                            "model": model,
                            "validation_pass_rate": round(_scenario_pass_rate(cases), 6),
                            "cases": cases,
                            "prompt": propagated_prompt,
                            "few_shot_example_ids": list(few_shot_ids),
                            "source_model": source_model,
                        }
                        if best is None or (
                            float(candidate["validation_pass_rate"]),
                            -len(propagated_prompt),
                        ) > (
                            float(best["validation_pass_rate"]),
                            -len(str(best["prompt"])),
                        ):
                            best = candidate
                        if candidate["validation_pass_rate"] >= quality_threshold:
                            break
                    if best is None:
                        raise ProviderError("No successful prompt package was available to propagate.")
                    best["propagation_spend_usd"] = round(model_budget.spent_usd, 10)
                    emit_progress({
                        "event": "prompt_propagation_completed",
                        "model": model,
                        "source_model": best["source_model"],
                        "validation_pass_rate": best["validation_pass_rate"],
                        "few_shot_example_ids": best["few_shot_example_ids"],
                        "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
                    })
                    return best

                if optimize_prompt and rewrite_packages and propagation_candidates:
                    with ThreadPoolExecutor(
                        max_workers=min(int(max_parallel_models), len(propagation_candidates))
                    ) as pool:
                        futures = {
                            pool.submit(screen_propagated, model): model
                            for model in propagation_candidates
                        }
                        for future in as_completed(futures):
                            model = futures[future]
                            try:
                                propagation_by_model[model] = future.result()
                            except BudgetExceeded as error:
                                incomplete_models.append({
                                    "model": model,
                                    "reason": str(error),
                                    "stage": "prompt_propagation",
                                })
                            except ProviderError as error:
                                unavailable_models.append({
                                    "model": model,
                                    "reason": str(error),
                                    "stage": "prompt_propagation",
                                })

                propagated_for_full = [
                    model for model in propagation_candidates
                    if model in propagation_by_model
                    and float(propagation_by_model[model]["validation_pass_rate"])
                    >= quality_threshold
                ]
                for model in propagated_for_full:
                    package = propagation_by_model[model]
                    seed_prompt_by_model[model] = str(package["prompt"])
                    seed_few_shot_ids_by_model[model] = list(
                        package["few_shot_example_ids"]
                    )
                    screening_cases_by_model[model] = list(package["cases"])
                    prompt_origin_by_model[model] = (
                        f"propagated_from:{package['source_model']}"
                    )
                if propagated_for_full:
                    attempted_for_full.extend(propagated_for_full)
                    results.extend(run_batch(propagated_for_full))

                screened_out = [
                    str(item["model"]) for item in screening_results
                    if item["model"] not in attempted_for_full
                ]
                pruned_models.extend(screened_out)
                attempted_set = set(attempted_for_full)
                for item in screening_results:
                    propagation = propagation_by_model.get(str(item["model"]))
                    if propagation:
                        item["propagated_prompt_validation_pass_rate"] = propagation[
                            "validation_pass_rate"
                        ]
                        item["propagated_from_model"] = propagation["source_model"]
                        item["propagation_spend_usd"] = propagation[
                            "propagation_spend_usd"
                        ]
                    item["status"] = (
                        "FULL_PROPAGATED"
                        if item["model"] in prompt_origin_by_model
                        else "FULL"
                        if item["model"] in attempted_set
                        else "PRUNED"
                    )
                for model in screened_out:
                    emit_progress({
                        "event": "model_pruned", "model": model,
                        "reason": "Validation screening did not earn a full prompt-search and final-test run.",
                    })
            else:
                broad_wave_size = max(
                    1, min(len(ordered_broad) - 1 or 1, int(max_parallel_models))
                )
                stop_after_first_pass = objective in {
                    "cheapest_passing", "cheapest_at_accuracy", "constrained",
                    "match_baseline_at_lowest_cost",
                }
                for wave_start in range(0, len(ordered_broad), broad_wave_size):
                    wave = ordered_broad[wave_start : wave_start + broad_wave_size]
                    results.extend(run_batch(wave))
                    if stop_after_first_pass and any(item.passed_quality_floor for item in results):
                        pruned_models.extend(ordered_broad[wave_start + broad_wave_size :])
                        break
            passing_broad = [item for item in results if item.passed_quality_floor]
            if passing_broad:
                cheapest_pass_cost = min(item.estimated_production_cost_per_call_usd for item in passing_broad)
                hone_base_models = {
                    item.model.split("#reasoning=", 1)[0]
                    for item in results
                    if item.passed_quality_floor
                    or item.estimated_production_cost_per_call_usd <= cheapest_pass_cost * 1.25 + 1e-12
                }
            else:
                best_quality = max((item.holdout_pass_rate for item in results), default=0.0)
                near_floor = max(0.0, min(best_quality - 0.20, quality_threshold - 0.20))
                hone_base_models = {
                    item.model.split("#reasoning=", 1)[0]
                    for item in results
                    if item.holdout_pass_rate >= near_floor
                }
            hone_models = model_list[len(broad_models):]
            eligible_hone = [
                model for model in hone_models
                if model.split("#reasoning=", 1)[0] in hone_base_models
            ]
            pruned_models.extend(model for model in hone_models if model not in eligible_hone)
            # Reasoning effort should be the lever under test, not a hidden change
            # to the prompt package. Seed each hone configuration from the cheapest
            # passing result for the same base model (or the cheapest global passing
            # package when needed). It may still rewrite if training/validation says
            # the shared package is inadequate, and every variant still faces the
            # untouched final test.
            passing_packages = sorted(
                (item for item in results if item.passed_quality_floor),
                key=lambda item: (
                    item.estimated_production_cost_per_call_usd,
                    item.model,
                ),
            )
            for model in eligible_hone:
                base_model = model.split("#reasoning=", 1)[0]
                source = next(
                    (
                        item for item in passing_packages
                        if item.model.split("#reasoning=", 1)[0] == base_model
                    ),
                    passing_packages[0] if passing_packages else None,
                )
                if source is None:
                    continue
                seed_prompt_by_model[model] = source.selected_prompt
                seed_few_shot_ids_by_model[model] = list(
                    source.few_shot_example_ids
                )
                prompt_origin_by_model[model] = (
                    f"reasoning_hone_from:{source.model}"
                )
            results.extend(run_batch(eligible_hone))
        else:
            results = run_batch(model_list)
        if not results:
            if incomplete_models:
                raise BudgetExceeded(incomplete_models[0]["reason"])
            failed = "; ".join(f"{item['model']}: {item['reason']}" for item in unavailable_models)
            raise ProviderError(
                f"No selected target model completed under the required provider policy. {failed}"
            )
        passing = [item for item in results if item.passed_quality_floor]
        within_cost = [
            item for item in results
            if max_cost_per_run_usd is None
            or item.estimated_production_cost_per_call_usd <= max_cost_per_run_usd + 1e-12
        ]
        within_latency = [
            item for item in results
            if max_p90_latency_seconds is None
            or item.target_latency_p90_ms <= max_p90_latency_seconds * 1000
        ]
        for item in results:
            item.passed_latency_ceiling = item in within_latency
        eligible_results = within_latency if max_p90_latency_seconds is not None and within_latency else results
        constrained = [
            item for item in passing
            if item in within_cost and item in within_latency
        ]
        def effective_cost(item: ModelResult) -> float:
            return (
                item.estimated_cost_per_successful_call_usd
                + latency_value_usd_per_second * item.target_latency_p90_ms / 1000
            )
        baseline = next((item for item in results if item.model == incumbent_model), results[0])
        required_baseline_quality = max(0.0, baseline.baseline_holdout_pass_rate - allowed_accuracy_regression)
        matched_baseline = [
            item for item in within_cost
            if item in within_latency and item.holdout_pass_rate >= required_baseline_quality
        ]
        if objective == "match_baseline_at_lowest_cost" and matched_baseline:
            winner = min(
                matched_baseline,
                key=lambda item: (
                    item.estimated_production_cost_per_call_usd,
                    -item.holdout_pass_rate,
                    item.model,
                ),
            )
        elif objective in {"cheapest_passing", "cheapest_at_accuracy", "constrained"} and constrained:
            winner = min(
                constrained,
                key=lambda item: (
                    effective_cost(item),
                    -item.holdout_pass_rate,
                    item.model,
                ),
            )
        elif objective == "best_within_cost" and [item for item in within_cost if item in within_latency]:
            winner = max(
                [item for item in within_cost if item in within_latency],
                key=lambda item: (
                    item.holdout_pass_rate,
                    -effective_cost(item),
                ),
            )
        else:
            winner = max(
                eligible_results,
                key=lambda item: (
                    item.holdout_pass_rate,
                    -effective_cost(item),
                ),
            )
        warnings = []
        if len(holdout) < 5:
            warnings.append(
                f"Only {len(holdout)} distinct final-test scenario(s): this is exploratory and not a reliability claim."
            )
        if not passing:
            warnings.append("No prompt/model pair cleared the requested quality threshold.")
        if max_cost_per_run_usd is not None and not within_cost:
            warnings.append("No tested configuration fit the requested production cost ceiling.")
        if max_p90_latency_seconds is not None and not [
            item for item in passing if item in within_latency
        ]:
            warnings.append("No passing configuration fit the requested measured p90 latency ceiling.")
        if objective == "constrained" and not constrained:
            warnings.append("No tested configuration satisfied both production cost and accuracy constraints.")
        if objective == "match_baseline_at_lowest_cost" and not matched_baseline:
            warnings.append("No tested configuration matched the incumbent's held-out quality within the allowed regression margin.")
        if unavailable_models:
            warnings.append(
                f"{len(unavailable_models)} selected model(s) were unavailable under the required provider policy."
            )
        if incomplete_models or skipped_budget_models:
            warnings.append("Coverage is partial: the winner is best only among fully completed targets.")
        if pruned_models:
            warnings.append(
                f"Adaptive search pruned {len(pruned_models)} reasoning configuration(s) outside the observed task-capability band."
            )
        if omitted_configurations:
            warnings.append(
                f"Preflight omitted {len(omitted_configurations)} known-incompatible configuration(s) before any target call."
            )
        frontier, diminishing = _quality_frontier(
            results, float(minimum_meaningful_quality_gain)
        )
        suite_payload = {
            "schema": "evalt-regression-suite-v1",
            "starting_prompt": prompt_text,
            "winning_prompt": winner.selected_prompt,
            "winning_few_shot_example_ids": winner.few_shot_example_ids,
            "winning_few_shot_provenance": winner.few_shot_provenance,
            "examples": [asdict(item) for item in example_list],
            "selected_model": winner.model,
            "known_models": model_list,
            "optimizer_model": optimizer_model,
            "evaluator_model": evaluator_model,
            "evaluator": evaluator_policy,
            "difficulty_thresholds": difficulty_floor_policy,
            "quality_threshold": quality_threshold,
            "incumbent_model": incumbent_model,
            "incumbent_baseline_holdout_pass_rate": (
                baseline.baseline_holdout_pass_rate if incumbent_model else None
            ),
            "allowed_accuracy_regression": allowed_accuracy_regression,
            "max_cost_per_run_usd": max_cost_per_run_usd,
            "max_p90_latency_seconds": max_p90_latency_seconds,
            "latency_value_usd_per_second": latency_value_usd_per_second,
            "representative_input_chars": representative_input_chars,
            "representative_output_tokens": representative_output_tokens,
            "holdout_repeats": int(holdout_repeats),
            "holdout_unique_scenarios": len(holdout),
            "minimum_meaningful_quality_gain": minimum_meaningful_quality_gain,
            "optimize_prompt": bool(optimize_prompt),
            "watch": {
                "enabled": False,
                "max_recheck_cost_usd": 0,
                "notice": "Enable explicitly before any automatic provider call.",
            },
        }
        suite_payload["suite_hash"] = hashlib.sha256(
            json.dumps(suite_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        comparison_integrity = {
            "single_frozen_run": True,
            "suite_hash": suite_payload["suite_hash"],
            "distinct_final_test_scenarios": len(holdout),
            "executions_per_final_test_scenario": int(holdout_repeats),
            "evaluator": evaluator_policy,
            "difficulty_thresholds": difficulty_floor_policy,
            "selection_protocol": {
                "prompt_modification_enabled": bool(optimize_prompt),
                "prompt_optimizer_inputs": (
                    "training split only" if optimize_prompt else "disabled"
                ),
                "few_shot_sources": (
                    "customer-approved training examples only"
                    if optimize_prompt and allow_few_shot else "disabled"
                ),
                "prompt_package_selection": (
                    "validation split" if optimize_prompt else "fixed supplied prompt"
                ),
                "promotion_gate": "untouched final-test split",
                "final_test_used_for_rewrite_or_selection": False,
            },
            "configurations": [
                {
                    "configuration": configuration,
                    "model": configuration.split("#reasoning=", 1)[0],
                    "requested_reasoning_effort": (
                        configuration.rsplit("#reasoning=", 1)[1]
                        if "#reasoning=" in configuration else "none"
                    ),
                }
                for configuration in model_list
            ],
            "claim_scope": (
                "Task-specific prompt/model/reasoning configuration result only; "
                "not a general model-intelligence ranking."
            ),
        }
        return OptimizationResult(
            objective=objective,
            quality_threshold=quality_threshold,
            exploratory=len(holdout) < 5,
            winner=winner,
            models=results,
            total_provider_spend_usd=round(budget.spent_usd, 10),
            warnings=warnings,
            quality_frontier=frontier,
            diminishing_returns=diminishing,
            regression_suite=suite_payload,
            elapsed_seconds=round(time.monotonic() - optimization_started, 3),
            comparison_integrity=comparison_integrity,
            omitted_configurations=omitted_configurations,
            unavailable_models=unavailable_models,
            incomplete_models=incomplete_models,
            skipped_budget_models=skipped_budget_models,
            pruned_models=pruned_models,
            screening_results=screening_results,
            winner_scope="Best among the completed adaptive search band" if pruned_models else "Best among fully completed eligible targets only" if incomplete_models or skipped_budget_models or unavailable_models else "Best among every capability-eligible requested target" if omitted_configurations else "Best among every requested target",
            quality_gate_status=(
                "QUALIFIED_ROUTE_SELECTED"
                if winner.passed_quality_floor and winner.passed_latency_ceiling
                else "NO_CONFIGURATION_PASSED"
            ),
        )

    @staticmethod
    def _validate(
        prompt: str,
        examples: list[Example],
        models: list[str],
        quality_threshold: float,
        max_cost: float,
        rounds: int,
        minimum_gain: float,
    ) -> None:
        if len(prompt) < 8:
            raise ValueError("The current prompt must contain at least eight characters.")
        if len(examples) < 3:
            raise ValueError("At least three approved examples are required to optimize.")
        if len(models) < 1 or len(models) > 25:
            raise ValueError("Choose between one and 25 target models per bounded run.")
        if not 0 < quality_threshold <= 1:
            raise ValueError("quality_threshold must be greater than zero and at most one.")
        if max_cost <= 0:
            raise ValueError("max_optimization_cost_usd must be positive.")
        if not 1 <= int(rounds) <= 8:
            raise ValueError("rounds must be between one and eight.")
        if not 0 <= float(minimum_gain) <= 1:
            raise ValueError("minimum_meaningful_quality_gain must be between zero and one.")
        if any(any(not turn.input or not turn.approved_output for turn in item.conversation()) for item in examples):
            raise ValueError("Every scenario turn requires an input and approved output.")
        if any(not math.isfinite(item.weight) or item.weight <= 0 for item in examples):
            raise ValueError("Every scenario weight must be positive and finite.")
        grouped = [item for item in examples if item.group]
        if grouped:
            if len(grouped) != len(examples):
                raise ValueError("Either every scenario must declare a group or none may declare one.")
            group_counts: dict[str, int] = {}
            for item in examples:
                group_counts[item.group] = group_counts.get(item.group, 0) + 1
            undersized = sorted(group for group, count in group_counts.items() if count < 5)
            if undersized:
                raise ValueError(
                    "Every stratified group needs at least five scenarios: "
                    + ", ".join(undersized)
                )

    def _evaluate_model(
        self,
        prompt: str,
        train: list[Example],
        dev: list[Example],
        holdout: list[Example],
        model: str,
        optimizer_model: str,
        evaluator_model: str,
        threshold: float,
        budget: _Budget,
        rounds: int,
        allow_few_shot: bool,
        max_few_shot_examples: int,
        representative_input_chars: int | None,
        representative_output_tokens: int | None,
        holdout_repeats: int,
        evaluator: dict[str, Any],
        difficulty_thresholds: dict[str, float],
        max_parallel_scenarios: int,
        baseline_dev_cases: list[CaseResult] | None = None,
        seed_prompt: str | None = None,
        seed_few_shot_ids: list[str] | None = None,
        prompt_origin: str = "starting_prompt",
        optimize_prompt: bool = True,
    ) -> ModelResult:
        started_spend = budget.spent_usd
        initial_prompt = seed_prompt or prompt
        initial_few_shot_ids = list(seed_few_shot_ids or [])
        # Validation remains the selection gate, but a small validation slice can
        # reach 100% by chance while the starting prompt still fails much of the
        # approved training evidence. Always measure both disclosed splits before
        # deciding that prompt learning is unnecessary. The frozen final test is
        # still never used to select or rewrite a prompt.
        baseline_dev = list(baseline_dev_cases) if baseline_dev_cases is not None else self._run_cases(
            initial_prompt, "baseline", dev, "dev", model, evaluator_model, budget,
            train, initial_few_shot_ids, evaluator=evaluator,
            max_parallel_scenarios=max_parallel_scenarios,
        )
        baseline_dev_rate = _pass_rate(baseline_dev)
        baseline_train = self._run_cases(
            initial_prompt, "baseline", train, "train", model, evaluator_model, budget,
            train, initial_few_shot_ids, evaluator=evaluator,
            max_parallel_scenarios=max_parallel_scenarios,
        )
        baseline_train_rate = _pass_rate(baseline_train)
        selected_prompt = initial_prompt
        selected_few_shot_ids: list[str] = initial_few_shot_ids
        selected_train = baseline_train or baseline_dev
        selected_train_rate = baseline_train_rate
        selected_dev = baseline_dev
        selected_rate = baseline_dev_rate
        candidate_cases: list[CaseResult] = []
        for round_number in (
            range(1, int(rounds) + 1)
            if optimize_prompt and (selected_rate < 1 or selected_train_rate < 1)
            else ()
        ):
            revised_prompt, revised_few_shot_ids = self._propose_prompt(
                selected_prompt,
                model,
                train,
                selected_train,
                optimizer_model,
                budget,
                allow_few_shot,
                max_few_shot_examples,
            )
            # Training evidence breaks a validation tie; the candidate must never
            # regress on validation. This avoids silently skipping a useful rewrite
            # after a lucky perfect score on a small validation slice.
            revised_train = self._run_cases(
                revised_prompt,
                f"candidate-{round_number}",
                train,
                "train",
                model,
                evaluator_model,
                budget,
                train,
                revised_few_shot_ids,
                evaluator=evaluator,
                max_parallel_scenarios=max_parallel_scenarios,
            )
            candidate_cases += revised_train
            revised_dev = self._run_cases(revised_prompt, f"candidate-{round_number}", dev, "dev", model, evaluator_model, budget, train, revised_few_shot_ids, evaluator=evaluator, max_parallel_scenarios=max_parallel_scenarios)
            candidate_cases += revised_dev
            revised_rate = _pass_rate(revised_dev)
            revised_train_rate = _pass_rate(revised_train)
            if (
                revised_rate > selected_rate
                or (
                    revised_rate == selected_rate
                    and revised_train_rate > selected_train_rate
                )
            ):
                selected_prompt = revised_prompt
                selected_few_shot_ids = revised_few_shot_ids
                selected_train = revised_train
                selected_train_rate = revised_train_rate
                selected_dev = revised_dev
                selected_rate = revised_rate
            if selected_rate >= 1 and selected_train_rate >= 1:
                break
        selected_kind = (
            "baseline"
            if selected_prompt == prompt and not selected_few_shot_ids
            else "candidate"
        )
        baseline_holdout = self._run_cases(
            prompt, "baseline", holdout, "holdout", model, evaluator_model, budget,
            repeats=holdout_repeats,
            evaluator=evaluator,
            max_parallel_scenarios=max_parallel_scenarios,
        )
        if selected_kind == "baseline":
            selected_holdout = baseline_holdout
        else:
            selected_holdout = self._run_cases(
                selected_prompt, "candidate", holdout, "holdout", model, evaluator_model, budget
                , train, selected_few_shot_ids, holdout_repeats, evaluator=evaluator,
                max_parallel_scenarios=max_parallel_scenarios,
            )
        holdout_execution_rate = _pass_rate(selected_holdout)
        holdout_rate = _scenario_pass_rate(selected_holdout)
        holdout_by_difficulty = _scenario_pass_rates_by_difficulty(selected_holdout)
        passed_difficulty_floors = all(
            difficulty in holdout_by_difficulty
            and holdout_by_difficulty[difficulty] >= floor
            for difficulty, floor in difficulty_thresholds.items()
        )
        observed_inputs = [sum(len(turn.input) for turn in item.conversation()) for item in train + dev + holdout]
        observed_outputs = [sum(len(turn.approved_output) for turn in item.conversation()) for item in train + dev + holdout]
        typical_input = representative_input_chars or _percentile(observed_inputs, 0.90)
        typical_output_tokens = representative_output_tokens or max(32, int(_percentile(observed_outputs, 0.90) / 3) + 1)
        production_messages = [{"role": "system", "content": selected_prompt}] + _few_shot_messages(train, selected_few_shot_ids) + [{"role": "user", "content": "x" * int(typical_input)}]
        # A completion allowance is a safety ceiling, not an expected bill. Price
        # the promoted route from the measured 90th-percentile successful final-test
        # call so generous first-request headroom does not make a tiny JSON response
        # look as if it always consumes the full context window.
        measured_target_costs = sorted(
            result.target_cost_usd for result in selected_holdout
            if result.target_cost_usd > 0
        )
        if measured_target_costs:
            production_cost = measured_target_costs[
                max(0, min(len(measured_target_costs) - 1, math.ceil(len(measured_target_costs) * 0.90) - 1))
            ]
        else:
            production_cost = self.transport.estimate_cost(model, production_messages, max_tokens=int(typical_output_tokens))
        cost_per_success = production_cost / max(holdout_rate, 0.01)
        measured_target_latencies = sorted(
            result.target_latency_ms for result in selected_holdout
            if result.target_latency_ms > 0
        )
        latency_p50 = _percentile(measured_target_latencies, 0.50) if measured_target_latencies else 0
        latency_p90 = _percentile(measured_target_latencies, 0.90) if measured_target_latencies else 0
        all_cases = baseline_train + baseline_dev + candidate_cases + baseline_holdout
        if selected_holdout is not baseline_holdout:
            all_cases += selected_holdout
        selected_prompt_origin = (
            prompt_origin
            if selected_prompt == initial_prompt
            and selected_few_shot_ids == initial_few_shot_ids
            else f"optimized_for:{model}"
        )
        return ModelResult(
            model=model,
            selected_prompt=selected_prompt,
            baseline_pass_rate=round(baseline_train_rate, 6),
            selected_pass_rate=round(selected_rate, 6),
            holdout_pass_rate=round(holdout_rate, 6),
            baseline_holdout_pass_rate=round(_scenario_pass_rate(baseline_holdout), 6),
            estimated_production_cost_per_call_usd=round(production_cost, 10),
            estimated_cost_per_successful_call_usd=round(cost_per_success, 10),
            optimization_spend_usd=round(budget.spent_usd - started_spend, 10),
            passed_quality_floor=holdout_rate >= threshold and passed_difficulty_floors,
            target_latency_p50_ms=round(latency_p50),
            target_latency_p90_ms=round(latency_p90),
            holdout_unique_scenarios=len(holdout),
            holdout_executions=len(selected_holdout),
            holdout_execution_pass_rate=round(holdout_execution_rate, 6),
            baseline_holdout_execution_pass_rate=round(_pass_rate(baseline_holdout), 6),
            holdout_pass_rates_by_difficulty=holdout_by_difficulty,
            passed_difficulty_floors=passed_difficulty_floors,
            prompt_origin=selected_prompt_origin,
            few_shot_example_ids=selected_few_shot_ids,
            few_shot_provenance=[
                {
                    "example_id": example_id,
                    "source_split": "train",
                    "customer_approved": True,
                    "eligible_for_validation_prompt": True,
                    "eligible_for_final_test_prompt": True,
                }
                for example_id in selected_few_shot_ids
            ],
            cases=all_cases,
        )

    def _run_cases(
        self,
        prompt: str,
        prompt_kind: str,
        examples: list[Example],
        split: str,
        target_model: str,
        evaluator_model: str,
        budget: _Budget,
        few_shot_source: list[Example] | None = None,
        few_shot_ids: list[str] | None = None,
        repeats: int = 1,
        evaluator: dict[str, Any] | None = None,
        max_parallel_scenarios: int = 1,
    ) -> list[CaseResult]:
        evaluator_policy = evaluator or {"type": "semantic"}
        target_max_tokens = {
            "exact_text": 64,
            "exact_json": 1024,
            "semantic": 8192,
        }[evaluator_policy["type"]]

        def run_execution(example: Example, repeat_index: int) -> list[CaseResult]:
            scenario_results: list[CaseResult] = []
            transcript: list[dict[str, str]] = []
            demonstrations = _few_shot_messages(few_shot_source or [], few_shot_ids or [], exclude_id=example.id if split == "train" else "")
            for turn_index, turn in enumerate(example.conversation()):
                target = self._call(
                    budget,
                    target_model,
                    [{"role": "system", "content": prompt}] + demonstrations + transcript + [{"role": "user", "content": turn.input}],
                    max_tokens=target_max_tokens,
                    response_schema=(
                        _exact_json_response_schema(turn.approved_output, evaluator_policy)
                        if evaluator_policy["type"] == "exact_json"
                        else None
                    ),
                )
                transcript += [{"role": "user", "content": turn.input}, {"role": "assistant", "content": target.content}]
                judgment, judge_completion = self._judge(example, turn, turn_index, transcript, target.content, evaluator_model, budget, evaluator_policy)
                repeat_suffix = f":repeat-{repeat_index + 1}" if repeats > 1 else ""
                scenario_results.append(
                    CaseResult(
                        example_id=f"{example.id}:turn-{turn_index + 1}{repeat_suffix}",
                        split=split,
                        prompt_kind=prompt_kind,
                        output=target.content,
                        approved_output=turn.approved_output,
                        passed=judgment.passed,
                        score=judgment.score,
                        reason=judgment.reason,
                        target_cost_usd=target.cost_usd,
                        evaluator_cost_usd=judge_completion.cost_usd,
                        target_generation_id=target.generation_id,
                        evaluator_generation_id=judge_completion.generation_id,
                        target_latency_ms=target.latency_ms,
                        evaluator_latency_ms=judge_completion.latency_ms,
                        group=example.group,
                        difficulty=example.difficulty,
                        weight=example.weight,
                        critical=example.critical,
                    )
                )
            return scenario_results

        executions = [(example, repeat_index) for example in examples for repeat_index in range(int(repeats))]
        if len(executions) <= 1 or int(max_parallel_scenarios) <= 1:
            return [result for example, repeat_index in executions for result in run_execution(example, repeat_index)]
        completed: dict[tuple[str, int], list[CaseResult]] = {}
        with ThreadPoolExecutor(max_workers=min(int(max_parallel_scenarios), len(executions))) as pool:
            futures = {pool.submit(run_execution, example, repeat_index): (example.id, repeat_index) for example, repeat_index in executions}
            for future in as_completed(futures):
                completed[futures[future]] = future.result()
        return [result for example, repeat_index in executions for result in completed[(example.id, repeat_index)]]

    def _judge(
        self,
        example: Example,
        turn: Turn,
        turn_index: int,
        transcript: list[dict[str, str]],
        output: str,
        evaluator_model: str,
        budget: _Budget,
        evaluator: dict[str, Any] | None = None,
    ) -> tuple[Judgment, Completion]:
        evaluator = evaluator or {"type": "semantic"}
        if evaluator["type"] != "semantic":
            judgment = _deterministic_judgment(output, turn.approved_output, evaluator)
            return judgment, Completion(
                content=json.dumps(asdict(judgment), separators=(",", ":")),
                model=f"deterministic/{evaluator['type']}",
                generation_id=f"deterministic:{evaluator['type']}",
                cost_usd=0.0,
            )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["passed", "score", "reason"],
            "properties": {
                "passed": {"type": "boolean"},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
        }
        completion = self._call(
            budget,
            evaluator_model,
            [
                {
                    "role": "system",
                    "content": (
                        "Judge whether the actual answer satisfies the behavior demonstrated "
                        "by the customer-approved answer. Do not require identical wording. "
                        "Return only the required JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "scenario_id": example.id,
                            "turn": turn_index + 1,
                            "transcript": transcript,
                            "input": turn.input,
                            "approved_answer": turn.approved_output,
                            "actual_answer": output,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=180,
            response_schema=schema,
        )
        try:
            value = _parse_json_object(completion.content)
            score = min(1.0, max(0.0, float(value["score"])))
            return Judgment(bool(value["passed"]), score, str(value["reason"])), completion
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ProviderError("The evaluator returned invalid structured judgment JSON.") from error

    def _propose_prompt(
        self,
        prompt: str,
        target_model: str,
        train: list[Example],
        baseline: list[CaseResult],
        optimizer_model: str,
        budget: _Budget,
        allow_few_shot: bool,
        max_few_shot_examples: int,
    ) -> tuple[str, list[str]]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["prompt", "hypothesis", "few_shot_example_ids"],
            "properties": {
                "prompt": {"type": "string"},
                "hypothesis": {"type": "string"},
                "few_shot_example_ids": {"type": "array", "items": {"type": "string"}},
            },
        }
        payload = {
            "target_model": target_model,
            "current_prompt": prompt,
            "examples": [asdict(item) for item in train],
            "baseline_results": [
                {
                    "example_id": item.example_id,
                    "output": item.output,
                    "passed": item.passed,
                    "reason": item.reason,
                }
                for item in baseline
            ],
            "few_shot_allowed": bool(allow_few_shot),
            "allowed_few_shot_example_ids": [item.id for item in train] if allow_few_shot else [],
            "max_few_shot_examples": int(max_few_shot_examples),
        }
        completion = self._call(
            budget,
            optimizer_model,
            [
                {
                    "role": "system",
                    "content": (
                        "Improve the current prompt package for the named target model. "
                        "You may rewrite the system prompt, select approved training examples as "
                        "few-shot demonstrations, do both, or keep the package. Before proposing, audit "
                        "every approved training scenario and every observed failure. Preserve every "
                        "label-changing boundary, exception, precedence rule, output constraint, and "
                        "multi-turn dependency; do not collapse cases with different approved behavior. "
                        "The production model will receive only the proposed system prompt, any selected "
                        "few-shot demonstrations, the current conversation transcript, and the new user "
                        "message. Make the package self-contained: never refer to a supplied policy, "
                        "training set, approved answer, test, or instruction that will not actually be "
                        "present at production time. Generalize rules from training evidence instead of "
                        "copying validation-specific wording. "
                        "Mentally replay the proposed package against all supplied scenarios. Use only "
                        "allowed IDs; demonstrations add production token cost. Return the package and "
                        "hypothesis as JSON."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=900,
            response_schema=schema,
        )
        try:
            parsed = _parse_json_object(completion.content)
            revised = str(parsed["prompt"]).strip()
            allowed = {item.id for item in train} if allow_few_shot else set()
            selected_ids = list(dict.fromkeys(str(value) for value in parsed.get("few_shot_example_ids", []) if str(value) in allowed))[: max(0, int(max_few_shot_examples))]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ProviderError("The optimizer returned invalid structured prompt JSON.") from error
        if len(revised) < 8:
            raise ProviderError("The optimizer returned an unusably short prompt.")
        return revised, selected_ids


def _parse_json_object(value: str) -> dict[str, Any]:
    text = str(value).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _exact_json_response_schema(
    approved_output: str, evaluator: dict[str, Any]
) -> dict[str, Any]:
    """Build a shape-only schema without leaking the approved answer."""
    expected = _parse_json_object(approved_output)
    required = list(evaluator.get("required_keys") or expected.keys())

    def shape(value: Any) -> dict[str, Any]:
        if value is None:
            return {"type": "null"}
        if isinstance(value, bool):
            return {"type": "boolean"}
        if isinstance(value, int):
            return {"type": "integer"}
        if isinstance(value, float):
            return {"type": "number"}
        if isinstance(value, list):
            return {"type": "array", "items": shape(value[0]) if value else {}}
        if isinstance(value, dict):
            return {
                "type": "object",
                "properties": {str(key): shape(item) for key, item in value.items()},
                "required": [str(key) for key in value],
                "additionalProperties": False,
            }
        return {"type": "string"}

    return {
        "type": "object",
        "properties": {key: shape(expected[key]) for key in required if key in expected},
        "required": required,
        "additionalProperties": bool(evaluator.get("allow_additional_properties", True)),
    }


def _validate_evaluator_policy(value: dict[str, Any] | None) -> dict[str, Any]:
    policy = dict(value or {"type": "semantic"})
    evaluator_type = str(policy.get("type", "")).strip().lower()
    if evaluator_type not in {"semantic", "exact_text", "exact_json"}:
        raise ValueError("evaluator.type must be semantic, exact_text, or exact_json.")
    policy["type"] = evaluator_type
    if evaluator_type == "exact_json":
        required_keys = policy.get("required_keys", [])
        if not isinstance(required_keys, list) or any(not str(key).strip() for key in required_keys):
            raise ValueError("evaluator.required_keys must be a list of non-empty strings.")
        policy["required_keys"] = list(dict.fromkeys(str(key).strip() for key in required_keys))
        policy["allow_additional_properties"] = bool(policy.get("allow_additional_properties", True))
        policy["normalize_rational_strings"] = bool(policy.get("normalize_rational_strings", False))
    return policy


def _validate_difficulty_thresholds(value: dict[str, float] | None) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for raw_name, raw_floor in dict(value or {}).items():
        name = str(raw_name).strip()
        floor = float(raw_floor)
        if not name:
            raise ValueError("difficulty_thresholds keys must be non-empty.")
        if not math.isfinite(floor) or not 0 < floor <= 1:
            raise ValueError("Every difficulty threshold must be greater than zero and at most one.")
        thresholds[name] = floor
    return thresholds


def _deterministic_judgment(
    output: str, approved_output: str, evaluator: dict[str, Any]
) -> Judgment:
    if evaluator["type"] == "exact_text":
        passed = str(output).strip() == str(approved_output).strip()
        return Judgment(passed, 1.0 if passed else 0.0, "Exact text matched." if passed else "Exact text differed.")
    try:
        actual = _parse_json_object(output)
        expected = _parse_json_object(approved_output)
    except (TypeError, ValueError, json.JSONDecodeError):
        return Judgment(False, 0.0, "Actual answer was not one valid JSON object.")
    required = set(evaluator.get("required_keys", []))
    actual_keys = set(actual)
    if not required.issubset(actual_keys):
        return Judgment(False, 0.0, f"Missing required JSON key(s): {', '.join(sorted(required - actual_keys))}.")
    if not evaluator.get("allow_additional_properties", True) and actual_keys != required:
        extras = actual_keys - required
        return Judgment(False, 0.0, f"Unexpected JSON key(s): {', '.join(sorted(extras))}.")
    comparison_keys = required or set(expected)
    for key in comparison_keys:
        if key not in expected or key not in actual:
            return Judgment(False, 0.0, f"JSON key {key!r} was not comparable.")
        actual_value = actual[key]
        expected_value = expected[key]
        if evaluator.get("normalize_rational_strings"):
            try:
                equal = Fraction(str(actual_value).strip()) == Fraction(str(expected_value).strip())
            except (ValueError, ZeroDivisionError):
                equal = actual_value == expected_value
        else:
            equal = actual_value == expected_value
        if not equal:
            return Judgment(False, 0.0, f"JSON value for {key!r} differed from the approved value.")
    return Judgment(True, 1.0, "JSON structure and approved values matched deterministically.")


def _safe_provider_error_detail(value: str) -> str:
    """Keep actionable provider diagnostics without exporting account identifiers."""
    try:
        payload = json.loads(str(value))
        error = payload.get("error") or {}
        safe: dict[str, Any] = {
            "message": str(error.get("message") or "Provider request failed."),
            "code": error.get("code"),
        }
        provider = (error.get("metadata") or {}).get("provider_name")
        if provider:
            safe["provider"] = str(provider)
        return json.dumps(safe, separators=(",", ":"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return "Provider request failed; raw provider detail was omitted."


def _split_examples(examples: list[Example], seed: str) -> tuple[list[Example], list[Example], list[Example]]:
    if examples and all(item.group for item in examples):
        grouped: dict[str, list[Example]] = {}
        for item in examples:
            grouped.setdefault(item.group, []).append(item)
        train: list[Example] = []
        dev: list[Example] = []
        holdout: list[Example] = []
        for group, items in sorted(grouped.items()):
            ranked_group = sorted(
                items,
                key=lambda item: hashlib.sha256(f"{seed}:{group}:{item.id}".encode()).hexdigest(),
            )
            holdout_count = max(1, len(ranked_group) // 5)
            dev_count = max(1, len(ranked_group) // 5)
            holdout.extend(ranked_group[:holdout_count])
            dev.extend(ranked_group[holdout_count : holdout_count + dev_count])
            train.extend(ranked_group[holdout_count + dev_count :])
        return train, dev, holdout
    ranked = sorted(
        examples,
        key=lambda item: hashlib.sha256(f"{seed}:{item.id}".encode()).hexdigest(),
    )
    holdout_count = max(1, len(ranked) // 5)
    dev_count = max(1, len(ranked) // 5)
    return ranked[holdout_count + dev_count :], ranked[holdout_count : holdout_count + dev_count], ranked[:holdout_count]


def _few_shot_messages(examples: list[Example], selected_ids: list[str], exclude_id: str = "") -> list[dict[str, str]]:
    selected = set(selected_ids)
    messages: list[dict[str, str]] = []
    for example in examples:
        if example.id not in selected or example.id == exclude_id:
            continue
        for turn in example.conversation():
            messages += [{"role": "user", "content": turn.input}, {"role": "assistant", "content": turn.approved_output}]
    return messages


def _pass_rate(results: list[CaseResult]) -> float:
    total_weight = sum(item.weight for item in results)
    return (
        sum(item.weight for item in results if item.passed) / total_weight
        if total_weight > 0
        else 0.0
    )


def _scenario_pass_rate(results: list[CaseResult]) -> float:
    """Count distinct scenarios once; every repeated execution must pass."""
    scenarios: dict[str, dict[str, Any]] = {}
    for item in results:
        scenario_id = item.example_id.split(":turn-", 1)[0]
        scenario = scenarios.setdefault(scenario_id, {"outcomes": [], "weight": item.weight})
        scenario["outcomes"].append(item.passed)
    total_weight = sum(float(value["weight"]) for value in scenarios.values())
    if total_weight <= 0:
        return 0.0
    passed_weight = sum(
        float(value["weight"])
        for value in scenarios.values()
        if all(value["outcomes"])
    )
    return passed_weight / total_weight


def _scenario_pass_rates_by_difficulty(results: list[CaseResult]) -> dict[str, float]:
    by_difficulty: dict[str, list[CaseResult]] = {}
    for item in results:
        by_difficulty.setdefault(item.difficulty or "typical", []).append(item)
    return {
        difficulty: round(_scenario_pass_rate(items), 6)
        for difficulty, items in sorted(by_difficulty.items())
    }


def _percentile(values: list[int], fraction: float) -> int:
    """Return a conservative nearest-rank percentile for production sizing."""
    if not values:
        return 1
    ordered = sorted(max(0, int(value)) for value in values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


def _quality_frontier(
    results: list[ModelResult], minimum_gain: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(
        results,
        key=lambda item: (
            item.estimated_cost_per_successful_call_usd,
            -item.holdout_pass_rate,
            item.model,
        ),
    )
    frontier: list[dict[str, Any]] = []
    best_quality = -1.0
    dominated: list[dict[str, Any]] = []
    for item in ordered:
        point = {
            "model": item.model,
            "holdout_pass_rate": item.holdout_pass_rate,
            "estimated_cost_per_successful_call_usd": item.estimated_cost_per_successful_call_usd,
        }
        gain = item.holdout_pass_rate - best_quality if best_quality >= 0 else item.holdout_pass_rate
        if item.holdout_pass_rate > best_quality:
            frontier.append(point)
            best_quality = item.holdout_pass_rate
        elif gain < minimum_gain:
            dominated.append({**point, "measured_quality_gain": round(gain, 6)})
    return frontier, {
        "minimum_meaningful_quality_gain": minimum_gain,
        "higher_cost_models_without_material_gain": dominated,
        "caveat": "Measured only on this frozen suite; price is not a proxy for intelligence.",
    }
