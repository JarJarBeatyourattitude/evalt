"""Typed, primary Evalt SDK surface.

The optimization engine remains import-compatible with the two earlier package names;
new code should use :class:`Suite` and :class:`Evalt` from this module.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass, field, replace
from functools import wraps
import copy
import json
from pathlib import Path
import re
import secrets
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping

from last_good_prompt.core import (
    BudgetExceeded,
    CaseResult,
    ChatTransport,
    Client,
    Completion,
    DraftAnswer,
    Example,
    Judgment,
    ModelResult,
    OpenRouterTransport,
    OptimizationResult,
    ProviderError,
    Turn,
    _Budget,
    _extract_single_numeric_scalar,
    _safe_provider_error_detail,
    _validate_evaluator_policy,
    normalize_request_options,
    request_options_fingerprint,
)
from .router import (
    DEFAULT_TARGETS, DurableRouter, RequestEnvelopeDriftWarning, RolePlan,
    RoutedAnswer, select_role_plan,
)
from .dashboard import DEFAULT_DASHBOARD_API_URL, WorkspaceSync, load_dashboard_config


_NUMBER_WORDS = {
    "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
    "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0,
    "nine": 9.0, "ten": 10.0, "hundred": 100.0,
}


def _infer_numeric_scale_contract(*values: str) -> tuple[float, float] | None:
    """Recover an explicit rating range from the customer-owned task contract."""

    text = " ".join(str(value) for value in values).casefold()
    numeric_patterns = (
        r"\bfrom\s+(-?\d+(?:\.\d+)?)\b.{0,120}?\bto\s+(-?\d+(?:\.\d+)?)\b",
        r"\bbetween\s+(-?\d+(?:\.\d+)?)\b.{0,80}?\band\s+(-?\d+(?:\.\d+)?)\b",
        r"\b(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(-?\d+(?:\.\d+)?)\b",
    )
    for pattern in numeric_patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            minimum, maximum = float(match.group(1)), float(match.group(2))
            if maximum > minimum:
                return minimum, maximum
    word_pattern = (
        r"\b(" + "|".join(_NUMBER_WORDS) + r")\s*(?:-|–|—|to)\s*("
        + "|".join(_NUMBER_WORDS) + r")\b"
    )
    match = re.search(word_pattern, text)
    if match:
        minimum = _NUMBER_WORDS[match.group(1)]
        maximum = _NUMBER_WORDS[match.group(2)]
        if maximum > minimum:
            return minimum, maximum
    return None


def _default_numeric_tolerance(
    task: str, prompt: str, minimum: float, maximum: float
) -> float:
    """Use a wider equivalence band only for irreducibly subjective ratings."""

    contract = f"{task} {prompt}".casefold()
    subjective_markers = (
        "sentiment", "satisfaction", "tone", "mood", "preference",
        "likelihood", "quality rating", "severity rating", "opinion",
    )
    fraction = 0.20 if any(marker in contract for marker in subjective_markers) else 0.10
    return (maximum - minimum) * fraction


def _automatic_target_max_tokens(
    evaluator: Mapping[str, Any], examples: Iterable[Example]
) -> int:
    """Choose a safe tested output envelope from the drafted task contract.

    A scalar or tiny exact label should not reserve the same 600-token worst case as
    prose, JSON, or code. Explicit developer ``max_tokens`` always bypasses this
    inference; this helper only shapes the default first-run tournament and installed
    route.
    """
    outputs = [
        turn.approved_output
        for example in examples
        for turn in example.conversation()
    ]
    longest_chars = max((len(value) for value in outputs), default=0)
    evaluator_type = str(evaluator.get("type") or "semantic")
    if evaluator_type == "numeric_tolerance":
        return 128
    if evaluator_type == "exact_text" and longest_chars <= 48:
        return 128
    estimated_tokens = max(1, (longest_chars + 2) // 3)
    if evaluator_type == "exact_json":
        return min(8192, max(512, estimated_tokens * 6 + 128))
    return min(8192, max(600, estimated_tokens * 6 + 128))


def _dashboard_run_scope(method):
    """Give one public SDK invocation a stable opaque dashboard lifecycle."""

    @wraps(method)
    def wrapped(self, suite_or_prompt, *args, **kwargs):
        current = getattr(self._dashboard_run_local, "context", None)
        if current is not None:
            return method(self, suite_or_prompt, *args, **kwargs)
        route = (
            str(getattr(suite_or_prompt, "name", "") or "default")
            if isinstance(suite_or_prompt, Suite)
            else str(kwargs.get("route") or "default")
        )
        context = {
            "run_id": f"evr_{secrets.token_urlsafe(12)}",
            "run_state": "running",
            "run_started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "route": route,
        }
        self._dashboard_run_local.context = context
        self._emit_dashboard_status(route, "connected" if self._dashboard_sync else "local")
        self._emit_progress({"event": "run_started", "route": route})
        try:
            result = method(self, suite_or_prompt, *args, **kwargs)
            context["run_state"] = "completed"
            context["run_finished_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            self._emit_progress({"event": "run_completed", "route": route})
            if self._dashboard_sync is not None:
                synced = self.flush_dashboard(timeout_seconds=8.0)
                self._emit_dashboard_status(route, "synced" if synced else "failed")
            return result
        except Exception:
            context["run_state"] = "failed"
            context["run_finished_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            self._emit_progress({"event": "run_failed", "route": route})
            if self._dashboard_sync is not None:
                synced = self.flush_dashboard(timeout_seconds=4.0)
                self._emit_dashboard_status(route, "synced" if synced else "failed")
            raise
        finally:
            try:
                del self._dashboard_run_local.context
            except AttributeError:
                pass

    return wrapped


@dataclass(frozen=True)
class Suite:
    """A reviewable optimization contract that can be validated without an API key."""

    prompt: str
    examples: tuple[Example, ...]
    models: tuple[str, ...]
    optimizer_model: str = "openai/gpt-5.6-luna"
    evaluator_model: str = "openai/gpt-5.6-luna"
    evaluator: Mapping[str, Any] = field(default_factory=lambda: {"type": "semantic"})
    difficulty_thresholds: Mapping[str, float] = field(default_factory=dict)
    objective: str = "lowest_cost_at_accuracy"
    quality_threshold: float = 0.95
    max_optimization_cost_usd: float = 2.00
    rounds: int = 3
    optimize_prompt: bool = True
    holdout_repeats: int = 2
    max_parallel_models: int = 16
    max_parallel_scenarios: int = 32
    request_timeout_seconds: float = 600
    target_max_tokens: int = 600
    request_options: Mapping[str, Any] = field(default_factory=dict)
    max_p90_latency_seconds: float | None = None
    latency_value_usd_per_second: float = 0.0
    minimum_meaningful_quality_gain: float = 0.03
    allow_few_shot: bool = True
    max_few_shot_examples: int = 3
    incumbent_model: str | None = None
    allowed_accuracy_regression: float = 0.0
    adaptive_search: bool = True
    evidence_provenance: str = "HUMAN_APPROVED"
    name: str = "evalt-suite"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Suite":
        try:
            examples = tuple(
                Example.from_value(item, index)
                for index, item in enumerate(value["examples"])
            )
            suite = cls(
                name=str(value.get("name") or "evalt-suite"),
                prompt=str(value["prompt"]),
                examples=examples,
                models=tuple(str(model) for model in value["models"]),
                optimizer_model=str(value.get("optimizer_model", "openai/gpt-5.6-luna")),
                evaluator_model=str(value.get("evaluator_model", "openai/gpt-5.6-luna")),
                evaluator=dict(value.get("evaluator") or {"type": "semantic"}),
                difficulty_thresholds={
                    str(name): float(floor)
                    for name, floor in dict(value.get("difficulty_thresholds") or {}).items()
                },
                objective=str(value.get("objective", "lowest_cost_at_accuracy")),
                quality_threshold=float(value.get("quality_threshold", 0.95)),
                max_optimization_cost_usd=float(value.get("max_optimization_cost_usd", 2.00)),
                rounds=int(value.get("rounds", 3)),
                optimize_prompt=bool(value.get("optimize_prompt", True)),
                holdout_repeats=int(value.get("holdout_repeats", 2)),
                max_parallel_models=int(value.get("max_parallel_models", 16)),
                max_parallel_scenarios=int(value.get("max_parallel_scenarios", 32)),
                request_timeout_seconds=float(value.get("request_timeout_seconds", 600)),
                target_max_tokens=int(value.get("target_max_tokens", 600)),
                request_options=normalize_request_options(value.get("request_options")),
                max_p90_latency_seconds=(
                    float(value["max_p90_latency_seconds"])
                    if value.get("max_p90_latency_seconds") is not None else None
                ),
                latency_value_usd_per_second=float(value.get("latency_value_usd_per_second", 0.0)),
                minimum_meaningful_quality_gain=float(value.get("minimum_meaningful_quality_gain", 0.03)),
                allow_few_shot=bool(value.get("allow_few_shot", True)),
                max_few_shot_examples=int(value.get("max_few_shot_examples", 3)),
                incumbent_model=str(value["incumbent_model"]) if value.get("incumbent_model") else None,
                allowed_accuracy_regression=float(value.get("allowed_accuracy_regression", 0.0)),
                adaptive_search=bool(value.get("adaptive_search", True)),
                evidence_provenance=str(
                    value.get("evidence_provenance", "HUMAN_APPROVED")
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid Evalt suite: {error}") from error
        suite.validate()
        return suite

    @classmethod
    def load(cls, path: str | Path) -> "Suite":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def validate(self) -> None:
        Client._validate(
            self.prompt.strip(),
            list(self.examples),
            list(dict.fromkeys(model.strip() for model in self.models if model.strip())),
            self.quality_threshold,
            self.max_optimization_cost_usd,
            self.rounds,
            self.minimum_meaningful_quality_gain,
        )
        if self.objective not in {
            "cheapest_passing", "cheapest_at_accuracy", "lowest_cost_at_accuracy",
            "highest_quality", "best_within_cost", "best_within_price", "constrained",
            "match_baseline_at_lowest_cost",
        }:
            raise ValueError("objective is not a supported cost/accuracy policy.")
        if not 0 <= self.max_few_shot_examples <= len(self.examples):
            raise ValueError("max_few_shot_examples must fit within the approved suite.")
        if self.incumbent_model and self.incumbent_model not in self.models:
            raise ValueError("incumbent_model must be one of the suite models.")
        if not 0 <= self.allowed_accuracy_regression < 1:
            raise ValueError("allowed_accuracy_regression must be between zero and one.")
        if not 1 <= self.holdout_repeats <= 5:
            raise ValueError("holdout_repeats must be between 1 and 5.")
        if not 1 <= self.max_parallel_models <= 32:
            raise ValueError("max_parallel_models must be between 1 and 32.")
        if not 1 <= self.max_parallel_scenarios <= 128:
            raise ValueError("max_parallel_scenarios must be between 1 and 128.")
        if not 0 < self.request_timeout_seconds <= 7200:
            raise ValueError("request_timeout_seconds must be greater than zero and no more than 7200 seconds.")
        if not 1 <= int(self.target_max_tokens) <= 131072:
            raise ValueError("target_max_tokens must be between 1 and 131072.")
        normalize_request_options(self.request_options)
        if self.max_p90_latency_seconds is not None and self.max_p90_latency_seconds <= 0:
            raise ValueError("max_p90_latency_seconds must be positive when provided.")
        if self.latency_value_usd_per_second < 0:
            raise ValueError("latency_value_usd_per_second cannot be negative.")
        _validate_evaluator_policy(dict(self.evaluator))
        if any(not str(name).strip() or not 0 < float(floor) <= 1 for name, floor in self.difficulty_thresholds.items()):
            raise ValueError("difficulty_thresholds must map non-empty names to values greater than zero and at most one.")
        if self.evidence_provenance not in {
            "HUMAN_APPROVED",
            "HUMAN_APPROVED_AI_DRAFT",
            "AI_GENERATED_AI_JUDGED",
        }:
            raise ValueError("evidence_provenance is not a supported trust level.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "evalt-suite-v1",
            "name": self.name,
            "prompt": self.prompt,
            "examples": [asdict(example) for example in self.examples],
            "models": list(self.models),
            "optimizer_model": self.optimizer_model,
            "evaluator_model": self.evaluator_model,
            "evaluator": dict(self.evaluator),
            "difficulty_thresholds": dict(self.difficulty_thresholds),
            "objective": self.objective,
            "quality_threshold": self.quality_threshold,
            "max_optimization_cost_usd": self.max_optimization_cost_usd,
            "rounds": self.rounds,
            "optimize_prompt": self.optimize_prompt,
            "holdout_repeats": self.holdout_repeats,
            "max_parallel_models": self.max_parallel_models,
            "max_parallel_scenarios": self.max_parallel_scenarios,
            "request_timeout_seconds": self.request_timeout_seconds,
            "target_max_tokens": self.target_max_tokens,
            "request_options": normalize_request_options(self.request_options),
            "request_options_sha256": request_options_fingerprint(self.request_options),
            "max_p90_latency_seconds": self.max_p90_latency_seconds,
            "latency_value_usd_per_second": self.latency_value_usd_per_second,
            "minimum_meaningful_quality_gain": self.minimum_meaningful_quality_gain,
            "allow_few_shot": self.allow_few_shot,
            "max_few_shot_examples": self.max_few_shot_examples,
            "incumbent_model": self.incumbent_model,
            "allowed_accuracy_regression": self.allowed_accuracy_regression,
            "adaptive_search": self.adaptive_search,
            "evidence_provenance": self.evidence_provenance,
        }

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def optimize_kwargs(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "examples": self.examples,
            "models": self.models,
            "optimizer_model": self.optimizer_model,
            "evaluator_model": self.evaluator_model,
            "evaluator": dict(self.evaluator),
            "difficulty_thresholds": dict(self.difficulty_thresholds),
            "objective": self.objective,
            "quality_threshold": self.quality_threshold,
            "max_optimization_cost_usd": self.max_optimization_cost_usd,
            "rounds": self.rounds,
            "optimize_prompt": self.optimize_prompt,
            "holdout_repeats": self.holdout_repeats,
            "max_parallel_models": self.max_parallel_models,
            "max_parallel_scenarios": self.max_parallel_scenarios,
            "target_max_tokens": self.target_max_tokens,
            "request_options": normalize_request_options(self.request_options),
            "max_p90_latency_seconds": self.max_p90_latency_seconds,
            "latency_value_usd_per_second": self.latency_value_usd_per_second,
            "minimum_meaningful_quality_gain": self.minimum_meaningful_quality_gain,
            "allow_few_shot": self.allow_few_shot,
            "max_few_shot_examples": self.max_few_shot_examples,
            "incumbent_model": self.incumbent_model,
            "allowed_accuracy_regression": self.allowed_accuracy_regression,
            "adaptive_search": self.adaptive_search,
        }


@dataclass(frozen=True)
class SuiteDraft:
    """AI-authored cases that cannot count as human evidence until approved."""

    task: str
    prompt: str
    examples: tuple[Example, ...]
    models: tuple[str, ...]
    designer_model: str
    evaluator_model: str
    evaluator: Mapping[str, Any]
    quality_threshold: float
    workflow_budget_usd: float
    designer_spend_usd: float
    objective: str = "lowest_cost_at_accuracy"
    optimize_prompt: bool = True
    holdout_repeats: int = 2
    max_parallel_models: int = 16
    max_parallel_scenarios: int = 32
    request_timeout_seconds: float = 120
    target_max_tokens: int = 600
    request_options: Mapping[str, Any] = field(default_factory=dict)
    max_p90_latency_seconds: float | None = None
    latency_value_usd_per_second: float = 0.0
    allow_few_shot: bool = True
    max_few_shot_examples: int = 3
    name: str = "evalt-suite"
    evidence_provenance: str = "AI_DRAFT_UNAPPROVED"
    design_notes: tuple[str, ...] = ()
    judge_calibration_checks: int = 0

    @property
    def remaining_optimization_budget_usd(self) -> float:
        return max(0.0, self.workflow_budget_usd - self.designer_spend_usd)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SuiteDraft":
        try:
            draft = cls(
                name=str(value.get("name") or "evalt-suite"),
                task=str(value["task"]),
                prompt=str(value["prompt"]),
                examples=tuple(
                    Example.from_value(item, index)
                    for index, item in enumerate(value["examples"])
                ),
                models=tuple(str(item) for item in value["models"]),
                designer_model=str(value["designer_model"]),
                evaluator_model=str(value["evaluator_model"]),
                evaluator=dict(value.get("evaluator") or {"type": "semantic"}),
                quality_threshold=float(value.get("quality_threshold", 0.95)),
                workflow_budget_usd=float(value["workflow_budget_usd"]),
                designer_spend_usd=float(value.get("designer_spend_usd", 0)),
                objective=str(value.get("objective", "lowest_cost_at_accuracy")),
                optimize_prompt=bool(value.get("optimize_prompt", True)),
                holdout_repeats=int(value.get("holdout_repeats", 2)),
                max_parallel_models=int(value.get("max_parallel_models", 16)),
                max_parallel_scenarios=int(value.get("max_parallel_scenarios", 32)),
                request_timeout_seconds=float(value.get("request_timeout_seconds", 120)),
                target_max_tokens=int(value.get("target_max_tokens", 600)),
                request_options=normalize_request_options(value.get("request_options")),
                max_p90_latency_seconds=(float(value["max_p90_latency_seconds"]) if value.get("max_p90_latency_seconds") is not None else None),
                latency_value_usd_per_second=float(value.get("latency_value_usd_per_second", 0.0)),
                allow_few_shot=bool(value.get("allow_few_shot", True)),
                max_few_shot_examples=int(value.get("max_few_shot_examples", 3)),
                evidence_provenance="AI_DRAFT_UNAPPROVED",
                design_notes=tuple(str(item) for item in value.get("design_notes", [])),
                judge_calibration_checks=int(value.get("judge_calibration_checks", 0)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid Evalt suite draft: {error}") from error
        if len(draft.examples) < 5:
            raise ValueError("An AI suite draft must contain at least five scenarios.")
        if draft.remaining_optimization_budget_usd <= 0:
            raise ValueError("The suite draft has no remaining tournament budget.")
        _validate_evaluator_policy(dict(draft.evaluator))
        return draft

    @classmethod
    def load(cls, path: str | Path) -> "SuiteDraft":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "evalt-suite-draft-v1",
            "name": self.name,
            "task": self.task,
            "prompt": self.prompt,
            "examples": [asdict(example) for example in self.examples],
            "models": list(self.models),
            "designer_model": self.designer_model,
            "evaluator_model": self.evaluator_model,
            "evaluator": dict(self.evaluator),
            "quality_threshold": self.quality_threshold,
            "workflow_budget_usd": self.workflow_budget_usd,
            "designer_spend_usd": self.designer_spend_usd,
            "remaining_optimization_budget_usd": self.remaining_optimization_budget_usd,
            "objective": self.objective,
            "optimize_prompt": self.optimize_prompt,
            "holdout_repeats": self.holdout_repeats,
            "max_parallel_models": self.max_parallel_models,
            "max_parallel_scenarios": self.max_parallel_scenarios,
            "request_timeout_seconds": self.request_timeout_seconds,
            "target_max_tokens": self.target_max_tokens,
            "request_options": normalize_request_options(self.request_options),
            "request_options_sha256": request_options_fingerprint(self.request_options),
            "max_p90_latency_seconds": self.max_p90_latency_seconds,
            "latency_value_usd_per_second": self.latency_value_usd_per_second,
            "allow_few_shot": self.allow_few_shot,
            "max_few_shot_examples": self.max_few_shot_examples,
            "evidence_provenance": self.evidence_provenance,
            "design_notes": list(self.design_notes),
            "judge_calibration_checks": self.judge_calibration_checks,
        }

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def approve(
        self,
        examples: Iterable[Example | Mapping[str, Any]] | None = None,
    ) -> Suite:
        """Explicitly approve the draft, optionally replacing/editing its cases."""
        approved = tuple(
            Example.from_value(item, index)
            for index, item in enumerate(examples if examples is not None else self.examples)
        )
        suite = Suite(
            name=self.name,
            prompt=self.prompt,
            examples=approved,
            models=self.models,
            optimizer_model=self.designer_model,
            evaluator_model=self.evaluator_model,
            evaluator=dict(self.evaluator),
            objective=self.objective,
            quality_threshold=self.quality_threshold,
            max_optimization_cost_usd=self.remaining_optimization_budget_usd,
            optimize_prompt=self.optimize_prompt,
            holdout_repeats=self.holdout_repeats,
            max_parallel_models=self.max_parallel_models,
            max_parallel_scenarios=self.max_parallel_scenarios,
            request_timeout_seconds=self.request_timeout_seconds,
            target_max_tokens=self.target_max_tokens,
            request_options=normalize_request_options(self.request_options),
            max_p90_latency_seconds=self.max_p90_latency_seconds,
            latency_value_usd_per_second=self.latency_value_usd_per_second,
            allow_few_shot=self.allow_few_shot,
            max_few_shot_examples=self.max_few_shot_examples,
            evidence_provenance="HUMAN_APPROVED_AI_DRAFT",
        )
        suite.validate()
        return suite

    def autopilot_suite(self) -> Suite:
        """Use the AI-authored contract without implying human verification."""
        suite = Suite(
            name=self.name,
            prompt=self.prompt,
            examples=self.examples,
            models=self.models,
            optimizer_model=self.designer_model,
            evaluator_model=self.evaluator_model,
            evaluator=dict(self.evaluator),
            objective=self.objective,
            quality_threshold=self.quality_threshold,
            max_optimization_cost_usd=self.remaining_optimization_budget_usd,
            optimize_prompt=self.optimize_prompt,
            holdout_repeats=self.holdout_repeats,
            max_parallel_models=self.max_parallel_models,
            max_parallel_scenarios=self.max_parallel_scenarios,
            request_timeout_seconds=self.request_timeout_seconds,
            target_max_tokens=self.target_max_tokens,
            request_options=normalize_request_options(self.request_options),
            max_p90_latency_seconds=self.max_p90_latency_seconds,
            latency_value_usd_per_second=self.latency_value_usd_per_second,
            allow_few_shot=self.allow_few_shot,
            max_few_shot_examples=self.max_few_shot_examples,
            evidence_provenance="AI_GENERATED_AI_JUDGED",
        )
        suite.validate()
        return suite


class Evalt:
    """Execute durable routes or run explicit optimization suites.

    ``run(Suite(...))`` remains the explicit optimizer.  ``run(prompt, input, ...)``
    is the primary production surface: it serves the remembered qualified route and
    records the model/prompt decision in local durable state.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        transport: ChatTransport | None = None,
        state_path: str | Path = ".evalt/evalt.db",
        request_timeout_seconds: float = 600,
        show_progress: bool | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        dashboard_token: str | None = None,
        dashboard_api_url: str | None = None,
    ) -> None:
        if transport is not None and request_timeout_seconds != 600:
            raise ValueError("request_timeout_seconds cannot be combined with a custom transport.")
        resolved_transport = transport or OpenRouterTransport(
            api_key=api_key, timeout_seconds=request_timeout_seconds
        )
        self.client = Client(api_key=api_key, transport=resolved_transport)
        self._state_path = Path(state_path)
        self._router: DurableRouter | None = None
        self._show_progress = (
            bool(getattr(sys.stderr, "isatty", lambda: False)())
            if show_progress is None else bool(show_progress)
        )
        self._progress_callback = progress_callback
        dashboard_config = None
        if dashboard_token is None:
            try:
                dashboard_config = load_dashboard_config(self._state_path)
            except (OSError, ValueError):
                # A missing, partial, or manually damaged optional dashboard config
                # cannot prevent the local production router from starting.
                dashboard_config = None
        resolved_dashboard_token = dashboard_token or (dashboard_config or {}).get("workspace_token")
        resolved_dashboard_url = dashboard_api_url or (dashboard_config or {}).get("api_url") or DEFAULT_DASHBOARD_API_URL
        self._dashboard_sync = (
            WorkspaceSync(resolved_dashboard_token, api_url=resolved_dashboard_url)
            if resolved_dashboard_token and resolved_dashboard_url
            else None
        )
        self._dashboard_run_local = threading.local()
        self._maintenance_guard = threading.Lock()
        self._maintenance_routes: set[str] = set()
        self._maintenance_threads: list[threading.Thread] = []

    def _emit_dashboard_status(self, route: str, status: str) -> None:
        """Report hosted visibility locally without recursively syncing the report."""

        if not self._show_progress:
            return
        workspace_id = str(
            getattr(self._dashboard_sync, "workspace_id", "unknown workspace")
        )
        if status == "connected":
            message = (
                f"Evalt · {route} · HOSTED WORKSPACE {workspace_id} · "
                "syncing private route metadata"
            )
        elif status == "synced":
            message = (
                f"Evalt · {route} · DASHBOARD SYNCED · {workspace_id} · "
                "open with: evalt dashboard"
            )
        elif status == "failed":
            detail = str(getattr(self._dashboard_sync, "last_error", "") or "")
            detail = f" · {detail}" if detail else ""
            message = (
                f"Evalt · {route} · DASHBOARD SYNC FAILED · {workspace_id} · "
                f"the local route is safe{detail}"
            )
        else:
            message = (
                f"Evalt · {route} · LOCAL WORKSPACE ONLY · "
                "run `evalt connect` to show this route at evalt.dev"
            )
        print(message, file=sys.stderr, flush=True)

    def _emit_progress(self, event: dict[str, Any]) -> None:
        event = dict(event)
        run_context = getattr(self._dashboard_run_local, "context", None)
        if run_context is not None:
            for key in (
                "run_id", "run_state", "run_started_at", "run_finished_at",
            ):
                if key in run_context:
                    event.setdefault(key, run_context[key])
        if self._dashboard_sync is not None:
            self._dashboard_sync.publish_event(event)
        if self._progress_callback is not None:
            self._progress_callback(dict(event))
        if not self._show_progress:
            return
        kind = str(event.get("event") or "progress")
        route = str(event.get("route") or "route")
        if kind == "production_call_started":
            spent = float(event.get("test_budget_spent_usd") or 0)
            message = (
                f"Evalt · {route} · serving one production call after "
                f"${spent:.6f} initial test spend"
                if spent
                else f"Evalt · {route} · bootstrap-only production call; no tournament spend"
            )
        elif kind == "production_call_completed":
            model = str(event.get("model") or "model")
            cost = float(event.get("provider_cost_usd") or 0)
            ceiling = float(event.get("effective_price_ceiling_usd") or 0)
            policy = str(event.get("price_policy") or "explicit")
            raw_reason = str(event.get("decision_reason") or "route")
            feedback_count = int(event.get("feedback_count") or 0)
            min_feedback = int(event.get("min_feedback") or 0)
            route_phase = str(event.get("route_phase") or "untested_bootstrap")
            if route_phase == "untested_bootstrap":
                message = (
                    f"Evalt · {route} · UNTESTED BOOTSTRAP · one provider call only · "
                    f"{model} · ${cost:.6f} · {feedback_count}/{min_feedback} labeled examples · "
                    "no tournament ran"
                )
            elif route_phase == "ai_tested":
                message = (
                    f"Evalt · {route} · AI-TESTED ROUTE · {model} · ${cost:.6f} · "
                    f"{policy} ceiling ${ceiling:.6f} · human feedback can strengthen the contract"
                )
            else:
                message = (
                    f"Evalt · {route} · QUALIFIED ROUTE · {model} · ${cost:.6f} · "
                    f"{policy} ceiling ${ceiling:.6f}"
                )
            if event.get("request_envelope_validated") is False:
                message += " | WARNING: request settings differ from the tested route"
        elif kind == "production_call_failed":
            message = f"Evalt · {route} · stopped before completion: {event.get('error')}"
        elif kind == "feedback_recorded":
            count = int(event.get("feedback_count") or 0)
            minimum = int(event.get("min_feedback") or 0)
            remaining = max(0, minimum - count)
            if remaining:
                message = (
                    f"Evalt · {route} · {event.get('verdict')} feedback saved · "
                    f"{count}/{minimum} labeled examples · {remaining} more before the first tournament"
                )
            else:
                message = (
                    f"Evalt · {route} · {event.get('verdict')} feedback saved · "
                    f"{count}/{minimum} labeled examples · tournament eligible"
                )
        elif kind == "suite_design_started":
            deadline = event.get("designer_timeout_seconds")
            deadline_text = (
                f" · deadline {float(deadline):g}s" if deadline is not None else ""
            )
            message = (
                f"Evalt · {route} · TEST DESIGN STARTED · "
                f"{int(event.get('case_count') or 0)} cases · "
                f"{event.get('designer_model', 'designer route')}{deadline_text} · "
                f"one workflow cap ${float(event.get('workflow_budget_usd') or 0):.2f}"
            )
        elif kind == "suite_design_attempt_started":
            message = (
                f"Evalt · {route} · DESIGNING TESTS · "
                f"{event.get('designer_model', 'model')} · "
                f"attempt {int(event.get('attempt') or 1)}/"
                f"{int(event.get('max_attempts') or 1)} · request started"
            )
        elif kind == "suite_design_heartbeat":
            message = (
                f"Evalt · {route} · DESIGNING TESTS · "
                f"{event.get('designer_model', 'model')} · "
                f"{float(event.get('elapsed_seconds') or 0):.0f}s elapsed · still working"
            )
        elif kind == "suite_design_completed":
            message = (
                f"Evalt · {route} · TEST DRAFT READY · "
                f"{int(event.get('case_count') or 0)} AI-authored cases · "
                f"spent ${float(event.get('designer_spend_usd') or 0):.6f} · "
                f"${float(event.get('remaining_budget_usd') or 0):.6f} remains for the tournament"
            )
        elif kind == "suite_designer_unavailable":
            message = (
                f"Evalt · {route} · DESIGNER ROUTE UNAVAILABLE · "
                f"{event.get('designer_model', 'model')} · trying the next cost-qualified role"
            )
        elif kind == "suite_designer_invalid":
            next_action = (
                "retrying this model within the workflow cap"
                if event.get("will_retry")
                else "trying the next cost-qualified role"
            )
            message = (
                f"Evalt · {route} · TEST DRAFT REJECTED · "
                f"{event.get('designer_model', 'model')} · "
                f"attempt {int(event.get('attempt') or 1)}/"
                f"{int(event.get('max_attempts') or 1)} · invalid structured output · "
                f"{next_action}"
            )
        elif kind == "judge_calibration_started":
            message = (
                f"Evalt · {route} · CALIBRATING JUDGE · "
                f"{event.get('evaluator_model', 'model')} · "
                f"{int(event.get('checks') or 0)} known pass/fail checks"
            )
        elif kind == "judge_calibration_completed":
            status = "PASSED" if event.get("passed") else "REJECTED"
            message = (
                f"Evalt · {route} · JUDGE {status} · "
                f"{event.get('evaluator_model', 'model')} · "
                f"{int(event.get('matched_checks') or 0)}/"
                f"{int(event.get('checks') or 0)} checks matched"
            )
        elif kind == "initial_optimization_started":
            message = (
                f"Evalt · {route} · FIRST-ROUTE OPTIMIZATION · AI is designing and testing "
                f"{int(event.get('case_count') or 0)} cases under one "
                f"${float(event.get('workflow_budget_usd') or 0):.2f} cap"
            )
        elif kind == "initial_optimization_completed":
            message = (
                f"Evalt · {route} · ROUTE SELECTED · {event.get('winner_model')} · "
                f"{float(event.get('holdout_pass_rate') or 0):.0%} final test · "
                f"${float(event.get('workflow_spend_usd') or 0):.6f} test spend"
            )
        elif kind == "maintenance_started":
            message = (
                f"Evalt · {route} · TOURNAMENT STARTED · "
                f"cap ${float(event['test_budget_usd']):.2f}"
            )
        elif kind == "maintenance_completed":
            message = (
                f"Evalt · {route} · TOURNAMENT COMPLETE · "
                f"{event.get('promoted_model') or 'no route promoted'} · "
                f"spent ${float(event.get('provider_spend_usd') or 0):.6f}"
            )
        elif kind == "maintenance_skipped":
            message = f"Evalt · {route} · NO TOURNAMENT RAN · {event.get('reason')}"
        elif kind == "maintenance_failed":
            message = f"Evalt · {route} · bounded retest stopped: {event.get('error')}"
        elif kind == "model_completed":
            final_scenarios = int(event.get("final_test_scenarios") or 0)
            final_result = (
                f"{float(event.get('final_test_pass_rate') or 0):.0%} final test · "
                f"{final_scenarios} scenario(s) / "
                f"{int(event.get('final_test_executions') or 0)} execution(s)"
                if final_scenarios
                else "final confirmation not run · validation did not qualify"
            )
            message = (
                f"Evalt · {route} · {event.get('model', 'model')} · "
                f"{final_result} · {int(event.get('prompt_candidates_tested') or 1)} "
                f"prompt package(s) · ${float(event.get('optimization_spend_usd') or 0):.6f} spent"
            )
        elif kind == "model_started":
            work = (
                "prompt search + final test"
                if event.get("optimize_prompt", True)
                else "shared prompt + final test"
            )
            message = (
                f"Evalt · {route} · DEEP TEST STARTED · "
                f"{event.get('model', 'model')} · {work}"
            )
        elif kind in {"reasoning_escalation_started", "reasoning_escalation_skipped"}:
            decision = "TRYING" if kind.endswith("started") else "SKIPPING"
            measured = event.get("validation_pass_rate")
            measured_text = (
                f"{float(measured):.0%} validation"
                if measured is not None else "no completed prior rung"
            )
            latency = event.get("target_latency_p90_ms")
            latency_text = (
                f" · p90 {int(latency)} ms" if latency is not None else ""
            )
            message = (
                f"Evalt · {route} · {decision} {event.get('to_effort')} REASONING · "
                f"{event.get('model', 'model')} · {measured_text}{latency_text} · "
                f"{event.get('reason', '')}"
            )
        elif kind == "final_confirmation_started":
            message = (
                f"Evalt · {route} · FINAL CONFIRMATION · {event.get('model', 'model')} · "
                f"{int(event.get('unique_scenarios') or 0)} unseen scenario(s) · "
                f"{int(event.get('executions') or 0)} execution(s) · prompt and model frozen"
            )
        elif kind == "final_confirmation_skipped":
            message = (
                f"Evalt · {route} · FINAL TEST SKIPPED · {event.get('model', 'model')} · "
                f"{float(event.get('validation_pass_rate') or 0):.0%} validation did not clear "
                f"{float(event.get('quality_threshold') or 0):.0%}"
            )
        elif kind == "broad_screen_started":
            message = (
                f"Evalt · {route} · BROAD SCREEN · {int(event.get('configurations') or 0)} "
                f"model configuration(s) · up to {int(event.get('parallel_models') or 1)} in parallel"
            )
        elif kind == "model_screen_completed":
            message = (
                f"Evalt · {route} · SCREENED · {event.get('model', 'model')} · "
                f"{float(event.get('validation_pass_rate') or 0):.0%} validation · "
                f"p90 {int(event.get('target_latency_p90_ms') or 0)} ms · "
                f"${float(event.get('screening_spend_usd') or 0):.6f} spent"
            )
        elif kind == "broad_screen_completed":
            message = (
                f"Evalt · {route} · BROAD SCREEN COMPLETE · "
                f"{int(event.get('completed_configurations') or 0)}/"
                f"{int(event.get('configurations') or 0)} configuration(s) settled · "
                f"{float(event.get('elapsed_seconds') or 0):.1f}s elapsed"
            )
        elif kind in {"model_unavailable", "model_incomplete"}:
            label = "UNAVAILABLE" if kind == "model_unavailable" else "INCOMPLETE"
            message = (
                f"Evalt · {route} · {label} · {event.get('model', 'model')} · "
                f"{event.get('reason', 'provider did not settle')}"
            )
        elif kind == "prompt_candidate_completed":
            candidate = int(event.get("candidate") or 0)
            label = "original prompt" if candidate == 0 else f"prompt rewrite {candidate}"
            decision = "selected so far" if event.get("selected") else "not selected"
            message = (
                f"Evalt · {route} · {event.get('model', 'model')} · {label} · "
                f"{float(event.get('validation_pass_rate') or 0):.0%} validation · "
                f"{int(event.get('few_shot_examples') or 0)} example(s) · {decision}"
            )
        else:
            return
        print(message, file=sys.stderr, flush=True)

    def _maintain_in_background(self, **kwargs: Any) -> None:
        route = str(kwargs.get("route") or "route")
        try:
            result = self.router.maintain(**kwargs)
            if result is None:
                status = self.router.status(route)
                last_event = status["decisions"][-1] if status["decisions"] else {}
                event_type = last_event.get("event_type")
                if event_type == "judge_calibration_waiting":
                    reason = (
                        "semantic judging needs at least two approved outputs and one corrected "
                        "failure; no provider budget was spent"
                    )
                elif event_type == "judge_calibration_failed":
                    reason = "the judge failed calibration; the existing route was kept"
                else:
                    reason = "another maintenance run owns this route or the evidence was not ready"
                self._emit_progress({
                    "event": "maintenance_skipped", "route": route, "reason": reason
                })
            else:
                self._emit_progress({
                    "event": "maintenance_completed",
                    "route": route,
                    "provider_spend_usd": result.total_provider_spend_usd,
                    "promoted_model": result.winner.model,
                })
        except Exception as error:  # background failures stay visible without killing the caller
            self._emit_progress({
                "event": "maintenance_failed", "route": route, "error": str(error)
            })
        finally:
            with self._maintenance_guard:
                self._maintenance_routes.discard(route)

    def _start_maintenance(self, *, route: str, test_budget_usd: float, **kwargs: Any) -> bool:
        with self._maintenance_guard:
            if route in self._maintenance_routes:
                return False
            self._maintenance_routes.add(route)
        self._emit_progress({
            "event": "maintenance_started",
            "route": route,
            "test_budget_usd": test_budget_usd,
        })
        thread = threading.Thread(
            target=self._maintain_in_background,
            kwargs={"route": route, "test_budget_usd": test_budget_usd, **kwargs},
            name=f"evalt-maintain-{route}",
            # Do not abandon already-authorized provider work when a short script exits.
            daemon=False,
        )
        with self._maintenance_guard:
            self._maintenance_threads.append(thread)
        thread.start()
        return True

    def wait_for_maintenance(self) -> None:
        """Wait for currently launched bounded tournaments to finish."""
        with self._maintenance_guard:
            threads = list(self._maintenance_threads)
        for thread in threads:
            thread.join()

    def design_suite(
        self,
        *,
        task: str,
        prompt: str,
        route: str = "evalt-suite",
        seed_examples: Iterable[Example | Mapping[str, Any]] = (),
        representative_inputs: Iterable[Any] = (),
        case_count: int = 25,
        workflow_budget_usd: float = 1.00,
        quality_threshold: float = 0.95,
        models: Iterable[str] | None = None,
        designer_model: str | None = None,
        evaluator_model: str | None = None,
        evaluator: Mapping[str, Any] | None = None,
        objective: str = "lowest_cost_at_accuracy",
        optimize_prompt: bool = True,
        holdout_repeats: int = 2,
        max_parallel_models: int = 16,
        max_parallel_scenarios: int = 32,
        request_timeout_seconds: float = 120,
        target_max_tokens: int = 600,
        request_options: Mapping[str, Any] | None = None,
        max_p90_latency_seconds: float | None = None,
        latency_value_usd_per_second: float = 0.0,
        allow_few_shot: bool = True,
        max_few_shot_examples: int = 3,
    ) -> SuiteDraft:
        """Use a smart model to draft a reviewable, budget-accounted test contract."""
        task_text = str(task).strip()
        prompt_text = str(prompt).strip()
        if len(task_text) < 8:
            raise ValueError("task must explain the recurring job in at least eight characters.")
        if len(prompt_text) < 8:
            raise ValueError("prompt must contain at least eight characters.")
        if not 5 <= int(case_count) <= 100:
            raise ValueError("case_count must be between 5 and 100; use 25 or more for five final-test cases.")
        if not 0 < float(workflow_budget_usd) <= 100:
            raise ValueError("workflow_budget_usd must be greater than zero and no more than 100.")
        if not 0 < float(quality_threshold) <= 1:
            raise ValueError("quality_threshold must be greater than zero and at most one.")
        if not 1 <= int(target_max_tokens) <= 131072:
            raise ValueError("target_max_tokens must be between 1 and 131072.")
        normalized_request_options = normalize_request_options(request_options)
        seeds = tuple(
            Example.from_value(item, index)
            for index, item in enumerate(seed_examples)
        )
        if len(seeds) > int(case_count):
            raise ValueError("seed_examples cannot outnumber case_count.")
        representative_context: list[dict[str, Any]] = []
        for raw_input in tuple(representative_inputs)[:3]:
            content = (
                raw_input
                if isinstance(raw_input, str)
                else json.dumps(
                    raw_input,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            representative_context.append({
                "content": content[:12000],
                "original_characters": len(content),
                "truncated": len(content) > 12000,
            })

        requested_models = tuple(models) if models is not None else DEFAULT_TARGETS
        catalog: list[Mapping[str, Any]] = []
        # Target candidates and orchestration roles are separate decisions. The
        # outer Evalt.run call passes an already-shortlisted target set here, but
        # the designer must still use the live catalog instead of falling back to
        # a static role merely because target models are explicit.
        if hasattr(self.client.transport, "model_catalog"):
            catalog = self.client.transport.model_catalog()
        role_plan = select_role_plan(
            catalog,
            maintenance_budget_usd=float(workflow_budget_usd),
            fallback_targets=requested_models,
        )
        if models is None and role_plan.catalog_revision != "fallback":
            requested_models = role_plan.target_models
        requested_models = tuple(dict.fromkeys(str(item).strip() for item in requested_models if str(item).strip()))
        if not requested_models:
            raise ValueError("At least one target model is required.")
        selected_designer = designer_model or role_plan.test_designer_model
        selected_evaluator = evaluator_model or role_plan.judge_model
        generated_count = int(case_count) - len(seeds)
        budget = _Budget(float(workflow_budget_usd))
        drafted_examples: list[Example] = []
        design_notes: list[str] = []
        suggested_evaluator: dict[str, Any] = dict(evaluator or {})
        judge_calibration_checks = 0

        if generated_count:
            use_generated_groups = int(case_count) >= 25 and not seeds
            self._emit_progress({
                "event": "suite_design_started",
                "route": route,
                "case_count": int(case_count),
                "workflow_budget_usd": float(workflow_budget_usd),
                "designer_model": selected_designer,
                "designer_timeout_seconds": getattr(
                    self.client.transport, "timeout_seconds", None
                ),
            })
            schema = {
                "type": "object",
                "additionalProperties": False,
                "required": ["evaluator", "judge_calibration", "scenarios", "design_notes"],
                "properties": {
                    "evaluator": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["type", "reason", "required_keys", "allow_additional_properties", "normalize_rational_strings"],
                        "properties": {
                            "type": {"type": "string", "enum": ["semantic", "exact_text", "exact_json", "numeric_tolerance"]},
                            "reason": {"type": "string"},
                            "required_keys": {"type": "array", "items": {"type": "string"}},
                            "allow_additional_properties": {"type": "boolean"},
                            "normalize_rational_strings": {"type": "boolean"},
                            "minimum": {"type": "number"},
                            "maximum": {"type": "number"},
                            "absolute_tolerance": {"type": "number", "exclusiveMinimum": 0},
                        },
                    },
                    "judge_calibration": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 6,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["input", "approved_output", "candidate_output", "should_pass"],
                            "properties": {
                                "input": {"type": "string"},
                                "approved_output": {"type": "string"},
                                "candidate_output": {"type": "string"},
                                "should_pass": {"type": "boolean"},
                            },
                        },
                    },
                    "scenarios": {
                        "type": "array",
                        "minItems": generated_count,
                        "maxItems": generated_count,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "group", "difficulty", "critical", "turns", "rationale"],
                            "properties": {
                                "id": {"type": "string"},
                                "group": {"type": "string"},
                                "difficulty": {"type": "string", "enum": ["routine", "complex", "adversarial"]},
                                "critical": {"type": "boolean"},
                                "turns": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 4,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["input", "approved_output"],
                                        "properties": {
                                            "input": {"type": "string"},
                                            "approved_output": {"type": "string"},
                                        },
                                    },
                                },
                                "rationale": {"type": "string"},
                            },
                        },
                    },
                    "design_notes": {"type": "array", "items": {"type": "string"}},
                },
            }
            payload = {
                "task": task_text,
                "current_prompt": prompt_text,
                "required_new_scenarios": generated_count,
                "seed_examples": [asdict(item) for item in seeds],
                "representative_inputs_without_labels": representative_context,
                "quality_target": quality_threshold,
            }
            coverage_focuses = (
                "Routine and direct requests",
                "Ambiguous and context-dependent requests",
                "Adversarial, sarcastic, or deceptive requests",
                "Boundary, neutral, and near-tie requests",
                "Realistic vocabulary, tone, and domain variation",
            )
            if use_generated_groups:
                base_count, remainder = divmod(generated_count, len(coverage_focuses))
                job_specs = [
                    (focus, base_count + int(index < remainder))
                    for index, focus in enumerate(coverage_focuses)
                ]
            else:
                job_specs = [("Exploratory coverage", generated_count)]
            design_jobs: list[tuple[str, int, dict[str, Any], list[dict[str, str]], int]] = []
            for focus, job_count in job_specs:
                requested_job_count = job_count + 2 if use_generated_groups else job_count
                job_schema = copy.deepcopy(schema)
                job_schema["properties"]["scenarios"]["minItems"] = requested_job_count
                job_schema["properties"]["scenarios"]["maxItems"] = requested_job_count
                job_payload = dict(payload)
                job_payload.update({
                    "required_new_scenarios": requested_job_count,
                    "coverage_focus": focus,
                    "required_output_shape": (
                        f"Exactly {requested_job_count} candidate scenarios for this coverage focus; "
                        f"Evalt will retain five valid, contract-faithful cases. "
                        f"Set every scenario group to {focus!r}."
                    ),
                })
                group_instruction = (
                    f"This is one of five parallel coverage batches. Focus only on {focus}. "
                    f"Return exactly {requested_job_count} distinct candidate scenarios and set every group to {focus!r}. "
                    "The other batches cover different behavior, so do not broaden into their territory."
                    if use_generated_groups
                    else "Set every scenario group to an empty string for this small exploratory draft."
                )
                job_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Design a balanced evaluation suite batch for a recurring production AI task. "
                            "Create genuinely distinct cases; do not merely paraphrase seeds. "
                            f"{group_instruction} "
                            "Preserve the customer's exact task contract. A scenario input is data the production "
                            "prompt will receive, not a new instruction that changes the requested output shape. "
                            "Never invent JSON, tables, multiple scores, explanations, IDs, or multi-turn behavior "
                            "unless the current task or prompt explicitly requires it. If the task returns one scalar "
                            "or one label, every approved output must contain exactly that one scalar or label. "
                            "Expected outputs describe the desired behavior, not what the current prompt happens to "
                            "produce. Make each case concrete enough for a human to approve or edit. All "
                            "cases are drafted before any train/validation/final-test split, so never label "
                            "or target a split. Use numeric_tolerance for a scalar rating or score where nearby "
                            "values are equivalent; include the stated minimum, maximum, and a defensible absolute "
                            "tolerance (twenty percent for subjective human ratings such as sentiment; tighter for objective values). Recommend exact_text or exact_json only "
                            "when equivalent answers truly must match that deterministic contract; otherwise use semantic. "
                            "Also create judge-calibration checks outside the scenario suite: at least two "
                            "clear passes and one clear failure, each labeled with should_pass. A pass must "
                            "fully satisfy every material requirement. A failure must be unmistakably wrong. "
                            "These are unapproved AI drafts. Representative inputs have no approved outputs; use them "
                            "only to understand realistic shape, length, and domain, and do not copy them "
                            "into the suite. Return only the required JSON."
                        ),
                    },
                    {"role": "user", "content": json.dumps(job_payload, ensure_ascii=False)},
                ]
                design_jobs.append((
                    focus,
                    job_count,
                    job_schema,
                    job_messages,
                    min(12000, max(4000, requested_job_count * 600)),
                ))
            designer_candidates = (
                (selected_designer,)
                if designer_model is not None
                else tuple(dict.fromkeys(
                    (selected_designer, *role_plan.designer_candidates)
                ))
            )
            designer_failures: list[str] = []
            completion = None
            parsed: dict[str, Any] | None = None
            max_structured_attempts = 2
            for designer_candidate in designer_candidates:
                for structured_attempt in range(1, max_structured_attempts + 1):
                    self._emit_progress({
                        "event": "suite_design_attempt_started",
                        "route": route,
                        "designer_model": designer_candidate,
                        "attempt": structured_attempt,
                        "max_attempts": max_structured_attempts,
                    })
                    attempt_started = time.monotonic()
                    try:
                        with ThreadPoolExecutor(max_workers=len(design_jobs)) as pool:
                            futures = [
                                pool.submit(
                                    self.client._call,
                                    budget,
                                    designer_candidate,
                                    job_messages,
                                    max_tokens=job_max_tokens,
                                    response_schema=job_schema,
                                )
                                for _focus, _count, job_schema, job_messages, job_max_tokens
                                in design_jobs
                            ]
                            candidate_completions: list[Completion] = []
                            for future in futures:
                                while True:
                                    try:
                                        candidate_completions.append(
                                            future.result(timeout=10)
                                        )
                                        break
                                    except FutureTimeoutError:
                                        self._emit_progress({
                                            "event": "suite_design_heartbeat",
                                            "route": route,
                                            "designer_model": designer_candidate,
                                            "attempt": structured_attempt,
                                            "max_attempts": max_structured_attempts,
                                            "elapsed_seconds": round(
                                                time.monotonic() - attempt_started, 1
                                            ),
                                        })
                    except ProviderError as error:
                        designer_failures.append(f"{designer_candidate}: {error}")
                        self._emit_progress({
                            "event": "suite_designer_unavailable",
                            "route": route,
                            "designer_model": designer_candidate,
                            "error": str(error),
                        })
                        break

                    try:
                        candidate_payloads: list[dict[str, Any]] = []
                        for job_index, (
                            focus, job_count, _job_schema, _job_messages, _job_tokens
                        ) in enumerate(design_jobs):
                            candidate_text = str(
                                candidate_completions[job_index].content
                            ).strip()
                            if candidate_text.startswith("```"):
                                candidate_text = (
                                    candidate_text.split("\n", 1)[-1]
                                    .rsplit("```", 1)[0]
                                    .strip()
                                )
                            job_payload = json.loads(candidate_text)
                            if not isinstance(job_payload, dict):
                                raise TypeError("designer payload must be an object")
                            required_payload_keys = {
                                "evaluator", "judge_calibration", "design_notes", "scenarios",
                            }
                            if not required_payload_keys.issubset(job_payload):
                                raise KeyError("designer payload is missing required fields")
                            job_scenarios = list(job_payload["scenarios"])
                            if len(job_scenarios) < job_count:
                                raise ValueError(
                                    f"{focus} returned {len(job_scenarios)} scenarios; needed at least {job_count}"
                                )
                            for scenario_index, raw_scenario in enumerate(job_scenarios):
                                scenario = dict(raw_scenario)
                                scenario["id"] = (
                                    f"batch-{job_index + 1}-{scenario_index + 1}-"
                                    f"{str(scenario.get('id') or 'case')}"
                                )
                                scenario["group"] = focus if use_generated_groups else ""
                                job_scenarios[scenario_index] = scenario
                            job_payload["scenarios"] = job_scenarios
                            candidate_evaluator = dict(job_payload["evaluator"])
                            if str(candidate_evaluator.get("type") or "") == "numeric_tolerance":
                                missing_scale = any(
                                    candidate_evaluator.get(key) is None
                                    for key in ("minimum", "maximum", "absolute_tolerance")
                                )
                                if missing_scale:
                                    inferred_scale = _infer_numeric_scale_contract(
                                        task_text, prompt_text
                                    )
                                    if inferred_scale is not None:
                                        minimum, maximum = inferred_scale
                                        candidate_evaluator.update({
                                            "minimum": minimum,
                                            "maximum": maximum,
                                            "absolute_tolerance": _default_numeric_tolerance(
                                                task_text, prompt_text, minimum, maximum
                                            ),
                                        })
                                        job_payload.setdefault("design_notes", []).append(
                                            "Evalt recovered the explicit numeric scale from the customer task contract and applied a task-sensitive equivalence tolerance."
                                        )
                            job_payload["evaluator"] = _validate_evaluator_policy(
                                candidate_evaluator
                            )
                            if job_payload["evaluator"]["type"] == "numeric_tolerance":
                                minimum = float(job_payload["evaluator"]["minimum"])
                                maximum = float(job_payload["evaluator"]["maximum"])
                                allows_multiturn = any(
                                    marker in f"{task_text} {prompt_text}".casefold()
                                    for marker in (
                                        "multi-turn", "multiturn", "conversation",
                                        "chat history", "previous message", "prior message",
                                    )
                                )
                                valid_scenarios: list[dict[str, Any]] = []
                                for scenario in job_scenarios:
                                    turns = list(scenario.get("turns") or [])
                                    if not allows_multiturn and len(turns) != 1:
                                        continue
                                    scenario_valid = True
                                    for turn in turns:
                                        scalar = _extract_single_numeric_scalar(
                                            turn.get("approved_output")
                                        )
                                        if scalar is None:
                                            scenario_valid = False
                                            break
                                        if not minimum <= scalar <= maximum:
                                            scenario_valid = False
                                            break
                                        turn["approved_output"] = f"{scalar:g}"
                                    if scenario_valid:
                                        valid_scenarios.append(scenario)
                                if len(valid_scenarios) < job_count:
                                    raise ValueError(
                                        f"{focus} produced fewer than {job_count} valid numeric scenarios"
                                    )
                                job_payload["scenarios"] = valid_scenarios[:job_count]
                            else:
                                job_payload["scenarios"] = job_scenarios[:job_count]
                            candidate_payloads.append(job_payload)
                        # The first batch defines one evaluator contract for the
                        # whole suite. Parallel batches may describe the same
                        # scalar as JSON or prose; normalize every approved answer
                        # against the shared contract before any split or spend.
                        shared_evaluator = dict(candidate_payloads[0]["evaluator"])
                        for job_payload in candidate_payloads:
                            job_payload["evaluator"] = dict(shared_evaluator)
                        if shared_evaluator["type"] == "numeric_tolerance":
                            minimum = float(shared_evaluator["minimum"])
                            maximum = float(shared_evaluator["maximum"])
                            for (focus, job_count, *_rest), job_payload in zip(
                                design_jobs, candidate_payloads
                            ):
                                valid_scenarios = []
                                for scenario in job_payload["scenarios"]:
                                    scenario_valid = True
                                    for turn in scenario.get("turns") or []:
                                        scalar = _extract_single_numeric_scalar(
                                            turn.get("approved_output")
                                        )
                                        if scalar is None:
                                            scenario_valid = False
                                            break
                                        if not minimum <= scalar <= maximum:
                                            scenario_valid = False
                                            break
                                        turn["approved_output"] = f"{scalar:g}"
                                    if scenario_valid:
                                        valid_scenarios.append(scenario)
                                if len(valid_scenarios) < job_count:
                                    raise ValueError(
                                        f"{focus} did not preserve enough one-scalar approved outputs"
                                    )
                                job_payload["scenarios"] = valid_scenarios[:job_count]
                        candidate_payload = dict(candidate_payloads[0])
                        if use_generated_groups:
                            candidate_payload["strata"] = [
                                {
                                    "group": focus,
                                    "scenarios": [
                                        {
                                            key: value for key, value in scenario.items()
                                            if key != "group"
                                        }
                                        for scenario in job_payload["scenarios"]
                                    ],
                                }
                                for (focus, _count, *_rest), job_payload
                                in zip(design_jobs, candidate_payloads)
                            ]
                            candidate_payload["design_notes"] = [
                                str(note)
                                for job_payload in candidate_payloads
                                for note in job_payload.get("design_notes", [])
                            ]
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                        will_retry = structured_attempt < max_structured_attempts
                        designer_failures.append(
                            f"{designer_candidate} attempt {structured_attempt}: "
                            f"invalid structured output ({error})"
                        )
                        self._emit_progress({
                            "event": "suite_designer_invalid",
                            "route": route,
                            "designer_model": designer_candidate,
                            "attempt": structured_attempt,
                            "max_attempts": max_structured_attempts,
                            "will_retry": will_retry,
                            "error": str(error),
                        })
                        continue

                    completion = candidate_completions[0]
                    parsed = candidate_payload
                    selected_designer = designer_candidate
                    break
                if completion is not None:
                    break
            if completion is None:
                raise ProviderError(
                    "No cost-qualified suite designer completed: "
                    + "; ".join(designer_failures)
                )
            try:
                if parsed is None:
                    raise ValueError("designer payload was not decoded")
                if use_generated_groups:
                    strata = list(parsed["strata"])
                    if len(strata) != 5:
                        raise ValueError("wrong stratum count")
                    group_names = [
                        str(item.get("group") or "").strip() for item in strata
                    ]
                    if any(not name for name in group_names) or len(set(group_names)) != 5:
                        raise ValueError("behavior strata must have five distinct names")
                    scenarios = []
                    for stratum, group_name in zip(strata, group_names):
                        for raw_scenario in list(stratum.get("scenarios") or []):
                            scenario = dict(raw_scenario)
                            scenario["group"] = group_name
                            scenarios.append(scenario)
                else:
                    scenarios = list(parsed["scenarios"])
                if len(scenarios) != generated_count:
                    raise ValueError("wrong scenario count")
                for index, item in enumerate(scenarios):
                    drafted_examples.append(Example.from_value(item, len(seeds) + index))
                    design_notes.append(f"{item.get('id')}: {str(item.get('rationale') or '').strip()}")
                design_notes.extend(str(item).strip() for item in parsed.get("design_notes", []) if str(item).strip())
                if not suggested_evaluator:
                    raw_evaluator = dict(parsed["evaluator"])
                    suggested_evaluator = {"type": raw_evaluator.get("type", "semantic")}
                    if suggested_evaluator["type"] == "exact_json":
                        suggested_evaluator.update({
                            "required_keys": list(raw_evaluator.get("required_keys") or []),
                            "allow_additional_properties": bool(raw_evaluator.get("allow_additional_properties", True)),
                            "normalize_rational_strings": bool(raw_evaluator.get("normalize_rational_strings", False)),
                        })
                    elif suggested_evaluator["type"] == "numeric_tolerance":
                        suggested_evaluator.update({
                            "minimum": raw_evaluator.get("minimum"),
                            "maximum": raw_evaluator.get("maximum"),
                            "absolute_tolerance": raw_evaluator.get("absolute_tolerance"),
                        })
                        _validate_evaluator_policy(suggested_evaluator)
                    design_notes.insert(0, f"Evaluator: {str(raw_evaluator.get('reason') or '').strip()}")
                calibration_rows = list(parsed.get("judge_calibration") or [])
                evaluator_type = str(suggested_evaluator.get("type") or "semantic")
                if evaluator_type in {"exact_text", "exact_json", "numeric_tolerance"}:
                    # AI-authored labels are unnecessary and can be internally
                    # inconsistent for deterministic evaluators. Construct known
                    # anchors whose outcomes follow directly from the evaluator's
                    # executable contract.
                    anchor_examples = drafted_examples[:2]
                    if len(anchor_examples) < 2:
                        raise ValueError("insufficient deterministic calibration anchors")
                    calibration_rows = []
                    for anchor in anchor_examples:
                        turn = anchor.conversation()[0]
                        calibration_rows.append({
                            "input": turn.input,
                            "approved_output": turn.approved_output,
                            "candidate_output": turn.approved_output,
                            "should_pass": True,
                        })
                    failure_turn = anchor_examples[0].conversation()[0]
                    if evaluator_type == "numeric_tolerance":
                        expected_score = _extract_single_numeric_scalar(
                            failure_turn.approved_output
                        )
                        if expected_score is None:
                            raise ValueError(
                                "numeric approved output did not contain one unambiguous scalar"
                            )
                        minimum = float(suggested_evaluator["minimum"])
                        maximum = float(suggested_evaluator["maximum"])
                        tolerance = float(suggested_evaluator["absolute_tolerance"])
                        failure_score = (
                            minimum
                            if abs(expected_score - minimum) > tolerance
                            else maximum
                        )
                        failure_output = f"{failure_score:g}"
                    else:
                        failure_output = (
                            "{not valid json"
                            if evaluator_type == "exact_json"
                            else failure_turn.approved_output
                            + "\n__EVALT_INTENTIONALLY_WRONG__"
                        )
                    calibration_rows.append({
                        "input": failure_turn.input,
                        "approved_output": failure_turn.approved_output,
                        "candidate_output": failure_output,
                        "should_pass": False,
                    })
                    evaluator_candidates = (f"deterministic/{evaluator_type}",)
                    design_notes.insert(
                        1,
                        f"Deterministic calibration: Evalt constructed two known passes and one known failure for {evaluator_type}.",
                    )
                else:
                    if len(calibration_rows) < 3:
                        raise ValueError("insufficient judge calibration")
                    known_passes = [
                        dict(item) for item in calibration_rows
                        if bool(item.get("should_pass"))
                    ]
                    known_failures = [
                        dict(item) for item in calibration_rows
                        if not bool(item.get("should_pass"))
                    ]
                    if len(known_passes) < 2 or not known_failures:
                        raise ValueError(
                            "semantic calibration requires two known passes and one known failure"
                        )
                    # A designer model cannot declare a numerically or materially
                    # different answer a "known pass."  The pass controls are
                    # deterministic identity checks; AI still proposes the clearly
                    # wrong negative controls that the judge must reject.
                    for calibration in known_passes:
                        calibration["candidate_output"] = calibration["approved_output"]
                        calibration["should_pass"] = True
                    calibration_rows = [*known_passes, *known_failures]
                    design_notes.insert(
                        1,
                        "Semantic calibration: Evalt made every known-pass control identical to its approved answer.",
                    )
                    if selected_evaluator == selected_designer:
                        raise ProviderError(
                            "AI-generated semantic suites require a judge model different from the suite designer; no correlated self-judge fallback ran."
                        )
                    evaluator_candidates = (selected_evaluator,)
                calibrated_evaluator: str | None = None
                calibration_summaries: list[str] = []
                for candidate in evaluator_candidates:
                    self._emit_progress({
                        "event": "judge_calibration_started",
                        "route": route,
                        "evaluator_model": candidate,
                        "checks": len(calibration_rows),
                    })
                    matched = True
                    matched_checks = 0
                    mismatch_reason = ""
                    for calibration_index, calibration in enumerate(calibration_rows):
                        calibration_example = Example.from_value({
                            "id": f"judge-calibration-{calibration_index + 1}",
                            "input": calibration["input"],
                            "approved_output": calibration["approved_output"],
                        }, calibration_index)
                        judgment, _completion = self.client._judge(
                            calibration_example,
                            calibration_example.conversation()[0],
                            0,
                            [],
                            str(calibration["candidate_output"]),
                            candidate,
                            budget,
                            dict(suggested_evaluator),
                        )
                        if judgment.passed is not bool(calibration["should_pass"]):
                            matched = False
                            mismatch_reason = (
                                f"check {calibration_index + 1}: expected "
                                f"{'pass' if calibration['should_pass'] else 'fail'}, got "
                                f"{'pass' if judgment.passed else 'fail'} ({judgment.reason[:160]})"
                            )
                            break
                        matched_checks += 1
                    self._emit_progress({
                        "event": "judge_calibration_completed",
                        "route": route,
                        "evaluator_model": candidate,
                        "checks": len(calibration_rows),
                        "matched_checks": matched_checks,
                        "passed": matched,
                        "mismatch_reason": mismatch_reason,
                    })
                    calibration_summaries.append(
                        f"{candidate}: {matched_checks}/{len(calibration_rows)}"
                        + (f"; {mismatch_reason}" if mismatch_reason else "")
                    )
                    if matched:
                        calibrated_evaluator = candidate
                        break
                if calibrated_evaluator is None:
                    raise ProviderError(
                        "No candidate judge passed the calibration checks; no tournament ran. "
                        + " | ".join(calibration_summaries)
                    )
                selected_evaluator = calibrated_evaluator
                judge_calibration_checks = len(calibration_rows)
                design_notes.insert(
                    1,
                    f"Judge calibration: {selected_evaluator} matched {len(calibration_rows)} labeled checks.",
                )
                if designer_failures:
                    design_notes.insert(
                        2,
                        f"Designer fallback: {selected_designer} completed after {len(designer_failures)} unavailable route(s).",
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ProviderError("The test designer returned an invalid structured suite; no draft was approved.") from error

        if generated_count and not (int(case_count) >= 25 and not seeds):
            drafted_examples = [replace(item, group="") for item in drafted_examples]

        all_examples = (*seeds, *drafted_examples)
        ids = [item.id for item in all_examples]
        if len(set(ids)) != len(ids):
            raise ProviderError("The test designer returned duplicate scenario IDs; no draft was approved.")
        if generated_count >= 25 and not seeds:
            generated_group_counts: dict[str, int] = {}
            for item in drafted_examples:
                if item.group.strip():
                    generated_group_counts[item.group.strip()] = (
                        generated_group_counts.get(item.group.strip(), 0) + 1
                    )
            if (
                len(generated_group_counts) < 5
                or any(count < 5 for count in generated_group_counts.values())
            ):
                raise ProviderError(
                    "The test designer did not provide at least five behavior strata with five cases each; no draft was approved."
                )
        if not suggested_evaluator:
            suggested_evaluator = {"type": "semantic"}
        _validate_evaluator_policy(suggested_evaluator)
        draft = SuiteDraft(
            name=route,
            task=task_text,
            prompt=prompt_text,
            examples=tuple(all_examples),
            models=requested_models,
            designer_model=selected_designer,
            evaluator_model=selected_evaluator,
            evaluator=suggested_evaluator,
            quality_threshold=float(quality_threshold),
            workflow_budget_usd=float(workflow_budget_usd),
            designer_spend_usd=round(budget.spent_usd, 10),
            objective=objective,
            optimize_prompt=bool(optimize_prompt),
            holdout_repeats=int(holdout_repeats),
            max_parallel_models=int(max_parallel_models),
            max_parallel_scenarios=int(max_parallel_scenarios),
            request_timeout_seconds=float(request_timeout_seconds),
            target_max_tokens=int(target_max_tokens),
            request_options=normalized_request_options,
            max_p90_latency_seconds=max_p90_latency_seconds,
            latency_value_usd_per_second=float(latency_value_usd_per_second),
            allow_few_shot=bool(allow_few_shot),
            max_few_shot_examples=int(max_few_shot_examples),
            design_notes=tuple(design_notes),
            judge_calibration_checks=judge_calibration_checks,
        )
        if draft.remaining_optimization_budget_usd <= 0:
            raise BudgetExceeded("Test design consumed the workflow cap before a tournament could run.")
        self._emit_progress({
            "event": "suite_design_completed",
            "route": route,
            "case_count": len(draft.examples),
            "designer_spend_usd": draft.designer_spend_usd,
            "remaining_budget_usd": draft.remaining_optimization_budget_usd,
            "evidence_provenance": draft.evidence_provenance,
            "judge_calibrated": draft.judge_calibration_checks >= 3,
            "judge_calibration_checks": draft.judge_calibration_checks,
            "evaluator_model": draft.evaluator_model,
        })
        return draft

    def optimize_task(
        self,
        *,
        task: str,
        prompt: str,
        case_control: str = "review",
        **kwargs: Any,
    ) -> SuiteDraft | OptimizationResult:
        """Draft a suite for review, or run a clearly labeled AI-only exploratory flow."""
        if case_control not in {"review", "autopilot"}:
            raise ValueError("case_control must be 'review' or 'autopilot'.")
        draft = self.design_suite(task=task, prompt=prompt, **kwargs)
        if case_control == "review":
            return draft
        result = self.run(draft.autopilot_suite())
        result.regression_suite["designer_spend_usd"] = draft.designer_spend_usd
        result.regression_suite["tournament_spend_usd"] = result.total_provider_spend_usd
        result.regression_suite["workflow_budget_usd"] = draft.workflow_budget_usd
        result.regression_suite["evidence_provenance"] = "AI_GENERATED_AI_JUDGED"
        result.total_provider_spend_usd = round(
            draft.designer_spend_usd + result.total_provider_spend_usd, 10
        )
        result.warnings.insert(
            0,
            "Autopilot used AI-generated cases and AI judging; treat this as directional evidence until a human approves the contract or real feedback reproduces it.",
        )
        return result

    @property
    def router(self) -> DurableRouter:
        if self._router is None:
            self._router = DurableRouter(self.client, state_path=self._state_path)
        return self._router

    @_dashboard_run_scope
    def run(
        self,
        suite_or_prompt: Suite | str,
        input: Any | None = None,
        *,
        route: str = "default",
        incumbent_model: str | None = None,
        price_usd: float | None = None,
        test_budget_usd: float | str = "auto",
        max_test_budget_usd: float = 1.00,
        target_accuracy: float = 0.95,
        objective: str = "lowest_cost_at_accuracy",
        optimize_prompt: bool = True,
        max_latency_seconds: float | None = None,
        max_p90_latency_seconds: float | None = None,
        latency_value_usd_per_second: float = 0.0,
        models: Iterable[str] | None = None,
        max_tokens: int | None = None,
        request_options: Mapping[str, Any] | None = None,
        strict_request_options: bool = False,
        budget_usd: float | None = None,
        quality_threshold: float | None = None,
        retest_after_calls: int = 500,
        min_feedback: int = 5,
        maintenance_budget_usd: float | None = None,
        auto_maintain: bool = True,
        task: str | None = None,
        first_run: str = "optimize",
        case_count: int = 25,
        optimization_rounds: int = 1,
        designer_model: str | None = None,
        evaluator_model: str | None = None,
        test_request_timeout_seconds: float = 120,
        designer_request_timeout_seconds: float = 45,
    ) -> OptimizationResult | RoutedAnswer:
        if isinstance(suite_or_prompt, Suite):
            if input is not None:
                raise ValueError("input is not used when running an explicit Suite.")
            suite_or_prompt.validate()
            if isinstance(self.client.transport, OpenRouterTransport):
                self.client.transport.set_timeout_seconds(
                    suite_or_prompt.request_timeout_seconds
                )
                self.client.transport.set_performance_policy(
                    preferred_max_latency_seconds=suite_or_prompt.max_p90_latency_seconds,
                    provider_sort=(
                        "latency" if suite_or_prompt.latency_value_usd_per_second > 0
                        else "price"
                    ),
                )
            optimize_kwargs = suite_or_prompt.optimize_kwargs()
            # A connected workspace is a progress consumer even when this is a
            # non-interactive script with console logging disabled. Without this
            # branch the dashboard received only the coarse outer run events and a
            # long tournament looked frozen until its final snapshot arrived.
            if (
                self._show_progress
                or self._progress_callback is not None
                or self._dashboard_sync is not None
            ):
                def route_progress(event: dict[str, Any]) -> None:
                    scoped = dict(event)
                    scoped.setdefault("route", suite_or_prompt.name)
                    self._emit_progress(scoped)

                optimize_kwargs["progress_callback"] = route_progress
            result = self.client.optimize(**optimize_kwargs)
            result.regression_suite["evidence_provenance"] = suite_or_prompt.evidence_provenance
            if suite_or_prompt.evidence_provenance == "AI_GENERATED_AI_JUDGED":
                result.warnings.insert(
                    0,
                    "This suite was AI-generated and AI-judged; it is directional evidence, not a human-verified regression contract.",
                )
            return result
        if input is None:
            raise ValueError("input is required when executing a prompt through Evalt.")
        normalized_request_options = (
            None if request_options is None else normalize_request_options(request_options)
        )
        if max_tokens is not None and not 1 <= int(max_tokens) <= 131072:
            raise ValueError("max_tokens must be between 1 and 131072.")
        if first_run not in {"optimize", "bootstrap"}:
            raise ValueError("first_run must be 'optimize' or 'bootstrap'.")
        if not 0 < float(test_request_timeout_seconds) <= 7200:
            raise ValueError(
                "test_request_timeout_seconds must be greater than zero and no more than 7200 seconds."
            )
        if not 0 < float(designer_request_timeout_seconds) <= 7200:
            raise ValueError(
                "designer_request_timeout_seconds must be greater than zero and no more than 7200 seconds."
            )
        if not 1 <= int(optimization_rounds) <= 8:
            raise ValueError("optimization_rounds must be between one and eight.")
        if isinstance(self.client.transport, OpenRouterTransport):
            self.client.transport.set_performance_policy(
                preferred_max_latency_seconds=max_p90_latency_seconds,
                provider_sort=(
                    "latency" if latency_value_usd_per_second > 0 else "price"
                ),
            )
        if price_usd is not None and budget_usd is not None and float(price_usd) != float(budget_usd):
            raise ValueError("Use price_usd; budget_usd is only a backward-compatible alias.")
        max_cost_per_run_usd = (
            float(price_usd if price_usd is not None else budget_usd)
            if price_usd is not None or budget_usd is not None
            else None
        )
        if quality_threshold is not None:
            if target_accuracy != 0.95 and float(target_accuracy) != float(quality_threshold):
                raise ValueError("Use target_accuracy; quality_threshold is only a backward-compatible alias.")
            target_accuracy = float(quality_threshold)
        if max_latency_seconds is not None:
            if (
                max_p90_latency_seconds is not None
                and float(max_latency_seconds) != float(max_p90_latency_seconds)
            ):
                raise ValueError(
                    "Use max_latency_seconds or max_p90_latency_seconds, not conflicting values."
                )
            max_p90_latency_seconds = float(max_latency_seconds)
        if max_p90_latency_seconds is not None and max_p90_latency_seconds <= 0:
            raise ValueError("max_latency_seconds must be positive when provided.")
        if latency_value_usd_per_second < 0:
            raise ValueError("latency_value_usd_per_second cannot be negative.")
        if maintenance_budget_usd is not None:
            if test_budget_usd != "auto" and float(test_budget_usd) != float(maintenance_budget_usd):
                raise ValueError("Use test_budget_usd; maintenance_budget_usd is only a backward-compatible alias.")
            test_budget_usd = float(maintenance_budget_usd)
        if not 0 < max_test_budget_usd <= 100:
            raise ValueError("max_test_budget_usd must be greater than 0 and no more than 100.")
        if test_budget_usd == "auto":
            if max_cost_per_run_usd is None:
                resolved_test_budget_usd = min(float(max_test_budget_usd), 1.00)
                test_budget_policy = (
                    "auto: up to $1.00 for first-route testing and bounded retests when no production price ceiling is set"
                )
            else:
                resolved_test_budget_usd = min(
                    float(max_test_budget_usd),
                    max(
                        0.25,
                        max_cost_per_run_usd * max(1, int(retest_after_calls)) * 0.10,
                    ),
                )
                test_budget_policy = (
                    "auto: 10% of one retest interval's production ceiling, floored at $0.25"
                )
        else:
            resolved_test_budget_usd = float(test_budget_usd)
            if not 0 <= resolved_test_budget_usd <= float(max_test_budget_usd):
                raise ValueError("test_budget_usd must be non-negative and no greater than max_test_budget_usd.")
            test_budget_policy = "explicit"
        requested_models = tuple(models) if models is not None else DEFAULT_TARGETS
        catalog: list[Mapping[str, Any]] = []
        if models is None and hasattr(self.client.transport, "model_catalog"):
            catalog = self.client.transport.model_catalog()
        role_plan = select_role_plan(
            catalog,
            maintenance_budget_usd=resolved_test_budget_usd,
            fallback_targets=requested_models,
        )
        if models is None and role_plan.catalog_revision != "fallback":
            requested_models = role_plan.target_models
        if incumbent_model:
            requested_models = tuple(dict.fromkeys((incumbent_model, *requested_models)))
        initial_test_spend_usd = 0.0
        try:
            if (
                first_run == "optimize"
                and resolved_test_budget_usd > 0
                and self.router.needs_initial_optimization(route, suite_or_prompt)
            ):
                self._emit_progress({
                    "event": "initial_optimization_started",
                    "route": route,
                    "case_count": int(case_count),
                    "workflow_budget_usd": resolved_test_budget_usd,
                })
                if isinstance(self.client.transport, OpenRouterTransport):
                    # Suite design is a one-time orchestration job and may need
                    # materially longer than an acceptable production response.
                    self.client.transport.set_timeout_seconds(
                        float(designer_request_timeout_seconds)
                    )
                draft = self.design_suite(
                    task=(task or suite_or_prompt),
                    prompt=suite_or_prompt,
                    route=route,
                    case_count=int(case_count),
                    workflow_budget_usd=resolved_test_budget_usd,
                    quality_threshold=target_accuracy,
                    models=requested_models,
                    designer_model=designer_model,
                    evaluator_model=evaluator_model,
                    representative_inputs=(input,),
                    objective=objective,
                    optimize_prompt=bool(optimize_prompt),
                    max_p90_latency_seconds=max_p90_latency_seconds,
                    latency_value_usd_per_second=latency_value_usd_per_second,
                    request_timeout_seconds=float(test_request_timeout_seconds),
                    target_max_tokens=int(max_tokens or 600),
                    request_options=normalized_request_options,
                )
                if max_tokens is None:
                    draft = replace(
                        draft,
                        target_max_tokens=_automatic_target_max_tokens(
                            draft.evaluator, draft.examples
                        ),
                    )
                initial_result = self.run(replace(
                    draft.autopilot_suite(), rounds=int(optimization_rounds)
                ))
                initial_test_spend_usd = round(
                    draft.designer_spend_usd + initial_result.total_provider_spend_usd,
                    10,
                )
                try:
                    summary = self.router.install_initial_result(
                        route=route,
                        prompt=suite_or_prompt,
                        models=requested_models,
                        quality_threshold=target_accuracy,
                        catalog_revision=role_plan.catalog_revision,
                        result=initial_result,
                        examples=draft.examples,
                        evidence_provenance="AI_GENERATED_AI_JUDGED",
                        total_workflow_spend_usd=initial_test_spend_usd,
                        designer_model=draft.designer_model,
                        evaluator_model=draft.evaluator_model,
                        judge_calibration_checks=draft.judge_calibration_checks,
                        target_max_tokens=draft.target_max_tokens,
                        request_options=draft.request_options,
                    )
                except ValueError as error:
                    raise ProviderError(str(error)) from error
                self._emit_progress({
                    "event": "initial_optimization_completed",
                    "route": route,
                    **summary,
                })
            if isinstance(self.client.transport, OpenRouterTransport):
                # Candidate tests and the production call use the tighter task
                # deadline. A timed-out effort cannot earn a higher reasoning rung.
                self.client.transport.set_timeout_seconds(
                    float(test_request_timeout_seconds)
                )
            self._emit_progress({
                "event": "production_call_started",
                "route": route,
                "target_accuracy": target_accuracy,
                "test_budget_usd": resolved_test_budget_usd,
                "test_budget_policy": test_budget_policy,
                "test_budget_spent_usd": initial_test_spend_usd,
            })
            answer = self.router.run(
                route=route,
                prompt=suite_or_prompt,
                input=input,
                max_cost_per_run_usd=max_cost_per_run_usd,
                models=requested_models,
                max_tokens=max_tokens,
                request_options=normalized_request_options,
                strict_request_options=bool(strict_request_options),
                target_accuracy=target_accuracy,
                objective=objective,
                optimize_prompt=bool(optimize_prompt),
                test_budget_usd=resolved_test_budget_usd,
                test_budget_policy=test_budget_policy,
                max_p90_latency_seconds=max_p90_latency_seconds,
                latency_value_usd_per_second=latency_value_usd_per_second,
                retest_after_calls=retest_after_calls,
                min_feedback=min_feedback,
                catalog_revision=role_plan.catalog_revision,
            )
        except Exception as error:
            self._emit_progress({
                "event": "production_call_failed", "route": route, "error": str(error)
            })
            # A one-shot script can exit immediately after the exception. Give an
            # explicitly connected dashboard a brief chance to receive the failure,
            # without ever turning dashboard availability into a provider failure.
            self.flush_dashboard(timeout_seconds=4.0)
            raise
        status = self.router.status(
            route, retest_after_calls=retest_after_calls, min_feedback=min_feedback
        )
        if self._dashboard_sync is not None:
            self._dashboard_sync.publish_route(status)
        self._emit_progress({
            "event": "production_call_completed",
            "route": route,
            "model": answer.model,
            "provider_cost_usd": answer.provider_cost_usd,
            "price_policy": status["price_policy"],
            "effective_price_ceiling_usd": status["effective_price_ceiling_usd"],
            "decision_reason": answer.decision_reason,
            "maintenance_due": list(answer.maintenance_due),
            "feedback_count": status["feedback_count"],
            "min_feedback": min_feedback,
            "route_phase": answer.route_phase,
            "evidence_provenance": answer.evidence_provenance,
            "request_envelope_validated": answer.request_envelope_validated,
            "request_options_sha256": answer.request_options_sha256,
        })
        # Most SDK examples are short scripts. The daemon worker keeps long-running
        # services fast, while this bounded flush makes the final route visible even
        # when Python exits on the next line.
        self.flush_dashboard(timeout_seconds=8.0)

        def on_feedback(receipt: dict[str, Any]) -> None:
            self._emit_progress(receipt)
            if self._dashboard_sync is not None:
                self._dashboard_sync.publish_route(self.router.status(route))
            if (
                auto_maintain
                and receipt.get("is_new")
                and resolved_test_budget_usd > 0
                and receipt.get("maintenance_due")
                and int(receipt.get("feedback_count") or 0) >= min_feedback
            ):
                self._start_maintenance(
                    route=route,
                    test_budget_usd=resolved_test_budget_usd,
                    role_plan=role_plan,
                    objective=objective,
                    max_cost_per_run_usd=max_cost_per_run_usd,
                    min_feedback=min_feedback,
                )
            self.flush_dashboard(timeout_seconds=5.0)

        answer._on_feedback = on_feedback
        return answer

    def maintain(self, route: str, *, test_budget_usd: float | None = None, maintenance_budget_usd: float | None = None, role_plan: RolePlan, objective: str = "cheapest_passing", max_cost_per_run_usd: float | None = None, rounds: int = 3, min_feedback: int = 5) -> OptimizationResult | None:
        """Run one bounded retest and promote only a route-specific passing winner."""
        if test_budget_usd is not None and maintenance_budget_usd is not None and float(test_budget_usd) != float(maintenance_budget_usd):
            raise ValueError("Use test_budget_usd; maintenance_budget_usd is only a backward-compatible alias.")
        resolved = test_budget_usd if test_budget_usd is not None else maintenance_budget_usd
        if resolved is None:
            raise ValueError("test_budget_usd is required for an explicit maintenance run.")
        return self.router.maintain(route, test_budget_usd=float(resolved), role_plan=role_plan, objective=objective, max_cost_per_run_usd=max_cost_per_run_usd, rounds=rounds, min_feedback=min_feedback)

    def route_status(self, route: str, *, retest_after_calls: int = 500, min_feedback: int = 5) -> dict[str, Any]:
        status = self.router.status(route, retest_after_calls=retest_after_calls, min_feedback=min_feedback)
        if self._dashboard_sync is not None:
            self._dashboard_sync.publish_route(status)
            self.flush_dashboard(timeout_seconds=5.0)
        return status

    def flush_dashboard(self, timeout_seconds: float = 10.0) -> bool:
        """Wait briefly for explicitly enabled dashboard metadata sync."""
        return True if self._dashboard_sync is None else self._dashboard_sync.flush(timeout_seconds)

    def draft(
        self,
        *,
        task: str,
        input: str,
        model: str = "openai/gpt-5-mini",
        max_cost_usd: float = 0.10,
    ) -> DraftAnswer:
        return self.client.draft_answer(
            task=task, input=input, model=model, max_cost_usd=max_cost_usd
        )


@dataclass(frozen=True)
class GateReport:
    passed: bool
    failures: tuple[str, ...] = field(default_factory=tuple)
    holdout_pass_rate: float = 0.0
    estimated_cost_per_successful_call_usd: float | None = None
    winner_scope: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_result(
    value: Mapping[str, Any],
    *,
    min_pass_rate: float = 0.95,
    max_cost_per_success_usd: float | None = None,
    require_complete_coverage: bool = False,
) -> GateReport:
    """Evaluate an exported SDK or web report without making a provider call."""

    winner = (
        value.get("winner")
        or value.get("best_candidate")
        or value.get("selected")
        or value.get("report")
        or value
    )
    if "selected" in winner and isinstance(winner["selected"], Mapping):
        winner = winner["selected"]
    holdout = winner.get("holdout_pass_rate") or winner.get("pass_rate")
    if holdout is None:
        holdout_record = winner.get("selectedHoldout") or winner.get("selected_holdout") or {}
        holdout = holdout_record.get("passRate") or holdout_record.get("pass_rate") or 0
    cost = (
        winner.get("estimated_cost_per_successful_call_usd")
        if winner.get("estimated_cost_per_successful_call_usd") is not None
        else winner.get("cost_per_success_usd")
    )
    scope = str(value.get("winner_scope") or winner.get("winner_scope") or "")
    failures: list[str] = []
    pass_rate = float(holdout or 0)
    if pass_rate < min_pass_rate:
        failures.append(
            f"holdout pass rate {pass_rate:.3f} is below required {min_pass_rate:.3f}"
        )
    if bool(value.get("exploratory")):
        failures.append("result is exploratory because the distinct final-test set is too small")
    if max_cost_per_success_usd is not None:
        if cost is None:
            failures.append("result does not contain estimated cost per successful call")
        elif float(cost) > max_cost_per_success_usd:
            failures.append(
                f"cost per successful call ${float(cost):.6f} exceeds ${max_cost_per_success_usd:.6f}"
            )
    normalized_scope = scope.lower()
    partial = (
        ("fully completed" in normalized_scope and "only" in normalized_scope)
        or bool(value.get("incomplete_models"))
        or bool(value.get("unavailable_models"))
        or bool(value.get("skipped_budget_models"))
        or value.get("coverage_complete") is False
    )
    if require_complete_coverage and partial:
        failures.append("the result covers only fully completed targets")
    return GateReport(
        passed=not failures,
        failures=tuple(failures),
        holdout_pass_rate=pass_rate,
        estimated_cost_per_successful_call_usd=float(cost) if cost is not None else None,
        winner_scope=scope,
    )


__all__ = [
    "BudgetExceeded",
    "CaseResult",
    "ChatTransport",
    "Client",
    "Completion",
    "DraftAnswer",
    "Evalt",
    "Example",
    "GateReport",
    "Judgment",
    "ModelResult",
    "OpenRouterTransport",
    "OptimizationResult",
    "ProviderError",
    "RequestEnvelopeDriftWarning",
    "RolePlan",
    "RoutedAnswer",
    "Suite",
    "Turn",
    "check_result",
    "select_role_plan",
    "_safe_provider_error_detail",
]
