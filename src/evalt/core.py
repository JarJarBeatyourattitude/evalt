"""Typed, primary Evalt SDK surface.

The optimization engine remains import-compatible with the two earlier package names;
new code should use :class:`Suite` and :class:`Evalt` from this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping

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
    _safe_provider_error_detail,
    _validate_evaluator_policy,
)
from .router import DEFAULT_TARGETS, DurableRouter, RolePlan, RoutedAnswer, select_role_plan


@dataclass(frozen=True)
class Suite:
    """A reviewable optimization contract that can be validated without an API key."""

    prompt: str
    examples: tuple[Example, ...]
    models: tuple[str, ...]
    optimizer_model: str = "openai/gpt-5.6-luna"
    evaluator_model: str = "openai/gpt-5.6-luna"
    evaluator: Mapping[str, Any] = field(default_factory=lambda: {"type": "semantic"})
    objective: str = "lowest_cost_at_accuracy"
    quality_threshold: float = 0.95
    max_optimization_cost_usd: float = 2.00
    rounds: int = 3
    holdout_repeats: int = 2
    max_parallel_models: int = 8
    max_parallel_scenarios: int = 16
    request_timeout_seconds: float = 600
    minimum_meaningful_quality_gain: float = 0.03
    allow_few_shot: bool = True
    max_few_shot_examples: int = 3
    incumbent_model: str | None = None
    allowed_accuracy_regression: float = 0.0
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
                objective=str(value.get("objective", "lowest_cost_at_accuracy")),
                quality_threshold=float(value.get("quality_threshold", 0.95)),
                max_optimization_cost_usd=float(value.get("max_optimization_cost_usd", 2.00)),
                rounds=int(value.get("rounds", 3)),
                holdout_repeats=int(value.get("holdout_repeats", 2)),
                max_parallel_models=int(value.get("max_parallel_models", 8)),
                max_parallel_scenarios=int(value.get("max_parallel_scenarios", 16)),
                request_timeout_seconds=float(value.get("request_timeout_seconds", 600)),
                minimum_meaningful_quality_gain=float(value.get("minimum_meaningful_quality_gain", 0.03)),
                allow_few_shot=bool(value.get("allow_few_shot", True)),
                max_few_shot_examples=int(value.get("max_few_shot_examples", 3)),
                incumbent_model=str(value["incumbent_model"]) if value.get("incumbent_model") else None,
                allowed_accuracy_regression=float(value.get("allowed_accuracy_regression", 0.0)),
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
        if not 1 <= self.max_parallel_models <= 16:
            raise ValueError("max_parallel_models must be between 1 and 16.")
        if not 1 <= self.max_parallel_scenarios <= 64:
            raise ValueError("max_parallel_scenarios must be between 1 and 64.")
        if not 0 < self.request_timeout_seconds <= 7200:
            raise ValueError("request_timeout_seconds must be greater than zero and no more than 7200 seconds.")
        _validate_evaluator_policy(dict(self.evaluator))

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
            "objective": self.objective,
            "quality_threshold": self.quality_threshold,
            "max_optimization_cost_usd": self.max_optimization_cost_usd,
            "rounds": self.rounds,
            "holdout_repeats": self.holdout_repeats,
            "max_parallel_models": self.max_parallel_models,
            "max_parallel_scenarios": self.max_parallel_scenarios,
            "request_timeout_seconds": self.request_timeout_seconds,
            "minimum_meaningful_quality_gain": self.minimum_meaningful_quality_gain,
            "allow_few_shot": self.allow_few_shot,
            "max_few_shot_examples": self.max_few_shot_examples,
            "incumbent_model": self.incumbent_model,
            "allowed_accuracy_regression": self.allowed_accuracy_regression,
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
            "objective": self.objective,
            "quality_threshold": self.quality_threshold,
            "max_optimization_cost_usd": self.max_optimization_cost_usd,
            "rounds": self.rounds,
            "holdout_repeats": self.holdout_repeats,
            "max_parallel_models": self.max_parallel_models,
            "max_parallel_scenarios": self.max_parallel_scenarios,
            "minimum_meaningful_quality_gain": self.minimum_meaningful_quality_gain,
            "allow_few_shot": self.allow_few_shot,
            "max_few_shot_examples": self.max_few_shot_examples,
            "incumbent_model": self.incumbent_model,
            "allowed_accuracy_regression": self.allowed_accuracy_regression,
            "adaptive_search": True,
        }


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
    ) -> None:
        if transport is not None and request_timeout_seconds != 600:
            raise ValueError("request_timeout_seconds cannot be combined with a custom transport.")
        resolved_transport = transport or OpenRouterTransport(
            api_key=api_key, timeout_seconds=request_timeout_seconds
        )
        self.client = Client(api_key=api_key, transport=resolved_transport)
        self._state_path = Path(state_path)
        self._router: DurableRouter | None = None

    @property
    def router(self) -> DurableRouter:
        if self._router is None:
            self._router = DurableRouter(self.client, state_path=self._state_path)
        return self._router

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
        models: Iterable[str] | None = None,
        max_tokens: int = 600,
        budget_usd: float | None = None,
        quality_threshold: float | None = None,
        retest_after_calls: int = 500,
        min_feedback: int = 5,
        maintenance_budget_usd: float | None = None,
        auto_maintain: bool = True,
    ) -> OptimizationResult | RoutedAnswer:
        if isinstance(suite_or_prompt, Suite):
            if input is not None:
                raise ValueError("input is not used when running an explicit Suite.")
            suite_or_prompt.validate()
            if isinstance(self.client.transport, OpenRouterTransport):
                self.client.transport.set_timeout_seconds(
                    suite_or_prompt.request_timeout_seconds
                )
            return self.client.optimize(**suite_or_prompt.optimize_kwargs())
        if input is None:
            raise ValueError("input is required when executing a prompt through Evalt.")
        if price_usd is not None and budget_usd is not None and float(price_usd) != float(budget_usd):
            raise ValueError("Use price_usd; budget_usd is only a backward-compatible alias.")
        if price_usd is None and budget_usd is None and incumbent_model:
            input_text = input if isinstance(input, str) else json.dumps(input, ensure_ascii=False, sort_keys=True)
            max_cost_per_run_usd = self.client.transport.estimate_cost(
                incumbent_model,
                [{"role": "system", "content": suite_or_prompt}, {"role": "user", "content": input_text}],
                max_tokens=max_tokens,
            )
        else:
            max_cost_per_run_usd = float(price_usd if price_usd is not None else (budget_usd if budget_usd is not None else 0.02))
        if quality_threshold is not None:
            if target_accuracy != 0.95 and float(target_accuracy) != float(quality_threshold):
                raise ValueError("Use target_accuracy; quality_threshold is only a backward-compatible alias.")
            target_accuracy = float(quality_threshold)
        if maintenance_budget_usd is not None:
            if test_budget_usd != "auto" and float(test_budget_usd) != float(maintenance_budget_usd):
                raise ValueError("Use test_budget_usd; maintenance_budget_usd is only a backward-compatible alias.")
            test_budget_usd = float(maintenance_budget_usd)
        if not 0 < max_test_budget_usd <= 100:
            raise ValueError("max_test_budget_usd must be greater than 0 and no more than 100.")
        if test_budget_usd == "auto":
            resolved_test_budget_usd = min(
                float(max_test_budget_usd),
                max(0.25, max_cost_per_run_usd * max(1, int(retest_after_calls)) * 0.10),
            )
            test_budget_policy = "auto: 10% of one retest interval's production ceiling, floored at $0.25"
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
        answer = self.router.run(
            route=route,
            prompt=suite_or_prompt,
            input=input,
            max_cost_per_run_usd=max_cost_per_run_usd,
            models=requested_models,
            max_tokens=max_tokens,
            target_accuracy=target_accuracy,
            objective=objective,
            test_budget_usd=resolved_test_budget_usd,
            test_budget_policy=test_budget_policy,
            retest_after_calls=retest_after_calls,
            min_feedback=min_feedback,
            catalog_revision=role_plan.catalog_revision,
        )
        if auto_maintain and resolved_test_budget_usd > 0 and answer.maintenance_due:
            status = self.router.status(route, retest_after_calls=retest_after_calls, min_feedback=min_feedback)
            if status["feedback_count"] >= min_feedback:
                threading.Thread(
                    target=self.router.maintain,
                    kwargs={
                        "route": route,
                        "test_budget_usd": resolved_test_budget_usd,
                        "role_plan": role_plan,
                        "objective": objective,
                        "max_cost_per_run_usd": max_cost_per_run_usd,
                        "min_feedback": min_feedback,
                    },
                    name=f"evalt-maintain-{route}",
                    daemon=True,
                ).start()
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
        return self.router.status(route, retest_after_calls=retest_after_calls, min_feedback=min_feedback)

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
    "RolePlan",
    "RoutedAnswer",
    "Suite",
    "Turn",
    "check_result",
    "select_role_plan",
    "_safe_provider_error_detail",
]
