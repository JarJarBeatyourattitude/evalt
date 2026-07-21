"""Typed, primary Evalt SDK surface.

The optimization engine remains import-compatible with the two earlier package names;
new code should use :class:`Suite` and :class:`Evalt` from this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
import threading
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
        self._maintenance_guard = threading.Lock()
        self._maintenance_routes: set[str] = set()
        self._maintenance_threads: list[threading.Thread] = []

    def _emit_progress(self, event: dict[str, Any]) -> None:
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
            message = (
                f"Evalt · {route} · TEST DESIGN STARTED · "
                f"{int(event.get('case_count') or 0)} cases · "
                f"one workflow cap ${float(event.get('workflow_budget_usd') or 0):.2f}"
            )
        elif kind == "suite_design_completed":
            message = (
                f"Evalt · {route} · TEST DRAFT READY · "
                f"{int(event.get('case_count') or 0)} AI-authored cases · "
                f"spent ${float(event.get('designer_spend_usd') or 0):.6f} · "
                f"${float(event.get('remaining_budget_usd') or 0):.6f} remains for the tournament"
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
            message = (
                f"Evalt · {event.get('model', 'route')} · "
                f"{float(event.get('final_test_pass_rate') or 0):.0%} final test · "
                f"{int(event.get('prompt_candidates_tested') or 1)} prompt package(s) · "
                f"${float(event.get('optimization_spend_usd') or 0):.6f} spent"
            )
        elif kind == "broad_screen_started":
            message = (
                f"Evalt · BROAD SCREEN · {int(event.get('configurations') or 0)} "
                f"model configuration(s) · up to {int(event.get('parallel_models') or 1)} in parallel"
            )
        elif kind == "model_screen_completed":
            message = (
                f"Evalt · SCREENED · {event.get('model', 'model')} · "
                f"{float(event.get('validation_pass_rate') or 0):.0%} validation · "
                f"p90 {int(event.get('target_latency_p90_ms') or 0)} ms · "
                f"${float(event.get('screening_spend_usd') or 0):.6f} spent"
            )
        elif kind == "broad_screen_completed":
            message = (
                f"Evalt · BROAD SCREEN COMPLETE · "
                f"{int(event.get('completed_configurations') or 0)}/"
                f"{int(event.get('configurations') or 0)} configuration(s) settled · "
                f"{float(event.get('elapsed_seconds') or 0):.1f}s elapsed"
            )
        elif kind in {"model_unavailable", "model_incomplete"}:
            label = "UNAVAILABLE" if kind == "model_unavailable" else "INCOMPLETE"
            message = (
                f"Evalt · {label} · {event.get('model', 'model')} · "
                f"{event.get('reason', 'provider did not settle')}"
            )
        elif kind == "prompt_candidate_completed":
            candidate = int(event.get("candidate") or 0)
            label = "original prompt" if candidate == 0 else f"prompt rewrite {candidate}"
            decision = "selected so far" if event.get("selected") else "not selected"
            message = (
                f"Evalt · {event.get('model', 'model')} · {label} · "
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
        if models is None and hasattr(self.client.transport, "model_catalog"):
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
            self._emit_progress({
                "event": "suite_design_started",
                "route": route,
                "case_count": int(case_count),
                "workflow_budget_usd": float(workflow_budget_usd),
                "designer_model": selected_designer,
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
                            "type": {"type": "string", "enum": ["semantic", "exact_text", "exact_json"]},
                            "reason": {"type": "string"},
                            "required_keys": {"type": "array", "items": {"type": "string"}},
                            "allow_additional_properties": {"type": "boolean"},
                            "normalize_rational_strings": {"type": "boolean"},
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
                            "required": ["id", "difficulty", "critical", "turns", "rationale"],
                            "properties": {
                                "id": {"type": "string"},
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
            max_tokens = min(32768, max(4000, generated_count * 500))
            completion = self.client._call(
                budget,
                selected_designer,
                [
                    {
                        "role": "system",
                        "content": (
                            "Design a balanced evaluation suite for a recurring production AI task. "
                            "Create genuinely distinct routine, complex, adversarial, boundary, format, "
                            "and multi-turn cases where relevant; do not merely paraphrase seeds. Expected "
                            "outputs describe the desired behavior, not what the current prompt happens to "
                            "produce. Make each case concrete enough for a human to approve or edit. All "
                            "cases are drafted before any train/validation/final-test split, so never label "
                            "or target a split. Recommend exact_text or exact_json only when equivalent "
                            "answers truly must match that deterministic contract; otherwise use semantic. "
                            "Also create judge-calibration checks outside the scenario suite: at least two "
                            "clear passes and one clear failure, each labeled with should_pass. These are "
                            "unapproved AI drafts. Representative inputs have no approved outputs; use them "
                            "only to understand realistic shape, length, and domain, and do not copy them "
                            "into the suite. Return only the required JSON."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_tokens=max_tokens,
                response_schema=schema,
            )
            try:
                text = str(completion.content).strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                parsed = json.loads(text)
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
                    design_notes.insert(0, f"Evaluator: {str(raw_evaluator.get('reason') or '').strip()}")
                calibration_rows = list(parsed.get("judge_calibration") or [])
                if len(calibration_rows) < 3:
                    raise ValueError("insufficient judge calibration")
                calibrated_evaluator: str | None = None
                for candidate in tuple(dict.fromkeys((selected_evaluator, selected_designer))):
                    matched = True
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
                            break
                    if matched:
                        calibrated_evaluator = candidate
                        break
                if calibrated_evaluator is None:
                    raise ProviderError(
                        "No candidate judge passed the AI-designed calibration checks; no tournament ran."
                    )
                selected_evaluator = calibrated_evaluator
                judge_calibration_checks = len(calibration_rows)
                design_notes.insert(
                    1,
                    f"Judge calibration: {selected_evaluator} matched {len(calibration_rows)} labeled checks.",
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ProviderError("The test designer returned an invalid structured suite; no draft was approved.") from error

        all_examples = (*seeds, *drafted_examples)
        ids = [item.id for item in all_examples]
        if len(set(ids)) != len(ids):
            raise ProviderError("The test designer returned duplicate scenario IDs; no draft was approved.")
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
        max_tokens: int = 600,
        budget_usd: float | None = None,
        quality_threshold: float | None = None,
        retest_after_calls: int = 500,
        min_feedback: int = 5,
        maintenance_budget_usd: float | None = None,
        auto_maintain: bool = True,
        task: str | None = None,
        first_run: str = "optimize",
        case_count: int = 25,
        designer_model: str | None = None,
        evaluator_model: str | None = None,
        test_request_timeout_seconds: float = 120,
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
            if self._show_progress or self._progress_callback is not None:
                optimize_kwargs["progress_callback"] = self._emit_progress
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
        if first_run not in {"optimize", "bootstrap"}:
            raise ValueError("first_run must be 'optimize' or 'bootstrap'.")
        if not 0 < float(test_request_timeout_seconds) <= 7200:
            raise ValueError(
                "test_request_timeout_seconds must be greater than zero and no more than 7200 seconds."
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
                draft = self.design_suite(
                    task=(task or suite_or_prompt),
                    prompt=suite_or_prompt,
                    route=route,
                    case_count=int(case_count),
                    workflow_budget_usd=resolved_test_budget_usd,
                    quality_threshold=target_accuracy,
                    models=requested_models,
                    designer_model=designer_model or role_plan.test_designer_model,
                    evaluator_model=evaluator_model or role_plan.judge_model,
                    representative_inputs=(input,),
                    objective=objective,
                    optimize_prompt=bool(optimize_prompt),
                    max_p90_latency_seconds=max_p90_latency_seconds,
                    latency_value_usd_per_second=latency_value_usd_per_second,
                    request_timeout_seconds=float(test_request_timeout_seconds),
                )
                initial_result = self.run(draft.autopilot_suite())
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
                    )
                except ValueError as error:
                    raise ProviderError(str(error)) from error
                self._emit_progress({
                    "event": "initial_optimization_completed",
                    "route": route,
                    **summary,
                })
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
            raise
        status = self.router.status(
            route, retest_after_calls=retest_after_calls, min_feedback=min_feedback
        )
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
        })

        def on_feedback(receipt: dict[str, Any]) -> None:
            self._emit_progress(receipt)
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
