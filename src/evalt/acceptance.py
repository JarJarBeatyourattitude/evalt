"""Evidence checks for Evalt's automatic first-route contract.

This module deliberately validates observable behavior rather than trusting a README
or a successful production answer.  It is used by the live release harness and kept
small enough for downstream teams to reuse in their own smoke tests.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping


class AcceptanceFailure(AssertionError):
    """The trace does not prove a complete automatic first-route tournament."""


def redact_trace(value: Any, forbidden_values: Iterable[str] = ()) -> Any:
    """Recursively remove exact secret values before a trace is written to disk."""
    secrets = tuple(item for item in forbidden_values if item)
    if isinstance(value, Mapping):
        return {
            str(key): redact_trace(item, secrets)
            for key, item in value.items()
            if str(key).lower() not in {"api_key", "authorization", "openrouter_api_key"}
        }
    if isinstance(value, (list, tuple)):
        return [redact_trace(item, secrets) for item in value]
    if isinstance(value, str):
        redacted = value
        for secret in secrets:
            redacted = redacted.replace(secret, "[REDACTED]")
        return redacted
    return value


def validate_auto_first_route_receipt(
    receipt: Mapping[str, Any],
    *,
    minimum_configurations: int = 3,
    required_case_count: int = 25,
) -> dict[str, Any]:
    """Fail closed unless a receipt proves design, judging, search, promotion, and reuse."""
    failures: list[str] = []
    first_events = list(receipt.get("first_call_events") or [])
    second_events = list(receipt.get("second_call_events") or [])
    event_names = [str(event.get("event") or "") for event in first_events]
    summary = dict((receipt.get("first_answer") or {}).get("initial_test_summary") or {})
    status = dict(receipt.get("route_status") or {})
    target_accuracy = float(receipt.get("target_accuracy") or 0.95)
    test_budget = float(receipt.get("test_budget_usd") or 0)

    for required in (
        "initial_optimization_started",
        "suite_design_started",
        "suite_design_completed",
        "initial_optimization_completed",
        "production_call_completed",
    ):
        if required not in event_names:
            failures.append(f"missing first-call event: {required}")

    design_events = [
        event for event in first_events if event.get("event") == "suite_design_completed"
    ]
    if not design_events:
        failures.append("no completed suite-design event")
    else:
        design = design_events[-1]
        if int(design.get("case_count") or 0) != required_case_count:
            failures.append(f"suite did not contain exactly {required_case_count} cases")
        if not design.get("judge_calibrated"):
            failures.append("judge was not reported as calibrated")
        if int(design.get("judge_calibration_checks") or 0) < 3:
            failures.append("fewer than three separate judge-calibration checks")

    completed = [
        event for event in first_events if event.get("event") == "model_completed"
    ]
    configurations = {str(event.get("model") or "") for event in completed}
    base_models = {item.split("#reasoning=", 1)[0] for item in configurations if item}
    reasoning_levels = {
        item.rsplit("#reasoning=", 1)[1]
        for item in configurations
        if "#reasoning=" in item
    }
    if len(completed) < minimum_configurations:
        failures.append(
            f"only {len(completed)} configurations settled; {minimum_configurations} required"
        )
    if len(base_models) < 2 and len(reasoning_levels) < 2:
        failures.append("search did not span at least two models or reasoning levels")

    prompt_events = [
        event for event in first_events
        if event.get("event") == "prompt_candidate_completed"
    ]
    if not any(event.get("kind") == "starting_prompt" for event in prompt_events):
        failures.append("original prompt was not measured")
    if not any(event.get("kind") == "rewrite" for event in prompt_events):
        failures.append("no prompt rewrite was measured")

    if int(summary.get("final_test_scenarios") or 0) < 5:
        failures.append("winner did not face at least five untouched final-test cases")
    if float(summary.get("holdout_pass_rate") or 0) < target_accuracy:
        failures.append("winner did not clear the requested final-test accuracy")
    if int(summary.get("tested_configurations") or 0) < minimum_configurations:
        failures.append("durable route summary records too few tested configurations")
    if int(summary.get("prompt_rewrites_tested") or 0) < 1:
        failures.append("durable route summary records no prompt rewrite")
    if summary.get("evidence_provenance") != "AI_GENERATED_AI_JUDGED":
        failures.append("route does not preserve AI-generated/AI-judged provenance")
    if not summary.get("judge_calibrated"):
        failures.append("durable route summary does not preserve judge calibration")
    if float(summary.get("workflow_spend_usd") or 0) > test_budget + 1e-9:
        failures.append("automatic test exceeded its declared workflow budget")

    first_answer = dict(receipt.get("first_answer") or {})
    second_answer = dict(receipt.get("second_answer") or {})
    if first_answer.get("route_phase") != "ai_tested":
        failures.append("first answer was not served through an AI-tested route")
    if second_answer.get("route_phase") != "ai_tested":
        failures.append("second answer did not reuse the AI-tested route")
    if status.get("route_phase") != "ai_tested":
        failures.append("durable route status is not ai_tested")
    if any(
        event.get("event") in {"initial_optimization_started", "suite_design_started"}
        for event in second_events
    ):
        failures.append("second call incorrectly designed or optimized the route again")
    if not any(event.get("event") == "production_call_completed" for event in second_events):
        failures.append("second production call did not complete")

    if failures:
        raise AcceptanceFailure("; ".join(failures))
    return {
        "status": "PASS",
        "cases": required_case_count,
        "judge_calibration_checks": int(design_events[-1]["judge_calibration_checks"]),
        "settled_configurations": len(completed),
        "distinct_models": len(base_models),
        "reasoning_levels": sorted(reasoning_levels),
        "prompt_candidates_measured": len(prompt_events),
        "prompt_rewrites_measured": sum(
            event.get("kind") == "rewrite" for event in prompt_events
        ),
        "winner": summary.get("winner_model"),
        "final_test_pass_rate": float(summary.get("holdout_pass_rate") or 0),
        "workflow_spend_usd": float(summary.get("workflow_spend_usd") or 0),
        "route_reused": True,
    }


def assert_no_secret(receipt: Mapping[str, Any], secrets: Iterable[str]) -> None:
    """Ensure no supplied secret remains in the serialized public receipt."""
    serialized = json.dumps(receipt, sort_keys=True, ensure_ascii=False)
    leaked = [secret for secret in secrets if secret and secret in serialized]
    if leaked:
        raise AcceptanceFailure("a credential remained in the redacted receipt")
