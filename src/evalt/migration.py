"""Conservative, offline migration helpers for historical eval exports."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


APPROVED_OUTPUT_KEYS = (
    "approved_output", "expected", "ideal", "reference_answer", "reference",
    "target", "label", "ground_truth",
)
INPUT_KEYS = ("input", "prompt_input", "question", "query")
NESTED_KEYS = ("sample", "item", "data", "record", "example")


@dataclass(frozen=True)
class MigrationResult:
    suite: dict[str, Any] | None
    report: dict[str, Any]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _mappings(row: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield only known containers, avoiding an ambiguous recursive key search."""
    yield row
    for key in NESTED_KEYS:
        nested = row.get(key)
        if isinstance(nested, Mapping):
            yield nested


def _extract_input(row: Mapping[str, Any]) -> tuple[str, str | None]:
    for container in _mappings(row):
        for key in INPUT_KEYS:
            text = _as_text(container.get(key))
            if text:
                return text, key
        messages = container.get("messages")
        if isinstance(messages, list):
            user_messages = [
                _as_text(message.get("content"))
                for message in messages
                if isinstance(message, Mapping) and message.get("role") == "user"
            ]
            user_messages = [message for message in user_messages if message]
            if user_messages:
                return user_messages[-1], "messages[last_user]"
    return "", None


def _extract_approved_output(row: Mapping[str, Any]) -> tuple[str, str | None]:
    for container in _mappings(row):
        for key in APPROVED_OUTPUT_KEYS:
            text = _as_text(container.get(key))
            if text:
                return text, key
    return "", None


def migrate_openai_results(
    input_path: str | Path,
    *,
    prompt: str,
    name: str,
    models: Iterable[str],
    quality_threshold: float = 0.95,
    max_optimization_cost_usd: float = 2.0,
) -> MigrationResult:
    """Recover reviewable cases without treating model responses as ground truth."""
    source = Path(input_path)
    if not prompt.strip():
        raise ValueError("The original task prompt is required; historical result JSONL cannot reconstruct it.")
    chosen_models = tuple(dict.fromkeys(model.strip() for model in models if model.strip()))
    if not chosen_models:
        raise ValueError("Provide at least one candidate model.")

    examples: list[dict[str, str]] = []
    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    total_rows = 0

    with source.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            total_rows += 1
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                malformed.append({"line": line_number, "reason": f"invalid JSON: {error.msg}"})
                continue
            if not isinstance(row, Mapping):
                malformed.append({"line": line_number, "reason": "row is not a JSON object"})
                continue

            input_text, input_field = _extract_input(row)
            approved_output, output_field = _extract_approved_output(row)
            missing = []
            if not input_text:
                missing.append("input")
            if not approved_output:
                missing.append("approved output/reference")
            if missing:
                skipped.append({
                    "line": line_number,
                    "reason": "missing " + " and ".join(missing),
                    "candidate_output_ignored": any(
                        key in row for key in ("output", "response", "completion")
                    ),
                })
                continue

            raw_id = _as_text(row.get("id") or row.get("sample_id") or row.get("eval_sample_id"))
            example_id = raw_id or f"openai-row-{line_number}"
            if example_id in seen_ids:
                example_id = f"{example_id}-line-{line_number}"
            seen_ids.add(example_id)
            examples.append({"id": example_id, "input": input_text, "approved_output": approved_output})
            imported.append({
                "line": line_number, "example_id": example_id,
                "input_field": input_field, "approved_output_field": output_field,
            })

    imported_rows = len(examples)
    report: dict[str, Any] = {
        "schema": "evalt-openai-results-migration-report-v1",
        "source": str(source),
        "total_nonempty_rows": total_rows,
        "imported_rows": imported_rows,
        "skipped_rows": len(skipped),
        "malformed_rows": len(malformed),
        "runnable_suite_created": imported_rows >= 3,
        "important_limit": (
            "Historical result JSONL is not a runnable eval definition. Evalt did not infer the "
            "original prompt, grader, model settings, or approved answers from candidate outputs."
        ),
        "imported": imported,
        "skipped": skipped,
        "malformed": malformed,
        "next_steps": (
            ["Review every imported approved output before optimization."]
            if imported_rows >= 3
            else ["Add explicit approved outputs/references until at least three reviewable cases exist."]
        ),
    }
    if imported_rows < 3:
        return MigrationResult(None, report)

    suite = {
        "schema": "evalt-suite-v1",
        "name": name,
        "prompt": prompt.strip(),
        "examples": examples,
        "models": list(chosen_models),
        "optimizer_model": "openai/gpt-5.6-luna",
        "evaluator_model": "openai/gpt-5.6-luna",
        "evaluator": {"type": "semantic"},
        "objective": "lowest_cost_at_accuracy",
        "quality_threshold": quality_threshold,
        "max_optimization_cost_usd": max_optimization_cost_usd,
        "rounds": 3,
        "holdout_repeats": 2,
        "max_parallel_models": 8,
        "max_parallel_scenarios": 16,
        "minimum_meaningful_quality_gain": 0.03,
        "allow_few_shot": True,
        "max_few_shot_examples": min(3, len(examples)),
    }
    return MigrationResult(suite, report)
