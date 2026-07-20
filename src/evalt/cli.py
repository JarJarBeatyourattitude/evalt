"""Command-line interface for Evalt projects, runs, and CI gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .core import BudgetExceeded, Evalt, ProviderError, Suite, check_result
from .migration import migrate_openai_results


STARTER_SUITE = {
    "schema": "evalt-suite-v1",
    "name": "support-routing",
    "prompt": "Classify the support message. Return one route label.",
    "examples": [
        {"id": "billing-1", "input": "I was charged twice", "approved_output": "billing"},
        {"id": "account-1", "input": "My reset link expired", "approved_output": "account"},
        {"id": "technical-1", "input": "The app freezes on launch", "approved_output": "technical"},
        {"id": "billing-2", "input": "Please send last month's invoice", "approved_output": "billing"},
        {"id": "account-2", "input": "I cannot sign in", "approved_output": "account"},
    ],
    "models": ["qwen/qwen3.5-9b", "google/gemini-3-flash-preview"],
    "optimizer_model": "openai/gpt-5.6-luna",
    "evaluator_model": "openai/gpt-5.6-luna",
    "objective": "lowest_cost_at_accuracy",
    "quality_threshold": 0.95,
    "max_optimization_cost_usd": 2.00,
    "rounds": 3,
    "minimum_meaningful_quality_gain": 0.03,
    "allow_few_shot": True,
    "max_few_shot_examples": 3,
}


class _OfflineTransport:
    def estimate_cost(self, *_args, **_kwargs):
        raise RuntimeError("Offline status does not make provider calls.")

    def complete(self, *_args, **_kwargs):
        raise RuntimeError("Offline status does not make provider calls.")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="evalt",
        description="Run prompts through a durable, tested, budget-bounded model route.",
    )
    root.add_argument("--version", action="version", version="evalt 0.8.12")
    commands = root.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="Write a reviewable starter suite; no provider call.")
    init.add_argument("path", nargs="?", default="evalt.json")
    init.add_argument("--force", action="store_true", help="Replace an existing file.")

    validate = commands.add_parser("validate", help="Validate a suite offline; no API key or spend.")
    validate.add_argument("suite")

    migrate = commands.add_parser(
        "import-openai-results",
        help="Recover reviewable cases from OpenAI Evals result JSONL; offline and conservative.",
    )
    migrate.add_argument("results", help="OpenAI Evals result JSONL export.")
    prompt_source = migrate.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument("--prompt", help="The original task prompt.")
    prompt_source.add_argument("--prompt-file", help="UTF-8 file containing the original task prompt.")
    migrate.add_argument("--output", default="evalt.json", help="Output Evalt suite path.")
    migrate.add_argument("--report", help="Migration report path; defaults beside --output.")
    migrate.add_argument("--name", default="openai-evals-migration")
    migrate.add_argument("--model", action="append", dest="models")
    migrate.add_argument("--quality-threshold", type=float, default=0.95)
    migrate.add_argument("--max-optimization-cost", type=float, default=2.0)

    draft = commands.add_parser("draft", help="Generate one bounded answer for approval or correction.")
    draft.add_argument("--task", required=True)
    draft.add_argument("--input", required=True)
    draft.add_argument("--model", default="openai/gpt-5-mini")
    draft.add_argument("--max-cost", type=float, default=0.10)

    run = commands.add_parser("run", help="Execute one prompt through a durable route.")
    run.add_argument("--route", required=True, help="Stable route name used to remember decisions.")
    run.add_argument("--prompt", required=True)
    run.add_argument("--input", required=True)
    run.add_argument("--price", "--budget", dest="price", type=float, required=True, help="Maximum provider price for one production response.")
    run.add_argument("--test-budget", "--maintenance-budget", dest="test_budget", default="auto", help="Automatic or numeric hard cap for a due retest.")
    run.add_argument("--max-test-budget", type=float, default=1.0, help="Hard ceiling when --test-budget=auto.")
    run.add_argument("--target-accuracy", type=float, default=0.95)
    run.add_argument("--objective", choices=("match_baseline_at_lowest_cost", "best_within_price", "lowest_cost_at_accuracy"), default="lowest_cost_at_accuracy")
    run.add_argument("--state", default=".evalt/evalt.db")
    run.add_argument("--model", action="append", dest="models")
    run.add_argument("--approved-output", help="Immediately record an accepted/corrected answer for future tests.")

    status = commands.add_parser("status", help="Show the durable decision trail for one route; no provider call.")
    status.add_argument("--route", required=True)
    status.add_argument("--state", default=".evalt/evalt.db")

    optimize = commands.add_parser("optimize", help="Run the suite under its hard provider-spend cap.")
    optimize.add_argument("suite")
    optimize.add_argument("--output", default="evalt-result.json")
    optimize.add_argument("--model", action="append", dest="models", help="Override the suite candidate list; repeat for each model/reasoning configuration.")
    optimize.add_argument("--max-parallel-models", type=int, help="Override parallel model lanes for this run.")
    optimize.add_argument("--max-parallel-scenarios", type=int, help="Override parallel scenario lanes per model for this run.")
    optimize.add_argument("--request-timeout", type=float, help="Override the suite's per-response wall-clock deadline (default 600 seconds; maximum 7200).")

    check = commands.add_parser("check", help="Gate an exported result for CI; no provider call.")
    check.add_argument("result")
    check.add_argument("--min-pass-rate", type=float, default=0.95)
    check.add_argument("--max-cost-per-success", type=float)
    check.add_argument("--require-complete-coverage", action="store_true")
    check.add_argument("--json", action="store_true", help="Print the gate report as JSON.")
    return root


def _write_starter(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to replace it.")
    path.write_text(json.dumps(STARTER_SUITE, indent=2) + "\n", encoding="utf-8")


def _summary(result, path: str) -> dict[str, object]:
    return {
        "winner_model": result.winner.model,
        "winner_prompt": result.winner.selected_prompt,
        "holdout_pass_rate": result.winner.holdout_pass_rate,
        "final_test_scenarios": result.winner.holdout_unique_scenarios,
        "final_test_executions": result.winner.holdout_executions,
        "final_test_execution_pass_rate": result.winner.holdout_execution_pass_rate,
        "estimated_cost_per_successful_call_usd": result.winner.estimated_cost_per_successful_call_usd,
        "optimization_spend_usd": result.total_provider_spend_usd,
        "elapsed_seconds": result.elapsed_seconds,
        "exploratory": result.exploratory,
        "winner_scope": result.winner_scope,
        "quality_frontier": result.quality_frontier,
        "diminishing_returns": result.diminishing_returns,
        "unavailable_models": result.unavailable_models,
        "result": path,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            path = Path(args.path)
            _write_starter(path, force=args.force)
            print(
                f"Created {path}. The five examples are a starter, not production evidence. "
                f"Add at least 25 approved scenarios, then run: evalt validate {path}"
            )
            return 0
        if args.command == "validate":
            suite = Suite.load(args.suite)
            print(json.dumps({
                "valid": True,
                "name": suite.name,
                "examples": len(suite.examples),
                "distinct_final_test_scenarios": max(1, len(suite.examples) // 5),
                "exploratory": max(1, len(suite.examples) // 5) < 5,
                "models": list(suite.models),
                "quality_threshold": suite.quality_threshold,
                "hard_provider_spend_cap_usd": suite.max_optimization_cost_usd,
                "per_provider_request_timeout_seconds": suite.request_timeout_seconds,
                "provider_call_started": False,
            }, indent=2))
            return 0
        if args.command == "import-openai-results":
            prompt = args.prompt
            if args.prompt_file:
                prompt = Path(args.prompt_file).read_text(encoding="utf-8")
            output_path = Path(args.output)
            report_path = Path(args.report) if args.report else output_path.with_suffix(
                output_path.suffix + ".migration-report.json"
            )
            migrated = migrate_openai_results(
                args.results,
                prompt=prompt or "",
                name=args.name,
                models=args.models or STARTER_SUITE["models"],
                quality_threshold=args.quality_threshold,
                max_optimization_cost_usd=args.max_optimization_cost,
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(migrated.report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            if migrated.suite is None:
                print(
                    f"No runnable suite written: only {migrated.report['imported_rows']} rows had both "
                    f"an input and explicit approved output. Review {report_path}.",
                    file=sys.stderr,
                )
                return 2
            Suite.from_dict(migrated.suite)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(migrated.suite, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({
                "suite": str(output_path),
                "report": str(report_path),
                "imported_rows": migrated.report["imported_rows"],
                "skipped_rows": migrated.report["skipped_rows"],
                "malformed_rows": migrated.report["malformed_rows"],
                "provider_call_started": False,
                "next": f"Review approved outputs, then run: evalt validate {output_path}",
            }, indent=2))
            return 0
        if args.command == "draft":
            draft = Evalt().draft(
                task=args.task, input=args.input, model=args.model, max_cost_usd=args.max_cost
            )
            print(json.dumps({
                "task": draft.task,
                "input": draft.input,
                "answer": draft.answer,
                "model": draft.model,
                "provider_cost_usd": draft.provider_cost_usd,
                "next": "Approve this answer or replace it with the answer you wanted.",
            }, indent=2, ensure_ascii=False))
            return 0
        if args.command == "run":
            answer = Evalt(state_path=args.state).run(
                args.prompt,
                args.input,
                route=args.route,
                price_usd=args.price,
                test_budget_usd=args.test_budget if args.test_budget == "auto" else float(args.test_budget),
                max_test_budget_usd=args.max_test_budget,
                target_accuracy=args.target_accuracy,
                objective=args.objective,
                models=args.models,
            )
            if args.approved_output is not None:
                if args.approved_output == answer.content:
                    answer.accept()
                else:
                    answer.correct(args.approved_output)
            print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
            return 0
        if args.command == "status":
            print(json.dumps(Evalt(transport=_OfflineTransport(), state_path=args.state).route_status(args.route), indent=2, ensure_ascii=False))
            return 0
        if args.command == "optimize":
            suite = Suite.load(args.suite)
            request_timeout_seconds = (
                args.request_timeout
                if args.request_timeout is not None
                else suite.request_timeout_seconds
            )
            client = Evalt(request_timeout_seconds=request_timeout_seconds)
            optimize_kwargs = suite.optimize_kwargs()
            if args.models:
                optimize_kwargs["models"] = args.models
            if args.max_parallel_models is not None:
                optimize_kwargs["max_parallel_models"] = args.max_parallel_models
            if args.max_parallel_scenarios is not None:
                optimize_kwargs["max_parallel_scenarios"] = args.max_parallel_scenarios
            try:
                result = client.client.optimize(
                    **optimize_kwargs,
                    progress_callback=lambda event: print(
                        json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True
                    ),
                )
            except (BudgetExceeded, ProviderError) as error:
                failure = {
                    "schema": "evalt-run-failure-v1",
                    "status": "INCOMPLETE",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "suite": str(args.suite),
                    "requested_models": list(optimize_kwargs.get("models") or []),
                    "output": str(args.output),
                    "provider_spend_usd": None,
                    "provider_spend_note": (
                        "A failed provider response may still be billable. Audit the provider account; "
                        "Evalt does not invent a zero-cost receipt when no lane completes."
                    ),
                }
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True)
                return 2
            result.save(args.output)
            print(json.dumps(_summary(result, args.output), indent=2, ensure_ascii=False))
            return 0
        with Path(args.result).open(encoding="utf-8") as handle:
            report = check_result(
                json.load(handle),
                min_pass_rate=args.min_pass_rate,
                max_cost_per_success_usd=args.max_cost_per_success,
                require_complete_coverage=args.require_complete_coverage,
            )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        elif report.passed:
            print(f"PASS: holdout pass rate {report.holdout_pass_rate:.1%}")
        else:
            print("FAIL: " + "; ".join(report.failures), file=sys.stderr)
        return 0 if report.passed else 1
    except (BudgetExceeded, ProviderError, ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        print(f"evalt: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
