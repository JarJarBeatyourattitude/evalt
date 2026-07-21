"""Command-line interface for Evalt projects, runs, and CI gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time

from .core import BudgetExceeded, Evalt, ProviderError, Suite, check_result
from .migration import migrate_openai_results
from .reporting import compare_results, render_comparison_html, write_reports


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
    "optimize_prompt": True,
    "minimum_meaningful_quality_gain": 0.03,
    "allow_few_shot": True,
    "max_few_shot_examples": 3,
}


class _CliProgress:
    """Readable TTY progress with JSONL preserved for pipes and automation."""

    def __init__(self, total: int) -> None:
        self.total = max(1, int(total))
        self.started_at = time.monotonic()
        self.active: set[str] = set()
        self.finished: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self._thread: threading.Thread | None = None
        if self._tty:
            self._thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._thread.start()

    def _status(self) -> str:
        elapsed = int(time.monotonic() - self.started_at)
        return (
            f"Evalt {elapsed // 60:02d}:{elapsed % 60:02d}  "
            f"{len(self.active)} active  {len(self.finished)}/{self.total} routes settled"
        )

    def _heartbeat(self) -> None:
        while not self._stop.wait(1):
            with self._lock:
                print(f"\r{self._status():<88}", end="", file=sys.stderr, flush=True)

    def __call__(self, event: dict) -> None:
        if not self._tty:
            print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)
            return
        model = str(event.get("model") or "route")
        kind = str(event.get("event") or "progress")
        with self._lock:
            if kind == "model_started":
                self.active.add(model)
                message = f"START  {model}"
            elif kind == "model_screen_started":
                self.active.add(model)
                message = f"SCREEN {model}"
            elif kind == "model_screen_completed":
                self.active.discard(model)
                rate = round(float(event.get("validation_pass_rate", 0)) * 100)
                p90 = int(event.get("target_latency_p90_ms") or 0)
                message = f"SCREEN {model}  {rate}% validation  p90 {p90 / 1000:.2f}s"
            elif kind == "prompt_propagation_started":
                self.active.add(model)
                count = int(event.get("candidate_prompt_packages") or 0)
                message = f"RETEST {model}  {count} learned prompt package(s)"
            elif kind == "prompt_propagation_completed":
                self.active.discard(model)
                rate = round(float(event.get("validation_pass_rate", 0)) * 100)
                source = str(event.get("source_model") or "another route")
                message = f"RETEST {model}  {rate}% validation  prompt from {source}"
            elif kind == "model_pruned":
                self.active.discard(model)
                self.finished.add(model)
                message = f"PRUNE  {model}  {event.get('reason', kind)}"
            elif kind == "model_completed":
                self.active.discard(model)
                self.finished.add(model)
                rate = round(float(event.get("final_test_pass_rate", 0)) * 100)
                p90 = int(event.get("target_latency_p90_ms") or 0)
                spend = float(event.get("optimization_spend_usd") or 0)
                message = f"DONE   {model}  {rate}% pass  p90 {p90 / 1000:.2f}s  ${spend:.4f}"
            elif kind in {"model_unavailable", "model_incomplete", "configuration_omitted"}:
                self.active.discard(model)
                self.finished.add(model)
                message = f"SKIP   {model}  {event.get('reason', kind)}"
            else:
                message = f"INFO   {model}  {kind.replace('_', ' ')}"
            print(f"\r{'':88}\r{message}", file=sys.stderr, flush=True)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self._tty:
            with self._lock:
                print(f"\r{'':88}\r{self._status()}", file=sys.stderr, flush=True)


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
    root.add_argument("--version", action="version", version="evalt 0.8.22")
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
    run.add_argument("--price", "--budget", dest="price", type=float, help="Optional hard maximum provider price for one production response; omitted uses a request-sized automatic safety ceiling.")
    run.add_argument("--test-budget", "--maintenance-budget", dest="test_budget", default="auto", help="Automatic or numeric hard cap for a due retest.")
    run.add_argument("--max-test-budget", type=float, default=1.0, help="Hard ceiling when --test-budget=auto.")
    run.add_argument("--target-accuracy", type=float, default=0.95)
    run.add_argument("--objective", choices=("match_baseline_at_lowest_cost", "best_within_price", "lowest_cost_at_accuracy"), default="lowest_cost_at_accuracy")
    run.add_argument("--state", default=".evalt/evalt.db")
    run.add_argument("--model", action="append", dest="models")
    run.add_argument("--fixed-prompt", action="store_true", help="Compare routes without rewriting the supplied prompt or adding few-shot examples.")
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
    optimize.add_argument("--fixed-prompt", action="store_true", help="Override the suite and compare routes without modifying its prompt.")
    optimize.add_argument("--html-report", help="Also write a self-contained offline HTML report.")
    optimize.add_argument("--junit-report", help="Also write case-level JUnit XML for CI.")

    report = commands.add_parser("report", help="Render saved JSON as HTML and/or JUnit; no provider call.")
    report.add_argument("result", help="Saved Evalt result JSON.")
    report.add_argument("--html", help="Self-contained HTML output path.")
    report.add_argument("--junit", help="JUnit XML output path.")
    report.add_argument("--title", default="Evalt evaluation report")

    compare = commands.add_parser(
        "compare", help="Diff two saved runs by final-test case; no provider call."
    )
    compare.add_argument("baseline", help="Earlier saved Evalt result JSON.")
    compare.add_argument("candidate", help="Candidate saved Evalt result JSON.")
    compare.add_argument("--output", help="Also write the structured comparison JSON.")
    compare.add_argument("--html", help="Also write a self-contained HTML comparison.")
    compare.add_argument("--title", default="Evalt comparison")

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
        "quality_gate_status": result.quality_gate_status,
        "continuation_recommendation": result.continuation_recommendation,
        "quality_frontier": result.quality_frontier,
        "diminishing_returns": result.diminishing_returns,
        "omitted_configurations": result.omitted_configurations,
        "unavailable_models": result.unavailable_models,
        "screening_results": result.screening_results,
        "result": path,
    }


def main(argv: list[str] | None = None) -> int:
    # Windows terminals may default to a legacy code page. Progress and summaries
    # are UTF-8 JSON contracts and must not fail after a paid result was saved.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
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
                optimize_prompt=not args.fixed_prompt,
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
        if args.command == "report":
            with Path(args.result).open(encoding="utf-8") as handle:
                result_payload = json.load(handle)
            written = write_reports(
                result_payload,
                html_path=args.html,
                junit_path=args.junit,
                title=args.title,
            )
            print(json.dumps({"provider_call_started": False, "reports": written}, indent=2))
            return 0
        if args.command == "compare":
            with Path(args.baseline).open(encoding="utf-8") as handle:
                baseline_payload = json.load(handle)
            with Path(args.candidate).open(encoding="utf-8") as handle:
                candidate_payload = json.load(handle)
            comparison = compare_results(baseline_payload, candidate_payload)
            written: dict[str, str] = {}
            if args.output:
                target = Path(args.output)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                written["json"] = str(target)
            if args.html:
                target = Path(args.html)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    render_comparison_html(comparison, title=args.title),
                    encoding="utf-8",
                )
                written["html"] = str(target)
            payload = {
                "provider_call_started": False,
                "comparison": comparison,
                "reports": written,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
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
            if args.fixed_prompt:
                optimize_kwargs["optimize_prompt"] = False
            progress = _CliProgress(len(optimize_kwargs.get("models") or []))
            try:
                result = client.client.optimize(
                    **optimize_kwargs,
                    progress_callback=progress,
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
            finally:
                progress.close()
            result.save(args.output)
            reports = {}
            if args.html_report or args.junit_report:
                reports = write_reports(
                    result.to_dict(),
                    html_path=args.html_report,
                    junit_path=args.junit_report,
                    title=f"Evalt · {suite.name}",
                )
            summary = _summary(result, args.output)
            if reports:
                summary["reports"] = reports
            print(json.dumps(summary, indent=2, ensure_ascii=False))
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
