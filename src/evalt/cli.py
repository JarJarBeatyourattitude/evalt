"""Command-line interface for Evalt projects, runs, and CI gates."""

from __future__ import annotations

import argparse
import certifi
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import webbrowser

from .core import BudgetExceeded, Evalt, ProviderError, Suite, check_result
from .migration import migrate_openai_results
from .reporting import (
    check_regression,
    compare_results,
    render_comparison_html,
    write_reports,
)
from .dashboard import (
    DEFAULT_DASHBOARD_API_URL,
    DEFAULT_DASHBOARD_APP_URL,
    dashboard_config_path,
    generate_workspace_token,
    inspect_workspace,
    load_dashboard_config,
    remove_dashboard_config,
    save_dashboard_config,
    workspace_fingerprint,
)


HOSTED_SDK_VERSION = "0.10.28"
HOSTED_WHEEL_URL = (
    "https://evalt.onrender.com/python-sdk/dist/"
    f"evalt-{HOSTED_SDK_VERSION}-py3-none-any.whl"
)
HOSTED_RELEASE_REQUIREMENTS_URL = "https://evalt.onrender.com/python-sdk/latest.txt"


def _sdk_version() -> str:
    from . import __version__
    return __version__


def _runtime_identity() -> dict[str, object]:
    executable = str(Path(sys.executable).resolve())
    installed_version = _sdk_version()
    hosted_current = installed_version == HOSTED_SDK_VERSION
    return {
        "sdk_version": installed_version,
        "installed_sdk_version": installed_version,
        "hosted_sdk_version": HOSTED_SDK_VERSION,
        "hosted_wheel_url": HOSTED_WHEEL_URL,
        "installed_matches_hosted": hosted_current,
        "python_executable": executable,
        "evalt_package": str(Path(__file__).resolve().parent),
        "tls_ca_bundle": certifi.where(),
        "same_interpreter_command": f'"{executable}" -m evalt doctor',
        "same_interpreter_install_command": (
            f'"{executable}" -m pip install --upgrade '
            f'-r {HOSTED_RELEASE_REQUIREMENTS_URL}'
        ),
        "upgrade_required": not hosted_current,
        "installation_channel_note": (
            "Use same_interpreter_install_command when versions differ. Bare pip may "
            "select another Python installation, and the public PyPI release may trail "
            "Evalt's exact hosted wheel. The stable latest.txt channel resolves to "
            "one versioned wheel so this command does not go stale with the next release."
        ),
    }


def _console_entrypoint_identity() -> dict[str, object]:
    """Compare the PATH console shim with this interpreter without provider work."""

    entrypoint = shutil.which("evalt")
    if not entrypoint:
        return {
            "console_entrypoint": None,
            "console_entrypoint_version": None,
            "console_entrypoint_matches_sdk": None,
            "installation_warning": None,
        }
    version = None
    error = None
    try:
        completed = subprocess.run(
            [entrypoint, "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        output = (completed.stdout or completed.stderr or "").strip()
        version = output.split()[-1] if completed.returncode == 0 and output else None
        if completed.returncode != 0:
            error = f"console entrypoint exited {completed.returncode}"
    except (OSError, subprocess.SubprocessError) as exc:
        error = str(exc)[:180] or exc.__class__.__name__
    matches = version == _sdk_version() if version else None
    warning = None
    if matches is False:
        warning = (
            f"The PATH evalt command reports {version}, but this Python imports "
            f"evalt {_sdk_version()}. Use the quoted python_executable with -m evalt "
            "for connect, doctor, dashboard, and runs."
        )
    elif error:
        warning = (
            "The PATH evalt command could not identify itself. Use the quoted "
            "python_executable with -m evalt to avoid a stale console shim."
        )
    return {
        "console_entrypoint": str(Path(entrypoint).resolve()),
        "console_entrypoint_version": version,
        "console_entrypoint_matches_sdk": matches,
        "installation_warning": warning,
    }


STARTER_SUITE = {
    "schema": "evalt-suite-v2",
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
    root.add_argument("--version", action="version", version=f"evalt {_sdk_version()}")
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

    run = commands.add_parser("run", help="Design, test, and execute one prompt through a durable route.")
    run.add_argument("--route", required=True, help="Stable route name used to remember decisions.")
    run.add_argument("--task", help="Plain-language recurring job; omitted uses the prompt as the task description.")
    run.add_argument("--prompt", required=True)
    run.add_argument("--input", required=True)
    run.add_argument("--price", "--budget", dest="price", type=float, help="Optional hard maximum provider price for one production response; omitted uses a request-sized automatic safety ceiling.")
    run.add_argument("--test-budget", "--maintenance-budget", dest="test_budget", default="auto", help="Automatic or numeric hard cap for a due retest.")
    run.add_argument("--max-test-budget", type=float, default=1.0, help="Hard ceiling when --test-budget=auto.")
    run.add_argument("--target-accuracy", type=float, default=0.95)
    run.add_argument("--objective", choices=("match_baseline_at_lowest_cost", "best_within_price", "lowest_cost_at_accuracy"), default="lowest_cost_at_accuracy")
    run.add_argument("--state", default=".evalt/evalt.db")
    run.add_argument("--model", action="append", dest="models")
    run.add_argument("--cases", type=int, default=25, help="AI-designed cases for a new route; 25 provides ten distinct final-test scenarios.")
    run.add_argument("--bootstrap-only", action="store_true", help="Skip first-route optimization and make one explicitly untested provider call.")
    run.add_argument("--fixed-prompt", action="store_true", help="Compare routes without rewriting the supplied prompt or adding few-shot examples.")
    run.add_argument("--approved-output", help="Immediately record an accepted/corrected answer for future tests.")

    status = commands.add_parser("status", help="Show the durable decision trail for one route; no provider call.")
    status.add_argument("--route", required=True)
    status.add_argument("--state", default=".evalt/evalt.db")

    versions = commands.add_parser(
        "versions",
        help="List immutable qualified packages for one local route; no provider call.",
    )
    versions.add_argument("--route", required=True)
    versions.add_argument("--state", default=".evalt/evalt.db")

    annotate_version = commands.add_parser(
        "annotate-version",
        help="Name or describe one qualified route version locally; no provider call or dashboard sync.",
    )
    annotate_version.add_argument("--route", required=True)
    annotate_version.add_argument(
        "--version",
        required=True,
        dest="package_id",
        help="Exact rv_ version ID from `evalt versions`.",
    )
    alias_action = annotate_version.add_mutually_exclusive_group()
    alias_action.add_argument("--alias", help="Private local alias, such as known-good.")
    alias_action.add_argument(
        "--clear-alias",
        action="store_true",
        help="Remove the private local alias.",
    )
    note_action = annotate_version.add_mutually_exclusive_group()
    note_action.add_argument("--note", help="Private local note (240 characters max).")
    note_action.add_argument(
        "--clear-note",
        action="store_true",
        help="Remove the private local note.",
    )
    annotate_version.add_argument(
        "--expected-alias",
        help="Only update if the current alias matches; use an empty value for no alias.",
    )
    annotate_version.add_argument("--state", default=".evalt/evalt.db")

    rollback = commands.add_parser(
        "rollback",
        help="Atomically restore a qualified local route package; no provider call.",
    )
    rollback.add_argument("--route", required=True)
    rollback.add_argument(
        "--version",
        required=True,
        dest="package_id",
        help="Exact rv_ version ID or unambiguous private local alias.",
    )
    rollback.add_argument("--state", default=".evalt/evalt.db")
    rollback.add_argument(
        "--yes",
        action="store_true",
        help="Confirm changing the locally serving package.",
    )

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
    check.add_argument(
        "--baseline",
        help=(
            "Earlier result from the same frozen suite. Rejects incompatible "
            "contracts and regressions without provider calls."
        ),
    )
    check.add_argument(
        "--max-regressions",
        type=int,
        default=0,
        help="Maximum previously passing final-test cases allowed to fail (default: 0).",
    )
    check.add_argument(
        "--max-quality-drop-pp",
        type=float,
        default=0.0,
        help="Maximum aggregate final-test quality drop in percentage points (default: 0).",
    )
    check.add_argument(
        "--max-cost-increase-pct",
        type=float,
        help="Optional maximum production cost increase versus the baseline, in percent.",
    )
    check.add_argument(
        "--max-p90-increase-ms",
        type=float,
        help="Optional maximum p90 latency increase versus the baseline, in milliseconds.",
    )
    check.add_argument("--json", action="store_true", help="Print the gate report as JSON.")

    connect = commands.add_parser("connect", help="Connect local route metadata to a private hosted workspace.")
    connect.add_argument("token", nargs="?", help="Existing evw_ workspace token; omitted creates one.")
    connect.add_argument("--state", help="Scope the connection to one route database; omitted saves it for all local projects.")
    connect.add_argument("--api-url", default=DEFAULT_DASHBOARD_API_URL)
    connect.add_argument("--app-url", default=DEFAULT_DASHBOARD_APP_URL)
    connect.add_argument("--no-open", action="store_true", help="Do not open the connected dashboard in a browser.")
    connect.add_argument("--no-sync-existing", action="store_true", help="Save the connection without publishing current sanitized route summaries.")

    dashboard = commands.add_parser("dashboard", help="Open the connected hosted workspace without exposing the token in output.")
    dashboard.add_argument("--state", help="Prefer a project-scoped connection for this route database.")
    dashboard.add_argument("--status", action="store_true", help="Show the connected workspace ID without opening a browser.")
    dashboard.add_argument("--sync-existing", action="store_true", help="Publish current sanitized route summaries without provider calls.")

    doctor = commands.add_parser("doctor", help="Diagnose the local package, route database, and hosted workspace without provider calls.")
    doctor.add_argument("--state", default=".evalt/evalt.db")

    disconnect = commands.add_parser("disconnect", help="Remove the local hosted-workspace connection.")
    disconnect.add_argument("--state", help="Remove a project-scoped connection instead of the user-wide default.")
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
        "final_test_evidence_status": result.winner.final_test_evidence_status,
        "final_test_confidence_level": result.winner.final_test_confidence_level,
        "final_test_accuracy_lower_bound": result.winner.final_test_accuracy_lower_bound,
        "target_accuracy_statistically_supported": result.winner.target_accuracy_statistically_supported,
        "minimum_zero_failure_scenarios": result.winner.minimum_zero_failure_scenarios,
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
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    # People naturally try `evalt --status`. Keep the documented nested command,
    # but make this safe shorthand diagnose the connected workspace too.
    if resolved_argv and resolved_argv[0] == "--status":
        resolved_argv = ["dashboard", "--status", *resolved_argv[1:]]
    args = parser().parse_args(resolved_argv)
    try:
        if args.command == "connect":
            token = args.token or generate_workspace_token()
            path = save_dashboard_config(
                token, state_path=args.state, api_url=args.api_url, app_url=args.app_url
            )
            dashboard_url = f"{str(args.app_url).rstrip('/')}#workspace={token}"
            opened = False if args.no_open else bool(webbrowser.open(dashboard_url))
            sync = {
                "local_route_count": 0,
                "route_summaries_queued": 0,
                "sync_succeeded": None,
                "sync_error": None,
            }
            if not args.no_sync_existing:
                sync = Evalt(
                    transport=_OfflineTransport(),
                    state_path=args.state or ".evalt/evalt.db",
                    dashboard_token=token,
                    dashboard_api_url=args.api_url,
                ).sync_existing_routes()
            hosted = inspect_workspace(token, api_url=args.api_url)
            payload = {
                "connected": True,
                "workspace_id": workspace_fingerprint(token),
                "config": str(path),
                "dashboard": str(args.app_url).rstrip("/"),
                "browser_opened": opened,
                **sync,
                **hosted,
                **_runtime_identity(),
                "sync_scope": "route metadata and bounded progress only; prompts, inputs, outputs, cases, provider keys, and raw responses stay local",
            }
            if args.no_open and args.token is None:
                payload["workspace_token"] = token
                payload["warning"] = "Treat workspace_token like a password; it grants access to synced route metadata."
            print(json.dumps(payload, indent=2))
            return 0 if hosted["hosted_reachable"] and sync["sync_succeeded"] is not False else 2
        if args.command == "dashboard":
            config = load_dashboard_config(args.state)
            if not config:
                raise ValueError(
                    "No hosted workspace is connected. Run: "
                    f'"{Path(sys.executable).resolve()}" -m evalt connect'
                )
            sync = None
            if args.sync_existing:
                sync = Evalt(
                    transport=_OfflineTransport(),
                    state_path=args.state or ".evalt/evalt.db",
                    dashboard_token=config["workspace_token"],
                    dashboard_api_url=config["api_url"],
                ).sync_existing_routes()
            opened = False if args.status else bool(webbrowser.open(f"{config['app_url']}#workspace={config['workspace_token']}"))
            payload = {
                "connected": True,
                "workspace_id": workspace_fingerprint(config["workspace_token"]),
                "opened": opened,
                "dashboard": config["app_url"],
                "config": config.get("config_path", str(dashboard_config_path(args.state))),
                "workspace_token_printed": False,
                **inspect_workspace(config["workspace_token"], api_url=config["api_url"]),
                **_runtime_identity(),
            }
            if sync is not None:
                payload.update(sync)
            print(json.dumps(payload, indent=2))
            return 0
        if args.command == "doctor":
            state = Path(args.state).expanduser().resolve()
            config = load_dashboard_config(state)
            local_route_count = 0
            if state.exists():
                local_route_count = len(
                    Evalt(transport=_OfflineTransport(), state_path=state).router.list_routes()
                )
            payload = {
                **_runtime_identity(),
                **_console_entrypoint_identity(),
                "route_state": str(state),
                "local_route_count": local_route_count,
                "connected": bool(config),
                "workspace_id": workspace_fingerprint(config["workspace_token"]) if config else None,
                "config": config.get("config_path") if config else None,
                "workspace_token_printed": False,
            }
            if config:
                payload.update(inspect_workspace(config["workspace_token"], api_url=config["api_url"]))
                payload["next"] = (
                    f'Run `"{Path(sys.executable).resolve()}" -m evalt dashboard '
                    f'--state "{state}" --sync-existing` to publish current route summaries.'
                    if local_route_count and not payload.get("remote_route_count")
                    else "Local and hosted workspace diagnostics completed."
                )
            else:
                payload.update({"hosted_reachable": False, "remote_route_count": None, "hosted_error": None})
                payload["next"] = (
                    f'Run `"{Path(sys.executable).resolve()}" -m evalt connect` '
                    "to create or attach a private hosted workspace."
                )
            print(json.dumps(payload, indent=2))
            return 0
        if args.command == "disconnect":
            removed = remove_dashboard_config(args.state)
            print(json.dumps({
                "connected": False,
                "local_config_removed": removed,
                "connection_scope": "project" if args.state else "all local projects",
                "remote_metadata_deleted": False,
            }, indent=2))
            return 0
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
            evalt = Evalt(state_path=args.state)
            answer = evalt.run(
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
                task=args.task,
                first_run="bootstrap" if args.bootstrap_only else "optimize",
                case_count=args.cases,
            )
            if args.approved_output is not None:
                if args.approved_output == answer.content:
                    answer.accept()
                else:
                    answer.correct(args.approved_output)
            evalt.flush_dashboard()
            print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
            return 0
        if args.command == "status":
            print(json.dumps(Evalt(transport=_OfflineTransport(), state_path=args.state).route_status(args.route), indent=2, ensure_ascii=False))
            return 0
        if args.command == "versions":
            evalt = Evalt(transport=_OfflineTransport(), state_path=args.state)
            versions = evalt.route_versions(args.route)
            print(json.dumps({
                "route": args.route,
                "qualified_package_count": len(versions),
                "versions": versions,
                "provider_call_started": False,
            }, indent=2, ensure_ascii=False))
            return 0
        if args.command == "annotate-version":
            alias = "" if args.clear_alias else args.alias
            note = "" if args.clear_note else args.note
            if alias is None and note is None:
                raise ValueError(
                    "Choose --alias, --clear-alias, --note, or --clear-note."
                )
            evalt = Evalt(transport=_OfflineTransport(), state_path=args.state)
            version = evalt.annotate_route_version(
                args.route,
                args.package_id,
                alias=alias,
                note=note,
                expected_alias=args.expected_alias,
            )
            print(json.dumps({
                "annotated": True,
                "route": args.route,
                "version": version,
                "provider_call_started": False,
                "dashboard_sync_started": False,
            }, indent=2, ensure_ascii=False))
            return 0
        if args.command == "rollback":
            if not args.yes:
                raise ValueError(
                    "Rollback changes the locally serving package. Review `evalt "
                    f"versions --route {args.route!r}` and repeat with --yes."
                )
            evalt = Evalt(transport=_OfflineTransport(), state_path=args.state)
            result = evalt.rollback_route(args.route, args.package_id)
            print(json.dumps({
                "rolled_back": True,
                "route": args.route,
                **result,
            }, indent=2, ensure_ascii=False))
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
            if args.models:
                suite = replace(suite, models=tuple(args.models))
            if args.max_parallel_models is not None:
                suite = replace(suite, max_parallel_models=args.max_parallel_models)
            if args.max_parallel_scenarios is not None:
                suite = replace(suite, max_parallel_scenarios=args.max_parallel_scenarios)
            if args.fixed_prompt:
                suite = replace(suite, optimize_prompt=False)
            progress = _CliProgress(len(suite.models))
            client = Evalt(
                request_timeout_seconds=request_timeout_seconds,
                progress_callback=progress,
                show_progress=False,
            )
            try:
                result = client.run(suite)
            except (BudgetExceeded, ProviderError) as error:
                failure = {
                    "schema": "evalt-run-failure-v1",
                    "status": "INCOMPLETE",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "suite": str(args.suite),
                    "requested_models": list(suite.models),
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
            candidate_payload = json.load(handle)
        if not args.baseline and (
            args.max_regressions != 0
            or args.max_quality_drop_pp != 0
            or args.max_cost_increase_pct is not None
            or args.max_p90_increase_ms is not None
        ):
            raise ValueError(
                "Baseline regression options require --baseline BASELINE_RESULT."
            )
        report = check_result(
            candidate_payload,
            min_pass_rate=args.min_pass_rate,
            max_cost_per_success_usd=args.max_cost_per_success,
            require_complete_coverage=args.require_complete_coverage,
        )
        baseline_report = None
        if args.baseline:
            with Path(args.baseline).open(encoding="utf-8") as handle:
                baseline_payload = json.load(handle)
            baseline_report = check_regression(
                baseline_payload,
                candidate_payload,
                max_regressions=args.max_regressions,
                max_quality_drop_percentage_points=args.max_quality_drop_pp,
                max_cost_increase_percent=args.max_cost_increase_pct,
                max_p90_increase_ms=args.max_p90_increase_ms,
            )
        failures = list(report.failures)
        if baseline_report is not None:
            failures.extend(baseline_report.failures)
        passed = not failures
        payload = {
            "schema": "evalt-ci-gate-v2",
            "passed": passed,
            "failures": failures,
            "provider_call_started": False,
            "holdout_pass_rate": report.holdout_pass_rate,
            "estimated_cost_per_successful_call_usd": (
                report.estimated_cost_per_successful_call_usd
            ),
            "winner_scope": report.winner_scope,
            "absolute_gate": report.to_dict(),
            "baseline_gate": (
                baseline_report.to_dict() if baseline_report is not None else None
            ),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        elif passed:
            suffix = ""
            if baseline_report is not None:
                suffix = (
                    f"; baseline {baseline_report.quality_delta_percentage_points:+.3f} pp, "
                    f"{baseline_report.regressions} case regressions"
                )
            print(f"PASS: holdout pass rate {report.holdout_pass_rate:.1%}{suffix}")
        else:
            print("FAIL: " + "; ".join(failures), file=sys.stderr)
        return 0 if passed else 1
    except (BudgetExceeded, ProviderError, ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        print(f"evalt: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
