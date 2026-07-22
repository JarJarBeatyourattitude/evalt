"""Explicit, metadata-only bridge from local Evalt routes to the hosted workspace."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import queue
import secrets
import threading
import time
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_DASHBOARD_API_URL = "https://evalt.onrender.com"
DEFAULT_DASHBOARD_APP_URL = "https://evalt.dev/app"
_TOKEN_PREFIX = "evw_"
_SENSITIVE_KEYS = {
    "prompt", "selected_prompt", "input", "output", "content", "messages", "cases",
    "examples", "api_key", "openrouter_api_key", "authorization", "request_options",
    "tested_request_options", "tool_calls", "raw_response",
}


def generate_workspace_token() -> str:
    return _TOKEN_PREFIX + secrets.token_urlsafe(32)


def validate_workspace_token(token: str) -> str:
    token = str(token or "").strip()
    if not token.startswith(_TOKEN_PREFIX) or len(token) < 44:
        raise ValueError("Workspace token must be an Evalt token beginning with evw_.")
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    if any(character not in allowed for character in token[len(_TOKEN_PREFIX):]):
        raise ValueError("Workspace token contains unsupported characters.")
    return token


def workspace_fingerprint(token: str) -> str:
    """Return a safe, stable identifier users can compare across devices."""

    validated = validate_workspace_token(token)
    digest = hashlib.sha256(validated.encode("utf-8")).hexdigest()[:12]
    return f"ws_{digest}"


def global_dashboard_config_path() -> Path:
    """Return the user-wide workspace connection used across project folders."""

    configured_home = os.environ.get("EVALT_CONFIG_HOME", "").strip()
    if configured_home:
        root = Path(configured_home).expanduser()
    else:
        try:
            root = Path.home() / ".evalt"
        except RuntimeError:
            # Minimal containers may omit every conventional home-directory
            # variable. Keep the optional dashboard bridge non-fatal there.
            root = Path.cwd() / ".evalt"
    return root.resolve() / "dashboard.json"


def dashboard_config_path(state_path: str | Path | None = None) -> Path:
    if state_path is None:
        return global_dashboard_config_path()
    return Path(state_path).expanduser().resolve().parent / "dashboard.json"


def save_dashboard_config(
    token: str,
    *,
    state_path: str | Path | None = None,
    api_url: str = DEFAULT_DASHBOARD_API_URL,
    app_url: str = DEFAULT_DASHBOARD_APP_URL,
) -> Path:
    token = validate_workspace_token(token)
    path = dashboard_config_path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "schema": "evalt-dashboard-config-v1",
        "workspace_token": token,
        "api_url": str(api_url).rstrip("/"),
        "app_url": str(app_url).rstrip("/"),
    }, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    temporary.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def load_dashboard_config(state_path: str | Path | None = None) -> dict[str, str] | None:
    token = os.environ.get("EVALT_WORKSPACE_TOKEN", "").strip()
    api_url = os.environ.get("EVALT_DASHBOARD_API_URL", DEFAULT_DASHBOARD_API_URL).strip()
    app_url = os.environ.get("EVALT_DASHBOARD_APP_URL", DEFAULT_DASHBOARD_APP_URL).strip()
    if token:
        return {
            "workspace_token": validate_workspace_token(token),
            "api_url": api_url.rstrip("/"),
            "app_url": app_url.rstrip("/"),
            "config_path": "environment:EVALT_WORKSPACE_TOKEN",
        }
    candidates = []
    if state_path is not None:
        candidates.append(dashboard_config_path(state_path))
    candidates.append(global_dashboard_config_path())
    checked: set[Path] = set()
    for path in candidates:
        if path in checked or not path.exists():
            continue
        checked.add(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        return {
            "workspace_token": validate_workspace_token(value.get("workspace_token", "")),
            "api_url": str(value.get("api_url") or api_url).rstrip("/"),
            "app_url": str(value.get("app_url") or app_url).rstrip("/"),
            "config_path": str(path),
        }
    return None


def remove_dashboard_config(state_path: str | Path | None = None) -> bool:
    path = dashboard_config_path(state_path)
    if not path.exists():
        return False
    path.unlink()
    return True


def _safe_scalar(value: Any, maximum: int = 240) -> Any:
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return str(value)[:maximum]


def sanitize_progress_event(event: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "event", "route", "model", "designer_model", "evaluator_model", "winner_model",
        "case_count", "completed", "total", "matched_checks", "checks", "passed",
        "validation_pass_rate", "holdout_pass_rate", "final_test_pass_rate",
        "provider_cost_usd", "workflow_spend_usd", "test_budget_spent_usd",
        "workflow_budget_usd", "remaining_budget_usd", "elapsed_seconds",
        "reasoning_effort", "to_effort", "phase", "error", "decision_reason",
        "evidence_provenance", "request_envelope_validated",
        "attempt", "max_attempts", "will_retry", "configurations",
        "completed_configurations", "parallel_models", "screening_scenarios",
        "validation_scenarios", "final_test_scenarios", "final_test_executions",
        "passed_quality_floor", "target_latency_p50_ms", "target_latency_p90_ms",
        "optimization_spend_usd", "screening_spend_usd",
        "estimated_production_cost_per_call_usd", "prompt_candidates_tested",
        "prompt_rewrites_tested", "selected_prompt_changed", "optimize_prompt",
        "candidate_prompt_packages", "source_model", "few_shot_examples",
        "status", "quality_gate_status", "judge_calibrated",
        "judge_calibration_checks", "tested_configurations", "candidate",
        "kind", "training_pass_rate", "selected", "quality_threshold",
        "reason", "prompt_hash",
        "run_id", "run_state", "run_started_at", "run_finished_at",
        "test_design_seconds", "tournament_seconds", "route_install_seconds",
        "production_call_seconds", "orchestration_seconds", "total_elapsed_seconds",
        "model_elapsed_seconds",
        "final_test_evidence_status", "final_test_confidence_level",
        "final_test_accuracy_lower_bound", "target_accuracy_statistically_supported",
        "minimum_zero_failure_scenarios",
    }
    safe = {
        key: (
            "Provider or workflow error; inspect local logs."
            if key == "error" and value
            else _safe_scalar(value, 180)
        )
        for key, value in event.items()
        if key in allowed and key.lower() not in _SENSITIVE_KEYS
    }
    safe["event"] = str(safe.get("event") or "progress")[:80]
    safe["route"] = str(safe.get("route") or "default")[:80]
    safe["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return safe


def sanitize_route_snapshot(status: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema", "route", "selected_model", "selected_prompt_version", "decision_reason",
        "route_phase", "evidence_provenance", "selected_few_shot_messages",
        "selected_few_shot_examples", "price_usd", "price_policy",
        "effective_price_ceiling_usd", "test_budget_usd", "test_budget_policy",
        "target_accuracy", "objective", "max_p90_latency_seconds", "optimize_prompt",
        "target_max_tokens", "total_calls", "feedback_count", "maintenance_due",
        "catalog_revision", "tested_catalog_revision",
        "latest_run_id", "latest_run_state", "latest_run_started_at",
        "latest_run_finished_at",
    }
    safe = {
        key: (
            [str(item)[:120] for item in value[:20]]
            if key == "maintenance_due" and isinstance(value, (list, tuple))
            else _safe_scalar(value)
        )
        for key, value in status.items()
        if key in allowed and key.lower() not in _SENSITIVE_KEYS
    }
    summary = status.get("last_test_summary")
    if isinstance(summary, Mapping):
        summary_allowed = {
            "winner_model", "winner_prompt_version", "holdout_pass_rate", "final_test_pass_rate",
            "final_test_scenarios", "final_test_executions", "estimated_cost_per_successful_call_usd",
            "optimization_spend_usd", "quality_gate_status", "few_shot_examples",
            "designer_model", "evaluator_model", "judge_calibration_checks", "request_options_sha256",
            "final_test_evidence_status", "final_test_confidence_level",
            "final_test_accuracy_lower_bound", "target_accuracy_statistically_supported",
            "minimum_zero_failure_scenarios",
        }
        safe["last_test_summary"] = {key: _safe_scalar(value) for key, value in summary.items() if key in summary_allowed}
    safe["schema"] = "evalt-workspace-route-v1"
    safe["route"] = str(status.get("route") or "default")[:80]
    safe["synced_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return safe


class WorkspaceSync:
    """Best-effort background publisher; dashboard failures never fail provider work."""

    def __init__(
        self,
        token: str,
        *,
        api_url: str = DEFAULT_DASHBOARD_API_URL,
        timeout_seconds: float = 8.0,
        sender: Callable[[str, str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.token = validate_workspace_token(token)
        self.workspace_id = workspace_fingerprint(self.token)
        self.api_url = str(api_url).rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.last_error: str | None = None
        self._sender = sender or self._send
        self._queue: queue.Queue[tuple[str, str, dict[str, Any]]] = queue.Queue(maxsize=128)
        self._worker = threading.Thread(target=self._work, name="evalt-dashboard-sync", daemon=True)
        self._worker.start()

    def _send(self, method: str, path: str, payload: dict[str, Any]) -> None:
        request = Request(
            f"{self.api_url}{path}",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            method=method,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            if response.status >= 300:
                raise OSError(f"dashboard returned HTTP {response.status}")

    def _work(self) -> None:
        while True:
            batch = [self._queue.get()]
            # A run often emits a progress milestone and its final route back
            # to back. Coalesce that burst into one durable hosted write so a
            # short command does not need to wait for two R2 round trips.
            time.sleep(0.08)
            while len(batch) < 50:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            grouped: dict[str, dict[str, Any]] = {}
            for method, _path, payload in batch:
                route = str(payload.get("route") or "default")[:80]
                group = grouped.setdefault(route, {"snapshot": None, "events": []})
                if method == "PUT":
                    group["snapshot"] = payload
                else:
                    group["events"].append(payload)
            last_error: str | None = None
            for route, payload in grouped.items():
                try:
                    self._sender("POST", f"/api/workspace/routes/{quote(route, safe='')}/sync", payload)
                except (HTTPError, URLError, OSError, TimeoutError, ValueError) as error:
                    last_error = str(error)[:300]
            self.last_error = last_error
            for _item in batch:
                self._queue.task_done()

    def _publish(self, method: str, path: str, payload: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait((method, path, payload))
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait((method, path, payload))

    def publish_event(self, event: Mapping[str, Any]) -> None:
        safe = sanitize_progress_event(event)
        self._publish("POST", f"/api/workspace/routes/{quote(safe['route'], safe='')}/events", safe)

    def publish_route(self, status: Mapping[str, Any]) -> None:
        safe = sanitize_route_snapshot(status)
        self._publish("PUT", f"/api/workspace/routes/{quote(safe['route'], safe='')}", safe)

    def flush(self, timeout_seconds: float = 10.0) -> bool:
        deadline = time.monotonic() + max(0, timeout_seconds)
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0 and self.last_error is None


__all__ = [
    "DEFAULT_DASHBOARD_API_URL", "DEFAULT_DASHBOARD_APP_URL", "WorkspaceSync",
    "dashboard_config_path", "generate_workspace_token", "global_dashboard_config_path", "load_dashboard_config",
    "remove_dashboard_config", "sanitize_progress_event", "sanitize_route_snapshot",
    "save_dashboard_config", "validate_workspace_token", "workspace_fingerprint",
]
