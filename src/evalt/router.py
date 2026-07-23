"""Durable, budget-bounded runtime routing for Evalt.

The router keeps serving a qualified prompt/model pair while collecting explicit
feedback.  When enough traffic and approved examples exist, a bounded maintenance
run may test current models and promote a cheaper passing pair.  SQLite is the
runtime source of truth; JSON is only an optional audit export.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import uuid
import warnings as runtime_warnings
from typing import Any, Callable, Iterable, Mapping, Sequence

from last_good_prompt.core import (
    BudgetExceeded, Client, Completion, Example, OptimizationResult, _Budget,
    normalize_message_content, normalize_request_options,
    request_options_fingerprint, required_input_modalities,
    safe_input_descriptor,
)


DEFAULT_TARGETS = (
    "openai/gpt-5-mini",
    "google/gemini-3-flash-preview",
    "qwen/qwen3.5-9b",
)


class RequestEnvelopeDriftWarning(UserWarning):
    """A production call changed settings that were part of route qualification."""


class RouteVersionMismatch(RuntimeError):
    """A production call cannot serve the exact qualified package it requested."""

    def __init__(
        self,
        *,
        route: str,
        expected_package_id: str,
        current_package_id: str | None,
        reason: str,
    ) -> None:
        self.route = route
        self.expected_package_id = expected_package_id
        self.current_package_id = current_package_id
        self.reason = reason
        self.provider_call_started = False
        current = current_package_id or "none"
        super().__init__(
            f"Route {route!r} cannot serve requested version "
            f"{expected_package_id!r}; current version is {current!r} ({reason}). "
            "No provider call was started. Inspect qualified versions with "
            f"'python3 -m evalt versions --route {json.dumps(route)}' and deliberately update "
            "the deployment pin or roll back the route."
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: Sequence[int], percentile: float) -> int | None:
    ordered = sorted(int(value) for value in values if int(value) > 0)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return round(
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _content(value: Any) -> str:
    descriptor = safe_input_descriptor(value)
    if isinstance(descriptor, str):
        return descriptor
    return json.dumps(
        descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _input_messages(value: Any) -> list[dict[str, Any]]:
    """Normalize one production input without permitting prompt replacement."""

    if isinstance(value, Mapping) and value.get("role"):
        candidates = [value]
    elif (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(item, Mapping) and item.get("role") for item in value)
    ):
        candidates = list(value)
    else:
        return [{"role": "user", "content": normalize_message_content(value)}]
    messages: list[dict[str, Any]] = []
    for raw in candidates:
        role = str(raw.get("role") or "").strip()
        if role not in {"user", "assistant", "tool"}:
            raise ValueError(
                "Production input messages may use user, assistant, or tool roles; "
                "the route's tested system prompt remains Evalt-owned."
            )
        message = dict(raw)
        message["role"] = role
        message["content"] = normalize_message_content(raw.get("content"))
        messages.append(message)
    return messages


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 1
    ordered = sorted(max(0, int(value)) for value in values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


def _route_phase(row: Mapping[str, Any]) -> str:
    """Describe only the strongest route evidence the database actually proves."""
    provenance = str(row["evidence_provenance"] or "LEGACY_UNKNOWN")
    if provenance == "HUMAN_FEEDBACK_CALIBRATED":
        return "human_calibrated"
    if provenance == "AI_GENERATED_AI_JUDGED":
        return "ai_tested"
    if provenance in {"HUMAN_APPROVED", "HUMAN_APPROVED_AI_DRAFT"}:
        return "human_approved"
    if provenance == "LEGACY_UNKNOWN":
        return "legacy_unknown"
    return "untested_bootstrap"


_QUALIFIED_EVIDENCE = {
    "AI_GENERATED_AI_JUDGED",
    "HUMAN_FEEDBACK_CALIBRATED",
    "HUMAN_APPROVED",
    "HUMAN_APPROVED_AI_DRAFT",
}

_PACKAGE_ID_RE = re.compile(r"^rv_[0-9a-f]{20}$")
_PACKAGE_ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,47}$")
_PACKAGE_NOTE_MAX_LENGTH = 240

_PACKAGE_FIELDS = (
    "prompt",
    "source_prompt_version",
    "prompt_version",
    "candidates_json",
    "selected_model",
    "selected_prompt",
    "selected_few_shot_json",
    "decision_reason",
    "quality_threshold",
    "tested_catalog_revision",
    "objective",
    "max_p90_latency_seconds",
    "latency_value_usd_per_second",
    "optimize_prompt",
    "evidence_provenance",
    "last_test_summary_json",
    "target_max_tokens",
    "tested_request_options_json",
    "tested_request_options_sha256",
    "tested_input_modalities_json",
)


@dataclass(frozen=True)
class RolePlan:
    """The three separately costed model roles used by a maintenance run."""

    tier: str
    test_designer_model: str
    judge_model: str
    target_models: tuple[str, ...]
    catalog_revision: str
    policy: str
    designer_candidates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["target_models"] = list(self.target_models)
        value["designer_candidates"] = list(self.designer_candidates)
        return value


def select_role_plan(
    catalog: Sequence[Mapping[str, Any]],
    *,
    maintenance_budget_usd: float,
    fallback_targets: Iterable[str] = DEFAULT_TARGETS,
    fallback_designer: str = "openai/gpt-5.6-luna",
    fallback_judge: str = "qwen/qwen3.5-35b-a3b",
) -> RolePlan:
    """Choose role candidates from price + intelligence metadata.

    General benchmarks only shortlist models.  Promotion still requires the
    route's frozen, human-approved examples.
    """

    normalized: list[dict[str, Any]] = []
    for item in catalog:
        supported = {str(value) for value in item.get("supported_parameters") or []}
        if supported and not ({"max_tokens", "max_completion_tokens"} & supported):
            continue
        try:
            raw_intelligence = item.get("intelligence")
            if raw_intelligence is None:
                # The models endpoint is requested in descending intelligence
                # order. Some catalog revisions omit the absolute benchmark value
                # while preserving that rank. Keep selecting dynamically instead
                # of silently dropping into a static role guess.
                rank = max(1, int(item["intelligence_rank"]))
                intelligence = max(0.0, 100.0 - 2.0 * (rank - 1))
                intelligence_source = "catalog_rank"
            else:
                intelligence = float(raw_intelligence)
                intelligence_source = "intelligence_index"
            price = float(item["blended_price"])
            model = str(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if model and intelligence >= 0 and price >= 0:
            reasoning = item.get("reasoning") or {}
            reasoning_supported = bool({"reasoning", "reasoning_effort"} & supported)
            reported_efforts = tuple(
                str(value) for value in reasoning.get("supported_efforts") or ()
            ) if reasoning_supported else ()
            # Preserve every endpoint-advertised effort. The optimizer stages
            # xhigh/max behind measured validation and latency evidence instead
            # of either launching them eagerly or hiding them from search.
            efforts = tuple(
                value for value in reported_efforts
                if value in {
                    "minimal", "low", "medium", "high", "xhigh", "max"
                }
            )
            if not reported_efforts and reasoning_supported:
                efforts = ("low", "medium", "high")
            if reasoning.get("mandatory") and not efforts:
                continue
            if not reasoning.get("mandatory"):
                efforts = ("none", *efforts)
            try:
                private_provider_routes = max(
                    0, int(item.get("private_provider_routes") or 0)
                )
            except (TypeError, ValueError):
                private_provider_routes = 0
            try:
                designer_provider_routes = max(
                    0,
                    int(
                        item.get("designer_provider_routes")
                        if item.get("designer_provider_routes") is not None
                        else private_provider_routes
                    ),
                )
            except (TypeError, ValueError):
                designer_provider_routes = 0
            try:
                designer_p90_latency_ms = float(item.get("designer_p90_latency_ms"))
            except (TypeError, ValueError):
                designer_p90_latency_ms = float("inf")
            designer_parameter_compatible = (
                not supported
                or bool({"response_format", "structured_outputs"} & supported)
            )
            normalized.append({
                "id": model,
                "intelligence": intelligence,
                "price": price,
                "private_provider_routes": private_provider_routes,
                "designer_provider_routes": designer_provider_routes,
                "designer_compatible": (
                    designer_parameter_compatible
                    and (designer_provider_routes > 0 or not supported)
                ),
                "designer_p90_latency_ms": designer_p90_latency_ms,
                "designer_default_effort": str(
                    reasoning.get("default_effort")
                    or ("medium" if reasoning.get("default_enabled") else "none")
                ),
                "reasoning_efforts": tuple(dict.fromkeys(efforts)),
                "intelligence_source": intelligence_source,
            })

    if maintenance_budget_usd < 0.50:
        tier, designer_delta, judge_delta, breadth = "lean", 12.0, 26.0, 5
    elif maintenance_budget_usd < 2.00:
        tier, designer_delta, judge_delta, breadth = "standard", 4.0, 18.0, 10
    else:
        tier, designer_delta, judge_delta, breadth = "deep", 0.0, 11.0, 16

    if not normalized:
        targets = tuple(dict.fromkeys(fallback_targets))[:breadth]
        return RolePlan(
            tier=tier,
            test_designer_model=fallback_designer,
            judge_model=fallback_judge,
            target_models=targets,
            catalog_revision="fallback",
            policy="No current intelligence metadata; explicit verified fallbacks are used until task tests can qualify a replacement.",
            designer_candidates=(fallback_designer,),
        )

    normalized.sort(key=lambda item: (item["price"], -item["intelligence"], item["id"]))
    maximum = max(item["intelligence"] for item in normalized)

    def cheapest_above(
        delta: float,
        *,
        exclude: set[str] | None = None,
        pool: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        excluded = exclude or set()
        source = list(pool) if pool is not None else normalized
        source_maximum = max(
            (item["intelligence"] for item in source),
            default=maximum,
        )
        eligible = [
            item for item in source
            if item["intelligence"] >= source_maximum - delta
            and item["id"] not in excluded
        ]
        if not eligible:
            eligible = [item for item in source if item["id"] not in excluded]
        if not eligible:
            eligible = list(normalized)
        return min(eligible, key=lambda item: (item["price"], -item["intelligence"]))["id"]

    frontier = [
        item for item in normalized
        if not any(
            other["id"] != item["id"]
            and other["price"] <= item["price"]
            and other["intelligence"] >= item["intelligence"]
            and (other["price"] < item["price"] or other["intelligence"] > item["intelligence"])
            for other in normalized
        )
    ]
    frontier.sort(key=lambda item: (item["price"], -item["intelligence"]))
    minimum_price = max(1e-12, min(item["price"] for item in frontier))
    knee = max(frontier, key=lambda item: item["intelligence"] - 8 * math.log10(max(item["price"], 1e-12) / minimum_price))
    strongest = max(frontier, key=lambda item: (item["intelligence"], -item["price"]))
    selected: list[dict[str, Any]] = []

    def add(item: dict[str, Any]) -> None:
        if item not in selected and len(selected) < breadth:
            selected.append(item)

    bootstrap_pool = [
        item for item in normalized
        if item["intelligence"] >= maximum - judge_delta
    ]
    bootstrap = min(
        bootstrap_pool,
        key=lambda item: (
            0 if item["private_provider_routes"] >= 2 else 1,
            item["price"],
            -item["intelligence"],
        ),
    )
    # The first production call occurs before route-specific feedback can qualify
    # a winner. Start it on a sufficiently intelligent route with provider
    # redundancy when available; the fragile absolute-cheapest model remains in
    # the measured tournament and can still win after evidence.
    add(bootstrap)
    add(frontier[0])
    add(knee)
    add(strongest)
    if breadth > len(selected):
        for index in range(breadth):
            add(frontier[round(index * (len(frontier) - 1) / max(1, breadth - 1))])
    for item in normalized:
        add(item)

    def base_effort(item: Mapping[str, Any]) -> str:
        efforts = item.get("reasoning_efforts") or ("none",)
        return "low" if "low" in efforts else "none" if "none" in efforts else str(efforts[0])

    # Broad phase: one honest endpoint-supported configuration per model.
    broad_ids = [f"{item['id']}#reasoning={base_effort(item)}" for item in selected]
    # Hone phase: alternate efforts stay behind breadth and are pruned unless the
    # broad result lands near the observed task-specific capability threshold.
    hone_ids: list[str] = []
    ordered_for_hone = sorted(
        selected, key=lambda value: (value["price"], -value["intelligence"])
    )
    # Interleave the next useful rung across models. Grouping every effort for
    # the cheapest model first meant the 25-configuration cap could omit medium
    # effort for GPT OSS entirely while testing several less useful variants of
    # one model. Medium is the first repair for the broad low-effort screen;
    # lower/no-reasoning and more extreme rungs remain available afterward.
    for effort in ("medium", "none", "high", "minimal", "xhigh", "max"):
        for item in ordered_for_hone:
            if effort not in (item.get("reasoning_efforts") or ("none",)):
                continue
            configured = f"{item['id']}#reasoning={effort}"
            if configured not in broad_ids and configured not in hone_ids:
                hone_ids.append(configured)
    target_ids = (broad_ids + hone_ids)[:25]
    revision = _hash(json.dumps(normalized, sort_keys=True, separators=(",", ":")))
    designer_pool = [item for item in normalized if item["designer_compatible"]]
    if not designer_pool:
        designer_pool = list(normalized)
    reliable_designer_pool = [
        item for item in designer_pool if item["designer_provider_routes"] >= 2
    ]
    if reliable_designer_pool:
        designer_pool = reliable_designer_pool
    designer_choice = cheapest_above(designer_delta, pool=designer_pool)
    judge_choice = cheapest_above(judge_delta, exclude={designer_choice})

    def bounded_designer_configuration(model_id: str) -> str:
        item = next(
            (value for value in designer_pool if value["id"] == model_id),
            None,
        )
        if (
            item is not None
            and item.get("designer_default_effort") != "none"
            and "none" in (item.get("reasoning_efforts") or ())
        ):
            return f"{model_id}#reasoning=none"
        return model_id
    fallback_designers = [
        item for item in designer_pool
        if item["intelligence"] >= max(
            value["intelligence"] for value in designer_pool
        ) - max(designer_delta, 12.0)
        and item["id"] not in {designer_choice, judge_choice}
    ]
    fallback_designers.sort(key=lambda item: (
        0 if item["designer_default_effort"] not in {"high", "xhigh", "max"} else 1,
        0 if item["designer_provider_routes"] >= 2 else 1,
        item["designer_p90_latency_ms"],
        item["price"],
        -item["intelligence"],
    ))
    designer_candidates = tuple(dict.fromkeys((
        bounded_designer_configuration(designer_choice),
        *(bounded_designer_configuration(item["id"]) for item in fallback_designers),
    )))[:4]
    return RolePlan(
        tier=tier,
        test_designer_model=bounded_designer_configuration(designer_choice),
        judge_model=judge_choice,
        target_models=tuple(target_ids),
        catalog_revision=revision,
        designer_candidates=designer_candidates,
        policy=(
            f"The suite designer uses the cheapest model within {designer_delta:g} intelligence points of the current benchmark leader; "
            f"judging uses the cheapest different model within {judge_delta:g} points and must pass route-specific calibration; "
            "designer failover stays inside a separate near-frontier live-catalog shortlist rather than borrowing the judge; "
            "the unqualified first call starts on a sufficiently intelligent provider-redundant route when available; production search then starts with one configuration across the price/intelligence frontier and spends remaining budget "
            "only on reasoning-effort variants in the task-specific capability band. Route holdouts, never benchmarks, promote the winner."
        ),
    )


@dataclass
class RoutedAnswer:
    content: str
    model: str
    route: str
    call_id: str
    provider_cost_usd: float
    prompt_version: str
    decision_reason: str
    maintenance_due: tuple[str, ...]
    route_version: str = ""
    route_phase: str = "untested_bootstrap"
    evidence_provenance: str = "UNTESTED_BOOTSTRAP"
    initial_test_summary: dict[str, Any] | None = None
    request_envelope_validated: bool = True
    request_options_sha256: str = ""
    tested_request_options_sha256: str = ""
    warnings: tuple[str, ...] = ()
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    message: dict[str, Any] = field(default_factory=dict)
    tool_calls: tuple[dict[str, Any], ...] = ()
    _router: "DurableRouter | None" = None
    _min_feedback: int = 5
    _retest_after_calls: int = 500
    _on_feedback: "Callable[[dict[str, Any]], None] | None" = None

    def accept(self) -> None:
        if not self._router:
            raise RuntimeError("This routed answer is detached from its Evalt state.")
        receipt = self._router.record_feedback(
            self.call_id,
            approved_output=self.content,
            verdict="accepted",
            min_feedback=self._min_feedback,
            retest_after_calls=self._retest_after_calls,
        )
        if self._on_feedback is not None:
            self._on_feedback(receipt)

    def correct(self, approved_output: Any) -> None:
        if not self._router:
            raise RuntimeError("This routed answer is detached from its Evalt state.")
        receipt = self._router.record_feedback(
            self.call_id,
            approved_output=_content(approved_output),
            verdict="corrected",
            min_feedback=self._min_feedback,
            retest_after_calls=self._retest_after_calls,
        )
        if self._on_feedback is not None:
            self._on_feedback(receipt)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "route": self.route,
            "call_id": self.call_id,
            "provider_cost_usd": self.provider_cost_usd,
            "prompt_version": self.prompt_version,
            "decision_reason": self.decision_reason,
            "maintenance_due": list(self.maintenance_due),
            "route_version": self.route_version,
            "route_phase": self.route_phase,
            "evidence_provenance": self.evidence_provenance,
            "initial_test_summary": self.initial_test_summary,
            "request_envelope_validated": self.request_envelope_validated,
            "request_options_sha256": self.request_options_sha256,
            "tested_request_options_sha256": self.tested_request_options_sha256,
            "warnings": list(self.warnings),
            "finish_reason": self.finish_reason,
            "native_finish_reason": self.native_finish_reason,
            "message": self.message,
            "tool_calls": list(self.tool_calls),
        }


class DurableRouter:
    """SQLite-backed prompt/model router with explicit bounded maintenance spend."""

    def __init__(self, client: Client, state_path: str | Path = ".evalt/evalt.db") -> None:
        self.client = client
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._maintenance_lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.state_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _db(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self._db() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS routes (
                    route TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    source_prompt_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    candidates_json TEXT NOT NULL,
                    selected_model TEXT NOT NULL,
                    selected_prompt TEXT NOT NULL,
                    decision_reason TEXT NOT NULL,
                    quality_threshold REAL NOT NULL,
                    total_calls INTEGER NOT NULL DEFAULT 0,
                    feedback_count INTEGER NOT NULL DEFAULT 0,
                    last_optimized_calls INTEGER NOT NULL DEFAULT 0,
                    last_optimized_feedback INTEGER NOT NULL DEFAULT 0,
                    catalog_revision TEXT NOT NULL DEFAULT 'unseen',
                    tested_catalog_revision TEXT NOT NULL DEFAULT 'unseen',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS calls (
                    call_id TEXT PRIMARY KEY,
                    route TEXT NOT NULL REFERENCES routes(route),
                    created_at TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    output_text TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    provider_cost_usd REAL NOT NULL,
                    budget_usd REAL NOT NULL,
                    decision_reason TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    call_id TEXT PRIMARY KEY REFERENCES calls(call_id),
                    route TEXT NOT NULL REFERENCES routes(route),
                    verdict TEXT NOT NULL,
                    approved_output TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    event_id TEXT PRIMARY KEY,
                    route TEXT NOT NULL REFERENCES routes(route),
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    detail_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS route_packages (
                    package_id TEXT PRIMARY KEY,
                    route TEXT NOT NULL REFERENCES routes(route),
                    created_at TEXT NOT NULL,
                    activation_reason TEXT NOT NULL,
                    rollback_of_package_id TEXT,
                    alias TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    annotation_updated_at TEXT,
                    snapshot_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS route_packages_route_created
                    ON route_packages(route, created_at DESC);
                """
            )
            columns = {row["name"] for row in db.execute("PRAGMA table_info(routes)")}
            if "source_prompt_version" not in columns:
                db.execute("ALTER TABLE routes ADD COLUMN source_prompt_version TEXT NOT NULL DEFAULT ''")
                db.execute("UPDATE routes SET source_prompt_version=prompt_version WHERE source_prompt_version='' ")
            for name, declaration in (
                ("max_cost_per_run_usd", "REAL NOT NULL DEFAULT 0.02"),
                ("price_policy", "TEXT NOT NULL DEFAULT 'legacy'"),
                ("test_budget_usd", "REAL NOT NULL DEFAULT 0"),
                ("objective", "TEXT NOT NULL DEFAULT 'best_within_price'"),
                ("test_budget_policy", "TEXT NOT NULL DEFAULT 'legacy'"),
                ("max_p90_latency_seconds", "REAL DEFAULT NULL"),
                ("latency_value_usd_per_second", "REAL NOT NULL DEFAULT 0"),
                ("optimize_prompt", "INTEGER NOT NULL DEFAULT 1"),
                ("selected_few_shot_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("evidence_provenance", "TEXT NOT NULL DEFAULT 'LEGACY_UNKNOWN'"),
                ("last_test_summary_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("target_max_tokens", "INTEGER NOT NULL DEFAULT 600"),
                ("tested_request_options_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("tested_request_options_sha256", "TEXT NOT NULL DEFAULT ''"),
                ("tested_input_modalities_json", "TEXT NOT NULL DEFAULT '[\"text\"]'"),
                ("current_package_id", "TEXT NOT NULL DEFAULT ''"),
            ):
                if name not in columns:
                    db.execute(f"ALTER TABLE routes ADD COLUMN {name} {declaration}")
            empty_options_hash = request_options_fingerprint({})
            db.execute(
                "UPDATE routes SET tested_request_options_sha256=? "
                "WHERE tested_request_options_sha256=''",
                (empty_options_hash,),
            )
            call_columns = {row["name"] for row in db.execute("PRAGMA table_info(calls)")}
            for name, declaration in (
                ("request_options_sha256", "TEXT NOT NULL DEFAULT ''"),
                ("request_envelope_validated", "INTEGER NOT NULL DEFAULT 1"),
                ("input_replayable", "INTEGER NOT NULL DEFAULT 1"),
                ("latency_ms", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if name not in call_columns:
                    db.execute(f"ALTER TABLE calls ADD COLUMN {name} {declaration}")
            package_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(route_packages)")
            }
            for name, declaration in (
                ("alias", "TEXT NOT NULL DEFAULT ''"),
                ("note", "TEXT NOT NULL DEFAULT ''"),
                ("annotation_updated_at", "TEXT DEFAULT NULL"),
            ):
                if name not in package_columns:
                    db.execute(
                        f"ALTER TABLE route_packages ADD COLUMN {name} {declaration}"
                    )
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS route_packages_route_alias "
                "ON route_packages(route,alias) WHERE alias<>''"
            )
            self._backfill_qualified_packages(db)

    @staticmethod
    def _package_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
        """Capture the complete locally serving, tested package without call data."""

        return {field: row[field] for field in _PACKAGE_FIELDS}

    @staticmethod
    def _qualified_snapshot(snapshot: Mapping[str, Any]) -> bool:
        provenance = str(snapshot.get("evidence_provenance") or "")
        try:
            summary = json.loads(str(snapshot.get("last_test_summary_json") or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return (
            provenance in _QUALIFIED_EVIDENCE
            and isinstance(summary, Mapping)
            and summary.get("quality_gate_status") == "QUALIFIED_ROUTE_SELECTED"
        )

    def _capture_package(
        self,
        db: sqlite3.Connection,
        route: str,
        *,
        activation_reason: str,
        rollback_of_package_id: str | None = None,
    ) -> str:
        row = db.execute("SELECT * FROM routes WHERE route=?", (route,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown Evalt route {route!r}.")
        snapshot = self._package_snapshot(row)
        if not self._qualified_snapshot(snapshot):
            raise ValueError("Only a qualified route package can be saved or restored.")
        package_id = f"rv_{uuid.uuid4().hex[:20]}"
        created_at = _now()
        db.execute(
            "INSERT INTO route_packages(package_id,route,created_at,activation_reason,"
            "rollback_of_package_id,snapshot_json) VALUES (?,?,?,?,?,?)",
            (
                package_id,
                route,
                created_at,
                activation_reason,
                rollback_of_package_id,
                json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ),
        )
        db.execute(
            "UPDATE routes SET current_package_id=?,updated_at=? WHERE route=?",
            (package_id, created_at, route),
        )
        return package_id

    def _backfill_qualified_packages(self, db: sqlite3.Connection) -> None:
        """Give qualified pre-versioning routes one immutable, idempotent package."""

        for row in db.execute(
            "SELECT * FROM routes WHERE current_package_id='' ORDER BY created_at,route"
        ).fetchall():
            snapshot = self._package_snapshot(row)
            if not self._qualified_snapshot(snapshot):
                continue
            package_id = self._capture_package(
                db, str(row["route"]), activation_reason="legacy_backfill"
            )
            self._event(
                db,
                str(row["route"]),
                "qualified_package_backfilled",
                {"package_id": package_id},
            )

    def _event(self, db: sqlite3.Connection, route: str, event_type: str, detail: Mapping[str, Any]) -> None:
        db.execute(
            "INSERT INTO decisions VALUES (?, ?, ?, ?, ?)",
            (f"evt-{uuid.uuid4().hex}", route, _now(), event_type, json.dumps(detail, sort_keys=True)),
        )

    def _ensure_route(
        self,
        *,
        route: str,
        prompt: str,
        models: Sequence[str],
        quality_threshold: float,
        catalog_revision: str,
        target_max_tokens: int | None = None,
        request_options: Mapping[str, Any] | None = None,
        input_modalities: Sequence[str] = ("text",),
    ) -> sqlite3.Row:
        if not route.strip():
            raise ValueError("route must be a stable non-empty name so Evalt can remember decisions.")
        if not prompt.strip():
            raise ValueError("prompt cannot be empty.")
        candidates = tuple(dict.fromkeys(model.strip() for model in models if model.strip()))
        if not candidates:
            raise ValueError("At least one candidate model is required.")
        version = _hash(prompt.strip())[:16]
        resolved_target_max_tokens = int(target_max_tokens or 600)
        if not 1 <= resolved_target_max_tokens <= 131072:
            raise ValueError("target_max_tokens must be between 1 and 131072.")
        options_were_provided = request_options is not None
        normalized_options = normalize_request_options(request_options)
        options_json = json.dumps(
            normalized_options, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        options_hash = request_options_fingerprint(normalized_options)
        modalities = tuple(dict.fromkeys(str(item) for item in input_modalities))
        modalities_json = json.dumps(modalities, separators=(",", ":"))
        now = _now()
        with self._db() as db:
            current = db.execute("SELECT * FROM routes WHERE route = ?", (route,)).fetchone()
            if current is None:
                db.execute(
                    "INSERT INTO routes(route,prompt,source_prompt_version,prompt_version,candidates_json,selected_model,selected_prompt,decision_reason,quality_threshold,catalog_revision,evidence_provenance,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (route, prompt.strip(), version, version, json.dumps(candidates), candidates[0], prompt.strip(), "bootstrap_unqualified", quality_threshold, catalog_revision, "UNTESTED_BOOTSTRAP", now, now),
                )
                db.execute(
                    "UPDATE routes SET target_max_tokens=?,tested_request_options_json=?,"
                    "tested_request_options_sha256=?,tested_input_modalities_json=? WHERE route=?",
                    (
                        resolved_target_max_tokens, options_json, options_hash,
                        modalities_json, route,
                    ),
                )
                self._event(db, route, "route_created", {"prompt_version": version, "bootstrap_model": candidates[0], "candidates": candidates})
            else:
                changed = current["source_prompt_version"] != version
                candidates_changed = json.loads(current["candidates_json"]) != list(candidates)
                if changed:
                    db.execute(
                        "UPDATE routes SET prompt=?,source_prompt_version=?,prompt_version=?,candidates_json=?,selected_model=?,selected_prompt=?,selected_few_shot_json='[]',evidence_provenance='UNTESTED_BOOTSTRAP',last_test_summary_json='{}',decision_reason='prompt_changed_unqualified',current_package_id='',feedback_count=0,last_optimized_feedback=0,last_optimized_calls=total_calls,tested_catalog_revision='',catalog_revision=?,target_max_tokens=?,tested_request_options_json=?,tested_request_options_sha256=?,tested_input_modalities_json=?,updated_at=? WHERE route=?",
                        (
                            prompt.strip(), version, version, json.dumps(candidates),
                            candidates[0], prompt.strip(), catalog_revision,
                            resolved_target_max_tokens, options_json, options_hash,
                            modalities_json, now, route,
                        ),
                    )
                    self._event(db, route, "prompt_changed", {
                        "prompt_version": version,
                        "bootstrap_model": candidates[0],
                        "evidence_policy": "Prior prompt-version feedback remains in the audit log but cannot qualify the changed prompt.",
                    })
                elif str(current["evidence_provenance"]) in {"LEGACY_UNKNOWN", "UNTESTED_BOOTSTRAP"}:
                    retained_target_max_tokens = (
                        resolved_target_max_tokens
                        if target_max_tokens is not None
                        else int(current["target_max_tokens"] or 600)
                    )
                    retained_options_json = (
                        options_json
                        if options_were_provided
                        else str(current["tested_request_options_json"] or "{}")
                    )
                    retained_options_hash = (
                        options_hash
                        if options_were_provided
                        else str(current["tested_request_options_sha256"])
                    )
                    retained_modalities_json = (
                        modalities_json
                        if input_modalities
                        else str(current["tested_input_modalities_json"] or '["text"]')
                    )
                    db.execute(
                        "UPDATE routes SET candidates_json=?,catalog_revision=?,target_max_tokens=?,"
                        "tested_request_options_json=?,tested_request_options_sha256=?,"
                        "tested_input_modalities_json=?,updated_at=? WHERE route=?",
                        (
                            json.dumps(candidates), catalog_revision, retained_target_max_tokens,
                            retained_options_json, retained_options_hash,
                            retained_modalities_json, now, route,
                        ),
                    )
                elif candidates_changed or current["catalog_revision"] != catalog_revision:
                    db.execute(
                        "UPDATE routes SET candidates_json=?,catalog_revision=?,updated_at=? WHERE route=?",
                        (json.dumps(candidates), catalog_revision, now, route),
                    )
                    self._event(db, route, "catalog_changed", {"catalog_revision": catalog_revision, "candidates": candidates})
            return db.execute("SELECT * FROM routes WHERE route = ?", (route,)).fetchone()

    def needs_initial_optimization(self, route: str, prompt: str) -> bool:
        """Return whether this exact source prompt lacks a tested durable package."""
        version = _hash(prompt.strip())[:16]
        with self._db() as db:
            row = db.execute("SELECT * FROM routes WHERE route=?", (route,)).fetchone()
        if row is None or row["source_prompt_version"] != version:
            return True
        if str(row["evidence_provenance"]) in {"LEGACY_UNKNOWN", "UNTESTED_BOOTSTRAP"}:
            return True
        return str(row["decision_reason"]).startswith(("bootstrap_", "prompt_changed_"))

    @staticmethod
    def _route_version_error(
        route: str,
        expected_package_id: str,
        current_package_id: str | None,
        reason: str,
    ) -> RouteVersionMismatch:
        return RouteVersionMismatch(
            route=route,
            expected_package_id=expected_package_id,
            current_package_id=current_package_id,
            reason=reason,
        )

    def require_version(
        self,
        route: str,
        expected_package_id: str,
        *,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Load one exact current qualified package without a provider call.

        The returned mapping is the immutable package snapshot plus bounded dynamic
        route counters. Production callers use this snapshot directly, so a
        concurrent later activation cannot switch the model or prompt mid-call.
        """

        expected = str(expected_package_id or "").strip()
        if re.fullmatch(r"rv_[0-9a-f]{20}", expected) is None:
            raise self._route_version_error(
                route, expected, None, "invalid route-version format"
            )
        with self._db() as db:
            current = db.execute(
                "SELECT * FROM routes WHERE route=?", (route,)
            ).fetchone()
            current_package_id = (
                str(current["current_package_id"] or "") if current is not None else ""
            )
            package = db.execute(
                "SELECT * FROM route_packages WHERE package_id=?", (expected,)
            ).fetchone()
            if current is None:
                raise self._route_version_error(
                    route, expected, None, "route does not exist"
                )
            if package is not None and str(package["route"]) != route:
                raise self._route_version_error(
                    route, expected, current_package_id or None,
                    f"version belongs to route {str(package['route'])!r}",
                )
            if not current_package_id:
                raise self._route_version_error(
                    route, expected, None,
                    "route has no qualified serving version",
                )
            if current_package_id != expected:
                raise self._route_version_error(
                    route, expected, current_package_id,
                    "serving version changed",
                )
            if package is None:
                raise self._route_version_error(
                    route, expected, current_package_id,
                    "serving package metadata is missing",
                )
            try:
                snapshot = json.loads(str(package["snapshot_json"]))
            except (TypeError, json.JSONDecodeError) as error:
                raise self._route_version_error(
                    route, expected, current_package_id,
                    "stored package metadata is damaged",
                ) from error
            if (
                not isinstance(snapshot, Mapping)
                or any(field not in snapshot for field in _PACKAGE_FIELDS)
                or not self._qualified_snapshot(snapshot)
            ):
                raise self._route_version_error(
                    route, expected, current_package_id,
                    "stored package is incomplete or not qualified",
                )
            if prompt is not None and str(snapshot["source_prompt_version"]) != _hash(
                prompt.strip()
            )[:16]:
                raise self._route_version_error(
                    route, expected, current_package_id,
                    "source prompt does not match the qualified version",
                )
            serving = dict(snapshot)
            for field in (
                "route",
                "total_calls",
                "feedback_count",
                "last_optimized_calls",
                "last_optimized_feedback",
                "catalog_revision",
                "created_at",
                "updated_at",
            ):
                serving[field] = current[field]
            serving["current_package_id"] = current_package_id
            return serving

    def install_initial_result(
        self,
        *,
        route: str,
        prompt: str,
        models: Sequence[str],
        quality_threshold: float,
        catalog_revision: str,
        result: OptimizationResult,
        examples: Sequence[Example],
        evidence_provenance: str,
        total_workflow_spend_usd: float,
        designer_model: str,
        evaluator_model: str,
        judge_calibration_checks: int,
        target_max_tokens: int,
        request_options: Mapping[str, Any] | None,
        objective: str = "lowest_cost_at_accuracy",
        max_p90_latency_seconds: float | None = None,
        latency_value_usd_per_second: float = 0.0,
        optimize_prompt: bool = True,
    ) -> dict[str, Any]:
        """Durably install one split-tested first-route package or fail closed."""
        suite_modalities = tuple(sorted({
            modality
            for example in examples
            for turn in example.conversation()
            for modality in required_input_modalities(turn.input)
        })) or ("text",)
        winner = result.winner
        passed = (
            result.quality_gate_status == "QUALIFIED_ROUTE_SELECTED"
            and not result.exploratory
            and winner.passed_quality_floor
            and winner.passed_difficulty_floors
            and winner.passed_latency_ceiling
            and winner.holdout_pass_rate >= float(quality_threshold)
        )
        if not passed:
            raise ValueError(
                "No configuration cleared the non-exploratory final-test gate; "
                "the route was not promoted."
            )
        self._ensure_route(
            route=route,
            prompt=prompt,
            models=models,
            quality_threshold=quality_threshold,
            catalog_revision=catalog_revision,
            target_max_tokens=target_max_tokens,
            request_options=request_options,
            input_modalities=suite_modalities,
        )
        normalized_options = normalize_request_options(request_options)
        options_json = json.dumps(
            normalized_options, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        options_hash = request_options_fingerprint(normalized_options)
        selected_ids = set(winner.few_shot_example_ids)
        few_shot_messages: list[dict[str, str]] = []
        persisted_few_shot_examples = 0
        for example in examples:
            if example.id not in selected_ids:
                continue
            turns = example.conversation()
            if any(
                "image" in required_input_modalities(turn.input)
                for turn in turns
            ):
                # Route state and version history must never become an image
                # store. The evaluated package remains valid without persisting
                # private visual exemplars as production few-shot messages.
                continue
            persisted_few_shot_examples += 1
            for turn in turns:
                few_shot_messages.extend((
                    {"role": "user", "content": turn.input},
                    {"role": "assistant", "content": turn.approved_output},
                ))
        summary = {
            "quality_gate_status": result.quality_gate_status,
            "winner_model": winner.model,
            "winner_prompt_version": _hash(winner.selected_prompt)[:16],
            "holdout_pass_rate": winner.holdout_pass_rate,
            "final_test_scenarios": winner.holdout_unique_scenarios,
            "final_test_executions": winner.holdout_executions,
            "final_test_evidence_status": winner.final_test_evidence_status,
            "final_test_confidence_level": winner.final_test_confidence_level,
            "final_test_accuracy_lower_bound": winner.final_test_accuracy_lower_bound,
            "target_accuracy_statistically_supported": winner.target_accuracy_statistically_supported,
            "minimum_zero_failure_scenarios": winner.minimum_zero_failure_scenarios,
            "tested_configurations": len(result.models),
            "prompt_candidates_tested": sum(
                item.prompt_candidates_tested for item in result.models
            ),
            "prompt_rewrites_tested": sum(
                item.prompt_rewrites_tested for item in result.models
            ),
            "winner_prompt_changed": winner.selected_prompt_changed,
            "few_shot_examples": persisted_few_shot_examples,
            "workflow_spend_usd": round(float(total_workflow_spend_usd), 10),
            "evidence_provenance": evidence_provenance,
            "designer_model": designer_model,
            "evaluator_model": evaluator_model,
            "judge_calibrated": int(judge_calibration_checks) >= 3,
            "judge_calibration_checks": int(judge_calibration_checks),
            "target_max_tokens": int(target_max_tokens),
            "request_options_sha256": options_hash,
            "input_modalities": list(suite_modalities),
            "suite_hash": result.regression_suite.get("suite_hash"),
            "evaluator_contract_hash": _hash(json.dumps(
                {
                    "evaluator": result.regression_suite.get("evaluator") or {
                        "type": "semantic"
                    },
                    "evaluator_model": evaluator_model,
                    "contract_version": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            )),
            "evaluator_type": str(
                (result.regression_suite.get("evaluator") or {}).get("type")
                or "semantic"
            ),
        }
        with self._db() as db:
            current = db.execute("SELECT * FROM routes WHERE route=?", (route,)).fetchone()
            db.execute(
                "UPDATE routes SET selected_model=?,selected_prompt=?,selected_few_shot_json=?,prompt_version=?,decision_reason='provisional_ai_qualified',evidence_provenance=?,last_test_summary_json=?,last_optimized_calls=?,last_optimized_feedback=?,tested_catalog_revision=?,objective=?,max_p90_latency_seconds=?,latency_value_usd_per_second=?,optimize_prompt=?,target_max_tokens=?,tested_request_options_json=?,tested_request_options_sha256=?,tested_input_modalities_json=?,updated_at=? WHERE route=?",
                (
                    winner.model,
                    winner.selected_prompt,
                    json.dumps(few_shot_messages, ensure_ascii=False, separators=(",", ":")),
                    _hash(winner.selected_prompt)[:16],
                    evidence_provenance,
                    json.dumps(summary, sort_keys=True),
                    current["total_calls"],
                    current["feedback_count"],
                    catalog_revision,
                    objective,
                    max_p90_latency_seconds,
                    float(latency_value_usd_per_second),
                    int(bool(optimize_prompt)),
                    int(target_max_tokens),
                    options_json,
                    options_hash,
                    json.dumps(suite_modalities, separators=(",", ":")),
                    _now(),
                    route,
                ),
            )
            package_id = self._capture_package(
                db, route, activation_reason="initial_qualification"
            )
            summary["package_id"] = package_id
            self._event(db, route, "initial_ai_route_promoted", summary)
        return summary

    def _due(self, row: sqlite3.Row, *, retest_after_calls: int, min_feedback: int) -> tuple[str, ...]:
        reasons: list[str] = []
        if row["feedback_count"] >= min_feedback and row["feedback_count"] > row["last_optimized_feedback"]:
            reasons.append("new_human_feedback")
        if row["total_calls"] - row["last_optimized_calls"] >= retest_after_calls:
            reasons.append("traffic_threshold")
        if row["catalog_revision"] != row["tested_catalog_revision"]:
            reasons.append("model_or_price_catalog_changed")
        return tuple(reasons)

    def run(
        self,
        *,
        route: str,
        prompt: str,
        input: Any,
        route_version: str | None = None,
        max_cost_per_run_usd: float | None,
        models: Sequence[str] = DEFAULT_TARGETS,
        max_tokens: int | None = None,
        request_options: Mapping[str, Any] | None = None,
        strict_request_options: bool = False,
        target_accuracy: float = 0.95,
        objective: str = "best_within_price",
        optimize_prompt: bool = True,
        test_budget_usd: float = 0.0,
        test_budget_policy: str = "explicit",
        max_p90_latency_seconds: float | None = None,
        latency_value_usd_per_second: float = 0.0,
        retest_after_calls: int = 500,
        min_feedback: int = 5,
        catalog_revision: str = "explicit-model-list",
    ) -> RoutedAnswer:
        if max_cost_per_run_usd is not None and not 0 < float(max_cost_per_run_usd) <= 10:
            raise ValueError("price_usd must be greater than 0 and no more than 10.")
        if objective not in {"match_baseline_at_lowest_cost", "best_within_price", "best_within_cost", "lowest_cost_at_accuracy", "cheapest_at_accuracy", "cheapest_passing", "constrained", "highest_quality"}:
            raise ValueError("objective is not a supported cost/accuracy policy.")
        if max_p90_latency_seconds is not None and max_p90_latency_seconds <= 0:
            raise ValueError("max_p90_latency_seconds must be positive when provided.")
        if latency_value_usd_per_second < 0:
            raise ValueError("latency_value_usd_per_second cannot be negative.")
        input_messages = _input_messages(input)
        active_modalities = tuple(sorted(required_input_modalities(input_messages)))
        row: Mapping[str, Any]
        if route_version is not None:
            row = self.require_version(route, route_version, prompt=prompt)
        else:
            row = self._ensure_route(
                route=route,
                prompt=prompt,
                models=models,
                quality_threshold=target_accuracy,
                catalog_revision=catalog_revision,
                target_max_tokens=max_tokens,
                request_options=request_options,
                input_modalities=active_modalities,
            )
        set_performance_policy = getattr(self.client.transport, "set_performance_policy", None)
        if callable(set_performance_policy):
            set_performance_policy(
                preferred_max_latency_seconds=max_p90_latency_seconds,
                provider_sort=("latency" if latency_value_usd_per_second > 0 else "price"),
            )
        input_text = _content(input_messages)
        tested_options = normalize_request_options(
            json.loads(row["tested_request_options_json"] or "{}")
        )
        active_options = (
            tested_options
            if request_options is None
            else normalize_request_options(request_options)
        )
        tested_options_hash = str(
            row["tested_request_options_sha256"]
            or request_options_fingerprint(tested_options)
        )
        active_options_hash = request_options_fingerprint(active_options)
        tested_modalities = tuple(
            json.loads(row["tested_input_modalities_json"] or '["text"]')
        )
        tested_max_tokens = int(row["target_max_tokens"] or 600)
        active_max_tokens = (
            tested_max_tokens if max_tokens is None else int(max_tokens)
        )
        if not 1 <= active_max_tokens <= 131072:
            raise ValueError("max_tokens must be between 1 and 131072.")
        envelope_drift = (
            active_options_hash != tested_options_hash
            or active_max_tokens != tested_max_tokens
            or set(active_modalities) != set(tested_modalities)
        )
        drift_message = (
            "This Evalt call changes the request settings or input modalities used to "
            "qualify the route; the saved accuracy result does not validate this response."
        )
        if envelope_drift and strict_request_options:
            raise ValueError(drift_message + " Remove the override or set strict_request_options=False.")
        if envelope_drift:
            runtime_warnings.warn(
                drift_message,
                RequestEnvelopeDriftWarning,
                stacklevel=3,
            )
            with self._db() as db:
                self._event(db, route, "request_envelope_drift", {
                    "tested_request_options_sha256": tested_options_hash,
                    "request_options_sha256": active_options_hash,
                    "tested_max_tokens": tested_max_tokens,
                    "requested_max_tokens": active_max_tokens,
                    "tested_input_modalities": list(tested_modalities),
                    "input_modalities": list(active_modalities),
                    "quality_claim_applies": False,
                })
        few_shot_messages = json.loads(row["selected_few_shot_json"] or "[]")
        messages = [
            {"role": "system", "content": row["selected_prompt"]},
            *few_shot_messages,
            *input_messages,
        ]
        estimate = self.client.transport.estimate_cost(row["selected_model"], messages, max_tokens=active_max_tokens)
        price_policy = "explicit" if max_cost_per_run_usd is not None else "automatic"
        effective_price_ceiling_usd = (
            float(max_cost_per_run_usd)
            if max_cost_per_run_usd is not None
            else min(10.0, max(0.01, float(estimate) * 1.10))
        )
        with self._db() as db:
            current = db.execute("SELECT * FROM routes WHERE route=?", (route,)).fetchone()
            if route_version is not None and (
                current is None
                or str(current["current_package_id"] or "") != route_version
            ):
                current_package_id = (
                    str(current["current_package_id"] or "")
                    if current is not None
                    else ""
                )
                raise self._route_version_error(
                    route,
                    route_version,
                    current_package_id or None,
                    "serving version changed before the provider call",
                )
            controls = (
                effective_price_ceiling_usd, price_policy, float(test_budget_usd),
                objective, test_budget_policy, max_p90_latency_seconds,
                float(latency_value_usd_per_second), int(bool(optimize_prompt)),
            )
            previous = (
                current["max_cost_per_run_usd"], current["price_policy"],
                current["test_budget_usd"], current["objective"],
                current["test_budget_policy"], current["max_p90_latency_seconds"],
                current["latency_value_usd_per_second"], current["optimize_prompt"],
            )
            if route_version is None:
                db.execute(
                    "UPDATE routes SET quality_threshold=?,max_cost_per_run_usd=?,price_policy=?,test_budget_usd=?,objective=?,test_budget_policy=?,max_p90_latency_seconds=?,latency_value_usd_per_second=?,optimize_prompt=?,updated_at=? WHERE route=?",
                    (float(target_accuracy), *controls, _now(), route),
                )
            else:
                db.execute(
                    "UPDATE routes SET max_cost_per_run_usd=?,price_policy=?,"
                    "test_budget_usd=?,test_budget_policy=?,updated_at=? WHERE route=?",
                    (
                        effective_price_ceiling_usd,
                        price_policy,
                        float(test_budget_usd),
                        test_budget_policy,
                        _now(),
                        route,
                    ),
                )
            if route_version is None and previous != controls:
                self._event(db, route, "routing_policy_configured", {
                    "price_usd": max_cost_per_run_usd,
                    "effective_price_ceiling_usd": effective_price_ceiling_usd,
                    "price_policy": price_policy,
                    "test_budget_usd": test_budget_usd,
                    "test_budget_policy": test_budget_policy,
                    "target_accuracy": target_accuracy,
                    "objective": objective,
                    "max_p90_latency_seconds": max_p90_latency_seconds,
                    "latency_value_usd_per_second": latency_value_usd_per_second,
                    "optimize_prompt": bool(optimize_prompt),
                })
            if route_version is None:
                row = db.execute(
                    "SELECT * FROM routes WHERE route=?", (route,)
                ).fetchone()
        if estimate > effective_price_ceiling_usd + 1e-12:
            raise BudgetExceeded(
                f"The selected route estimates ${estimate:.6f}, above this call's ${effective_price_ceiling_usd:.6f} price ceiling."
            )
        completion_kwargs: dict[str, Any] = {"max_tokens": active_max_tokens}
        if active_options:
            completion_kwargs["request_options"] = active_options
        try:
            completion: Completion = self.client.transport.complete(
                row["selected_model"], messages, **completion_kwargs
            )
        except TypeError as error:
            if active_options and "request_options" in str(error):
                raise ValueError(
                    "The custom transport does not support request_options."
                ) from error
            raise
        if completion.cost_usd > effective_price_ceiling_usd + 1e-12:
            raise BudgetExceeded("The provider-reported cost exceeded this call's hard cap.")
        call_id = f"call-{uuid.uuid4().hex}"
        with self._db() as db:
            db.execute(
                "INSERT INTO calls(call_id,route,created_at,input_text,output_text,model,prompt_version,provider_cost_usd,budget_usd,decision_reason,request_options_sha256,request_envelope_validated,input_replayable,latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    call_id, route, _now(), input_text, completion.content,
                    completion.model, row["prompt_version"], completion.cost_usd,
                    effective_price_ceiling_usd, row["decision_reason"],
                    active_options_hash, int(not envelope_drift),
                    int("image" not in active_modalities),
                    max(0, int(completion.latency_ms or 0)),
                ),
            )
            db.execute("UPDATE routes SET total_calls=total_calls+1,updated_at=? WHERE route=?", (_now(), route))
            current = db.execute("SELECT * FROM routes WHERE route=?", (route,)).fetchone()
            due = self._due(current, retest_after_calls=retest_after_calls, min_feedback=min_feedback)
        return RoutedAnswer(
            content=completion.content,
            model=completion.model,
            route=route,
            call_id=call_id,
            provider_cost_usd=completion.cost_usd,
            prompt_version=row["prompt_version"],
            decision_reason=row["decision_reason"],
            maintenance_due=due,
            route_version=str(route_version or row["current_package_id"] or ""),
            route_phase=(
                _route_phase(row)
            ),
            evidence_provenance=row["evidence_provenance"],
            initial_test_summary=(json.loads(row["last_test_summary_json"] or "{}") or None),
            request_envelope_validated=not envelope_drift,
            request_options_sha256=active_options_hash,
            tested_request_options_sha256=tested_options_hash,
            warnings=((drift_message,) if envelope_drift else ()),
            finish_reason=completion.finish_reason,
            native_finish_reason=completion.native_finish_reason,
            message=dict(completion.message),
            tool_calls=tuple(completion.tool_calls),
            _router=self,
            _min_feedback=min_feedback,
            _retest_after_calls=retest_after_calls,
        )

    def record_feedback(
        self,
        call_id: str,
        *,
        approved_output: str,
        verdict: str,
        min_feedback: int = 5,
        retest_after_calls: int = 500,
    ) -> dict[str, Any]:
        if verdict not in {"accepted", "corrected"}:
            raise ValueError("verdict must be accepted or corrected.")
        with self._db() as db:
            call = db.execute("SELECT * FROM calls WHERE call_id=?", (call_id,)).fetchone()
            if call is None:
                raise KeyError(f"Unknown Evalt call {call_id!r}.")
            existed = db.execute("SELECT 1 FROM feedback WHERE call_id=?", (call_id,)).fetchone()
            db.execute(
                "INSERT OR REPLACE INTO feedback(call_id,route,verdict,approved_output,created_at) VALUES (?,?,?,?,?)",
                (call_id, call["route"], verdict, approved_output, _now()),
            )
            replayable = bool(call["input_replayable"])
            if existed is None and replayable:
                db.execute("UPDATE routes SET feedback_count=feedback_count+1,updated_at=? WHERE route=?", (_now(), call["route"]))
            self._event(db, call["route"], "feedback_recorded", {
                "call_id": call_id,
                "verdict": verdict,
                "replayable_for_maintenance": replayable,
            })
            row = db.execute("SELECT * FROM routes WHERE route=?", (call["route"],)).fetchone()
            return {
                "event": "feedback_recorded",
                "route": call["route"],
                "verdict": verdict,
                "is_new": existed is None,
                "replayable_for_maintenance": replayable,
                "feedback_count": int(row["feedback_count"]),
                "min_feedback": int(min_feedback),
                "maintenance_due": list(
                    self._due(
                        row,
                        retest_after_calls=retest_after_calls,
                        min_feedback=min_feedback,
                    )
                ),
            }

    def maintain(
        self,
        route: str,
        *,
        test_budget_usd: float,
        role_plan: RolePlan,
        objective: str = "best_within_price",
        max_cost_per_run_usd: float | None = None,
        rounds: int = 3,
        min_feedback: int = 5,
    ) -> OptimizationResult | None:
        if test_budget_usd <= 0:
            return None
        if not self._maintenance_lock.acquire(blocking=False):
            return None
        try:
            with self._db() as db:
                row = db.execute("SELECT * FROM routes WHERE route=?", (route,)).fetchone()
                if row is None:
                    raise KeyError(f"Unknown Evalt route {route!r}.")
                feedback = db.execute(
                    "SELECT calls.input_text,calls.output_text,feedback.approved_output,feedback.verdict FROM feedback JOIN calls USING(call_id) WHERE feedback.route=? AND calls.prompt_version=? AND calls.input_replayable=1 ORDER BY feedback.created_at",
                    (route, row["source_prompt_version"]),
                ).fetchall()
            if len(feedback) < min_feedback:
                return None
            examples = [Example.from_value({"id": f"feedback-{index+1}", "input": item["input_text"], "approved_output": item["approved_output"]}, index) for index, item in enumerate(feedback)]
            calibration: list[tuple[Example, str, bool]] = []
            for example, item in zip(examples, feedback):
                calibration.append((example, item["approved_output"], True))
                if item["verdict"] == "corrected" and item["output_text"].strip() != item["approved_output"].strip():
                    calibration.append((example, item["output_text"], False))
            positives = sum(1 for _example, _output, expected in calibration if expected)
            negatives = sum(1 for _example, _output, expected in calibration if not expected)
            if positives < 2 or negatives < 1:
                with self._db() as db:
                    self._event(db, route, "judge_calibration_waiting", {"known_passes": positives, "known_failures": negatives, "required": "2 passes and 1 human-corrected failure"})
                return None

            maintenance_budget = _Budget(test_budget_usd)
            calibrated_judge: str | None = None
            calibration_checks = calibration[:4]
            for judge in dict.fromkeys((role_plan.judge_model, role_plan.test_designer_model)):
                matched = True
                for example, candidate_output, expected in calibration_checks:
                    turn = example.conversation()[0]
                    judgment, _completion = self.client._judge(example, turn, 0, [], candidate_output, judge, maintenance_budget)
                    if judgment.passed is not expected:
                        matched = False
                if matched:
                    calibrated_judge = judge
                    break
            if calibrated_judge is None:
                with self._db() as db:
                    current = db.execute("SELECT * FROM routes WHERE route=?", (route,)).fetchone()
                    db.execute(
                        "UPDATE routes SET last_optimized_calls=?,last_optimized_feedback=?,tested_catalog_revision=?,updated_at=? WHERE route=?",
                        (current["total_calls"], current["feedback_count"], role_plan.catalog_revision, _now(), route),
                    )
                    self._event(db, route, "judge_calibration_failed", {"models": list(dict.fromkeys((role_plan.judge_model, role_plan.test_designer_model))), "checks": len(calibration_checks), "provider_spend_usd": maintenance_budget.spent_usd})
                return None
            role_plan = replace(role_plan, judge_model=calibrated_judge)
            remaining_budget = test_budget_usd - maintenance_budget.spent_usd
            if remaining_budget <= 0:
                raise BudgetExceeded("Judge calibration consumed the maintenance cap before the target tournament.")
            input_lengths = [len(item["input_text"]) for item in feedback]
            output_lengths = [len(item["approved_output"]) for item in feedback]
            representative_input_chars = _percentile(input_lengths, 0.90)
            representative_output_tokens = max(32, int(_percentile(output_lengths, 0.90) / 3) + 1)
            result = self.client.optimize(
                prompt=row["prompt"],
                examples=examples,
                models=tuple(dict.fromkeys((row["selected_model"], *role_plan.target_models))),
                optimizer_model=role_plan.test_designer_model,
                evaluator_model=role_plan.judge_model,
                objective=objective,
                quality_threshold=float(row["quality_threshold"]),
                max_optimization_cost_usd=remaining_budget,
                rounds=rounds,
                optimize_prompt=bool(row["optimize_prompt"]),
                max_cost_per_run_usd=max_cost_per_run_usd,
                representative_input_chars=representative_input_chars,
                representative_output_tokens=representative_output_tokens,
                incumbent_model=row["selected_model"],
                adaptive_search=True,
                max_p90_latency_seconds=row["max_p90_latency_seconds"],
                latency_value_usd_per_second=row["latency_value_usd_per_second"],
                target_max_tokens=int(row["target_max_tokens"] or 600),
                request_options=json.loads(row["tested_request_options_json"] or "{}"),
            )
            winner = result.winner
            within_price = max_cost_per_run_usd is None or winner.estimated_production_cost_per_call_usd <= max_cost_per_run_usd + 1e-12
            within_latency = winner.passed_latency_ceiling
            if objective == "match_baseline_at_lowest_cost":
                baseline_quality = float(result.regression_suite["incumbent_baseline_holdout_pass_rate"])
                accuracy_met = winner.holdout_pass_rate >= baseline_quality
            else:
                accuracy_met = winner.holdout_pass_rate >= float(row["quality_threshold"])
            promoted = within_price and within_latency and (
                objective in {"best_within_price", "best_within_cost", "highest_quality"}
                or accuracy_met
            )
            with self._db() as db:
                current = db.execute("SELECT * FROM routes WHERE route=?", (route,)).fetchone()
                detail = {
                    "quality_gate_status": (
                        "QUALIFIED_ROUTE_SELECTED" if promoted else "NO_QUALIFIED_ROUTE"
                    ),
                    "role_plan": role_plan.to_dict(),
                    "judge_calibration_spend_usd": maintenance_budget.spent_usd,
                    "optimization_spend_usd": result.total_provider_spend_usd,
                    "total_maintenance_spend_usd": maintenance_budget.spent_usd + result.total_provider_spend_usd,
                    "winner_model": winner.model,
                    "winner_prompt_version": _hash(winner.selected_prompt)[:16],
                    "few_shot_examples": len(winner.few_shot_example_ids),
                    "holdout_pass_rate": winner.holdout_pass_rate,
                    "cost_per_success_usd": winner.estimated_cost_per_successful_call_usd,
                    "promoted": promoted,
                    "target_accuracy_met": accuracy_met,
                    "production_price_ceiling_met": within_price,
                    "p90_latency_seconds": winner.target_latency_p90_ms / 1000,
                    "latency_ceiling_met": within_latency,
                    "max_p90_latency_seconds": row["max_p90_latency_seconds"],
                    "latency_value_usd_per_second": row["latency_value_usd_per_second"],
                    "representative_input_chars_p90": representative_input_chars,
                    "representative_output_tokens_p90": representative_output_tokens,
                    "suite_hash": result.regression_suite.get("suite_hash"),
                    "evaluator_contract_hash": _hash(json.dumps(
                        {
                            "evaluator": result.regression_suite.get("evaluator") or {
                                "type": "semantic"
                            },
                            "evaluator_model": role_plan.judge_model,
                            "contract_version": 1,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )),
                    "evaluator_type": str(
                        (result.regression_suite.get("evaluator") or {}).get("type")
                        or "semantic"
                    ),
                }
                if promoted:
                    selected_ids = set(winner.few_shot_example_ids)
                    few_shot_messages: list[dict[str, str]] = []
                    for example in examples:
                        if example.id not in selected_ids:
                            continue
                        for turn in example.conversation():
                            few_shot_messages.extend((
                                {"role": "user", "content": turn.input},
                                {"role": "assistant", "content": turn.approved_output},
                            ))
                    db.execute(
                        "UPDATE routes SET selected_model=?,selected_prompt=?,selected_few_shot_json=?,prompt_version=?,decision_reason=?,evidence_provenance='HUMAN_FEEDBACK_CALIBRATED',last_test_summary_json=?,last_optimized_calls=?,last_optimized_feedback=?,tested_catalog_revision=?,updated_at=? WHERE route=?",
                        (
                            winner.model,
                            winner.selected_prompt,
                            json.dumps(few_shot_messages, ensure_ascii=False, separators=(",", ":")),
                            _hash(winner.selected_prompt)[:16],
                            f"qualified_{objective}",
                            json.dumps(detail, sort_keys=True),
                            current["total_calls"],
                            current["feedback_count"],
                            role_plan.catalog_revision,
                            _now(),
                            route,
                        ),
                    )
                    package_id = self._capture_package(
                        db, route, activation_reason="maintenance_promotion"
                    )
                    detail["package_id"] = package_id
                    self._event(db, route, "route_promoted", detail)
                else:
                    db.execute(
                        "UPDATE routes SET last_optimized_calls=?,last_optimized_feedback=?,tested_catalog_revision=?,updated_at=? WHERE route=?",
                        (current["total_calls"], current["feedback_count"], role_plan.catalog_revision, _now(), route),
                    )
                    self._event(db, route, "maintenance_no_promotion", detail)
            return result
        finally:
            self._maintenance_lock.release()

    def status(self, route: str, *, retest_after_calls: int = 500, min_feedback: int = 5) -> dict[str, Any]:
        with self._db() as db:
            row = db.execute("SELECT * FROM routes WHERE route=?", (route,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown Evalt route {route!r}.")
            decisions = [
                {"created_at": item["created_at"], "event_type": item["event_type"], "detail": json.loads(item["detail_json"])}
                for item in db.execute("SELECT * FROM decisions WHERE route=? ORDER BY created_at", (route,))
            ]
            call_metrics = list(db.execute(
                "SELECT provider_cost_usd,latency_ms FROM calls WHERE route=? ORDER BY created_at",
                (route,),
            ))
            packages = self._list_versions(db, route)
        last_test_summary = json.loads(row["last_test_summary_json"] or "{}")
        production_latencies = [int(item["latency_ms"] or 0) for item in call_metrics]
        candidate_models = list(last_test_summary.get("candidate_models") or [])
        if not candidate_models:
            candidate_models = list(json.loads(row["candidates_json"] or "[]"))
        return {
            "schema": "evalt-route-audit-v1",
            "route": route,
            "selected_model": row["selected_model"],
            "selected_prompt_version": row["prompt_version"],
            "decision_reason": row["decision_reason"],
            "route_phase": _route_phase(row),
            "evidence_provenance": row["evidence_provenance"],
            "selected_few_shot_messages": len(json.loads(row["selected_few_shot_json"] or "[]")),
            "selected_few_shot_examples": int(
                json.loads(row["last_test_summary_json"] or "{}").get(
                    "few_shot_examples", 0
                )
            ),
            "last_test_summary": last_test_summary,
            "price_usd": (
                row["max_cost_per_run_usd"]
                if row["price_policy"] == "explicit" else None
            ),
            "price_policy": row["price_policy"],
            "effective_price_ceiling_usd": row["max_cost_per_run_usd"],
            "test_budget_usd": row["test_budget_usd"],
            "test_budget_policy": row["test_budget_policy"],
            "target_accuracy": row["quality_threshold"],
            "objective": row["objective"],
            "max_p90_latency_seconds": row["max_p90_latency_seconds"],
            "latency_value_usd_per_second": row["latency_value_usd_per_second"],
            "optimize_prompt": bool(row["optimize_prompt"]),
            "target_max_tokens": int(row["target_max_tokens"] or 600),
            "tested_request_options": json.loads(row["tested_request_options_json"] or "{}"),
            "tested_request_options_sha256": row["tested_request_options_sha256"],
            "tested_input_modalities": json.loads(
                row["tested_input_modalities_json"] or '["text"]'
            ),
            "input_modalities": json.loads(
                row["tested_input_modalities_json"] or '["text"]'
            ),
            "case_count": int(
                last_test_summary.get("case_count")
                or last_test_summary.get("final_test_scenarios")
                or 0
            ),
            "candidate_models": candidate_models,
            "total_calls": row["total_calls"],
            "production_cost_total_usd": round(sum(
                float(item["provider_cost_usd"] or 0) for item in call_metrics
            ), 10),
            "production_latency_p50_ms": _percentile(production_latencies, 0.50),
            "production_latency_p90_ms": _percentile(production_latencies, 0.90),
            "feedback_count": row["feedback_count"],
            "feedback_needed_for_first_test": max(
                0, int(min_feedback) - int(row["feedback_count"])
            ),
            "maintenance_due": list(self._due(row, retest_after_calls=retest_after_calls, min_feedback=min_feedback)),
            "catalog_revision": row["catalog_revision"],
            "tested_catalog_revision": row["tested_catalog_revision"],
            "current_package_id": row["current_package_id"],
            "qualified_package_count": len(packages),
            "recent_route_versions": packages[:5],
            "decisions": decisions,
        }

    @staticmethod
    def _version_summary(
        package: Mapping[str, Any], current_package_id: str
    ) -> dict[str, Any]:
        base = {
            "package_id": str(package["package_id"]),
            "alias": str(package["alias"] or ""),
            "note": str(package["note"] or ""),
            "annotation_updated_at": (
                str(package["annotation_updated_at"])
                if package["annotation_updated_at"] else None
            ),
            "activated_at": str(package["created_at"]),
            "activation_reason": str(package["activation_reason"]),
            "rollback_of_package_id": (
                str(package["rollback_of_package_id"])
                if package["rollback_of_package_id"] else None
            ),
            "current": str(package["package_id"]) == current_package_id,
        }
        try:
            snapshot = json.loads(str(package["snapshot_json"]))
        except (TypeError, json.JSONDecodeError):
            return {
                **base,
                "restorable": False,
                "problem": "Stored package metadata is damaged.",
            }
        if (
            not isinstance(snapshot, Mapping)
            or any(field not in snapshot for field in _PACKAGE_FIELDS)
            or not DurableRouter._qualified_snapshot(snapshot)
        ):
            return {
                **base,
                "restorable": False,
                "problem": "Stored package is incomplete or not qualified.",
            }
        summary = json.loads(str(snapshot.get("last_test_summary_json") or "{}"))
        return {
            **base,
            "restorable": True,
            "selected_model": str(snapshot.get("selected_model") or ""),
            "prompt_version": str(snapshot.get("prompt_version") or ""),
            "evidence_provenance": str(snapshot.get("evidence_provenance") or ""),
            "holdout_pass_rate": summary.get("holdout_pass_rate"),
            "final_test_accuracy_lower_bound": summary.get(
                "final_test_accuracy_lower_bound"
            ),
            "workflow_spend_usd": summary.get("workflow_spend_usd"),
            "target_latency_p90_ms": summary.get("target_latency_p90_ms"),
        }

    def _list_versions(
        self, db: sqlite3.Connection, route: str
    ) -> list[dict[str, Any]]:
        current = db.execute(
            "SELECT current_package_id FROM routes WHERE route=?", (route,)
        ).fetchone()
        if current is None:
            raise KeyError(f"Unknown Evalt route {route!r}.")
        return [
            self._version_summary(package, str(current["current_package_id"] or ""))
            for package in db.execute(
                "SELECT * FROM route_packages WHERE route=? "
                "ORDER BY created_at DESC,package_id DESC",
                (route,),
            )
        ]

    def list_versions(self, route: str) -> list[dict[str, Any]]:
        """List bounded metadata for immutable qualified packages; never provider calls."""

        with self._db() as db:
            return self._list_versions(db, route)

    @staticmethod
    def _validated_package_alias(alias: str) -> str:
        value = str(alias).strip()
        if not _PACKAGE_ALIAS_RE.fullmatch(value):
            raise ValueError(
                "alias must be 1-48 characters using lowercase letters, numbers, "
                "'.', '_', or '-', and must start with a letter or number."
            )
        return value

    @staticmethod
    def _validated_package_note(note: str) -> str:
        value = str(note).strip()
        if len(value) > _PACKAGE_NOTE_MAX_LENGTH:
            raise ValueError(
                f"note must be {_PACKAGE_NOTE_MAX_LENGTH} characters or fewer."
            )
        if any(ord(character) < 32 for character in value):
            raise ValueError("note cannot contain control characters.")
        return value

    def annotate_version(
        self,
        route: str,
        package_id: str,
        *,
        alias: str | None = None,
        note: str | None = None,
        expected_alias: str | None = None,
    ) -> dict[str, Any]:
        """Set or clear private local annotations without changing a package."""

        route_name = str(route).strip()
        version_id = str(package_id).strip()
        if not route_name:
            raise ValueError("route must be a stable non-empty name.")
        if not _PACKAGE_ID_RE.fullmatch(version_id):
            raise ValueError("package_id must be an exact Evalt version such as rv_….")
        if alias is None and note is None:
            raise ValueError("Provide alias or note; use an empty value to clear it.")
        new_alias = None if alias is None else (
            "" if not str(alias).strip() else self._validated_package_alias(alias)
        )
        new_note = None if note is None else self._validated_package_note(note)
        expected = None
        if expected_alias is not None:
            expected = (
                ""
                if not str(expected_alias).strip()
                else self._validated_package_alias(expected_alias)
            )

        with self._db() as db:
            current = db.execute(
                "SELECT current_package_id FROM routes WHERE route=?", (route_name,)
            ).fetchone()
            if current is None:
                raise KeyError(f"Unknown Evalt route {route_name!r}.")
            package = db.execute(
                "SELECT * FROM route_packages WHERE package_id=? AND route=?",
                (version_id, route_name),
            ).fetchone()
            if package is None:
                raise KeyError(
                    f"Route {route_name!r} has no qualified package {version_id!r}."
                )
            summary = self._version_summary(
                package, str(current["current_package_id"] or "")
            )
            if not summary.get("restorable"):
                raise ValueError(
                    "The stored route package is damaged or not qualified; "
                    "annotations were not changed."
                )
            old_alias = str(package["alias"] or "")
            old_note = str(package["note"] or "")
            if expected is not None and expected != old_alias:
                raise RuntimeError(
                    "The version alias changed concurrently; reload the version list "
                    "before retrying."
                )
            resolved_alias = old_alias if new_alias is None else new_alias
            resolved_note = old_note if new_note is None else new_note
            if resolved_alias == old_alias and resolved_note == old_note:
                return summary
            collision = db.execute(
                "SELECT package_id FROM route_packages "
                "WHERE route=? AND alias=? AND package_id<>?",
                (route_name, resolved_alias, version_id),
            ).fetchone() if resolved_alias else None
            if collision is not None:
                raise ValueError(
                    f"Alias {resolved_alias!r} already names another version of "
                    f"route {route_name!r}."
                )
            updated_at = _now()
            try:
                changed = db.execute(
                    "UPDATE route_packages SET alias=?,note=?,annotation_updated_at=? "
                    "WHERE package_id=? AND route=? AND alias=? AND note=?",
                    (
                        resolved_alias,
                        resolved_note,
                        updated_at,
                        version_id,
                        route_name,
                        old_alias,
                        old_note,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"Alias {resolved_alias!r} already names another version of "
                    f"route {route_name!r}."
                ) from exc
            if changed.rowcount != 1:
                raise RuntimeError(
                    "The version annotation changed concurrently; reload the version "
                    "list before retrying."
                )
            self._event(
                db,
                route_name,
                "qualified_package_annotation_changed",
                {
                    "package_id": version_id,
                    "alias_changed": resolved_alias != old_alias,
                    "note_changed": resolved_note != old_note,
                    "provider_call_started": False,
                },
            )
            refreshed = db.execute(
                "SELECT * FROM route_packages WHERE package_id=? AND route=?",
                (version_id, route_name),
            ).fetchone()
            return self._version_summary(
                refreshed, str(current["current_package_id"] or "")
            )

    def _resolve_version_reference(
        self, db: sqlite3.Connection, route: str, reference: str
    ) -> sqlite3.Row:
        value = str(reference).strip()
        if _PACKAGE_ID_RE.fullmatch(value):
            package = db.execute(
                "SELECT * FROM route_packages WHERE package_id=? AND route=?",
                (value, route),
            ).fetchone()
            if package is None:
                raise KeyError(
                    f"Route {route!r} has no qualified package {value!r}."
                )
            return package
        alias = self._validated_package_alias(value)
        matches = db.execute(
            "SELECT * FROM route_packages WHERE route=? AND alias=? "
            "ORDER BY created_at DESC,package_id DESC",
            (route, alias),
        ).fetchall()
        if not matches:
            raise KeyError(
                f"Route {route!r} has no qualified package or alias {alias!r}."
            )
        if len(matches) != 1:
            raise ValueError(
                f"Alias {alias!r} is ambiguous for route {route!r}; use an exact rv_ ID."
            )
        return matches[0]

    def rollback(self, route: str, package_id: str) -> dict[str, Any]:
        """Atomically restore one qualified local package without calling a provider."""

        with self._db() as db:
            current = db.execute(
                "SELECT * FROM routes WHERE route=?", (route,)
            ).fetchone()
            if current is None:
                raise KeyError(f"Unknown Evalt route {route!r}.")
            package = self._resolve_version_reference(db, route, package_id)
            resolved_package_id = str(package["package_id"])
            if str(current["current_package_id"] or "") == resolved_package_id:
                raise ValueError("That qualified package is already serving.")
            try:
                snapshot = json.loads(str(package["snapshot_json"]))
            except json.JSONDecodeError as error:
                raise ValueError("The stored route package is damaged; no changes were made.") from error
            if not isinstance(snapshot, Mapping) or not self._qualified_snapshot(snapshot):
                raise ValueError("The stored route package is not qualified; no changes were made.")
            missing = [field for field in _PACKAGE_FIELDS if field not in snapshot]
            if missing:
                raise ValueError("The stored route package is incomplete; no changes were made.")
            assignments = ",".join(f"{field}=?" for field in _PACKAGE_FIELDS)
            db.execute(
                f"UPDATE routes SET {assignments},updated_at=? WHERE route=?",
                (*[snapshot[field] for field in _PACKAGE_FIELDS], _now(), route),
            )
            new_package_id = self._capture_package(
                db,
                route,
                activation_reason="rollback",
                rollback_of_package_id=resolved_package_id,
            )
            detail = {
                "package_id": new_package_id,
                "restored_package_id": resolved_package_id,
                "previous_package_id": str(current["current_package_id"] or ""),
                "selected_model": str(snapshot["selected_model"]),
                "prompt_version": str(snapshot["prompt_version"]),
                "provider_call_started": False,
            }
            self._event(db, route, "route_rolled_back", detail)
            return detail

    def list_routes(self) -> tuple[str, ...]:
        """Return durable route names without reading prompts, calls, or outputs."""

        with self._db() as db:
            return tuple(
                str(item["route"])
                for item in db.execute(
                    "SELECT route FROM routes ORDER BY updated_at DESC, route ASC"
                )
            )

    def export_audit(self, route: str, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.status(route), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = [
    "DEFAULT_TARGETS", "DurableRouter", "RequestEnvelopeDriftWarning",
    "RouteVersionMismatch", "RolePlan", "RoutedAnswer", "select_role_plan",
]
