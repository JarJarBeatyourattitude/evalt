"""Durable, budget-bounded runtime routing for Evalt.

The router keeps serving a qualified prompt/model pair while collecting explicit
feedback.  When enough traffic and approved examples exist, a bounded maintenance
run may test current models and promote a cheaper passing pair.  SQLite is the
runtime source of truth; JSON is only an optional audit export.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import uuid
from typing import Any, Iterable, Mapping, Sequence

from last_good_prompt.core import BudgetExceeded, Client, Completion, Example, OptimizationResult, _Budget


DEFAULT_TARGETS = (
    "openai/gpt-5-mini",
    "google/gemini-3-flash-preview",
    "qwen/qwen3.5-9b",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 1
    ordered = sorted(max(0, int(value)) for value in values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


@dataclass(frozen=True)
class RolePlan:
    """The three separately costed model roles used by a maintenance run."""

    tier: str
    test_designer_model: str
    judge_model: str
    target_models: tuple[str, ...]
    catalog_revision: str
    policy: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["target_models"] = list(self.target_models)
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
            intelligence = float(item["intelligence"])
            price = float(item["blended_price"])
            model = str(item["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if model and intelligence >= 0 and price >= 0:
            reasoning = item.get("reasoning") or {}
            reasoning_supported = bool({"reasoning", "reasoning_effort"} & supported)
            efforts = tuple(str(value) for value in reasoning.get("supported_efforts") or ()) if reasoning_supported else ()
            if not efforts and reasoning_supported:
                efforts = ("low", "medium", "high")
            if not reasoning.get("mandatory"):
                efforts = ("none", *efforts)
            try:
                private_provider_routes = max(
                    0, int(item.get("private_provider_routes") or 0)
                )
            except (TypeError, ValueError):
                private_provider_routes = 0
            normalized.append({
                "id": model,
                "intelligence": intelligence,
                "price": price,
                "private_provider_routes": private_provider_routes,
                "reasoning_efforts": tuple(dict.fromkeys(efforts)),
            })

    if maintenance_budget_usd < 0.50:
        tier, designer_delta, judge_delta, breadth = "lean", 16.0, 26.0, 3
    elif maintenance_budget_usd < 2.00:
        tier, designer_delta, judge_delta, breadth = "standard", 9.0, 18.0, 6
    else:
        tier, designer_delta, judge_delta, breadth = "deep", 4.0, 11.0, 12

    if not normalized:
        targets = tuple(dict.fromkeys(fallback_targets))[:breadth]
        return RolePlan(
            tier=tier,
            test_designer_model=fallback_designer,
            judge_model=fallback_judge,
            target_models=targets,
            catalog_revision="fallback",
            policy="No current intelligence metadata; explicit verified fallbacks are used until task tests can qualify a replacement.",
        )

    normalized.sort(key=lambda item: (item["price"], -item["intelligence"], item["id"]))
    maximum = max(item["intelligence"] for item in normalized)

    def cheapest_above(delta: float) -> str:
        eligible = [item for item in normalized if item["intelligence"] >= maximum - delta]
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
    for item in sorted(selected, key=lambda value: (value["price"], -value["intelligence"])):
        for effort in item.get("reasoning_efforts") or ("none",):
            configured = f"{item['id']}#reasoning={effort}"
            if configured not in broad_ids and configured not in hone_ids:
                hone_ids.append(configured)
    target_ids = (broad_ids + hone_ids)[:25]
    revision = _hash(json.dumps(normalized, sort_keys=True, separators=(",", ":")))
    return RolePlan(
        tier=tier,
        test_designer_model=cheapest_above(designer_delta),
        judge_model=cheapest_above(judge_delta),
        target_models=tuple(target_ids),
        catalog_revision=revision,
        policy=(
            "The smartest role is protected for test design; judging uses the cheapest model above a lower intelligence floor; "
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
    _router: "DurableRouter | None" = None

    def accept(self) -> None:
        if not self._router:
            raise RuntimeError("This routed answer is detached from its Evalt state.")
        self._router.record_feedback(self.call_id, approved_output=self.content, verdict="accepted")

    def correct(self, approved_output: Any) -> None:
        if not self._router:
            raise RuntimeError("This routed answer is detached from its Evalt state.")
        self._router.record_feedback(self.call_id, approved_output=_content(approved_output), verdict="corrected")

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
                """
            )
            columns = {row["name"] for row in db.execute("PRAGMA table_info(routes)")}
            if "source_prompt_version" not in columns:
                db.execute("ALTER TABLE routes ADD COLUMN source_prompt_version TEXT NOT NULL DEFAULT ''")
                db.execute("UPDATE routes SET source_prompt_version=prompt_version WHERE source_prompt_version='' ")
            for name, declaration in (
                ("max_cost_per_run_usd", "REAL NOT NULL DEFAULT 0.02"),
                ("test_budget_usd", "REAL NOT NULL DEFAULT 0"),
                ("objective", "TEXT NOT NULL DEFAULT 'best_within_price'"),
                ("test_budget_policy", "TEXT NOT NULL DEFAULT 'legacy'"),
                ("max_p90_latency_seconds", "REAL DEFAULT NULL"),
                ("latency_value_usd_per_second", "REAL NOT NULL DEFAULT 0"),
                ("optimize_prompt", "INTEGER NOT NULL DEFAULT 1"),
            ):
                if name not in columns:
                    db.execute(f"ALTER TABLE routes ADD COLUMN {name} {declaration}")

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
    ) -> sqlite3.Row:
        if not route.strip():
            raise ValueError("route must be a stable non-empty name so Evalt can remember decisions.")
        if not prompt.strip():
            raise ValueError("prompt cannot be empty.")
        candidates = tuple(dict.fromkeys(model.strip() for model in models if model.strip()))
        if not candidates:
            raise ValueError("At least one candidate model is required.")
        version = _hash(prompt.strip())[:16]
        now = _now()
        with self._db() as db:
            current = db.execute("SELECT * FROM routes WHERE route = ?", (route,)).fetchone()
            if current is None:
                db.execute(
                    "INSERT INTO routes(route,prompt,source_prompt_version,prompt_version,candidates_json,selected_model,selected_prompt,decision_reason,quality_threshold,catalog_revision,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (route, prompt.strip(), version, version, json.dumps(candidates), candidates[0], prompt.strip(), "bootstrap_unqualified", quality_threshold, catalog_revision, now, now),
                )
                self._event(db, route, "route_created", {"prompt_version": version, "bootstrap_model": candidates[0], "candidates": candidates})
            else:
                changed = current["source_prompt_version"] != version
                candidates_changed = json.loads(current["candidates_json"]) != list(candidates)
                if changed:
                    db.execute(
                        "UPDATE routes SET prompt=?,source_prompt_version=?,prompt_version=?,candidates_json=?,selected_model=?,selected_prompt=?,decision_reason='prompt_changed_unqualified',catalog_revision=?,updated_at=? WHERE route=?",
                        (prompt.strip(), version, version, json.dumps(candidates), candidates[0], prompt.strip(), catalog_revision, now, route),
                    )
                    self._event(db, route, "prompt_changed", {"prompt_version": version, "bootstrap_model": candidates[0]})
                elif candidates_changed or current["catalog_revision"] != catalog_revision:
                    db.execute(
                        "UPDATE routes SET candidates_json=?,catalog_revision=?,updated_at=? WHERE route=?",
                        (json.dumps(candidates), catalog_revision, now, route),
                    )
                    self._event(db, route, "catalog_changed", {"catalog_revision": catalog_revision, "candidates": candidates})
            return db.execute("SELECT * FROM routes WHERE route = ?", (route,)).fetchone()

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
        max_cost_per_run_usd: float,
        models: Sequence[str] = DEFAULT_TARGETS,
        max_tokens: int = 600,
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
        if not 0 < float(max_cost_per_run_usd) <= 10:
            raise ValueError("price_usd must be greater than 0 and no more than 10.")
        if objective not in {"match_baseline_at_lowest_cost", "best_within_price", "best_within_cost", "lowest_cost_at_accuracy", "cheapest_at_accuracy", "cheapest_passing", "constrained", "highest_quality"}:
            raise ValueError("objective is not a supported cost/accuracy policy.")
        if max_p90_latency_seconds is not None and max_p90_latency_seconds <= 0:
            raise ValueError("max_p90_latency_seconds must be positive when provided.")
        if latency_value_usd_per_second < 0:
            raise ValueError("latency_value_usd_per_second cannot be negative.")
        row = self._ensure_route(
            route=route,
            prompt=prompt,
            models=models,
            quality_threshold=target_accuracy,
            catalog_revision=catalog_revision,
        )
        with self._db() as db:
            current = db.execute("SELECT * FROM routes WHERE route=?", (route,)).fetchone()
            controls = (
                float(max_cost_per_run_usd), float(test_budget_usd), objective,
                test_budget_policy, max_p90_latency_seconds,
                float(latency_value_usd_per_second), int(bool(optimize_prompt)),
            )
            previous = (
                current["max_cost_per_run_usd"], current["test_budget_usd"],
                current["objective"], current["test_budget_policy"],
                current["max_p90_latency_seconds"],
                current["latency_value_usd_per_second"],
                current["optimize_prompt"],
            )
            db.execute(
                "UPDATE routes SET quality_threshold=?,max_cost_per_run_usd=?,test_budget_usd=?,objective=?,test_budget_policy=?,max_p90_latency_seconds=?,latency_value_usd_per_second=?,optimize_prompt=?,updated_at=? WHERE route=?",
                (float(target_accuracy), *controls, _now(), route),
            )
            if previous != controls:
                self._event(db, route, "routing_policy_configured", {
                    "price_usd": max_cost_per_run_usd,
                    "test_budget_usd": test_budget_usd,
                    "test_budget_policy": test_budget_policy,
                    "target_accuracy": target_accuracy,
                    "objective": objective,
                    "max_p90_latency_seconds": max_p90_latency_seconds,
                    "latency_value_usd_per_second": latency_value_usd_per_second,
                    "optimize_prompt": bool(optimize_prompt),
                })
            row = db.execute("SELECT * FROM routes WHERE route=?", (route,)).fetchone()
        set_performance_policy = getattr(self.client.transport, "set_performance_policy", None)
        if callable(set_performance_policy):
            set_performance_policy(
                preferred_max_latency_seconds=max_p90_latency_seconds,
                provider_sort=("latency" if latency_value_usd_per_second > 0 else "price"),
            )
        input_text = _content(input)
        messages = [{"role": "system", "content": row["selected_prompt"]}, {"role": "user", "content": input_text}]
        estimate = self.client.transport.estimate_cost(row["selected_model"], messages, max_tokens=max_tokens)
        if estimate > float(max_cost_per_run_usd) + 1e-12:
            raise BudgetExceeded(
                f"The selected route estimates ${estimate:.6f}, above this call's ${float(max_cost_per_run_usd):.6f} price ceiling."
            )
        completion: Completion = self.client.transport.complete(row["selected_model"], messages, max_tokens=max_tokens)
        if completion.cost_usd > float(max_cost_per_run_usd) + 1e-12:
            raise BudgetExceeded("The provider-reported cost exceeded this call's hard cap.")
        call_id = f"call-{uuid.uuid4().hex}"
        with self._db() as db:
            db.execute(
                "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?)",
                (call_id, route, _now(), input_text, completion.content, completion.model, row["prompt_version"], completion.cost_usd, float(max_cost_per_run_usd), row["decision_reason"]),
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
            _router=self,
        )

    def record_feedback(self, call_id: str, *, approved_output: str, verdict: str) -> None:
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
            if existed is None:
                db.execute("UPDATE routes SET feedback_count=feedback_count+1,updated_at=? WHERE route=?", (_now(), call["route"]))
            self._event(db, call["route"], "feedback_recorded", {"call_id": call_id, "verdict": verdict})

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
                    "SELECT calls.input_text,calls.output_text,feedback.approved_output,feedback.verdict FROM feedback JOIN calls USING(call_id) WHERE feedback.route=? ORDER BY feedback.created_at",
                    (route,),
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
                    "role_plan": role_plan.to_dict(),
                    "judge_calibration_spend_usd": maintenance_budget.spent_usd,
                    "optimization_spend_usd": result.total_provider_spend_usd,
                    "total_maintenance_spend_usd": maintenance_budget.spent_usd + result.total_provider_spend_usd,
                    "winner_model": winner.model,
                    "winner_prompt_version": _hash(winner.selected_prompt)[:16],
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
                }
                if promoted:
                    db.execute(
                        "UPDATE routes SET selected_model=?,selected_prompt=?,prompt_version=?,decision_reason=?,last_optimized_calls=?,last_optimized_feedback=?,tested_catalog_revision=?,updated_at=? WHERE route=?",
                        (winner.model, winner.selected_prompt, _hash(winner.selected_prompt)[:16], f"qualified_{objective}", current["total_calls"], current["feedback_count"], role_plan.catalog_revision, _now(), route),
                    )
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
        return {
            "schema": "evalt-route-audit-v1",
            "route": route,
            "selected_model": row["selected_model"],
            "selected_prompt_version": row["prompt_version"],
            "decision_reason": row["decision_reason"],
            "price_usd": row["max_cost_per_run_usd"],
            "test_budget_usd": row["test_budget_usd"],
            "test_budget_policy": row["test_budget_policy"],
            "target_accuracy": row["quality_threshold"],
            "objective": row["objective"],
            "max_p90_latency_seconds": row["max_p90_latency_seconds"],
            "latency_value_usd_per_second": row["latency_value_usd_per_second"],
            "optimize_prompt": bool(row["optimize_prompt"]),
            "total_calls": row["total_calls"],
            "feedback_count": row["feedback_count"],
            "maintenance_due": list(self._due(row, retest_after_calls=retest_after_calls, min_feedback=min_feedback)),
            "catalog_revision": row["catalog_revision"],
            "tested_catalog_revision": row["tested_catalog_revision"],
            "decisions": decisions,
        }

    def export_audit(self, route: str, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.status(route), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = ["DEFAULT_TARGETS", "DurableRouter", "RolePlan", "RoutedAnswer", "select_role_plan"]
