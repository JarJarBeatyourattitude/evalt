import json
import os
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest import mock

from evalt import BudgetExceeded, Client, Evalt, ProviderError, Suite, check_result, select_role_plan
from evalt.cli import STARTER_SUITE, main as cli_main
from evalt.core import Completion, OpenRouterTransport, _safe_provider_error_detail
from evalt.migration import migrate_openai_results
from modelsieve import Client as ModelSieveClient
from last_good_prompt import Client as LegacyClient


class CompatibilityTests(unittest.TestCase):
    def test_earlier_imports_resolve_to_evalt_client(self):
        self.assertIs(LegacyClient, Client)
        self.assertIs(ModelSieveClient, Client)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class FakeTransport:
    def __init__(self):
        self.calls = []

    def estimate_cost(self, model, messages, *, max_tokens):
        return 0.001 if model == "expensive" else 0.0001

    def complete(self, model, messages, *, max_tokens, response_schema=None):
        self.calls.append((model, messages, max_tokens, response_schema))
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if response_schema and "Improve the current prompt" in system:
            content = json.dumps({"prompt": "Return the approved route label only.", "hypothesis": "Remove prose."})
        elif response_schema and "Judge whether" in system:
            value = json.loads(user)
            passed = value["actual_answer"].strip() == value["approved_answer"].strip()
            content = json.dumps({"passed": passed, "score": 1 if passed else 0, "reason": "exact fixture judgment"})
        else:
            raw_input = user.lower()
            if "approved route label" in system:
                if "charged" in raw_input:
                    content = "billing"
                elif "reset" in raw_input:
                    content = "account"
                else:
                    content = "technical"
            else:
                content = "Here is a verbose answer"
        return Completion(
            content=content,
            model=model,
            generation_id=f"gen-{len(self.calls)}",
            cost_usd=self.estimate_cost(model, messages, max_tokens=max_tokens),
            prompt_tokens=10,
            completion_tokens=3,
        )


class PartlyUnavailableTransport(FakeTransport):
    def complete(self, model, messages, *, max_tokens, response_schema=None):
        if model == "unavailable":
            raise ProviderError("No strict-ZDR route is currently available.")
        return super().complete(
            model, messages, max_tokens=max_tokens, response_schema=response_schema
        )


class PolicyTransport(FakeTransport):
    def __init__(self):
        super().__init__()
        self.performance_policies = []

    def set_performance_policy(self, *, preferred_max_latency_seconds=None, provider_sort="price"):
        self.performance_policies.append((preferred_max_latency_seconds, provider_sort))


class SlowTransport(FakeTransport):
    def complete(self, model, messages, *, max_tokens, response_schema=None):
        result = super().complete(
            model, messages, max_tokens=max_tokens, response_schema=response_schema
        )
        return Completion(
            result.content, result.model, result.generation_id, result.cost_usd,
            result.prompt_tokens, result.completion_tokens, 5_000,
        )


class FewShotTransport(FakeTransport):
    def complete(self, model, messages, *, max_tokens, response_schema=None):
        if response_schema and "Improve the current prompt" in messages[0]["content"]:
            self.calls.append((model, messages, max_tokens, response_schema))
            payload = json.loads(messages[-1]["content"])
            chosen = payload["allowed_few_shot_example_ids"][:1]
            return Completion(json.dumps({"prompt": "Return the approved route label only.", "hypothesis": "Use one approved demonstration.", "few_shot_example_ids": chosen}), model, f"gen-{len(self.calls)}", self.estimate_cost(model, messages, max_tokens=max_tokens))
        return super().complete(model, messages, max_tokens=max_tokens, response_schema=response_schema)


class PriceFrontierTransport(FakeTransport):
    def complete(self, model, messages, *, max_tokens, response_schema=None):
        if response_schema:
            return super().complete(model, messages, max_tokens=max_tokens, response_schema=response_schema)
        self.calls.append((model, messages, max_tokens, response_schema))
        if model == "cheap":
            content = "wrong"
        else:
            user = messages[-1]["content"].lower()
            content = "billing" if any(term in user for term in ("charged", "charge", "invoice", "refund")) else "account" if "reset" in user else "technical"
        return Completion(content, model, f"gen-{len(self.calls)}", self.estimate_cost(model, messages, max_tokens=max_tokens))


EXAMPLES = [
    {"id": "billing", "input": "I was charged twice", "approved_output": "billing"},
    {"id": "account", "input": "My reset link expired", "approved_output": "account"},
    {"id": "technical", "input": "The app freezes", "approved_output": "technical"},
    {"id": "billing-2", "input": "Refund my charge", "approved_output": "billing"},
    {"id": "account-2", "input": "I cannot sign in", "approved_output": "technical"},
]


class SdkTests(unittest.TestCase):
    def test_checked_in_quickstart_is_syntactically_runnable(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"
        compile(example.read_text(encoding="utf-8"), str(example), "exec")

    def test_evalt_loads_openrouter_key_from_local_dotenv(self):
        with TemporaryDirectory() as directory:
            Path(directory, ".env").write_text("OPENROUTER_API_KEY=from-dotenv\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(Path, "cwd", return_value=Path(directory)):
                evalt = Evalt()
                self.assertEqual(evalt.client.transport._api_key, "from-dotenv")

    def test_process_environment_wins_over_local_dotenv(self):
        with TemporaryDirectory() as directory:
            Path(directory, ".env").write_text("OPENROUTER_API_KEY=from-dotenv\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "from-process"}, clear=True), mock.patch.object(Path, "cwd", return_value=Path(directory)):
                evalt = Evalt()
                self.assertEqual(evalt.client.transport._api_key, "from-process")

    def test_missing_openrouter_key_names_all_supported_sources(self):
        with TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(Path, "cwd", return_value=Path(directory)):
                with self.assertRaisesRegex(ValueError, r"environment or in a \.env file"):
                    Evalt()

    def test_openai_results_migration_is_offline_conservative_and_valid(self):
        rows = [
            {"id": "one", "input": "2 + 2", "ideal": "4", "output": "5"},
            {"sample_id": "two", "sample": {"input": "3 + 3", "expected": "6"}},
            {"messages": [{"role": "user", "content": "4 + 4"}], "ground_truth": "8"},
            {"id": "candidate-only", "input": "5 + 5", "output": "10"},
        ]
        with TemporaryDirectory() as directory:
            source = Path(directory) / "results.jsonl"
            source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            result = migrate_openai_results(
                source, prompt="Return only the number.", name="math", models=["cheap"],
            )
        self.assertIsNotNone(result.suite)
        self.assertEqual(result.report["imported_rows"], 3)
        self.assertEqual(result.report["skipped_rows"], 1)
        self.assertTrue(result.report["skipped"][0]["candidate_output_ignored"])
        self.assertEqual(result.suite["examples"][0]["approved_output"], "4")
        Suite.from_dict(result.suite)

    def test_openai_results_migration_refuses_to_invent_a_runnable_suite(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "results.jsonl"
            source.write_text(
                json.dumps({"input": "hello", "output": "historical candidate answer"}),
                encoding="utf-8",
            )
            result = migrate_openai_results(
                source, prompt="Answer.", name="missing-labels", models=["cheap"],
            )
        self.assertIsNone(result.suite)
        self.assertFalse(result.report["runnable_suite_created"])
        self.assertRegex(result.report["important_limit"], "did not infer")

    def test_openai_results_cli_writes_report_even_when_suite_is_not_runnable(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "results.jsonl"
            output = Path(directory) / "evalt.json"
            source.write_text(json.dumps({"input": "hello", "response": "candidate"}), encoding="utf-8")
            stderr = StringIO()
            with redirect_stderr(stderr):
                code = cli_main([
                    "import-openai-results", str(source), "--prompt", "Answer.",
                    "--output", str(output),
                ])
            report = Path(str(output) + ".migration-report.json")
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())
            self.assertTrue(report.exists())
            self.assertIn("No runnable suite written", stderr.getvalue())

    def test_default_model_lanes_overlap_while_preserving_requested_result_order(self):
        class ParallelTransport(FakeTransport):
            def __init__(self):
                super().__init__()
                self.barrier = threading.Barrier(2)
                self.first_target_seen = set()
                self.lock = threading.Lock()

            def complete(self, model, messages, *, max_tokens, response_schema=None):
                should_wait = False
                if response_schema is None and model in {"cheap-a", "cheap-b"}:
                    with self.lock:
                        if model not in self.first_target_seen:
                            self.first_target_seen.add(model)
                            should_wait = True
                if should_wait:
                    self.barrier.wait(timeout=1)
                return super().complete(model, messages, max_tokens=max_tokens, response_schema=response_schema)

        result = Client(transport=ParallelTransport()).optimize(
            prompt="Return the approved route label only.", examples=EXAMPLES,
            models=["cheap-a", "cheap-b"], optimizer_model="optimizer",
            evaluator_model="evaluator", max_optimization_cost_usd=1,
        )
        self.assertEqual([item.model for item in result.models], ["cheap-a", "cheap-b"])

    def test_parallel_progress_events_are_complete_and_machine_readable(self):
        events = []
        result = Client(transport=FakeTransport()).optimize(
            prompt="Return the approved route label only.", examples=EXAMPLES,
            models=["cheap-a", "cheap-b"], optimizer_model="optimizer",
            evaluator_model="evaluator", max_optimization_cost_usd=1,
            progress_callback=events.append,
        )
        self.assertEqual(len(result.models), 2)
        started = {event["model"] for event in events if event["event"] == "model_started"}
        completed = {event["model"] for event in events if event["event"] == "model_completed"}
        self.assertEqual(started, {"cheap-a", "cheap-b"})
        self.assertEqual(completed, started)
        self.assertTrue(all("target_latency_p90_ms" in event for event in events if event["event"] == "model_completed"))

    def test_latency_ceiling_can_reject_the_cheapest_otherwise_passing_route(self):
        class LatencyTransport(FakeTransport):
            def estimate_cost(self, model, messages, *, max_tokens):
                return 0.0001 if model == "slow-cheap" else 0.0002

            def complete(self, model, messages, *, max_tokens, response_schema=None):
                completion = super().complete(
                    model, messages, max_tokens=max_tokens, response_schema=response_schema
                )
                latency_ms = 9000 if model == "slow-cheap" else 400
                return Completion(
                    completion.content, completion.model, completion.generation_id,
                    completion.cost_usd, completion.prompt_tokens,
                    completion.completion_tokens, latency_ms,
                )

        result = Client(transport=LatencyTransport()).optimize(
            prompt="Return the approved route label only.", examples=EXAMPLES,
            models=["slow-cheap", "fast-costlier"], optimizer_model="optimizer",
            evaluator_model="evaluator", max_optimization_cost_usd=1,
            max_p90_latency_seconds=2,
        )
        self.assertEqual(result.winner.model, "fast-costlier")
        by_model = {item.model: item for item in result.models}
        self.assertFalse(by_model["slow-cheap"].passed_latency_ceiling)
        self.assertEqual(by_model["slow-cheap"].target_latency_p90_ms, 9000)
        self.assertTrue(by_model["fast-costlier"].passed_latency_ceiling)

    def test_scenario_lanes_overlap_but_each_multiturn_transcript_stays_ordered(self):
        class ScenarioParallelTransport(FakeTransport):
            def __init__(self):
                super().__init__()
                self.barrier = threading.Barrier(3)
                self.first_inputs = set()
                self.lock = threading.Lock()

            def complete(self, model, messages, *, max_tokens, response_schema=None):
                should_wait = False
                if response_schema is None and model == "cheap":
                    user = messages[-1]["content"]
                    with self.lock:
                        if user not in self.first_inputs and len(self.first_inputs) < 3:
                            self.first_inputs.add(user)
                            should_wait = True
                if should_wait:
                    self.barrier.wait(timeout=1)
                return super().complete(model, messages, max_tokens=max_tokens, response_schema=response_schema)

        parallel_examples = [
            {"id": f"billing-{index}", "turns": [
                {"input": f"I was charged twice {index}", "approved_output": "billing"},
                {"input": f"The charge is still there {index}", "approved_output": "billing"},
            ]}
            for index in range(15)
        ]
        result = Client(transport=ScenarioParallelTransport()).optimize(
            prompt="Return the approved route label only.", examples=parallel_examples,
            models=["cheap"], optimizer_model="optimizer", evaluator_model="evaluator",
            max_optimization_cost_usd=1, max_parallel_models=1, max_parallel_scenarios=3,
        )
        self.assertEqual(result.winner.model, "cheap")

    def test_repeated_case_executions_use_the_thirty_two_lane_default(self):
        class MeasuredParallelTransport(FakeTransport):
            def __init__(self):
                super().__init__()
                self.active = 0
                self.peak = 0
                self.lock = threading.Lock()

            def complete(self, model, messages, *, max_tokens, response_schema=None):
                if response_schema is None and model == "cheap":
                    with self.lock:
                        self.active += 1
                        self.peak = max(self.peak, self.active)
                    time.sleep(0.025)
                    try:
                        return super().complete(model, messages, max_tokens=max_tokens, response_schema=response_schema)
                    finally:
                        with self.lock:
                            self.active -= 1
                return super().complete(model, messages, max_tokens=max_tokens, response_schema=response_schema)

        examples = [
            {"id": f"parallel-{index}", "input": f"I was charged twice {index}", "approved_output": "billing"}
            for index in range(50)
        ]
        transport = MeasuredParallelTransport()
        result = Client(transport=transport).optimize(
            prompt="Return the approved route label only.", examples=examples,
            models=["cheap"], optimizer_model="optimizer", evaluator_model="unused",
            evaluator={"type": "exact_text"}, rounds=1, holdout_repeats=2,
            max_optimization_cost_usd=2,
        )
        self.assertEqual(result.winner.model, "cheap")
        self.assertGreater(transport.peak, 16)
        self.assertLessEqual(transport.peak, 32)

    def test_exact_json_evaluator_is_zero_cost_and_normalizes_rational_values(self):
        class ExactJsonTransport(FakeTransport):
            def complete(self, model, messages, *, max_tokens, response_schema=None):
                if response_schema:
                    return super().complete(model, messages, max_tokens=max_tokens, response_schema=response_schema)
                self.calls.append((model, messages, max_tokens, response_schema))
                return Completion('{"x":"2/4","y":"-4/3"}', model, f"gen-{len(self.calls)}", 0.0001)

        examples = [
            {"id": f"system-{index}", "input": f"Solve fixture {index}", "approved_output": '{"x":"1/2","y":"-4/3"}'}
            for index in range(5)
        ]
        transport = ExactJsonTransport()
        result = Client(transport=transport).optimize(
            prompt="Return exact x and y as JSON.", examples=examples, models=["target"],
            optimizer_model="optimizer", evaluator_model="must-not-run", rounds=1,
            max_optimization_cost_usd=1,
            evaluator={
                "type": "exact_json", "required_keys": ["x", "y"],
                "allow_additional_properties": False, "normalize_rational_strings": True,
            },
        )
        self.assertEqual(result.winner.holdout_pass_rate, 1)
        self.assertTrue(all(case.evaluator_cost_usd == 0 for case in result.winner.cases))
        self.assertFalse(any(model == "must-not-run" for model, *_rest in transport.calls))

    def test_exact_json_evaluator_rejects_wrong_values_and_extra_keys(self):
        class InvalidJsonTransport(FakeTransport):
            def complete(self, model, messages, *, max_tokens, response_schema=None):
                if response_schema:
                    return super().complete(model, messages, max_tokens=max_tokens, response_schema=response_schema)
                self.calls.append((model, messages, max_tokens, response_schema))
                return Completion('{"x":"1/2","y":"9","note":"guess"}', model, f"gen-{len(self.calls)}", 0.0001)

        examples = [
            {"id": f"system-{index}", "input": f"Solve fixture {index}", "approved_output": '{"x":"1/2","y":"-4/3"}'}
            for index in range(5)
        ]
        result = Client(transport=InvalidJsonTransport()).optimize(
            prompt="Return exact x and y as JSON.", examples=examples, models=["target"],
            optimizer_model="optimizer", evaluator_model="must-not-run", rounds=1,
            max_optimization_cost_usd=1,
            evaluator={
                "type": "exact_json", "required_keys": ["x", "y"],
                "allow_additional_properties": False, "normalize_rational_strings": True,
            },
        )
        self.assertEqual(result.winner.holdout_pass_rate, 0)
        self.assertTrue(any("Unexpected JSON key" in case.reason for case in result.winner.cases))

    def test_perfect_validation_skips_training_and_prompt_rewrite(self):
        class AlreadyCorrectTransport(FakeTransport):
            def complete(self, model, messages, *, max_tokens, response_schema=None):
                self.calls.append((model, messages, max_tokens, response_schema))
                return Completion("billing", model, f"gen-{len(self.calls)}", 0.0001)

        examples = [
            {"id": f"billing-{index}", "input": f"charged twice {index}", "approved_output": "billing"}
            for index in range(10)
        ]
        transport = AlreadyCorrectTransport()
        result = Client(transport=transport).optimize(
            prompt="Return the approved route label only.", examples=examples,
            models=["target"], optimizer_model="optimizer", evaluator_model="unused",
            evaluator={"type": "exact_text"}, rounds=3,
            max_optimization_cost_usd=1,
        )

        called_models = [model for model, *_rest in transport.calls]
        self.assertNotIn("optimizer", called_models)
        self.assertFalse(any(case.split == "train" for case in result.winner.cases))
        self.assertEqual(result.winner.holdout_pass_rate, 1)

    def test_price_first_api_keeps_test_budget_and_accuracy_as_separate_controls(self):
        with TemporaryDirectory() as directory:
            evalt = Evalt(transport=FakeTransport(), state_path=Path(directory) / "evalt.db")
            answer = evalt.run(
                "Return the approved route label only.", "charged twice",
                route="price-first", price_usd=0.003, test_budget_usd=0.40,
                target_accuracy=0.95, objective="best_within_price",
                models=["cheap"], auto_maintain=False,
            )
            self.assertEqual(answer.content, "billing")
            status = evalt.route_status("price-first")
            self.assertEqual(status["price_usd"], 0.003)
            self.assertEqual(status["test_budget_usd"], 0.40)
            self.assertEqual(status["target_accuracy"], 0.95)
            self.assertEqual(status["objective"], "best_within_price")

    def test_default_route_uses_accuracy_target_without_an_incumbent(self):
        with TemporaryDirectory() as directory:
            evalt = Evalt(transport=FakeTransport(), state_path=Path(directory) / "evalt.db")
            answer = evalt.run(
                "Return the correct label.", "test input",
                price_usd=0.05,
                models=["cheap"], auto_maintain=False,
            )
            self.assertEqual(answer.model, "cheap")
            status = evalt.route_status("default")
            self.assertEqual(status["objective"], "lowest_cost_at_accuracy")
            self.assertEqual(status["target_accuracy"], 0.95)
            self.assertIsNone(status["max_p90_latency_seconds"])

    def test_primary_run_is_a_durable_budget_bounded_router_not_a_json_export(self):
        with TemporaryDirectory() as directory:
            state = Path(directory) / "evalt.db"
            first = Evalt(transport=FakeTransport(), state_path=state).run(
                "Return the approved route label only.",
                "I was charged twice",
                route="support-route",
                budget_usd=0.01,
                models=["cheap", "expensive"],
                incumbent_model="cheap",
                retest_after_calls=2,
                min_feedback=1,
            )
            self.assertEqual(first.content, "billing")
            self.assertEqual(first.model, "cheap")
            self.assertEqual(first.decision_reason, "bootstrap_unqualified")
            first.accept()

            restarted = Evalt(transport=FakeTransport(), state_path=state)
            status = restarted.route_status("support-route", retest_after_calls=2, min_feedback=1)
            self.assertEqual(status["total_calls"], 1)
            self.assertEqual(status["feedback_count"], 1)
            self.assertIn("new_human_feedback", status["maintenance_due"])
            self.assertTrue(status["decisions"])

    def test_named_routes_keep_multiple_tasks_and_their_evidence_isolated(self):
        with TemporaryDirectory() as directory:
            state = Path(directory) / "evalt.db"
            evalt = Evalt(transport=FakeTransport(), state_path=state)
            support = evalt.run(
                "Return the approved route label only.", "I was charged twice",
                route="support-routing", budget_usd=0.01, models=["cheap"], auto_maintain=False,
            )
            incident = evalt.run(
                "Return the approved route label only. This route triages incidents.", "The app freezes",
                route="incident-triage", budget_usd=0.01, models=["cheap"], auto_maintain=False,
            )
            support.accept()
            incident.correct("urgent")

            support_status = evalt.route_status("support-routing")
            incident_status = evalt.route_status("incident-triage")
            self.assertEqual(support.content, "billing")
            self.assertEqual(incident.content, "technical")
            self.assertEqual(support_status["route"], "support-routing")
            self.assertEqual(incident_status["route"], "incident-triage")
            self.assertEqual(support_status["total_calls"], 1)
            self.assertEqual(incident_status["total_calls"], 1)
            self.assertEqual(support_status["feedback_count"], 1)
            self.assertEqual(incident_status["feedback_count"], 1)
            self.assertNotEqual(support.prompt_version, incident.prompt_version)

    def test_durable_route_persists_latency_policy_and_applies_provider_preference(self):
        with TemporaryDirectory() as directory:
            transport = PolicyTransport()
            evalt = Evalt(transport=transport, state_path=Path(directory) / "evalt.db")
            evalt.run(
                "Return one label.", "hello", route="speed-sensitive",
                price_usd=0.01, models=["cheap"], auto_maintain=False,
                max_p90_latency_seconds=2.5,
                latency_value_usd_per_second=0.0002,
            )
            status = evalt.route_status("speed-sensitive")
            self.assertEqual(status["max_p90_latency_seconds"], 2.5)
            self.assertEqual(status["latency_value_usd_per_second"], 0.0002)
            self.assertEqual(transport.performance_policies[-1], (2.5, "latency"))
            policy_events = [
                item for item in status["decisions"]
                if item["event_type"] == "routing_policy_configured"
            ]
            self.assertEqual(policy_events[-1]["detail"]["max_p90_latency_seconds"], 2.5)

    def test_simple_latency_alias_persists_a_conservative_p90_ceiling(self):
        with TemporaryDirectory() as directory:
            transport = PolicyTransport()
            evalt = Evalt(transport=transport, state_path=Path(directory) / "evalt.db")
            evalt.run(
                "Return one label.", "hello", route="simple-speed-ceiling",
                price_usd=0.01, models=["cheap"], auto_maintain=False,
                max_latency_seconds=3.0,
            )
            status = evalt.route_status("simple-speed-ceiling")
            self.assertEqual(status["max_p90_latency_seconds"], 3.0)
            self.assertEqual(transport.performance_policies[-1], (3.0, "price"))

    def test_simple_and_advanced_latency_names_cannot_conflict(self):
        with TemporaryDirectory() as directory:
            evalt = Evalt(transport=FakeTransport(), state_path=Path(directory) / "evalt.db")
            with self.assertRaisesRegex(ValueError, "not conflicting values"):
                evalt.run(
                    "Return one label.", "hello", models=["cheap"], auto_maintain=False,
                    max_latency_seconds=3.0, max_p90_latency_seconds=2.0,
                )

    def test_durable_maintenance_never_promotes_a_route_that_misses_latency_ceiling(self):
        with TemporaryDirectory() as directory:
            evalt = Evalt(transport=SlowTransport(), state_path=Path(directory) / "evalt.db")
            cases = [("charged twice", "billing"), ("reset expired", "account"), ("app freezes", "technical"), ("charged again", "billing"), ("reset broken", "account")]
            for input_text, approved in cases:
                answer = evalt.run(
                    "Write a helpful classification for this message.", input_text,
                    route="latency-gated", price_usd=0.01, models=["cheap"],
                    incumbent_model="cheap", min_feedback=5,
                    max_p90_latency_seconds=1,
                )
                answer.correct(approved)
            plan = select_role_plan([], maintenance_budget_usd=1, fallback_targets=["cheap"], fallback_designer="optimizer", fallback_judge="evaluator")
            result = evalt.maintain("latency-gated", test_budget_usd=1, role_plan=plan, min_feedback=5)
            self.assertIsNotNone(result)
            self.assertFalse(result.winner.passed_latency_ceiling)
            status = evalt.route_status("latency-gated")
            self.assertNotEqual(status["decision_reason"], "qualified_cheapest_passing")
            self.assertIn("maintenance_no_promotion", [item["event_type"] for item in status["decisions"]])

    def test_router_rejects_a_call_before_provider_use_when_request_cap_is_too_low(self):
        with TemporaryDirectory() as directory:
            transport = FakeTransport()
            with self.assertRaises(BudgetExceeded):
                Evalt(transport=transport, state_path=Path(directory) / "evalt.db").run(
                    "Return one label.", "hello", route="tiny-budget", budget_usd=0.00001, models=["cheap"], incumbent_model="cheap"
                )
            self.assertEqual(transport.calls, [])

    def test_prompt_change_is_logged_and_reverts_to_unqualified_bootstrap(self):
        with TemporaryDirectory() as directory:
            state = Path(directory) / "evalt.db"
            evalt = Evalt(transport=FakeTransport(), state_path=state)
            first = evalt.run("Return the approved route label only.", "charged", route="support", budget_usd=0.01, models=["cheap"], incumbent_model="cheap")
            second = evalt.run("Return only billing, account, or technical.", "charged", route="support", budget_usd=0.01, models=["cheap"], incumbent_model="cheap")
            self.assertNotEqual(first.prompt_version, second.prompt_version)
            self.assertEqual(second.decision_reason, "prompt_changed_unqualified")
            events = [item["event_type"] for item in evalt.route_status("support")["decisions"]]
            self.assertIn("prompt_changed", events)

    def test_role_policy_protects_design_quality_and_expands_breadth_with_budget(self):
        catalog = [
            {"id": "tiny", "intelligence": 55, "blended_price": 0.1},
            {"id": "balanced", "intelligence": 78, "blended_price": 0.8},
            {"id": "smart", "intelligence": 91, "blended_price": 4.0},
            {"id": "frontier", "intelligence": 96, "blended_price": 9.0},
        ]
        lean = select_role_plan(catalog, maintenance_budget_usd=0.25)
        deep = select_role_plan(catalog, maintenance_budget_usd=3.0)
        self.assertEqual(lean.tier, "lean")
        self.assertEqual(deep.tier, "deep")
        self.assertEqual(deep.test_designer_model, "frontier")
        self.assertGreaterEqual(len(deep.target_models), len(lean.target_models))
        self.assertNotEqual(lean.judge_model, "tiny")

    def test_catalog_revision_changes_when_openrouter_price_changes(self):
        before = select_role_plan([
            {"id": "tiny", "intelligence": 55, "blended_price": 0.10},
            {"id": "smart", "intelligence": 91, "blended_price": 1.00},
        ], maintenance_budget_usd=0.50)
        after = select_role_plan([
            {"id": "tiny", "intelligence": 55, "blended_price": 0.30},
            {"id": "smart", "intelligence": 91, "blended_price": 1.00},
        ], maintenance_budget_usd=0.50)
        self.assertNotEqual(before.catalog_revision, after.catalog_revision)

        with TemporaryDirectory() as directory:
            evalt = Evalt(transport=FakeTransport(), state_path=Path(directory) / "evalt.db")
            answer = evalt.router.run(
                route="repriced", prompt="Return one label.", input="hello",
                max_cost_per_run_usd=0.01, models=["cheap"], catalog_revision=after.catalog_revision,
            )
            self.assertIn("model_or_price_catalog_changed", answer.maintenance_due)

    def test_openrouter_catalog_refreshes_prices_after_the_ttl(self):
        model_catalog_calls = 0

        def opener(request, timeout):
            nonlocal model_catalog_calls
            if "endpoints/zdr" in request.full_url:
                prompt_price = "0.000001" if model_catalog_calls == 1 else "0.000003"
                return FakeResponse({"data": [{
                    "model_id": "priced-model",
                    "pricing": {"prompt": prompt_price, "completion": "0.000002"},
                    "supported_parameters": ["max_completion_tokens"],
                }]})
            model_catalog_calls += 1
            prompt_price = "0.000001" if model_catalog_calls == 1 else "0.000003"
            return FakeResponse({"data": [{
                "id": "priced-model",
                "pricing": {"prompt": prompt_price, "completion": "0.000002"},
                "supported_parameters": ["max_completion_tokens"],
                "benchmarks": {"artificial_analysis": {"intelligence_index": 70}},
            }]})

        transport = OpenRouterTransport(
            "sk-or-v1-test-key", opener=opener, catalog_ttl_seconds=0,
        )
        first = transport.model_catalog()[0]["blended_price"]
        second = transport.model_catalog()[0]["blended_price"]
        self.assertNotEqual(first, second)
        self.assertEqual(model_catalog_calls, 2)

    def test_maintenance_calibrates_the_cheaper_judge_then_promotes_a_passing_route(self):
        with TemporaryDirectory() as directory:
            evalt = Evalt(transport=FakeTransport(), state_path=Path(directory) / "evalt.db")
            answers = []
            cases = [("charged twice", "billing"), ("reset expired", "account"), ("app freezes", "technical"), ("charged again", "billing"), ("reset broken", "account")]
            for input_text, approved in cases:
                answer = evalt.run(
                    "Write a helpful classification for this message.", input_text,
                    route="maintained", budget_usd=0.01, models=["cheap"], incumbent_model="cheap", min_feedback=5,
                )
                answers.append(answer)
                answer.correct(approved)
            plan = select_role_plan([], maintenance_budget_usd=1, fallback_targets=["cheap"], fallback_designer="optimizer", fallback_judge="evaluator")
            result = evalt.maintain("maintained", maintenance_budget_usd=1, role_plan=plan, min_feedback=5)
            self.assertIsNotNone(result)
            status = evalt.route_status("maintained")
            self.assertEqual(status["decision_reason"], "qualified_cheapest_passing")
            self.assertIn("route_promoted", [item["event_type"] for item in status["decisions"]])
            served = evalt.run(
                "Write a helpful classification for this message.", "charged once more",
                route="maintained", budget_usd=0.01, models=["cheap"], incumbent_model="cheap", min_feedback=5,
            )
            self.assertEqual(served.decision_reason, "qualified_cheapest_passing")

    def test_typed_suite_validates_offline_and_runs_through_primary_evalt_api(self):
        suite = Suite.from_dict({
            "name": "support-routing",
            "prompt": "Write a helpful classification for this message.",
            "examples": EXAMPLES,
            "models": ["cheap"],
            "optimizer_model": "optimizer",
            "evaluator_model": "evaluator",
            "max_optimization_cost_usd": 1,
        })
        result = Evalt(transport=FakeTransport()).run(suite)
        self.assertEqual(suite.name, "support-routing")
        self.assertTrue(suite.optimize_kwargs()["adaptive_search"])
        self.assertEqual(result.winner.model, "cheap")
        self.assertFalse(check_result(result.to_dict(), min_pass_rate=0.9).passed)
        self.assertIn("exploratory", check_result(result.to_dict(), min_pass_rate=0.9).failures[0])

    def test_suite_persists_a_long_configurable_provider_deadline(self):
        suite = Suite.from_dict({
            "name": "long-context-routing",
            "prompt": "Return the approved route label only.",
            "examples": EXAMPLES,
            "models": ["cheap"],
            "request_timeout_seconds": 1200,
        })
        self.assertEqual(suite.request_timeout_seconds, 1200)
        self.assertEqual(suite.to_dict()["request_timeout_seconds"], 1200)
        latency_suite = Suite.from_dict({
            "prompt": "Return the approved route label only.",
            "examples": EXAMPLES,
            "models": ["cheap"],
            "max_p90_latency_seconds": 3.5,
            "latency_value_usd_per_second": 0.00002,
        })
        self.assertEqual(latency_suite.optimize_kwargs()["max_p90_latency_seconds"], 3.5)
        self.assertEqual(latency_suite.to_dict()["latency_value_usd_per_second"], 0.00002)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            Suite.from_dict({
                "prompt": "Return the approved route label only.",
                "examples": EXAMPLES,
                "models": ["cheap"],
                "request_timeout_seconds": 0,
            })

    def test_transport_defaults_to_ten_minutes_and_allows_long_complex_jobs(self):
        transport = OpenRouterTransport("sk-or-v1-test-key")
        self.assertEqual(transport.timeout_seconds, 600)
        transport.set_timeout_seconds(1800)
        self.assertEqual(transport.timeout_seconds, 1800)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            transport.set_timeout_seconds(0)

    def test_result_gate_fails_quality_cost_and_partial_coverage_for_ci(self):
        report = check_result({
            "winner": {
                "holdout_pass_rate": 0.8,
                "estimated_cost_per_successful_call_usd": 0.01,
            },
            "winner_scope": "Best among fully completed targets only",
        }, min_pass_rate=0.9, max_cost_per_success_usd=0.005, require_complete_coverage=True)
        self.assertFalse(report.passed)
        self.assertEqual(len(report.failures), 3)

    def test_result_gate_accepts_compact_web_result_shape(self):
        report = check_result({
            "best_candidate": {
                "pass_rate": 1.0,
                "cost_per_success_usd": 0.001,
            },
            "coverage_complete": True,
        }, min_pass_rate=0.95, max_cost_per_success_usd=0.01, require_complete_coverage=True)
        self.assertTrue(report.passed)
        self.assertEqual(report.holdout_pass_rate, 1.0)

    def test_cli_initializes_validates_and_gates_without_provider_calls(self):
        with TemporaryDirectory() as directory:
            suite_path = Path(directory) / "evalt.json"
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(cli_main(["init", str(suite_path)]), 0)
                self.assertEqual(cli_main(["validate", str(suite_path)]), 0)
            self.assertIn("not production evidence", output.getvalue())
            self.assertIn('"exploratory": true', output.getvalue())
            self.assertIn('"per_provider_request_timeout_seconds": 600', output.getvalue())
            result_path = Path(directory) / "result.json"
            result_path.write_text(json.dumps({
                "winner": {"holdout_pass_rate": 1.0, "estimated_cost_per_successful_call_usd": 0.001},
                "winner_scope": "Best among every requested target",
            }), encoding="utf-8")
            with redirect_stdout(output), redirect_stderr(output):
                self.assertEqual(cli_main(["check", str(result_path), "--min-pass-rate", "0.95"]), 0)
            self.assertIn("provider_call_started", output.getvalue())

    def test_cli_persists_failure_receipt_when_no_provider_lane_completes(self):
        class FailedClient:
            def optimize(self, **_kwargs):
                raise ProviderError("provider timed out")

        class FailedEvalt:
            def __init__(self, **_kwargs):
                self.client = FailedClient()

        with TemporaryDirectory() as directory:
            suite_path = Path(directory) / "suite.json"
            output_path = Path(directory) / "result.json"
            suite_path.write_text(json.dumps({
                **STARTER_SUITE,
                "models": ["example/model#reasoning=high"],
            }), encoding="utf-8")
            with mock.patch("evalt.cli.Evalt", FailedEvalt):
                self.assertEqual(cli_main([
                    "optimize", str(suite_path), "--output", str(output_path),
                ]), 2)
            failure = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(failure["schema"], "evalt-run-failure-v1")
            self.assertEqual(failure["status"], "INCOMPLETE")
            self.assertEqual(failure["error_type"], "ProviderError")
            self.assertIsNone(failure["provider_spend_usd"])
            self.assertRegex(failure["provider_spend_note"], "may still be billable")

    def test_provider_errors_omit_account_identifiers(self):
        detail = _safe_provider_error_detail(json.dumps({
            "error": {"message": "Reasoning is mandatory.", "code": 400},
            "user_id": "private-account-id",
        }))
        self.assertIn("Reasoning is mandatory.", detail)
        self.assertNotIn("private-account-id", detail)
        self.assertNotIn("user_id", detail)

    def test_transport_enforces_a_total_response_deadline_not_only_socket_activity(self):
        class DripResponse(FakeResponse):
            def read1(self, _size):
                return b"x"

        transport = OpenRouterTransport(
            "sk-or-v1-test-key", timeout_seconds=0.000000001,
            opener=lambda *_args, **_kwargs: DripResponse({}),
        )
        with self.assertRaisesRegex(ProviderError, "total deadline"):
            transport._request("https://example.invalid")

    def test_live_transport_sends_only_catalog_supported_parameters(self):
        requests = []

        def opener(request, timeout):
            requests.append(json.loads(request.data) if request.data else None)
            if request.data is None:
                if "endpoints/zdr" in request.full_url:
                    return FakeResponse({"data": [{
                        "model_id": "model-without-temperature", "tag": "fixture/full", "context_length": 131072,
                        "max_completion_tokens": 131072,
                        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                        "supported_parameters": ["max_completion_tokens", "structured_outputs"],
                    }]})
                return FakeResponse({"data": [{
                    "id": "model-without-temperature",
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    "supported_parameters": ["max_completion_tokens", "structured_outputs"],
                }]})
            return FakeResponse({
                "id": "gen-1", "model": "model-without-temperature",
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"cost": 0.0001, "prompt_tokens": 4, "completion_tokens": 1},
            })

        transport = OpenRouterTransport("sk-or-v1-test-key", opener=opener)
        transport.complete(
            "model-without-temperature", [{"role": "user", "content": "hello"}],
            max_tokens=25, response_schema={"type": "object"},
        )
        sent = requests[-1]
        self.assertEqual(sent["max_completion_tokens"], 32768)
        self.assertNotIn("max_tokens", sent)
        self.assertNotIn("temperature", sent)
        self.assertEqual(sent["provider"]["sort"], "price")
        self.assertEqual(sent["usage"], {"include": True})
        self.assertEqual(sent["response_format"]["type"], "json_schema")

    def test_reasoning_effort_is_a_costed_auditable_model_configuration(self):
        requests = []

        def opener(request, timeout):
            requests.append(json.loads(request.data) if request.data else None)
            if request.data is None:
                if "endpoints/zdr" in request.full_url:
                    return FakeResponse({"data": [{
                        "model_id": "reasoning-model", "tag": "fixture/full", "context_length": 131072,
                        "max_completion_tokens": 131072,
                        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                        "supported_parameters": ["max_completion_tokens", "reasoning"],
                    }]})
                return FakeResponse({"data": [{
                    "id": "reasoning-model",
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    "supported_parameters": ["max_completion_tokens", "reasoning"],
                    "reasoning": {"supported_efforts": ["low", "medium", "high"], "mandatory": False},
                }]})
            return FakeResponse({
                "id": "gen-reasoning", "model": "reasoning-model",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"cost": 0.0002, "prompt_tokens": 4, "completion_tokens": 20},
            })

        transport = OpenRouterTransport("sk-or-v1-test-key", opener=opener)
        result = transport.complete(
            "reasoning-model#reasoning=low",
            [{"role": "user", "content": "hello"}], max_tokens=100,
        )
        self.assertEqual(requests[-1]["reasoning"], {"effort": "low", "exclude": True})
        self.assertEqual(requests[-1]["max_completion_tokens"], 65536)
        self.assertEqual(result.model, "reasoning-model#reasoning=low")

    def test_mandatory_reasoning_catalog_never_emits_a_none_configuration(self):
        plan = select_role_plan([{
            "id": "openai/gpt-oss-120b",
            "intelligence": 70,
            "blended_price": 0.5,
            "supported_parameters": ["max_tokens", "reasoning"],
            "reasoning": {"mandatory": True, "supported_efforts": ["low", "medium", "high"]},
        }], maintenance_budget_usd=3)
        self.assertTrue(plan.target_models)
        self.assertTrue(all("#reasoning=none" not in model for model in plan.target_models))

    def test_optimizer_omits_known_unsupported_configuration_before_target_calls(self):
        class CapabilityTransport(FakeTransport):
            def __init__(self):
                super().__init__()
                self.target_models_seen = []

            def model_catalog(self):
                return []

            def configuration_support(self, configuration):
                if configuration.endswith("#reasoning=none"):
                    return {"supported": False, "reason": "mandatory reasoning"}
                return {"supported": True, "reason": "supported"}

            def complete(self, model, messages, *, max_tokens, response_schema=None):
                if model.startswith("target"):
                    self.target_models_seen.append(model)
                return super().complete(
                    model, messages, max_tokens=max_tokens, response_schema=response_schema,
                )

        transport = CapabilityTransport()
        result = Client(transport=transport).optimize(
            prompt="Return one label.", examples=EXAMPLES,
            models=["target#reasoning=none", "target#reasoning=low"],
            optimizer_model="optimizer", evaluator_model="evaluator",
            max_optimization_cost_usd=1,
        )
        self.assertTrue(transport.target_models_seen)
        self.assertTrue(all(model.endswith("#reasoning=low") for model in transport.target_models_seen))
        self.assertEqual(result.unavailable_models, [])
        self.assertEqual(result.omitted_configurations[0]["model"], "target#reasoning=none")
        self.assertEqual(result.omitted_configurations[0]["stage"], "preflight")

    def test_transport_clamps_reasoning_headroom_to_endpoint_and_context_limits(self):
        requests = []

        def opener(request, timeout):
            if request.data is not None:
                requests.append(json.loads(request.data))
                return FakeResponse({
                    "id": "gen-limit", "choices": [{"finish_reason": "length", "message": {"content": "partial"}}],
                    "usage": {"cost": 0.001, "prompt_tokens": 100, "completion_tokens": 2048},
                })
            if "endpoints/zdr" in request.full_url:
                return FakeResponse({"data": [{
                    "model_id": "bounded-reasoner", "context_length": 4096, "max_completion_tokens": 2048,
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    "supported_parameters": ["max_tokens", "reasoning"],
                }]})
            return FakeResponse({"data": [{
                "id": "bounded-reasoner", "context_length": 8192,
                "top_provider": {"context_length": 8192, "max_completion_tokens": 4096},
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "supported_parameters": ["max_tokens", "reasoning"],
                "reasoning": {"mandatory": True, "supported_efforts": ["low", "medium", "high"]},
            }]})

        transport = OpenRouterTransport("sk-or-v1-test-key", opener=opener)
        with self.assertRaises(ProviderError) as raised:
            transport.complete(
                "bounded-reasoner#reasoning=high",
                [{"role": "user", "content": "hello"}], max_tokens=100,
            )
        self.assertEqual(requests[-1]["max_tokens"], 2048)
        self.assertEqual(raised.exception.code, "PROVIDER_TRUNCATED")
        self.assertFalse(raised.exception.retry_with_more_tokens)

    def test_transport_selects_and_pins_the_cheapest_route_with_full_first_call_headroom(self):
        requests = []

        def opener(request, timeout):
            if request.data is not None:
                requests.append(json.loads(request.data))
                return FakeResponse({
                    "id": "gen-full-headroom", "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                    "usage": {"cost": 0.001, "prompt_tokens": 10, "completion_tokens": 20},
                })
            if "endpoints/zdr" in request.full_url:
                return FakeResponse({"data": [
                    {
                        "model_id": "reasoner", "tag": "cheap/capped", "context_length": 131072,
                        "max_completion_tokens": 32768, "pricing": {"prompt": "0.00000001", "completion": "0.00000001"},
                        "supported_parameters": ["max_completion_tokens", "reasoning"],
                    },
                    {
                        "model_id": "reasoner", "tag": "full/capacity", "context_length": 131072,
                        "max_completion_tokens": 131072, "pricing": {"prompt": "0.00000002", "completion": "0.00000002"},
                        "supported_parameters": ["max_completion_tokens", "reasoning"],
                    },
                ]})
            return FakeResponse({"data": [{
                "id": "reasoner", "context_length": 131072,
                "top_provider": {"context_length": 131072, "max_completion_tokens": 131072},
                "pricing": {"prompt": "0.00000001", "completion": "0.00000001"},
                "supported_parameters": ["max_completion_tokens", "reasoning"],
                "reasoning": {"mandatory": True, "supported_efforts": ["low", "medium", "high"]},
            }]})

        transport = OpenRouterTransport("sk-or-v1-test-key", opener=opener)
        transport.complete(
            "reasoner#reasoning=high", [{"role": "user", "content": "hello"}], max_tokens=100,
        )
        sent = requests[-1]
        self.assertEqual(sent["provider"]["only"], ["full/capacity"])
        self.assertFalse(sent["provider"]["allow_fallbacks"])
        self.assertGreater(sent["max_completion_tokens"], 100000)

    def test_missing_visible_content_at_length_is_typed_as_truncation(self):
        def opener(request, timeout):
            if request.data is None:
                if "endpoints/zdr" in request.full_url:
                    return FakeResponse({"data": [{
                        "model_id": "reasoner", "tag": "fixture/full", "context_length": 131072,
                        "max_completion_tokens": 131072,
                        "pricing": {"prompt": "0.00000001", "completion": "0.00000001"},
                        "supported_parameters": ["max_completion_tokens", "reasoning"],
                    }]})
                return FakeResponse({"data": [{
                    "id": "reasoner", "context_length": 131072,
                    "top_provider": {"context_length": 131072, "max_completion_tokens": 131072},
                    "pricing": {"prompt": "0.00000001", "completion": "0.00000001"},
                    "supported_parameters": ["max_completion_tokens", "reasoning"],
                    "reasoning": {"mandatory": True, "supported_efforts": ["high"]},
                }]})
            return FakeResponse({
                "id": "gen-no-visible-answer",
                "choices": [{"finish_reason": "length", "message": {"reasoning": "hidden"}}],
                "usage": {"cost": 0.001, "prompt_tokens": 10, "completion_tokens": 131000},
            })

        transport = OpenRouterTransport("sk-or-v1-test-key", opener=opener)
        with self.assertRaises(ProviderError) as raised:
            transport.complete(
                "reasoner#reasoning=high", [{"role": "user", "content": "hello"}], max_tokens=100,
            )
        self.assertEqual(raised.exception.code, "PROVIDER_TRUNCATED")
        self.assertFalse(raised.exception.retry_with_more_tokens)

    def test_empty_answer_retries_once_with_a_larger_budgeted_response_target(self):
        class EmptyOnceTransport(FakeTransport):
            def __init__(self):
                super().__init__()
                self.max_tokens_seen = []

            def complete(self, model, messages, *, max_tokens, response_schema=None):
                self.max_tokens_seen.append(max_tokens)
                if len(self.max_tokens_seen) == 1:
                    error = ProviderError("empty")
                    error.code = "PROVIDER_EMPTY"
                    error.cost_usd = 0.001
                    raise error
                return Completion("complete answer", model, "gen-retry", 0.002)

        transport = EmptyOnceTransport()
        answer = Client(transport=transport).draft_answer(
            task="Answer completely.", input="Test", max_cost_usd=0.10,
        )
        self.assertEqual(answer.answer, "complete answer")
        self.assertEqual(transport.max_tokens_seen, [8192, 131072])

    def test_truncated_answer_gets_one_genuinely_larger_budgeted_retry(self):
        class TruncatedTransport(FakeTransport):
            def __init__(self):
                super().__init__()
                self.max_tokens_seen = []

            def complete(self, model, messages, *, max_tokens, response_schema=None):
                self.max_tokens_seen.append(max_tokens)
                if len(self.max_tokens_seen) == 1:
                    error = ProviderError("response limit")
                    error.code = "PROVIDER_TRUNCATED"
                    error.cost_usd = 0.001
                    raise error
                return super().complete(
                    model, messages, max_tokens=max_tokens, response_schema=response_schema,
                )

        transport = TruncatedTransport()
        result = Client(transport=transport).optimize(
            prompt="Return one label.", examples=EXAMPLES, models=["target"],
            optimizer_model="optimizer", evaluator_model="evaluator",
            max_optimization_cost_usd=1,
        )
        self.assertEqual(result.winner.model, "target")
        self.assertEqual(transport.max_tokens_seen[:2], [8192, 131072])

    def test_repeated_truncation_stops_after_one_expansion(self):
        class AlwaysTruncatedTransport(FakeTransport):
            def __init__(self):
                super().__init__()
                self.max_tokens_seen = []

            def complete(self, model, messages, *, max_tokens, response_schema=None):
                self.max_tokens_seen.append(max_tokens)
                error = ProviderError("response limit")
                error.code = "PROVIDER_TRUNCATED"
                raise error

        transport = AlwaysTruncatedTransport()
        with self.assertRaises(ProviderError):
            Client(transport=transport).optimize(
                prompt="Return one label.", examples=EXAMPLES, models=["target"],
                optimizer_model="optimizer", evaluator_model="evaluator",
                max_optimization_cost_usd=1,
            )
        self.assertEqual(transport.max_tokens_seen, [8192, 131072])

    def test_reasoning_effort_fails_closed_when_the_current_zdr_endpoint_cannot_honor_it(self):
        def opener(request, timeout):
            if "endpoints/zdr" in request.full_url:
                return FakeResponse({"data": [{
                    "model_id": "catalog-reasoning-only",
                    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    "supported_parameters": ["max_completion_tokens"],
                }]})
            return FakeResponse({"data": [{
                "id": "catalog-reasoning-only",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "supported_parameters": ["max_completion_tokens", "reasoning"],
                "reasoning": {"supported_efforts": ["low", "high"], "mandatory": False},
                "benchmarks": {"artificial_analysis": {"intelligence_index": 50}},
            }]})

        transport = OpenRouterTransport("sk-or-v1-test-key", opener=opener)
        with self.assertRaisesRegex(ProviderError, "current ZDR endpoint"):
            transport.complete(
                "catalog-reasoning-only#reasoning=high",
                [{"role": "user", "content": "hello"}], max_tokens=100,
            )
        plan = select_role_plan(transport.model_catalog(), maintenance_budget_usd=0.25)
        self.assertEqual(plan.target_models, ("catalog-reasoning-only#reasoning=none",))

    def test_adaptive_search_broadens_first_then_prunes_effort_variants_outside_the_capability_band(self):
        class AdaptiveTransport(FakeTransport):
            @staticmethod
            def base(model):
                return model.split("#reasoning=", 1)[0]

            def estimate_cost(self, model, messages, *, max_tokens):
                base = {"cheap": 0.0001, "near": 0.001, "far": 0.005}.get(self.base(model), 0.0001)
                return base * (2 if model.endswith("#reasoning=high") else 1)

            def complete(self, model, messages, *, max_tokens, response_schema=None):
                if response_schema:
                    return super().complete(model, messages, max_tokens=max_tokens, response_schema=response_schema)
                self.calls.append((model, messages, max_tokens, response_schema))
                user = messages[-1]["content"].lower()
                if self.base(model) in {"cheap", "far"}:
                    content = "wrong"
                else:
                    content = "billing" if "charged" in user or "refund" in user else "account" if "reset" in user else "technical"
                return Completion(content, model, f"gen-{len(self.calls)}", self.estimate_cost(model, messages, max_tokens=max_tokens))

        result = Client(transport=AdaptiveTransport()).optimize(
            prompt="Return the approved route label only.", examples=EXAMPLES,
            models=[
                "cheap#reasoning=low", "near#reasoning=low", "far#reasoning=low",
                "cheap#reasoning=high", "near#reasoning=high", "far#reasoning=high",
            ],
            optimizer_model="optimizer", evaluator_model="evaluator",
            max_optimization_cost_usd=2, adaptive_search=True,
        )
        self.assertIn("far#reasoning=high", result.pruned_models)
        self.assertNotIn("cheap#reasoning=high", result.pruned_models)
        self.assertEqual(result.winner.model, "near#reasoning=low")
        self.assertIn("adaptive search band", result.winner_scope)

    def test_adaptive_cheapest_passing_stops_before_an_expensive_rescue_wave(self):
        transport = FakeTransport()
        passing_examples = [
            {"id": f"billing-{index}", "input": f"charged twice {index}", "approved_output": "billing"}
            for index in range(5)
        ]
        result = Client(transport=transport).optimize(
            prompt="Return the approved route label only.", examples=passing_examples,
            models=["cheap-a", "cheap-b", "cheap-c", "expensive"],
            optimizer_model="optimizer", evaluator_model="evaluator",
            max_optimization_cost_usd=2, adaptive_search=True,
        )
        self.assertIn("expensive", result.pruned_models)
        target_models = [model for model, _messages, _tokens, schema in transport.calls if schema is None]
        self.assertNotIn("expensive", target_models)

    def test_draft_becomes_approved_or_corrected_example(self):
        client = Client(transport=FakeTransport())
        draft = client.draft_answer(task="Classify support", input="I was charged twice")
        self.assertEqual(draft.approve().approved_output, "Here is a verbose answer")
        self.assertEqual(draft.correct("billing").approved_output, "billing")

    def test_cheapest_passing_selects_cost_not_model_order(self):
        client = Client(transport=FakeTransport())
        result = client.optimize(
            prompt="Write a helpful classification for this message.",
            examples=EXAMPLES,
            models=["expensive", "cheap"],
            optimizer_model="optimizer",
            evaluator_model="evaluator",
            quality_threshold=0.9,
            max_optimization_cost_usd=1,
        )
        self.assertEqual(result.winner.model, "cheap")
        self.assertEqual(result.winner.selected_prompt, "Return the approved route label only.")
        self.assertEqual(result.winner.holdout_pass_rate, 1)
        self.assertTrue(result.exploratory)
        self.assertEqual(result.winner.holdout_unique_scenarios, 1)
        self.assertEqual(result.winner.holdout_executions, 2)
        self.assertGreater(result.total_provider_spend_usd, 0)
        self.assertEqual(result.quality_frontier[0]["model"], "cheap")
        self.assertEqual(
            result.diminishing_returns["higher_cost_models_without_material_gain"][0]["model"],
            "expensive",
        )
        self.assertFalse(result.regression_suite["watch"]["enabled"])
        self.assertRegex(result.regression_suite["suite_hash"], r"^[0-9a-f]{64}$")
        self.assertTrue(result.comparison_integrity["single_frozen_run"])
        self.assertEqual(result.comparison_integrity["distinct_final_test_scenarios"], 1)
        self.assertIn("not a general model-intelligence ranking", result.comparison_integrity["claim_scope"])

    def test_price_first_objective_maximizes_accuracy_inside_the_price_ceiling(self):
        client = Client(transport=PriceFrontierTransport())
        roomy = client.optimize(
            prompt="Return the approved route label only.", examples=EXAMPLES,
            models=["cheap", "expensive"], optimizer_model="optimizer",
            evaluator_model="evaluator", objective="best_within_price",
            max_cost_per_run_usd=0.002, max_optimization_cost_usd=1,
        )
        self.assertEqual(roomy.winner.model, "expensive")
        constrained = client.optimize(
            prompt="Return the approved route label only.", examples=EXAMPLES,
            models=["cheap", "expensive"], optimizer_model="optimizer",
            evaluator_model="evaluator", objective="best_within_price",
            max_cost_per_run_usd=0.0005, max_optimization_cost_usd=1,
        )
        self.assertEqual(constrained.winner.model, "cheap")

    def test_optional_incumbent_objective_matches_baseline_then_minimizes_cost(self):
        result = Client(transport=PriceFrontierTransport()).optimize(
            prompt="Return the approved route label only.", examples=EXAMPLES,
            models=["expensive", "cheap", "cheap-good"], incumbent_model="expensive",
            optimizer_model="optimizer", evaluator_model="evaluator",
            objective="match_baseline_at_lowest_cost", max_optimization_cost_usd=1,
        )
        self.assertEqual(result.winner.model, "cheap-good")
        self.assertEqual(result.winner.holdout_pass_rate, result.regression_suite["incumbent_baseline_holdout_pass_rate"])

    def test_small_suite_is_labeled_exploratory(self):
        client = Client(transport=FakeTransport())
        result = client.optimize(
            prompt="Write a helpful classification for this message.",
            examples=EXAMPLES[:3],
            models=["cheap"],
            optimizer_model="optimizer",
            evaluator_model="evaluator",
            max_optimization_cost_usd=1,
        )
        self.assertTrue(result.exploratory)
        self.assertRegex(result.warnings[0], "distinct final-test")

    def test_unavailable_target_does_not_abort_other_selected_models(self):
        client = Client(transport=PartlyUnavailableTransport())
        result = client.optimize(
            prompt="Write a helpful classification for this message.",
            examples=EXAMPLES,
            models=["unavailable", "cheap"],
            optimizer_model="optimizer",
            evaluator_model="evaluator",
            max_optimization_cost_usd=1,
        )
        self.assertEqual(result.winner.model, "cheap")
        self.assertEqual(result.unavailable_models[0]["model"], "unavailable")
        self.assertRegex(result.warnings[-1], "1 selected model")

    def test_budget_fails_before_unapproved_call(self):
        transport = FakeTransport()
        client = Client(transport=transport)
        with self.assertRaises(BudgetExceeded):
            client.optimize(
                prompt="Write a helpful classification for this message.",
                examples=EXAMPLES,
                models=["cheap"],
                optimizer_model="optimizer",
                evaluator_model="evaluator",
                max_optimization_cost_usd=0.00005,
            )
        self.assertEqual(transport.calls, [])

    def test_at_least_three_examples_required(self):
        client = Client(transport=FakeTransport())
        with self.assertRaisesRegex(ValueError, "At least three"):
            client.optimize(
                prompt="Write a helpful classification for this message.",
                examples=EXAMPLES[:2],
                models=["cheap"],
            )

    def test_more_than_five_models_are_allowed_when_one_shared_budget_is_explicit(self):
        client = Client(transport=FakeTransport())
        result = client.optimize(
            prompt="Write a helpful classification for this message.", examples=EXAMPLES,
            models=[f"cheap-{index}" for index in range(6)], optimizer_model="optimizer",
            evaluator_model="evaluator", max_optimization_cost_usd=1,
        )
        self.assertEqual(len(result.models), 6)
        self.assertEqual(result.winner_scope, "Best among every requested target")

    def test_budget_exhaustion_returns_partial_coverage_after_one_model_finishes(self):
        client = Client(transport=FakeTransport())
        result = client.optimize(
            prompt="Write a helpful classification for this message.", examples=EXAMPLES,
            models=["cheap", "expensive", "later"], optimizer_model="optimizer",
            evaluator_model="evaluator", max_optimization_cost_usd=0.0025,
            max_parallel_models=1, max_parallel_scenarios=1,
        )
        self.assertEqual(result.winner.model, "cheap")
        self.assertTrue(result.incomplete_models)
        self.assertEqual(result.skipped_budget_models, ["later"])
        self.assertEqual(result.winner_scope, "Best among fully completed eligible targets only")

    def test_few_shot_selection_is_training_only_and_is_included_in_costed_package(self):
        transport = FewShotTransport()
        result = Client(transport=transport).optimize(
            prompt="Write a helpful classification for this message.", examples=EXAMPLES,
            models=["cheap"], optimizer_model="optimizer", evaluator_model="evaluator",
            max_optimization_cost_usd=1, max_few_shot_examples=1,
        )
        self.assertEqual(len(result.winner.few_shot_example_ids), 1)
        suite_ids = {item["id"] for item in result.regression_suite["examples"]}
        self.assertTrue(set(result.winner.few_shot_example_ids) <= suite_ids)
        self.assertEqual(result.regression_suite["winning_few_shot_example_ids"], result.winner.few_shot_example_ids)

    def test_multi_turn_scenarios_replay_prior_assistant_context(self):
        scenarios = [
            {"id": f"scenario-{index}", "turns": [
                {"input": f"Remember code {index}", "approved_output": f"stored {index}"},
                {"input": "What code?", "approved_output": str(index)},
            ]}
            for index in range(5)
        ]
        transport = FakeTransport()
        Client(transport=transport).optimize(
            prompt="Remember the code and answer the later question.", examples=scenarios,
            models=["cheap"], optimizer_model="optimizer", evaluator_model="evaluator",
            max_optimization_cost_usd=1, rounds=1,
        )
        target_calls = [messages for _model, messages, _tokens, schema in transport.calls if schema is None]
        self.assertTrue(any(len(messages) >= 4 and any(item["role"] == "assistant" for item in messages[:-1]) for messages in target_calls))


if __name__ == "__main__":
    unittest.main()
