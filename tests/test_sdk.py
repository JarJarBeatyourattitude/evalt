import hashlib
from io import BytesIO
import json
import os
import sqlite3
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest import mock
from urllib.error import HTTPError

from evalt import BudgetExceeded, Client, Evalt, ProviderError, Suite, check_result, compare_results, render_comparison_html, render_html_report, render_junit_report, select_role_plan
from evalt.cli import STARTER_SUITE, main as cli_main, parser as cli_parser
from evalt.acceptance import AcceptanceFailure, redact_trace, validate_auto_first_route_receipt
from evalt.core import Completion, OpenRouterTransport, _safe_provider_error_detail
from evalt.migration import migrate_openai_results
from modelsieve import Client as ModelSieveClient
from last_good_prompt import Client as LegacyClient
from last_good_prompt.core import _Budget, ModelResult


class CompatibilityTests(unittest.TestCase):
    def test_earlier_imports_resolve_to_evalt_client(self):
        self.assertIs(LegacyClient, Client)
        self.assertIs(ModelSieveClient, Client)


class AutomaticFirstRouteAcceptanceTests(unittest.TestCase):
    def receipt(self):
        summary = {
            "final_test_scenarios": 5,
            "holdout_pass_rate": 1.0,
            "tested_configurations": 3,
            "prompt_candidates_tested": 6,
            "prompt_rewrites_tested": 3,
            "evidence_provenance": "AI_GENERATED_AI_JUDGED",
            "judge_calibrated": True,
            "workflow_spend_usd": 0.75,
            "winner_model": "cheap-a#reasoning=low",
        }
        return {
            "target_accuracy": 0.95,
            "test_budget_usd": 1.0,
            "first_answer": {
                "route_phase": "ai_tested",
                "initial_test_summary": summary,
            },
            "second_answer": {"route_phase": "ai_tested"},
            "route_status": {"route_phase": "ai_tested"},
            "first_call_events": [
                {"event": "initial_optimization_started"},
                {"event": "suite_design_started"},
                {
                    "event": "suite_design_completed",
                    "case_count": 25,
                    "judge_calibrated": True,
                    "judge_calibration_checks": 4,
                },
                {
                    "event": "prompt_candidate_completed",
                    "model": "cheap-a#reasoning=low",
                    "kind": "starting_prompt",
                },
                {
                    "event": "prompt_candidate_completed",
                    "model": "cheap-a#reasoning=low",
                    "kind": "rewrite",
                },
                {"event": "model_completed", "model": "cheap-a#reasoning=low"},
                {"event": "model_completed", "model": "cheap-b#reasoning=low"},
                {"event": "model_completed", "model": "cheap-b#reasoning=high"},
                {"event": "initial_optimization_completed"},
                {"event": "production_call_completed"},
            ],
            "second_call_events": [{"event": "production_call_completed"}],
        }

    def test_acceptance_requires_the_observable_full_tournament_and_reuse(self):
        report = validate_auto_first_route_receipt(self.receipt())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["settled_configurations"], 3)
        self.assertTrue(report["route_reused"])

    def test_acceptance_rejects_the_old_instant_bootstrap_path(self):
        receipt = self.receipt()
        receipt["first_answer"]["route_phase"] = "untested_bootstrap"
        receipt["first_call_events"] = [{"event": "production_call_completed"}]
        with self.assertRaisesRegex(AcceptanceFailure, "missing first-call event"):
            validate_auto_first_route_receipt(receipt)

    def test_acceptance_rejects_a_second_call_that_restarts_design(self):
        receipt = self.receipt()
        receipt["second_call_events"].insert(0, {"event": "suite_design_started"})
        with self.assertRaisesRegex(AcceptanceFailure, "second call incorrectly"):
            validate_auto_first_route_receipt(receipt)

    def test_trace_redaction_removes_secret_keys_and_values(self):
        redacted = redact_trace(
            {"authorization": "Bearer secret-value", "error": "secret-value failed"},
            ("secret-value",),
        )
        self.assertNotIn("authorization", redacted)
        self.assertEqual(redacted["error"], "[REDACTED] failed")


class PortableReportTests(unittest.TestCase):
    def fixture(self):
        case = {
            "example_id": "final-1", "split": "holdout", "difficulty": "complex",
            "passed": False, "reason": "wrong label", "output": "billing <unsafe>",
            "approved_output": "technical", "target_latency_ms": 1250,
        }
        winner = {
            "model": "fixture/cheap", "holdout_pass_rate": 0.96,
            "estimated_cost_per_successful_call_usd": 0.0002, "cases": [case],
        }
        return {
            "quality_threshold": 0.95, "winner": winner, "models": [winner],
            "total_provider_spend_usd": 0.12, "winner_scope": "All requested targets",
            "regression_suite": {"suite_hash": "abc123"}, "elapsed_seconds": 3.5,
        }

    def test_html_report_is_standalone_and_escapes_model_outputs(self):
        html = render_html_report(self.fixture(), title="Fixture report")
        self.assertIn("<!doctype html>", html)
        self.assertIn("fixture/cheap", html)
        self.assertIn("billing &lt;unsafe&gt;", html)
        self.assertNotIn("billing <unsafe>", html)

    def test_junit_report_preserves_case_failure_and_route_metadata(self):
        junit = render_junit_report(self.fixture(), suite_name="fixture-route")
        self.assertIn('tests="1"', junit)
        self.assertIn('failures="1"', junit)
        self.assertIn('name="winner_model" value="fixture/cheap"', junit)
        self.assertIn("wrong label", junit)

    def test_cli_report_writes_html_and_junit_without_provider_access(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.json"
            html = root / "report.html"
            junit = root / "report.xml"
            result.write_text(json.dumps(self.fixture()), encoding="utf-8")
            with redirect_stdout(StringIO()):
                code = cli_main(["report", str(result), "--html", str(html), "--junit", str(junit)])
            self.assertEqual(code, 0)
            self.assertTrue(html.exists())
            self.assertTrue(junit.exists())

    def test_comparison_reports_case_regressions_and_cost_delta(self):
        baseline = self.fixture()
        candidate = json.loads(json.dumps(baseline))
        candidate["winner"]["model"] = "fixture/new"
        candidate["winner"]["holdout_pass_rate"] = 1.0
        candidate["winner"]["estimated_cost_per_successful_call_usd"] = 0.0001
        candidate["winner"]["cases"][0]["passed"] = True
        candidate["winner"]["cases"][0]["output"] = "technical"
        comparison = compare_results(baseline, candidate)
        self.assertTrue(comparison["comparable_contract"])
        self.assertEqual(comparison["case_summary"]["improvements"], 1)
        self.assertEqual(comparison["case_summary"]["regressions"], 0)
        self.assertAlmostEqual(
            comparison["delta"]["cost_per_1k_successful_calls_usd"], -0.1
        )

    def test_comparison_refuses_to_imply_a_shared_gate_when_hashes_differ(self):
        baseline = self.fixture()
        candidate = json.loads(json.dumps(baseline))
        candidate["regression_suite"]["suite_hash"] = "different"
        comparison = compare_results(baseline, candidate)
        self.assertFalse(comparison["comparable_contract"])
        self.assertIn("must not be used as a promotion gate", comparison["contract"]["warning"])

    def test_cli_compare_writes_offline_json_and_escaped_html(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            output = root / "comparison.json"
            html = root / "comparison.html"
            baseline.write_text(json.dumps(self.fixture()), encoding="utf-8")
            candidate_payload = self.fixture()
            candidate_payload["winner"]["model"] = "new/<unsafe>"
            candidate.write_text(json.dumps(candidate_payload), encoding="utf-8")
            with redirect_stdout(StringIO()):
                code = cli_main([
                    "compare", str(baseline), str(candidate),
                    "--output", str(output), "--html", str(html),
                ])
            self.assertEqual(code, 0)
            self.assertTrue(output.exists())
            rendered = html.read_text(encoding="utf-8")
            self.assertIn("new/&lt;unsafe&gt;", rendered)
            self.assertNotIn("new/<unsafe>", rendered)


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


class CostlyBootstrapTransport(FakeTransport):
    def estimate_cost(self, model, messages, *, max_tokens):
        return 0.057045


class CaseDesignerTransport(FakeTransport):
    def complete(self, model, messages, *, max_tokens, response_schema=None):
        if response_schema and "Design a balanced evaluation suite" in messages[0]["content"]:
            self.calls.append((model, messages, max_tokens, response_schema))
            scenarios = []
            for index in range(25):
                category = index % 3
                if category == 0:
                    input_text, approved = f"I was charged twice, case {index}", "billing"
                elif category == 1:
                    input_text, approved = f"My reset link expired, case {index}", "account"
                else:
                    input_text, approved = f"The app freezes, case {index}", "technical"
                scenarios.append({
                    "id": f"designed-{index + 1}",
                    "group": f"family-{index % 5 + 1}",
                    "difficulty": ("routine", "complex", "adversarial")[category],
                    "critical": category == 2,
                    "turns": [{"input": input_text, "approved_output": approved}],
                    "rationale": "Exercise a distinct support-routing boundary.",
                })
            design_payload = {
                "evaluator": {
                    "type": "exact_text",
                    "reason": "The production contract requires one exact lowercase label.",
                    "required_keys": [],
                    "allow_additional_properties": True,
                    "normalize_rational_strings": False,
                },
                "judge_calibration": [
                    {
                        "input": "I was charged twice",
                        "approved_output": "billing",
                        "candidate_output": "billing",
                        "should_pass": True,
                    },
                    {
                        "input": "My reset link expired",
                        "approved_output": "account",
                        "candidate_output": "account",
                        "should_pass": True,
                    },
                    {
                        "input": "The app freezes",
                        "approved_output": "technical",
                        "candidate_output": "billing",
                        "should_pass": False,
                    },
                ],
                "design_notes": ["Balanced three routing labels before splitting."],
            }
            if "strata" in response_schema.get("properties", {}):
                design_payload["strata"] = [
                    {
                        "group": f"family-{group_index + 1}",
                        "scenarios": [
                            {
                                key: value for key, value in scenario.items()
                                if key != "group"
                            }
                            for scenario in scenarios
                            if scenario["group"] == f"family-{group_index + 1}"
                        ],
                    }
                    for group_index in range(5)
                ]
            else:
                design_payload["scenarios"] = scenarios
            content = json.dumps(design_payload)
            return Completion(
                content, model, f"gen-{len(self.calls)}",
                self.estimate_cost(model, messages, max_tokens=max_tokens),
            )
        return super().complete(
            model, messages, max_tokens=max_tokens, response_schema=response_schema
        )


class CaseDesignerFewShotTransport(CaseDesignerTransport):
    def complete(self, model, messages, *, max_tokens, response_schema=None):
        if response_schema and "Improve the current prompt" in messages[0]["content"]:
            self.calls.append((model, messages, max_tokens, response_schema))
            payload = json.loads(messages[-1]["content"])
            chosen = payload["allowed_few_shot_example_ids"][:1]
            return Completion(
                json.dumps({
                    "prompt": "Return the approved route label only.",
                    "hypothesis": "Constrain the output and include one training-only demonstration.",
                    "few_shot_example_ids": chosen,
                }),
                model,
                f"gen-{len(self.calls)}",
                self.estimate_cost(model, messages, max_tokens=max_tokens),
            )
        return super().complete(
            model, messages, max_tokens=max_tokens, response_schema=response_schema
        )


class DesignerFallbackTransport(CaseDesignerTransport):
    def model_catalog(self):
        return [
            {
                "id": "primary-designer",
                "intelligence": 100,
                "blended_price": 1.0,
                "supported_parameters": ["max_tokens"],
            },
            {
                "id": "secondary-designer",
                "intelligence": 90,
                "blended_price": 0.2,
                "supported_parameters": ["max_tokens"],
            },
            {
                "id": "cheap-target",
                "intelligence": 65,
                "blended_price": 0.01,
                "supported_parameters": ["max_tokens"],
            },
        ]

    def complete(self, model, messages, *, max_tokens, response_schema=None):
        if (
            model == "primary-designer"
            and response_schema
            and "Design a balanced evaluation suite" in messages[0]["content"]
        ):
            raise ProviderError("primary designer timed out")
        return super().complete(
            model, messages, max_tokens=max_tokens, response_schema=response_schema
        )


class MalformedThenValidDesignerTransport(CaseDesignerTransport):
    def __init__(self):
        super().__init__()
        self.design_attempts = 0

    def complete(self, model, messages, *, max_tokens, response_schema=None):
        if response_schema and "Design a balanced evaluation suite" in messages[0]["content"]:
            self.design_attempts += 1
            if self.design_attempts == 1:
                self.calls.append((model, messages, max_tokens, response_schema))
                return Completion(
                    '{"evaluator":{"type":"exact_text",',
                    model,
                    f"gen-{len(self.calls)}",
                    self.estimate_cost(model, messages, max_tokens=max_tokens),
                )
        return super().complete(
            model, messages, max_tokens=max_tokens, response_schema=response_schema
        )


class SemanticMiscalibratedDesignerTransport(CaseDesignerTransport):
    def complete(self, model, messages, *, max_tokens, response_schema=None):
        if response_schema and "Design a balanced evaluation suite" in messages[0]["content"]:
            completion = super().complete(
                model, messages, max_tokens=max_tokens, response_schema=response_schema
            )
            payload = json.loads(completion.content)
            payload["evaluator"]["type"] = "semantic"
            payload["evaluator"]["reason"] = "Equivalent sentiment scores may vary slightly."
            payload["judge_calibration"][0]["candidate_output"] = "8"
            payload["judge_calibration"][0]["approved_output"] = "5"
            payload["judge_calibration"][1]["candidate_output"] = "84"
            payload["judge_calibration"][1]["approved_output"] = "80"
            return Completion(
                json.dumps(payload),
                completion.model,
                completion.generation_id,
                completion.cost_usd,
            )
        return super().complete(
            model, messages, max_tokens=max_tokens, response_schema=response_schema
        )


class FailingCaseDesignerTransport(CaseDesignerTransport):
    def complete(self, model, messages, *, max_tokens, response_schema=None):
        if response_schema:
            return super().complete(
                model, messages, max_tokens=max_tokens, response_schema=response_schema
            )
        self.calls.append((model, messages, max_tokens, response_schema))
        return Completion(
            "wrong",
            model,
            f"gen-{len(self.calls)}",
            self.estimate_cost(model, messages, max_tokens=max_tokens),
        )


class AlwaysPassJudgeDesignerTransport(CaseDesignerTransport):
    """Known-invalid semantic judge used to prove calibration fails closed."""

    def complete(self, model, messages, *, max_tokens, response_schema=None):
        system = messages[0]["content"]
        if response_schema and "Design a balanced evaluation suite" in system:
            completion = super().complete(
                model, messages, max_tokens=max_tokens, response_schema=response_schema
            )
            payload = json.loads(completion.content)
            payload["evaluator"]["type"] = "semantic"
            return Completion(
                json.dumps(payload),
                completion.model,
                completion.generation_id,
                completion.cost_usd,
            )
        if response_schema and "Judge whether" in system:
            self.calls.append((model, messages, max_tokens, response_schema))
            return Completion(
                json.dumps({"passed": True, "score": 1, "reason": "always passes"}),
                model,
                f"gen-{len(self.calls)}",
                self.estimate_cost(model, messages, max_tokens=max_tokens),
            )
        return super().complete(
            model, messages, max_tokens=max_tokens, response_schema=response_schema
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
    def test_parallel_budget_reservations_wait_instead_of_creating_a_false_failure(self):
        budget = _Budget(0.10)
        first_reserved = threading.Event()
        release_first = threading.Event()
        second_authorized = threading.Event()

        def first_call():
            budget.authorize(0.06)
            first_reserved.set()
            release_first.wait(timeout=1)
            budget.commit(0.01, 0.06)

        def second_call():
            first_reserved.wait(timeout=1)
            budget.authorize(0.06)
            second_authorized.set()
            budget.commit(0.01, 0.06)

        first = threading.Thread(target=first_call)
        second = threading.Thread(target=second_call)
        first.start()
        second.start()
        self.assertTrue(first_reserved.wait(timeout=1))
        self.assertFalse(second_authorized.wait(timeout=0.05))
        release_first.set()
        first.join(timeout=1)
        second.join(timeout=1)
        self.assertTrue(second_authorized.is_set())
        self.assertAlmostEqual(budget.spent_usd, 0.02)
        self.assertAlmostEqual(budget.reserved_usd, 0.0)

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
        prompt_events = [
            event for event in events if event["event"] == "prompt_candidate_completed"
        ]
        self.assertEqual(
            {event["model"] for event in prompt_events}, {"cheap-a", "cheap-b"}
        )
        self.assertTrue(any(event["kind"] == "rewrite" for event in prompt_events))
        self.assertTrue(all("prompt_hash" in event for event in prompt_events))
        self.assertTrue(all(item.prompt_rewrites_tested >= 1 for item in result.models))

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
                if model == "optimizer":
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
                if model == "optimizer":
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

    def test_perfect_training_and_validation_still_measure_one_prompt_rewrite(self):
        class AlreadyCorrectTransport(FakeTransport):
            def complete(self, model, messages, *, max_tokens, response_schema=None):
                self.calls.append((model, messages, max_tokens, response_schema))
                if response_schema and "Improve the current prompt" in messages[0]["content"]:
                    return Completion(
                        json.dumps({
                            "prompt": "Return exactly the lowercase label billing.",
                            "hypothesis": "Make the already observed output contract explicit.",
                            "few_shot_example_ids": [],
                        }),
                        model,
                        f"gen-{len(self.calls)}",
                        0.0001,
                    )
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
        self.assertIn("optimizer", called_models)
        self.assertEqual(result.winner.prompt_rewrites_tested, 1)
        self.assertEqual(result.winner.prompt_candidates_tested, 2)
        self.assertTrue(any(case.split == "train" for case in result.winner.cases))
        self.assertTrue(all(max_tokens == 64 for model, _messages, max_tokens, _schema in transport.calls if model == "target"))
        self.assertEqual(result.winner.holdout_pass_rate, 1)

    def test_perfect_small_validation_does_not_hide_training_failures(self):
        prompt = "Apply the private policy and return approved or rejected."
        examples = [
            {"id": f"policy-{index}", "input": f"case {index}", "approved_output": "approved"}
            for index in range(10)
        ]
        ranked = sorted(
            examples,
            key=lambda item: hashlib.sha256(f"{prompt}:{item['id']}".encode()).hexdigest(),
        )
        dev_ids = {item["id"] for item in ranked[2:4]}

        class ValidationFlukeTransport(FakeTransport):
            def complete(self, model, messages, *, max_tokens, response_schema=None):
                self.calls.append((model, messages, max_tokens, response_schema))
                system = messages[0]["content"]
                user = messages[-1]["content"]
                if response_schema and "Improve the current prompt" in system:
                    return Completion(
                        json.dumps({
                            "prompt": "Rewritten policy: always return approved.",
                            "hypothesis": "Encode the rule demonstrated across training evidence.",
                            "few_shot_example_ids": [],
                        }),
                        model, f"gen-{len(self.calls)}", 0.0001,
                    )
                case_id = user.removeprefix("case ")
                original_is_lucky = f"policy-{case_id}" in dev_ids
                content = "approved" if "Rewritten policy" in system or original_is_lucky else "rejected"
                return Completion(content, model, f"gen-{len(self.calls)}", 0.0001)

        result = Client(transport=ValidationFlukeTransport()).optimize(
            prompt=prompt,
            examples=examples,
            models=["target"],
            optimizer_model="optimizer",
            evaluator_model="unused",
            evaluator={"type": "exact_text"},
            rounds=1,
            max_optimization_cost_usd=1,
            holdout_repeats=1,
        )

        self.assertEqual(result.winner.baseline_pass_rate, 0)
        self.assertEqual(result.winner.selected_pass_rate, 1)
        self.assertEqual(result.winner.baseline_holdout_pass_rate, 0)
        self.assertEqual(result.winner.holdout_pass_rate, 1)
        self.assertEqual(result.winner.selected_prompt, "Rewritten policy: always return approved.")
        self.assertEqual(result.winner.prompt_origin, "optimized_for:target")
        self.assertEqual(result.quality_gate_status, "QUALIFIED_ROUTE_SELECTED")

    def test_prompt_optimization_can_be_disabled_without_disabling_model_evaluation(self):
        transport = FakeTransport()
        supplied_prompt = "Keep this exact production prompt unchanged."
        result = Client(transport=transport).optimize(
            prompt=supplied_prompt,
            examples=EXAMPLES,
            models=["cheap", "expensive"],
            optimizer_model="optimizer",
            evaluator_model="evaluator",
            optimize_prompt=False,
            allow_few_shot=True,
            max_optimization_cost_usd=1,
        )
        self.assertTrue(result.models)
        self.assertTrue(all(item.selected_prompt == supplied_prompt for item in result.models))
        self.assertTrue(all(item.few_shot_example_ids == [] for item in result.models))
        self.assertFalse(result.regression_suite["optimize_prompt"])
        self.assertFalse(
            result.comparison_integrity["selection_protocol"]["prompt_modification_enabled"]
        )
        self.assertEqual(result.quality_gate_status, "NO_CONFIGURATION_PASSED")
        called_models = [model for model, *_rest in transport.calls]
        self.assertNotIn("optimizer", called_models)

    def test_stratified_groups_reach_every_split_and_hard_floor_blocks_promotion(self):
        class DifficultyTransport(FakeTransport):
            def complete(self, model, messages, *, max_tokens, response_schema=None):
                self.calls.append((model, messages, max_tokens, response_schema))
                if response_schema and "Improve the current prompt" in messages[0]["content"]:
                    return Completion(
                        json.dumps({
                            "prompt": "Return approved for routine cases and rejected for hard cases.",
                            "hypothesis": "Preserve the observed difficulty behavior.",
                            "few_shot_example_ids": [],
                        }),
                        model, f"gen-{len(self.calls)}", 0.0001,
                    )
                content = "rejected" if "hard" in messages[-1]["content"] else "approved"
                return Completion(content, model, f"gen-{len(self.calls)}", 0.0001)

        examples = [
            {
                "id": f"routine-{index}", "group": "routine-policy",
                "difficulty": "routine", "input": f"routine case {index}",
                "approved_output": "approved",
            }
            for index in range(5)
        ] + [
            {
                "id": f"hard-{index}", "group": "hard-policy",
                "difficulty": "hard", "input": f"hard case {index}",
                "approved_output": "approved", "critical": True,
            }
            for index in range(5)
        ]
        result = Client(transport=DifficultyTransport()).optimize(
            prompt="Return the approved policy label only.",
            examples=examples,
            models=["target"],
            optimizer_model="optimizer",
            evaluator_model="unused",
            evaluator={"type": "exact_text"},
            difficulty_thresholds={"routine": 1.0, "hard": 1.0},
            quality_threshold=0.5,
            rounds=1,
            holdout_repeats=1,
            max_optimization_cost_usd=1,
        )

        self.assertEqual(result.winner.holdout_pass_rate, 0.5)
        self.assertEqual(result.winner.holdout_pass_rates_by_difficulty, {"hard": 0.0, "routine": 1.0})
        self.assertFalse(result.winner.passed_difficulty_floors)
        self.assertFalse(result.winner.passed_quality_floor)
        final_groups = {case.group for case in result.winner.cases if case.split == "holdout"}
        validation_groups = {case.group for case in result.winner.cases if case.split == "dev"}
        self.assertEqual(final_groups, {"routine-policy", "hard-policy"})
        self.assertEqual(validation_groups, {"routine-policy", "hard-policy"})

    def test_stratified_group_requires_five_scenarios(self):
        with self.assertRaisesRegex(ValueError, "at least five"):
            Client(transport=FakeTransport()).optimize(
                prompt="Return the approved policy label only.",
                examples=[
                    {"id": f"thin-{index}", "group": "thin", "input": f"case {index}", "approved_output": "ok"}
                    for index in range(4)
                ],
                models=["target"],
                max_optimization_cost_usd=1,
            )

    def test_price_first_api_keeps_test_budget_and_accuracy_as_separate_controls(self):
        with TemporaryDirectory() as directory:
            evalt = Evalt(transport=FakeTransport(), state_path=Path(directory) / "evalt.db")
            answer = evalt.run(
                "Return the approved route label only.", "charged twice",
                route="price-first", price_usd=0.003, test_budget_usd=0.40,
                target_accuracy=0.95, objective="best_within_price",
                models=["cheap"], auto_maintain=False, first_run="bootstrap",
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
                models=["cheap"], auto_maintain=False, first_run="bootstrap",
            )
            self.assertEqual(answer.model, "cheap")
            status = evalt.route_status("default")
            self.assertEqual(status["objective"], "lowest_cost_at_accuracy")
            self.assertEqual(status["target_accuracy"], 0.95)
            self.assertIsNone(status["max_p90_latency_seconds"])

    def test_omitted_price_uses_request_sized_ceiling_separate_from_auto_test_budget(self):
        with TemporaryDirectory() as directory:
            evalt = Evalt(
                transport=CostlyBootstrapTransport(),
                state_path=Path(directory) / "evalt.db",
            )
            answer = evalt.run(
                "Classify this request as billing, account, or technical.",
                "Please, everything is broken and the website will not load.",
                route="support-routing",
                target_accuracy=0.95,
                test_budget_usd="auto",
                models=["costly-bootstrap"],
                auto_maintain=False, first_run="bootstrap",
            )
            self.assertEqual(answer.content, "Here is a verbose answer")
            status = evalt.route_status("support-routing")
            self.assertIsNone(status["price_usd"])
            self.assertEqual(status["price_policy"], "automatic")
            self.assertAlmostEqual(status["effective_price_ceiling_usd"], 0.0627495)
            self.assertEqual(status["test_budget_usd"], 1.0)
            self.assertIn("no production price ceiling", status["test_budget_policy"])

    def test_interactive_progress_shows_cost_without_polluting_answer_content(self):
        with TemporaryDirectory() as directory:
            events = []
            stream = StringIO()
            evalt = Evalt(
                transport=CostlyBootstrapTransport(),
                state_path=Path(directory) / "evalt.db",
                show_progress=True,
                progress_callback=events.append,
            )
            with redirect_stderr(stream):
                answer = evalt.run(
                    "Return the approved route label only.",
                    "The website will not load.",
                    route="visible-support",
                    test_budget_usd="auto",
                    models=["costly-bootstrap"],
                    auto_maintain=False, first_run="bootstrap",
                )
            rendered = stream.getvalue()
            self.assertEqual(answer.content, "technical")
            self.assertIn("bootstrap-only production call; no tournament spend", rendered)
            self.assertIn("$0.057045", rendered)
            self.assertIn("UNTESTED BOOTSTRAP", rendered)
            self.assertIn("0/5 labeled examples", rendered)
            self.assertIn("no tournament ran", rendered)
            self.assertEqual(
                [event["event"] for event in events],
                ["production_call_started", "production_call_completed"],
            )

    def test_interactive_progress_surfaces_the_parallel_broad_screen(self):
        stream = StringIO()
        evalt = Evalt(transport=FakeTransport(), show_progress=True)
        with redirect_stderr(stream):
            evalt._emit_progress({
                "event": "suite_design_started",
                "route": "support-routing",
                "case_count": 25,
                "workflow_budget_usd": 1,
                "designer_model": "smart-designer",
                "designer_timeout_seconds": 300,
            })
            evalt._emit_progress({
                "event": "suite_design_attempt_started",
                "route": "support-routing",
                "designer_model": "smart-designer",
                "attempt": 1,
                "max_attempts": 2,
            })
            evalt._emit_progress({
                "event": "suite_designer_invalid",
                "route": "support-routing",
                "designer_model": "smart-designer",
                "attempt": 1,
                "max_attempts": 2,
                "will_retry": True,
            })
            evalt._emit_progress({
                "event": "suite_design_heartbeat",
                "route": "support-routing",
                "designer_model": "smart-designer",
                "elapsed_seconds": 20,
            })
            evalt._emit_progress({
                "event": "broad_screen_started",
                "route": "support-routing",
                "configurations": 12,
                "parallel_models": 12,
            })
            evalt._emit_progress({
                "event": "model_screen_completed",
                "route": "support-routing",
                "model": "cheap#reasoning=low",
                "validation_pass_rate": 0.8,
                "target_latency_p90_ms": 432,
                "screening_spend_usd": 0.0042,
            })
            evalt._emit_progress({
                "event": "model_started",
                "route": "support-routing",
                "model": "cheap#reasoning=low",
            })
            evalt._emit_progress({
                "event": "broad_screen_completed",
                "route": "support-routing",
                "configurations": 12,
                "completed_configurations": 11,
                "elapsed_seconds": 24.6,
            })
        rendered = stream.getvalue()
        self.assertIn("25 cases · smart-designer · deadline 300s", rendered)
        self.assertIn("smart-designer · attempt 1/2 · request started", rendered)
        self.assertIn("TEST DRAFT REJECTED · smart-designer · attempt 1/2", rendered)
        self.assertIn("retrying this model within the workflow cap", rendered)
        self.assertIn("smart-designer · 20s elapsed · still working", rendered)
        self.assertIn("support-routing · BROAD SCREEN · 12 model configuration(s) · up to 12 in parallel", rendered)
        self.assertIn("support-routing · SCREENED · cheap#reasoning=low · 80% validation", rendered)
        self.assertIn("support-routing · DEEP TEST STARTED · cheap#reasoning=low", rendered)
        self.assertIn("support-routing · BROAD SCREEN COMPLETE · 11/12 configuration(s) settled · 24.6s", rendered)

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
                auto_maintain=False, first_run="bootstrap",
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

    def test_existing_route_database_migrates_without_inventing_evidence_provenance(self):
        with TemporaryDirectory() as directory:
            state = Path(directory) / "evalt.db"
            prompt = "Return the approved route label only."
            version = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            db = sqlite3.connect(state)
            try:
                db.execute(
                    """CREATE TABLE routes (
                    route TEXT PRIMARY KEY,prompt TEXT NOT NULL,source_prompt_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,candidates_json TEXT NOT NULL,selected_model TEXT NOT NULL,
                    selected_prompt TEXT NOT NULL,decision_reason TEXT NOT NULL,quality_threshold REAL NOT NULL,
                    total_calls INTEGER NOT NULL DEFAULT 0,feedback_count INTEGER NOT NULL DEFAULT 0,
                    last_optimized_calls INTEGER NOT NULL DEFAULT 0,last_optimized_feedback INTEGER NOT NULL DEFAULT 0,
                    catalog_revision TEXT NOT NULL DEFAULT 'unseen',tested_catalog_revision TEXT NOT NULL DEFAULT 'unseen',
                    created_at TEXT NOT NULL,updated_at TEXT NOT NULL)"""
                )
                db.execute(
                    "INSERT INTO routes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "legacy-route", prompt, version, version, '["cheap"]', "cheap", prompt,
                        "qualified_lowest_cost_at_accuracy", 0.95, 3, 2, 3, 2, "old", "old",
                        "2026-07-20T00:00:00+00:00", "2026-07-20T00:00:00+00:00",
                    ),
                )
                db.commit()
            finally:
                db.close()
            evalt = Evalt(transport=FakeTransport(), state_path=state)
            status = evalt.route_status("legacy-route")
            self.assertEqual(status["evidence_provenance"], "LEGACY_UNKNOWN")
            self.assertEqual(status["route_phase"], "legacy_unknown")
            self.assertEqual(status["selected_few_shot_messages"], 0)
            self.assertTrue(evalt.router.needs_initial_optimization("legacy-route", prompt))
            answer = evalt.run(
                prompt,
                "charged twice",
                route="legacy-route",
                models=["cheap"],
                first_run="bootstrap",
            )
            self.assertEqual(answer.content, "billing")
            self.assertEqual(answer.evidence_provenance, "LEGACY_UNKNOWN")
            self.assertEqual(answer.route_phase, "legacy_unknown")

    def test_first_run_automatically_designs_tests_promotes_and_reuses_a_route(self):
        with TemporaryDirectory() as directory:
            transport = CaseDesignerFewShotTransport()
            events = []
            evalt = Evalt(
                transport=transport,
                state_path=Path(directory) / "evalt.db",
                progress_callback=events.append,
            )
            first = evalt.run(
                "Write a helpful classification for this message.",
                "Please, everything is broken and the website will not load.",
                task="Route recurring support tickets to billing, account, or technical.",
                route="automatic-first-route",
                test_budget_usd=1,
                models=["cheap"],
                designer_model="designer",
                evaluator_model="evaluator",
            )
            self.assertEqual(first.content, "technical")
            self.assertEqual(first.route_phase, "ai_tested")
            self.assertEqual(first.evidence_provenance, "AI_GENERATED_AI_JUDGED")
            self.assertEqual(first.decision_reason, "provisional_ai_qualified")
            self.assertEqual(first.initial_test_summary["final_test_scenarios"], 5)
            self.assertGreaterEqual(first.initial_test_summary["tested_configurations"], 1)
            self.assertGreaterEqual(first.initial_test_summary["prompt_candidates_tested"], 2)
            self.assertGreaterEqual(first.initial_test_summary["prompt_rewrites_tested"], 1)
            self.assertEqual(first.initial_test_summary["few_shot_examples"], 1)
            status = evalt.route_status("automatic-first-route")
            self.assertEqual(status["route_phase"], "ai_tested")
            self.assertEqual(status["selected_few_shot_messages"], 2)
            self.assertEqual(status["selected_few_shot_examples"], 1)
            self.assertIn(
                "initial_ai_route_promoted",
                [item["event_type"] for item in status["decisions"]],
            )
            designed_before = sum(
                "Design a balanced evaluation suite" in messages[0]["content"]
                for _model, messages, _tokens, _schema in transport.calls
            )
            design_call = next(
                messages
                for _model, messages, _tokens, _schema in transport.calls
                if "Design a balanced evaluation suite" in messages[0]["content"]
            )
            design_payload = json.loads(design_call[-1]["content"])
            self.assertEqual(
                design_payload["representative_inputs_without_labels"][0]["content"],
                "Please, everything is broken and the website will not load.",
            )
            second = evalt.run(
                "Write a helpful classification for this message.",
                "I was charged twice.",
                task="Route recurring support tickets to billing, account, or technical.",
                route="automatic-first-route",
                test_budget_usd=1,
                models=["cheap"],
                designer_model="designer",
                evaluator_model="evaluator",
            )
            designed_after = sum(
                "Design a balanced evaluation suite" in messages[0]["content"]
                for _model, messages, _tokens, _schema in transport.calls
            )
            self.assertEqual(second.content, "billing")
            self.assertEqual(designed_after, designed_before)
            self.assertEqual(evalt.route_status("automatic-first-route")["total_calls"], 2)
            event_names = [event["event"] for event in events]
            self.assertIn("initial_optimization_started", event_names)
            self.assertIn("initial_optimization_completed", event_names)
            self.assertIn("prompt_candidate_completed", event_names)

    def test_bootstrap_only_is_an_explicit_escape_hatch(self):
        with TemporaryDirectory() as directory:
            transport = FakeTransport()
            answer = Evalt(
                transport=transport,
                state_path=Path(directory) / "evalt.db",
            ).run(
                "Return the approved route label only.",
                "The website will not load.",
                route="explicit-bootstrap",
                first_run="bootstrap",
                models=["cheap"],
            )
            self.assertEqual(answer.route_phase, "untested_bootstrap")
            self.assertEqual(answer.evidence_provenance, "UNTESTED_BOOTSTRAP")
            self.assertIsNone(answer.initial_test_summary)
            self.assertEqual(len(transport.calls), 1)

    def test_automatic_first_route_fails_closed_when_no_configuration_passes(self):
        with TemporaryDirectory() as directory:
            evalt = Evalt(
                transport=FailingCaseDesignerTransport(),
                state_path=Path(directory) / "evalt.db",
            )
            with self.assertRaisesRegex(ProviderError, "route was not promoted"):
                evalt.run(
                    "Return exactly one lowercase label: billing, account, or technical.",
                    "The website will not load.",
                    task="Route recurring support tickets to billing, account, or technical.",
                    route="no-passing-route",
                    test_budget_usd=1,
                    models=["cheap"],
                    designer_model="designer",
                    evaluator_model="evaluator",
                )
            with self.assertRaises(KeyError):
                evalt.route_status("no-passing-route")

    def test_automatic_first_route_never_spends_past_the_shared_test_cap(self):
        with TemporaryDirectory() as directory:
            transport = CaseDesignerTransport()
            with self.assertRaises(BudgetExceeded):
                Evalt(
                    transport=transport,
                    state_path=Path(directory) / "evalt.db",
                ).run(
                    "Return exactly one lowercase label: billing, account, or technical.",
                    "The website will not load.",
                    task="Route recurring support tickets to billing, account, or technical.",
                    route="tiny-first-test",
                    test_budget_usd=0.00005,
                    models=["cheap"],
                    designer_model="designer",
                    evaluator_model="evaluator",
                )
            self.assertEqual(transport.calls, [])

    def test_named_routes_keep_multiple_tasks_and_their_evidence_isolated(self):
        with TemporaryDirectory() as directory:
            state = Path(directory) / "evalt.db"
            evalt = Evalt(transport=FakeTransport(), state_path=state)
            support = evalt.run(
                "Return the approved route label only.", "I was charged twice",
                route="support-routing", budget_usd=0.01, models=["cheap"], auto_maintain=False, first_run="bootstrap",
            )
            incident = evalt.run(
                "Return the approved route label only. This route triages incidents.", "The app freezes",
                route="incident-triage", budget_usd=0.01, models=["cheap"], auto_maintain=False, first_run="bootstrap",
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
                first_run="bootstrap",
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
                first_run="bootstrap",
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
                    "Return one label.", "hello", models=["cheap"], auto_maintain=False, first_run="bootstrap",
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
                    auto_maintain=False, first_run="bootstrap",
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
                    "Return one label.", "hello", route="tiny-budget", budget_usd=0.00001, models=["cheap"], incumbent_model="cheap", first_run="bootstrap"
                )
            self.assertEqual(transport.calls, [])

    def test_prompt_change_is_logged_and_reverts_to_unqualified_bootstrap(self):
        with TemporaryDirectory() as directory:
            state = Path(directory) / "evalt.db"
            evalt = Evalt(transport=FakeTransport(), state_path=state)
            first = evalt.run("Return the approved route label only.", "charged", route="support", budget_usd=0.01, models=["cheap"], incumbent_model="cheap", auto_maintain=False, first_run="bootstrap")
            first.accept()
            self.assertEqual(evalt.route_status("support")["feedback_count"], 1)
            second = evalt.run("Return only billing, account, or technical.", "charged", route="support", budget_usd=0.01, models=["cheap"], incumbent_model="cheap", auto_maintain=False, first_run="bootstrap")
            self.assertNotEqual(first.prompt_version, second.prompt_version)
            self.assertEqual(second.decision_reason, "prompt_changed_unqualified")
            status = evalt.route_status("support")
            self.assertEqual(status["feedback_count"], 0)
            self.assertEqual(status["route_phase"], "untested_bootstrap")
            events = [item["event_type"] for item in status["decisions"]]
            self.assertIn("prompt_changed", events)

    def test_feedback_progress_is_explicit_and_launches_first_real_tournament(self):
        with TemporaryDirectory() as directory:
            stream = StringIO()
            evalt = Evalt(
                transport=FakeTransport(),
                state_path=Path(directory) / "evalt.db",
                show_progress=True,
            )
            cases = [
                ("charged twice", "billing"),
                ("reset expired", "account"),
                ("app freezes", "technical"),
                ("charged again", "billing"),
                ("reset broken", "account"),
            ]
            with redirect_stderr(stream):
                for input_text, approved in cases:
                    answer = evalt.run(
                        "Write a helpful classification for this message.",
                        input_text,
                        route="automatic-maintenance",
                        budget_usd=0.01,
                        models=["cheap"],
                        incumbent_model="cheap",
                        min_feedback=5,
                        first_run="bootstrap",
                    )
                    answer.correct(approved)
                evalt.wait_for_maintenance()
            rendered = stream.getvalue()
            self.assertIn("1/5 labeled examples", rendered)
            self.assertIn("5/5 labeled examples · tournament eligible", rendered)
            self.assertIn("TOURNAMENT STARTED", rendered)
            self.assertIn("TOURNAMENT COMPLETE", rendered)
            self.assertEqual(
                evalt.route_status("automatic-maintenance")["route_phase"],
                "human_calibrated",
            )

    def test_durable_route_persists_fixed_prompt_policy(self):
        with TemporaryDirectory() as directory:
            evalt = Evalt(
                transport=FakeTransport(),
                state_path=Path(directory) / "evalt.db",
            )
            evalt.run(
                "Return one route label only.",
                "charged twice",
                route="fixed-prompt",
                models=["cheap"],
                incumbent_model="cheap",
                optimize_prompt=False,
                budget_usd=0.01,
                first_run="bootstrap",
            )
            self.assertFalse(evalt.route_status("fixed-prompt")["optimize_prompt"])

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

    def test_standard_role_policy_screens_ten_distinct_models_before_reasoning_hone(self):
        catalog = [
            {
                "id": f"model-{index}",
                "intelligence": 60 + index,
                "blended_price": 0.05 * (index + 1),
                "supported_parameters": ["max_tokens", "reasoning"],
                "reasoning": {"supported_efforts": ["low", "medium", "high"]},
            }
            for index in range(14)
        ]
        plan = select_role_plan(catalog, maintenance_budget_usd=1.00)
        broad = plan.target_models[:10]
        self.assertEqual(plan.tier, "standard")
        self.assertEqual(len(broad), 10)
        self.assertEqual(len({item.split("#", 1)[0] for item in broad}), 10)

    def test_automatic_role_policy_preserves_extreme_efforts_for_staged_search(self):
        catalog = [
            {
                "id": "wide-effort-model",
                "intelligence": 90,
                "blended_price": 0.1,
                "supported_parameters": ["max_tokens", "reasoning"],
                "reasoning": {
                    "supported_efforts": ["low", "medium", "high", "xhigh", "max"]
                },
            },
            {"id": "plain-model", "intelligence": 80, "blended_price": 0.2},
        ]
        plan = select_role_plan(catalog, maintenance_budget_usd=1.0)
        self.assertTrue(any("reasoning=xhigh" in item for item in plan.target_models))
        self.assertTrue(any("reasoning=max" in item for item in plan.target_models))
        self.assertTrue(any("reasoning=high" in item for item in plan.target_models))

    def test_role_policy_bootstraps_on_a_capable_redundant_route_not_a_fragile_cheapest_model(self):
        catalog = [
            {"id": "fragile-cheap", "intelligence": 80, "blended_price": 0.1, "private_provider_routes": 1},
            {"id": "reliable", "intelligence": 82, "blended_price": 0.3, "private_provider_routes": 3},
            {"id": "strong", "intelligence": 96, "blended_price": 4.0, "private_provider_routes": 2},
        ]
        plan = select_role_plan(catalog, maintenance_budget_usd=0.25)
        self.assertTrue(plan.target_models[0].startswith("reliable#reasoning="))
        self.assertTrue(any(model.startswith("fragile-cheap#reasoning=") for model in plan.target_models))

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
                    route="maintained", budget_usd=0.01, models=["cheap"], incumbent_model="cheap", min_feedback=5, auto_maintain=False, first_run="bootstrap",
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
                first_run="bootstrap",
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

    def test_ai_suite_design_is_budgeted_reviewable_and_not_silently_approved(self):
        with TemporaryDirectory() as directory:
            evalt = Evalt(
                transport=CaseDesignerTransport(),
                state_path=Path(directory) / "evalt.db",
            )
            draft = evalt.design_suite(
                task="Classify recurring support tickets into one routing label.",
                prompt="Return exactly one lowercase label: billing, account, or technical.",
                route="designed-support",
                case_count=25,
                workflow_budget_usd=1,
                models=["cheap"],
                designer_model="designer",
                evaluator_model="evaluator",
            )
            self.assertEqual(len(draft.examples), 25)
            self.assertEqual(draft.evidence_provenance, "AI_DRAFT_UNAPPROVED")
            self.assertGreater(draft.designer_spend_usd, 0)
            self.assertLess(draft.remaining_optimization_budget_usd, 1)
            self.assertEqual(draft.request_timeout_seconds, 120)
            self.assertTrue(any("Judge calibration" in note for note in draft.design_notes))
            suite = draft.approve()
            self.assertEqual(suite.evidence_provenance, "HUMAN_APPROVED_AI_DRAFT")
            self.assertEqual(suite.evaluator["type"], "exact_text")
            self.assertEqual(suite.max_optimization_cost_usd, draft.remaining_optimization_budget_usd)

    def test_ai_suite_design_falls_back_to_the_independent_judge_role(self):
        events: list[dict] = []
        evalt = Evalt(
            transport=DesignerFallbackTransport(),
            progress_callback=events.append,
        )
        draft = evalt.design_suite(
            task="Classify recurring support tickets into one routing label.",
            prompt="Return exactly one lowercase label: billing, account, or technical.",
            route="designer-fallback",
            case_count=25,
            workflow_budget_usd=1,
        )
        self.assertEqual(draft.designer_model, "secondary-designer")
        self.assertTrue(any("Designer fallback" in note for note in draft.design_notes))
        self.assertEqual(
            [event["event"] for event in events if event["event"] == "suite_designer_unavailable"],
            ["suite_designer_unavailable"],
        )
        self.assertEqual(
            [
                event["designer_model"] for event in events
                if event["event"] == "suite_design_attempt_started"
            ],
            ["primary-designer", "secondary-designer"],
        )

    def test_ai_suite_design_rejects_a_judge_that_cannot_detect_known_failure(self):
        with TemporaryDirectory() as directory:
            evalt = Evalt(
                transport=AlwaysPassJudgeDesignerTransport(),
                state_path=Path(directory) / "evalt.db",
            )
            with self.assertRaisesRegex(
                ProviderError, "No candidate judge passed the calibration"
            ):
                evalt.design_suite(
                    task="Classify recurring support tickets into one routing label.",
                    prompt="Return exactly one lowercase label: billing, account, or technical.",
                    route="bad-judge",
                    case_count=25,
                    workflow_budget_usd=1,
                    models=["cheap"],
                    designer_model="designer",
                    evaluator_model="evaluator",
                )
            with self.assertRaises(KeyError):
                evalt.route_status("bad-judge")

    def test_ai_suite_design_retries_malformed_structured_output_with_visible_attempts(self):
        events: list[dict] = []
        transport = MalformedThenValidDesignerTransport()
        evalt = Evalt(transport=transport, progress_callback=events.append)
        draft = evalt.design_suite(
            task="Classify recurring support tickets into one routing label.",
            prompt="Return exactly one lowercase label: billing, account, or technical.",
            route="designer-structured-retry",
            case_count=25,
            workflow_budget_usd=1,
            models=["cheap"],
            designer_model="designer",
            evaluator_model="evaluator",
        )
        self.assertEqual(len(draft.examples), 25)
        self.assertEqual(transport.design_attempts, 2)
        attempts = [
            event for event in events
            if event["event"] == "suite_design_attempt_started"
        ]
        self.assertEqual([event["attempt"] for event in attempts], [1, 2])
        rejected = [
            event for event in events
            if event["event"] == "suite_designer_invalid"
        ]
        self.assertEqual(len(rejected), 1)
        self.assertTrue(rejected[0]["will_retry"])

    def test_semantic_calibration_cannot_label_a_different_answer_as_a_known_pass(self):
        events: list[dict] = []
        transport = SemanticMiscalibratedDesignerTransport()
        evalt = Evalt(transport=transport, progress_callback=events.append)
        draft = evalt.design_suite(
            task="Score recurring customer sentiment on a consistent scale.",
            prompt="Return a sentiment score from zero to one hundred.",
            route="semantic-calibration-identity",
            case_count=25,
            workflow_budget_usd=1,
            models=["cheap"],
            designer_model="designer",
            evaluator_model="evaluator",
        )
        self.assertEqual(draft.evaluator["type"], "semantic")
        self.assertTrue(any(
            "known-pass control identical" in note for note in draft.design_notes
        ))
        calibration_events = [
            event for event in events
            if event["event"] == "judge_calibration_completed"
        ]
        self.assertTrue(calibration_events[-1]["passed"])
        judgment_payloads = [
            json.loads(messages[-1]["content"])
            for _model, messages, _max_tokens, response_schema in transport.calls
            if response_schema and "Judge whether" in messages[0]["content"]
        ]
        self.assertEqual(
            judgment_payloads[0]["actual_answer"],
            judgment_payloads[0]["approved_answer"],
        )
        self.assertEqual(
            judgment_payloads[1]["actual_answer"],
            judgment_payloads[1]["approved_answer"],
        )

    def test_autopilot_design_runs_full_tournament_but_labels_ai_evidence(self):
        with TemporaryDirectory() as directory:
            evalt = Evalt(
                transport=CaseDesignerTransport(),
                state_path=Path(directory) / "evalt.db",
            )
            result = evalt.optimize_task(
                task="Classify recurring support tickets into one routing label.",
                prompt="Return exactly one lowercase label: billing, account, or technical.",
                route="autopilot-support",
                case_control="autopilot",
                case_count=25,
                workflow_budget_usd=1,
                models=["cheap"],
                designer_model="designer",
                evaluator_model="evaluator",
            )
            self.assertEqual(
                result.regression_suite["evidence_provenance"],
                "AI_GENERATED_AI_JUDGED",
            )
            self.assertIn("AI-generated", result.warnings[0])
            self.assertEqual(result.regression_suite["holdout_unique_scenarios"], 5)
            self.assertGreater(result.total_provider_spend_usd, 0)

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

    def test_automatic_first_route_uses_a_separate_longer_ai_design_deadline(self):
        with TemporaryDirectory() as directory:
            evalt = Evalt(
                api_key="sk-or-v1-test-key",
                state_path=Path(directory) / "evalt.db",
                show_progress=False,
            )
            observed: dict[str, float] = {}

            def stop_after_observation(**_kwargs):
                observed["timeout_seconds"] = evalt.client.transport.timeout_seconds
                raise RuntimeError("design observation complete")

            evalt.design_suite = stop_after_observation  # type: ignore[method-assign]
            with self.assertRaisesRegex(RuntimeError, "design observation complete"):
                evalt.run(
                    "Return exactly one lowercase route label.",
                    "the website will not load",
                    task="Route recurring customer support tickets.",
                    route="deadline-before-design",
                    models=["cheap"],
                    test_budget_usd=0.25,
                    max_test_budget_usd=0.25,
                    test_request_timeout_seconds=37,
                    designer_request_timeout_seconds=211,
                )
            self.assertEqual(observed["timeout_seconds"], 211)

    def test_transport_defaults_to_ten_minutes_and_allows_long_complex_jobs(self):
        transport = OpenRouterTransport("sk-or-v1-test-key")
        self.assertEqual(transport.timeout_seconds, 600)
        transport.set_timeout_seconds(1800)
        self.assertEqual(transport.timeout_seconds, 1800)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            transport.set_timeout_seconds(0)

    def test_default_transport_uses_a_bundled_verified_ca_context(self):
        with mock.patch("last_good_prompt.core.urlopen", return_value=FakeResponse({"ok": True})) as opener:
            transport = OpenRouterTransport("sk-or-v1-test-key")
            self.assertEqual(transport._request("https://openrouter.ai/test"), {"ok": True})
        context = opener.call_args.kwargs["context"]
        self.assertTrue(context.check_hostname)
        self.assertGreater(len(context.get_ca_certs()), 0)

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

    def test_cli_exposes_fixed_prompt_mode_for_routes_and_explicit_suites(self):
        run_args = cli_parser().parse_args([
            "run", "--route", "support", "--prompt", "Return one route label.",
            "--input", "The website is broken.", "--price", "0.01", "--fixed-prompt",
        ])
        optimize_args = cli_parser().parse_args([
            "optimize", "evalt.json", "--fixed-prompt",
        ])
        self.assertTrue(run_args.fixed_prompt)
        self.assertTrue(optimize_args.fixed_prompt)

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
        self.assertEqual(sent["max_completion_tokens"], 25)
        self.assertNotIn("max_tokens", sent)
        self.assertNotIn("temperature", sent)
        self.assertEqual(sent["provider"]["sort"], "price")
        self.assertEqual(sent["usage"], {"include": True})
        self.assertEqual(sent["response_format"]["type"], "json_schema")

    def test_exact_json_targets_receive_a_shape_only_response_schema(self):
        transport = FakeTransport()
        Client(transport=transport).optimize(
            prompt="Return exact JSON only.",
            examples=[
                {"id": "one", "input": "first", "approved_output": '{"x":"17/3","y":"-2"}'},
                {"id": "two", "input": "second", "approved_output": '{"x":"5","y":"11/7"}'},
                {"id": "three", "input": "third", "approved_output": '{"x":"-9/4","y":"6"}'},
            ],
            models=["target"],
            optimizer_model="optimizer",
            evaluator_model="evaluator",
            evaluator={
                "type": "exact_json",
                "required_keys": ["x", "y"],
                "allow_additional_properties": False,
                "normalize_rational_strings": True,
            },
            max_optimization_cost_usd=1,
            rounds=1,
        )
        target_schemas = [
            response_schema
            for model, messages, _max_tokens, response_schema in transport.calls
            if model == "target" and messages[0]["content"] != "Improve the current prompt using only the supplied customer-approved training evidence. Return a deployable prompt and a short hypothesis."
        ]
        self.assertTrue(target_schemas)
        for schema in target_schemas:
            self.assertEqual(schema["required"], ["x", "y"])
            self.assertEqual(schema["properties"], {"x": {"type": "string"}, "y": {"type": "string"}})
            self.assertFalse(schema["additionalProperties"])
            self.assertNotIn("17/3", json.dumps(schema))

    def test_endpoint_selection_prefers_a_schema_capable_route_at_the_same_capacity(self):
        requests = []

        def opener(request, timeout):
            if request.data is not None:
                requests.append(json.loads(request.data))
                return FakeResponse({
                    "id": "gen-schema", "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                    "usage": {"cost": 0.0001, "prompt_tokens": 4, "completion_tokens": 1},
                })
            if "endpoints/zdr" in request.full_url:
                return FakeResponse({"data": [
                    {
                        "model_id": "schema-choice", "tag": "cheap/unstructured", "context_length": 131072,
                        "max_completion_tokens": 131072, "pricing": {"prompt": "0.00000001", "completion": "0.00000001"},
                        "supported_parameters": ["max_completion_tokens"],
                    },
                    {
                        "model_id": "schema-choice", "tag": "reliable/structured", "context_length": 131072,
                        "max_completion_tokens": 131072, "pricing": {"prompt": "0.00000002", "completion": "0.00000002"},
                        "supported_parameters": ["max_completion_tokens", "response_format", "structured_outputs"],
                    },
                ]})
            return FakeResponse({"data": [{
                "id": "schema-choice", "context_length": 131072,
                "top_provider": {"context_length": 131072, "max_completion_tokens": 131072},
                "pricing": {"prompt": "0.00000001", "completion": "0.00000001"},
                "supported_parameters": ["max_completion_tokens", "response_format", "structured_outputs"],
            }]})

        transport = OpenRouterTransport("sk-or-v1-test-key", opener=opener)
        transport.complete(
            "schema-choice", [{"role": "user", "content": "hello"}], max_tokens=25,
            response_schema={"type": "object"},
        )
        sent = requests[-1]
        self.assertEqual(sent["provider"]["only"], ["reliable/structured"])
        self.assertEqual(sent["response_format"]["type"], "json_schema")

    def test_provider_429_retries_the_same_model_on_a_preflighted_fallback_route(self):
        requests = []

        def opener(request, timeout):
            if request.data is not None:
                body = json.loads(request.data)
                requests.append(body)
                if len(requests) == 1:
                    detail = json.dumps({
                        "error": {
                            "message": "rate limited",
                            "code": 429,
                            "metadata": {"provider_name": "First"},
                        }
                    }).encode()
                    raise HTTPError(request.full_url, 429, "rate limited", {}, BytesIO(detail))
                return FakeResponse({
                    "id": "gen-fallback", "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                    "usage": {"cost": 0.0001, "prompt_tokens": 4, "completion_tokens": 1},
                })
            if "endpoints/zdr" in request.full_url:
                return FakeResponse({"data": [
                    {
                        "model_id": "fallback-model", "tag": "first/route", "provider_name": "First",
                        "context_length": 131072, "max_completion_tokens": 131072,
                        "pricing": {"prompt": "0.00000001", "completion": "0.00000001"},
                        "supported_parameters": ["max_completion_tokens", "structured_outputs"],
                    },
                    {
                        "model_id": "fallback-model", "tag": "second/route", "provider_name": "Second",
                        "context_length": 131072, "max_completion_tokens": 131072,
                        "pricing": {"prompt": "0.00000002", "completion": "0.00000002"},
                        "supported_parameters": ["max_completion_tokens", "structured_outputs"],
                    },
                ]})
            return FakeResponse({"data": [{
                "id": "fallback-model", "context_length": 131072,
                "top_provider": {"context_length": 131072, "max_completion_tokens": 131072},
                "pricing": {"prompt": "0.00000001", "completion": "0.00000001"},
                "supported_parameters": ["max_completion_tokens", "structured_outputs"],
            }]})

        transport = OpenRouterTransport("sk-or-v1-test-key", opener=opener)
        completion = transport.complete(
            "fallback-model", [{"role": "user", "content": "hello"}], max_tokens=25,
            response_schema={"type": "object"},
        )
        self.assertEqual(completion.generation_id, "gen-fallback")
        self.assertEqual(requests[0]["provider"]["only"], ["first/route", "second/route"])
        self.assertEqual(requests[1]["provider"]["only"], ["second/route"])

    def test_empty_completion_falls_back_to_another_preflighted_provider_without_expanding_tokens(self):
        requests = []

        def opener(request, timeout):
            if request.data is not None:
                body = json.loads(request.data)
                requests.append(body)
                if len(requests) == 1:
                    return FakeResponse({
                        "id": "gen-empty", "provider": "First",
                        "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
                        "usage": {"cost": 0.00003, "prompt_tokens": 4, "completion_tokens": 0},
                    })
                return FakeResponse({
                    "id": "gen-good", "provider": "Second",
                    "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                    "usage": {"cost": 0.0001, "prompt_tokens": 4, "completion_tokens": 1},
                })
            if "endpoints/zdr" in request.full_url:
                return FakeResponse({"data": [
                    {
                        "model_id": "fallback-model", "tag": "first/route", "provider_name": "First",
                        "context_length": 131072, "max_completion_tokens": 131072,
                        "pricing": {"prompt": "0.00000001", "completion": "0.00000001"},
                        "supported_parameters": ["max_completion_tokens", "structured_outputs"],
                    },
                    {
                        "model_id": "fallback-model", "tag": "second/route", "provider_name": "Second",
                        "context_length": 131072, "max_completion_tokens": 131072,
                        "pricing": {"prompt": "0.00000002", "completion": "0.00000002"},
                        "supported_parameters": ["max_completion_tokens", "structured_outputs"],
                    },
                ]})
            return FakeResponse({"data": [{
                "id": "fallback-model", "context_length": 131072,
                "top_provider": {"context_length": 131072, "max_completion_tokens": 131072},
                "pricing": {"prompt": "0.00000001", "completion": "0.00000001"},
                "supported_parameters": ["max_completion_tokens", "structured_outputs"],
            }]})

        completion = OpenRouterTransport(
            "sk-or-v1-test-key", opener=opener,
        ).complete(
            "fallback-model", [{"role": "user", "content": "hello"}], max_tokens=25,
            response_schema={"type": "object"},
        )
        self.assertEqual(completion.generation_id, "gen-good")
        self.assertAlmostEqual(completion.cost_usd, 0.00013)
        self.assertEqual(requests[0]["max_completion_tokens"], requests[1]["max_completion_tokens"])
        self.assertEqual(requests[1]["provider"]["only"], ["second/route"])

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

    def test_empty_answer_does_not_misdiagnose_the_failure_as_a_token_limit(self):
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
                return Completion("unexpected retry", model, "gen-retry", 0.002)

        transport = EmptyOnceTransport()
        with self.assertRaises(ProviderError):
            Client(transport=transport).draft_answer(
                task="Answer completely.", input="Test", max_cost_usd=0.10,
            )
        self.assertEqual(transport.max_tokens_seen, [8192])

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
        self.assertIn("near#reasoning=high", result.pruned_models)
        self.assertEqual(result.winner.model, "near#reasoning=low")
        self.assertTrue(all(
            item.prompt_rewrites_tested == 0
            for item in result.models
            if item.model == "cheap#reasoning=high"
        ))
        self.assertIn("adaptive search band", result.winner_scope)

    def test_extreme_reasoning_requires_close_validation_not_a_lucky_final_test(self):
        class LadderClient(Client):
            def __init__(self):
                super().__init__(transport=FakeTransport())
                self.evaluated = []

            def _evaluate_model(self, *args, **kwargs):
                model = args[4]
                self.evaluated.append(model)
                effort = model.rsplit("#reasoning=", 1)[-1]
                validation = {"low": 0.40, "high": 0.40}.get(effort, 1.0)
                return ModelResult(
                    model=model,
                    selected_prompt=args[0],
                    baseline_pass_rate=validation,
                    selected_pass_rate=validation,
                    holdout_pass_rate=1.0,
                    baseline_holdout_pass_rate=1.0,
                    estimated_production_cost_per_call_usd=0.001,
                    estimated_cost_per_successful_call_usd=0.001,
                    optimization_spend_usd=0.0,
                    passed_quality_floor=True,
                    target_latency_p90_ms=500,
                )

        client = LadderClient()
        examples = [
            {
                "id": f"case-{index}",
                "input": f"request {index}",
                "approved_output": "billing",
            }
            for index in range(25)
        ]
        events = []
        client.optimize(
            prompt="Return one label.",
            examples=examples,
            models=[
                "candidate#reasoning=low", "candidate#reasoning=high",
                "candidate#reasoning=xhigh", "candidate#reasoning=max",
            ],
            optimizer_model="optimizer",
            evaluator_model="evaluator",
            adaptive_search=True,
            max_optimization_cost_usd=1,
            progress_callback=events.append,
        )
        self.assertIn("candidate#reasoning=high", client.evaluated)
        self.assertNotIn("candidate#reasoning=xhigh", client.evaluated)
        self.assertNotIn("candidate#reasoning=max", client.evaluated)
        self.assertTrue(any(
            event["event"] == "reasoning_escalation_skipped"
            and event["to_effort"] == "xhigh"
            for event in events
        ))

    def test_extreme_reasoning_climbs_one_rung_at_a_time_and_stops_on_regression(self):
        class LadderClient(Client):
            def __init__(self):
                super().__init__(transport=FakeTransport())
                self.evaluated = []

            def _evaluate_model(self, *args, **kwargs):
                model = args[4]
                self.evaluated.append(model)
                effort = model.rsplit("#reasoning=", 1)[-1]
                validation = {
                    "low": 0.60, "high": 0.80, "xhigh": 0.60, "max": 1.0,
                }[effort]
                return ModelResult(
                    model=model,
                    selected_prompt=args[0],
                    baseline_pass_rate=validation,
                    selected_pass_rate=validation,
                    holdout_pass_rate=validation,
                    baseline_holdout_pass_rate=validation,
                    estimated_production_cost_per_call_usd=0.001,
                    estimated_cost_per_successful_call_usd=0.001,
                    optimization_spend_usd=0.0,
                    passed_quality_floor=validation >= 0.95,
                    target_latency_p90_ms=500,
                )

        client = LadderClient()
        examples = [
            {
                "id": f"case-{index}",
                "input": f"request {index}",
                "approved_output": "billing",
            }
            for index in range(25)
        ]
        events = []
        client.optimize(
            prompt="Return one label.",
            examples=examples,
            models=[
                "candidate#reasoning=low", "candidate#reasoning=high",
                "candidate#reasoning=xhigh", "candidate#reasoning=max",
            ],
            optimizer_model="optimizer",
            evaluator_model="evaluator",
            adaptive_search=True,
            max_optimization_cost_usd=1,
            progress_callback=events.append,
        )
        self.assertIn("candidate#reasoning=xhigh", client.evaluated)
        self.assertNotIn("candidate#reasoning=max", client.evaluated)
        self.assertTrue(any(
            event["event"] == "reasoning_escalation_skipped"
            and event["to_effort"] == "max"
            for event in events
        ))

    def test_extreme_reasoning_does_not_escalate_past_the_latency_ceiling(self):
        class LadderClient(Client):
            def __init__(self):
                super().__init__(transport=FakeTransport())
                self.evaluated = []

            def _evaluate_model(self, *args, **kwargs):
                model = args[4]
                self.evaluated.append(model)
                effort = model.rsplit("#reasoning=", 1)[-1]
                validation = {"low": 0.60, "high": 0.80}.get(effort, 1.0)
                return ModelResult(
                    model=model,
                    selected_prompt=args[0],
                    baseline_pass_rate=validation,
                    selected_pass_rate=validation,
                    holdout_pass_rate=validation,
                    baseline_holdout_pass_rate=validation,
                    estimated_production_cost_per_call_usd=0.001,
                    estimated_cost_per_successful_call_usd=0.001,
                    optimization_spend_usd=0.0,
                    passed_quality_floor=False,
                    target_latency_p90_ms=4_000 if effort == "high" else 500,
                )

        client = LadderClient()
        examples = [
            {
                "id": f"case-{index}",
                "input": f"request {index}",
                "approved_output": "billing",
            }
            for index in range(25)
        ]
        client.optimize(
            prompt="Return one label.",
            examples=examples,
            models=[
                "candidate#reasoning=low", "candidate#reasoning=high",
                "candidate#reasoning=xhigh", "candidate#reasoning=max",
            ],
            optimizer_model="optimizer",
            evaluator_model="evaluator",
            adaptive_search=True,
            max_optimization_cost_usd=1,
            max_p90_latency_seconds=3.0,
        )
        self.assertNotIn("candidate#reasoning=xhigh", client.evaluated)
        self.assertNotIn("candidate#reasoning=max", client.evaluated)

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

    def test_adaptive_search_screens_broad_models_before_full_final_test(self):
        class ScreeningTransport(FakeTransport):
            def estimate_cost(self, model, messages, *, max_tokens):
                return {
                    "strong-cheap": 0.0002,
                    "strong-expensive": 0.002,
                    "weak-cheapest": 0.00001,
                }.get(model, 0.0001)

            def complete(self, model, messages, *, max_tokens, response_schema=None):
                if model == "optimizer":
                    return super().complete(
                        model, messages, max_tokens=max_tokens, response_schema=response_schema,
                    )
                self.calls.append((model, messages, max_tokens, response_schema))
                content = "billing" if model.startswith("strong-") else "wrong"
                return Completion(
                    content, model, f"gen-{len(self.calls)}",
                    self.estimate_cost(model, messages, max_tokens=max_tokens),
                    latency_ms=10,
                )

        examples = [
            {"id": f"billing-{index}", "input": f"charged twice {index}", "approved_output": "billing"}
            for index in range(15)
        ]
        events = []
        result = Client(transport=ScreeningTransport()).optimize(
            prompt="Return the approved route label only.",
            examples=examples,
            models=[
                "weak-cheapest", "weak-two", "strong-expensive",
                "weak-three", "strong-cheap", "weak-four",
            ],
            optimizer_model="optimizer",
            evaluator_model="unused",
            evaluator={"type": "exact_text"},
            rounds=1,
            adaptive_search=True,
            max_optimization_cost_usd=5,
            progress_callback=events.append,
        )
        self.assertEqual(len(result.screening_results), 6)
        self.assertEqual(len(result.models), 4)
        self.assertEqual(result.winner.model, "strong-cheap")
        self.assertEqual(len(result.pruned_models), 2)
        self.assertTrue(all(model.startswith("weak-") for model in result.pruned_models))
        self.assertTrue(any(event["event"] == "model_screen_completed" for event in events))
        self.assertTrue(any(event["event"] == "model_pruned" for event in events))

    def test_weak_adaptive_screen_preserves_one_catalog_intelligence_anchor(self):
        class AnchorTransport(FakeTransport):
            def model_catalog(self):
                return [
                    {"id": f"model-{index}", "intelligence": index}
                    for index in range(1, 7)
                ]

            def estimate_cost(self, model, messages, *, max_tokens):
                if model.startswith("model-"):
                    return int(model.rsplit("-", 1)[1]) / 10_000
                return super().estimate_cost(model, messages, max_tokens=max_tokens)

            def complete(self, model, messages, *, max_tokens, response_schema=None):
                if response_schema:
                    return super().complete(
                        model, messages, max_tokens=max_tokens,
                        response_schema=response_schema,
                    )
                self.calls.append((model, messages, max_tokens, response_schema))
                return Completion(
                    "wrong", model, f"gen-{len(self.calls)}",
                    self.estimate_cost(model, messages, max_tokens=max_tokens),
                )

        examples = [
            {"id": f"billing-{index}", "input": f"charged twice {index}", "approved_output": "billing"}
            for index in range(15)
        ]
        events = []
        result = Client(transport=AnchorTransport()).optimize(
            prompt="Classify this support request.", examples=examples,
            models=[f"model-{index}" for index in range(1, 7)],
            optimizer_model="optimizer", evaluator_model="unused",
            evaluator={"type": "exact_text"}, rounds=1,
            adaptive_search=True, max_optimization_cost_usd=5,
            progress_callback=events.append,
        )
        self.assertIn("model-6", {item.model for item in result.models})
        self.assertNotIn("model-6", result.pruned_models)
        self.assertEqual(result.quality_gate_status, "NO_CONFIGURATION_PASSED")
        deep_starts = [
            event["model"] for event in events if event["event"] == "model_started"
        ]
        self.assertEqual(deep_starts[0], "model-6")

    def test_adaptive_search_backfills_failed_deep_finalists(self):
        class BackfillClient(Client):
            def _evaluate_model(self, *args, **kwargs):
                model = args[4]
                if model in {"candidate-1", "candidate-2"}:
                    raise ProviderError("simulated provider failure after screening")
                return super()._evaluate_model(*args, **kwargs)

        class UniformTransport(FakeTransport):
            def estimate_cost(self, model, messages, *, max_tokens):
                if model.startswith("candidate-"):
                    return int(model.rsplit("-", 1)[1]) / 10_000
                return super().estimate_cost(model, messages, max_tokens=max_tokens)

            def complete(self, model, messages, *, max_tokens, response_schema=None):
                if model.startswith("candidate-"):
                    self.calls.append((model, messages, max_tokens, response_schema))
                    return Completion(
                        "billing", model, f"gen-{len(self.calls)}",
                        self.estimate_cost(model, messages, max_tokens=max_tokens),
                    )
                return super().complete(
                    model, messages, max_tokens=max_tokens, response_schema=response_schema,
                )

        examples = [
            {"id": f"billing-{index}", "input": f"charged twice {index}", "approved_output": "billing"}
            for index in range(15)
        ]
        result = BackfillClient(transport=UniformTransport()).optimize(
            prompt="Return the approved route label only.",
            examples=examples,
            models=[f"candidate-{index}" for index in range(1, 7)],
            optimizer_model="optimizer",
            evaluator_model="unused",
            evaluator={"type": "exact_text"},
            rounds=1,
            adaptive_search=True,
            max_optimization_cost_usd=5,
        )
        self.assertEqual(len(result.screening_results), 6)
        self.assertEqual(len(result.models), 4)
        self.assertEqual(result.pruned_models, [])
        self.assertEqual(
            {item["status"] for item in result.screening_results}, {"FULL"},
        )

    def test_successful_training_rewrite_propagates_before_cheap_models_are_pruned(self):
        class PropagationTransport(FakeTransport):
            def estimate_cost(self, model, messages, *, max_tokens):
                if model.startswith("candidate-"):
                    return int(model.rsplit("-", 1)[1]) / 100_000
                return super().estimate_cost(model, messages, max_tokens=max_tokens)

            def complete(self, model, messages, *, max_tokens, response_schema=None):
                self.calls.append((model, messages, max_tokens, response_schema))
                if model == "optimizer":
                    return Completion(
                        json.dumps({
                            "prompt": "Return billing for every supplied support request.",
                            "hypothesis": "The approved training set establishes the route.",
                            "few_shot_example_ids": [],
                        }),
                        model,
                        f"gen-{len(self.calls)}",
                        0.0001,
                    )
                system = messages[0]["content"]
                content = "billing" if "Return billing" in system else "wrong"
                return Completion(
                    content,
                    model,
                    f"gen-{len(self.calls)}",
                    self.estimate_cost(model, messages, max_tokens=max_tokens),
                )

        examples = [
            {"id": f"billing-{index}", "input": f"charged twice {index}", "approved_output": "billing"}
            for index in range(15)
        ]
        result = Client(transport=PropagationTransport()).optimize(
            prompt="Classify this request.",
            examples=examples,
            models=[f"candidate-{index}" for index in range(1, 7)],
            optimizer_model="optimizer",
            evaluator_model="unused",
            evaluator={"type": "exact_text"},
            rounds=1,
            adaptive_search=True,
            max_optimization_cost_usd=5,
        )
        propagated = [
            item for item in result.models
            if item.prompt_origin.startswith("propagated_from:")
        ]
        self.assertEqual(len(result.models), 6)
        self.assertEqual(len(propagated), 2)
        self.assertEqual(result.pruned_models, [])
        self.assertEqual(
            sum(item["status"] == "FULL_PROPAGATED" for item in result.screening_results),
            2,
        )
        self.assertTrue(all(item.holdout_pass_rate == 1 for item in propagated))

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
        self.assertIsNone(result.regression_suite["incumbent_model"])
        self.assertEqual(result.regression_suite["selected_model"], "cheap")
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
            evaluator_model="evaluator", max_optimization_cost_usd=0.004, rounds=1,
            max_parallel_models=1, max_parallel_scenarios=1,
        )
        self.assertEqual(result.winner.model, "cheap")
        self.assertTrue(result.incomplete_models)
        self.assertEqual(result.skipped_budget_models, ["later"])
        self.assertEqual(result.winner_scope, "Best among fully completed eligible targets only")
        self.assertTrue(result.continuation_recommendation["recommended"])
        self.assertEqual(
            result.continuation_recommendation["unfinished_configurations"],
            ["expensive", "later"],
        )
        self.assertEqual(
            result.continuation_recommendation["suggested_next_test_budget_usd"],
            0.25,
        )
        self.assertFalse(result.continuation_recommendation["automatic_spend"])

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
        self.assertEqual(
            result.regression_suite["winning_few_shot_provenance"],
            result.winner.few_shot_provenance,
        )
        self.assertTrue(result.winner.few_shot_provenance)
        self.assertEqual(
            {item["source_split"] for item in result.winner.few_shot_provenance},
            {"train"},
        )
        self.assertTrue(
            all(item["customer_approved"] for item in result.winner.few_shot_provenance)
        )
        self.assertEqual(result.winner.prompt_origin, "optimized_for:cheap")
        optimizer_system_prompts = [
            messages[0]["content"]
            for model, messages, _tokens, _schema in transport.calls
            if model == "optimizer"
        ]
        self.assertTrue(optimizer_system_prompts)
        self.assertTrue(all("Make the package self-contained" in text for text in optimizer_system_prompts))
        self.assertTrue(all("present at production time" in text for text in optimizer_system_prompts))

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
