import json
import os
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from evalt import (
    CommandScorer,
    CustomScorerError,
    Evalt,
    Example,
    ScoreRequest,
    ScoreResult,
    Suite,
    compare_results,
    render_html_report,
    render_junit_report,
)
from evalt.cli import _custom_scorer_registry, main as cli_main, parser
from evalt.dashboard import sanitize_route_snapshot
from last_good_prompt.core import Client, Completion


class FixedTransport:
    def __init__(self) -> None:
        self.calls = []

    def estimate_cost(self, model, messages, *, max_tokens):
        return 0.0001

    def complete(
        self, model, messages, *, max_tokens, response_schema=None,
        request_options=None,
    ):
        self.calls.append({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "response_schema": response_schema,
        })
        return Completion(
            content=str(messages[-1]["content"]).split(":", 1)[-1].strip(),
            model=model,
            generation_id=f"target-{len(self.calls)}",
            cost_usd=0.0001,
        )


class ExactLocalScorer:
    scorer_id = "domain-rubric"
    scorer_version = "1.0"
    thread_safe = True

    def __init__(self) -> None:
        self.requests = []

    def score(self, request: ScoreRequest) -> ScoreResult:
        self.requests.append(request)
        passed = request.actual_output == request.approved_output
        return ScoreResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            reason="Local domain rule matched." if passed else "Local domain rule differed.",
        )


def custom_suite() -> Suite:
    return Suite(
        name="custom-scorer-suite",
        prompt="Return the text after the colon.",
        examples=tuple(
            Example(f"case: label-{index}", f"label-{index}", f"case-{index}")
            for index in range(1, 7)
        ),
        models=("fixture/target",),
        evaluator_model="unused/custom",
        evaluator={
            "type": "custom",
            "scorer_id": "domain-rubric",
            "scorer_version": "1.0",
        },
        optimize_prompt=False,
        rounds=1,
        holdout_repeats=1,
        max_parallel_models=1,
        max_parallel_scenarios=1,
        max_optimization_cost_usd=0.10,
    )


class CustomScorerContractTests(unittest.TestCase):
    def test_custom_scorer_runs_end_to_end_at_zero_evaluator_cost(self):
        transport = FixedTransport()
        scorer = ExactLocalScorer()
        result = Evalt(
            transport=transport,
            custom_scorers={scorer.scorer_id: scorer},
            show_progress=False,
        ).run(custom_suite())
        self.assertEqual(result.winner.holdout_pass_rate, 1.0)
        self.assertTrue(result.winner.cases)
        self.assertTrue(
            all(case.evaluator_cost_usd == 0 for case in result.winner.cases)
        )
        self.assertTrue(
            all(
                case.evaluator_generation_id == "custom:domain-rubric@1.0"
                for case in result.winner.cases
            )
        )
        self.assertTrue(scorer.requests)
        self.assertEqual(
            result.regression_suite["evaluator"],
            {
                "type": "custom",
                "scorer_id": "domain-rubric",
                "scorer_version": "1.0",
            },
        )
        self.assertEqual(
            result.regression_suite["evaluator_model"],
            "custom/domain-rubric@1.0",
        )
        self.assertEqual(
            custom_suite().to_dict()["evaluator_model"],
            "custom/domain-rubric@1.0",
        )
        self.assertTrue(result.regression_suite["suite_hash"])

    def test_missing_registration_fails_before_any_provider_call(self):
        transport = FixedTransport()
        with self.assertRaisesRegex(ValueError, "not registered locally"):
            Evalt(transport=transport, show_progress=False).run(custom_suite())
        self.assertEqual(transport.calls, [])

    def test_version_mismatch_fails_before_any_provider_call(self):
        transport = FixedTransport()
        scorer = ExactLocalScorer()
        scorer.scorer_version = "2.0"
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            Evalt(
                transport=transport,
                custom_scorers={scorer.scorer_id: scorer},
                show_progress=False,
            ).run(custom_suite())
        self.assertEqual(transport.calls, [])

    def test_suite_cannot_select_a_command_or_module(self):
        for field, value in (
            ("command", ["python", "score.py"]),
            ("argv", ["python"]),
            ("module", "scorer"),
            ("path", "./score.py"),
            ("executable", "python"),
        ):
            suite = custom_suite()
            evaluator = dict(suite.evaluator)
            evaluator[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "cannot select executable code"
            ):
                Suite(
                    **{
                        **suite.__dict__,
                        "evaluator": evaluator,
                    }
                ).validate()

    def test_custom_scorer_identity_is_typed_and_nonempty(self):
        suite = custom_suite()
        for evaluator in (
            {"type": "custom", "scorer_id": None, "scorer_version": "1"},
            {"type": "custom", "scorer_id": "rubric", "scorer_version": None},
            {"type": "custom", "scorer_id": "", "scorer_version": "1"},
            {"type": "custom", "scorer_id": "rubric", "scorer_version": "bad version"},
        ):
            with self.subTest(evaluator=evaluator), self.assertRaises(ValueError):
                Suite(**{**suite.__dict__, "evaluator": evaluator}).validate()

    def test_invalid_result_shapes_fail_closed(self):
        invalid_values = [
            {"passed": True},
            {"passed": "yes", "score": 1},
            {"passed": True, "score": float("nan")},
            {"passed": True, "score": 1.1},
            {"passed": True, "score": 1, "extra": "no"},
            {"passed": True, "score": 1, "reason": None},
            {"passed": True, "score": 1, "reason": "x" * 2001},
        ]
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(CustomScorerError):
                ScoreResult.from_value(value)

    def test_monitor_requires_same_registration_before_provider_activity(self):
        scorer = ExactLocalScorer()
        baseline = Evalt(
            transport=FixedTransport(),
            custom_scorers={scorer.scorer_id: scorer},
            show_progress=False,
        ).run(custom_suite()).to_dict()
        monitor_transport = FixedTransport()
        with self.assertRaisesRegex(ValueError, "not registered locally"):
            Evalt(
                transport=monitor_transport, show_progress=False
            ).monitor(baseline, max_cost_usd=0.10)
        self.assertEqual(monitor_transport.calls, [])

    def test_monitor_replays_the_same_registered_custom_contract(self):
        scorer = ExactLocalScorer()
        baseline = Evalt(
            transport=FixedTransport(),
            custom_scorers={scorer.scorer_id: scorer},
            show_progress=False,
        ).run(custom_suite()).to_dict()
        monitor_scorer = ExactLocalScorer()
        monitored = Evalt(
            transport=FixedTransport(),
            custom_scorers={monitor_scorer.scorer_id: monitor_scorer},
            show_progress=False,
        ).monitor(
            baseline,
            max_cost_usd=0.10,
            source_suite=custom_suite(),
        )
        self.assertEqual(monitored.status, "HEALTHY")
        self.assertTrue(monitored.passed)
        self.assertTrue(monitor_scorer.requests)
        self.assertEqual(
            monitored.candidate.regression_suite["evaluator_model"],
            "custom/domain-rubric@1.0",
        )

    def test_offline_reports_and_comparison_preserve_scorer_lineage(self):
        scorer = ExactLocalScorer()
        payload = Evalt(
            transport=FixedTransport(),
            custom_scorers={scorer.scorer_id: scorer},
            show_progress=False,
        ).run(custom_suite()).to_dict()
        html = render_html_report(payload)
        junit = render_junit_report(payload)
        comparison = compare_results(payload, payload)
        self.assertIn("Custom / domain-rubric @ 1.0", html)
        self.assertIn('name="evaluator_type" value="custom"', junit)
        self.assertIn('name="custom_scorer_id" value="domain-rubric"', junit)
        self.assertEqual(
            comparison["contract"]["candidate_evaluator"],
            {
                "type": "custom",
                "label": "Custom / domain-rubric @ 1.0",
                "scorer_id": "domain-rubric",
                "scorer_version": "1.0",
            },
        )
        self.assertNotIn("custom_scorer.py", html + junit)

    def test_dashboard_sync_keeps_only_custom_type_and_keyed_lineage(self):
        snapshot = sanitize_route_snapshot({
            "route": "custom-route",
            "last_test_summary": {
                "evaluator_type": "custom",
                "evaluator_model": "custom/private-rubric-name@internal-build",
                "evaluator_contract_hash": "b" * 64,
                "evaluator_contract_id": "evaluator_" + "c" * 24,
                "scorer_id": "private-rubric-name",
                "scorer_version": "internal-build",
            },
        })
        summary = snapshot["last_test_summary"]
        self.assertEqual(summary["evaluator_type"], "custom")
        self.assertEqual(summary["evaluator_contract_id"], "evaluator_" + "c" * 24)
        self.assertNotIn("evaluator_contract_hash", summary)
        self.assertNotIn("evaluator_model", summary)
        self.assertNotIn("scorer_id", summary)
        self.assertNotIn("scorer_version", summary)


class CommandScorerTests(unittest.TestCase):
    @staticmethod
    def request() -> ScoreRequest:
        return ScoreRequest(
            scenario_id="case-1",
            turn=1,
            input="hello",
            transcript=({"role": "assistant", "content": "actual"},),
            approved_output="approved",
            actual_output="actual",
            group="support",
            difficulty="edge",
        )

    def command(self, source: str, **kwargs) -> CommandScorer:
        return CommandScorer(
            "command-rubric",
            "2026.07",
            [sys.executable, "-c", source],
            **kwargs,
        )

    def test_command_receives_strict_request_and_returns_strict_result(self):
        source = (
            "import json,sys;"
            "x=json.load(sys.stdin);"
            "print(json.dumps({'passed':x['schema']=='evalt-custom-score-request-v1',"
            "'score':0.75,'reason':x['scenario_id']}))"
        )
        result = self.command(source).score(self.request())
        self.assertEqual(result, ScoreResult(True, 0.75, "case-1"))

    def test_command_is_not_run_through_a_shell(self):
        source = (
            "import json,sys;"
            "json.load(sys.stdin);"
            "print(json.dumps({'passed':sys.argv[1]=='&&',"
            "'score':1,'reason':'argv preserved'}))"
        )
        scorer = CommandScorer(
            "command-rubric",
            "2026.07",
            [sys.executable, "-c", source, "&&", "echo", "unsafe"],
        )
        self.assertTrue(scorer.score(self.request()).passed)

    def test_provider_credentials_are_not_inherited_by_default(self):
        source = (
            "import json,os,sys;"
            "json.load(sys.stdin);"
            "print(json.dumps({'passed':'OPENROUTER_API_KEY' not in os.environ,"
            "'score':1,'reason':'minimal environment'}))"
        )
        with mock.patch.dict(
            os.environ, {"OPENROUTER_API_KEY": "must-not-cross-boundary"}, clear=False
        ):
            self.assertTrue(self.command(source).score(self.request()).passed)

    def test_timeout_nonzero_malformed_and_oversized_output_fail_closed(self):
        cases = [
            (
                "timeout",
                self.command(
                    "import time;time.sleep(1)",
                    timeout_seconds=0.05,
                ),
                "timed out",
            ),
            (
                "nonzero",
                self.command("import sys;sys.stderr.write('no');sys.exit(7)"),
                "exited with code 7",
            ),
            (
                "malformed",
                self.command("print('not-json')"),
                "valid JSON object",
            ),
            (
                "oversized",
                self.command("print('x'*1000)", max_output_bytes=100),
                "stdout exceeded",
            ),
        ]
        for name, scorer, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(
                CustomScorerError, message
            ):
                scorer.score(self.request())

    def test_oversized_input_and_stderr_fail_closed(self):
        with self.assertRaisesRegex(CustomScorerError, "request exceeded"):
            self.command(
                "print('{}')", max_input_bytes=10
            ).score(self.request())
        source = (
            "import json,sys;"
            "json.load(sys.stdin);"
            "sys.stderr.write('x'*1000);"
            "print(json.dumps({'passed':True,'score':1}))"
        )
        with self.assertRaisesRegex(CustomScorerError, "stderr exceeded"):
            self.command(source, max_output_bytes=100).score(self.request())


class CustomScorerCliTests(unittest.TestCase):
    def test_cli_registration_is_explicit_argv_not_suite_data(self):
        args = parser().parse_args([
            "optimize",
            "suite.json",
            "--custom-scorer-id", "domain-rubric",
            "--custom-scorer-version", "1.0",
            "--custom-scorer-executable", sys.executable,
            "--custom-scorer-arg=-c",
            "--custom-scorer-arg", "print('{}')",
        ])
        registry = _custom_scorer_registry(args)
        scorer = registry["domain-rubric"]
        self.assertEqual(
            scorer.argv,
            (sys.executable, "-c", "print('{}')"),
        )

    def test_partial_cli_registration_is_rejected(self):
        args = parser().parse_args([
            "monitor",
            "baseline.json",
            "--max-cost-usd", "0.1",
            "--custom-scorer-id", "domain-rubric",
        ])
        with self.assertRaisesRegex(ValueError, "requires"):
            _custom_scorer_registry(args)

    def test_cli_resolves_custom_registration_before_provider_setup(self):
        class UnexpectedEvalt:
            def __init__(self, **_kwargs):
                raise AssertionError("provider setup must not begin")

        with TemporaryDirectory() as directory:
            suite_path = Path(directory) / "suite.json"
            custom_suite().save(suite_path)
            stderr = StringIO()
            with mock.patch("evalt.cli.Evalt", UnexpectedEvalt), redirect_stderr(stderr):
                code = cli_main([
                    "optimize",
                    str(suite_path),
                    "--output",
                    str(Path(directory) / "must-not-exist.json"),
                ])

            self.assertEqual(code, 2)
            self.assertIn("not registered locally", stderr.getvalue())

    def test_cli_writes_a_typed_failure_without_a_traceback(self):
        class FailingEvalt:
            def __init__(self, **_kwargs):
                pass

            def run(self, _suite):
                raise CustomScorerError("Custom scorer timed out.")

        with TemporaryDirectory() as directory:
            suite_path = Path(directory) / "suite.json"
            output_path = Path(directory) / "result.json"
            custom_suite().save(suite_path)
            stderr = StringIO()
            with mock.patch("evalt.cli.Evalt", FailingEvalt), redirect_stderr(stderr):
                code = cli_main([
                    "optimize",
                    str(suite_path),
                    "--output",
                    str(output_path),
                    "--custom-scorer-id",
                    "domain-rubric",
                    "--custom-scorer-version",
                    "1.0",
                    "--custom-scorer-executable",
                    sys.executable,
                ])
            self.assertEqual(code, 2)
            failure = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(failure["error_type"], "CustomScorerError")
            self.assertIn("may still be billable", failure["provider_spend_note"])
            self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
