from __future__ import annotations

import copy
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from evalt import (
    BudgetExceeded,
    Evalt,
    Example,
    ImageInput,
    ProviderError,
    Suite,
    multimodal_input,
)
from last_good_prompt.core import Completion
from evalt.cli import main as cli_main
from evalt.dashboard import sanitize_progress_event


EXAMPLES = [
    {"id": "billing-1", "input": "billing charge", "approved_output": "billing"},
    {"id": "billing-2", "input": "send invoice", "approved_output": "billing"},
    {"id": "billing-3", "input": "refund charge", "approved_output": "billing"},
    {"id": "account-1", "input": "login failed", "approved_output": "account"},
    {"id": "account-2", "input": "reset password", "approved_output": "account"},
]


class LabelTransport:
    def __init__(
        self,
        *,
        wrong: bool = False,
        estimated_cost: float = 0.0001,
        latency_ms: int = 25,
    ):
        self.calls: list[dict] = []
        self.wrong = wrong
        self.estimated_cost = estimated_cost
        self.latency_ms = latency_ms

    def estimate_cost(self, _model, _messages, *, max_tokens):
        return self.estimated_cost

    def complete(
        self,
        model,
        messages,
        *,
        max_tokens,
        response_schema=None,
        request_options=None,
    ):
        self.calls.append({
            "model": model,
            "messages": copy.deepcopy(messages),
            "max_tokens": max_tokens,
            "response_schema": copy.deepcopy(response_schema),
            "request_options": copy.deepcopy(request_options),
        })
        text = str(messages[-1]["content"]).casefold()
        answer = (
            "billing"
            if any(value in text for value in ("billing", "charge", "invoice", "refund"))
            else "account"
        )
        if self.wrong:
            answer = "account" if answer == "billing" else "billing"
        return Completion(
            content=answer,
            model=model,
            generation_id=f"generation-{len(self.calls)}",
            cost_usd=self.estimated_cost,
            latency_ms=self.latency_ms,
        )


class ImageTransport(LabelTransport):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    def complete(
        self,
        model,
        messages,
        *,
        max_tokens,
        response_schema=None,
        request_options=None,
    ):
        self.calls.append({
            "model": model,
            "messages": copy.deepcopy(messages),
            "max_tokens": max_tokens,
            "response_schema": copy.deepcopy(response_schema),
            "request_options": copy.deepcopy(request_options),
        })
        content = messages[-1]["content"]
        image_part = next(
            part for part in content if part.get("type") == "image_url"
        )
        encoded = image_part["image_url"]["url"].split(";base64,", 1)[1]
        digest = hashlib.sha256(base64.b64decode(encoded)).hexdigest()
        return Completion(
            content=self.labels[digest],
            model=model,
            generation_id=f"image-{len(self.calls)}",
            cost_usd=self.estimated_cost,
            latency_ms=40,
        )


class ConversationTransport(LabelTransport):
    def complete(
        self,
        model,
        messages,
        *,
        max_tokens,
        response_schema=None,
        request_options=None,
    ):
        self.calls.append({
            "model": model,
            "messages": copy.deepcopy(messages),
            "max_tokens": max_tokens,
            "response_schema": copy.deepcopy(response_schema),
            "request_options": copy.deepcopy(request_options),
        })
        user = str(messages[-1]["content"])
        if user.startswith("Remember code "):
            answer = "stored " + user.rsplit(" ", 1)[-1]
        else:
            prior = next(
                item["content"]
                for item in reversed(messages[:-1])
                if item.get("role") == "assistant"
                and str(item.get("content")).startswith("stored ")
            )
            answer = prior.rsplit(" ", 1)[-1]
        return Completion(
            content=answer,
            model=model,
            generation_id=f"conversation-{len(self.calls)}",
            cost_usd=self.estimated_cost,
            latency_ms=15,
        )


class SemanticTransport(LabelTransport):
    def complete(
        self,
        model,
        messages,
        *,
        max_tokens,
        response_schema=None,
        request_options=None,
    ):
        if response_schema:
            self.calls.append({
                "model": model,
                "messages": copy.deepcopy(messages),
                "max_tokens": max_tokens,
                "response_schema": copy.deepcopy(response_schema),
                "request_options": copy.deepcopy(request_options),
            })
            judgment_input = json.loads(messages[-1]["content"])
            passed = (
                judgment_input["actual_answer"].strip()
                == judgment_input["approved_answer"].strip()
            )
            return Completion(
                content=json.dumps({
                    "passed": passed,
                    "score": 1 if passed else 0,
                    "reason": "fixture semantic judgment",
                }),
                model=model,
                generation_id=f"judge-{len(self.calls)}",
                cost_usd=self.estimated_cost,
                latency_ms=self.latency_ms,
            )
        return super().complete(
            model,
            messages,
            max_tokens=max_tokens,
            response_schema=response_schema,
            request_options=request_options,
        )


class FailingTransport(LabelTransport):
    def complete(self, *args, **kwargs):
        self.calls.append({"attempted": True})
        raise ProviderError("fixture provider unavailable")


class AggregateSync:
    def __init__(self):
        self.events = []

    def publish_event(self, event):
        self.events.append(sanitize_progress_event(event))

    def flush(self, timeout_seconds=10):
        return True


def suite() -> Suite:
    return Suite.from_dict({
        "schema": "evalt-suite-v2",
        "name": "monitor-fixture",
        "prompt": "Return exactly billing or account.",
        "examples": EXAMPLES,
        "models": ["fixture/model"],
        "optimizer_model": "fixture/optimizer",
        "evaluator_model": "fixture/evaluator",
        "evaluator": {"type": "exact_text"},
        "quality_threshold": 0.8,
        "max_optimization_cost_usd": 1,
        "rounds": 1,
        "optimize_prompt": False,
        "holdout_repeats": 2,
        "request_options": {"temperature": 0},
        "target_max_tokens": 16,
    })


def baseline_result():
    transport = LabelTransport()
    result = Evalt(transport=transport, show_progress=False).run(suite())
    return result.to_dict()


class MonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = baseline_result()

    def test_rechecks_only_frozen_final_test_and_returns_healthy(self):
        transport = LabelTransport()
        evalt = Evalt(transport=transport, show_progress=False)
        result = evalt.monitor(self.baseline, max_cost_usd=0.10)

        self.assertEqual(result.status, "HEALTHY")
        self.assertTrue(result.passed)
        self.assertEqual(
            len(transport.calls),
            self.baseline["regression_suite"]["holdout_unique_scenarios"]
            * self.baseline["regression_suite"]["holdout_repeats"],
        )
        self.assertTrue(
            all(
                call["messages"][0]["content"]
                == self.baseline["regression_suite"]["winning_prompt"]
                for call in transport.calls
            )
        )
        self.assertTrue(
            all(call["model"] == "fixture/model" for call in transport.calls)
        )
        self.assertTrue(
            all(call["request_options"] == {"temperature": 0} for call in transport.calls)
        )
        self.assertEqual(result.candidate.winner.prompt_origin, "frozen_monitor")
        self.assertFalse(result.candidate.winner.selected_prompt_changed)
        self.assertFalse(result.to_dict()["route_mutated"])
        self.assertFalse(result.to_dict()["dashboard_sync_started"])

    def test_detects_measured_regression_without_changing_contract(self):
        result = Evalt(
            transport=LabelTransport(wrong=True), show_progress=False
        ).monitor(self.baseline, max_cost_usd=0.10)

        self.assertEqual(result.status, "REGRESSION")
        self.assertFalse(result.passed)
        self.assertGreater(result.regression_gate["regressions"], 0)
        self.assertEqual(
            result.candidate.regression_suite["suite_hash"],
            self.baseline["regression_suite"]["suite_hash"],
        )

    def test_tampered_contract_fails_before_provider_call(self):
        tampered = copy.deepcopy(self.baseline)
        tampered["regression_suite"]["winning_prompt"] = "Changed after qualification."
        transport = LabelTransport()

        with self.assertRaisesRegex(ValueError, "modified after qualification"):
            Evalt(transport=transport, show_progress=False).monitor(
                tampered, max_cost_usd=0.10
            )
        self.assertEqual(transport.calls, [])

    def test_non_positive_or_non_finite_cap_fails_before_provider_call(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                transport = LabelTransport()
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    Evalt(transport=transport, show_progress=False).monitor(
                        self.baseline, max_cost_usd=value
                    )
                self.assertEqual(transport.calls, [])

    def test_budget_reservation_fails_before_unapproved_call(self):
        transport = LabelTransport(estimated_cost=1)
        with self.assertRaises(BudgetExceeded):
            Evalt(transport=transport, show_progress=False).monitor(
                self.baseline, max_cost_usd=0.10
            )
        self.assertEqual(transport.calls, [])

    def test_provider_failure_is_not_mislabeled_as_a_regression(self):
        transport = FailingTransport()
        with self.assertRaisesRegex(ProviderError, "fixture provider unavailable"):
            Evalt(transport=transport, show_progress=False).monitor(
                self.baseline, max_cost_usd=0.10
            )
        self.assertGreaterEqual(len(transport.calls), 1)

    def test_cost_and_latency_policy_can_fail_a_quality_stable_route(self):
        result = Evalt(
            transport=LabelTransport(estimated_cost=0.0002, latency_ms=100),
            show_progress=False,
        ).monitor(
            self.baseline,
            max_cost_usd=0.10,
            max_cost_increase_percent=10,
            max_p90_increase_ms=10,
        )

        self.assertEqual(result.candidate.winner.holdout_pass_rate, 1)
        self.assertEqual(result.status, "REGRESSION")
        self.assertTrue(
            any("cost" in failure for failure in result.regression_gate["failures"])
        )
        self.assertTrue(
            any("latency" in failure for failure in result.regression_gate["failures"])
        )

    def test_history_is_aggregate_only(self):
        result = Evalt(
            transport=LabelTransport(), show_progress=False
        ).monitor(self.baseline, max_cost_usd=0.10)
        history = result.history_record()
        serialized = json.dumps(history).casefold()

        self.assertEqual(history["schema"], "evalt-monitor-history-v1")
        for forbidden in (
            "winning_prompt",
            "approved_output",
            "input",
            "output",
            "reason",
            "generation_id",
            "messages",
            "image_url",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_saved_monitor_result_remains_check_and_compare_compatible(self):
        from evalt.reporting import check_regression, compare_results

        result = Evalt(
            transport=LabelTransport(), show_progress=False
        ).monitor(self.baseline, max_cost_usd=0.10)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "monitor.json"
            result.save(target)
            payload = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema"], "evalt-monitor-result-v1")
        self.assertIn("winner", payload)
        self.assertIn("regression_suite", payload)
        self.assertEqual(payload["monitor_status"], "HEALTHY")
        comparison = compare_results(self.baseline, payload)
        self.assertTrue(comparison["comparable_contract"])
        self.assertEqual(comparison["case_summary"]["regressions"], 0)
        self.assertTrue(check_regression(self.baseline, payload).passed)

    def test_image_monitor_requires_and_verifies_original_local_suite(self):
        png_a = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1Pe"
            "AAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
        )
        png_b = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1Pe"
            "AAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
        )
        labels = {
            hashlib.sha256(png_a).hexdigest(): "damaged",
            hashlib.sha256(png_b).hexdigest(): "intact",
        }
        image_examples = tuple(
            Example(
                multimodal_input(
                    "Return damaged or intact.",
                    ImageInput.from_url(
                        "data:image/png;base64,"
                        + base64.b64encode(
                            png_a if index % 2 == 0 else png_b
                        ).decode("ascii")
                    ),
                ),
                "damaged" if index % 2 == 0 else "intact",
                f"image-{index}",
            )
            for index in range(5)
        )
        image_suite = Suite(
            name="image-monitor",
            prompt="Inspect the image and return damaged or intact.",
            examples=image_examples,
            models=("fixture/vision",),
            optimizer_model="fixture/optimizer",
            evaluator_model="fixture/evaluator",
            evaluator={"type": "exact_text"},
            quality_threshold=0.8,
            max_optimization_cost_usd=1,
            rounds=1,
            optimize_prompt=False,
            holdout_repeats=1,
            max_parallel_models=1,
            max_parallel_scenarios=1,
            target_max_tokens=16,
        )
        baseline = Evalt(
            transport=ImageTransport(labels), show_progress=False
        ).run(image_suite).to_dict()
        self.assertFalse(
            baseline["regression_suite"]["examples"][0]["input_replayable"]
        )
        self.assertNotIn("data:image", json.dumps(baseline))

        missing_source_transport = ImageTransport(labels)
        with self.assertRaisesRegex(ValueError, "original reviewed Suite"):
            Evalt(
                transport=missing_source_transport, show_progress=False
            ).monitor(baseline, max_cost_usd=0.10)
        self.assertEqual(missing_source_transport.calls, [])

        monitor_transport = ImageTransport(labels)
        monitored = Evalt(
            transport=monitor_transport, show_progress=False
        ).monitor(
            baseline,
            max_cost_usd=0.10,
            source_suite=image_suite,
        )
        self.assertEqual(monitored.status, "HEALTHY")
        self.assertTrue(monitor_transport.calls)
        self.assertNotIn("data:image", json.dumps(monitored.to_dict()))
        self.assertNotIn("data:image", json.dumps(monitored.history_record()))

    def test_multi_turn_monitor_replays_frozen_conversation_context(self):
        conversation_suite = Suite.from_dict({
            "schema": "evalt-suite-v2",
            "name": "conversation-monitor",
            "prompt": "Remember each code and answer the follow-up exactly.",
            "examples": [
                {
                    "id": f"conversation-{index}",
                    "turns": [
                        {
                            "input": f"Remember code {index}",
                            "approved_output": f"stored {index}",
                        },
                        {"input": "What code?", "approved_output": str(index)},
                    ],
                }
                for index in range(5)
            ],
            "models": ["fixture/conversation"],
            "optimizer_model": "fixture/optimizer",
            "evaluator_model": "fixture/evaluator",
            "evaluator": {"type": "exact_text"},
            "quality_threshold": 0.8,
            "max_optimization_cost_usd": 1,
            "rounds": 1,
            "optimize_prompt": False,
            "holdout_repeats": 1,
            "target_max_tokens": 16,
        })
        baseline = Evalt(
            transport=ConversationTransport(), show_progress=False
        ).run(conversation_suite).to_dict()
        transport = ConversationTransport()
        monitored = Evalt(
            transport=transport, show_progress=False
        ).monitor(baseline, max_cost_usd=0.10)

        self.assertEqual(monitored.status, "HEALTHY")
        follow_up_calls = [
            call for call in transport.calls
            if call["messages"][-1]["content"] == "What code?"
        ]
        self.assertTrue(follow_up_calls)
        self.assertTrue(
            all(
                any(message["role"] == "assistant" for message in call["messages"][:-1])
                for call in follow_up_calls
            )
        )

    def test_semantic_monitor_preserves_independent_evaluator_model(self):
        semantic_suite = Suite.from_dict({
            **suite().to_dict(),
            "name": "semantic-monitor",
            "evaluator": {"type": "semantic"},
        })
        baseline = Evalt(
            transport=SemanticTransport(), show_progress=False
        ).run(semantic_suite).to_dict()
        transport = SemanticTransport()
        monitored = Evalt(
            transport=transport, show_progress=False
        ).monitor(baseline, max_cost_usd=0.10)

        self.assertEqual(monitored.status, "HEALTHY")
        target_models = {
            call["model"]
            for call in transport.calls
            if not call.get("response_schema")
        }
        judge_models = {
            call["model"]
            for call in transport.calls
            if call.get("response_schema")
        }
        self.assertEqual(target_models, {"fixture/model"})
        self.assertEqual(judge_models, {"fixture/evaluator"})

    def test_monitor_never_opens_or_changes_the_route_database(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "evalt.db"
            original = b"route state must remain byte-for-byte unchanged"
            state.write_bytes(original)
            result = Evalt(
                transport=LabelTransport(),
                state_path=state,
                show_progress=False,
            ).monitor(self.baseline, max_cost_usd=0.10)

            self.assertEqual(result.status, "HEALTHY")
            self.assertEqual(state.read_bytes(), original)

    def test_named_monitor_syncs_only_aggregate_route_health_metadata(self):
        evalt = Evalt(transport=LabelTransport(), show_progress=False)
        sync = AggregateSync()
        evalt._dashboard_sync = sync
        monitored = evalt.monitor(
            self.baseline,
            max_cost_usd=0.10,
            route="support-routing",
        )

        self.assertTrue(monitored.dashboard_sync_started)
        self.assertTrue(monitored.dashboard_sync_succeeded)
        self.assertEqual(monitored.route, "support-routing")
        self.assertEqual(
            monitored.history_record()["route"], "support-routing"
        )
        self.assertEqual(len(sync.events), 1)
        event = sync.events[0]
        self.assertEqual(event["event"], "route_health_checked")
        self.assertEqual(event["status"], "HEALTHY")
        self.assertEqual(event["regressions"], 0)
        serialized = json.dumps(event).casefold()
        for forbidden in (
            "prompt",
            "input",
            "output",
            "approved",
            "messages",
            "generation",
            "image",
            "suite_hash",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_cli_writes_result_and_aggregate_history_with_exit_zero(self):
        monitored = Evalt(
            transport=LabelTransport(), show_progress=False
        ).monitor(self.baseline, max_cost_usd=0.10)
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            output = Path(directory) / "monitor.json"
            history = Path(directory) / "history.jsonl"
            baseline.write_text(
                json.dumps(self.baseline), encoding="utf-8"
            )
            with mock.patch("evalt.cli.Evalt") as evalt_class:
                evalt_class.return_value.monitor.return_value = monitored
                with redirect_stdout(StringIO()) as stdout:
                    code = cli_main([
                        "monitor",
                        str(baseline),
                        "--max-cost-usd",
                        "0.10",
                        "--output",
                        str(output),
                        "--history",
                        str(history),
                    ])

            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads(stdout.getvalue())["status"], "HEALTHY"
            )
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["monitor_status"],
                "HEALTHY",
            )
            history_payload = json.loads(
                history.read_text(encoding="utf-8")
            )
            self.assertEqual(
                history_payload["schema"], "evalt-monitor-history-v1"
            )
            self.assertNotIn(
                "approved_output", history.read_text(encoding="utf-8")
            )

    def test_cli_tamper_and_invalid_cap_fail_before_client_construction(self):
        tampered = copy.deepcopy(self.baseline)
        tampered["regression_suite"]["winning_prompt"] = "tampered"
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "tampered.json"
            baseline.write_text(json.dumps(tampered), encoding="utf-8")
            with mock.patch("evalt.cli.Evalt") as evalt_class:
                with redirect_stderr(StringIO()):
                    tamper_code = cli_main([
                        "monitor",
                        str(baseline),
                        "--max-cost-usd",
                        "0.10",
                    ])
                    cap_code = cli_main([
                        "monitor",
                        str(baseline),
                        "--max-cost-usd",
                        "0",
                    ])

            self.assertEqual(tamper_code, 2)
            self.assertEqual(cap_code, 2)
            evalt_class.assert_not_called()

    def test_cli_provider_failure_preserves_machine_readable_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            output = Path(directory) / "failure.json"
            baseline.write_text(json.dumps(self.baseline), encoding="utf-8")
            with mock.patch("evalt.cli.Evalt") as evalt_class:
                evalt_class.return_value.monitor.side_effect = ProviderError(
                    "fixture provider unavailable"
                )
                with redirect_stderr(StringIO()):
                    code = cli_main([
                        "monitor",
                        str(baseline),
                        "--max-cost-usd",
                        "0.10",
                        "--output",
                        str(output),
                    ])

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(code, 2)
            self.assertEqual(payload["schema"], "evalt-monitor-failure-v1")
            self.assertEqual(payload["status"], "ERROR")
            self.assertIsNone(payload["provider_spend_usd"])
            self.assertFalse(payload["route_mutated"])


if __name__ == "__main__":
    unittest.main()
