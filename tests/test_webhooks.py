from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import evalt.webhooks as webhook_module
from evalt import (
    WebhookConfigurationError,
    WebhookDestination,
    WebhookEvent,
    ci_gate_event,
    deliver_webhook,
    replay_webhook,
    route_health_event,
)
from evalt.cli import _webhook_destination, parser


PUBLIC_ADDRESS = [
    (2, 1, 6, "", ("93.184.216.34", 443)),
]
PRIVATE_ADDRESS = [
    (2, 1, 6, "", ("127.0.0.1", 443)),
]


class FixtureTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, **request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def resolver(rows):
    return lambda *_args, **_kwargs: rows


def destination(directory: str, **overrides):
    values = {
        "url": "https://hooks.example.test/evalt",
        "secret": "fixture-secret-at-least-16-bytes",
        "destination_id": "incident-pipeline",
        "audit_path": Path(directory) / "webhook-audit.jsonl",
        "backoff_seconds": 0,
    }
    values.update(overrides)
    return WebhookDestination(**values)


def event(event_id: str = "evt_" + "a" * 32):
    return WebhookEvent(
        schema="evalt-webhook-event-v1",
        event_id=event_id,
        idempotency_key=event_id,
        type="route.health.degraded",
        occurred_at="2026-07-23T16:00:00+00:00",
        source="evalt.monitor",
        data={
            "route_ref": "route_fixture",
            "status": "REGRESSION",
            "regressions": 2,
        },
    )


def monitor_result(status: str, directory: str):
    winner = SimpleNamespace(
        holdout_pass_rate=1.0 if status == "HEALTHY" else 0.6,
        target_latency_p90_ms=25,
        holdout_unique_scenarios=5,
        holdout_executions=10,
    )
    candidate = SimpleNamespace(
        winner=winner,
        regression_suite={"suite_hash": "b" * 64},
    )
    return SimpleNamespace(
        route="billing-route",
        status=status,
        regression_gate={
            "quality_delta_percentage_points": 0 if status == "HEALTHY" else -40,
            "regressions": 0 if status == "HEALTHY" else 2,
            "missing_cases": 0,
            "cost_increase_percent": 0,
            "p90_increase_ms": 0,
        },
        candidate=candidate,
        provider_spend_usd=0.001,
        finished_at=(
            "2026-07-23T16:00:00+00:00"
            if status == "REGRESSION"
            else "2026-07-23T16:05:00+00:00"
        ),
        directory=directory,
    )


class WebhookTests(unittest.TestCase):
    def test_signed_delivery_is_bounded_and_audit_omits_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            target = destination(directory)
            transport = FixtureTransport([(204, {})])
            delivered = deliver_webhook(
                target,
                event(),
                transport=transport,
                resolver=resolver(PUBLIC_ADDRESS),
            )

            self.assertTrue(delivered.delivered)
            self.assertEqual(len(transport.requests), 1)
            request = transport.requests[0]
            body = request["body"]
            timestamp = request["headers"]["X-Evalt-Timestamp"]
            expected = hmac.new(
                target.secret.encode(),
                timestamp.encode() + b"." + body,
                hashlib.sha256,
            ).hexdigest()
            self.assertEqual(
                request["headers"]["X-Evalt-Signature"],
                f"sha256={expected}",
            )
            self.assertEqual(request["headers"]["Idempotency-Key"], event().event_id)
            self.assertLessEqual(len(body), target.max_body_bytes)
            audit = Path(target.audit_path).read_text(encoding="utf-8")
            self.assertNotIn(target.url, audit)
            self.assertNotIn(target.secret, audit)
            for forbidden in (
                "prompt",
                "approved_output",
                "image_url",
                "scorer_id",
                "scorer_code",
                "provider_key",
                "response_body",
            ):
                self.assertNotIn(forbidden, audit.casefold())

    def test_private_address_is_denied_before_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            target = destination(directory, max_attempts=1)
            transport = FixtureTransport([(204, {})])
            delivered = deliver_webhook(
                target,
                event(),
                transport=transport,
                resolver=resolver(PRIVATE_ADDRESS),
            )

            self.assertFalse(delivered.delivered)
            self.assertEqual(delivered.attempts[0].result, "unsafe_or_unresolved_destination")
            self.assertEqual(transport.requests, [])

    def test_redirect_is_rejected_and_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = FixtureTransport([(302, {"location": "https://other.test"})])
            delivered = deliver_webhook(
                destination(directory),
                event(),
                transport=transport,
                resolver=resolver(PUBLIC_ADDRESS),
            )

            self.assertFalse(delivered.delivered)
            self.assertEqual(len(delivered.attempts), 1)
            self.assertEqual(delivered.attempts[0].result, "redirect_rejected")

    def test_retry_after_is_capped_and_event_identity_stays_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            pauses = []
            transport = FixtureTransport([
                (429, {"retry-after": "999"}),
                (503, {}),
                (202, {}),
            ])
            target = destination(
                directory,
                max_retry_after_seconds=2,
                backoff_seconds=0.5,
            )
            delivered = deliver_webhook(
                target,
                event(),
                transport=transport,
                resolver=resolver(PUBLIC_ADDRESS),
                sleep=pauses.append,
            )

            self.assertTrue(delivered.delivered)
            self.assertEqual(pauses, [2, 1.0])
            self.assertEqual(
                {request["headers"]["Idempotency-Key"] for request in transport.requests},
                {event().event_id},
            )
            self.assertEqual(
                {request["body"] for request in transport.requests},
                {transport.requests[0]["body"]},
            )

    def test_replay_preserves_exact_event_and_idempotency_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            target = destination(directory, max_attempts=1)
            first_transport = FixtureTransport([(500, {})])
            first = deliver_webhook(
                target,
                event(),
                transport=first_transport,
                resolver=resolver(PUBLIC_ADDRESS),
            )
            replay_transport = FixtureTransport([(204, {})])
            replayed = replay_webhook(
                target,
                event_id=first.event_id,
                transport=replay_transport,
                resolver=resolver(PUBLIC_ADDRESS),
            )

            self.assertFalse(first.delivered)
            self.assertTrue(replayed.delivered)
            self.assertTrue(replayed.replay)
            self.assertEqual(replayed.event_id, first.event_id)
            self.assertEqual(
                replay_transport.requests[0]["headers"]["Idempotency-Key"],
                first.event_id,
            )
            lines = Path(target.audit_path).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(json.loads(lines[-1])["replay"])

    def test_replay_reads_only_a_bounded_audit_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            target = destination(directory, max_attempts=1)
            Path(target.audit_path).write_text("x" * 4096 + "\n", encoding="utf-8")
            first = deliver_webhook(
                target,
                event(),
                transport=FixtureTransport([(500, {})]),
                resolver=resolver(PUBLIC_ADDRESS),
            )
            with mock.patch.object(
                webhook_module, "_MAX_AUDIT_READ_BYTES", 2048
            ):
                replayed = replay_webhook(
                    target,
                    event_id=first.event_id,
                    transport=FixtureTransport([(204, {})]),
                    resolver=resolver(PUBLIC_ADDRESS),
                )

            self.assertTrue(replayed.delivered)
            self.assertTrue(replayed.replay)

    def test_route_health_transition_names_degradation_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            target = destination(directory)
            degraded = route_health_event(
                monitor_result("REGRESSION", directory), target
            )
            self.assertEqual(degraded.type, "route.health.degraded")
            deliver_webhook(
                target,
                degraded,
                transport=FixtureTransport([(204, {})]),
                resolver=resolver(PUBLIC_ADDRESS),
            )
            recovered = route_health_event(
                monitor_result("HEALTHY", directory), target
            )
            self.assertEqual(recovered.type, "route.health.recovered")
            self.assertEqual(recovered.data["previous_status"], "REGRESSION")
            self.assertNotIn("route", recovered.data)

    def test_ci_event_contains_only_aggregate_decision_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            target = destination(directory)
            built = ci_gate_event(
                status="FAIL",
                result={
                    "route": "private-customer-route",
                    "winner": {
                        "holdout_pass_rate": 0.8,
                        "prompt": "never send this",
                    },
                    "regression_suite": {
                        "suite_hash": "c" * 64,
                        "examples": ["never send this either"],
                    },
                },
                gate={
                    "baseline_gate": {
                        "regressions": 1,
                        "quality_delta_percentage_points": -10,
                    },
                    "failures": ["private case detail"],
                },
                destination=target,
                occurred_at="2026-07-23T16:00:00+00:00",
            )

            serialized = json.dumps(built.to_dict())
            self.assertEqual(built.type, "ci.gate.fail")
            self.assertNotIn("private-customer-route", serialized)
            self.assertNotIn("never send", serialized)
            self.assertNotIn("private case detail", serialized)
            self.assertEqual(built.data["regressions"], 1)

    def test_configuration_fails_closed_without_leaking_secret(self):
        target = WebhookDestination(
            url="http://127.0.0.1/hook",
            secret="fixture-secret-at-least-16-bytes",
        )
        with self.assertRaises(WebhookConfigurationError):
            target.validate()
        self.assertNotIn(target.secret, repr(target))

    def test_cli_reads_the_secret_value_only_from_the_named_environment(self):
        args = parser().parse_args([
            "monitor",
            "baseline.json",
            "--max-cost-usd",
            "0.10",
            "--webhook-url",
            "https://hooks.example.test/evalt",
            "--webhook-secret-env",
            "PRIVATE_ALERT_SECRET",
            "--webhook-destination-id",
            "incident-pipeline",
            "--webhook-required",
        ])
        with mock.patch.dict(
            "os.environ",
            {"PRIVATE_ALERT_SECRET": "fixture-secret-at-least-16-bytes"},
            clear=True,
        ):
            configured = _webhook_destination(args)

        self.assertIsNotNone(configured)
        self.assertEqual(configured.destination_id, "incident-pipeline")
        self.assertNotIn(configured.secret, repr(configured))


if __name__ == "__main__":
    unittest.main()
