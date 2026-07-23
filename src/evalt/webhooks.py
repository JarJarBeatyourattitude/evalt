"""Signed, privacy-safe outbound events for Evalt decisions.

Webhook destinations are an explicit runtime capability. URLs and secrets never enter
suite files, result files, dashboard synchronization, or the delivery audit. Event
bodies contain only aggregate decision evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import ssl
import time
from typing import Any, Callable, Iterable, Mapping, Protocol
from urllib.parse import urlsplit

import certifi


_DESTINATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_EVENT_ID = re.compile(r"^evt_[0-9a-f]{32}$")
_SECRET_ENV = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_MAX_RESPONSE_BYTES = 4096
_MAX_AUDIT_READ_BYTES = 16 * 1024 * 1024
_MAX_AUDIT_RECORDS = 5000
_RETRYABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504}


class WebhookError(RuntimeError):
    """Base error for webhook configuration, delivery, and audit failures."""


class WebhookConfigurationError(WebhookError, ValueError):
    """The destination is unsafe or incomplete."""


class WebhookDeliveryError(WebhookError):
    """A delivery could not be completed."""


@dataclass(frozen=True)
class WebhookDestination:
    """One explicitly trusted HTTPS destination.

    ``secret`` is intentionally excluded from repr and every serialized artifact.
    """

    url: str = field(repr=False)
    secret: str = field(repr=False)
    destination_id: str = "default"
    audit_path: str | Path = ".evalt/webhook-deliveries.jsonl"
    timeout_seconds: float = 5.0
    max_attempts: int = 3
    backoff_seconds: float = 0.25
    max_retry_after_seconds: float = 30.0
    max_body_bytes: int = 16_384
    allow_private_network: bool = False
    include_route_name: bool = False

    def validate(self) -> None:
        parsed = urlsplit(str(self.url))
        if parsed.scheme != "https":
            raise WebhookConfigurationError("Webhook URL must use https://.")
        if not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
            raise WebhookConfigurationError(
                "Webhook URL requires a host and cannot contain credentials or a fragment."
            )
        try:
            port = parsed.port or 443
        except ValueError as exc:
            raise WebhookConfigurationError("Webhook URL has an invalid port.") from exc
        if not 1 <= port <= 65_535:
            raise WebhookConfigurationError("Webhook URL port is invalid.")
        if not _DESTINATION_ID.fullmatch(str(self.destination_id)):
            raise WebhookConfigurationError(
                "destination_id must contain 1-64 letters, numbers, dots, dashes, or underscores."
            )
        secret_bytes = str(self.secret).encode("utf-8")
        if not 16 <= len(secret_bytes) <= 4096:
            raise WebhookConfigurationError(
                "Webhook secret must be between 16 and 4096 UTF-8 bytes."
            )
        if not 0 < float(self.timeout_seconds) <= 30:
            raise WebhookConfigurationError(
                "Webhook timeout must be greater than zero and no more than 30 seconds."
            )
        if not 1 <= int(self.max_attempts) <= 5:
            raise WebhookConfigurationError("Webhook max_attempts must be from 1 through 5.")
        if not 0 <= float(self.backoff_seconds) <= 10:
            raise WebhookConfigurationError(
                "Webhook backoff must be from zero through 10 seconds."
            )
        if not 0 <= float(self.max_retry_after_seconds) <= 60:
            raise WebhookConfigurationError(
                "Webhook Retry-After cap must be from zero through 60 seconds."
            )
        if not 1024 <= int(self.max_body_bytes) <= 65_536:
            raise WebhookConfigurationError(
                "Webhook body limit must be from 1024 through 65536 bytes."
            )
        path = Path(self.audit_path)
        if path.exists() and not path.is_file():
            raise WebhookConfigurationError("Webhook audit path must be a file.")

    @classmethod
    def from_secret_environment(
        cls,
        *,
        url: str,
        secret_env: str = "EVALT_WEBHOOK_SECRET",
        **kwargs: Any,
    ) -> "WebhookDestination":
        if not _SECRET_ENV.fullmatch(str(secret_env)):
            raise WebhookConfigurationError(
                "Webhook secret environment name must be uppercase letters, numbers, and underscores."
            )
        secret = os.environ.get(secret_env)
        if not secret:
            raise WebhookConfigurationError(
                f"Webhook secret environment variable {secret_env} is not set."
            )
        return cls(url=url, secret=secret, **kwargs)


@dataclass(frozen=True)
class WebhookEvent:
    schema: str
    event_id: str
    idempotency_key: str
    type: str
    occurred_at: str
    source: str
    data: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "event_id": self.event_id,
            "idempotency_key": self.idempotency_key,
            "type": self.type,
            "occurred_at": self.occurred_at,
            "source": self.source,
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WebhookEvent":
        event = cls(
            schema=str(value.get("schema") or ""),
            event_id=str(value.get("event_id") or ""),
            idempotency_key=str(value.get("idempotency_key") or ""),
            type=str(value.get("type") or ""),
            occurred_at=str(value.get("occurred_at") or ""),
            source=str(value.get("source") or ""),
            data=dict(value.get("data") or {}),
        )
        if (
            event.schema != "evalt-webhook-event-v1"
            or not _EVENT_ID.fullmatch(event.event_id)
            or event.idempotency_key != event.event_id
            or not event.type.startswith(("route.health.", "ci.gate."))
            or not event.occurred_at
            or not event.source
        ):
            raise WebhookError("Webhook audit contains an invalid event.")
        return event


@dataclass(frozen=True)
class WebhookAttempt:
    attempt: int
    started_at: str
    finished_at: str
    status_code: int | None
    result: str
    retry_after_seconds: float | None = None


@dataclass(frozen=True)
class WebhookDelivery:
    event_id: str
    event_type: str
    destination_id: str
    delivered: bool
    attempts: tuple[WebhookAttempt, ...]
    audit_path: str
    replay: bool = False

    def summary(self) -> dict[str, Any]:
        last = self.attempts[-1] if self.attempts else None
        return {
            "schema": "evalt-webhook-delivery-summary-v1",
            "event_id": self.event_id,
            "event_type": self.event_type,
            "destination_id": self.destination_id,
            "delivered": self.delivered,
            "attempts": len(self.attempts),
            "status_code": last.status_code if last else None,
            "result": last.result if last else "not_attempted",
            "audit_path": self.audit_path,
            "replay": self.replay,
        }


class WebhookTransport(Protocol):
    def post(
        self,
        *,
        ip_address: str,
        hostname: str,
        port: int,
        target: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> tuple[int, Mapping[str, str]]:
        """Send one request without following redirects."""


class VerifiedHttpsTransport:
    """TLS transport pinned to an address that passed the public-IP check."""

    def post(
        self,
        *,
        ip_address: str,
        hostname: str,
        port: int,
        target: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> tuple[int, Mapping[str, str]]:
        raw = socket.create_connection((ip_address, port), timeout=timeout_seconds)
        context = ssl.create_default_context(cafile=certifi.where())
        wrapped = context.wrap_socket(raw, server_hostname=hostname)
        connection = http.client.HTTPConnection(hostname, port, timeout=timeout_seconds)
        connection.sock = wrapped
        try:
            connection.request("POST", target, body=body, headers=dict(headers))
            response = connection.getresponse()
            response.read(_MAX_RESPONSE_BYTES + 1)
            return int(response.status), {
                str(key).casefold(): str(value)
                for key, value in response.getheaders()
            }
        finally:
            connection.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public_addresses(
    hostname: str,
    port: int,
    *,
    allow_private_network: bool,
    resolver: Callable[..., Iterable[Any]] = socket.getaddrinfo,
) -> tuple[str, ...]:
    try:
        rows = resolver(hostname, port, type=socket.SOCK_STREAM)
    except (OSError, socket.gaierror) as exc:
        raise WebhookDeliveryError("Webhook host could not be resolved.") from exc
    addresses: list[str] = []
    for row in rows:
        try:
            address = str(row[4][0])
            parsed = ipaddress.ip_address(address)
        except (IndexError, TypeError, ValueError):
            continue
        if not allow_private_network and not parsed.is_global:
            raise WebhookDeliveryError(
                "Webhook host resolved to a private, local, reserved, or non-global address."
            )
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise WebhookDeliveryError("Webhook host did not resolve to a usable address.")
    return tuple(addresses)


def _event_bytes(event: WebhookEvent, maximum: int) -> bytes:
    body = json.dumps(
        event.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if len(body) > maximum:
        raise WebhookDeliveryError(
            f"Webhook event is {len(body)} bytes; destination limit is {maximum}."
        )
    return body


def _route_reference(route: str | None) -> str:
    digest = hashlib.sha256(
        ("evalt-webhook-route-v1:" + str(route or "default")).encode("utf-8")
    ).hexdigest()[:24]
    return f"route_{digest}"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _bounded_number(value: Any, *, digits: int = 6) -> float | int | None:
    number = _finite(value)
    if number is None:
        return None
    if float(number).is_integer():
        return int(number)
    return round(number, digits)


def _event_id(kind: str, occurred_at: str, data: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"kind": kind, "occurred_at": occurred_at, "data": data},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "evt_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _recent_audit_records(path: Path) -> Iterable[Mapping[str, Any]]:
    """Read a bounded tail of an append-only audit.

    Webhook delivery must not turn a long-lived JSONL audit into an unbounded
    memory read. The newest 16 MiB / 5,000 records is enough for state
    transitions and explicit replay while keeping the runtime ceiling stable.
    """

    if not path.is_file():
        return ()
    size = path.stat().st_size
    start = max(0, size - _MAX_AUDIT_READ_BYTES)
    with path.open("rb") as handle:
        handle.seek(start)
        raw = handle.read(_MAX_AUDIT_READ_BYTES)
    lines = raw.decode("utf-8", errors="replace").splitlines()
    if start and lines:
        # The first segment can begin in the middle of a JSONL record.
        lines = lines[1:]
    records: list[Mapping[str, Any]] = []
    for line in lines[-_MAX_AUDIT_RECORDS:]:
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, Mapping):
            records.append(value)
    return tuple(records)


def _last_route_status(
    path: Path,
    *,
    destination_id: str,
    route_ref: str,
) -> str | None:
    latest: str | None = None
    for record in _recent_audit_records(path):
        if (
            record.get("schema") == "evalt-webhook-delivery-v1"
            and record.get("destination_id") == destination_id
            and record.get("route_ref") == route_ref
        ):
            status = (record.get("event") or {}).get("data", {}).get("status")
            if status in {"HEALTHY", "REGRESSION", "ERROR"}:
                latest = status
    return latest


def route_health_event(
    result: Any,
    destination: WebhookDestination,
) -> WebhookEvent:
    """Build one aggregate route-health event, including transition semantics."""

    route_ref = _route_reference(getattr(result, "route", None))
    previous = _last_route_status(
        Path(destination.audit_path),
        destination_id=destination.destination_id,
        route_ref=route_ref,
    )
    status = str(getattr(result, "status", "ERROR"))
    if status == "REGRESSION" and previous != "REGRESSION":
        kind = "route.health.degraded"
    elif previous == "REGRESSION" and status == "HEALTHY":
        kind = "route.health.recovered"
    else:
        kind = "route.health.checked"
    gate = dict(getattr(result, "regression_gate", {}) or {})
    winner = getattr(getattr(result, "candidate", None), "winner", None)
    suite = dict(getattr(getattr(result, "candidate", None), "regression_suite", {}) or {})
    data: dict[str, Any] = {
        "route_ref": route_ref,
        "status": status,
        "previous_status": previous,
        "suite_hash": suite.get("suite_hash"),
        "holdout_pass_rate": _bounded_number(
            getattr(winner, "holdout_pass_rate", None)
        ),
        "quality_delta_percentage_points": _bounded_number(
            gate.get("quality_delta_percentage_points")
        ),
        "regressions": _bounded_number(gate.get("regressions"), digits=0),
        "missing_cases": _bounded_number(gate.get("missing_cases"), digits=0),
        "cost_increase_percent": _bounded_number(gate.get("cost_increase_percent")),
        "p90_increase_ms": _bounded_number(gate.get("p90_increase_ms")),
        "target_latency_p90_ms": _bounded_number(
            getattr(winner, "target_latency_p90_ms", None)
        ),
        "provider_spend_usd": _bounded_number(
            getattr(result, "provider_spend_usd", None), digits=10
        ),
        "final_test_scenarios": _bounded_number(
            getattr(winner, "holdout_unique_scenarios", None), digits=0
        ),
        "final_test_executions": _bounded_number(
            getattr(winner, "holdout_executions", None), digits=0
        ),
    }
    if destination.include_route_name:
        data["route"] = str(getattr(result, "route", None) or "default")[:120]
    data = {key: value for key, value in data.items() if value is not None}
    occurred_at = str(getattr(result, "finished_at", None) or _utc_now())
    event_id = _event_id(kind, occurred_at, data)
    return WebhookEvent(
        schema="evalt-webhook-event-v1",
        event_id=event_id,
        idempotency_key=event_id,
        type=kind,
        occurred_at=occurred_at,
        source="evalt.monitor",
        data=data,
    )


def ci_gate_event(
    *,
    status: str,
    result: Mapping[str, Any],
    gate: Mapping[str, Any],
    destination: WebhookDestination,
    occurred_at: str | None = None,
) -> WebhookEvent:
    """Build one aggregate CI gate event without result content."""

    normalized = str(status).upper()
    if normalized not in {"PASS", "FAIL", "ERROR"}:
        raise WebhookConfigurationError("CI webhook status must be PASS, FAIL, or ERROR.")
    winner = result.get("winner") or result.get("selected") or {}
    if not isinstance(winner, Mapping):
        winner = {}
    baseline = gate.get("baseline_gate")
    if not isinstance(baseline, Mapping):
        baseline = {}
    suite = result.get("regression_suite")
    if not isinstance(suite, Mapping):
        suite = {}
    route = str(result.get("route") or result.get("route_name") or "default")
    data: dict[str, Any] = {
        "route_ref": _route_reference(route),
        "status": normalized,
        "suite_hash": suite.get("suite_hash"),
        "holdout_pass_rate": _bounded_number(
            winner.get("holdout_pass_rate", winner.get("pass_rate"))
        ),
        "quality_delta_percentage_points": _bounded_number(
            baseline.get("quality_delta_percentage_points")
        ),
        "regressions": _bounded_number(baseline.get("regressions"), digits=0),
        "missing_cases": _bounded_number(baseline.get("missing_cases"), digits=0),
        "cost_increase_percent": _bounded_number(
            baseline.get("cost_increase_percent")
        ),
        "p90_increase_ms": _bounded_number(baseline.get("p90_increase_ms")),
    }
    if destination.include_route_name:
        data["route"] = route[:120]
    data = {key: value for key, value in data.items() if value is not None}
    timestamp = occurred_at or _utc_now()
    kind = f"ci.gate.{normalized.casefold()}"
    event_id = _event_id(kind, timestamp, data)
    return WebhookEvent(
        schema="evalt-webhook-event-v1",
        event_id=event_id,
        idempotency_key=event_id,
        type=kind,
        occurred_at=timestamp,
        source="evalt.ci",
        data=data,
    )


def _retry_after(headers: Mapping[str, str], maximum: float) -> float | None:
    raw = str(headers.get("retry-after") or "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None
    if seconds < 0:
        return None
    return min(seconds, maximum)


@contextmanager
def _audit_lock(path: Path, timeout_seconds: float = 5.0):
    lock = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise WebhookError("Webhook audit is locked by another writer.")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _append_audit(
    path: Path,
    *,
    destination_id: str,
    event: WebhookEvent,
    delivery: WebhookDelivery,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    route_ref = str(event.data.get("route_ref") or "")
    record = {
        "schema": "evalt-webhook-delivery-v1",
        "recorded_at": _utc_now(),
        "event_id": event.event_id,
        "event_type": event.type,
        "destination_id": destination_id,
        "route_ref": route_ref,
        "delivered": delivery.delivered,
        "replay": delivery.replay,
        "attempts": [asdict(attempt) for attempt in delivery.attempts],
        "event": event.to_dict(),
        "privacy": "aggregate decision only; runtime capabilities and customer content omitted",
    }
    serialized = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    with _audit_lock(path):
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def deliver_webhook(
    destination: WebhookDestination,
    event: WebhookEvent,
    *,
    transport: WebhookTransport | None = None,
    resolver: Callable[..., Iterable[Any]] = socket.getaddrinfo,
    sleep: Callable[[float], None] = time.sleep,
    replay: bool = False,
) -> WebhookDelivery:
    """Deliver and audit one event. Redirects are never followed."""

    destination.validate()
    body = _event_bytes(event, int(destination.max_body_bytes))
    parsed = urlsplit(destination.url)
    hostname = str(parsed.hostname)
    port = parsed.port or 443
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    try:
        target.encode("ascii")
    except UnicodeEncodeError as exc:
        raise WebhookConfigurationError(
            "Webhook path and query must be percent-encoded ASCII."
        ) from exc
    timestamp = str(int(time.time()))
    signature = hmac.new(
        destination.secret.encode("utf-8"),
        timestamp.encode("ascii") + b"." + body,
        hashlib.sha256,
    ).hexdigest()
    host_header = hostname if port == 443 else f"{hostname}:{port}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "evalt-webhook/1",
        "Host": host_header,
        "Content-Length": str(len(body)),
        "Idempotency-Key": event.idempotency_key,
        "X-Evalt-Event": event.type,
        "X-Evalt-Timestamp": timestamp,
        "X-Evalt-Signature": f"sha256={signature}",
    }
    sender = transport or VerifiedHttpsTransport()
    attempts: list[WebhookAttempt] = []
    for attempt_index in range(1, int(destination.max_attempts) + 1):
        started = _utc_now()
        status_code: int | None = None
        retry_after: float | None = None
        try:
            addresses = _public_addresses(
                hostname,
                port,
                allow_private_network=destination.allow_private_network,
                resolver=resolver,
            )
            status_code, response_headers = sender.post(
                ip_address=addresses[(attempt_index - 1) % len(addresses)],
                hostname=hostname,
                port=port,
                target=target,
                headers=headers,
                body=body,
                timeout_seconds=float(destination.timeout_seconds),
            )
            if 200 <= status_code < 300:
                result = "delivered"
            elif 300 <= status_code < 400:
                result = "redirect_rejected"
            else:
                result = "http_error"
            retry_after = _retry_after(
                response_headers, float(destination.max_retry_after_seconds)
            )
        except WebhookDeliveryError:
            result = "unsafe_or_unresolved_destination"
        except (OSError, TimeoutError, ssl.SSLError, http.client.HTTPException):
            result = "transport_error"
        attempts.append(
            WebhookAttempt(
                attempt=attempt_index,
                started_at=started,
                finished_at=_utc_now(),
                status_code=status_code,
                result=result,
                retry_after_seconds=retry_after,
            )
        )
        if result == "delivered":
            break
        retryable = (
            result in {"transport_error", "unsafe_or_unresolved_destination"}
            or status_code in _RETRYABLE_STATUSES
        )
        if not retryable or attempt_index >= int(destination.max_attempts):
            break
        delay = retry_after
        if delay is None:
            delay = min(
                float(destination.backoff_seconds) * (2 ** (attempt_index - 1)),
                10.0,
            )
        sleep(delay)
    delivery = WebhookDelivery(
        event_id=event.event_id,
        event_type=event.type,
        destination_id=destination.destination_id,
        delivered=bool(attempts and attempts[-1].result == "delivered"),
        attempts=tuple(attempts),
        audit_path=str(Path(destination.audit_path)),
        replay=replay,
    )
    _append_audit(
        Path(destination.audit_path),
        destination_id=destination.destination_id,
        event=event,
        delivery=delivery,
    )
    return delivery


def replay_webhook(
    destination: WebhookDestination,
    *,
    event_id: str,
    transport: WebhookTransport | None = None,
    resolver: Callable[..., Iterable[Any]] = socket.getaddrinfo,
    sleep: Callable[[float], None] = time.sleep,
) -> WebhookDelivery:
    """Replay the exact aggregate event and preserve its idempotency identity."""

    if not _EVENT_ID.fullmatch(str(event_id)):
        raise WebhookError("Replay requires an Evalt event ID such as evt_<32 hex>.")
    path = Path(destination.audit_path)
    if not path.is_file():
        raise WebhookError("Webhook audit does not exist.")
    event: WebhookEvent | None = None
    for record in _recent_audit_records(path):
        if (
            record.get("schema") == "evalt-webhook-delivery-v1"
            and record.get("event_id") == event_id
            and record.get("destination_id") == destination.destination_id
        ):
            event = WebhookEvent.from_dict(record.get("event") or {})
    if event is None:
        raise WebhookError(
            "No matching recent event exists for this destination and audit."
        )
    return deliver_webhook(
        destination,
        event,
        transport=transport,
        resolver=resolver,
        sleep=sleep,
        replay=True,
    )
