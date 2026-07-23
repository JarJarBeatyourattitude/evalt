"""Local custom-scorer contracts used by Evalt's evaluation engine.

Downloaded suites identify a scorer, but can never select executable code. Users
register a scorer object explicitly in trusted local Python or through explicit
CLI arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_MAX_REASON_CHARS = 2_000
_MAX_ENVIRONMENT_ENTRIES = 128
_MAX_ENVIRONMENT_VALUE_CHARS = 16_384
_SAFE_INHERITED_ENVIRONMENT = {
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "WINDIR",
}


class CustomScorerError(RuntimeError):
    """A registered custom scorer could not produce one valid score."""


@dataclass(frozen=True)
class ScoreRequest:
    """One local evaluation request.

    Input content may contain private text or embedded image bytes. Evalt passes
    it only to the scorer explicitly registered by the caller and never adds it
    to dashboard synchronization.
    """

    scenario_id: str
    turn: int
    input: Any
    transcript: tuple[Mapping[str, Any], ...]
    approved_output: str
    actual_output: str
    group: str | None = None
    difficulty: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "evalt-custom-score-request-v1",
            "scenario_id": self.scenario_id,
            "turn": self.turn,
            "input": self.input,
            "transcript": [dict(message) for message in self.transcript],
            "approved_output": self.approved_output,
            "actual_output": self.actual_output,
            "group": self.group,
            "difficulty": self.difficulty,
        }


@dataclass(frozen=True)
class ScoreResult:
    """One strict custom-scorer result."""

    passed: bool
    score: float
    reason: str = ""

    @classmethod
    def from_value(cls, value: "ScoreResult | Mapping[str, Any]") -> "ScoreResult":
        if isinstance(value, cls):
            passed = value.passed
            score = value.score
            reason = value.reason
        elif isinstance(value, Mapping):
            unexpected = set(value) - {"passed", "score", "reason"}
            missing = {"passed", "score"} - set(value)
            if unexpected or missing:
                details = []
                if missing:
                    details.append("missing " + ", ".join(sorted(missing)))
                if unexpected:
                    details.append("unexpected " + ", ".join(sorted(unexpected)))
                raise CustomScorerError(
                    "Custom scorer result must contain only passed, score, and "
                    "optional reason (" + "; ".join(details) + ")."
                )
            passed = value["passed"]
            score = value["score"]
            reason = value.get("reason", "")
        else:
            raise CustomScorerError(
                "Custom scorer must return ScoreResult or a mapping."
            )
        if type(passed) is not bool:
            raise CustomScorerError("Custom scorer result passed must be a boolean.")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise CustomScorerError("Custom scorer result score must be a number.")
        normalized_score = float(score)
        if not math.isfinite(normalized_score) or not 0 <= normalized_score <= 1:
            raise CustomScorerError(
                "Custom scorer result score must be finite and between zero and one."
            )
        if not isinstance(reason, str):
            raise CustomScorerError("Custom scorer result reason must be a string.")
        normalized_reason = reason
        if len(normalized_reason) > _MAX_REASON_CHARS:
            raise CustomScorerError(
                f"Custom scorer result reason cannot exceed {_MAX_REASON_CHARS} characters."
            )
        return cls(passed=passed, score=normalized_score, reason=normalized_reason)


@runtime_checkable
class Scorer(Protocol):
    """Protocol for an explicitly registered local scorer."""

    scorer_id: str
    scorer_version: str

    def score(self, request: ScoreRequest) -> ScoreResult | Mapping[str, Any]:
        """Score one actual output against its approved example."""


def validate_scorer_identity(scorer_id: str, scorer_version: str) -> tuple[str, str]:
    if not isinstance(scorer_id, str) or not isinstance(scorer_version, str):
        raise ValueError("scorer_id and scorer_version must be strings.")
    normalized_id = scorer_id.strip()
    normalized_version = scorer_version.strip()
    if not _IDENTIFIER_PATTERN.fullmatch(normalized_id):
        raise ValueError(
            "scorer_id must start with a letter or number and contain at most 80 "
            "letters, numbers, dots, underscores, or hyphens."
        )
    if not _IDENTIFIER_PATTERN.fullmatch(normalized_version):
        raise ValueError(
            "scorer_version must start with a letter or number and contain at most "
            "80 letters, numbers, dots, underscores, or hyphens."
        )
    return normalized_id, normalized_version


def normalize_scorer_registry(
    scorers: Mapping[str, Scorer] | None,
) -> dict[str, Scorer]:
    """Validate explicit registrations without importing or executing scorer code."""

    registry: dict[str, Scorer] = {}
    for raw_key, scorer in dict(scorers or {}).items():
        key = str(raw_key).strip()
        if not isinstance(scorer, Scorer):
            raise ValueError(
                f"Custom scorer {key or raw_key!r} must expose scorer_id, "
                "scorer_version, and score(request)."
            )
        scorer_id, scorer_version = validate_scorer_identity(
            scorer.scorer_id, scorer.scorer_version
        )
        if key != scorer_id:
            raise ValueError(
                f"Custom scorer registry key {key!r} does not match scorer_id "
                f"{scorer_id!r}."
            )
        if scorer_id in registry:
            raise ValueError(f"Custom scorer {scorer_id!r} was registered twice.")
        registry[scorer_id] = scorer
    return registry


def resolve_registered_scorer(
    evaluator: Mapping[str, Any], registry: Mapping[str, Scorer]
) -> Scorer:
    scorer_id = str(evaluator.get("scorer_id") or "")
    scorer_version = str(evaluator.get("scorer_version") or "")
    scorer = registry.get(scorer_id)
    if scorer is None:
        raise ValueError(
            f"Custom scorer {scorer_id!r} is not registered locally. Downloaded "
            "suite files cannot select executable code; pass it explicitly with "
            "Evalt(custom_scorers={...}) or the custom-scorer CLI options."
        )
    registered_id, registered_version = validate_scorer_identity(
        scorer.scorer_id, scorer.scorer_version
    )
    if registered_id != scorer_id or registered_version != scorer_version:
        raise ValueError(
            f"Custom scorer {scorer_id!r} version mismatch: suite requires "
            f"{scorer_version!r}, locally registered scorer is "
            f"{registered_version!r}."
        )
    return scorer


class CommandScorer:
    """Run one explicitly configured local command without a shell.

    Requests are JSON on stdin and one strict JSON object is required on stdout.
    Output is streamed and discarded beyond the configured bound, so an abusive
    process cannot make Evalt retain unlimited stdout or stderr in memory.
    """

    def __init__(
        self,
        scorer_id: str,
        scorer_version: str,
        argv: Sequence[str],
        *,
        timeout_seconds: float = 10.0,
        max_input_bytes: int = 8 * 1024 * 1024,
        max_output_bytes: int = 64 * 1024,
        cwd: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
        inherit_environment: bool = False,
    ) -> None:
        self.scorer_id, self.scorer_version = validate_scorer_identity(
            scorer_id, scorer_version
        )
        self.argv = tuple(str(value) for value in argv)
        if not self.argv or any(not value for value in self.argv):
            raise ValueError("CommandScorer argv must contain non-empty arguments.")
        if not math.isfinite(float(timeout_seconds)) or not 0 < float(timeout_seconds) <= 300:
            raise ValueError(
                "CommandScorer timeout_seconds must be greater than zero and at most 300."
            )
        if not 1 <= int(max_input_bytes) <= 64 * 1024 * 1024:
            raise ValueError(
                "CommandScorer max_input_bytes must be between 1 byte and 64 MiB."
            )
        if not 1 <= int(max_output_bytes) <= 1024 * 1024:
            raise ValueError(
                "CommandScorer max_output_bytes must be between 1 byte and 1 MiB."
            )
        self.timeout_seconds = float(timeout_seconds)
        self.max_input_bytes = int(max_input_bytes)
        self.max_output_bytes = int(max_output_bytes)
        self.cwd = str(Path(cwd).resolve()) if cwd is not None else None
        self.inherit_environment = bool(inherit_environment)
        explicit_environment = dict(environment or {})
        if len(explicit_environment) > _MAX_ENVIRONMENT_ENTRIES:
            raise ValueError(
                f"CommandScorer environment cannot exceed {_MAX_ENVIRONMENT_ENTRIES} entries."
            )
        self.environment: dict[str, str] = {}
        for raw_name, raw_value in explicit_environment.items():
            name = str(raw_name)
            value = str(raw_value)
            if not name or "=" in name or "\x00" in name or "\x00" in value:
                raise ValueError("CommandScorer environment contains an invalid entry.")
            if len(value) > _MAX_ENVIRONMENT_VALUE_CHARS:
                raise ValueError(
                    "CommandScorer environment values cannot exceed "
                    f"{_MAX_ENVIRONMENT_VALUE_CHARS} characters."
                )
            self.environment[name] = value

    def _subprocess_environment(self) -> dict[str, str]:
        if self.inherit_environment:
            result = dict(os.environ)
        else:
            result = {
                name: value
                for name, value in os.environ.items()
                if name.upper() in _SAFE_INHERITED_ENVIRONMENT
            }
        result.update(self.environment)
        return result

    def score(self, request: ScoreRequest) -> ScoreResult:
        payload = json.dumps(
            request.to_dict(), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        if len(payload) > self.max_input_bytes:
            raise CustomScorerError(
                f"Custom scorer request exceeded {self.max_input_bytes} bytes."
            )
        try:
            process = subprocess.Popen(
                self.argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                env=self._subprocess_environment(),
                shell=False,
            )
        except OSError as error:
            raise CustomScorerError(
                f"Custom scorer {self.scorer_id!r} could not start: {error}"
            ) from error

        stdout = bytearray()
        stderr = bytearray()
        stdout_total = [0]
        stderr_total = [0]
        stream_errors: list[BaseException] = []

        def write_stdin() -> None:
            try:
                assert process.stdin is not None
                process.stdin.write(payload)
                process.stdin.close()
            except (BrokenPipeError, OSError) as error:
                stream_errors.append(error)

        def read_stream(
            stream: Any, retained: bytearray, total: list[int]
        ) -> None:
            try:
                while True:
                    chunk = stream.read(65_536)
                    if not chunk:
                        return
                    total[0] += len(chunk)
                    remaining = self.max_output_bytes + 1 - len(retained)
                    if remaining > 0:
                        retained.extend(chunk[:remaining])
            except OSError as error:
                stream_errors.append(error)
            finally:
                try:
                    stream.close()
                except OSError:
                    pass

        threads = [
            threading.Thread(target=write_stdin, daemon=True),
            threading.Thread(
                target=read_stream,
                args=(process.stdout, stdout, stdout_total),
                daemon=True,
            ),
            threading.Thread(
                target=read_stream,
                args=(process.stderr, stderr, stderr_total),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        try:
            return_code = process.wait(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait()
            for thread in threads:
                thread.join(timeout=1)
            raise CustomScorerError(
                f"Custom scorer {self.scorer_id!r} timed out after "
                f"{self.timeout_seconds:g} seconds."
            ) from error
        for thread in threads:
            thread.join(timeout=1)
        if any(thread.is_alive() for thread in threads):
            raise CustomScorerError(
                f"Custom scorer {self.scorer_id!r} did not close its streams."
            )
        if stream_errors and not (
            return_code != 0 and all(isinstance(error, BrokenPipeError) for error in stream_errors)
        ):
            raise CustomScorerError(
                f"Custom scorer {self.scorer_id!r} stream communication failed."
            )
        if stdout_total[0] > self.max_output_bytes:
            raise CustomScorerError(
                f"Custom scorer stdout exceeded {self.max_output_bytes} bytes."
            )
        if stderr_total[0] > self.max_output_bytes:
            raise CustomScorerError(
                f"Custom scorer stderr exceeded {self.max_output_bytes} bytes."
            )
        if return_code != 0:
            raise CustomScorerError(
                f"Custom scorer {self.scorer_id!r} exited with code "
                f"{return_code}; stderr was omitted."
            )
        try:
            decoded = bytes(stdout).decode("utf-8")
        except UnicodeDecodeError as error:
            raise CustomScorerError(
                "Custom scorer stdout must be valid UTF-8 JSON."
            ) from error
        try:
            value = json.loads(decoded)
        except json.JSONDecodeError as error:
            raise CustomScorerError(
                "Custom scorer stdout must contain exactly one valid JSON object."
            ) from error
        if not isinstance(value, Mapping):
            raise CustomScorerError(
                "Custom scorer stdout must contain exactly one JSON object."
            )
        return ScoreResult.from_value(value)
