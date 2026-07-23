"""Private, content-addressed storage for reusable Evalt evidence.

The library is deliberately local.  It never talks to the hosted workspace and its
index contains no source paths or customer-content excerpts.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence
import uuid


_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$")
_TAG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,30}[a-z0-9])?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INDEX_SCHEMA = "evalt-local-evidence-library-v1"
_KINDS = {"suite", "result"}
_MAX_OBJECT_BYTES = 128 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _private_permissions(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decode_json_object(data: bytes, *, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON number {value}.")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}.")
            value[key] = item
        return value

    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON.") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object.")
    return value


def _validated_name(value: str) -> str:
    name = str(value or "").strip().casefold()
    if not _NAME_RE.fullmatch(name):
        raise ValueError(
            "Library names must be 1-64 lowercase letters, numbers, dots, "
            "underscores, or hyphens, with an alphanumeric first and last character."
        )
    return name


def _validated_tags(values: Sequence[str] | None) -> tuple[str, ...]:
    tags: set[str] = set()
    for raw in values or ():
        tag = str(raw or "").strip().casefold()
        if not _TAG_RE.fullmatch(tag):
            raise ValueError(
                "Library tags must be 1-32 lowercase letters, numbers, dots, "
                "underscores, or hyphens, with an alphanumeric first and last character."
            )
        tags.add(tag)
    if len(tags) > 12:
        raise ValueError("A library entry may have at most 12 distinct tags.")
    return tuple(sorted(tags))


def _validate_result(value: Mapping[str, Any]) -> None:
    schema = str(value.get("schema") or "")
    if schema not in {"", "evalt-result-v1", "evalt-monitor-result-v1"}:
        raise ValueError("Result files must use a supported Evalt result schema.")
    winner = value.get("winner")
    if not (
        isinstance(winner, Mapping)
        and isinstance(value.get("regression_suite"), Mapping)
        and "total_provider_spend_usd" in value
        and (
            schema == "evalt-result-v1"
            or isinstance(value.get("models"), list)
        )
    ):
        raise ValueError(
            "Result files must be exported Evalt optimization or monitor results."
        )
    selected = winner if isinstance(winner, Mapping) else {}
    if not str(selected.get("model") or "").strip():
        raise ValueError("Evalt result winner is missing its selected model.")
    try:
        quality = float(selected["holdout_pass_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Evalt result winner is missing a numeric holdout_pass_rate."
        ) from error
    if not 0 <= quality <= 1:
        raise ValueError("Evalt result holdout_pass_rate must be between zero and one.")


def _contains_image(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("type") == "image_url":
            return True
        return any(_contains_image(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_image(item) for item in value)
    return False


def _classify(value: Mapping[str, Any], requested: str = "auto") -> str:
    if requested not in {*_KINDS, "auto"}:
        raise ValueError("Library kind must be auto, suite, or result.")
    schema = str(value.get("schema") or "")
    inferred = "suite" if schema in {"evalt-suite-v1", "evalt-suite-v2"} else "result"
    if requested != "auto" and requested != inferred:
        raise ValueError(
            f"The JSON is an Evalt {inferred}, not the requested {requested} kind."
        )
    if inferred == "suite":
        # Import lazily so core.py can expose EvidenceLibrary without a cycle.
        from .core import Suite

        Suite.from_dict(value)
    else:
        _validate_result(value)
    return inferred


def _summary(kind: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if kind == "suite":
        examples = value.get("examples") or []
        models = value.get("models") or []
        has_images = any(
            isinstance(item, Mapping)
            and _contains_image(item.get("input"))
            for item in examples
        )
        return {
            "suite_name": str(value.get("name") or "evalt-suite"),
            "examples": len(examples) if isinstance(examples, list) else 0,
            "models": len(models) if isinstance(models, list) else 0,
            "quality_threshold": value.get("quality_threshold"),
            "has_images": has_images,
        }
    winner = value.get("winner")
    winner = winner if isinstance(winner, Mapping) else {}
    regression_suite = value.get("regression_suite")
    regression_suite = (
        regression_suite if isinstance(regression_suite, Mapping) else {}
    )
    return {
        "selected_model": str(winner.get("model") or ""),
        "holdout_pass_rate": winner.get("holdout_pass_rate"),
        "suite_hash": str(regression_suite.get("suite_hash") or "") or None,
        "monitor_status": value.get("monitor_status"),
    }


@dataclass(frozen=True)
class LibraryEntry:
    name: str
    kind: str
    tags: tuple[str, ...]
    sha256: str
    bytes: int
    added_at: str
    summary: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LibraryEntry":
        try:
            entry = cls(
                name=_validated_name(str(value["name"])),
                kind=str(value["kind"]),
                tags=_validated_tags(tuple(value.get("tags") or ())),
                sha256=str(value["sha256"]),
                bytes=int(value["bytes"]),
                added_at=str(value["added_at"]),
                summary=dict(value.get("summary") or {}),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid library entry: {error}") from error
        if entry.kind not in _KINDS:
            raise ValueError("Invalid library entry kind.")
        if not _SHA256_RE.fullmatch(entry.sha256):
            raise ValueError("Invalid library entry SHA-256.")
        if entry.bytes <= 0:
            raise ValueError("Invalid library entry byte count.")
        return entry

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tags"] = list(self.tags)
        value["summary"] = dict(self.summary)
        return value


class EvidenceLibrary:
    """An offline, immutable library of approved suites and result receipts."""

    def __init__(self, root: str | Path | None = None):
        configured = (
            Path(root)
            if root is not None
            else Path(os.environ.get("EVALT_LIBRARY_HOME", ".evalt/library"))
        )
        self.root = configured.expanduser().resolve()
        self.objects = self.root / "objects"
        self.index_path = self.root / "index.json"
        self.lock_path = self.root / ".lock"

    def _ensure_directories(self) -> None:
        self.objects.mkdir(parents=True, exist_ok=True)
        _private_permissions(self.root, 0o700)
        _private_permissions(self.objects, 0o700)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self._ensure_directories()
        with self.lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            _private_permissions(self.lock_path, 0o600)
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _index_integrity(entries: Sequence[Mapping[str, Any]]) -> str:
        return _digest(_canonical(list(entries)))

    def _load_index(self) -> list[LibraryEntry]:
        if not self.index_path.exists():
            return []
        try:
            raw = self.index_path.read_bytes()
            value = _decode_json_object(raw, label="The Evalt library index")
        except (OSError, ValueError) as error:
            raise ValueError("The Evalt library index is unreadable or malformed.") from error
        if value.get("schema") != _INDEX_SCHEMA:
            raise ValueError("The Evalt library index schema is invalid.")
        entries_value = value.get("entries")
        if not isinstance(entries_value, list):
            raise ValueError("The Evalt library index entries are invalid.")
        expected = self._index_integrity(entries_value)
        if value.get("integrity_sha256") != expected:
            raise ValueError(
                "The Evalt library index was modified outside Evalt; restore it "
                "from version control or a trusted backup."
            )
        entries = [LibraryEntry.from_dict(item) for item in entries_value]
        names = [entry.name for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("The Evalt library index contains duplicate names.")
        return entries

    def _write_index(self, entries: Sequence[LibraryEntry]) -> None:
        values = [
            entry.to_dict()
            for entry in sorted(entries, key=lambda item: (item.kind, item.name))
        ]
        payload = {
            "schema": _INDEX_SCHEMA,
            "privacy": "local-only; never synchronized",
            "entries": values,
            "integrity_sha256": self._index_integrity(values),
        }
        temporary = self.root / f".index-{uuid.uuid4().hex}.tmp"
        data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            _private_permissions(temporary, 0o600)
            os.replace(temporary, self.index_path)
            _private_permissions(self.index_path, 0o600)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _object_path(self, sha256: str) -> Path:
        if not _SHA256_RE.fullmatch(sha256):
            raise ValueError("Invalid evidence object SHA-256.")
        return self.objects / f"{sha256}.json"

    def _read_object(self, entry: LibraryEntry) -> bytes:
        path = self._object_path(entry.sha256)
        try:
            data = path.read_bytes()
        except FileNotFoundError as error:
            raise ValueError(
                f"Library object for {entry.name!r} is missing."
            ) from error
        if len(data) != entry.bytes or _digest(data) != entry.sha256:
            raise ValueError(
                f"Library object for {entry.name!r} failed its SHA-256 integrity check."
            )
        try:
            value = _decode_json_object(
                data, label=f"Library object for {entry.name!r}"
            )
        except ValueError as error:
            raise ValueError(
                f"Library object for {entry.name!r} is no longer a valid Evalt JSON object."
            ) from error
        _classify(value, entry.kind)
        return data

    def add(
        self,
        source: str | Path,
        *,
        name: str,
        tags: Sequence[str] | None = None,
        kind: str = "auto",
    ) -> LibraryEntry:
        source_path = Path(source).expanduser()
        try:
            size = source_path.stat().st_size
            if size > _MAX_OBJECT_BYTES:
                raise ValueError(
                    "Evidence files may be at most 128 MiB."
                )
            data = source_path.read_bytes()
        except OSError as error:
            raise ValueError(f"Cannot read evidence file {source_path}: {error}") from error
        if not data:
            raise ValueError("Evidence files cannot be empty.")
        if len(data) > _MAX_OBJECT_BYTES:
            raise ValueError("Evidence files may be at most 128 MiB.")
        value = _decode_json_object(data, label="Evidence files")
        resolved_kind = _classify(value, kind)
        resolved_name = _validated_name(name)
        resolved_tags = _validated_tags(tags)
        sha256 = _digest(data)
        entry = LibraryEntry(
            name=resolved_name,
            kind=resolved_kind,
            tags=resolved_tags,
            sha256=sha256,
            bytes=len(data),
            added_at=_utc_now(),
            summary=_summary(resolved_kind, value),
        )
        with self._lock():
            entries = self._load_index()
            existing = next(
                (item for item in entries if item.name == resolved_name), None
            )
            if existing:
                if (
                    existing.sha256 == entry.sha256
                    and existing.kind == entry.kind
                    and existing.tags == entry.tags
                ):
                    self._read_object(existing)
                    return existing
                raise ValueError(
                    f"Library name {resolved_name!r} already identifies different "
                    "content or tags; choose a new immutable name."
                )
            target = self._object_path(sha256)
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError("A content-addressed library object is corrupted.")
            else:
                try:
                    with target.open("xb") as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                except FileExistsError:
                    if target.read_bytes() != data:
                        raise ValueError(
                            "A concurrent content-addressed library object is corrupted."
                        )
                _private_permissions(target, 0o600)
            entries.append(entry)
            self._write_index(entries)
        return entry

    def list(
        self,
        *,
        kind: str | None = None,
        tag: str | None = None,
        query: str | None = None,
    ) -> list[LibraryEntry]:
        if kind is not None and kind not in _KINDS:
            raise ValueError("Library kind filter must be suite or result.")
        selected_tag = _validated_tags([tag])[0] if tag else None
        selected_query = str(query or "").strip().casefold()
        entries = self._load_index()
        return [
            entry
            for entry in entries
            if (kind is None or entry.kind == kind)
            and (selected_tag is None or selected_tag in entry.tags)
            and (
                not selected_query
                or selected_query in entry.name
                or any(selected_query in item for item in entry.tags)
            )
        ]

    def entry(self, name: str, *, expected_kind: str | None = None) -> LibraryEntry:
        resolved_name = _validated_name(name)
        entry = next(
            (item for item in self._load_index() if item.name == resolved_name), None
        )
        if entry is None:
            raise ValueError(f"Library has no evidence named {resolved_name!r}.")
        if expected_kind is not None and entry.kind != expected_kind:
            raise ValueError(
                f"Library evidence {resolved_name!r} is a {entry.kind}, "
                f"not a {expected_kind}."
            )
        self._read_object(entry)
        return entry

    def read(
        self, name: str, *, expected_kind: str | None = None
    ) -> dict[str, Any]:
        entry = self.entry(name, expected_kind=expected_kind)
        return json.loads(self._read_object(entry).decode("utf-8"))

    def resolve(
        self, name: str, *, expected_kind: str | None = None
    ) -> Path:
        entry = self.entry(name, expected_kind=expected_kind)
        return self._object_path(entry.sha256)

    def export(
        self,
        name: str,
        output: str | Path,
        *,
        force: bool = False,
    ) -> Path:
        entry = self.entry(name)
        data = self._read_object(entry)
        target = Path(output).expanduser()
        if target.exists() and not force:
            if target.is_file() and target.read_bytes() == data:
                return target
            raise FileExistsError(
                f"{target} already exists; pass force=True or --force to replace it."
            )
        if target.exists() and target.is_dir():
            raise IsADirectoryError(f"{target} is a directory, not an export file.")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / f".{target.name}-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return target


def resolve_evidence_reference(
    value: str | Path,
    *,
    root: str | Path | None = None,
    expected_kind: str | None = None,
) -> Path:
    """Resolve ``@private-name`` or return a normal caller-supplied path."""

    text = str(value)
    if not text.startswith("@"):
        return Path(value)
    name = text[1:]
    if not name:
        raise ValueError("Library references require a name after @.")
    return EvidenceLibrary(root).resolve(name, expected_kind=expected_kind)


__all__ = [
    "EvidenceLibrary",
    "LibraryEntry",
    "resolve_evidence_reference",
]
