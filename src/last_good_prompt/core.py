"""A bounded prompt/model tournament with exact provider-cost accounting."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
import contextvars
from dataclasses import asdict, dataclass, field
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import ssl
import statistics
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zlib

import certifi
from dotenv import load_dotenv

from .scorers import (
    CustomScorerError,
    ScoreRequest,
    ScoreResult,
    Scorer,
    normalize_scorer_registry,
    resolve_registered_scorer,
    validate_scorer_identity,
)


def _submit_with_context(
    pool: ThreadPoolExecutor, function: Callable[..., Any], *args: Any
):
    """Submit work with the caller's request limits and deadlines attached."""
    context = contextvars.copy_context()
    return pool.submit(context.run, function, *args)


OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models?output_modalities=text&sort=intelligence-high-to-low"
OPENROUTER_ZDR_ENDPOINTS_URL = "https://openrouter.ai/api/v1/endpoints/zdr"


# These fields define Evalt's own routing and accounting contract. Everything else
# in the Chat Completions body is accepted through ``request_options`` so new
# OpenRouter fields do not require an Evalt release merely to round-trip them.
_RESERVED_OPENROUTER_REQUEST_FIELDS = {
    "model", "models", "messages", "prompt", "stream", "usage",
    "max_tokens", "max_completion_tokens",
}
_SECRET_FIELD_NAMES = {
    "api_key", "apikey", "authorization", "openrouter_api_key",
}

_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_IMAGE_REQUEST_BYTES = 40 * 1024 * 1024
_MAX_DECODED_IMAGE_BYTES = 128 * 1024 * 1024
_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
_IMAGE_DETAILS = {"auto", "low", "high"}


def _validate_png(data: bytes) -> None:
    if len(data) < 33:
        raise ValueError("PNG data is truncated.")
    offset = 8
    saw_header = False
    saw_pixels = False
    compressed = bytearray()
    expected_decoded = 0
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("PNG data is truncated.")
        length = int.from_bytes(data[offset:offset + 4], "big")
        kind = data[offset + 4:offset + 8]
        end = offset + 12 + length
        if end > len(data):
            raise ValueError("PNG data is truncated.")
        payload = data[offset + 8:offset + 8 + length]
        expected_crc = int.from_bytes(data[offset + 8 + length:end], "big")
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise ValueError("PNG data has an invalid checksum.")
        if not saw_header:
            if kind != b"IHDR" or length != 13:
                raise ValueError("PNG data is missing its image header.")
            width = int.from_bytes(payload[:4], "big")
            height = int.from_bytes(payload[4:8], "big")
            if not width or not height:
                raise ValueError("PNG dimensions must be positive.")
            bit_depth, color_type, compression, filtering, interlace = payload[8:13]
            valid_depths = {
                0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8},
                4: {8, 16}, 6: {8, 16},
            }
            if bit_depth not in valid_depths.get(color_type, set()):
                raise ValueError("PNG color type and bit depth are invalid.")
            if compression != 0 or filtering != 0 or interlace not in {0, 1}:
                raise ValueError("PNG header methods are invalid.")
            channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
            bits_per_pixel = channels * bit_depth
            if interlace == 0:
                expected_decoded = height * (1 + ((width * bits_per_pixel + 7) // 8))
            else:
                passes = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8),
                          (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2),
                          (0, 1, 1, 2))
                for x_start, y_start, x_step, y_step in passes:
                    pass_width = max(0, (width - x_start + x_step - 1) // x_step)
                    pass_height = max(0, (height - y_start + y_step - 1) // y_step)
                    if pass_width and pass_height:
                        expected_decoded += pass_height * (
                            1 + ((pass_width * bits_per_pixel + 7) // 8)
                        )
            if expected_decoded > _MAX_DECODED_IMAGE_BYTES:
                raise ValueError("PNG expands beyond the 128 MiB decoded safety limit.")
            saw_header = True
        elif kind == b"IDAT":
            compressed.extend(payload)
            saw_pixels = True
        elif kind == b"IEND":
            if length or end != len(data) or not saw_pixels:
                raise ValueError("PNG data has an invalid image terminator.")
            try:
                decoder = zlib.decompressobj()
                decoded = decoder.decompress(bytes(compressed), expected_decoded + 1)
                if len(decoded) > expected_decoded or decoder.unconsumed_tail:
                    raise ValueError("PNG decoded pixel data exceeds its dimensions.")
                decoded += decoder.flush(expected_decoded + 1 - len(decoded))
                if len(decoded) != expected_decoded or not decoder.eof or decoder.unused_data:
                    raise ValueError("PNG pixel data cannot be decoded completely.")
            except zlib.error as error:
                raise ValueError("PNG pixel data cannot be decoded.") from error
            return
        offset = end
    raise ValueError("PNG data is missing its image terminator.")


def _validate_jpeg(data: bytes) -> None:
    if len(data) < 12 or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise ValueError("JPEG data is truncated.")
    offset = 2
    saw_dimensions = False
    while offset < len(data) - 2:
        if data[offset] != 0xFF:
            raise ValueError("JPEG data has an invalid marker.")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            raise ValueError("JPEG data is truncated.")
        marker = data[offset]
        offset += 1
        if marker == 0xDA:
            if not saw_dimensions or offset + 2 > len(data):
                raise ValueError("JPEG data is missing dimensions or scan data.")
            segment_length = int.from_bytes(data[offset:offset + 2], "big")
            if segment_length < 2 or offset + segment_length >= len(data) - 1:
                raise ValueError("JPEG scan data is truncated.")
            return
        if marker in {0x01, *range(0xD0, 0xD9)}:
            continue
        if offset + 2 > len(data):
            raise ValueError("JPEG data is truncated.")
        segment_length = int.from_bytes(data[offset:offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(data) - 2:
            raise ValueError("JPEG data is truncated.")
        if marker in {
            0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
        }:
            if segment_length < 8:
                raise ValueError("JPEG dimensions are truncated.")
            height = int.from_bytes(data[offset + 3:offset + 5], "big")
            width = int.from_bytes(data[offset + 5:offset + 7], "big")
            if not width or not height:
                raise ValueError("JPEG dimensions must be positive.")
            saw_dimensions = True
        offset += segment_length
    raise ValueError("JPEG data is missing scan data.")


def _skip_gif_sub_blocks(data: bytes, offset: int) -> int:
    while True:
        if offset >= len(data):
            raise ValueError("GIF data is truncated.")
        length = data[offset]
        offset += 1
        if not length:
            return offset
        if offset + length > len(data):
            raise ValueError("GIF data is truncated.")
        offset += length


def _validate_gif(data: bytes) -> None:
    if len(data) < 14:
        raise ValueError("GIF data is truncated.")
    width = int.from_bytes(data[6:8], "little")
    height = int.from_bytes(data[8:10], "little")
    if not width or not height:
        raise ValueError("GIF dimensions must be positive.")
    packed = data[10]
    offset = 13 + (3 * (2 ** ((packed & 0x07) + 1)) if packed & 0x80 else 0)
    saw_image = False
    while offset < len(data):
        marker = data[offset]
        offset += 1
        if marker == 0x3B:
            if not saw_image or offset != len(data):
                raise ValueError("GIF data has an invalid image terminator.")
            return
        if marker == 0x21:
            if offset >= len(data):
                raise ValueError("GIF extension data is truncated.")
            offset = _skip_gif_sub_blocks(data, offset + 1)
            continue
        if marker != 0x2C or offset + 9 > len(data):
            raise ValueError("GIF data has an invalid image block.")
        descriptor = data[offset:offset + 9]
        local_packed = descriptor[8]
        offset += 9
        if local_packed & 0x80:
            offset += 3 * (2 ** ((local_packed & 0x07) + 1))
        if offset >= len(data):
            raise ValueError("GIF image data is truncated.")
        offset = _skip_gif_sub_blocks(data, offset + 1)
        saw_image = True
    raise ValueError("GIF data is missing its image terminator.")


def _validate_webp(data: bytes) -> None:
    if len(data) < 20 or int.from_bytes(data[4:8], "little") + 8 != len(data):
        raise ValueError("WebP data is truncated.")
    offset = 12
    saw_image = False
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("WebP chunk data is truncated.")
        kind = data[offset:offset + 4]
        length = int.from_bytes(data[offset + 4:offset + 8], "little")
        start = offset + 8
        end = start + length
        if end > len(data):
            raise ValueError("WebP chunk data is truncated.")
        payload = data[start:end]
        if kind == b"VP8X":
            if len(payload) < 10:
                raise ValueError("WebP dimensions are truncated.")
            width = 1 + int.from_bytes(payload[4:7], "little")
            height = 1 + int.from_bytes(payload[7:10], "little")
            saw_image = bool(width and height)
        elif kind == b"VP8L":
            if len(payload) < 5 or payload[0] != 0x2F:
                raise ValueError("WebP lossless image data is invalid.")
            bits = int.from_bytes(payload[1:5], "little")
            saw_image = bool((bits & 0x3FFF) + 1 and ((bits >> 14) & 0x3FFF) + 1)
        elif kind == b"VP8 ":
            if len(payload) < 10 or payload[3:6] != b"\x9d\x01\x2a":
                raise ValueError("WebP image data is invalid.")
            width = int.from_bytes(payload[6:8], "little") & 0x3FFF
            height = int.from_bytes(payload[8:10], "little") & 0x3FFF
            saw_image = bool(width and height)
        offset = end + (length & 1)
    if offset != len(data) or not saw_image:
        raise ValueError("WebP data contains no decodable image frame.")


def _image_media_type(data: bytes, suffix: str = "") -> str:
    """Structurally validate supported image bytes and return their media type."""
    detected = ""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        detected = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        detected = "image/jpeg"
    elif len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        detected = "image/webp"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        detected = "image/gif"
    expected = _IMAGE_MEDIA_TYPES.get(str(suffix).casefold())
    if not detected or (expected and detected != expected):
        raise ValueError("Images must be valid PNG, JPEG, WebP, or GIF files.")
    validators = {
        "image/png": _validate_png,
        "image/jpeg": _validate_jpeg,
        "image/webp": _validate_webp,
        "image/gif": _validate_gif,
    }
    validators[detected](data)
    return detected


def _embedded_image_payload_bytes(value: Any) -> int:
    url = str(value or "")
    if not url.casefold().startswith("data:"):
        return 0
    separator = url.find(",")
    if separator < 0:
        return 0
    encoded = "".join(url[separator + 1:].split())
    padding = len(encoded) - len(encoded.rstrip("="))
    return max(0, (len(encoded) * 3) // 4 - padding)


def _normalize_image_url(value: Any) -> str:
    url = str(value or "").strip()
    if url.casefold().startswith("data:"):
        match = re.fullmatch(
            r"data:(image/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/=\s]+)",
            url,
            flags=re.IGNORECASE,
        )
        if not match:
            raise ValueError("Image data URLs must be base64 PNG, JPEG, WebP, or GIF data.")
        try:
            payload = base64.b64decode("".join(match.group(2).split()), validate=True)
        except (ValueError, TypeError) as error:
            raise ValueError("Image data URLs must contain valid base64 data.") from error
        if not payload or len(payload) > _MAX_IMAGE_BYTES:
            raise ValueError("Images must be non-empty and no larger than 20 MiB.")
        media_type = _image_media_type(payload)
        if media_type.casefold() != match.group(1).casefold():
            raise ValueError("The image data URL media type does not match its bytes.")
        return f"data:{media_type};base64,{base64.b64encode(payload).decode('ascii')}"
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Remote images must use an HTTPS URL without embedded credentials.")
    if Path(parsed.path).suffix.casefold() not in _IMAGE_MEDIA_TYPES:
        raise ValueError("HTTPS image URLs must end in .png, .jpg, .jpeg, .webp, or .gif.")
    return url


@dataclass(frozen=True)
class ImageInput:
    """A validated OpenRouter image input from HTTPS or a local image file."""

    source: str | os.PathLike[str]
    detail: str = "auto"

    @classmethod
    def from_path(
        cls, path: str | os.PathLike[str], *, detail: str = "auto"
    ) -> "ImageInput":
        return cls(path, detail=detail)

    @classmethod
    def from_url(cls, url: str, *, detail: str = "auto") -> "ImageInput":
        return cls(url, detail=detail)

    def __post_init__(self) -> None:
        detail = str(self.detail).strip().casefold()
        if detail not in _IMAGE_DETAILS:
            raise ValueError("image detail must be auto, low, or high.")
        raw_source = os.fspath(self.source)
        if str(raw_source).strip().casefold().startswith(("https://", "data:")):
            normalized = _normalize_image_url(raw_source)
        else:
            path = Path(raw_source).expanduser()
            if not path.is_file():
                raise ValueError(f"Image file does not exist: {path}")
            size = path.stat().st_size
            if not 0 < size <= _MAX_IMAGE_BYTES:
                raise ValueError("Images must be non-empty and no larger than 20 MiB.")
            payload = path.read_bytes()
            if not 0 < len(payload) <= _MAX_IMAGE_BYTES:
                raise ValueError("Images must be non-empty and no larger than 20 MiB.")
            media_type = _image_media_type(payload, path.suffix)
            normalized = (
                f"data:{media_type};base64,{base64.b64encode(payload).decode('ascii')}"
            )
        object.__setattr__(self, "source", normalized)
        object.__setattr__(self, "detail", detail)

    def to_content_part(self) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": str(self.source), "detail": self.detail},
        }


def normalize_message_content(value: Any) -> Any:
    """Normalize strings and multimodal parts to Chat Completions content."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, ImageInput):
        return [value.to_content_part()]
    if isinstance(value, Mapping):
        parts = [value]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise TypeError("Message content must be text, ImageInput, or content parts.")
    normalized: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, ImageInput):
            normalized.append(part.to_content_part())
            continue
        if isinstance(part, str):
            text = part.strip()
            if text:
                normalized.append({"type": "text", "text": text})
            continue
        if not isinstance(part, Mapping):
            raise TypeError("Every content part must be text, ImageInput, or a mapping.")
        part_type = str(part.get("type") or "").strip()
        if part_type == "text":
            text = str(part.get("text") or "").strip()
            if not text:
                raise ValueError("Text content parts cannot be blank.")
            normalized.append({"type": "text", "text": text})
            continue
        if part_type == "image_url":
            image_value = part.get("image_url")
            if isinstance(image_value, str):
                image_value = {"url": image_value}
            if not isinstance(image_value, Mapping):
                raise ValueError("image_url content parts require a URL object.")
            detail = str(image_value.get("detail") or "auto").strip().casefold()
            if detail not in _IMAGE_DETAILS:
                raise ValueError("image detail must be auto, low, or high.")
            normalized.append({
                "type": "image_url",
                "image_url": {
                    "url": _normalize_image_url(image_value.get("url")),
                    "detail": detail,
                },
            })
            continue
        raise ValueError(f"Unsupported message content part type {part_type!r}.")
    if not normalized:
        raise ValueError("Message content cannot be blank.")
    total_image_bytes = sum(
        _embedded_image_payload_bytes(part.get("image_url", {}).get("url"))
        for part in normalized
        if part.get("type") == "image_url"
    )
    if total_image_bytes > _MAX_IMAGE_REQUEST_BYTES:
        raise ValueError("One message can contain at most 40 MiB of embedded image data.")
    return normalized


def multimodal_input(text: str, *images: ImageInput | str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Build standard text/image_url parts from text and image sources."""
    parts: list[Any] = []
    if str(text).strip():
        parts.append({"type": "text", "text": str(text).strip()})
    parts.extend(image if isinstance(image, ImageInput) else ImageInput(image) for image in images)
    normalized = normalize_message_content(parts)
    if not isinstance(normalized, list):
        raise AssertionError("multimodal input normalization returned an invalid value")
    return normalized


def required_input_modalities(value: Any) -> set[str]:
    """Return the input modalities required by message content or a message list."""
    modalities: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, ImageInput):
            modalities.add("image")
        elif isinstance(item, str):
            if item.strip():
                modalities.add("text")
        elif isinstance(item, Mapping):
            if item.get("type") == "image_url":
                modalities.add("image")
            elif item.get("type") == "text":
                modalities.add("text")
            elif "content" in item:
                visit(item.get("content"))
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return modalities


def safe_input_descriptor(value: Any) -> Any:
    """Describe multimodal content without copying image bytes or signed URLs."""
    if isinstance(value, Mapping) and value.get("role"):
        descriptor = {
            "role": str(value.get("role")),
            "content": safe_input_descriptor(value.get("content")),
        }
        if value.get("tool_call_id") is not None:
            descriptor["tool_call_id"] = str(value.get("tool_call_id"))
        return descriptor
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(item, Mapping) and item.get("role") for item in value)
    ):
        return [safe_input_descriptor(item) for item in value]
    normalized = normalize_message_content(value)
    if isinstance(normalized, str) or normalized is None:
        return normalized
    result: list[dict[str, str]] = []
    for part in normalized:
        if part["type"] == "text":
            result.append({"type": "text", "text": str(part["text"])})
            continue
        image_url = str(part["image_url"]["url"])
        if image_url.startswith("data:"):
            media_type, encoded = image_url[5:].split(";base64,", 1)
            payload = base64.b64decode(encoded, validate=True)
            digest = hashlib.sha256(payload).hexdigest()[:16]
            descriptor = f"[{media_type} image, {len(payload)} bytes, sha256:{digest}]"
        else:
            parsed = urlsplit(image_url)
            descriptor = f"[HTTPS image from {parsed.hostname or 'remote host'}]"
        result.append({"type": "image_descriptor", "text": descriptor})
    return result


def _text_character_count(value: Any) -> int:
    """Count only textual request data; image base64 is never treated as tokens."""
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, ImageInput):
        return 0
    if isinstance(value, Mapping):
        if value.get("type") == "image_url":
            return 0
        if value.get("type") == "text":
            return len(str(value.get("text") or ""))
        return sum(
            _text_character_count(child)
            for key, child in value.items()
            if key not in {"image_url"}
        )
    if isinstance(value, (list, tuple)):
        return sum(_text_character_count(child) for child in value)
    return len(str(value))


def _image_count(value: Any) -> int:
    if isinstance(value, ImageInput):
        return 1
    if isinstance(value, Mapping):
        if value.get("type") == "image_url":
            return 1
        return sum(_image_count(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_image_count(child) for child in value)
    return 0


def _normalize_messages(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("Every chat message must be a mapping.")
        item = dict(message)
        if "content" in item:
            item["content"] = normalize_message_content(item.get("content"))
        normalized.append(item)
    return normalized


def normalize_request_options(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a canonical, persistence-safe OpenRouter target request envelope.

    The envelope intentionally covers the Chat Completions body rather than other
    OpenRouter endpoint families. Model selection, messages, streaming lifecycle,
    cost accounting, output-token safety and the tuned reasoning effort remain
    Evalt-owned so tests and production cannot silently disagree about them.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("request_options must be a JSON-serializable mapping.")
    try:
        normalized = json.loads(
            json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("request_options must contain only JSON-serializable values.") from error
    if not isinstance(normalized, dict):
        raise TypeError("request_options must be a JSON object.")
    reserved = sorted(_RESERVED_OPENROUTER_REQUEST_FIELDS & normalized.keys())
    if reserved:
        if "stream" in reserved:
            raise ValueError(
                "Evalt.run() is a durable non-streaming call; stream belongs to a "
                "separate iterator lifecycle and cannot be hidden in request_options."
            )
        raise ValueError(
            "Evalt owns these request fields: " + ", ".join(reserved) + "."
        )

    def reject_secrets(item: Any, path: str = "request_options") -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                label = str(key)
                if label.casefold() in _SECRET_FIELD_NAMES:
                    raise ValueError(f"Secret-bearing field {path}.{label} is not allowed.")
                reject_secrets(child, f"{path}.{label}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                reject_secrets(child, f"{path}[{index}]")

    reject_secrets(normalized)
    reasoning = normalized.get("reasoning")
    if isinstance(reasoning, dict) and "effort" in reasoning:
        raise ValueError(
            "Evalt tunes reasoning effort as part of the model configuration; "
            "do not set request_options['reasoning']['effort']."
        )
    if "reasoning_effort" in normalized:
        raise ValueError(
            "Evalt tunes reasoning_effort as part of the model configuration."
        )
    return normalized


def request_options_fingerprint(value: Mapping[str, Any] | None) -> str:
    normalized = normalize_request_options(value)
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ProviderError(RuntimeError):
    """The model provider could not complete a request."""

    provider_spend_usd: float | None = None
    test_budget_usd: float | None = None


class BudgetExceeded(RuntimeError):
    """A new provider call would exceed the customer-approved hard cap."""


@dataclass(frozen=True)
class Turn:
    input: Any
    approved_output: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "input", normalize_message_content(self.input))
        object.__setattr__(self, "approved_output", str(self.approved_output).strip())


@dataclass(frozen=True)
class Example:
    input: Any
    approved_output: str
    id: str = ""
    turns: tuple[Turn, ...] = ()
    group: str = ""
    difficulty: str = "typical"
    weight: float = 1.0
    critical: bool = False

    def conversation(self) -> tuple[Turn, ...]:
        return self.turns or (Turn(self.input, self.approved_output),)

    @classmethod
    def from_value(cls, value: Example | dict[str, Any], index: int = 0) -> Example:
        if isinstance(value, cls):
            return value
        turns = tuple(
            Turn(item.get("input", ""), str(item.get("approved_output", item.get("expected", ""))).strip())
            for item in value.get("turns", [])
        )
        first = turns[0] if turns else Turn(value.get("input", ""), str(value.get("approved_output", "")).strip())
        return cls(
            first.input,
            turns[-1].approved_output if turns else first.approved_output,
            str(value.get("id", "")).strip() or f"example-{index + 1}",
            turns,
            str(value.get("group", "")).strip(),
            str(value.get("difficulty", "typical")).strip() or "typical",
            float(value.get("weight", 1.0)),
            bool(value.get("critical", False)),
        )


def _safe_example_payload(example: Example) -> dict[str, Any]:
    """Serialize a result receipt without embedding image bytes or source URLs."""
    payload = asdict(example)
    payload["input"] = safe_input_descriptor(example.input)
    payload["turns"] = [
        {
            "input": safe_input_descriptor(turn.input),
            "approved_output": turn.approved_output,
        }
        for turn in example.turns
    ]
    payload["input_replayable"] = "image" not in {
        modality
        for turn in example.conversation()
        for modality in required_input_modalities(turn.input)
    }
    return payload


@dataclass(frozen=True)
class Completion:
    content: str
    model: str
    generation_id: str
    cost_usd: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    message: dict[str, Any] = field(default_factory=dict)
    tool_calls: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class Judgment:
    passed: bool
    score: float
    reason: str


@dataclass(frozen=True)
class CaseResult:
    example_id: str
    split: str
    prompt_kind: str
    output: str
    approved_output: str
    passed: bool
    score: float
    reason: str
    target_cost_usd: float
    evaluator_cost_usd: float
    target_generation_id: str
    evaluator_generation_id: str
    target_latency_ms: int = 0
    evaluator_latency_ms: int = 0
    group: str = ""
    difficulty: str = "typical"
    weight: float = 1.0
    critical: bool = False


@dataclass
class ModelResult:
    model: str
    selected_prompt: str
    baseline_pass_rate: float
    selected_pass_rate: float
    holdout_pass_rate: float
    baseline_holdout_pass_rate: float
    estimated_production_cost_per_call_usd: float
    estimated_cost_per_successful_call_usd: float
    optimization_spend_usd: float
    passed_quality_floor: bool
    target_latency_p50_ms: int = 0
    target_latency_p90_ms: int = 0
    passed_latency_ceiling: bool = True
    holdout_unique_scenarios: int = 0
    holdout_executions: int = 0
    holdout_execution_pass_rate: float = 0.0
    baseline_holdout_execution_pass_rate: float = 0.0
    holdout_pass_rates_by_difficulty: dict[str, float] = field(default_factory=dict)
    passed_difficulty_floors: bool = True
    prompt_origin: str = "starting_prompt"
    prompt_candidates_tested: int = 1
    prompt_rewrites_tested: int = 0
    selected_prompt_changed: bool = False
    few_shot_example_ids: list[str] = field(default_factory=list)
    few_shot_provenance: list[dict[str, Any]] = field(default_factory=list)
    cases: list[CaseResult] = field(default_factory=list)
    final_test_evidence_status: str = "NO_FINAL_TEST"
    final_test_confidence_level: float = 0.95
    final_test_accuracy_lower_bound: float | None = None
    target_accuracy_statistically_supported: bool = False
    minimum_zero_failure_scenarios: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationResult:
    objective: str
    quality_threshold: float
    exploratory: bool
    winner: ModelResult
    models: list[ModelResult]
    total_provider_spend_usd: float
    warnings: list[str]
    quality_frontier: list[dict[str, Any]]
    diminishing_returns: dict[str, Any]
    regression_suite: dict[str, Any]
    elapsed_seconds: float = 0.0
    comparison_integrity: dict[str, Any] = field(default_factory=dict)
    omitted_configurations: list[dict[str, str]] = field(default_factory=list)
    unavailable_models: list[dict[str, str]] = field(default_factory=list)
    incomplete_models: list[dict[str, str]] = field(default_factory=list)
    skipped_budget_models: list[str] = field(default_factory=list)
    pruned_models: list[str] = field(default_factory=list)
    screening_results: list[dict[str, Any]] = field(default_factory=list)
    winner_scope: str = "Best among every requested target"
    quality_gate_status: str = "QUALIFIED_ROUTE_SELECTED"
    continuation_recommendation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def save_regression_suite(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.regression_suite, handle, indent=2, ensure_ascii=False)
            handle.write("\n")


@dataclass(frozen=True)
class DraftAnswer:
    task: str
    input: str
    answer: str
    model: str
    provider_cost_usd: float

    def approve(self, example_id: str = "example-1") -> Example:
        return Example(self.input, self.answer, example_id)

    def correct(self, approved_output: str, example_id: str = "example-1") -> Example:
        corrected = str(approved_output).strip()
        if not corrected:
            raise ValueError("A corrected answer cannot be blank.")
        return Example(self.input, corrected, example_id)


class ChatTransport(Protocol):
    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
        request_options: Mapping[str, Any] | None = None,
    ) -> Completion: ...

    def estimate_cost(
        self, model: str, messages: list[dict[str, Any]], *, max_tokens: int
    ) -> float: ...


class OpenRouterTransport:
    """Small dependency-free OpenRouter client with current model-price lookup."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout_seconds: float = 600,
        app_url: str = "https://evalt.dev",
        catalog_ttl_seconds: float = 3600,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if not api_key and not os.environ.get("OPENROUTER_API_KEY"):
            dotenv_path = Path.cwd() / ".env"
            if dotenv_path.is_file():
                load_dotenv(dotenv_path=dotenv_path, override=False)
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Set OPENROUTER_API_KEY in the environment or in a .env file in the current "
                "working directory, or pass api_key= to Evalt()."
            )
        self.set_timeout_seconds(timeout_seconds)
        self._app_url = app_url
        self._opener = opener
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())
        self._catalog_ttl_seconds = max(0.0, float(catalog_ttl_seconds))
        self._catalog_loaded_at = 0.0
        self._prices: dict[str, tuple[float, float]] | None = None
        self._image_prices: dict[str, float] = {}
        self._input_modalities: dict[str, set[str]] = {}
        self._supported_parameters: dict[str, set[str]] = {}
        self._reasoning: dict[str, dict[str, Any]] = {}
        self._limits: dict[str, tuple[int | None, int | None]] = {}
        self._providers: dict[str, list[str]] = {}
        self._provider_route_names: dict[str, dict[str, str]] = {}
        self._endpoint_routes: dict[str, list[dict[str, Any]]] = {}
        self._catalog_items: list[dict[str, Any]] = []
        self._preferred_max_latency_seconds: float | None = None
        self._provider_sort = "price"
        self._request_timeout_override_var: contextvars.ContextVar[float | None] = (
            contextvars.ContextVar("evalt_request_timeout_override", default=None)
        )
        self._request_deadline_var: contextvars.ContextVar[float | None] = (
            contextvars.ContextVar("evalt_request_deadline", default=None)
        )
        self._request_deadline_limit_var: contextvars.ContextVar[float | None] = (
            contextvars.ContextVar("evalt_request_deadline_limit", default=None)
        )

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    def set_timeout_seconds(self, value: float) -> None:
        resolved = float(value)
        if not 0 < resolved <= 7200:
            raise ValueError("timeout_seconds must be greater than zero and no more than 7200 seconds.")
        self._timeout = resolved

    @contextmanager
    def request_timeout_override(self, value: float):
        """Apply a per-thread request deadline without racing other model lanes."""
        resolved = float(value)
        if not 0 < resolved <= 7200:
            raise ValueError("timeout override must be greater than zero and no more than 7200 seconds.")
        token = self._request_timeout_override_var.set(resolved)
        try:
            yield
        finally:
            self._request_timeout_override_var.reset(token)

    @contextmanager
    def request_deadline_override(self, value: float):
        """Bound every request in one model lane by a shared wall-clock deadline."""
        resolved = float(value)
        if not 0 < resolved <= 7200:
            raise ValueError("request deadline must be greater than zero and no more than 7200 seconds.")
        deadline_token = self._request_deadline_var.set(
            time.monotonic() + resolved
        )
        limit_token = self._request_deadline_limit_var.set(resolved)
        try:
            yield
        finally:
            self._request_deadline_var.reset(deadline_token)
            self._request_deadline_limit_var.reset(limit_token)

    def set_performance_policy(
        self,
        *,
        preferred_max_latency_seconds: float | None = None,
        provider_sort: str = "price",
    ) -> None:
        """Set current provider-performance preferences for subsequent calls."""
        if provider_sort not in {"price", "latency", "throughput"}:
            raise ValueError("provider_sort must be price, latency, or throughput.")
        if preferred_max_latency_seconds is not None and preferred_max_latency_seconds <= 0:
            raise ValueError("preferred_max_latency_seconds must be positive when provided.")
        self._provider_sort = provider_sort
        self._preferred_max_latency_seconds = (
            float(preferred_max_latency_seconds)
            if preferred_max_latency_seconds is not None else None
        )

    def _request(self, url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "HTTP-Referer": self._app_url,
            "X-Title": "Evalt Python SDK",
        }
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
        started = time.monotonic()
        request_override = self._request_timeout_override_var.get()
        request_timeout = float(
            request_override if request_override is not None else self._timeout
        )
        absolute_deadline = self._request_deadline_var.get()
        if absolute_deadline is not None:
            remaining = float(absolute_deadline) - started
            if remaining <= 0:
                limit = float(self._request_deadline_limit_var.get() or 0.0)
                raise ProviderError(
                    f"model evaluation exceeded its {limit:g}s total deadline"
                )
            request_timeout = min(request_timeout, remaining)
        try:
            response_context = (
                self._opener(request, timeout=request_timeout)
                if self._opener is not None
                else urlopen(request, timeout=request_timeout, context=self._ssl_context)
            )
            with response_context as response:
                if not hasattr(response, "read1"):
                    raw = response.read()
                else:
                    chunks: list[bytes] = []
                    while True:
                        if time.monotonic() - started > request_timeout:
                            raise TimeoutError(
                                f"provider response exceeded the {request_timeout:g}s total deadline"
                            )
                        chunk = response.read1(65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                return json.loads(raw.decode("utf-8"))
        except HTTPError as error:
            raw_detail = error.read().decode("utf-8", errors="replace")
            detail = _safe_provider_error_detail(raw_detail)
            provider_error = ProviderError(f"OpenRouter returned HTTP {error.code}: {detail}")
            provider_error.status_code = int(error.code)
            try:
                payload = json.loads(raw_detail)
                provider_error.provider_name = str(
                    ((payload.get("error") or {}).get("metadata") or {}).get("provider_name")
                    or payload.get("provider")
                    or ""
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                provider_error.provider_name = ""
            raise provider_error from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ProviderError(f"OpenRouter request failed: {error}") from error

    def _load_prices(self) -> dict[str, tuple[float, float]]:
        if self._prices is not None and time.monotonic() - self._catalog_loaded_at < self._catalog_ttl_seconds:
            return self._prices
        payload = self._request(OPENROUTER_MODELS_URL)
        endpoint_catalog_available = True
        try:
            endpoint_payload = self._request(OPENROUTER_ZDR_ENDPOINTS_URL)
        except ProviderError:
            endpoint_catalog_available = False
            endpoint_payload = {"data": []}
        endpoint_candidates: dict[str, list[dict[str, Any]]] = {}
        recognized_endpoints = 0
        for endpoint in endpoint_payload.get("data", []):
            model_id = str(endpoint.get("model_id") or "")
            parameters = {str(value) for value in endpoint.get("supported_parameters") or []}
            if model_id and ({"max_tokens", "max_completion_tokens"} & parameters):
                endpoint_candidates.setdefault(model_id, []).append(endpoint)
                recognized_endpoints += 1
        if endpoint_catalog_available and endpoint_payload.get("data") and not recognized_endpoints:
            # Test doubles and legacy gateways may return model-shaped data at the
            # endpoint URL. Do not mistake that malformed catalog for a proof that
            # every ZDR route disappeared.
            endpoint_catalog_available = False
        prices: dict[str, tuple[float, float]] = {}
        catalog_items: list[dict[str, Any]] = []
        self._supported_parameters = {}
        self._image_prices = {}
        self._input_modalities = {}
        self._reasoning = {}
        self._limits = {}
        self._providers = {}
        self._provider_route_names = {}
        self._endpoint_routes = {}
        for intelligence_rank, item in enumerate(payload.get("data", []), start=1):
            pricing = item.get("pricing") or {}
            try:
                model_id = str(item["id"])
                endpoints = endpoint_candidates.get(model_id, [])
                if endpoint_catalog_available and not endpoints:
                    # A model-level listing is not an executable private route. Omit
                    # it before spend instead of launching a guaranteed ZDR 404.
                    continue
                architecture = item.get("architecture") or {}
                model_modalities = {
                    str(value).strip().casefold()
                    for value in architecture.get("input_modalities") or ["text"]
                    if str(value).strip()
                }
                for route in endpoints:
                    route_architecture = route.get("architecture") or {}
                    if not route.get("input_modalities") and not route_architecture.get("input_modalities"):
                        # OpenRouter's current flat /endpoints/zdr feed proves the
                        # executable private route but omits modality fields. Its
                        # documented model catalog remains the capability source in
                        # that schema. An explicit endpoint declaration still wins,
                        # so a text-only route is never upgraded by this fallback.
                        route["input_modalities"] = sorted(model_modalities)
                self._endpoint_routes[model_id] = [dict(value) for value in endpoints]
                top_provider = item.get("top_provider") or {}

                def endpoint_capacity(value: dict[str, Any]) -> int:
                    raw = value.get("max_completion_tokens") or value.get("context_length") or 0
                    try:
                        return max(0, int(raw))
                    except (TypeError, ValueError):
                        return 0

                def endpoint_price(value: dict[str, Any]) -> float:
                    route = value.get("pricing") or {}
                    return float(route.get("prompt") or "inf") + float(route.get("completion") or "inf")

                model_ceiling_raw = top_provider.get("max_completion_tokens") or item.get("context_length") or 131072
                try:
                    preferred_capacity = min(131072, max(1, int(model_ceiling_raw)))
                except (TypeError, ValueError):
                    preferred_capacity = 131072
                capacity_eligible = [value for value in endpoints if endpoint_capacity(value) >= preferred_capacity]

                def endpoint_contract_score(value: dict[str, Any]) -> int:
                    parameters = {str(item) for item in value.get("supported_parameters") or []}
                    # Structured output prevents correct answers from failing an
                    # exact-JSON task merely because a provider wrapped them in
                    # prose. Adjustable reasoning is the next most useful search
                    # lever. Prefer the strongest executable contract first, then
                    # choose the cheapest route that provides it.
                    return (
                        2 * int(bool({"structured_outputs", "response_format"} & parameters))
                        + int(bool({"reasoning", "reasoning_effort"} & parameters))
                    )

                if capacity_eligible:
                    best_contract = max(endpoint_contract_score(value) for value in capacity_eligible)
                    provider_pool = sorted(
                        (value for value in capacity_eligible if endpoint_contract_score(value) == best_contract),
                        key=endpoint_price,
                    )
                    endpoint = provider_pool[0]
                elif endpoints:
                    maximum_capacity = max(endpoint_capacity(value) for value in endpoints)
                    maximum_capacity_pool = [
                        value for value in endpoints if endpoint_capacity(value) == maximum_capacity
                    ]
                    best_contract = max(endpoint_contract_score(value) for value in maximum_capacity_pool)
                    provider_pool = sorted(
                        (value for value in maximum_capacity_pool if endpoint_contract_score(value) == best_contract),
                        key=endpoint_price,
                    )
                    endpoint = provider_pool[0]
                else:
                    provider_pool = []
                    endpoint = None
                route_pricing = (endpoint or {}).get("pricing") or pricing
                prices[model_id] = (
                    float(route_pricing.get("prompt") or 0),
                    float(route_pricing.get("completion") or 0),
                )
                self._image_prices[model_id] = float(
                    route_pricing.get("image") or pricing.get("image") or 0
                )
                if endpoint_catalog_available:
                    executable_endpoints = [
                        value for value in endpoints if value.get("status") in {None, 0}
                    ]
                    endpoint_modalities = [
                        self._endpoint_input_modalities(value)
                        for value in executable_endpoints
                    ]
                    self._input_modalities[model_id] = set().union(
                        *endpoint_modalities
                    ) if endpoint_modalities else {"text"}
                    image_route_prices = []
                    for value in executable_endpoints:
                        if "image" not in self._endpoint_input_modalities(value):
                            continue
                        try:
                            image_route_prices.append(float((value.get("pricing") or {}).get("image") or 0))
                        except (TypeError, ValueError):
                            pass
                    if image_route_prices:
                        self._image_prices[model_id] = max(image_route_prices)
                else:
                    modality_source = architecture.get("input_modalities") or ["text"]
                    self._input_modalities[model_id] = {
                        str(value).strip().casefold()
                        for value in modality_source if str(value).strip()
                    }
                supported_source = (endpoint or {}).get("supported_parameters") or item.get("supported_parameters") or []
                self._supported_parameters[model_id] = {str(value) for value in supported_source}
                if not endpoint_catalog_available:
                    # Model-level metadata cannot prove that the selected private endpoint
                    # accepts optional reasoning controls, so fail closed on that lever.
                    self._supported_parameters[model_id] -= {"reasoning", "reasoning_effort"}
                self._reasoning[model_id] = dict(item.get("reasoning") or {})
                context_value = (endpoint or {}).get("context_length") or top_provider.get("context_length") or item.get("context_length")
                completion_value = (endpoint or {}).get("max_completion_tokens") or top_provider.get("max_completion_tokens") or context_value
                try:
                    context_limit = int(context_value) if context_value is not None else None
                except (TypeError, ValueError):
                    context_limit = None
                try:
                    completion_limit = int(completion_value) if completion_value is not None else None
                except (TypeError, ValueError):
                    completion_limit = None
                self._limits[model_id] = (context_limit, completion_limit)
                provider_tags = [str(value["tag"]) for value in provider_pool if value.get("tag")]
                provider_families = {
                    str(value.get("provider_name") or value.get("tag") or "").strip().casefold()
                    for value in provider_pool
                    if value.get("provider_name") or value.get("tag")
                }
                designer_endpoints = [
                    value for value in endpoints
                    if endpoint_capacity(value) >= 12000
                    and ({"response_format", "structured_outputs"} & {
                        str(parameter)
                        for parameter in value.get("supported_parameters") or []
                    })
                    and value.get("status") in {None, 0}
                ]
                designer_provider_families = {
                    str(value.get("provider_name") or value.get("tag") or "")
                    .strip().casefold()
                    for value in designer_endpoints
                    if value.get("provider_name") or value.get("tag")
                }
                designer_latencies = []
                for value in designer_endpoints:
                    try:
                        designer_latencies.append(
                            float((value.get("latency_last_30m") or {})["p90"])
                        )
                    except (KeyError, TypeError, ValueError):
                        pass
                if provider_tags:
                    self._providers[model_id] = provider_tags[:3]
                    self._provider_route_names[model_id] = {
                        str(value["tag"]): str(value.get("provider_name") or "")
                        for value in provider_pool[:3] if value.get("tag")
                    }
                intelligence = ((item.get("benchmarks") or {}).get("artificial_analysis") or {}).get("intelligence_index")
                try:
                    intelligence = float(intelligence) if intelligence is not None else None
                except (TypeError, ValueError):
                    intelligence = None
                catalog_items.append({
                    "id": str(item["id"]),
                    "intelligence": intelligence,
                    "intelligence_rank": intelligence_rank,
                    "blended_price": prices[model_id][0] * 1_000_000 + prices[model_id][1] * 2_000_000,
                    "supported_parameters": sorted(self._supported_parameters[model_id]),
                    "input_modalities": sorted(self._input_modalities[model_id]),
                    "image_price": self._image_prices[model_id],
                    "reasoning": item.get("reasoning") or {},
                    "context_length": context_limit,
                    "max_completion_tokens": completion_limit,
                    # Count genuinely independent serving organizations, not two
                    # regional endpoints owned by the same provider. This number
                    # drives reliability preferences in role shortlisting.
                    "private_provider_routes": len(provider_families),
                    "designer_provider_routes": len(designer_provider_families),
                    "designer_p90_latency_ms": (
                        min(designer_latencies) if designer_latencies else None
                    ),
                })
            except (KeyError, TypeError, ValueError):
                continue
        self._prices = prices
        self._catalog_items = catalog_items
        self._catalog_loaded_at = time.monotonic()
        return prices

    def model_catalog(self) -> list[dict[str, Any]]:
        """Return the current provider catalog used for role shortlisting.

        Intelligence metadata never qualifies a production route by itself; Evalt's
        task-specific holdouts remain the promotion gate.
        """
        self._load_prices()
        return [dict(item) for item in self._catalog_items]

    @staticmethod
    def _endpoint_capacity(endpoint: Mapping[str, Any]) -> int:
        raw = endpoint.get("max_completion_tokens") or endpoint.get("context_length") or 0
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _endpoint_input_modalities(endpoint: Mapping[str, Any]) -> set[str]:
        """Return modalities explicitly executable on one provider endpoint.

        Model-level vision metadata is not endpoint proof. When an endpoint omits
        modality metadata, retain ordinary text routing but fail closed for images.
        """
        source = (
            endpoint.get("input_modalities")
            or (endpoint.get("architecture") or {}).get("input_modalities")
            or ["text"]
        )
        return {
            str(value).strip().casefold()
            for value in source if str(value).strip()
        }

    @staticmethod
    def _endpoint_latency(endpoint: Mapping[str, Any]) -> float:
        try:
            return max(0.0, float((endpoint.get("latency_last_30m") or {})["p90"]))
        except (KeyError, TypeError, ValueError):
            return float("inf")

    @staticmethod
    def _provider_family_slug(endpoint_tag: Any) -> str:
        """Return the resilient provider-family slug for an endpoint catalog tag.

        OpenRouter's ZDR endpoint feed includes region and quantization tags such as
        ``google-vertex/global`` and ``deepinfra/fp4``. Those exact endpoint tags can
        disappear between catalog fetch and execution even while another endpoint in
        the same provider family is healthy. Base slugs still let
        ``require_parameters`` enforce the request envelope without pinning a brittle
        variant.
        """
        return str(endpoint_tag or "").strip().split("/", 1)[0]

    @classmethod
    def _portable_response_schema(cls, value: Any) -> Any:
        """Return the strict-schema subset shared by current routed providers.

        Some provider-native compilers reject otherwise valid JSON Schema keywords.
        Amazon Bedrock, for example, accepts ``minItems`` only for 0 or 1 and rejects
        numeric bounds and ``maxItems``. Evalt validates those stronger constraints
        after decoding, so omitting them from the provider grammar keeps routing
        portable without weakening the executable suite contract.
        """
        if isinstance(value, list):
            return [cls._portable_response_schema(item) for item in value]
        if not isinstance(value, dict):
            return value
        unsupported = {
            "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
            "multipleOf", "maxItems", "minLength", "maxLength",
        }
        portable: dict[str, Any] = {}
        for key, item in value.items():
            if key in unsupported:
                continue
            if key == "minItems" and item not in {0, 1}:
                continue
            portable[str(key)] = cls._portable_response_schema(item)
        return portable

    def _routes_for_request(
        self,
        model: str,
        *,
        max_tokens: int,
        response_schema: Mapping[str, Any] | None,
        reasoning_requested: bool,
        required_modalities: Iterable[str] | None = None,
    ) -> tuple[list[str], dict[str, str], str | None, set[str], list[Mapping[str, Any]]]:
        """Choose current endpoints for this request, not for the model's maximum size.

        The catalog's largest-context endpoint is often slower or less capable than a
        smaller endpoint that easily fits a 4k-token orchestration call.  Pinning the
        former made suite design both slow and brittle.  This preflight uses the actual
        bounded request and only returns routes that can honor its envelope.
        """
        required = {
            str(value).strip().casefold()
            for value in (required_modalities or ()) if str(value).strip()
        }
        endpoints = [
            value for value in self._endpoint_routes.get(model, [])
            if value.get("status") in {None, 0}
            and self._endpoint_capacity(value) >= int(max_tokens)
            and required.issubset(self._endpoint_input_modalities(value))
        ]
        if response_schema:
            exact = [
                value for value in endpoints
                if "response_format" in {
                    str(parameter)
                    for parameter in value.get("supported_parameters") or []
                }
            ]
            if exact:
                endpoints = exact
            else:
                endpoints = [
                    value for value in endpoints
                    if "structured_outputs" in {
                        str(parameter)
                        for parameter in value.get("supported_parameters") or []
                    }
                ]
        if reasoning_requested:
            endpoints = [
                value for value in endpoints
                if {"reasoning", "reasoning_effort"} & {
                    str(parameter)
                    for parameter in value.get("supported_parameters") or []
                }
            ]
        if not endpoints:
            return [], {}, None, set(), []

        token_fields = []
        for token_field in ("max_tokens", "max_completion_tokens"):
            supporting = [
                value for value in endpoints
                if token_field in {
                    str(parameter)
                    for parameter in value.get("supported_parameters") or []
                }
            ]
            if supporting:
                families = {
                    str(value.get("provider_name") or value.get("tag") or "").casefold()
                    for value in supporting
                }
                token_fields.append((len(families), len(supporting), token_field, supporting))
        if not token_fields:
            return [], {}, None, set(), []
        _family_count, _route_count, token_field, endpoints = max(
            token_fields,
            key=lambda item: (item[0], item[1], item[2] == "max_tokens"),
        )

        def route_price(value: Mapping[str, Any]) -> float:
            try:
                pricing = value.get("pricing") or {}
                return float(pricing.get("prompt") or 0) + float(
                    pricing.get("completion") or 0
                )
            except (TypeError, ValueError):
                return float("inf")

        endpoints.sort(key=lambda value: (
            0 if float(value.get("uptime_last_5m") or 0) >= 99 else 1,
            self._endpoint_latency(value),
            route_price(value),
        ))
        selected: list[Mapping[str, Any]] = []
        selected_families: set[str] = set()
        for value in endpoints:
            family = str(value.get("provider_name") or value.get("tag") or "").casefold()
            if family in selected_families:
                continue
            selected.append(value)
            selected_families.add(family)
            if len(selected) == 3:
                break
        if not selected:
            selected = endpoints[:1]
        tags = list(dict.fromkeys(
            self._provider_family_slug(value.get("tag"))
            for value in selected
            if self._provider_family_slug(value.get("tag"))
        ))
        names = {
            self._provider_family_slug(value.get("tag")): str(value.get("provider_name") or "")
            for value in selected if value.get("tag")
        }
        common_parameters = set.intersection(*(
            {
                str(parameter)
                for parameter in value.get("supported_parameters") or []
            }
            for value in selected
        ))
        return tags, names, token_field, common_parameters, selected

    def configuration_support(
        self,
        configuration: str,
        required_modalities: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Preflight a model/effort pair against current routed capabilities."""
        explicit_reasoning = "#reasoning=" in configuration
        model, effort = self._split_configuration(configuration)
        prices = self._load_prices()
        if model not in prices:
            return {
                "supported": False,
                "reason": f"OpenRouter did not return a current priced ZDR route for {model!r}.",
            }
        required = {
            str(value).strip().casefold()
            for value in (required_modalities or ()) if str(value).strip()
        }
        routed_endpoints = self._endpoint_routes.get(model, [])
        modality_endpoints = [
            value for value in routed_endpoints
            if value.get("status") in {None, 0}
            and required.issubset(self._endpoint_input_modalities(value))
        ]
        unsupported_modalities = required - self._input_modalities.get(model, {"text"})
        if unsupported_modalities or (routed_endpoints and not modality_endpoints):
            return {
                "supported": False,
                "reason": (
                    f"The current OpenRouter catalog does not list {model!r} as accepting "
                    f"required input modality/modalities: {', '.join(sorted(unsupported_modalities))}."
                ),
            }
        supported = (
            set().union(*(
                {str(parameter) for parameter in value.get("supported_parameters") or []}
                for value in modality_endpoints
            ))
            if modality_endpoints else self._supported_parameters.get(model, set())
        )
        reasoning = self._reasoning.get(model, {})
        reasoning_supported = bool({"reasoning", "reasoning_effort"} & supported)
        if explicit_reasoning and effort == "none" and reasoning.get("mandatory"):
            return {
                "supported": False,
                "reason": f"Reasoning is mandatory for {model!r}; the no-reasoning configuration was omitted before spend.",
            }
        if explicit_reasoning and effort != "none" and not reasoning_supported:
            return {
                "supported": False,
                "reason": f"The current ZDR endpoint for {model!r} does not support adjustable reasoning; this configuration was omitted before spend.",
            }
        supported_efforts = {
            str(value) for value in reasoning.get("supported_efforts") or []
        }
        if explicit_reasoning and effort != "none" and supported_efforts and effort not in supported_efforts:
            return {
                "supported": False,
                "reason": f"The current ZDR route for {model!r} does not list reasoning effort {effort!r}; this configuration was omitted before spend.",
            }
        return {
            "supported": True,
            "reason": "Current routed capability metadata accepts this configuration.",
        }

    def estimate_cost(
        self, model: str, messages: list[dict[str, Any]], *, max_tokens: int
    ) -> float:
        messages = _normalize_messages(messages)
        support = self.configuration_support(
            model, required_input_modalities(messages)
        )
        if not support.get("supported"):
            raise ProviderError(str(support.get("reason") or "Unsupported model configuration."))
        explicit_reasoning = "#reasoning=" in model
        model, effort = self._split_configuration(model)
        prices = self._load_prices()
        reasoning_metadata = self._reasoning.get(model, {})
        if not explicit_reasoning and reasoning_metadata.get("mandatory"):
            effort = str(reasoning_metadata.get("default_effort") or "medium")
        max_tokens = self._bounded_output_tokens(model, messages, max_tokens, effort)
        if model not in prices:
            raise ProviderError(
                f"OpenRouter did not return current pricing for model {model!r}; "
                "the SDK will not start an unpriced call."
            )
        required_modalities = required_input_modalities(messages)
        reasoning_is_requested = bool(
            explicit_reasoning or reasoning_metadata.get("mandatory")
        )
        _tags, _names, _token_field, _supported, selected_routes = self._routes_for_request(
            model,
            max_tokens=max_tokens,
            response_schema=None,
            reasoning_requested=reasoning_is_requested,
            required_modalities=required_modalities,
        )
        if self._endpoint_routes.get(model) and not selected_routes:
            raise ProviderError(
                f"No current ZDR endpoint for {model!r} accepts the request modalities and output envelope."
            )
        if selected_routes:
            route_prices = [value.get("pricing") or {} for value in selected_routes]
            prompt_price = max(float(value.get("prompt") or 0) for value in route_prices)
            completion_price = max(float(value.get("completion") or 0) for value in route_prices)
            routed_image_price = max(float(value.get("image") or 0) for value in route_prices)
        else:
            prompt_price, completion_price = prices[model]
            routed_image_price = self._image_prices.get(model, 0.0)
        estimated_prompt_tokens = max(1, math.ceil(_text_character_count(messages) / 3))
        images = _image_count(messages)
        # Catalog image prices are charged directly when present. Some providers
        # instead fold vision into prompt tokens, so reserve a deliberately large
        # 8k-token image allowance (and a small absolute floor) before authorizing.
        conservative_image_price = max(
            routed_image_price, prompt_price * 8192, 0.001
        )
        return (
            estimated_prompt_tokens * prompt_price
            + max_tokens * completion_price
            + images * conservative_image_price
        )

    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
        request_options: Mapping[str, Any] | None = None,
    ) -> Completion:
        configuration = model
        messages = _normalize_messages(messages)
        support = self.configuration_support(
            configuration, required_input_modalities(messages)
        )
        if not support.get("supported"):
            raise ProviderError(str(support.get("reason") or "Unsupported model configuration."))
        explicit_reasoning = "#reasoning=" in model
        model, reasoning_effort = self._split_configuration(model)
        self._load_prices()
        reasoning_metadata = self._reasoning.get(model, {})
        if not explicit_reasoning and reasoning_metadata.get("mandatory"):
            reasoning_effort = str(reasoning_metadata.get("default_effort") or "medium")
        max_tokens = self._bounded_output_tokens(model, messages, max_tokens, reasoning_effort)
        supported = self._supported_parameters.get(model, set())
        target_options = normalize_request_options(request_options)
        requested_reasoning_option = target_options.get("reasoning")
        reasoning_is_requested = bool(
            explicit_reasoning
            or reasoning_metadata.get("mandatory")
            or requested_reasoning_option is not None
        )
        required_modalities = required_input_modalities(messages)
        route_tags, route_names, routed_token_field, routed_supported, selected_routes = (
            self._routes_for_request(
                model,
                max_tokens=max_tokens,
                response_schema=response_schema,
                reasoning_requested=reasoning_is_requested,
                required_modalities=required_modalities,
            )
        )
        if self._endpoint_routes.get(model) and (
            not selected_routes or ("image" in required_modalities and not route_tags)
        ):
            raise ProviderError(
                f"No current ZDR endpoint for {model!r} accepts the request modalities and output envelope."
            )
        if route_tags:
            self._providers[model] = route_tags
            self._provider_route_names[model] = route_names
            supported = routed_supported
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "provider": {
                "zdr": True,
                "data_collection": "deny",
                "require_parameters": True,
                "sort": self._provider_sort,
            },
            "usage": {"include": True},
        }
        if self._providers.get(model):
            body["provider"].update({
                "only": self._providers[model],
                "allow_fallbacks": len(self._providers[model]) > 1,
            })
        if self._preferred_max_latency_seconds is not None:
            body["provider"]["preferred_max_latency"] = {
                "p90": self._preferred_max_latency_seconds
            }
        if routed_token_field == "max_tokens":
            body["max_tokens"] = max_tokens
        elif routed_token_field == "max_completion_tokens":
            body["max_completion_tokens"] = max_tokens
        elif "max_completion_tokens" in supported:
            body["max_completion_tokens"] = max_tokens
        elif "max_tokens" in supported:
            body["max_tokens"] = max_tokens
        else:
            raise ProviderError(
                f"OpenRouter did not report a bounded output-token parameter for {model!r}."
            )
        if "temperature" in supported:
            body["temperature"] = 0
        reasoning_supported = bool({"reasoning", "reasoning_effort"} & supported)
        if explicit_reasoning and reasoning_effort != "none" and not reasoning_supported:
            raise ProviderError(f"The current ZDR endpoint for {model!r} does not support adjustable reasoning.")
        if explicit_reasoning and reasoning_effort == "none" and reasoning_metadata.get("mandatory"):
            raise ProviderError(f"Reasoning is mandatory for {model!r}; choose low, medium, high, or another model.")
        supported_efforts = {str(value) for value in reasoning_metadata.get("supported_efforts") or []}
        if explicit_reasoning and reasoning_effort != "none" and supported_efforts and reasoning_effort not in supported_efforts:
            raise ProviderError(
                f"The current ZDR endpoint for {model!r} does not list reasoning effort {reasoning_effort!r}."
            )
        if reasoning_supported and (explicit_reasoning or reasoning_metadata.get("mandatory")):
            body["reasoning"] = (
                {"enabled": False, "exclude": True}
                if reasoning_effort == "none"
                else {"effort": reasoning_effort, "exclude": True}
            )
        if response_schema and ({"structured_outputs", "response_format"} & supported):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "evalt_result",
                    "strict": True,
                    "schema": self._portable_response_schema(response_schema),
                },
            }
        # The customer envelope wins over Evalt defaults but never over the
        # route-owned fields rejected by normalize_request_options(). Provider
        # preferences are merged so callers can use every OpenRouter routing
        # control without accidentally dropping Evalt's safe defaults.
        requested_provider = target_options.pop("provider", None)
        if requested_provider is not None:
            if not isinstance(requested_provider, dict):
                raise ValueError("request_options['provider'] must be a JSON object.")
            body["provider"].update(requested_provider)
        if route_tags:
            # Endpoint capability preflight owns the executable provider set.
            # Customer routing preferences may narrow or order that safe set, but
            # must never replace it with an endpoint that was not verified for the
            # request's image/schema/reasoning/token envelope.
            compatible = list(route_tags)
            requested_only = (
                requested_provider.get("only")
                if isinstance(requested_provider, dict) else None
            )
            if requested_only is not None:
                if not isinstance(requested_only, (list, tuple)):
                    raise ValueError("request_options['provider']['only'] must be a list.")
                allowed = {str(value).strip().casefold() for value in requested_only}
                compatible = [
                    tag for tag in compatible
                    if tag.casefold() in allowed
                    or str(route_names.get(tag) or "").casefold() in allowed
                ]
            requested_ignore = (
                requested_provider.get("ignore")
                if isinstance(requested_provider, dict) else None
            )
            if requested_ignore is not None:
                if not isinstance(requested_ignore, (list, tuple)):
                    raise ValueError("request_options['provider']['ignore'] must be a list.")
                ignored = {str(value).strip().casefold() for value in requested_ignore}
                compatible = [
                    tag for tag in compatible
                    if tag.casefold() not in ignored
                    and str(route_names.get(tag) or "").casefold() not in ignored
                ]
            if not compatible:
                modalities = ", ".join(sorted(required_modalities)) or "text"
                raise ProviderError(
                    "The requested provider controls exclude every current ZDR "
                    f"endpoint verified for {model!r} and modalities: {modalities}."
                )
            body["provider"]["only"] = compatible
            body["provider"]["allow_fallbacks"] = (
                bool(body["provider"].get("allow_fallbacks", True))
                and len(compatible) > 1
            )
        # Privacy and parameter compatibility are Evalt safety invariants, not
        # tunable production behavior. All other OpenRouter provider-routing
        # controls remain available in the tested request envelope.
        body["provider"].update({
            "zdr": True,
            "data_collection": "deny",
            "require_parameters": True,
        })
        requested_reasoning = target_options.pop("reasoning", None)
        if requested_reasoning is not None:
            if not isinstance(requested_reasoning, dict):
                raise ValueError("request_options['reasoning'] must be a JSON object.")
            body.setdefault("reasoning", {}).update(requested_reasoning)
        body.update(target_options)
        body["model"] = model
        body["messages"] = messages
        body["usage"] = {"include": True}
        started = time.monotonic()
        fallback_spend_usd = 0.0
        transient_same_route_retries = 0
        while True:
            try:
                payload = self._request(OPENROUTER_CHAT_URL, body)
            except ProviderError as error:
                failed_provider = str(getattr(error, "provider_name", "") or "")
                route_names = self._provider_route_names.get(model, {})
                current_routes = list(body["provider"].get("only") or [])
                remaining_routes = [
                    tag for tag in current_routes
                    if not failed_provider or route_names.get(tag) != failed_provider
                ]
                retryable_status = getattr(error, "status_code", None) in {
                    408, 409, 429, 500, 502, 503, 504,
                }
                if (
                    retryable_status
                    and current_routes
                    and not remaining_routes
                    and transient_same_route_retries < 1
                ):
                    # A sole private route can be healthy immediately before a
                    # burst-level 429. One short same-envelope retry is cheaper and
                    # more reliable than abandoning an otherwise qualified model;
                    # configuration errors (400/422) never retry.
                    transient_same_route_retries += 1
                    time.sleep(0.25)
                    continue
                if (
                    getattr(error, "status_code", None)
                    not in {400, 408, 409, 422, 429, 500, 502, 503, 504}
                    or not failed_provider
                    or not remaining_routes
                    or remaining_routes == current_routes
                ):
                    raise
                # OpenRouter can surface one provider's 429 even with fallbacks
                # enabled. Retry the same model/configuration against the already
                # preflighted remaining private routes, never against a new model.
                body["provider"]["only"] = remaining_routes
                body["provider"]["allow_fallbacks"] = len(remaining_routes) > 1
                continue
            try:
                choice = payload["choices"][0]
                message = dict(choice.get("message") or {})
                content = message.get("content")
                tool_calls = tuple(
                    dict(item) for item in (message.get("tool_calls") or [])
                    if isinstance(item, dict)
                )
                usage = payload.get("usage") or {}
                billed_cost = float(usage.get("cost") or 0)
                if choice.get("finish_reason") == "length":
                    error = ProviderError("OpenRouter reached the response limit before finishing.")
                    error.code = "PROVIDER_TRUNCATED"
                    expanded = self._bounded_output_tokens(model, messages, 131072, reasoning_effort)
                    error.retry_with_more_tokens = expanded > max_tokens
                    error.cost_usd = fallback_spend_usd + billed_cost
                    error.generation_id = str(payload.get("id") or "")
                    raise error
                if (content is None or not str(content).strip()) and tool_calls:
                    # A tool call is a valid assistant answer. Canonical JSON keeps
                    # it judgeable in suites while the structured form remains on
                    # Completion/RoutedAnswer for production execution.
                    content = json.dumps(
                        {"tool_calls": list(tool_calls)},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                if content is None or not str(content).strip():
                    # An empty answer is not evidence that more output allowance is
                    # needed. If OpenRouter identifies the serving provider, move to
                    # another already-preflighted compatible route at the same token
                    # ceiling. Preserve the failed route's reported cost.
                    failed_provider = str(payload.get("provider") or "")
                    route_names = self._provider_route_names.get(model, {})
                    current_routes = list(body["provider"].get("only") or [])
                    remaining_routes = [
                        tag for tag in current_routes
                        if not failed_provider
                        or (
                            tag.casefold() != failed_provider.casefold()
                            and route_names.get(tag, "").casefold() != failed_provider.casefold()
                        )
                    ]
                    if failed_provider and remaining_routes and remaining_routes != current_routes:
                        fallback_spend_usd += billed_cost
                        body["provider"]["only"] = remaining_routes
                        body["provider"]["allow_fallbacks"] = len(remaining_routes) > 1
                        continue
                    error = ProviderError("OpenRouter returned an empty completion for this model configuration.")
                    error.code = "PROVIDER_EMPTY"
                    error.retry_with_more_tokens = False
                    error.cost_usd = fallback_spend_usd + billed_cost
                    error.generation_id = str(payload.get("id") or "")
                    raise error
                return Completion(
                    content=str(content),
                    model=configuration,
                    generation_id=str(payload.get("id") or ""),
                    cost_usd=fallback_spend_usd + billed_cost,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    latency_ms=round((time.monotonic() - started) * 1000),
                    finish_reason=(
                        str(choice.get("finish_reason"))
                        if choice.get("finish_reason") is not None else None
                    ),
                    native_finish_reason=(
                        str(choice.get("native_finish_reason"))
                        if choice.get("native_finish_reason") is not None else None
                    ),
                    message=message,
                    tool_calls=tool_calls,
                )
            except (KeyError, IndexError, TypeError, ValueError) as error:
                detail = _safe_provider_error_detail(json.dumps(payload, ensure_ascii=False))
                raise ProviderError(f"OpenRouter returned an invalid completion payload: {detail}") from error

    @staticmethod
    def _split_configuration(configuration: str) -> tuple[str, str]:
        """Decode an auditable model + reasoning-effort candidate identifier."""
        marker = "#reasoning="
        if marker not in configuration:
            return configuration, "none"
        model, effort = configuration.rsplit(marker, 1)
        if effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"Unsupported reasoning effort {effort!r}.")
        return model, effort

    @staticmethod
    def _reasoning_token_ceiling(requested_tokens: int, effort: str) -> int:
        # OpenRouter counts hidden reasoning inside the completion allowance.  The
        # visible answer may be a two-field JSON object while a reasoning model still
        # consumes tens of thousands of tokens before emitting it.  Keep these
        # defaults deliberately generous so a paid evaluation is not thrown away by
        # an allowance tuned to visible output length.
        minimums = {
            "none": 0,
            "minimal": 32768,
            "low": 65536,
            "medium": 98304,
            "high": 131072,
            "xhigh": 131072,
            "max": 131072,
        }
        return max(1, int(requested_tokens), minimums.get(effort, 0))

    def _bounded_output_tokens(
        self,
        model: str,
        messages: list[dict[str, Any]],
        requested_tokens: int,
        effort: str,
    ) -> int:
        desired = self._reasoning_token_ceiling(requested_tokens, effort)
        context_limit, completion_limit = self._limits.get(model, (None, None))
        # Use a conservative character/token ratio and reserve framing overhead so
        # Evalt never asks for a full context window on top of a non-empty prompt.
        estimated_prompt_tokens = max(1, math.ceil(_text_character_count(messages) / 3) + 128)
        limits = [desired]
        if completion_limit:
            limits.append(max(1, completion_limit))
        if context_limit:
            limits.append(max(1, context_limit - estimated_prompt_tokens))
        return max(1, min(limits))


class _Budget:
    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = float(limit_usd)
        self.spent_usd = 0.0
        self.reserved_usd = 0.0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def authorize(self, estimate_usd: float) -> None:
        estimate = float(estimate_usd)
        with self._condition:
            if estimate < 0:
                raise BudgetExceeded("A provider-cost estimate cannot be negative.")
            # Parallel calls reserve their worst-case completion allowance. A lane
            # that would fit after an in-flight reservation settles must wait, not
            # become a false budget failure. Once no reservation can free enough
            # room, the hard cap still fails closed before another provider call.
            while self.spent_usd + self.reserved_usd + estimate > self.limit_usd + 1e-12:
                if self.reserved_usd <= 1e-12:
                    raise BudgetExceeded(
                        f"The next estimated call would exceed the ${self.limit_usd:.4f} cap."
                    )
                self._condition.wait()
            self.reserved_usd += estimate

    def commit(self, actual_usd: float, reserved_estimate_usd: float = 0.0) -> None:
        actual = max(0.0, float(actual_usd))
        reserved = max(0.0, float(reserved_estimate_usd))
        with self._condition:
            self.reserved_usd = max(0.0, self.reserved_usd - reserved)
            self.spent_usd += actual
            exceeded = self.spent_usd + self.reserved_usd > self.limit_usd + 1e-12
            self._condition.notify_all()
            if exceeded:
                raise BudgetExceeded(
                    "The provider-reported cost exceeded the customer-approved hard cap."
                )

    def release(self, reserved_estimate_usd: float) -> None:
        with self._condition:
            self.reserved_usd = max(0.0, self.reserved_usd - max(0.0, float(reserved_estimate_usd)))
            self._condition.notify_all()


class _BudgetScope:
    """Charge one parallel lane to the shared cap while retaining lane-local spend."""

    def __init__(self, shared: _Budget) -> None:
        self.shared = shared
        self.limit_usd = shared.limit_usd
        self.spent_usd = 0.0

    def authorize(self, estimate_usd: float) -> None:
        self.shared.authorize(estimate_usd)

    def commit(self, actual_usd: float, reserved_estimate_usd: float = 0.0) -> None:
        self.shared.commit(actual_usd, reserved_estimate_usd)
        self.spent_usd += max(0.0, float(actual_usd))

    def release(self, reserved_estimate_usd: float) -> None:
        self.shared.release(reserved_estimate_usd)


class Client:
    """Optimize prompts and compare target models using customer-approved examples."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        transport: ChatTransport | None = None,
        custom_scorers: Mapping[str, Scorer] | None = None,
    ) -> None:
        self.transport = transport or OpenRouterTransport(api_key)
        self.custom_scorers = normalize_scorer_registry(custom_scorers)
        self._custom_scorer_locks = {
            scorer_id: threading.Lock() for scorer_id in self.custom_scorers
        }

    def validate_evaluator_runtime(
        self, evaluator: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """Validate an evaluator and resolve local code before provider activity."""

        policy = _validate_evaluator_policy(dict(evaluator) if evaluator else None)
        if policy["type"] == "custom":
            resolve_registered_scorer(policy, self.custom_scorers)
        return policy

    def _call(
        self,
        budget: _Budget,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
        request_options: Mapping[str, Any] | None = None,
    ) -> Completion:
        for output_expansion_attempt in range(2):
            # A simple 2x retry was ineffective for reasoning models: 600 -> 1,200
            # was mapped to the same reasoning floor on both attempts.  A truncation
            # retry now requests a genuinely larger provider ceiling.
            attempt_max_tokens = (
                int(max_tokens)
                if output_expansion_attempt == 0
                else max(int(max_tokens) * 2, 131072)
            )
            estimate = self.transport.estimate_cost(model, messages, max_tokens=attempt_max_tokens)
            budget.authorize(estimate)
            try:
                complete_kwargs: dict[str, Any] = {
                    "max_tokens": attempt_max_tokens,
                    "response_schema": response_schema,
                }
                if request_options:
                    complete_kwargs["request_options"] = request_options
                try:
                    completion = self.transport.complete(
                        model, messages, **complete_kwargs
                    )
                except TypeError as error:
                    if request_options and "request_options" in str(error):
                        raise ProviderError(
                            "The custom transport does not implement Evalt's "
                            "request_options contract."
                        ) from error
                    raise
            except ProviderError as error:
                billed_cost = float(getattr(error, "cost_usd", 0) or 0)
                if billed_cost:
                    budget.commit(billed_cost, estimate)
                else:
                    budget.release(estimate)
                if output_expansion_attempt == 0 and getattr(error, "code", "") == "PROVIDER_TRUNCATED" and getattr(error, "retry_with_more_tokens", True):
                    continue
                raise
            except Exception:
                budget.release(estimate)
                raise
            budget.commit(completion.cost_usd, estimate)
            return completion
        raise ProviderError("The provider did not return a complete answer after one budgeted expansion.")

    def draft_answer(
        self,
        *,
        task: str,
        input: str,
        model: str = "openai/gpt-5-mini",
        max_cost_usd: float = 0.10,
    ) -> DraftAnswer:
        task_text = str(task).strip()
        input_text = str(input).strip()
        if not task_text or not input_text:
            raise ValueError("Both task and input are required.")
        budget = _Budget(max_cost_usd)
        response = self._call(
            budget,
            model,
            [
                {"role": "system", "content": task_text},
                {"role": "user", "content": input_text},
            ],
            max_tokens=8192,
        )
        return DraftAnswer(task_text, input_text, response.content, response.model, response.cost_usd)

    def optimize(
        self,
        *,
        prompt: str,
        examples: Iterable[Example | dict[str, Any]],
        models: Iterable[str],
        optimizer_model: str = "openai/gpt-5.6-luna",
        evaluator_model: str = "openai/gpt-5.6-luna",
        objective: str = "cheapest_passing",
        quality_threshold: float = 0.95,
        max_optimization_cost_usd: float = 2.00,
        rounds: int = 3,
        optimize_prompt: bool = True,
        minimum_meaningful_quality_gain: float = 0.03,
        allow_few_shot: bool = True,
        max_few_shot_examples: int = 3,
        max_cost_per_run_usd: float | None = None,
        representative_input_chars: int | None = None,
        representative_output_tokens: int | None = None,
        incumbent_model: str | None = None,
        allowed_accuracy_regression: float = 0.0,
        adaptive_search: bool = False,
        holdout_repeats: int = 2,
        evaluator: dict[str, Any] | None = None,
        difficulty_thresholds: dict[str, float] | None = None,
        max_parallel_models: int = 16,
        max_parallel_scenarios: int = 32,
        max_p90_latency_seconds: float | None = None,
        latency_value_usd_per_second: float = 0.0,
        target_max_tokens: int | None = None,
        request_options: Mapping[str, Any] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> OptimizationResult:
        optimization_started = time.monotonic()
        prompt_text = str(prompt).strip()
        model_list = list(dict.fromkeys(str(model).strip() for model in models if str(model).strip()))
        example_list = [Example.from_value(value, index) for index, value in enumerate(examples)]
        suite_required_modalities = {"text"}
        for example in example_list:
            for turn in example.conversation():
                suite_required_modalities.update(required_input_modalities(turn.input))
        if "image" in suite_required_modalities and optimize_prompt:
            raise ValueError(
                "Image suites require optimize_prompt=False because the prompt optimizer "
                "does not inspect or rewrite against image pixels."
            )
        self._validate(
            prompt_text,
            example_list,
            model_list,
            quality_threshold,
            max_optimization_cost_usd,
            rounds,
            minimum_meaningful_quality_gain,
        )
        allowed_objectives = {
            "cheapest_passing", "cheapest_at_accuracy", "lowest_cost_at_accuracy",
            "highest_quality", "best_within_cost", "best_within_price", "constrained",
            "match_baseline_at_lowest_cost",
        }
        if objective not in allowed_objectives:
            raise ValueError(f"objective must be one of {sorted(allowed_objectives)}.")
        if max_cost_per_run_usd is not None and max_cost_per_run_usd <= 0:
            raise ValueError("max_cost_per_run_usd must be positive when provided.")
        if not 0 <= allowed_accuracy_regression < 1:
            raise ValueError("allowed_accuracy_regression must be between zero and one.")
        if not 1 <= int(holdout_repeats) <= 5:
            raise ValueError("holdout_repeats must be between one and five.")
        if not 1 <= int(max_parallel_models) <= 32:
            raise ValueError("max_parallel_models must be between one and thirty-two.")
        if not 1 <= int(max_parallel_scenarios) <= 128:
            raise ValueError("max_parallel_scenarios must be between one and one hundred twenty-eight.")
        if max_p90_latency_seconds is not None and max_p90_latency_seconds <= 0:
            raise ValueError("max_p90_latency_seconds must be positive when provided.")
        if latency_value_usd_per_second < 0:
            raise ValueError("latency_value_usd_per_second cannot be negative.")
        if target_max_tokens is not None and not 1 <= int(target_max_tokens) <= 131072:
            raise ValueError("target_max_tokens must be between one and 131072.")
        target_request_options = normalize_request_options(request_options)
        evaluator_policy = self.validate_evaluator_runtime(evaluator)
        if evaluator_policy["type"] == "custom":
            evaluator_model = (
                f"custom/{evaluator_policy['scorer_id']}@"
                f"{evaluator_policy['scorer_version']}"
            )
        difficulty_floor_policy = _validate_difficulty_thresholds(difficulty_thresholds)
        performance_setter = getattr(self.transport, "set_performance_policy", None)
        if callable(performance_setter):
            performance_setter(
                preferred_max_latency_seconds=max_p90_latency_seconds,
                provider_sort="latency" if latency_value_usd_per_second > 0 else "price",
            )
        objective = {
            "lowest_cost_at_accuracy": "cheapest_at_accuracy",
            "best_within_price": "best_within_cost",
        }.get(objective, objective)
        train, dev, holdout = _split_examples(example_list, prompt_text)
        budget = _Budget(max_optimization_cost_usd)
        results = []
        unavailable_models = []
        incomplete_models = []
        skipped_budget_models = []
        pruned_models = []
        screening_results: list[dict[str, Any]] = []
        screening_cases_by_model: dict[str, list[CaseResult]] = {}
        seed_prompt_by_model: dict[str, str] = {}
        seed_few_shot_ids_by_model: dict[str, list[str]] = {}
        prompt_origin_by_model: dict[str, str] = {}
        fixed_prompt_models: set[str] = set()
        omitted_configurations = []
        progress_lock = threading.Lock()

        def request_timeout_context(cap_seconds: float):
            override = getattr(self.transport, "request_timeout_override", None)
            if not callable(override):
                return nullcontext()
            configured = float(
                getattr(self.transport, "timeout_seconds", cap_seconds)
            )
            return override(min(float(cap_seconds), configured))

        def request_deadline_context(cap_seconds: float):
            override = getattr(self.transport, "request_deadline_override", None)
            if not callable(override):
                return nullcontext()
            configured = float(
                getattr(self.transport, "timeout_seconds", cap_seconds)
            )
            return override(min(float(cap_seconds), configured))

        resolved_target_envelope = int(target_max_tokens) if target_max_tokens is not None else {
            "exact_text": 64,
            "exact_json": 1024,
            "numeric_tolerance": 64,
            "custom": 8192,
            "semantic": 8192,
        }[evaluator_policy["type"]]
        # A recurring label/score route is not viable when one provider call can
        # stall the whole tournament for two minutes. Larger-output and semantic
        # jobs retain materially larger deadlines; explicit suites can also raise
        # their output envelope when long reasoning is genuinely part of the job.
        deep_call_deadline = (
            45.0 if resolved_target_envelope <= 128
            else 90.0 if resolved_target_envelope <= 1024
            else float(getattr(self.transport, "timeout_seconds", 600.0))
        )
        # The request cap above applies to one provider response. Prompt search,
        # validation, and final confirmation share a second lane-wide deadline so
        # a sequence of individually legal calls cannot turn a simple recurring
        # task into a five-minute finalist.
        deep_model_deadline = min(
            float(getattr(self.transport, "timeout_seconds", deep_call_deadline * 2)),
            deep_call_deadline * 2,
        )

        def emit_progress(event: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            # Multiple model lanes finish concurrently. Serialize callbacks so CLI
            # progress remains valid line-delimited JSON and consumer callbacks do
            # not need their own synchronization.
            with progress_lock:
                progress_callback(dict(event))

        catalog_loader = getattr(self.transport, "model_catalog", None)
        catalog_snapshot: list[dict[str, Any]] = []
        if callable(catalog_loader) and model_list:
            catalog_snapshot = list(catalog_loader() or [])
        catalog_intelligence: dict[str, float] = {}
        catalog_modalities: dict[str, set[str]] = {}
        for catalog_item in catalog_snapshot:
            try:
                catalog_modalities[str(catalog_item["id"])] = {
                    str(value).strip().casefold()
                    for value in catalog_item.get("input_modalities") or ["text"]
                    if str(value).strip()
                }
                intelligence = catalog_item.get("intelligence")
                if intelligence is not None:
                    catalog_intelligence[str(catalog_item["id"])] = float(intelligence)
            except (KeyError, TypeError, ValueError):
                continue
        support_checker = getattr(self.transport, "configuration_support", None)
        if callable(support_checker):
            eligible_models: list[str] = []
            for configuration in model_list:
                base_model = configuration.split("#reasoning=", 1)[0]
                listed_modalities = catalog_modalities.get(base_model)
                if listed_modalities is not None and not suite_required_modalities.issubset(listed_modalities):
                    support = {
                        "supported": False,
                        "reason": (
                            f"The current model catalog does not list {base_model!r} as "
                            "accepting all suite input modalities: "
                            f"{', '.join(sorted(suite_required_modalities))}."
                        ),
                    }
                else:
                    try:
                        support = support_checker(
                            configuration,
                            required_modalities=suite_required_modalities,
                        )
                    except TypeError:
                        # Preserve compatibility with custom transports implementing
                        # the original one-argument capability hook.
                        support = support_checker(configuration)
                if support.get("supported"):
                    eligible_models.append(configuration)
                    continue
                omitted_configurations.append({
                    "model": configuration,
                    "reason": str(support.get("reason") or "Unsupported by current provider capability metadata."),
                    "stage": "preflight",
                })
                emit_progress({
                    "event": "configuration_omitted",
                    "model": configuration,
                    "reason": omitted_configurations[-1]["reason"],
                    "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
                })
            model_list = eligible_models
        if not model_list:
            reasons = "; ".join(
                f"{item['model']}: {item['reason']}" for item in omitted_configurations
            )
            raise ProviderError(
                "No requested configuration is compatible with current provider capability metadata. "
                + reasons
            )

        broad_models: list[str] = []
        if adaptive_search:
            seen_base_models: set[str] = set()
            hone_models: list[str] = []
            for configuration in model_list:
                base_model = configuration.split("#reasoning=", 1)[0]
                if base_model in seen_base_models:
                    hone_models.append(configuration)
                else:
                    seen_base_models.add(base_model)
                    broad_models.append(configuration)
            model_list = broad_models + hone_models
        model_started_at: dict[str, float] = {}
        model_timing_lock = threading.Lock()

        def model_elapsed_seconds(model: str) -> float | None:
            with model_timing_lock:
                started_at = model_started_at.get(model)
            return (
                round(time.monotonic() - started_at, 3)
                if started_at is not None else None
            )

        def evaluate(model: str) -> ModelResult:
            with model_timing_lock:
                model_started_at[model] = time.monotonic()
            emit_progress({
                "event": "model_started", "model": model,
                "optimize_prompt": bool(
                    optimize_prompt and model not in fixed_prompt_models
                ),
                "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
            })
            model_budget = _BudgetScope(budget)
            with request_deadline_context(deep_model_deadline):
                with request_timeout_context(deep_call_deadline):
                    return self._evaluate_model(
                        prompt_text, train, dev, holdout, model, optimizer_model,
                        evaluator_model, quality_threshold, model_budget, rounds,
                        bool(allow_few_shot and optimize_prompt),
                        max_few_shot_examples, representative_input_chars,
                        representative_output_tokens, int(holdout_repeats), evaluator_policy,
                        difficulty_floor_policy, int(max_parallel_scenarios),
                        baseline_dev_cases=screening_cases_by_model.get(model),
                        seed_prompt=seed_prompt_by_model.get(model),
                        seed_few_shot_ids=seed_few_shot_ids_by_model.get(model),
                        prompt_origin=prompt_origin_by_model.get(model, "starting_prompt"),
                        optimize_prompt=bool(
                            optimize_prompt and model not in fixed_prompt_models
                        ),
                        progress_callback=emit_progress,
                        target_max_tokens=(int(target_max_tokens) if target_max_tokens is not None else None),
                        request_options=target_request_options,
                    )

        def run_batch(configurations: list[str]) -> list[ModelResult]:
            if not configurations:
                return []
            if len(configurations) == 1 or int(max_parallel_models) == 1:
                completed_sequential: list[ModelResult] = []
                for model_index, model in enumerate(configurations):
                    try:
                        item = evaluate(model)
                        completed_sequential.append(item)
                        emit_progress({
                            "event": "model_completed", "model": model,
                            "final_test_pass_rate": item.holdout_pass_rate,
                            "final_test_scenarios": item.holdout_unique_scenarios,
                            "final_test_executions": item.holdout_executions,
                            "passed_quality_floor": item.passed_quality_floor,
                            "target_latency_p50_ms": item.target_latency_p50_ms,
                            "target_latency_p90_ms": item.target_latency_p90_ms,
                            "optimization_spend_usd": item.optimization_spend_usd,
                            "prompt_candidates_tested": item.prompt_candidates_tested,
                            "prompt_rewrites_tested": item.prompt_rewrites_tested,
                            "selected_prompt_changed": item.selected_prompt_changed,
                            "final_test_evidence_status": item.final_test_evidence_status,
                            "final_test_confidence_level": item.final_test_confidence_level,
                            "final_test_accuracy_lower_bound": item.final_test_accuracy_lower_bound,
                            "target_accuracy_statistically_supported": item.target_accuracy_statistically_supported,
                            "minimum_zero_failure_scenarios": item.minimum_zero_failure_scenarios,
                            "model_elapsed_seconds": model_elapsed_seconds(model),
                            "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
                        })
                    except BudgetExceeded as error:
                        incomplete_models.append({"model": model, "reason": str(error)})
                        emit_progress({
                            "event": "model_incomplete", "model": model,
                            "reason": str(error),
                            "model_elapsed_seconds": model_elapsed_seconds(model),
                        })
                        skipped_budget_models.extend(configurations[model_index + 1 :])
                        break
                    except ProviderError as error:
                        unavailable_models.append({"model": model, "reason": str(error)})
                        emit_progress({
                            "event": "model_unavailable", "model": model,
                            "reason": str(error),
                            "model_elapsed_seconds": model_elapsed_seconds(model),
                        })
                return completed_sequential
            completed: dict[str, ModelResult] = {}
            with ThreadPoolExecutor(max_workers=min(int(max_parallel_models), len(configurations))) as pool:
                futures = {
                    _submit_with_context(pool, evaluate, model): model
                    for model in configurations
                }
                for future in as_completed(futures):
                    model = futures[future]
                    try:
                        item = future.result()
                        completed[model] = item
                        emit_progress({
                            "event": "model_completed", "model": model,
                            "final_test_pass_rate": item.holdout_pass_rate,
                            "final_test_scenarios": item.holdout_unique_scenarios,
                            "final_test_executions": item.holdout_executions,
                            "passed_quality_floor": item.passed_quality_floor,
                            "target_latency_p50_ms": item.target_latency_p50_ms,
                            "target_latency_p90_ms": item.target_latency_p90_ms,
                            "optimization_spend_usd": item.optimization_spend_usd,
                            "prompt_candidates_tested": item.prompt_candidates_tested,
                            "prompt_rewrites_tested": item.prompt_rewrites_tested,
                            "selected_prompt_changed": item.selected_prompt_changed,
                            "final_test_evidence_status": item.final_test_evidence_status,
                            "final_test_confidence_level": item.final_test_confidence_level,
                            "final_test_accuracy_lower_bound": item.final_test_accuracy_lower_bound,
                            "target_accuracy_statistically_supported": item.target_accuracy_statistically_supported,
                            "minimum_zero_failure_scenarios": item.minimum_zero_failure_scenarios,
                            "model_elapsed_seconds": model_elapsed_seconds(model),
                            "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
                        })
                    except BudgetExceeded as error:
                        incomplete_models.append({"model": model, "reason": str(error)})
                        emit_progress({
                            "event": "model_incomplete", "model": model,
                            "reason": str(error),
                            "model_elapsed_seconds": model_elapsed_seconds(model),
                        })
                    except ProviderError as error:
                        unavailable_models.append({"model": model, "reason": str(error)})
                        emit_progress({
                            "event": "model_unavailable", "model": model,
                            "reason": str(error),
                            "model_elapsed_seconds": model_elapsed_seconds(model),
                        })
            return [completed[model] for model in configurations if model in completed]

        def screen_model(model: str) -> dict[str, Any]:
            emit_progress({
                "event": "model_screen_started", "model": model,
                "screening_scenarios": len(dev),
                "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
            })
            model_budget = _BudgetScope(budget)
            # The broad screen exists to reject obviously weak, costly, or slow
            # configurations before deep prompt search. A single provider route
            # must not hold the entire first-run tournament hostage for the full
            # (potentially much larger) final-test deadline. The override is
            # thread-local, so ten parallel model lanes cannot change one another's
            # request contract. Deep testing still receives the suite's full limit.
            with request_timeout_context(30.0):
                cases = self._run_cases(
                    prompt_text, "screening-baseline", dev, "dev", model,
                    evaluator_model, model_budget, evaluator=evaluator_policy,
                    max_parallel_scenarios=int(max_parallel_scenarios),
                    target_max_tokens=(int(target_max_tokens) if target_max_tokens is not None else None),
                    request_options=target_request_options,
                )
            screening_cases_by_model[model] = cases
            costs = sorted(case.target_cost_usd for case in cases if case.target_cost_usd > 0)
            estimated_cost = (
                costs[max(0, min(len(costs) - 1, math.ceil(len(costs) * 0.90) - 1))]
                if costs else float("inf")
            )
            latencies = [case.target_latency_ms for case in cases if case.target_latency_ms > 0]
            result = {
                "model": model,
                "validation_pass_rate": round(_scenario_pass_rate(cases), 6),
                "validation_scenarios": len(dev),
                "estimated_production_cost_per_call_usd": round(estimated_cost, 10),
                "screening_spend_usd": round(model_budget.spent_usd, 10),
                "target_latency_p90_ms": round(_percentile(latencies, 0.90)) if latencies else 0,
                "status": "SCREENED",
            }
            emit_progress({
                "event": "model_screen_completed", **result,
                "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
            })
            return result

        def run_screening(configurations: list[str]) -> list[dict[str, Any]]:
            completed: dict[str, dict[str, Any]] = {}
            emit_progress({
                "event": "broad_screen_started",
                "configurations": len(configurations),
                "parallel_models": min(int(max_parallel_models), len(configurations)),
                "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
            })
            with ThreadPoolExecutor(max_workers=min(int(max_parallel_models), len(configurations))) as pool:
                futures = {
                    _submit_with_context(pool, screen_model, model): model
                    for model in configurations
                }
                for future in as_completed(futures):
                    model = futures[future]
                    try:
                        completed[model] = future.result()
                    except BudgetExceeded as error:
                        incomplete_models.append({"model": model, "reason": str(error), "stage": "screening"})
                        emit_progress({"event": "model_incomplete", "model": model, "reason": str(error)})
                    except ProviderError as error:
                        unavailable_models.append({"model": model, "reason": str(error), "stage": "screening"})
                        emit_progress({"event": "model_unavailable", "model": model, "reason": str(error)})
            emit_progress({
                "event": "broad_screen_completed",
                "configurations": len(configurations),
                "completed_configurations": len(completed),
                "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
            })
            return [completed[model] for model in configurations if model in completed]

        if adaptive_search:
            sizing_input = representative_input_chars or max(
                1, int(_percentile([
                    _text_character_count(turn.input)
                    for item in example_list for turn in item.conversation()
                ], 0.90))
            )
            sizing_output = representative_output_tokens or 64
            sizing_content: Any = "x" * sizing_input
            if "image" in suite_required_modalities:
                sizing_content = max(
                    (
                        turn.input
                        for item in example_list for turn in item.conversation()
                        if "image" in required_input_modalities(turn.input)
                    ),
                    key=lambda value: (_image_count(value), _text_character_count(value)),
                )
            sizing_messages = [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": sizing_content},
            ]
            broad_estimates: dict[str, float] = {}
            for model in broad_models:
                try:
                    broad_estimates[model] = self.transport.estimate_cost(
                        model, sizing_messages, max_tokens=int(sizing_output)
                    )
                except ProviderError:
                    broad_estimates[model] = float("inf")
            ordered_broad = sorted(
                broad_models,
                key=lambda model: (
                    0 if model == incumbent_model else 1,
                    broad_estimates[model],
                    model,
                ),
            )
            results = []
            if len(ordered_broad) > 4 and len(dev) >= 3:
                screening_results.extend(run_screening(ordered_broad))
                if not screening_results:
                    raise ProviderError("No broad model completed the validation screening stage.")
                # Screening is deliberately broad and cheap. Full prompt search plus
                # repeated final testing is deeper, but it must not collapse to a
                # single completed route merely because two initially selected
                # providers fail. Aim for roughly 1.5*sqrt(N) fully settled routes,
                # with a floor of four, and backfill failures from the measured
                # screening frontier until that target is actually complete.
                full_limit = min(
                    len(screening_results),
                    max(4, math.ceil(1.5 * math.sqrt(len(screening_results)))),
                )
                ranked_quality = sorted(
                    screening_results,
                    key=lambda item: (
                        -float(item["validation_pass_rate"]),
                        float(item["estimated_production_cost_per_call_usd"]),
                        str(item["model"]),
                    ),
                )
                measured_frontier = [
                    item for item in screening_results
                    if not any(
                        other["model"] != item["model"]
                        and float(other["validation_pass_rate"]) >= float(item["validation_pass_rate"])
                        and float(other["estimated_production_cost_per_call_usd"])
                        <= float(item["estimated_production_cost_per_call_usd"])
                        and (
                            float(other["validation_pass_rate"]) > float(item["validation_pass_rate"])
                            or float(other["estimated_production_cost_per_call_usd"])
                            < float(item["estimated_production_cost_per_call_usd"])
                        )
                        for other in screening_results
                    )
                ]
                measured_frontier.sort(
                    key=lambda item: (
                        -float(item["validation_pass_rate"]),
                        float(item["estimated_production_cost_per_call_usd"]),
                    )
                )
                full_priority: list[str] = []
                intelligence_anchor_model: str | None = None

                def prioritize(model: str) -> None:
                    if model not in full_priority:
                        full_priority.append(model)

                if incumbent_model and any(item["model"] == incumbent_model for item in screening_results):
                    prioritize(incumbent_model)
                # When every original-prompt screen is below the quality gate,
                # preserve one intelligence anchor from the requested field. A
                # weak prompt can flatten cheap-model scores; the anchor gives the
                # optimizer one capable lane on which to learn a prompt package
                # that can then be propagated back down the cost frontier.
                if max(float(item["validation_pass_rate"]) for item in screening_results) < quality_threshold:
                    anchor_candidates = [
                        item for item in screening_results
                        if item["model"].split("#reasoning=", 1)[0] in catalog_intelligence
                    ]
                    if anchor_candidates:
                        anchor = max(
                            anchor_candidates,
                            key=lambda item: (
                                catalog_intelligence[item["model"].split("#reasoning=", 1)[0]],
                                -float(item["estimated_production_cost_per_call_usd"]),
                            ),
                        )
                        intelligence_anchor_model = str(anchor["model"])
                        prioritize(intelligence_anchor_model)
                for item in measured_frontier:
                    prioritize(str(item["model"]))
                for item in ranked_quality[:3]:
                    prioritize(str(item["model"]))
                prioritize(str(min(
                    screening_results,
                    key=lambda item: (
                        float(item["estimated_production_cost_per_call_usd"]),
                        -float(item["validation_pass_rate"]),
                    ),
                )["model"]))
                for item in ranked_quality:
                    prioritize(str(item["model"]))

                attempted_for_full: list[str] = []
                full_results: list[ModelResult] = []
                # The shared budget already makes in-flight reservations wait
                # instead of failing. Keep the intelligence anchor first in the
                # priority list, but launch it in the same bounded wave as the
                # other finalists. Running the anchor alone added a complete
                # prompt-search/final-test round to every weak-starting-prompt
                # workflow without buying stronger evidence.
                # Launch the complete deep finalist set in one bounded wave. This
                # spends a little redundant work when a very early lane passes,
                # but avoids serial deep waves and gives a cheapest-route search a
                # real cross-model frontier before prompt propagation or reasoning
                # hone starts.
                completed_floor = full_limit
                while len(full_results) < completed_floor:
                    remaining = [model for model in full_priority if model not in attempted_for_full]
                    if not remaining:
                        break
                    needed = max(1, completed_floor - len(full_results))
                    # Provider failures are common enough that launching exactly
                    # ``needed`` lanes turns one timeout into a whole extra serial
                    # wave. Overbook by at most two independent finalists inside
                    # the same shared budget. Successful redundant lanes remain
                    # useful measured frontier evidence; the cap still fails
                    # closed before any unaffordable call starts.
                    failure_headroom = min(2, max(0, len(remaining) - needed))
                    wave = remaining[: needed + failure_headroom]
                    attempted_for_full.extend(wave)
                    full_results.extend(run_batch(wave))
                results.extend(full_results)

                # A weak starting prompt can make a capable cheap model look bad in
                # the first screen. Re-test the not-yet-deep models with up to two
                # successful prompt packages learned strictly from training cases.
                # Validation may promote that package; final-test answers remain
                # untouched until the model receives its full evaluation.
                rewrite_packages: list[tuple[str, list[str], str]] = []
                seen_packages: set[tuple[str, tuple[str, ...]]] = set()
                for source in sorted(
                    full_results,
                    key=lambda item: (
                        0 if item.passed_quality_floor else 1,
                        -item.selected_pass_rate,
                        item.estimated_production_cost_per_call_usd,
                    ),
                ):
                    package_key = (
                        source.selected_prompt,
                        tuple(source.few_shot_example_ids),
                    )
                    if (
                        package_key in seen_packages
                        or (
                            source.selected_prompt == prompt_text
                            and not source.few_shot_example_ids
                        )
                    ):
                        continue
                    seen_packages.add(package_key)
                    rewrite_packages.append(
                        (
                            source.selected_prompt,
                            list(source.few_shot_example_ids),
                            source.model,
                        )
                    )
                    if len(rewrite_packages) >= 2:
                        break

                propagation_candidates = [
                    str(item["model"]) for item in screening_results
                    if item["model"] not in attempted_for_full
                ]
                validation_passing_packages = [
                    item for item in full_results
                    if item.selected_pass_rate >= quality_threshold
                ]
                if validation_passing_packages:
                    validation_passing_cost = min(
                        item.estimated_production_cost_per_call_usd
                        for item in validation_passing_packages
                    )
                    screen_cost_by_model = {
                        str(item["model"]): float(
                            item["estimated_production_cost_per_call_usd"]
                        )
                        for item in screening_results
                    }
                    propagation_candidates = [
                        model for model in propagation_candidates
                        if screen_cost_by_model.get(model, float("inf"))
                        < validation_passing_cost
                    ]
                propagation_by_model: dict[str, dict[str, Any]] = {}

                def screen_propagated(model: str) -> dict[str, Any]:
                    emit_progress({
                        "event": "prompt_propagation_started",
                        "model": model,
                        "candidate_prompt_packages": len(rewrite_packages),
                        "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
                    })
                    model_budget = _BudgetScope(budget)
                    best: dict[str, Any] | None = None
                    for propagated_prompt, few_shot_ids, source_model in rewrite_packages:
                        with request_timeout_context(30.0):
                            cases = self._run_cases(
                                propagated_prompt,
                                "propagated-screen",
                                dev,
                                "dev",
                                model,
                                evaluator_model,
                                model_budget,
                                train,
                                few_shot_ids,
                                evaluator=evaluator_policy,
                                max_parallel_scenarios=int(max_parallel_scenarios),
                                target_max_tokens=(int(target_max_tokens) if target_max_tokens is not None else None),
                                request_options=target_request_options,
                            )
                        candidate = {
                            "model": model,
                            "validation_pass_rate": round(_scenario_pass_rate(cases), 6),
                            "cases": cases,
                            "prompt": propagated_prompt,
                            "few_shot_example_ids": list(few_shot_ids),
                            "source_model": source_model,
                        }
                        if best is None or (
                            float(candidate["validation_pass_rate"]),
                            -len(propagated_prompt),
                        ) > (
                            float(best["validation_pass_rate"]),
                            -len(str(best["prompt"])),
                        ):
                            best = candidate
                        if candidate["validation_pass_rate"] >= quality_threshold:
                            break
                    if best is None:
                        raise ProviderError("No successful prompt package was available to propagate.")
                    best["propagation_spend_usd"] = round(model_budget.spent_usd, 10)
                    emit_progress({
                        "event": "prompt_propagation_completed",
                        "model": model,
                        "source_model": best["source_model"],
                        "validation_pass_rate": best["validation_pass_rate"],
                        "few_shot_example_ids": best["few_shot_example_ids"],
                        "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
                    })
                    return best

                if optimize_prompt and rewrite_packages and propagation_candidates:
                    with ThreadPoolExecutor(
                        max_workers=min(int(max_parallel_models), len(propagation_candidates))
                    ) as pool:
                        futures = {
                            _submit_with_context(pool, screen_propagated, model): model
                            for model in propagation_candidates
                        }
                        for future in as_completed(futures):
                            model = futures[future]
                            try:
                                propagation_by_model[model] = future.result()
                            except BudgetExceeded as error:
                                incomplete_models.append({
                                    "model": model,
                                    "reason": str(error),
                                    "stage": "prompt_propagation",
                                })
                            except ProviderError as error:
                                unavailable_models.append({
                                    "model": model,
                                    "reason": str(error),
                                    "stage": "prompt_propagation",
                                })

                propagated_for_full = [
                    model for model in propagation_candidates
                    if model in propagation_by_model
                    and float(propagation_by_model[model]["validation_pass_rate"])
                    >= quality_threshold
                ]
                for model in propagated_for_full:
                    package = propagation_by_model[model]
                    seed_prompt_by_model[model] = str(package["prompt"])
                    seed_few_shot_ids_by_model[model] = list(
                        package["few_shot_example_ids"]
                    )
                    screening_cases_by_model[model] = list(package["cases"])
                    prompt_origin_by_model[model] = (
                        f"propagated_from:{package['source_model']}"
                    )
                if propagated_for_full:
                    attempted_for_full.extend(propagated_for_full)
                    results.extend(run_batch(propagated_for_full))

                screened_out = [
                    str(item["model"]) for item in screening_results
                    if item["model"] not in attempted_for_full
                ]
                pruned_models.extend(screened_out)
                attempted_set = set(attempted_for_full)
                for item in screening_results:
                    propagation = propagation_by_model.get(str(item["model"]))
                    if propagation:
                        item["propagated_prompt_validation_pass_rate"] = propagation[
                            "validation_pass_rate"
                        ]
                        item["propagated_from_model"] = propagation["source_model"]
                        item["propagation_spend_usd"] = propagation[
                            "propagation_spend_usd"
                        ]
                    item["status"] = (
                        "FULL_PROPAGATED"
                        if item["model"] in prompt_origin_by_model
                        else "FULL"
                        if item["model"] in attempted_set
                        else "PRUNED"
                    )
                for model in screened_out:
                    emit_progress({
                        "event": "model_pruned", "model": model,
                        "reason": "Validation screening did not earn a full prompt-search and final-test run.",
                    })
            else:
                broad_wave_size = max(
                    1, min(len(ordered_broad) - 1 or 1, int(max_parallel_models))
                )
                stop_after_first_pass = objective in {
                    "cheapest_passing", "cheapest_at_accuracy", "constrained",
                    "match_baseline_at_lowest_cost",
                }
                for wave_start in range(0, len(ordered_broad), broad_wave_size):
                    wave = ordered_broad[wave_start : wave_start + broad_wave_size]
                    results.extend(run_batch(wave))
                    if stop_after_first_pass and any(item.passed_quality_floor for item in results):
                        pruned_models.extend(ordered_broad[wave_start + broad_wave_size :])
                        break
            # The final test qualifies a route; it must not choose which reasoning
            # rung Evalt tries next. Adaptive search decisions use validation only.
            validation_passing_broad = [
                item for item in results
                if item.selected_pass_rate >= quality_threshold
            ]
            if validation_passing_broad:
                cheapest_pass_cost = min(
                    item.estimated_production_cost_per_call_usd
                    for item in validation_passing_broad
                )
                hone_base_models = {
                    item.model.split("#reasoning=", 1)[0]
                    for item in results
                    if item.selected_pass_rate >= quality_threshold
                    or item.estimated_production_cost_per_call_usd <= cheapest_pass_cost * 1.25 + 1e-12
                }
            else:
                best_quality = max(
                    (item.selected_pass_rate for item in results), default=0.0
                )
                near_floor = max(0.0, min(best_quality - 0.20, quality_threshold - 0.20))
                hone_base_models = {
                    item.model.split("#reasoning=", 1)[0]
                    for item in results
                    if item.selected_pass_rate >= near_floor
                }
            hone_models = model_list[len(broad_models):]
            broad_result_by_base = {
                item.model.split("#reasoning=", 1)[0]: item
                for item in results
                if item.model in broad_models
            }
            effort_rank = {
                "none": 0,
                "minimal": 1,
                "low": 2,
                "medium": 3,
                "high": 4,
                "xhigh": 5,
                "max": 6,
            }

            def split_effort(configuration: str) -> tuple[str, str]:
                if "#reasoning=" not in configuration:
                    return configuration, "none"
                base_model, effort = configuration.rsplit("#reasoning=", 1)
                return base_model, effort

            def reasoning_hone_can_improve(model: str) -> bool:
                base_model, candidate_effort = split_effort(model)
                broad_result = broad_result_by_base.get(base_model)
                if broad_result is None or base_model not in hone_base_models:
                    return False
                _broad_model, broad_effort = split_effort(broad_result.model)
                candidate_rank = effort_rank[candidate_effort]
                broad_rank = effort_rank[broad_effort]
                if broad_result.selected_pass_rate >= quality_threshold:
                    # A stronger rung is dominated once the same base model clears
                    # validation in a lowest-cost search. Still measure every
                    # cheaper lower rung so the route is not promoted merely
                    # because the first passing effort happened to be expensive.
                    return candidate_rank < broad_rank
                # If the broad effort missed, only extra reasoning can plausibly
                # repair accuracy.  Do not spend on a still-weaker configuration.
                return candidate_rank > broad_rank

            ordinary_hone = [
                model for model in hone_models
                if split_effort(model)[1] not in {"xhigh", "max"}
                and reasoning_hone_can_improve(model)
            ]
            extreme_hone = [
                model for model in hone_models
                if split_effort(model)[1] in {"xhigh", "max"}
                and reasoning_hone_can_improve(model)
            ]
            pruned_models.extend(
                model for model in hone_models
                if model not in ordinary_hone and model not in extreme_hone
            )
            # Reasoning effort should be the lever under test, not a hidden change
            # to the prompt package. Seed each rung from the strongest validation
            # package already measured for that model. It may still rewrite from
            # training/validation, and every completed variant faces the untouched
            # final test.
            def seed_reasoning_hone(
                configurations: list[str], available: list[ModelResult]
            ) -> None:
                for configuration in configurations:
                    base_model, _effort = split_effort(configuration)
                    same_base = [
                        item for item in available
                        if split_effort(item.model)[0] == base_model
                    ]
                    candidates = same_base or available
                    if not candidates:
                        continue
                    source = min(
                        candidates,
                        key=lambda item: (
                            -item.selected_pass_rate,
                            item.estimated_production_cost_per_call_usd,
                            -effort_rank[split_effort(item.model)[1]],
                            item.model,
                        ),
                    )
                    seed_prompt_by_model[configuration] = source.selected_prompt
                    seed_few_shot_ids_by_model[configuration] = list(
                        source.few_shot_example_ids
                    )
                    prompt_origin_by_model[configuration] = (
                        f"reasoning_hone_from:{source.model}"
                    )
                    # Hold the learned prompt package fixed so reasoning effort is
                    # the only lever under test. This also prevents optional hone
                    # lanes from repeating a full prompt-search tournament.
                    fixed_prompt_models.add(configuration)

            seed_reasoning_hone(ordinary_hone, results)
            results.extend(run_batch(ordinary_hone))

            # Extreme reasoning is a sequential evidence ladder. High must land
            # within one validation case of the requested quality gate and inside
            # the production latency ceiling before xhigh earns spend. Max earns
            # spend only when xhigh stays close, does not regress, and also remains
            # inside the ceiling. Final-test performance is never consulted here.
            validation_step = 1.0 / max(1, len(dev))
            close_floor = max(0.0, quality_threshold - validation_step - 1e-12)

            def completed_effort(base_model: str, effort: str) -> ModelResult | None:
                matches = [
                    item for item in results
                    if split_effort(item.model) == (base_model, effort)
                ]
                return min(
                    matches,
                    key=lambda item: (
                        -item.selected_pass_rate,
                        item.estimated_production_cost_per_call_usd,
                    ),
                ) if matches else None

            def inside_latency(item: ModelResult | None) -> bool:
                return bool(
                    item is not None
                    and (
                        max_p90_latency_seconds is None
                        or item.target_latency_p90_ms
                        <= max_p90_latency_seconds * 1000
                    )
                )

            xhigh_candidates: list[str] = []
            for configuration in extreme_hone:
                base_model, effort = split_effort(configuration)
                if effort != "xhigh":
                    continue
                high_result = completed_effort(base_model, "high")
                earned = bool(
                    high_result is not None
                    and high_result.selected_pass_rate < quality_threshold
                    and high_result.selected_pass_rate >= close_floor
                    and inside_latency(high_result)
                )
                emit_progress({
                    "event": (
                        "reasoning_escalation_started"
                        if earned else "reasoning_escalation_skipped"
                    ),
                    "model": base_model,
                    "from_effort": "high",
                    "to_effort": "xhigh",
                    "validation_pass_rate": (
                        high_result.selected_pass_rate if high_result else None
                    ),
                    "target_latency_p90_ms": (
                        high_result.target_latency_p90_ms if high_result else None
                    ),
                    "quality_threshold": quality_threshold,
                    "close_floor": close_floor,
                    "reason": (
                        "high is within one validation case and the latency ceiling"
                        if earned else
                        "high was absent, not close enough, already passed, or too slow"
                    ),
                })
                if earned:
                    xhigh_candidates.append(configuration)
                else:
                    pruned_models.append(configuration)
            seed_reasoning_hone(xhigh_candidates, results)
            results.extend(run_batch(xhigh_candidates))

            max_candidates: list[str] = []
            for configuration in extreme_hone:
                base_model, effort = split_effort(configuration)
                if effort != "max":
                    continue
                high_result = completed_effort(base_model, "high")
                xhigh_result = completed_effort(base_model, "xhigh")
                earned = bool(
                    high_result is not None
                    and xhigh_result is not None
                    and xhigh_result.selected_pass_rate < quality_threshold
                    and xhigh_result.selected_pass_rate >= close_floor
                    and xhigh_result.selected_pass_rate
                    >= high_result.selected_pass_rate
                    and inside_latency(xhigh_result)
                )
                emit_progress({
                    "event": (
                        "reasoning_escalation_started"
                        if earned else "reasoning_escalation_skipped"
                    ),
                    "model": base_model,
                    "from_effort": "xhigh",
                    "to_effort": "max",
                    "validation_pass_rate": (
                        xhigh_result.selected_pass_rate if xhigh_result else None
                    ),
                    "target_latency_p90_ms": (
                        xhigh_result.target_latency_p90_ms if xhigh_result else None
                    ),
                    "quality_threshold": quality_threshold,
                    "close_floor": close_floor,
                    "reason": (
                        "xhigh stayed close without regressing and met the latency ceiling"
                        if earned else
                        "xhigh was absent, regressed, already passed, not close, or too slow"
                    ),
                })
                if earned:
                    max_candidates.append(configuration)
                else:
                    pruned_models.append(configuration)
            seed_reasoning_hone(max_candidates, results)
            results.extend(run_batch(max_candidates))
        else:
            results = run_batch(model_list)
        if not results:
            if incomplete_models:
                raise BudgetExceeded(incomplete_models[0]["reason"])
            failed = "; ".join(f"{item['model']}: {item['reason']}" for item in unavailable_models)
            raise ProviderError(
                f"No selected target model completed under the required provider policy. {failed}"
            )
        passing = [item for item in results if item.passed_quality_floor]
        within_cost = [
            item for item in results
            if max_cost_per_run_usd is None
            or item.estimated_production_cost_per_call_usd <= max_cost_per_run_usd + 1e-12
        ]
        within_latency = [
            item for item in results
            if max_p90_latency_seconds is None
            or item.target_latency_p90_ms <= max_p90_latency_seconds * 1000
        ]
        for item in results:
            item.passed_latency_ceiling = item in within_latency
        eligible_results = within_latency if max_p90_latency_seconds is not None and within_latency else results
        constrained = [
            item for item in passing
            if item in within_cost and item in within_latency
        ]
        def effective_cost(item: ModelResult) -> float:
            return (
                item.estimated_cost_per_successful_call_usd
                + latency_value_usd_per_second * item.target_latency_p90_ms / 1000
            )
        baseline = next((item for item in results if item.model == incumbent_model), results[0])
        required_baseline_quality = max(0.0, baseline.baseline_holdout_pass_rate - allowed_accuracy_regression)
        matched_baseline = [
            item for item in within_cost
            if item in within_latency and item.holdout_pass_rate >= required_baseline_quality
        ]
        if objective == "match_baseline_at_lowest_cost" and matched_baseline:
            winner = min(
                matched_baseline,
                key=lambda item: (
                    item.estimated_production_cost_per_call_usd,
                    -item.holdout_pass_rate,
                    item.model,
                ),
            )
        elif objective in {"cheapest_passing", "cheapest_at_accuracy", "constrained"} and constrained:
            winner = min(
                constrained,
                key=lambda item: (
                    effective_cost(item),
                    -item.holdout_pass_rate,
                    item.model,
                ),
            )
        elif objective == "best_within_cost" and [item for item in within_cost if item in within_latency]:
            winner = max(
                [item for item in within_cost if item in within_latency],
                key=lambda item: (
                    item.holdout_pass_rate,
                    -effective_cost(item),
                ),
            )
        else:
            winner = max(
                eligible_results,
                key=lambda item: (
                    item.holdout_pass_rate,
                    -effective_cost(item),
                ),
            )
        warnings = []
        if len(holdout) < 5:
            warnings.append(
                f"Only {len(holdout)} distinct final-test scenario(s): this is exploratory and not a reliability claim."
            )
        elif winner.passed_quality_floor and not winner.target_accuracy_statistically_supported:
            if winner.final_test_accuracy_lower_bound is None:
                warnings.append(
                    "The observed final-test rate cleared the route gate, but weighted or non-binomial evidence cannot establish a confidence bound; the route remains provisional."
                )
            else:
                warnings.append(
                    f"The observed final-test rate cleared the route gate, but the one-sided {winner.final_test_confidence_level:.0%} exact lower bound is {winner.final_test_accuracy_lower_bound:.1%}, below the {quality_threshold:.1%} target; the route remains provisional."
                )
        if not passing:
            warnings.append("No prompt/model pair cleared the requested quality threshold.")
        if max_cost_per_run_usd is not None and not within_cost:
            warnings.append("No tested configuration fit the requested production cost ceiling.")
        if max_p90_latency_seconds is not None and not [
            item for item in passing if item in within_latency
        ]:
            warnings.append("No passing configuration fit the requested measured p90 latency ceiling.")
        if objective == "constrained" and not constrained:
            warnings.append("No tested configuration satisfied both production cost and accuracy constraints.")
        if objective == "match_baseline_at_lowest_cost" and not matched_baseline:
            warnings.append("No tested configuration matched the incumbent's held-out quality within the allowed regression margin.")
        if unavailable_models:
            warnings.append(
                f"{len(unavailable_models)} selected model(s) were unavailable under the required provider policy."
            )
        if incomplete_models or skipped_budget_models:
            warnings.append("Coverage is partial: the winner is best only among fully completed targets.")
        if pruned_models:
            warnings.append(
                f"Adaptive search pruned {len(pruned_models)} reasoning configuration(s) outside the observed task-capability band."
            )
        if omitted_configurations:
            warnings.append(
                f"Preflight omitted {len(omitted_configurations)} known-incompatible configuration(s) before any target call."
            )
        frontier, diminishing = _quality_frontier(
            results, float(minimum_meaningful_quality_gain)
        )
        suite_payload = {
            "schema": "evalt-regression-suite-v1",
            "starting_prompt": prompt_text,
            "winning_prompt": winner.selected_prompt,
            "winning_few_shot_example_ids": winner.few_shot_example_ids,
            "winning_few_shot_provenance": winner.few_shot_provenance,
            "examples": [_safe_example_payload(item) for item in example_list],
            "selected_model": winner.model,
            "known_models": model_list,
            "optimizer_model": optimizer_model,
            "evaluator_model": evaluator_model,
            "evaluator": evaluator_policy,
            "difficulty_thresholds": difficulty_floor_policy,
            "quality_threshold": quality_threshold,
            "incumbent_model": incumbent_model,
            "incumbent_baseline_holdout_pass_rate": (
                baseline.baseline_holdout_pass_rate if incumbent_model else None
            ),
            "allowed_accuracy_regression": allowed_accuracy_regression,
            "max_cost_per_run_usd": max_cost_per_run_usd,
            "max_p90_latency_seconds": max_p90_latency_seconds,
            "latency_value_usd_per_second": latency_value_usd_per_second,
            "representative_input_chars": representative_input_chars,
            "representative_output_tokens": representative_output_tokens,
            "target_max_tokens": (
                int(target_max_tokens) if target_max_tokens is not None else None
            ),
            "request_options": target_request_options,
            "request_options_sha256": request_options_fingerprint(target_request_options),
            "holdout_repeats": int(holdout_repeats),
            "holdout_unique_scenarios": len(holdout),
            "final_test_evidence_status": winner.final_test_evidence_status,
            "final_test_confidence_level": winner.final_test_confidence_level,
            "final_test_accuracy_lower_bound": winner.final_test_accuracy_lower_bound,
            "target_accuracy_statistically_supported": winner.target_accuracy_statistically_supported,
            "minimum_zero_failure_scenarios": winner.minimum_zero_failure_scenarios,
            "minimum_meaningful_quality_gain": minimum_meaningful_quality_gain,
            "optimize_prompt": bool(optimize_prompt),
            "watch": {
                "enabled": False,
                "max_recheck_cost_usd": 0,
                "notice": "Enable explicitly before any automatic provider call.",
            },
        }
        suite_payload["suite_hash"] = hashlib.sha256(
            json.dumps(suite_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        comparison_integrity = {
            "single_frozen_run": True,
            "suite_hash": suite_payload["suite_hash"],
            "distinct_final_test_scenarios": len(holdout),
            "executions_per_final_test_scenario": int(holdout_repeats),
            "evaluator": evaluator_policy,
            "difficulty_thresholds": difficulty_floor_policy,
            "selection_protocol": {
                "prompt_modification_enabled": bool(optimize_prompt),
                "prompt_optimizer_inputs": (
                    "training split only" if optimize_prompt else "disabled"
                ),
                "few_shot_sources": (
                    "customer-approved training examples only"
                    if optimize_prompt and allow_few_shot else "disabled"
                ),
                "prompt_package_selection": (
                    "validation split" if optimize_prompt else "fixed supplied prompt"
                ),
                "promotion_gate": "untouched final-test split",
                "final_test_used_for_rewrite_or_selection": False,
            },
            "configurations": [
                {
                    "configuration": configuration,
                    "model": configuration.split("#reasoning=", 1)[0],
                    "requested_reasoning_effort": (
                        configuration.rsplit("#reasoning=", 1)[1]
                        if "#reasoning=" in configuration else "none"
                    ),
                }
                for configuration in model_list
            ],
            "claim_scope": (
                "Task-specific prompt/model/reasoning configuration result only; "
                "not a general model-intelligence ranking."
            ),
        }
        budget_limited_configurations = list(dict.fromkeys([
            *[
                str(item["model"])
                for item in incomplete_models
                if "budget" in str(item.get("reason", "")).lower()
                or "cap" in str(item.get("reason", "")).lower()
            ],
            *[str(model) for model in skipped_budget_models],
        ]))
        continuation_recommendation = None
        if budget_limited_configurations and max_optimization_cost_usd < 100:
            suggested_budget = min(
                100.0,
                max(
                    float(max_optimization_cost_usd) + 0.25,
                    float(max_optimization_cost_usd) * 1.5,
                ),
            )
            suggested_budget = round(suggested_budget + 1e-12, 2)
            qualified = winner.passed_quality_floor and winner.passed_latency_ceiling
            continuation_recommendation = {
                "recommended": True,
                "reason": (
                    "COMPLETE_COST_FRONTIER"
                    if qualified
                    else "FIND_QUALIFIED_ROUTE"
                ),
                "unfinished_configurations": budget_limited_configurations,
                "current_test_budget_usd": round(float(max_optimization_cost_usd), 2),
                "suggested_next_test_budget_usd": suggested_budget,
                "suggested_additional_budget_usd": round(
                    suggested_budget - float(max_optimization_cost_usd), 2
                ),
                "basis": (
                    "Bounded 1.5x-cap heuristic (at least $0.25 more); review the "
                    "unfinished configurations before explicitly approving a rerun."
                ),
                "automatic_spend": False,
            }
            emit_progress({
                "event": "continuation_recommended",
                **continuation_recommendation,
                "elapsed_seconds": round(time.monotonic() - optimization_started, 3),
            })
        return OptimizationResult(
            objective=objective,
            quality_threshold=quality_threshold,
            exploratory=len(holdout) < 5,
            winner=winner,
            models=results,
            total_provider_spend_usd=round(budget.spent_usd, 10),
            warnings=warnings,
            quality_frontier=frontier,
            diminishing_returns=diminishing,
            regression_suite=suite_payload,
            elapsed_seconds=round(time.monotonic() - optimization_started, 3),
            comparison_integrity=comparison_integrity,
            omitted_configurations=omitted_configurations,
            unavailable_models=unavailable_models,
            incomplete_models=incomplete_models,
            skipped_budget_models=skipped_budget_models,
            pruned_models=pruned_models,
            screening_results=screening_results,
            winner_scope="Best among the completed adaptive search band" if pruned_models else "Best among fully completed eligible targets only" if incomplete_models or skipped_budget_models or unavailable_models else "Best among every capability-eligible requested target" if omitted_configurations else "Best among every requested target",
            quality_gate_status=(
                "QUALIFIED_ROUTE_SELECTED"
                if winner.passed_quality_floor and winner.passed_latency_ceiling
                else "NO_CONFIGURATION_PASSED"
            ),
            continuation_recommendation=continuation_recommendation,
        )

    @staticmethod
    def _validate(
        prompt: str,
        examples: list[Example],
        models: list[str],
        quality_threshold: float,
        max_cost: float,
        rounds: int,
        minimum_gain: float,
    ) -> None:
        if len(prompt) < 8:
            raise ValueError("The current prompt must contain at least eight characters.")
        if len(examples) < 3:
            raise ValueError("At least three approved examples are required to optimize.")
        if len(models) < 1 or len(models) > 25:
            raise ValueError("Choose between one and 25 target models per bounded run.")
        if not 0 < quality_threshold <= 1:
            raise ValueError("quality_threshold must be greater than zero and at most one.")
        if max_cost <= 0:
            raise ValueError("max_optimization_cost_usd must be positive.")
        if not 1 <= int(rounds) <= 8:
            raise ValueError("rounds must be between one and eight.")
        if not 0 <= float(minimum_gain) <= 1:
            raise ValueError("minimum_meaningful_quality_gain must be between zero and one.")
        if any(any(not turn.input or not turn.approved_output for turn in item.conversation()) for item in examples):
            raise ValueError("Every scenario turn requires an input and approved output.")
        if any(not math.isfinite(item.weight) or item.weight <= 0 for item in examples):
            raise ValueError("Every scenario weight must be positive and finite.")
        grouped = [item for item in examples if item.group]
        if grouped:
            if len(grouped) != len(examples):
                raise ValueError("Either every scenario must declare a group or none may declare one.")
            group_counts: dict[str, int] = {}
            for item in examples:
                group_counts[item.group] = group_counts.get(item.group, 0) + 1
            undersized = sorted(group for group, count in group_counts.items() if count < 5)
            if undersized:
                raise ValueError(
                    "Every stratified group needs at least five scenarios: "
                    + ", ".join(undersized)
                )

    def _evaluate_model(
        self,
        prompt: str,
        train: list[Example],
        dev: list[Example],
        holdout: list[Example],
        model: str,
        optimizer_model: str,
        evaluator_model: str,
        threshold: float,
        budget: _Budget,
        rounds: int,
        allow_few_shot: bool,
        max_few_shot_examples: int,
        representative_input_chars: int | None,
        representative_output_tokens: int | None,
        holdout_repeats: int,
        evaluator: dict[str, Any],
        difficulty_thresholds: dict[str, float],
        max_parallel_scenarios: int,
        baseline_dev_cases: list[CaseResult] | None = None,
        seed_prompt: str | None = None,
        seed_few_shot_ids: list[str] | None = None,
        prompt_origin: str = "starting_prompt",
        optimize_prompt: bool = True,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        target_max_tokens: int | None = None,
        request_options: Mapping[str, Any] | None = None,
    ) -> ModelResult:
        started_spend = budget.spent_usd
        initial_prompt = seed_prompt or prompt
        initial_few_shot_ids = list(seed_few_shot_ids or [])
        # Validation remains the selection gate, but a small validation slice can
        # reach 100% by chance while the starting prompt still fails much of the
        # approved training evidence. Always measure both disclosed splits before
        # deciding that prompt learning is unnecessary. The frozen final test is
        # still never used to select or rewrite a prompt.
        def run_split(
            split_examples: list[Example], split: str, candidate_prompt: str,
            prompt_kind: str, few_shot_ids: list[str],
        ) -> list[CaseResult]:
            return self._run_cases(
                candidate_prompt, prompt_kind, split_examples, split, model,
                evaluator_model, budget, train, few_shot_ids,
                evaluator=evaluator,
                max_parallel_scenarios=max_parallel_scenarios,
                target_max_tokens=target_max_tokens,
                request_options=request_options,
            )

        if (
            baseline_dev_cases is None
            and train and dev
            and int(max_parallel_scenarios) > 1
        ):
            with ThreadPoolExecutor(max_workers=2) as pool:
                train_future = _submit_with_context(
                    pool,
                    run_split, train, "train", initial_prompt, "baseline",
                    initial_few_shot_ids,
                )
                dev_future = _submit_with_context(
                    pool,
                    run_split, dev, "dev", initial_prompt, "baseline",
                    initial_few_shot_ids,
                )
                baseline_train = train_future.result()
                baseline_dev = dev_future.result()
        else:
            baseline_dev = (
                list(baseline_dev_cases)
                if baseline_dev_cases is not None
                else run_split(
                    dev, "dev", initial_prompt, "baseline",
                    initial_few_shot_ids,
                )
            )
            baseline_train = run_split(
                train, "train", initial_prompt, "baseline",
                initial_few_shot_ids,
            )
        baseline_dev_rate = _pass_rate(baseline_dev)
        baseline_train_rate = _pass_rate(baseline_train)
        selected_prompt = initial_prompt
        selected_few_shot_ids: list[str] = initial_few_shot_ids
        selected_train = baseline_train or baseline_dev
        selected_train_rate = baseline_train_rate
        selected_dev = baseline_dev
        selected_rate = baseline_dev_rate
        candidate_cases: list[CaseResult] = []
        prompt_candidates_tested = 1
        if progress_callback is not None:
            progress_callback({
                "event": "prompt_candidate_completed",
                "model": model,
                "candidate": 0,
                "kind": "starting_prompt",
                "prompt_hash": hashlib.sha256(initial_prompt.encode("utf-8")).hexdigest()[:16],
                "few_shot_examples": len(initial_few_shot_ids),
                "training_pass_rate": round(baseline_train_rate, 6),
                "validation_pass_rate": round(baseline_dev_rate, 6),
                "selected": True,
            })
        should_search_prompt = bool(optimize_prompt)
        for round_number in (
            range(1, int(rounds) + 1) if should_search_prompt else ()
        ):
            try:
                revised_prompt, revised_few_shot_ids = self._propose_prompt(
                    selected_prompt,
                    model,
                    train,
                    selected_train,
                    optimizer_model,
                    budget,
                    allow_few_shot,
                    max_few_shot_examples,
                )
                # Training evidence breaks a validation tie; the candidate must never
                # regress on validation. This avoids silently skipping a useful rewrite
                # after a lucky perfect score on a small validation slice.
                if train and dev and int(max_parallel_scenarios) > 1:
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        train_future = _submit_with_context(
                            pool,
                            run_split, train, "train", revised_prompt,
                            f"candidate-{round_number}", revised_few_shot_ids,
                        )
                        dev_future = _submit_with_context(
                            pool,
                            run_split, dev, "dev", revised_prompt,
                            f"candidate-{round_number}", revised_few_shot_ids,
                        )
                        revised_train = train_future.result()
                        revised_dev = dev_future.result()
                else:
                    revised_train = run_split(
                        train, "train", revised_prompt,
                        f"candidate-{round_number}", revised_few_shot_ids,
                    )
                    revised_dev = run_split(
                        dev, "dev", revised_prompt,
                        f"candidate-{round_number}", revised_few_shot_ids,
                    )
                candidate_cases += revised_train
                candidate_cases += revised_dev
            except BudgetExceeded as error:
                # Prompt search is optional. Preserve the fully measured starting
                # package and continue to final confirmation when a rewrite cannot
                # fit under the shared tournament cap.
                if progress_callback is not None:
                    progress_callback({
                        "event": "prompt_candidate_skipped_budget",
                        "model": model,
                        "candidate": round_number,
                        "reason": str(error),
                        "validation_pass_rate": round(selected_rate, 6),
                    })
                break
            revised_rate = _pass_rate(revised_dev)
            revised_train_rate = _pass_rate(revised_train)
            prompt_candidates_tested += 1
            selected_candidate = False
            if (
                revised_rate > selected_rate
                or (
                    revised_rate == selected_rate
                    and revised_train_rate > selected_train_rate
                )
            ):
                selected_prompt = revised_prompt
                selected_few_shot_ids = revised_few_shot_ids
                selected_train = revised_train
                selected_train_rate = revised_train_rate
                selected_dev = revised_dev
                selected_rate = revised_rate
                selected_candidate = True
            if progress_callback is not None:
                progress_callback({
                    "event": "prompt_candidate_completed",
                    "model": model,
                    "candidate": round_number,
                    "kind": "rewrite",
                    "prompt_hash": hashlib.sha256(revised_prompt.encode("utf-8")).hexdigest()[:16],
                    "few_shot_examples": len(revised_few_shot_ids),
                    "training_pass_rate": round(revised_train_rate, 6),
                    "validation_pass_rate": round(revised_rate, 6),
                    "selected": selected_candidate,
                })
            if selected_rate >= 1 and selected_train_rate >= 1:
                break
        selected_kind = (
            "baseline"
            if selected_prompt == prompt and not selected_few_shot_ids
            else "candidate"
        )
        if selected_rate >= threshold:
            if progress_callback is not None:
                progress_callback({
                    "event": "final_confirmation_started",
                    "model": model,
                    "unique_scenarios": len(holdout),
                    "executions": len(holdout) * int(holdout_repeats),
                    "validation_pass_rate": round(selected_rate, 6),
                })
            baseline_holdout = self._run_cases(
                prompt, "baseline", holdout, "holdout", model, evaluator_model, budget,
                repeats=holdout_repeats,
                evaluator=evaluator,
                max_parallel_scenarios=max_parallel_scenarios,
                target_max_tokens=target_max_tokens, request_options=request_options,
            )
            if selected_kind == "baseline":
                selected_holdout = baseline_holdout
            else:
                selected_holdout = self._run_cases(
                    selected_prompt, "candidate", holdout, "holdout", model, evaluator_model, budget
                    , train, selected_few_shot_ids, holdout_repeats, evaluator=evaluator,
                    max_parallel_scenarios=max_parallel_scenarios,
                    target_max_tokens=target_max_tokens, request_options=request_options,
                )
        else:
            baseline_holdout = []
            selected_holdout = []
            if progress_callback is not None:
                progress_callback({
                    "event": "final_confirmation_skipped",
                    "model": model,
                    "unique_scenarios": len(holdout),
                    "executions": len(holdout) * int(holdout_repeats),
                    "validation_pass_rate": round(selected_rate, 6),
                    "quality_threshold": threshold,
                    "reason": "The frozen package did not clear validation, so final-test spend could not qualify it.",
                })
        holdout_execution_rate = _pass_rate(selected_holdout)
        holdout_rate = _scenario_pass_rate(selected_holdout)
        final_test_evidence = _final_test_evidence(
            selected_holdout, target_accuracy=threshold
        )
        holdout_by_difficulty = _scenario_pass_rates_by_difficulty(selected_holdout)
        passed_difficulty_floors = all(
            difficulty in holdout_by_difficulty
            and holdout_by_difficulty[difficulty] >= floor
            for difficulty, floor in difficulty_thresholds.items()
        )
        observed_inputs = [
            sum(_text_character_count(turn.input) for turn in item.conversation())
            for item in train + dev + holdout
        ]
        observed_outputs = [sum(len(turn.approved_output) for turn in item.conversation()) for item in train + dev + holdout]
        typical_input = representative_input_chars or _percentile(observed_inputs, 0.90)
        typical_output_tokens = representative_output_tokens or max(32, int(_percentile(observed_outputs, 0.90) / 3) + 1)
        representative_content: Any = "x" * int(typical_input)
        image_inputs = [
            turn.input for item in train + dev + holdout for turn in item.conversation()
            if "image" in required_input_modalities(turn.input)
        ]
        if image_inputs:
            representative_content = max(
                image_inputs,
                key=lambda value: (_image_count(value), _text_character_count(value)),
            )
        production_messages = [{"role": "system", "content": selected_prompt}] + _few_shot_messages(train, selected_few_shot_ids) + [{"role": "user", "content": representative_content}]
        # A completion allowance is a safety ceiling, not an expected bill. Price
        # the promoted route from the measured 90th-percentile successful final-test
        # call so generous first-request headroom does not make a tiny JSON response
        # look as if it always consumes the full context window.
        measured_target_costs = sorted(
            result.target_cost_usd for result in selected_holdout
            if result.target_cost_usd > 0
        )
        if measured_target_costs:
            production_cost = measured_target_costs[
                max(0, min(len(measured_target_costs) - 1, math.ceil(len(measured_target_costs) * 0.90) - 1))
            ]
        else:
            production_cost = self.transport.estimate_cost(model, production_messages, max_tokens=int(typical_output_tokens))
        cost_per_success = production_cost / max(holdout_rate, 0.01)
        measured_target_latencies = sorted(
            result.target_latency_ms for result in selected_holdout
            if result.target_latency_ms > 0
        )
        latency_p50 = _percentile(measured_target_latencies, 0.50) if measured_target_latencies else 0
        latency_p90 = _percentile(measured_target_latencies, 0.90) if measured_target_latencies else 0
        all_cases = baseline_train + baseline_dev + candidate_cases + baseline_holdout
        if selected_holdout is not baseline_holdout:
            all_cases += selected_holdout
        selected_prompt_origin = (
            prompt_origin
            if selected_prompt == initial_prompt
            and selected_few_shot_ids == initial_few_shot_ids
            else f"optimized_for:{model}"
        )
        return ModelResult(
            model=model,
            selected_prompt=selected_prompt,
            baseline_pass_rate=round(baseline_train_rate, 6),
            selected_pass_rate=round(selected_rate, 6),
            holdout_pass_rate=round(holdout_rate, 6),
            baseline_holdout_pass_rate=round(_scenario_pass_rate(baseline_holdout), 6),
            estimated_production_cost_per_call_usd=round(production_cost, 10),
            estimated_cost_per_successful_call_usd=round(cost_per_success, 10),
            optimization_spend_usd=round(budget.spent_usd - started_spend, 10),
            passed_quality_floor=holdout_rate >= threshold and passed_difficulty_floors,
            target_latency_p50_ms=round(latency_p50),
            target_latency_p90_ms=round(latency_p90),
            holdout_unique_scenarios=len(holdout) if selected_holdout else 0,
            holdout_executions=len(selected_holdout),
            holdout_execution_pass_rate=round(holdout_execution_rate, 6),
            baseline_holdout_execution_pass_rate=round(_pass_rate(baseline_holdout), 6),
            holdout_pass_rates_by_difficulty=holdout_by_difficulty,
            passed_difficulty_floors=passed_difficulty_floors,
            prompt_origin=selected_prompt_origin,
            prompt_candidates_tested=prompt_candidates_tested,
            prompt_rewrites_tested=max(0, prompt_candidates_tested - 1),
            selected_prompt_changed=(
                selected_prompt != prompt or bool(selected_few_shot_ids)
            ),
            few_shot_example_ids=selected_few_shot_ids,
            few_shot_provenance=[
                {
                    "example_id": example_id,
                    "source_split": "train",
                    "customer_approved": True,
                    "eligible_for_validation_prompt": True,
                    "eligible_for_final_test_prompt": True,
                }
                for example_id in selected_few_shot_ids
            ],
            final_test_evidence_status=final_test_evidence["status"],
            final_test_confidence_level=final_test_evidence["confidence_level"],
            final_test_accuracy_lower_bound=final_test_evidence["accuracy_lower_bound"],
            target_accuracy_statistically_supported=final_test_evidence["target_supported"],
            minimum_zero_failure_scenarios=final_test_evidence["minimum_zero_failure_scenarios"],
            cases=all_cases,
        )

    def _run_cases(
        self,
        prompt: str,
        prompt_kind: str,
        examples: list[Example],
        split: str,
        target_model: str,
        evaluator_model: str,
        budget: _Budget,
        few_shot_source: list[Example] | None = None,
        few_shot_ids: list[str] | None = None,
        repeats: int = 1,
        evaluator: dict[str, Any] | None = None,
        max_parallel_scenarios: int = 1,
        target_max_tokens: int | None = None,
        request_options: Mapping[str, Any] | None = None,
    ) -> list[CaseResult]:
        evaluator_policy = evaluator or {"type": "semantic"}
        normalized_target_options = normalize_request_options(request_options)
        resolved_target_max_tokens = int(target_max_tokens) if target_max_tokens is not None else {
            "exact_text": 64,
            "exact_json": 1024,
            "numeric_tolerance": 64,
            "custom": 8192,
            "semantic": 8192,
        }[evaluator_policy["type"]]

        def run_execution(example: Example, repeat_index: int) -> list[CaseResult]:
            scenario_results: list[CaseResult] = []
            transcript: list[dict[str, Any]] = []
            demonstrations = _few_shot_messages(few_shot_source or [], few_shot_ids or [], exclude_id=example.id if split == "train" else "")
            for turn_index, turn in enumerate(example.conversation()):
                target = self._call(
                    budget,
                    target_model,
                    [{"role": "system", "content": prompt}] + demonstrations + transcript + [{"role": "user", "content": turn.input}],
                    max_tokens=resolved_target_max_tokens,
                    response_schema=(
                        _exact_json_response_schema(turn.approved_output, evaluator_policy)
                        if evaluator_policy["type"] == "exact_json"
                        and "response_format" not in normalized_target_options
                        else None
                    ),
                    request_options=normalized_target_options,
                )
                transcript += [{"role": "user", "content": turn.input}, {"role": "assistant", "content": target.content}]
                judgment, judge_completion = self._judge(example, turn, turn_index, transcript, target.content, evaluator_model, budget, evaluator_policy)
                repeat_suffix = f":repeat-{repeat_index + 1}" if repeats > 1 else ""
                scenario_results.append(
                    CaseResult(
                        example_id=f"{example.id}:turn-{turn_index + 1}{repeat_suffix}",
                        split=split,
                        prompt_kind=prompt_kind,
                        output=target.content,
                        approved_output=turn.approved_output,
                        passed=judgment.passed,
                        score=judgment.score,
                        reason=judgment.reason,
                        target_cost_usd=target.cost_usd,
                        evaluator_cost_usd=judge_completion.cost_usd,
                        target_generation_id=target.generation_id,
                        evaluator_generation_id=judge_completion.generation_id,
                        target_latency_ms=target.latency_ms,
                        evaluator_latency_ms=judge_completion.latency_ms,
                        group=example.group,
                        difficulty=example.difficulty,
                        weight=example.weight,
                        critical=example.critical,
                    )
                )
            return scenario_results

        executions = [(example, repeat_index) for example in examples for repeat_index in range(int(repeats))]
        if len(executions) <= 1 or int(max_parallel_scenarios) <= 1:
            return [result for example, repeat_index in executions for result in run_execution(example, repeat_index)]
        completed: dict[tuple[str, int], list[CaseResult]] = {}
        with ThreadPoolExecutor(max_workers=min(int(max_parallel_scenarios), len(executions))) as pool:
            futures = {
                _submit_with_context(pool, run_execution, example, repeat_index):
                (example.id, repeat_index)
                for example, repeat_index in executions
            }
            for future in as_completed(futures):
                completed[futures[future]] = future.result()
        return [result for example, repeat_index in executions for result in completed[(example.id, repeat_index)]]

    def _judge(
        self,
        example: Example,
        turn: Turn,
        turn_index: int,
        transcript: list[dict[str, Any]],
        output: str,
        evaluator_model: str,
        budget: _Budget,
        evaluator: dict[str, Any] | None = None,
    ) -> tuple[Judgment, Completion]:
        evaluator = evaluator or {"type": "semantic"}
        if evaluator["type"] == "custom":
            scorer = resolve_registered_scorer(evaluator, self.custom_scorers)
            request = ScoreRequest(
                scenario_id=example.id,
                turn=turn_index + 1,
                input=turn.input,
                transcript=tuple(dict(message) for message in transcript),
                approved_output=turn.approved_output,
                actual_output=output,
                group=example.group,
                difficulty=example.difficulty,
            )
            try:
                if bool(getattr(scorer, "thread_safe", False)):
                    raw_result = scorer.score(request)
                else:
                    with self._custom_scorer_locks[str(evaluator["scorer_id"])]:
                        raw_result = scorer.score(request)
                score_result = ScoreResult.from_value(raw_result)
            except CustomScorerError:
                raise
            except Exception as error:
                raise CustomScorerError(
                    f"Custom scorer {evaluator['scorer_id']!r} raised "
                    f"{type(error).__name__}; its message was omitted."
                ) from error
            judgment = Judgment(
                score_result.passed, score_result.score, score_result.reason
            )
            scorer_label = (
                f"{evaluator['scorer_id']}@{evaluator['scorer_version']}"
            )
            return judgment, Completion(
                content=json.dumps(asdict(judgment), separators=(",", ":")),
                model=f"custom/{scorer_label}",
                generation_id=f"custom:{scorer_label}",
                cost_usd=0.0,
            )
        if evaluator["type"] != "semantic":
            judgment = _deterministic_judgment(output, turn.approved_output, evaluator)
            return judgment, Completion(
                content=json.dumps(asdict(judgment), separators=(",", ":")),
                model=f"deterministic/{evaluator['type']}",
                generation_id=f"deterministic:{evaluator['type']}",
                cost_usd=0.0,
            )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["passed", "score", "reason"],
            "properties": {
                "passed": {"type": "boolean"},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
        }
        completion = self._call(
            budget,
            evaluator_model,
            [
                {
                    "role": "system",
                    "content": (
                        "Judge whether the actual answer satisfies the behavior demonstrated "
                        "by the customer-approved answer. Do not require identical wording. "
                        "Return only the required JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "scenario_id": example.id,
                            "turn": turn_index + 1,
                            "transcript": [
                                {
                                    **message,
                                    "content": safe_input_descriptor(message.get("content")),
                                }
                                for message in transcript
                            ],
                            "input": safe_input_descriptor(turn.input),
                            "approved_answer": turn.approved_output,
                            "actual_answer": output,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=180,
            response_schema=schema,
        )
        try:
            value = _parse_json_object(completion.content)
            score = min(1.0, max(0.0, float(value["score"])))
            return Judgment(bool(value["passed"]), score, str(value["reason"])), completion
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ProviderError("The evaluator returned invalid structured judgment JSON.") from error

    def _propose_prompt(
        self,
        prompt: str,
        target_model: str,
        train: list[Example],
        baseline: list[CaseResult],
        optimizer_model: str,
        budget: _Budget,
        allow_few_shot: bool,
        max_few_shot_examples: int,
    ) -> tuple[str, list[str]]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["prompt", "hypothesis", "few_shot_example_ids"],
            "properties": {
                "prompt": {"type": "string"},
                "hypothesis": {"type": "string"},
                "few_shot_example_ids": {"type": "array", "items": {"type": "string"}},
            },
        }
        payload = {
            "target_model": target_model,
            "current_prompt": prompt,
            "examples": [
                {
                    **asdict(item),
                    "input": safe_input_descriptor(item.input),
                    "turns": [
                        {
                            "input": safe_input_descriptor(turn.input),
                            "approved_output": turn.approved_output,
                        }
                        for turn in item.conversation()
                    ],
                }
                for item in train
            ],
            "baseline_results": [
                {
                    "example_id": item.example_id,
                    "output": item.output,
                    "passed": item.passed,
                    "reason": item.reason,
                }
                for item in baseline
            ],
            "few_shot_allowed": bool(allow_few_shot),
            "allowed_few_shot_example_ids": [item.id for item in train] if allow_few_shot else [],
            "max_few_shot_examples": int(max_few_shot_examples),
        }
        completion = self._call(
            budget,
            optimizer_model,
            [
                {
                    "role": "system",
                    "content": (
                        "Improve the current prompt package for the named target model. "
                        "You may rewrite the system prompt, select approved training examples as "
                        "few-shot demonstrations, do both, or keep the package. Before proposing, audit "
                        "every approved training scenario and every observed failure. Preserve every "
                        "label-changing boundary, exception, precedence rule, output constraint, and "
                        "multi-turn dependency; do not collapse cases with different approved behavior. "
                        "The production model will receive only the proposed system prompt, any selected "
                        "few-shot demonstrations, the current conversation transcript, and the new user "
                        "message. Make the package self-contained: never refer to a supplied policy, "
                        "training set, approved answer, test, or instruction that will not actually be "
                        "present at production time. Generalize rules from training evidence instead of "
                        "copying validation-specific wording. "
                        "Mentally replay the proposed package against all supplied scenarios. Use only "
                        "allowed IDs; demonstrations add production token cost. Return the package and "
                        "hypothesis as JSON."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=900,
            response_schema=schema,
        )
        try:
            parsed = _parse_json_object(completion.content)
            revised = str(parsed["prompt"]).strip()
            allowed = {item.id for item in train} if allow_few_shot else set()
            selected_ids = list(dict.fromkeys(str(value) for value in parsed.get("few_shot_example_ids", []) if str(value) in allowed))[: max(0, int(max_few_shot_examples))]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ProviderError("The optimizer returned invalid structured prompt JSON.") from error
        if len(revised) < 8:
            raise ProviderError("The optimizer returned an unusably short prompt.")
        return revised, selected_ids


def _parse_json_object(value: str) -> dict[str, Any]:
    text = str(value).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _exact_json_response_schema(
    approved_output: str, evaluator: dict[str, Any]
) -> dict[str, Any]:
    """Build a shape-only schema without leaking the approved answer."""
    expected = _parse_json_object(approved_output)
    required = list(evaluator.get("required_keys") or expected.keys())

    def shape(value: Any) -> dict[str, Any]:
        if value is None:
            return {"type": "null"}
        if isinstance(value, bool):
            return {"type": "boolean"}
        if isinstance(value, int):
            return {"type": "integer"}
        if isinstance(value, float):
            return {"type": "number"}
        if isinstance(value, list):
            return {"type": "array", "items": shape(value[0]) if value else {}}
        if isinstance(value, dict):
            return {
                "type": "object",
                "properties": {str(key): shape(item) for key, item in value.items()},
                "required": [str(key) for key in value],
                "additionalProperties": False,
            }
        return {"type": "string"}

    return {
        "type": "object",
        "properties": {key: shape(expected[key]) for key in required if key in expected},
        "required": required,
        "additionalProperties": bool(evaluator.get("allow_additional_properties", True)),
    }


def _validate_evaluator_policy(value: dict[str, Any] | None) -> dict[str, Any]:
    policy = dict(value or {"type": "semantic"})
    evaluator_type = str(policy.get("type", "")).strip().lower()
    if evaluator_type not in {
        "semantic", "exact_text", "exact_json", "numeric_tolerance", "custom"
    }:
        raise ValueError(
            "evaluator.type must be semantic, exact_text, exact_json, "
            "numeric_tolerance, or custom."
        )
    policy["type"] = evaluator_type
    if evaluator_type == "custom":
        allowed_fields = {"type", "scorer_id", "scorer_version"}
        unexpected = set(policy) - allowed_fields
        if unexpected:
            raise ValueError(
                "A custom evaluator may contain only type, scorer_id, and "
                "scorer_version. Suite files cannot select executable code "
                f"(unexpected: {', '.join(sorted(unexpected))})."
            )
        try:
            scorer_id, scorer_version = validate_scorer_identity(
                policy["scorer_id"], policy["scorer_version"]
            )
        except KeyError as error:
            raise ValueError(
                "A custom evaluator requires scorer_id and scorer_version."
            ) from error
        policy["scorer_id"] = scorer_id
        policy["scorer_version"] = scorer_version
    if evaluator_type == "exact_json":
        required_keys = policy.get("required_keys", [])
        if not isinstance(required_keys, list) or any(not str(key).strip() for key in required_keys):
            raise ValueError("evaluator.required_keys must be a list of non-empty strings.")
        policy["required_keys"] = list(dict.fromkeys(str(key).strip() for key in required_keys))
        policy["allow_additional_properties"] = bool(policy.get("allow_additional_properties", True))
        policy["normalize_rational_strings"] = bool(policy.get("normalize_rational_strings", False))
    if evaluator_type == "numeric_tolerance":
        try:
            minimum = float(policy["minimum"])
            maximum = float(policy["maximum"])
            tolerance = float(policy["absolute_tolerance"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "numeric_tolerance requires numeric minimum, maximum, and absolute_tolerance."
            ) from error
        if not all(math.isfinite(item) for item in (minimum, maximum, tolerance)):
            raise ValueError("numeric_tolerance values must be finite.")
        if maximum <= minimum:
            raise ValueError("numeric_tolerance maximum must be greater than minimum.")
        if tolerance <= 0 or tolerance >= maximum - minimum:
            raise ValueError(
                "numeric_tolerance absolute_tolerance must be greater than zero and smaller than the scale."
            )
        policy["minimum"] = minimum
        policy["maximum"] = maximum
        policy["absolute_tolerance"] = tolerance
    return policy


def _extract_single_numeric_scalar(value: Any) -> float | None:
    """Extract one unambiguous scalar without pretending multi-number prose is one."""

    text = str(value).strip()
    try:
        return float(text)
    except (TypeError, ValueError):
        matches = re.findall(
            r"(?<![\w.])-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])",
            text,
        )
        if len(matches) != 1:
            return None
        try:
            return float(matches[0])
        except ValueError:
            return None


def _validate_difficulty_thresholds(value: dict[str, float] | None) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for raw_name, raw_floor in dict(value or {}).items():
        name = str(raw_name).strip()
        floor = float(raw_floor)
        if not name:
            raise ValueError("difficulty_thresholds keys must be non-empty.")
        if not math.isfinite(floor) or not 0 < floor <= 1:
            raise ValueError("Every difficulty threshold must be greater than zero and at most one.")
        thresholds[name] = floor
    return thresholds


def _deterministic_judgment(
    output: str, approved_output: str, evaluator: dict[str, Any]
) -> Judgment:
    if evaluator["type"] == "exact_text":
        passed = str(output).strip() == str(approved_output).strip()
        return Judgment(passed, 1.0 if passed else 0.0, "Exact text matched." if passed else "Exact text differed.")
    if evaluator["type"] == "numeric_tolerance":
        actual = _extract_single_numeric_scalar(output)
        expected = _extract_single_numeric_scalar(approved_output)
        if actual is None or expected is None:
            return Judgment(False, 0.0, "Actual and approved answers must each contain one unambiguous numeric scalar.")
        minimum = float(evaluator["minimum"])
        maximum = float(evaluator["maximum"])
        tolerance = float(evaluator["absolute_tolerance"])
        if not math.isfinite(actual) or not math.isfinite(expected):
            return Judgment(False, 0.0, "Numeric answers must be finite.")
        if not minimum <= actual <= maximum:
            return Judgment(False, 0.0, f"Actual score was outside the allowed {minimum:g} to {maximum:g} scale.")
        difference = abs(actual - expected)
        passed = difference <= tolerance
        score = max(0.0, 1.0 - difference / max(tolerance, 1e-12)) if passed else 0.0
        return Judgment(
            passed,
            score,
            (
                f"Numeric score was within ±{tolerance:g} of the approved score."
                if passed
                else f"Numeric score differed by {difference:g}, above the ±{tolerance:g} tolerance."
            ),
        )
    try:
        actual = _parse_json_object(output)
        expected = _parse_json_object(approved_output)
    except (TypeError, ValueError, json.JSONDecodeError):
        return Judgment(False, 0.0, "Actual answer was not one valid JSON object.")
    required = set(evaluator.get("required_keys", []))
    actual_keys = set(actual)
    if not required.issubset(actual_keys):
        return Judgment(False, 0.0, f"Missing required JSON key(s): {', '.join(sorted(required - actual_keys))}.")
    if not evaluator.get("allow_additional_properties", True) and actual_keys != required:
        extras = actual_keys - required
        return Judgment(False, 0.0, f"Unexpected JSON key(s): {', '.join(sorted(extras))}.")
    comparison_keys = required or set(expected)
    for key in comparison_keys:
        if key not in expected or key not in actual:
            return Judgment(False, 0.0, f"JSON key {key!r} was not comparable.")
        actual_value = actual[key]
        expected_value = expected[key]
        if evaluator.get("normalize_rational_strings"):
            try:
                equal = Fraction(str(actual_value).strip()) == Fraction(str(expected_value).strip())
            except (ValueError, ZeroDivisionError):
                equal = actual_value == expected_value
        else:
            equal = actual_value == expected_value
        if not equal:
            return Judgment(False, 0.0, f"JSON value for {key!r} differed from the approved value.")
    return Judgment(True, 1.0, "JSON structure and approved values matched deterministically.")


def _safe_provider_error_detail(value: str) -> str:
    """Keep actionable provider diagnostics without exporting account identifiers."""
    try:
        payload = json.loads(str(value))
        error = payload.get("error") or {}
        safe: dict[str, Any] = {
            "message": str(error.get("message") or "Provider request failed."),
            "code": error.get("code"),
        }
        provider = (error.get("metadata") or {}).get("provider_name")
        if provider:
            safe["provider"] = str(provider)
        return json.dumps(safe, separators=(",", ":"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return "Provider request failed; raw provider detail was omitted."


def _split_examples(examples: list[Example], seed: str) -> tuple[list[Example], list[Example], list[Example]]:
    # A 25-case automatic suite used to spend only five unique scenarios on the
    # final confirmation. Preserve a useful development set, but reserve 40% of
    # substantive suites for the untouched final test. With five balanced strata
    # of five, this yields 10 training, 5 development, and 10 final scenarios.
    deep_confirmation = len(examples) >= 25
    if examples and all(item.group for item in examples):
        grouped: dict[str, list[Example]] = {}
        for item in examples:
            grouped.setdefault(item.group, []).append(item)
        train: list[Example] = []
        dev: list[Example] = []
        holdout: list[Example] = []
        for group, items in sorted(grouped.items()):
            ranked_group = sorted(
                items,
                key=lambda item: hashlib.sha256(f"{seed}:{group}:{item.id}".encode()).hexdigest(),
            )
            holdout_count = max(
                1,
                (len(ranked_group) * (2 if deep_confirmation else 1)) // 5,
            )
            dev_count = max(1, len(ranked_group) // 5)
            holdout.extend(ranked_group[:holdout_count])
            dev.extend(ranked_group[holdout_count : holdout_count + dev_count])
            train.extend(ranked_group[holdout_count + dev_count :])
        return train, dev, holdout
    ranked = sorted(
        examples,
        key=lambda item: hashlib.sha256(f"{seed}:{item.id}".encode()).hexdigest(),
    )
    holdout_count = max(
        1,
        (len(ranked) * (2 if deep_confirmation else 1)) // 5,
    )
    dev_count = max(1, len(ranked) // 5)
    return ranked[holdout_count + dev_count :], ranked[holdout_count : holdout_count + dev_count], ranked[:holdout_count]


def _few_shot_messages(examples: list[Example], selected_ids: list[str], exclude_id: str = "") -> list[dict[str, Any]]:
    selected = set(selected_ids)
    messages: list[dict[str, Any]] = []
    for example in examples:
        if example.id not in selected or example.id == exclude_id:
            continue
        for turn in example.conversation():
            messages += [{"role": "user", "content": turn.input}, {"role": "assistant", "content": turn.approved_output}]
    return messages


def _pass_rate(results: list[CaseResult]) -> float:
    total_weight = sum(item.weight for item in results)
    return (
        sum(item.weight for item in results if item.passed) / total_weight
        if total_weight > 0
        else 0.0
    )


def _scenario_pass_rate(results: list[CaseResult]) -> float:
    """Count distinct scenarios once; every repeated execution must pass."""
    scenarios: dict[str, dict[str, Any]] = {}
    for item in results:
        scenario_id = item.example_id.split(":turn-", 1)[0]
        scenario = scenarios.setdefault(scenario_id, {"outcomes": [], "weight": item.weight})
        scenario["outcomes"].append(item.passed)
    total_weight = sum(float(value["weight"]) for value in scenarios.values())
    if total_weight <= 0:
        return 0.0
    passed_weight = sum(
        float(value["weight"])
        for value in scenarios.values()
        if all(value["outcomes"])
    )
    return passed_weight / total_weight


def _binomial_exact_lower_bound(
    successes: int, trials: int, confidence_level: float = 0.95
) -> float | None:
    """One-sided Clopper-Pearson lower bound without a statistics dependency."""

    if trials <= 0 or successes < 0 or successes > trials:
        return None
    if successes == 0:
        return 0.0
    alpha = 1.0 - float(confidence_level)
    low, high = 0.0, 1.0
    for _ in range(80):
        probability = (low + high) / 2
        upper_tail = sum(
            math.comb(trials, count)
            * probability ** count
            * (1 - probability) ** (trials - count)
            for count in range(successes, trials + 1)
        )
        if upper_tail < alpha:
            low = probability
        else:
            high = probability
    return round((low + high) / 2, 6)


def _final_test_evidence(
    results: list[CaseResult], *, target_accuracy: float, confidence_level: float = 0.95
) -> dict[str, Any]:
    """Describe sampling precision without treating repeats as new scenarios."""

    scenarios: dict[str, dict[str, Any]] = {}
    for item in results:
        scenario_id = item.example_id.split(":turn-", 1)[0]
        scenario = scenarios.setdefault(
            scenario_id, {"outcomes": [], "weight": float(item.weight)}
        )
        scenario["outcomes"].append(bool(item.passed))
    trials = len(scenarios)
    minimum = None
    if 0 < target_accuracy < 1 and 0 < confidence_level < 1:
        minimum = math.ceil(
            math.log(1 - confidence_level) / math.log(target_accuracy)
        )
    base = {
        "confidence_level": float(confidence_level),
        "accuracy_lower_bound": None,
        "target_supported": False,
        "minimum_zero_failure_scenarios": minimum,
    }
    if not trials:
        return {"status": "NO_FINAL_TEST", **base}
    weights = [float(item["weight"]) for item in scenarios.values()]
    if max(weights) - min(weights) > 1e-12:
        return {"status": "PROVISIONAL_WEIGHTED_FINAL_TEST", **base}
    successes = sum(
        1 for item in scenarios.values() if all(item["outcomes"])
    )
    lower_bound = _binomial_exact_lower_bound(
        successes, trials, confidence_level
    )
    supported = bool(
        lower_bound is not None and lower_bound >= float(target_accuracy)
    )
    return {
        "status": (
            "TARGET_SUPPORTED_UNDER_BINOMIAL_ASSUMPTIONS"
            if supported
            else "PROVISIONAL_SMALL_FINAL_TEST"
        ),
        **base,
        "accuracy_lower_bound": lower_bound,
        "target_supported": supported,
    }


def _scenario_pass_rates_by_difficulty(results: list[CaseResult]) -> dict[str, float]:
    by_difficulty: dict[str, list[CaseResult]] = {}
    for item in results:
        by_difficulty.setdefault(item.difficulty or "typical", []).append(item)
    return {
        difficulty: round(_scenario_pass_rate(items), 6)
        for difficulty, items in sorted(by_difficulty.items())
    }


def _percentile(values: list[int], fraction: float) -> int:
    """Return a conservative nearest-rank percentile for production sizing."""
    if not values:
        return 1
    ordered = sorted(max(0, int(value)) for value in values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


def _quality_frontier(
    results: list[ModelResult], minimum_gain: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(
        results,
        key=lambda item: (
            item.estimated_cost_per_successful_call_usd,
            -item.holdout_pass_rate,
            item.model,
        ),
    )
    frontier: list[dict[str, Any]] = []
    best_quality = -1.0
    dominated: list[dict[str, Any]] = []
    for item in ordered:
        point = {
            "model": item.model,
            "holdout_pass_rate": item.holdout_pass_rate,
            "estimated_cost_per_successful_call_usd": item.estimated_cost_per_successful_call_usd,
        }
        gain = item.holdout_pass_rate - best_quality if best_quality >= 0 else item.holdout_pass_rate
        if item.holdout_pass_rate > best_quality:
            frontier.append(point)
            best_quality = item.holdout_pass_rate
        elif gain < minimum_gain:
            dominated.append({**point, "measured_quality_gain": round(gain, 6)})
    return frontier, {
        "minimum_meaningful_quality_gain": minimum_gain,
        "higher_cost_models_without_material_gain": dominated,
        "caveat": "Measured only on this frozen suite; price is not a proxy for intelligence.",
    }
