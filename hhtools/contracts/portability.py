"""Transport-neutral guards for bounded, host-independent public JSON."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

# Public contracts contain metadata, not bulk payloads. Binary output belongs in
# an artifact; the aggregate budget also prevents splitting one payload across fields.
MAX_PORTABLE_STRING_BYTES = 1024 * 1024
MAX_PORTABLE_DOCUMENT_STRING_BYTES = 2 * 1024 * 1024
MAX_PORTABLE_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_INLINE_BASE64_DECODED_BYTES = 64 * 1024
MAX_PORTABLE_CONTAINER_ITEMS = 10_000
MAX_PORTABLE_NODES = 262_144
MAX_PORTABLE_DEPTH = 64

PORTABLE_URI_PORT_PATTERN = (
    r"(?:0|[1-9][0-9]{0,3}|[1-5][0-9]{4}|6[0-4][0-9]{3}|"
    r"65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])"
)
_IPV6_ADDRESS_PATTERN = (
    r"(?:"
    r"(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}|"
    r"(?:[0-9A-Fa-f]{1,4}:){1,7}:|"
    r"(?:[0-9A-Fa-f]{1,4}:){1,6}:[0-9A-Fa-f]{1,4}|"
    r"(?:[0-9A-Fa-f]{1,4}:){1,5}(?::[0-9A-Fa-f]{1,4}){1,2}|"
    r"(?:[0-9A-Fa-f]{1,4}:){1,4}(?::[0-9A-Fa-f]{1,4}){1,3}|"
    r"(?:[0-9A-Fa-f]{1,4}:){1,3}(?::[0-9A-Fa-f]{1,4}){1,4}|"
    r"(?:[0-9A-Fa-f]{1,4}:){1,2}(?::[0-9A-Fa-f]{1,4}){1,5}|"
    r"[0-9A-Fa-f]{1,4}:(?:(?::[0-9A-Fa-f]{1,4}){1,6})|"
    r":(?:(?::[0-9A-Fa-f]{1,4}){1,7}|:)"
    r")"
)
PORTABLE_URI_HOST_PATTERN = rf"(?:\[(?:{_IPV6_ADDRESS_PATTERN})\]|[A-Za-z0-9._~-]+)"
_URI_PCHAR_PATTERN = r"(?:[A-Za-z0-9._~!$&'()*+,;=:@-]|%[0-9A-Fa-f]{2})"
PORTABLE_URI_TAIL_PATTERN = rf"(?:[/?#](?:{_URI_PCHAR_PATTERN}|[/?#])*)?"

_MAX_PERCENT_DECODE_ROUNDS = 8
_MAX_NESTED_URI_DEPTH = 4
_CONTROLLED_URI_TOKEN = re.compile(
    r"(?:hhtools|https?)://[^\s\"'{}()<>]+",
    re.IGNORECASE,
)
_EMBEDDED_URI_SCHEME = re.compile(r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9+.-]*)://")
_EMBEDDED_FILE_URI = re.compile(r"(?<![A-Za-z0-9])file:", re.IGNORECASE)
_EMBEDDED_DATA_URI = re.compile(r"(?<![A-Za-z0-9])data:", re.IGNORECASE)
_EMBEDDED_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)[^\s\"']*")
_WINDOWS_DRIVE_ANYWHERE = re.compile(r"[A-Za-z]:[\\/][^\s\"']*")
_EMBEDDED_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9/])/(?![/\s])[^\s\"']*")
_EMBEDDED_PROTOCOL_RELATIVE = re.compile(r"(?<![A-Za-z0-9/])//[^\s\"']+")
_EMBEDDED_SENSITIVE_POSIX_PATH = re.compile(
    r"/(?:Applications|Library|System|Users|Volumes|app|bin|boot|code|data|datasets|"
    r"dev|etc|gpfs|home|lib|lib64|lustre|media|mnt|models|nfs|opt|private|proc|"
    r"project|projects|raid|root|run|sbin|scratch|srv|storage|sys|tmp|usr|var|"
    r"workspace|workspaces)(?:/|$)[^\s\"']*"
)
_URI_WITH_USERINFO = re.compile(
    r"(?:hhtools|https?)://[^\s/?#]*@",
    re.IGNORECASE,
)
_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
_SAFE_URI_SEGMENT = re.compile(r"^[A-Za-z0-9._~:-]+$")
_BASE64 = re.compile(r"^[A-Za-z0-9+/_-]*={0,2}$")
_EXPLICIT_BASE64_PREFIX = re.compile(
    r"^(?:base64:|[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+\s*;\s*base64\s*,)",
    re.IGNORECASE,
)
_UI_QUERY_KEY = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SAFE_UI_QUERY_KEYS = frozenset({"calibrate", "panel", "robot", "view"})
_SAFE_WEB_ROUTE_QUERY_KEYS = frozenset({"callback", "next", "redirect", "return"})
_SENSITIVE_PUBLIC_KEY = re.compile(
    r"(?:api[_-]?key|authorization|credential|password|secret|token)$",
    re.IGNORECASE,
)
_PORTABLE_IPV6_ADDRESS = re.compile(rf"^(?:{_IPV6_ADDRESS_PATTERN})$")
_PORTABLE_HOSTNAME = re.compile(r"^[A-Za-z0-9._~-]+$")
_PORTABLE_HTTP_PATH = re.compile(rf"^(?:{_URI_PCHAR_PATTERN}|/)*$")
_PORTABLE_HTTP_QUERY_FRAGMENT = re.compile(rf"^(?:{_URI_PCHAR_PATTERN}|[/?])*$")


class PortableJsonError(ValueError):
    """A public JSON document contains a non-portable value."""


def _decoded_layers(value: str) -> tuple[str, ...]:
    """Return bounded percent-decoding layers, failing closed on deeper nesting."""

    layers = [value]
    for _ in range(_MAX_PERCENT_DECODE_ROUNDS):
        if _PERCENT_ESCAPE.search(layers[-1]) is None:
            break
        try:
            decoded = unquote(layers[-1], errors="strict")
        except UnicodeError:
            # ``looks_like_host_path`` is a predicate used at protocol
            # boundaries.  Malformed percent-encoded bytes must be rejected,
            # not escape as an implementation-specific decoder exception.
            layers.append("/invalid-percent-encoding")
            break
        if decoded == layers[-1]:
            break
        layers.append(decoded)
    if _PERCENT_ESCAPE.search(layers[-1]) is not None:
        layers.append("/encoded-path-depth-exceeded")
    return tuple(layers)


def _raw_host_path(value: str) -> bool:
    if _EMBEDDED_FILE_URI.search(value) or _EMBEDDED_DATA_URI.search(value):
        return True
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or windows.root:
        return True
    return bool(
        _EMBEDDED_WINDOWS_PATH.search(value)
        or _WINDOWS_DRIVE_ANYWHERE.search(value)
        or _EMBEDDED_SENSITIVE_POSIX_PATH.search(value)
        or _EMBEDDED_POSIX_PATH.search(value)
        or _EMBEDDED_PROTOCOL_RELATIVE.search(value)
    )


def _decoded_value_has_host_path(value: str, *, nested_depth: int = 0) -> bool:
    for layer in _decoded_layers(value):
        try:
            parsed = urlsplit(layer)
        except (UnicodeError, ValueError):
            return True
        scheme = parsed.scheme.casefold()
        if scheme in {"http", "https"}:
            if nested_depth >= _MAX_NESTED_URI_DEPTH or _http_uri_has_host_path(
                layer,
                nested_depth=nested_depth + 1,
            ):
                return True
            continue
        if scheme == "hhtools":
            if not _canonical_hhtools_uri(layer):
                return True
            continue
        if scheme or _raw_host_path(layer):
            return True
    return False


def _canonical_hhtools_uri(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        invalid = (
            parsed.scheme.casefold() != "hhtools"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.query
            or parsed.fragment
            or "%" in value
        )
    except (UnicodeError, ValueError):
        return False
    if invalid:
        return False
    authority = parsed.netloc
    parts = parsed.path.removeprefix("/").split("/") if parsed.path else []
    if any(_SAFE_URI_SEGMENT.fullmatch(part) is None for part in parts):
        return False
    valid = False
    if authority == "capabilities":
        valid = not parts
    elif authority == "schemas":
        valid = len(parts) == 3 and parts[:2] == ["agent", "v1"]
    elif authority in {"robots", "plans"}:
        valid = len(parts) == 1
    elif authority == "assets":
        valid = len(parts) == 2 and parts[1] == "manifest"
    elif authority == "jobs":
        report = len(parts) == 2 and parts[1] in {
            "status",
            "manifest",
            "evaluation",
            "failures",
        }
        valid = report or (len(parts) == 3 and parts[1] == "artifacts")
    return valid


def _http_uri_has_host_path(  # noqa: PLR0911 - fail closed at each URI boundary
    value: str,
    *,
    nested_depth: int = 0,
) -> bool:
    try:
        parsed = urlsplit(value)
        _port = parsed.port
    except (UnicodeError, ValueError):
        return True
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        return True
    if (
        _PORTABLE_IPV6_ADDRESS.fullmatch(parsed.hostname) is None
        if ":" in parsed.hostname
        else _PORTABLE_HOSTNAME.fullmatch(parsed.hostname) is None
    ):
        return True
    if (
        _PORTABLE_HTTP_PATH.fullmatch(parsed.path) is None
        or _PORTABLE_HTTP_QUERY_FRAGMENT.fullmatch(parsed.query) is None
        or _PORTABLE_HTTP_QUERY_FRAGMENT.fullmatch(parsed.fragment) is None
    ):
        return True

    # A URL path is portable, but Windows/file syntax in it is not. Query values
    # and fragments are data, so absolute path syntax there is always a leak.
    for layer in _decoded_layers(parsed.path):
        if _EMBEDDED_FILE_URI.search(layer) or _WINDOWS_DRIVE_ANYWHERE.search(layer):
            return True
    try:
        query_items = parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=32)
    except (UnicodeError, ValueError):
        return True
    for key, item in query_items:
        if _decoded_value_has_host_path(key, nested_depth=nested_depth):
            return True
        if not _decoded_value_has_host_path(item, nested_depth=nested_depth):
            continue
        if key in _SAFE_WEB_ROUTE_QUERY_KEYS and _safe_root_relative_web_path(item):
            continue
        return True
    return _decoded_value_has_host_path(parsed.fragment, nested_depth=nested_depth)


def _safe_root_relative_web_path(value: str) -> bool:
    """Allow a bounded web route without treating it as a filesystem path."""

    for layer in _decoded_layers(value):
        if layer in {"/encoded-path-depth-exceeded", "/invalid-percent-encoding"}:
            return False
        if (
            not layer.startswith("/")
            or layer.startswith("//")
            or len(layer) > 2048
            or _EMBEDDED_FILE_URI.search(layer)
            or _WINDOWS_DRIVE_ANYWHERE.search(layer)
            or _EMBEDDED_SENSITIVE_POSIX_PATH.search(layer)
        ):
            return False
    return True


def _safe_next_action_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (UnicodeError, ValueError):
        return False
    if parsed.scheme:
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or port is None
        ):
            return False
    elif parsed.netloc or not value.startswith("/"):
        return False
    if parsed.path != "/" or parsed.fragment:
        return False
    try:
        fields = parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=16)
    except (UnicodeError, ValueError):
        return False
    return all(
        key in _SAFE_UI_QUERY_KEYS
        and _UI_QUERY_KEY.fullmatch(key) is not None
        and len(item) <= 256
        and not _decoded_value_has_host_path(item)
        for key, item in fields
    )


def _looks_like_host_path_once(value: str) -> bool:
    # Inspect path syntax before masking a controlled URI.  Otherwise a
    # malicious URI authority such as ``https://C:\\Users\\...@host`` can make
    # the URI regex consume the drive prefix and leave only a harmless-looking
    # suffix behind.
    if (
        _EMBEDDED_FILE_URI.search(value)
        or _EMBEDDED_DATA_URI.search(value)
        or _URI_WITH_USERINFO.search(value)
    ):
        return True
    for match in _EMBEDDED_URI_SCHEME.finditer(value):
        if match.group(1).casefold() not in {"hhtools", "http", "https"}:
            return True

    unsafe_uri = False

    def portable_uri(uri: str) -> bool:
        return (
            _canonical_hhtools_uri(uri)
            if uri.casefold().startswith("hhtools://")
            else not _http_uri_has_host_path(uri)
        )

    def mask_uri(match: re.Match[str]) -> str:
        nonlocal unsafe_uri
        uri = match.group(0)
        safe = portable_uri(uri)
        # Square brackets commonly wrap a URI in prose and tests.  Keep a
        # single closing wrapper outside the mask only when removing it turns
        # the complete URI into a valid portable token; malformed IPv6 remains
        # fail-closed because its truncated candidate is invalid too.
        if not safe and uri.endswith("]") and portable_uri(uri[:-1]):
            return "]"
        if not safe:
            unsafe_uri = True
            return uri
        return ""

    masked = _CONTROLLED_URI_TOKEN.sub(mask_uri, value)
    return unsafe_uri or _raw_host_path(masked)


def looks_like_host_path(
    value: str,
    *,
    allow_same_origin_ui_url: bool = False,
) -> bool:
    """Detect raw or encoded POSIX, Windows, UNC, and file paths."""

    try:
        value.encode("utf-8")
    except UnicodeError:
        return True
    if allow_same_origin_ui_url and _safe_next_action_url(value):
        return False
    return any(_looks_like_host_path_once(layer) for layer in _decoded_layers(value))


def _safe_documentation_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        _port = parsed.port
    except (UnicodeError, ValueError):
        return False
    return bool(
        parsed.scheme == "https"
        and parsed.hostname
        and parsed.username is None
        and parsed.password is None
        and not _http_uri_has_host_path(value)
    )


def is_portable_next_action_url(value: str) -> bool:
    """Whether a NextAction URL is a bounded local UI route or HTTPS documentation."""

    return _safe_next_action_url(value) or _safe_documentation_url(value)


def is_portable_resource_uri(value: str) -> bool:
    """Whether a public resource URI is canonical and host independent."""

    try:
        parsed = urlsplit(value)
    except (UnicodeError, ValueError):
        return False
    if parsed.scheme.casefold() == "hhtools":
        return _canonical_hhtools_uri(value)
    if parsed.scheme.casefold() in {"http", "https"}:
        return not _http_uri_has_host_path(value)
    return False


def _looks_like_large_base64(value: str) -> bool:
    explicit = _EXPLICIT_BASE64_PREFIX.match(value)
    payload = value[explicit.end() :] if explicit is not None else value
    compact = (
        "".join(payload.split())
        if explicit is not None
        else payload.replace("\r", "").replace("\n", "")
    )
    if _BASE64.fullmatch(compact) is None:
        return False
    core = compact.rstrip("=")
    padding = len(compact) - len(core)
    # RFC 4648 URL-safe values are commonly emitted without trailing padding.
    # A one-character remainder can never be valid Base64; padded values must
    # retain the normal four-character block shape.
    if len(core) % 4 == 1 or (padding and len(compact) % 4):
        return False
    decoded_size = (len(core) * 6) // 8
    return decoded_size > MAX_INLINE_BASE64_DECODED_BYTES


@dataclass
class _StringBudget:
    used_bytes: int = 0
    nodes: int = 0

    def enter(self, *, depth: int) -> None:
        if depth > MAX_PORTABLE_DEPTH:
            raise PortableJsonError("document nesting too deep")
        self.nodes += 1
        if self.nodes > MAX_PORTABLE_NODES:
            raise PortableJsonError("document node budget exceeded")

    @staticmethod
    def check_container(size: int) -> None:
        if size > MAX_PORTABLE_CONTAINER_ITEMS:
            raise PortableJsonError("container item budget exceeded")

    def consume(self, value: str) -> None:
        if len(value) > MAX_PORTABLE_STRING_BYTES:
            raise PortableJsonError("string too large")
        try:
            size = len(value.encode("utf-8"))
        except UnicodeError as error:
            raise PortableJsonError("invalid UTF-8 string") from error
        if size > MAX_PORTABLE_STRING_BYTES:
            raise PortableJsonError("string too large")
        if _looks_like_large_base64(value):
            raise PortableJsonError("inline base64 payload too large")
        self.used_bytes += size
        if self.used_bytes > MAX_PORTABLE_DOCUMENT_STRING_BYTES:
            raise PortableJsonError("document string budget exceeded")


def _validate_portable_json(
    value: Any,
    budget: _StringBudget,
    *,
    depth: int,
) -> None:
    budget.enter(depth=depth)
    if value is None or isinstance(value, bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PortableJsonError("non-finite number")
        return
    if isinstance(value, str):
        budget.consume(value)
        if looks_like_host_path(value):
            raise PortableJsonError("host path")
        return
    if isinstance(value, list):
        budget.check_container(len(value))
        for item in value:
            _validate_portable_json(item, budget, depth=depth + 1)
        return
    if isinstance(value, dict):
        budget.check_container(len(value))
        actor = value.get("actor")
        next_action = (
            isinstance(actor, str)
            and actor in {"agent", "human", "system"}
            and isinstance(value.get("action"), str)
        )
        for key, item in value.items():
            if not isinstance(key, str):
                raise PortableJsonError("invalid object key")
            budget.consume(key)
            if looks_like_host_path(key) or _SENSITIVE_PUBLIC_KEY.search(key):
                raise PortableJsonError("invalid object key")
            if next_action and key == "url":
                if item is None:
                    continue
                if not isinstance(item, str) or not is_portable_next_action_url(item):
                    raise PortableJsonError("host path")
                budget.consume(item)
                continue
            _validate_portable_json(item, budget, depth=depth + 1)
        return
    raise PortableJsonError("non-JSON value")


def validate_portable_json(value: Any) -> None:
    """Require finite, bounded public JSON without host-local payloads."""

    _validate_portable_json(value, _StringBudget(), depth=0)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as error:
        raise PortableJsonError("non-JSON value") from error
    if len(encoded) > MAX_PORTABLE_DOCUMENT_BYTES:
        raise PortableJsonError("document byte budget exceeded")


__all__ = [
    "MAX_INLINE_BASE64_DECODED_BYTES",
    "MAX_PORTABLE_CONTAINER_ITEMS",
    "MAX_PORTABLE_DEPTH",
    "MAX_PORTABLE_DOCUMENT_BYTES",
    "MAX_PORTABLE_DOCUMENT_STRING_BYTES",
    "MAX_PORTABLE_NODES",
    "MAX_PORTABLE_STRING_BYTES",
    "PORTABLE_URI_HOST_PATTERN",
    "PORTABLE_URI_PORT_PATTERN",
    "PORTABLE_URI_TAIL_PATTERN",
    "PortableJsonError",
    "is_portable_next_action_url",
    "is_portable_resource_uri",
    "looks_like_host_path",
    "validate_portable_json",
]
