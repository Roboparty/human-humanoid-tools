"""Local-only transport boundary for the versioned Agent HTTP API.

The Agent API deliberately has no remote authentication in v1.  This pure
ASGI middleware therefore treats a literal loopback connection *and* a
loopback ``Host`` header as part of the protocol boundary.  It also buffers a
small, bounded request body before FastAPI or Pydantic can parse it, so a
missing ``Content-Length`` or chunked transfer cannot bypass the limit.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

from starlette.responses import JSONResponse
from starlette.types import Message, Receive, Scope, Send

from hhtools.contracts import ApiError, ErrorStage

AGENT_API_PREFIX = "/api/agent/v1"
LEGACY_UPGRADE_PATH = f"{AGENT_API_PREFIX}/legacy/jobspec-v1/upgrade"
AGENT_MAX_BODY_BYTES = 1024 * 1024
LEGACY_UPGRADE_MAX_BODY_BYTES = 64 * 1024

_CONTENT_LENGTH_RE = re.compile(rb"[0-9]+\Z")

type ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def is_agent_path(path: str) -> bool:
    """Return whether ``path`` is inside the exact v1 Agent namespace."""

    return path == AGENT_API_PREFIX or path.startswith(f"{AGENT_API_PREFIX}/")


def agent_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    stage: ErrorStage = ErrorStage.REQUEST,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Build the one versioned error envelope used by early Agent failures."""

    response_headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    if headers is not None:
        response_headers.update(headers)
    error = ApiError(
        code=code,
        message=message,
        retryable=retryable,
        stage=stage,
        details=dict(details or {}),
    )
    return JSONResponse(
        status_code=status_code,
        content=error.model_dump(mode="json", exclude_none=True),
        headers=response_headers,
    )


def _is_loopback_literal(value: str | None) -> bool:
    if value is None:
        return False
    try:
        address = ipaddress.ip_address(value)
    except (TypeError, ValueError):
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped.is_loopback
    return address.is_loopback


def _host_name(value: str) -> str | None:  # noqa: PLR0911
    """Parse a strict HTTP Host value without DNS resolution."""

    if not value or any(character.isspace() for character in value):
        return None
    if value.startswith("["):
        closing = value.find("]")
        if closing <= 1:
            return None
        name = value[1:closing]
        remainder = value[closing + 1 :]
        if remainder and (not remainder.startswith(":") or not _valid_port(remainder[1:])):
            return None
        return name

    # An unbracketed IPv6 Host is ambiguous with a port and is not valid per
    # the HTTP URI grammar.  Require the normal ``[::1]:port`` spelling.
    if value.count(":") > 1:
        return None
    if ":" in value:
        name, port = value.rsplit(":", 1)
        if not name or not _valid_port(port):
            return None
        return name
    return value


def _valid_port(value: str) -> bool:
    return (
        value.isascii() and value.isdigit() and 1 <= len(value) <= 5 and 1 <= int(value) <= 65_535
    )


def _loopback_host(headers: list[tuple[bytes, bytes]]) -> bool:
    values = [value for name, value in headers if name.lower() == b"host"]
    if len(values) != 1:
        return False
    try:
        raw = values[0].decode("ascii")
    except UnicodeDecodeError:
        return False
    name = _host_name(raw)
    if name is None:
        return False
    return name.casefold() == "localhost" or _is_loopback_literal(name)


def _content_length(headers: list[tuple[bytes, bytes]]) -> tuple[int | None, bool]:
    values = [value for name, value in headers if name.lower() == b"content-length"]
    if not values:
        return None, True
    if len(values) != 1 or _CONTENT_LENGTH_RE.fullmatch(values[0]) is None:
        return None, False
    try:
        return int(values[0]), True
    except ValueError:  # pragma: no cover - guarded by the decimal regex
        return None, False


def _identity_content_encoding(headers: list[tuple[bytes, bytes]]) -> bool:
    values = [value for name, value in headers if name.lower() == b"content-encoding"]
    if not values:
        return True
    if len(values) != 1:
        return False
    try:
        return values[0].decode("ascii").casefold() == "identity"
    except UnicodeDecodeError:
        return False


def _loopback_origin(headers: list[tuple[bytes, bytes]]) -> bool:
    """Allow absent CLI Origin or one syntactically valid loopback Web origin."""

    values = [value for name, value in headers if name.lower() == b"origin"]
    if not values:
        return True
    if len(values) != 1:
        return False
    try:
        raw = values[0].decode("ascii")
        parsed = urlsplit(raw)
        # An Origin is only scheme + authority.  Userinfo, paths, query and
        # fragments are rejected instead of being normalized permissively.
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            return False
        if parsed.port is not None and not 1 <= parsed.port <= 65_535:
            return False
    except (UnicodeDecodeError, ValueError):
        return False
    hostname = parsed.hostname
    return hostname is not None and (
        hostname.casefold() == "localhost" or _is_loopback_literal(hostname)
    )


async def _bounded_body(
    receive: Receive,
    *,
    maximum: int,
) -> tuple[bytes | None, bool]:
    """Read one HTTP body, returning ``(body, disconnected)``.

    ``None`` means the limit was crossed.  The check happens while chunks are
    received, so it also covers HTTP/1.1 chunked bodies and HTTP/2 requests
    where no trustworthy Content-Length exists.
    """

    body = bytearray()
    while True:
        message = await receive()
        message_type = message.get("type")
        if message_type == "http.disconnect":
            return bytes(body), True
        if message_type != "http.request":
            continue
        chunk = message.get("body", b"")
        if chunk:
            body.extend(chunk)
            if len(body) > maximum:
                return None, False
        if not message.get("more_body", False):
            return bytes(body), False


class AgentBoundaryMiddleware:
    """Enforce the local-only, bounded-body v1 Agent transport boundary."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(  # noqa: PLR0911
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope.get("type") != "http" or not is_agent_path(str(scope.get("path", ""))):
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        client_host = client[0] if isinstance(client, tuple) and client else None
        if not _is_loopback_literal(client_host):
            await agent_error_response(
                status_code=403,
                code="LOOPBACK_REQUIRED",
                message="The Agent API only accepts loopback clients.",
            )(scope, receive, send)
            return

        headers = list(scope.get("headers", []))
        if not _loopback_host(headers):
            await agent_error_response(
                status_code=400,
                code="INVALID_HOST",
                message="The Agent API requires a loopback Host header.",
            )(scope, receive, send)
            return

        # CLI clients normally omit Origin.  When a browser supplies one, a
        # second loopback check prevents a public site from issuing a simple
        # cross-origin mutation against this unauthenticated localhost API.
        if not _loopback_origin(headers):
            await agent_error_response(
                status_code=403,
                code="ORIGIN_FORBIDDEN",
                message="Browser requests to the Agent API require a loopback Origin.",
            )(scope, receive, send)
            return

        declared_length, valid_length = _content_length(headers)
        if not valid_length:
            await agent_error_response(
                status_code=400,
                code="INVALID_CONTENT_LENGTH",
                message="Content-Length must be one non-negative decimal value.",
            )(scope, receive, send)
            return

        if not _identity_content_encoding(headers):
            await agent_error_response(
                status_code=415,
                code="UNSUPPORTED_CONTENT_ENCODING",
                message="Compressed Agent request bodies are not supported.",
            )(scope, receive, send)
            return

        maximum = (
            LEGACY_UPGRADE_MAX_BODY_BYTES
            if scope.get("path") == LEGACY_UPGRADE_PATH
            else AGENT_MAX_BODY_BYTES
        )
        if declared_length is not None and declared_length > maximum:
            await self._too_large(scope, receive, send, maximum=maximum)
            return

        body, disconnected = await _bounded_body(receive, maximum=maximum)
        if body is None:
            await self._too_large(scope, receive, send, maximum=maximum)
            return
        if disconnected:
            await agent_error_response(
                status_code=400,
                code="INCOMPLETE_REQUEST_BODY",
                message="The Agent request body ended before it was complete.",
            )(scope, receive, send)
            return
        if declared_length is not None and declared_length != len(body):
            await agent_error_response(
                status_code=400,
                code="INVALID_CONTENT_LENGTH",
                message="Content-Length does not match the received request body.",
            )(scope, receive, send)
            return

        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            # The original channel is now waiting for a real disconnect.  A
            # StreamingResponse listener may await it, and its task group will
            # cancel that wait as soon as streaming completes.
            return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _too_large(
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        maximum: int,
    ) -> None:
        await agent_error_response(
            status_code=413,
            code="REQUEST_TOO_LARGE",
            message="The Agent request body exceeds the route limit.",
            details={"max_bytes": maximum},
        )(scope, receive, send)


__all__ = [
    "AGENT_API_PREFIX",
    "AGENT_MAX_BODY_BYTES",
    "AgentBoundaryMiddleware",
    "LEGACY_UPGRADE_MAX_BODY_BYTES",
    "LEGACY_UPGRADE_PATH",
    "agent_error_response",
    "is_agent_path",
]
