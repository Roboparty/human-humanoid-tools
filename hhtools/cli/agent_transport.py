"""Strict HTTP transport for the versioned Agent JSON CLI.

The CLI is intentionally a client of the long-lived Web composition root.  It
does not construct an ``AssetRegistry``, ``JobManager``, scheduler, or solver of
its own; otherwise a submitted job would be owned by a short-lived CLI process.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from hhtools.contracts import ApiError, ArtifactDescriptor, ErrorStage, NextAction
from hhtools.contracts.portability import (
    PortableJsonError as ContractPortableJsonError,
)
from hhtools.contracts.portability import (
    looks_like_host_path as contract_looks_like_host_path,
)
from hhtools.contracts.portability import (
    validate_portable_json as validate_contract_portable_json,
)

_DEFAULT_MAX_JSON_BYTES = 16 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_MAX_JSON_NUMBER_TOKEN = 128


class AgentTransportError(RuntimeError):
    """Expected transport failure expressed with the public error contract."""

    def __init__(self, error: ApiError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")


class StrictJsonError(ValueError):
    """The document uses a non-standard constant or a duplicate object key."""


class PortableJsonError(ValueError):
    """The public JSON document contains a host-specific path or unsafe value."""


def _reject_json_constant(value: str) -> None:
    raise StrictJsonError(f"non-standard JSON constant: {value}")


def _strict_json_int(value: str) -> int:
    if len(value) > _MAX_JSON_NUMBER_TOKEN:
        raise StrictJsonError("integer token is too long")
    return int(value)


def _strict_json_float(value: str) -> float:
    if len(value) > _MAX_JSON_NUMBER_TOKEN:
        raise StrictJsonError("float token is too long")
    result = float(value)
    if not math.isfinite(result):
        raise StrictJsonError("non-finite number")
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise StrictJsonError("duplicate JSON object key")
        document[key] = value
    return document


def loads_strict_json(payload: str | bytes) -> Any:
    """Decode RFC-style JSON without Python's NaN or duplicate-key extensions."""

    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    except UnicodeDecodeError as error:
        raise StrictJsonError("JSON is not valid UTF-8") from error
    try:
        return json.loads(
            text,
            parse_float=_strict_json_float,
            parse_int=_strict_json_int,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise StrictJsonError("invalid JSON syntax") from error
    except RecursionError as error:
        raise StrictJsonError("JSON nesting exceeds the supported depth") from error


def _looks_like_host_path(
    value: str,
    *,
    allow_same_origin_ui_url: bool = False,
) -> bool:
    return contract_looks_like_host_path(
        value,
        allow_same_origin_ui_url=allow_same_origin_ui_url,
    )


def ensure_portable_json(
    value: Any,
    *,
    allow_same_origin_ui_url: bool = False,
) -> None:
    """Fail closed when public JSON could reveal a host path or local URI."""

    try:
        if allow_same_origin_ui_url and isinstance(value, str):
            if contract_looks_like_host_path(
                value,
                allow_same_origin_ui_url=True,
            ):
                raise ContractPortableJsonError("host path")
            return
        validate_contract_portable_json(value)
    except ContractPortableJsonError as error:
        raise PortableJsonError(str(error)) from error


class AgentTransport(Protocol):
    """Small injectable boundary used by the command adapter and its tests."""

    def request_json(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str | int | float | bool | None] | None = None,
        document: Mapping[str, Any] | None = None,
    ) -> Any: ...

    def download_artifact(
        self,
        path: str,
        *,
        destination: Path,
        descriptor: ArtifactDescriptor,
        overwrite: bool,
    ) -> None: ...


def _transport_error(
    code: str,
    message: str,
    *,
    stage: ErrorStage = ErrorStage.INTERNAL,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
    next_action: NextAction | None = None,
) -> AgentTransportError:
    return AgentTransportError(
        ApiError(
            code=code,
            message=message,
            retryable=retryable,
            stage=stage,
            details=dict(details or {}),
            next_action=next_action,
        )
    )


def _output_write_error(error: OSError) -> AgentTransportError:
    return _transport_error(
        "OUTPUT_WRITE_FAILED",
        "The artifact output could not be written or published.",
        stage=ErrorStage.ARTIFACT,
        retryable=False,
    )


def _validate_base_url(value: str) -> str:
    """Accept one HTTP(S) Agent namespace URL without credentials or query."""

    parsed = urlsplit(value.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise _transport_error(
            "INVALID_PARAMETER",
            "The Agent base URL must be an HTTP(S) URL without credentials, query, or fragment.",
            stage=ErrorStage.REQUEST,
            details={"field": "base_url"},
        )
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _api_error_from_body(payload: bytes) -> ApiError | None:
    try:
        document = loads_strict_json(payload)
        return ApiError.model_validate(document)
    except (ValueError, TypeError):
        return None


class HttpAgentTransport:
    """stdlib-only HTTP transport for an already-running Agent REST service."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        max_json_bytes: int = _DEFAULT_MAX_JSON_BYTES,
    ) -> None:
        self._base_url = _validate_base_url(base_url)
        if not 0.1 <= timeout_seconds <= 3_600:
            raise _transport_error(
                "INVALID_PARAMETER",
                "The Agent timeout must be between 0.1 and 3600 seconds.",
                stage=ErrorStage.REQUEST,
                details={"field": "timeout"},
            )
        self._timeout_seconds = float(timeout_seconds)
        self._max_json_bytes = int(max_json_bytes)

    def _url(
        self,
        path: str,
        query: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> str:
        if not path.startswith("/") or ".." in path.split("/"):
            raise _transport_error(
                "INTERNAL_ERROR",
                "The CLI constructed an invalid Agent endpoint.",
            )
        values = {
            key: str(value).lower() if isinstance(value, bool) else str(value)
            for key, value in (query or {}).items()
            if value is not None
        }
        suffix = f"?{urlencode(values)}" if values else ""
        return f"{self._base_url}{path}{suffix}"

    def _raise_http_error(self, error: HTTPError) -> None:
        payload = error.read(self._max_json_bytes + 1)
        parsed = _api_error_from_body(payload)
        if parsed is not None:
            raise AgentTransportError(parsed) from error
        raise _transport_error(
            "REMOTE_PROTOCOL_ERROR",
            "The Agent service returned an error outside the versioned contract.",
            retryable=error.code >= 500,
            details={"http_status": error.code},
        ) from error

    def _raise_connection_error(self, error: BaseException) -> None:
        raise _transport_error(
            "AGENT_SERVICE_UNAVAILABLE",
            "The Agent service is unavailable at the configured base URL.",
            retryable=True,
            next_action=NextAction(
                actor="human",
                action="start_agent_service",
                message="Start `hhtools web` or select the correct local Agent endpoint.",
            ),
        ) from error

    def request_json(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str | int | float | bool | None] | None = None,
        document: Mapping[str, Any] | None = None,
    ) -> Any:
        payload = None
        headers = {"Accept": "application/json", "User-Agent": "hhtools-agent-json/1"}
        if document is not None:
            payload = json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            self._url(path, query),
            data=payload,
            headers=headers,
            method=method.upper(),
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                body = response.read(self._max_json_bytes + 1)
        except HTTPError as error:
            self._raise_http_error(error)
            raise AssertionError("unreachable") from error
        except (OSError, TimeoutError, URLError) as error:
            self._raise_connection_error(error)
            raise AssertionError("unreachable") from error
        if len(body) > self._max_json_bytes:
            raise _transport_error(
                "REMOTE_PROTOCOL_ERROR",
                "The Agent JSON response exceeds the CLI safety limit.",
            )
        try:
            return loads_strict_json(body)
        except StrictJsonError as error:
            raise _transport_error(
                "REMOTE_PROTOCOL_ERROR",
                "The Agent service did not return one valid UTF-8 JSON document.",
            ) from error

    def download_artifact(
        self,
        path: str,
        *,
        destination: Path,
        descriptor: ArtifactDescriptor,
        overwrite: bool,
    ) -> None:
        """Stream one authorized artifact to an explicit local destination.

        The descriptor is fetched and membership-checked before this method is
        called.  Server and client hashes are both verified; bytes never enter
        the JSON response or an in-memory Base64 representation.
        """

        target = Path(destination).expanduser()
        try:
            parent = target.parent.resolve(strict=True)
        except OSError as error:
            raise _transport_error(
                "INVALID_PARAMETER",
                "The artifact output parent directory does not exist or is unavailable.",
                stage=ErrorStage.REQUEST,
                details={"field": "output"},
            ) from error
        if not parent.is_dir() or target.name in {"", ".", ".."}:
            raise _transport_error(
                "INVALID_PARAMETER",
                "The artifact output must name a file inside an existing directory.",
                stage=ErrorStage.REQUEST,
                details={"field": "output"},
            )
        target = parent / target.name
        if target.exists() and (target.is_dir() or not overwrite):
            raise _transport_error(
                "OUTPUT_EXISTS",
                "The artifact output already exists; pass --force to replace a file.",
                stage=ErrorStage.REQUEST,
                details={"field": "output"},
            )

        request = Request(
            self._url(path),
            headers={
                "Accept": "application/octet-stream",
                "User-Agent": "hhtools-agent-json/1",
            },
            method="GET",
        )
        try:
            temporary_fd, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.hhtools-",
                suffix=".tmp",
                dir=parent,
            )
        except OSError as error:
            raise _output_write_error(error) from error
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        size = 0
        try:
            try:
                with os.fdopen(temporary_fd, "wb") as stream:
                    try:
                        with urlopen(  # noqa: S310
                            request,
                            timeout=self._timeout_seconds,
                        ) as response:
                            while True:
                                try:
                                    chunk = response.read(_COPY_CHUNK_BYTES)
                                except (OSError, TimeoutError, URLError) as error:
                                    self._raise_connection_error(error)
                                if not chunk:
                                    break
                                try:
                                    stream.write(chunk)
                                except OSError as error:
                                    raise _output_write_error(error) from error
                                digest.update(chunk)
                                size += len(chunk)
                    except HTTPError as error:
                        self._raise_http_error(error)
                    except (OSError, TimeoutError, URLError) as error:
                        self._raise_connection_error(error)
                    try:
                        stream.flush()
                        os.fsync(stream.fileno())
                    except OSError as error:
                        raise _output_write_error(error) from error
            except AgentTransportError:
                raise
            except OSError as error:
                raise _output_write_error(error) from error

            if descriptor.size_bytes is not None and size != descriptor.size_bytes:
                raise _transport_error(
                    "ARTIFACT_HASH_MISMATCH",
                    "The downloaded artifact size differs from its descriptor.",
                    stage=ErrorStage.ARTIFACT,
                    retryable=True,
                )
            if descriptor.sha256 is not None and digest.hexdigest() != descriptor.sha256:
                raise _transport_error(
                    "ARTIFACT_HASH_MISMATCH",
                    "The downloaded artifact hash differs from its descriptor.",
                    stage=ErrorStage.ARTIFACT,
                    retryable=True,
                )

            try:
                if overwrite:
                    os.replace(temporary, target)
                else:
                    # A hard-link publication is atomic and fails if another process
                    # created the destination after the initial existence check.
                    os.link(temporary, target)
            except FileExistsError as error:
                raise _transport_error(
                    "OUTPUT_EXISTS",
                    "The artifact output was created concurrently.",
                    stage=ErrorStage.REQUEST,
                    details={"field": "output"},
                ) from error
            except OSError as error:
                raise _output_write_error(error) from error
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = [
    "AgentTransport",
    "AgentTransportError",
    "HttpAgentTransport",
    "PortableJsonError",
    "StrictJsonError",
    "ensure_portable_json",
    "loads_strict_json",
]
