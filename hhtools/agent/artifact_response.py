"""Race-resistant HTTP streaming for canonically authorized Agent artifacts."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import stat
from collections.abc import Iterator

from starlette.responses import Response, StreamingResponse

from hhtools.contracts import ApiError, ArtifactDescriptor, ErrorStage
from hhtools.services.artifacts import StoredArtifact
from hhtools.services.jobs import JobManagerError

_READ_CHUNK_BYTES = 1024 * 1024
_SAFE_JOB_ID = re.compile(r"^job:[A-Za-z0-9][A-Za-z0-9._~-]{0,251}$")


def _artifact_failure(message: str, *, retryable: bool = True) -> JobManagerError:
    return JobManagerError(
        ApiError(
            code="ARTIFACT_HASH_MISMATCH",
            message=message,
            retryable=retryable,
            stage=ErrorStage.ARTIFACT,
        )
    )


def _internal_failure(message: str) -> JobManagerError:
    return JobManagerError(
        ApiError(
            code="INTERNAL_ERROR",
            message=message,
            retryable=False,
            stage=ErrorStage.INTERNAL,
        )
    )


def _validated_descriptor(stored: StoredArtifact) -> ArtifactDescriptor:
    """Revalidate even a test double constructed without Pydantic validation."""

    try:
        descriptor = ArtifactDescriptor.model_validate(stored.descriptor.model_dump(mode="python"))
        if _SAFE_JOB_ID.fullmatch(descriptor.job_id) is None:
            raise ValueError("unsafe job id")
        return descriptor
    except (AttributeError, TypeError, ValueError) as exc:
        raise _internal_failure(
            "The managed artifact descriptor is not safe for transport."
        ) from exc


def verified_artifact_response(stored: StoredArtifact) -> Response:
    """Hash and stream bytes through one open file handle.

    ``FileResponse`` opens a path later, after route authorization and optional
    verification.  A path swap in that interval could make the response serve
    different bytes.  This helper opens once, verifies SHA-256 and length on
    that exact descriptor, rewinds it, and gives the same handle to
    ``StreamingResponse``. JSON artifacts retain the exact verified bytes in a
    buffered response so the route can apply its strict JSON portability guard.
    It intentionally does not implement Range or the ASGI ``pathsend`` extension.
    """

    descriptor = _validated_descriptor(stored)
    if descriptor.sha256 is None or descriptor.size_bytes is None:
        raise _internal_failure("The managed artifact is missing required integrity metadata.")
    media_type = descriptor.media_type or "application/octet-stream"
    content_type = media_type.partition(";")[0].strip().lower()
    json_chunks: list[bytes] | None = (
        [] if content_type == "application/json" or content_type.endswith("+json") else None
    )

    try:
        handle = stored.path.open("rb")
    except OSError as exc:
        raise _artifact_failure("The managed artifact is unavailable for download.") from exc

    try:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise _artifact_failure(
                "The managed artifact is not a regular file.",
                retryable=False,
            )

        digest = hashlib.sha256()
        observed_size = 0
        while chunk := handle.read(_READ_CHUNK_BYTES):
            observed_size += len(chunk)
            digest.update(chunk)
            if json_chunks is not None:
                json_chunks.append(chunk)
        after = os.fstat(handle.fileno())
        stable_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        observed_sha256 = digest.hexdigest()
        if (
            not stable_identity
            or observed_size != descriptor.size_bytes
            or after.st_size != descriptor.size_bytes
            or not hmac.compare_digest(observed_sha256, descriptor.sha256)
        ):
            raise _artifact_failure(
                "The managed artifact no longer matches its canonical descriptor."
            )
        if json_chunks is None:
            handle.seek(0)
        else:
            handle.close()
    except Exception:
        handle.close()
        raise

    def stream_same_handle() -> Iterator[bytes]:
        try:
            while chunk := handle.read(_READ_CHUNK_BYTES):
                yield chunk
        finally:
            handle.close()

    digest_bytes = bytes.fromhex(descriptor.sha256)
    filename = descriptor.kind
    if descriptor.format is not None:
        filename = f"{filename}.{descriptor.format}"
    headers = {
        "Cache-Control": "no-store",
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Digest": f"sha-256=:{base64.b64encode(digest_bytes).decode('ascii')}:",
        "Content-Length": str(descriptor.size_bytes),
        "ETag": f'"sha256:{descriptor.sha256}"',
        "X-Content-SHA256": descriptor.sha256,
        "X-Content-Type-Options": "nosniff",
        "X-HHTools-Artifact-Id": descriptor.artifact_id,
        "X-HHTools-Job-Id": descriptor.job_id,
    }
    if json_chunks is not None:
        return Response(
            content=b"".join(json_chunks),
            status_code=200,
            media_type=media_type,
            headers=headers,
        )
    return StreamingResponse(
        stream_same_handle(),
        status_code=200,
        media_type=media_type,
        headers=headers,
    )


__all__ = ["verified_artifact_response"]
