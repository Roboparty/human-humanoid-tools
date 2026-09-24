"""Immutable managed artifacts for Agent-facing jobs.

The store owns the bytes below its data directory and exposes only controlled
``hhtools://`` resource URIs.  Host paths are never persisted in public
descriptors or metadata, which keeps manifests portable across Windows, Linux,
and macOS deployments.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO

from pydantic import ValidationError

from hhtools.contracts import ApiError, ArtifactDescriptor, ErrorStage

_CHUNK_SIZE = 1024 * 1024
_KIND = re.compile(r"^[a-z][a-z0-9_-]{0,127}$")
_FORMAT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,31}$")
_MAX_METADATA_BYTES = 64 * 1024
_OBJECT_WRITE_LOCK = threading.Lock()
_WINDOWS_PUBLISH_RETRIES = 20


class ArtifactStoreError(RuntimeError):
    """Expected artifact failure with a transport-neutral error body."""

    def __init__(self, error: ApiError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")

    @property
    def api_error(self) -> ApiError:
        return self.error

    @property
    def code(self) -> str:
        return self.error.code


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """One validated descriptor plus its private managed object path."""

    descriptor: ArtifactDescriptor
    path: Path


class _InvalidArtifactError(ValueError):
    """Private validation failure that must not expose caller values."""


def _error(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    stage: ErrorStage = ErrorStage.ARTIFACT,
    details: Mapping[str, Any] | None = None,
) -> ArtifactStoreError:
    return ArtifactStoreError(
        ApiError(
            code=code,
            message=message,
            retryable=retryable,
            stage=stage,
            details=dict(details or {}),
        )
    )


def _looks_like_host_path(value: str) -> bool:
    if re.match(r"^(?:hhtools|https?)://[^\s]+$", value):
        return False
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return posix.is_absolute() or windows.is_absolute() or bool(windows.drive) or bool(windows.root)


def _validate_portable_json(value: Any) -> None:
    if value is None or isinstance(value, bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _InvalidArtifactError("non-finite number")
        return
    if isinstance(value, str):
        if _looks_like_host_path(value):
            raise _InvalidArtifactError("host path")
        return
    if isinstance(value, list):
        for item in value:
            _validate_portable_json(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or _looks_like_host_path(key):
                raise _InvalidArtifactError("invalid object key")
            _validate_portable_json(item)
        return
    raise _InvalidArtifactError("non-JSON value")


def _canonical_json(value: Any) -> str:
    _validate_portable_json(value)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise _InvalidArtifactError("invalid JSON") from exc
    return encoded


def _validate_job_id(job_id: str) -> None:
    # Job ids are minted by JobStore.  Keeping the URI segment strict prevents
    # a future HTTP/resource adapter from having to reinterpret slashes or URLs.
    if not re.fullmatch(r"job:[A-Za-z0-9._~-]{1,240}", job_id):
        raise _error("INVALID_PARAMETER", "The artifact job id is invalid.")


def _validate_kind(kind: str) -> None:
    if _KIND.fullmatch(kind) is None:
        raise _error("INVALID_PARAMETER", "The artifact kind is invalid.")


def _validate_format(format_name: str | None) -> None:
    if format_name is not None and _FORMAT.fullmatch(format_name) is None:
        raise _error("INVALID_PARAMETER", "The artifact format is invalid.")


def _hash_stream(stream: BinaryIO, target: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(_CHUNK_SIZE):
        digest.update(chunk)
        target.write(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


class ArtifactStore:
    """SQLite-indexed, immutable artifact bytes below one managed root."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        self._object_root = self._data_dir / "artifact-objects"
        self._database_path = self._data_dir / "artifacts.sqlite3"
        try:
            self._object_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The artifact store directory could not be initialized.",
                retryable=True,
                stage=ErrorStage.INTERNAL,
            ) from exc
        self._initialize_database()

    @property
    def database_path(self) -> Path:
        """Internal database location for deployment diagnostics only."""

        return self._database_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS artifacts (
                        artifact_id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        format TEXT,
                        media_type TEXT,
                        size_bytes INTEGER NOT NULL,
                        sha256 TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        object_path TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS artifacts_job_created
                    ON artifacts(job_id, created_at, artifact_id)
                    """
                )
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The artifact store database could not be initialized.",
                retryable=True,
                stage=ErrorStage.INTERNAL,
            ) from exc

    def put_json(
        self,
        *,
        job_id: str,
        kind: str,
        document: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactDescriptor:
        """Persist one canonical portable JSON artifact."""

        try:
            payload = _canonical_json(document).encode("utf-8")
        except _InvalidArtifactError as exc:
            raise _error(
                "INVALID_PARAMETER",
                "Artifact JSON must contain finite portable values without host paths.",
            ) from exc
        return self.put_bytes(
            job_id=job_id,
            kind=kind,
            payload=payload,
            format="json",
            media_type="application/json",
            metadata=metadata,
        )

    def put_bytes(
        self,
        *,
        job_id: str,
        kind: str,
        payload: bytes,
        format: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactDescriptor:
        """Persist immutable in-memory bytes and return their compact descriptor."""

        if not isinstance(payload, bytes):
            raise _error("INVALID_PARAMETER", "Artifact payload must be bytes.")
        return self._put_stream(
            job_id=job_id,
            kind=kind,
            stream_factory=lambda: io.BytesIO(payload),
            format=format,
            media_type=media_type,
            metadata=metadata,
        )

    def put_file(
        self,
        *,
        job_id: str,
        kind: str,
        source: Path,
        format: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactDescriptor:
        """Copy one stable internal output file into managed artifact storage."""

        path = Path(source)
        try:
            before = path.stat()
            if not path.is_file():
                raise OSError("not a regular file")
        except OSError as exc:
            raise _error(
                "OUTPUT_WRITE_FAILED",
                "The output artifact is missing or unreadable.",
                retryable=True,
            ) from exc

        def source_is_stable(copied_size: int) -> bool:
            try:
                after = path.stat()
            except OSError:
                return False
            snapshot_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            snapshot_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            return snapshot_before == snapshot_after and copied_size == after.st_size

        return self._put_stream(
            job_id=job_id,
            kind=kind,
            stream_factory=lambda: path.open("rb"),
            format=format,
            media_type=media_type,
            metadata=metadata,
            stability_check=source_is_stable,
        )

    def _put_stream(
        self,
        *,
        job_id: str,
        kind: str,
        stream_factory: Callable[[], BinaryIO],
        format: str | None,
        media_type: str | None,
        metadata: Mapping[str, Any] | None,
        stability_check: Callable[[int], bool] | None = None,
    ) -> ArtifactDescriptor:
        _validate_job_id(job_id)
        _validate_kind(kind)
        _validate_format(format)
        try:
            metadata_json = _canonical_json(dict(metadata or {}))
        except _InvalidArtifactError as exc:
            raise _error(
                "INVALID_PARAMETER",
                "Artifact metadata must be finite portable JSON without host paths.",
            ) from exc
        if len(metadata_json.encode("utf-8")) > _MAX_METADATA_BYTES:
            raise _error("INVALID_PARAMETER", "Artifact metadata is too large.")

        temporary = self._object_root / f".{uuid.uuid4().hex}.tmp"
        try:
            with stream_factory() as source_stream, temporary.open("xb") as target:
                sha256, size = _hash_stream(source_stream, target)
                target.flush()
                os.fsync(target.fileno())
            relative_object = f"artifact-objects/{sha256[:2]}/{sha256}"
            object_path = self._data_dir / PurePosixPath(relative_object)
            object_path.parent.mkdir(parents=True, exist_ok=True)
            self._publish_object(
                temporary,
                object_path,
                sha256=sha256,
                size=size,
            )
            if stability_check is not None and not stability_check(size):
                raise _error(
                    "OUTPUT_WRITE_FAILED",
                    "The output artifact changed while it was being stored.",
                    retryable=True,
                )
        except ArtifactStoreError:
            temporary.unlink(missing_ok=True)
            raise
        except (OSError, TypeError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            raise _error(
                "OUTPUT_WRITE_FAILED",
                "The artifact could not be written to managed storage.",
                retryable=True,
            ) from exc

        identity = _canonical_json(
            {
                "format": format,
                "job_id": job_id,
                "kind": kind,
                "media_type": media_type,
                "metadata": json.loads(metadata_json),
                "sha256": sha256,
            }
        )
        identity_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        artifact_id = f"artifact:{kind}:{identity_digest}"
        created_at = datetime.now(UTC).isoformat()
        row = (
            artifact_id,
            job_id,
            kind,
            format,
            media_type,
            size,
            sha256,
            created_at,
            metadata_json,
            relative_object,
        )
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT OR IGNORE INTO artifacts (
                        artifact_id, job_id, kind, format, media_type,
                        size_bytes, sha256, created_at, metadata_json, object_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )
                stored = connection.execute(
                    "SELECT * FROM artifacts WHERE artifact_id = ?",
                    (artifact_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The artifact descriptor could not be persisted.",
                retryable=True,
                stage=ErrorStage.INTERNAL,
            ) from exc
        if stored is None:
            raise _error(
                "INTERNAL_ERROR",
                "The artifact descriptor is unavailable after persistence.",
                stage=ErrorStage.INTERNAL,
            )
        decoded = self._decode_row(stored)
        expected = row[:7] + (metadata_json, relative_object)
        actual = (
            stored["artifact_id"],
            stored["job_id"],
            stored["kind"],
            stored["format"],
            stored["media_type"],
            stored["size_bytes"],
            stored["sha256"],
            stored["metadata_json"],
            stored["object_path"],
        )
        if actual != expected:
            raise _error(
                "INTERNAL_ERROR",
                "The artifact id is already bound to different content.",
                stage=ErrorStage.INTERNAL,
            )
        return decoded.descriptor

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(_CHUNK_SIZE):
                digest.update(chunk)
        return digest.hexdigest()

    def _publish_object(
        self,
        temporary: Path,
        object_path: Path,
        *,
        sha256: str,
        size: int,
    ) -> None:
        """Publish one content object without racing another Windows writer.

        Windows does not allow ``os.replace`` while another thread/process has
        the destination open for verification.  Serialize local publishers and
        retry the cross-process winner path; content addressing makes accepting
        an already verified destination equivalent to our own rename.
        """

        with _OBJECT_WRITE_LOCK:
            for attempt in range(_WINDOWS_PUBLISH_RETRIES):
                if object_path.exists():
                    try:
                        valid = (
                            object_path.stat().st_size == size
                            and self._hash_file(object_path) == sha256
                        )
                    except PermissionError:
                        valid = False
                    else:
                        if not valid:
                            raise _error(
                                "INTERNAL_ERROR",
                                "Managed artifact content is corrupted.",
                                stage=ErrorStage.INTERNAL,
                            )
                        temporary.unlink(missing_ok=True)
                        return
                try:
                    os.replace(temporary, object_path)
                    return
                except PermissionError:
                    if attempt + 1 >= _WINDOWS_PUBLISH_RETRIES:
                        raise
                    time.sleep(0.005)

    def _decode_row(self, row: sqlite3.Row) -> StoredArtifact:
        try:
            metadata = json.loads(row["metadata_json"])
            if _canonical_json(metadata) != row["metadata_json"]:
                raise _InvalidArtifactError("non-canonical metadata")
            relative = PurePosixPath(row["object_path"])
            if (
                relative.is_absolute()
                or any(part in {"", ".", ".."} for part in relative.parts)
                or relative.parts[:1] != ("artifact-objects",)
            ):
                raise _InvalidArtifactError("invalid object path")
            object_path = (self._data_dir / relative).resolve(strict=False)
            object_path.relative_to(self._data_dir.resolve())
            descriptor = ArtifactDescriptor(
                artifact_id=row["artifact_id"],
                job_id=row["job_id"],
                kind=row["kind"],
                format=row["format"],
                resource_uri=(f"hhtools://jobs/{row['job_id']}/artifacts/{row['artifact_id']}"),
                media_type=row["media_type"],
                size_bytes=row["size_bytes"],
                sha256=row["sha256"],
                created_at=row["created_at"],
                metadata=metadata,
            )
        except (
            _InvalidArtifactError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            ValidationError,
        ) as exc:
            raise _error(
                "INTERNAL_ERROR",
                "A persisted artifact descriptor is invalid.",
                stage=ErrorStage.INTERNAL,
            ) from exc
        expected_path = (
            self._data_dir / "artifact-objects" / descriptor.sha256[:2] / descriptor.sha256
        )
        if object_path != expected_path.resolve(strict=False):
            raise _error(
                "INTERNAL_ERROR",
                "A persisted artifact object path is inconsistent.",
                stage=ErrorStage.INTERNAL,
            )
        return StoredArtifact(descriptor=descriptor, path=object_path)

    def get(self, artifact_id: str, *, verify: bool = False) -> StoredArtifact:
        """Return one managed artifact and optionally verify its current bytes."""

        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM artifacts WHERE artifact_id = ?",
                    (artifact_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The artifact store could not be read.",
                retryable=True,
                stage=ErrorStage.INTERNAL,
            ) from exc
        if row is None:
            raise _error("ARTIFACT_NOT_FOUND", "No artifact has the requested id.")
        stored = self._decode_row(row)
        if verify:
            try:
                valid = (
                    stored.path.is_file()
                    and stored.path.stat().st_size == stored.descriptor.size_bytes
                    and self._hash_file(stored.path) == stored.descriptor.sha256
                )
            except OSError:
                valid = False
            if not valid:
                raise _error(
                    "ARTIFACT_HASH_MISMATCH",
                    "The managed artifact no longer matches its descriptor.",
                    retryable=True,
                )
        return stored

    def list_candidates_for_job(self, job_id: str) -> list[ArtifactDescriptor]:
        """List all managed candidates for a job in stable creation order.

        Candidates are not an authorization or lifecycle-membership boundary.
        A write can survive a failed JobStore CAS or a process interruption, so
        callers serving Agent APIs must list JobStore's canonical descriptors
        instead of exposing this raw catalog.
        """

        _validate_job_id(job_id)
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT * FROM artifacts
                    WHERE job_id = ?
                    ORDER BY created_at, artifact_id
                    """,
                    (job_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The artifact store could not be read.",
                retryable=True,
                stage=ErrorStage.INTERNAL,
            ) from exc
        return [self._decode_row(row).descriptor for row in rows]

    def list_for_job(self, job_id: str) -> list[ArtifactDescriptor]:
        """Compatibility alias for the raw candidate catalog.

        This method may include unbound artifacts.  Use JobManager's canonical
        artifact APIs for user-visible listing and access control.
        """

        return self.list_candidates_for_job(job_id)


__all__ = [
    "ArtifactStore",
    "ArtifactStoreError",
    "StoredArtifact",
]
