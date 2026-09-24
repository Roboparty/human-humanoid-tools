"""Safe delivery of job-scoped artifacts to one server-configured export root."""

from __future__ import annotations

import hashlib
import os
import stat
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from hhtools.contracts import ApiError, ErrorStage
from hhtools.contracts.artifact_exports import ArtifactExportReceipt

from .artifacts import StoredArtifact
from .jobs import JobManager, JobManagerError

AGENT_EXPORT_ROOT_ID = "agent-exports"
_COPY_CHUNK_BYTES = 1024 * 1024
_PUBLISH_RETRIES = 20
_PUBLISH_LOCK = threading.Lock()


class ArtifactExportError(RuntimeError):
    """Expected export failure with a transport-neutral public error."""

    def __init__(self, error: ApiError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")

    @property
    def api_error(self) -> ApiError:
        return self.error

    @property
    def code(self) -> str:
        return self.error.code


def _error(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    stage: ErrorStage = ErrorStage.ARTIFACT,
    details: Mapping[str, str] | None = None,
) -> ArtifactExportError:
    return ArtifactExportError(
        ApiError(
            code=code,
            message=message,
            retryable=retryable,
            stage=stage,
            details=dict(details or {}),
        )
    )


def _copy_job_error(error: ApiError) -> ArtifactExportError:
    return ArtifactExportError(ApiError.model_validate_json(error.model_dump_json()))


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


class ArtifactExportService:
    """Copy verified managed artifacts without accepting caller-selected paths."""

    def __init__(self, jobs: JobManager, export_root: Path) -> None:
        self._jobs = jobs
        try:
            root = Path(export_root).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            self._root = root.resolve(strict=True)
            if not self._root.is_dir():
                raise OSError("export root is not a directory")
        except OSError as exc:
            raise _error(
                "ARTIFACT_EXPORT_FAILED",
                "The configured artifact export root is unavailable.",
                retryable=True,
                stage=ErrorStage.INTERNAL,
                details={"root_id": AGENT_EXPORT_ROOT_ID},
            ) from exc

    @staticmethod
    def _relative_path(stored: StoredArtifact) -> PurePosixPath:
        descriptor = stored.descriptor
        job_token = hashlib.sha256(descriptor.job_id.encode("utf-8")).hexdigest()
        artifact_token = hashlib.sha256(descriptor.artifact_id.encode("utf-8")).hexdigest()
        extension = descriptor.format.casefold() if descriptor.format is not None else "bin"
        return PurePosixPath("jobs", job_token, f"{artifact_token}.{extension}")

    @staticmethod
    def _details(stored: StoredArtifact) -> dict[str, str]:
        return {
            "artifact_id": stored.descriptor.artifact_id,
            "job_id": stored.descriptor.job_id,
            "root_id": AGENT_EXPORT_ROOT_ID,
        }

    @staticmethod
    def _receipt(
        stored: StoredArtifact,
        relative_path: PurePosixPath,
    ) -> ArtifactExportReceipt:
        descriptor = stored.descriptor
        if descriptor.size_bytes is None or descriptor.sha256 is None:
            raise _error(
                "INTERNAL_ERROR",
                "The managed artifact lacks required integrity metadata.",
                retryable=True,
                details=ArtifactExportService._details(stored),
            )
        try:
            return ArtifactExportReceipt(
                relative_path=relative_path.as_posix(),
                job_id=descriptor.job_id,
                artifact_id=descriptor.artifact_id,
                kind=descriptor.kind,
                format=descriptor.format,
                media_type=descriptor.media_type,
                size_bytes=descriptor.size_bytes,
                sha256=descriptor.sha256,
            )
        except ValidationError as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The managed artifact cannot be represented by an export receipt.",
                details=ArtifactExportService._details(stored),
            ) from exc

    def _destination(self, relative_path: PurePosixPath, stored: StoredArtifact) -> Path:
        destination = self._root.joinpath(*relative_path.parts)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            resolved_parent = destination.parent.resolve(strict=True)
            resolved_parent.relative_to(self._root)
        except (OSError, ValueError) as exc:
            raise _error(
                "ARTIFACT_EXPORT_FAILED",
                "The artifact export destination is unavailable.",
                retryable=True,
                details=self._details(stored),
            ) from exc
        return resolved_parent / destination.name

    def _destination_matches(self, destination: Path, stored: StoredArtifact) -> bool:
        descriptor = stored.descriptor
        try:
            metadata = destination.lstat()
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise _error(
                "ARTIFACT_EXPORT_FAILED",
                "The existing artifact export could not be inspected.",
                retryable=True,
                details=self._details(stored),
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            return False
        if metadata.st_size != descriptor.size_bytes:
            return False
        try:
            digest, size = _hash_file(destination)
        except OSError as exc:
            raise _error(
                "ARTIFACT_EXPORT_FAILED",
                "The existing artifact export could not be verified.",
                retryable=True,
                details=self._details(stored),
            ) from exc
        return size == descriptor.size_bytes and digest == descriptor.sha256

    def _copy_to_temporary(self, stored: StoredArtifact, destination: Path) -> Path:
        descriptor = stored.descriptor
        temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        digest = hashlib.sha256()
        size = 0
        try:
            with stored.path.open("rb") as source, temporary.open("xb") as target:
                while chunk := source.read(_COPY_CHUNK_BYTES):
                    digest.update(chunk)
                    target.write(chunk)
                    size += len(chunk)
                target.flush()
                os.fsync(target.fileno())
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise _error(
                "ARTIFACT_EXPORT_FAILED",
                "The managed artifact could not be copied to the export root.",
                retryable=True,
                details=self._details(stored),
            ) from exc
        if size != descriptor.size_bytes or digest.hexdigest() != descriptor.sha256:
            temporary.unlink(missing_ok=True)
            raise _error(
                "ARTIFACT_HASH_MISMATCH",
                "The managed artifact changed while it was being exported.",
                retryable=True,
                details=self._details(stored),
            )
        return temporary

    def _publish(self, temporary: Path, destination: Path, stored: StoredArtifact) -> None:
        with _PUBLISH_LOCK:
            if self._destination_matches(destination, stored):
                temporary.unlink(missing_ok=True)
                return
            for attempt in range(_PUBLISH_RETRIES):
                try:
                    os.replace(temporary, destination)
                    return
                except PermissionError as exc:
                    if self._destination_matches(destination, stored):
                        temporary.unlink(missing_ok=True)
                        return
                    if attempt + 1 >= _PUBLISH_RETRIES:
                        raise _error(
                            "ARTIFACT_EXPORT_FAILED",
                            "The artifact export could not be published atomically.",
                            retryable=True,
                            details=self._details(stored),
                        ) from exc
                    time.sleep(0.005)
                except OSError as exc:
                    raise _error(
                        "ARTIFACT_EXPORT_FAILED",
                        "The artifact export could not be published atomically.",
                        retryable=True,
                        details=self._details(stored),
                    ) from exc

    def export(self, job_id: str, artifact_id: str) -> ArtifactExportReceipt:
        """Export one canonically job-bound artifact and return portable metadata."""

        try:
            stored = self._jobs.get_artifact(job_id, artifact_id, verify=True)
        except JobManagerError as exc:
            raise _copy_job_error(exc.api_error) from exc
        except Exception as exc:  # noqa: BLE001 - application-service boundary
            raise _error(
                "INTERNAL_ERROR",
                "The managed artifact could not be resolved for export.",
                retryable=True,
                stage=ErrorStage.INTERNAL,
                details={"root_id": AGENT_EXPORT_ROOT_ID},
            ) from exc

        relative_path = self._relative_path(stored)
        receipt = self._receipt(stored, relative_path)
        destination = self._destination(relative_path, stored)
        with _PUBLISH_LOCK:
            if self._destination_matches(destination, stored):
                return receipt
        temporary = self._copy_to_temporary(stored, destination)
        try:
            self._publish(temporary, destination, stored)
        finally:
            temporary.unlink(missing_ok=True)
        return receipt


__all__ = [
    "AGENT_EXPORT_ROOT_ID",
    "ArtifactExportError",
    "ArtifactExportService",
]
