"""Durable job identity and lifecycle storage for agent-facing execution.

``JobStore`` owns durable facts only: the immutable :class:`JobSpecV2`, the
current lifecycle revision, outcome, progress, artifact descriptors, retry
lineage, and a cooperative cancellation request.  It deliberately does not
reserve scheduler capacity, select a device, start a worker, invoke a solver,
or write artifact payload bytes.

The idempotency fingerprint is the SHA-256 digest of the complete canonical
JobSpec v2 document.  Consequently, a key can be retried safely for exactly
the same execution identity, while reusing it for a changed plan, parameter,
asset, provenance snapshot, or output policy is a conflict.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, NoReturn

from pydantic import ValidationError

from hhtools.contracts import (
    AgentJobView,
    ApiError,
    ArtifactDescriptor,
    ErrorStage,
    JobOutcome,
    JobProgress,
    JobSpecV2,
    JobState,
    NextAction,
)

_IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~:-]{0,255}$")
_JOB_ID_PATTERN = re.compile(r"^job:[A-Za-z0-9][A-Za-z0-9._~-]{0,251}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_JOB_WAIT_SECONDS = 60.0
_TERMINAL_STATES = frozenset({JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED})
_ALLOWED_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.QUEUED: frozenset({JobState.RUNNING, JobState.FAILED, JobState.CANCELLED}),
    JobState.RUNNING: frozenset({JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}),
    JobState.COMPLETED: frozenset(),
    JobState.FAILED: frozenset(),
    JobState.CANCELLED: frozenset(),
}

_SELECT_JOB = """
    SELECT
        job_id,
        idempotency_key,
        request_fingerprint,
        spec_sha256,
        spec_json,
        state,
        outcome,
        progress_json,
        summary_json,
        error_json,
        next_action_json,
        revision,
        cancel_requested,
        parent_job_id,
        root_job_id,
        attempt,
        artifacts_json,
        submitted_at,
        started_at,
        completed_at,
        cancel_requested_at,
        poll_after_ms
    FROM jobs
"""


class JobStoreError(RuntimeError):
    """Expected job-store failure with a transport-neutral error body."""

    def __init__(self, error: ApiError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")

    @property
    def api_error(self) -> ApiError:
        """Alias used by REST, CLI, and MCP adapters."""

        return self.error

    @property
    def code(self) -> str:
        """Return the stable machine code without parsing prose."""

        return self.error.code


@dataclass(frozen=True, slots=True)
class StoredJob:
    """Validated snapshot returned by the durable job store.

    A fresh instance is decoded for every operation.  This is significant
    because JobSpec v2 is frozen at the model boundary but intentionally
    contains JSON dictionaries; mutating a returned dictionary cannot mutate
    the persisted execution identity.
    """

    spec: JobSpecV2
    view: AgentJobView
    idempotency_key: str
    request_fingerprint: str
    artifacts: tuple[ArtifactDescriptor, ...]
    cancel_requested: bool
    cancel_requested_at: datetime | None
    created: bool = False

    @property
    def job_id(self) -> str:
        """Return the public job id."""

        return self.view.job_id

    @property
    def revision(self) -> int:
        """Return the store-owned lifecycle revision."""

        return self.view.progress.revision


class _InvalidStoredJobError(ValueError):
    """Private signal for malformed or internally inconsistent rows."""


def _error(
    code: str,
    message: str,
    *,
    stage: ErrorStage = ErrorStage.EXECUTION,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
) -> JobStoreError:
    return JobStoreError(
        ApiError(
            code=code,
            message=message,
            retryable=retryable,
            stage=stage,
            details=dict(details or {}),
        )
    )


def _reject_json_constant(value: str) -> NoReturn:
    raise _InvalidStoredJobError(f"non-finite JSON number: {value}")


def _object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidStoredJobError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _strict_json_loads(payload: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError) as exc:
        raise _InvalidStoredJobError("invalid JSON document") from exc


def _looks_like_absolute_path(value: str) -> bool:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return posix.is_absolute() or windows.is_absolute() or bool(windows.drive) or bool(windows.root)


def _validate_portable_json(value: Any, *, location: str = "$") -> None:
    """Reject non-JSON values, non-finite numbers, and host absolute paths."""

    if value is None or isinstance(value, bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _InvalidStoredJobError(f"non-finite number at {location}")
        return
    if isinstance(value, str):
        if _looks_like_absolute_path(value):
            raise _InvalidStoredJobError(f"absolute host path at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_portable_json(item, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise _InvalidStoredJobError(f"non-string object key at {location}")
            if _looks_like_absolute_path(key):
                raise _InvalidStoredJobError(f"absolute host path key at {location}")
            _validate_portable_json(item, location=f"{location}.{key}")
        return
    raise _InvalidStoredJobError(f"non-JSON value at {location}")


def _canonical_json(document: Any) -> str:
    _validate_portable_json(document)
    try:
        return json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise _InvalidStoredJobError("document cannot be encoded as canonical JSON") from exc


def _encode_spec(spec: JobSpecV2) -> tuple[str, JobSpecV2]:
    if not isinstance(spec, JobSpecV2):
        raise _error(
            "INVALID_PARAMETER",
            "A valid JobSpec v2 is required to create a job.",
            stage=ErrorStage.REQUEST,
        )
    try:
        encoded = _canonical_json(spec.model_dump(mode="json"))
        restored = JobSpecV2.model_validate_json(encoded)
        if _canonical_json(restored.model_dump(mode="json")) != encoded:
            raise _InvalidStoredJobError("JobSpec v2 did not survive a canonical round trip")
    except (_InvalidStoredJobError, TypeError, ValueError, ValidationError) as exc:
        raise _error(
            "INVALID_PARAMETER",
            "JobSpec v2 must be lossless portable JSON without host paths.",
            stage=ErrorStage.REQUEST,
        ) from exc
    return encoded, restored


def compute_request_fingerprint(spec: JobSpecV2) -> str:
    """Return the canonical SHA-256 request identity for ``spec``."""

    encoded, _ = _encode_spec(spec)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _summary_for_spec(spec: JobSpecV2) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "input_count": len(spec.inputs),
        "robot_id": spec.robot.robot_id,
        "backend": spec.backend,
    }
    run_mode = spec.effective_parameters.get("run_mode")
    if isinstance(run_mode, str) and run_mode:
        summary["run_mode"] = run_mode
    return summary


def _normalize_idempotency_key(value: str) -> str:
    if not isinstance(value, str) or _IDEMPOTENCY_KEY_PATTERN.fullmatch(value) is None:
        raise _error(
            "INVALID_PARAMETER",
            "The idempotency key must be 1-256 portable token characters.",
            stage=ErrorStage.REQUEST,
        )
    return value


def _normalize_fingerprint(value: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise _error(
            "INVALID_PARAMETER",
            "The request fingerprint must be a lower-case SHA-256 digest.",
            stage=ErrorStage.REQUEST,
        )
    return value


def _normalize_job_id(value: str) -> str:
    if not isinstance(value, str) or _JOB_ID_PATTERN.fullmatch(value) is None:
        raise _error(
            "INVALID_PARAMETER",
            "The job id is not a valid HHTools job identifier.",
            stage=ErrorStage.REQUEST,
        )
    return value


def _normalize_revision(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _error(
            "INVALID_PARAMETER",
            "The expected job revision must be a non-negative integer.",
            stage=ErrorStage.REQUEST,
        )
    return value


def _normalize_wait_timeout(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
        or value > _MAX_JOB_WAIT_SECONDS
    ):
        raise _error(
            "INVALID_PARAMETER",
            "The job wait timeout must be a finite number from 0 through 60 seconds.",
            stage=ErrorStage.REQUEST,
        )
    return float(value)


def _normalize_poll_after_ms(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 300_000:
        raise _error(
            "INVALID_PARAMETER",
            "The polling interval must be an integer from 0 to 300000 milliseconds.",
            stage=ErrorStage.REQUEST,
        )
    return value


def _normalize_next_action(value: NextAction | None) -> NextAction | None:
    if value is not None and not isinstance(value, NextAction):
        raise _error(
            "INVALID_PARAMETER",
            "The next action must use the NextAction contract.",
            stage=ErrorStage.REQUEST,
        )
    return value


def _parse_datetime(value: Any, *, required: bool) -> datetime | None:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise _InvalidStoredJobError("persisted timestamp has an invalid type")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise _InvalidStoredJobError("persisted timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _InvalidStoredJobError("persisted timestamp is not timezone aware")
    return parsed


def _decode_optional_model(
    payload: Any,
    model_type: type[ApiError] | type[NextAction],
) -> ApiError | NextAction | None:
    if payload is None:
        return None
    if not isinstance(payload, str):
        raise _InvalidStoredJobError("persisted optional model has an invalid type")
    document = _strict_json_loads(payload)
    if not isinstance(document, dict) or _canonical_json(document) != payload:
        raise _InvalidStoredJobError("persisted optional model is not canonical JSON")
    return model_type.model_validate(document)


def _validate_artifact(artifact: ArtifactDescriptor, *, job_id: str) -> str:
    if not isinstance(artifact, ArtifactDescriptor):
        raise _InvalidStoredJobError("artifact does not use ArtifactDescriptor")
    if artifact.job_id != job_id:
        raise _InvalidStoredJobError("artifact job id does not match its owner")
    if artifact.sha256 is None or artifact.size_bytes is None or artifact.created_at is None:
        raise _InvalidStoredJobError("artifact requires sha256, size_bytes, and created_at")
    encoded = _canonical_json(artifact.model_dump(mode="json"))
    restored = ArtifactDescriptor.model_validate_json(encoded)
    if _canonical_json(restored.model_dump(mode="json")) != encoded:
        raise _InvalidStoredJobError("artifact did not survive a canonical round trip")
    return encoded


def _normalize_artifacts(
    job_id: str,
    artifacts: Sequence[ArtifactDescriptor],
) -> tuple[ArtifactDescriptor, ...]:
    if isinstance(artifacts, str | bytes) or not isinstance(artifacts, Sequence):
        raise _error(
            "INVALID_PARAMETER",
            "Artifacts must be a non-empty sequence of ArtifactDescriptor values.",
            stage=ErrorStage.REQUEST,
        )
    if not artifacts:
        raise _error(
            "INVALID_PARAMETER",
            "At least one artifact descriptor is required.",
            stage=ErrorStage.REQUEST,
        )

    normalized: list[ArtifactDescriptor] = []
    by_id: dict[str, str] = {}
    try:
        for artifact in artifacts:
            encoded = _validate_artifact(artifact, job_id=job_id)
            existing = by_id.get(artifact.artifact_id)
            if existing is not None:
                if existing != encoded:
                    raise _InvalidStoredJobError(
                        "one request contains divergent descriptors for an artifact id"
                    )
                continue
            by_id[artifact.artifact_id] = encoded
            normalized.append(ArtifactDescriptor.model_validate_json(encoded))
    except (_InvalidStoredJobError, TypeError, ValueError, ValidationError) as exc:
        raise _error(
            "INVALID_PARAMETER",
            "Artifact descriptors must be complete, portable, and owned by the job.",
            stage=ErrorStage.REQUEST,
            details={"job_id": job_id},
        ) from exc
    return tuple(normalized)


def _decode_artifacts(payload: Any, *, job_id: str) -> tuple[ArtifactDescriptor, ...]:
    if not isinstance(payload, str):
        raise _InvalidStoredJobError("persisted artifacts have an invalid type")
    document = _strict_json_loads(payload)
    if not isinstance(document, list) or _canonical_json(document) != payload:
        raise _InvalidStoredJobError("persisted artifacts are not canonical JSON")

    result: list[ArtifactDescriptor] = []
    identities: dict[str, str] = {}
    for item in document:
        artifact = ArtifactDescriptor.model_validate(item)
        encoded = _validate_artifact(artifact, job_id=job_id)
        existing = identities.get(artifact.artifact_id)
        if existing is not None:
            raise _InvalidStoredJobError("persisted artifact ids are not unique")
        identities[artifact.artifact_id] = encoded
        result.append(artifact)
    return tuple(result)


def _encode_artifacts(artifacts: Sequence[ArtifactDescriptor]) -> str:
    return _canonical_json([artifact.model_dump(mode="json") for artifact in artifacts])


def _merge_artifacts(
    current: StoredJob,
    additions: Sequence[ArtifactDescriptor],
) -> tuple[tuple[ArtifactDescriptor, ...], bool]:
    merged = list(current.artifacts)
    by_id = {
        artifact.artifact_id: _validate_artifact(artifact, job_id=current.job_id)
        for artifact in current.artifacts
    }
    changed = False
    for artifact in additions:
        encoded = _validate_artifact(artifact, job_id=current.job_id)
        existing = by_id.get(artifact.artifact_id)
        if existing is not None:
            if existing != encoded:
                raise _error(
                    "JOB_CONFLICT",
                    "The artifact id is already bound to another descriptor.",
                    details={
                        "job_id": current.job_id,
                        "artifact_id": artifact.artifact_id,
                    },
                )
            continue
        by_id[artifact.artifact_id] = encoded
        merged.append(artifact)
        changed = True
    return tuple(merged), changed


class JobStore:
    """SQLite WAL store for immutable JobSpec v2 and atomic lifecycle facts."""

    def __init__(
        self,
        data_dir: Path,
        *,
        clock: Callable[[], datetime] | None = None,
        job_id_provider: Callable[[], str] | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._database_path = self._data_dir / "jobs.sqlite3"
        self._clock = clock or (lambda: datetime.now(UTC))
        self._job_id_provider = job_id_provider or (lambda: f"job:{uuid.uuid4().hex}")
        self._changes = threading.Condition()
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The job store directory could not be initialized.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            ) from exc
        self._initialize_database()

    @property
    def database_path(self) -> Path:
        """Return the SQLite path for deployment diagnostics."""

        return self._database_path

    def _notify_change(self) -> None:
        """Wake in-process revision waiters after a durable row mutation."""

        with self._changes:
            self._changes.notify_all()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize_database(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=NORMAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        idempotency_key TEXT NOT NULL UNIQUE,
                        request_fingerprint TEXT NOT NULL,
                        spec_sha256 TEXT NOT NULL,
                        spec_json TEXT NOT NULL,
                        state TEXT NOT NULL,
                        outcome TEXT,
                        progress_json TEXT NOT NULL,
                        summary_json TEXT NOT NULL,
                        error_json TEXT,
                        next_action_json TEXT,
                        revision INTEGER NOT NULL CHECK (revision >= 0),
                        cancel_requested INTEGER NOT NULL DEFAULT 0
                            CHECK (cancel_requested IN (0, 1)),
                        parent_job_id TEXT,
                        root_job_id TEXT,
                        attempt INTEGER NOT NULL DEFAULT 1 CHECK (attempt >= 1),
                        artifacts_json TEXT NOT NULL DEFAULT '[]',
                        submitted_at TEXT NOT NULL,
                        started_at TEXT,
                        completed_at TEXT,
                        cancel_requested_at TEXT,
                        poll_after_ms INTEGER
                            CHECK (poll_after_ms IS NULL OR
                                   (poll_after_ms >= 0 AND poll_after_ms <= 300000)),
                        FOREIGN KEY (parent_job_id) REFERENCES jobs (job_id),
                        FOREIGN KEY (root_job_id) REFERENCES jobs (job_id)
                    )
                    """
                )
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
                }
                migrations = {
                    "parent_job_id": "ALTER TABLE jobs ADD COLUMN parent_job_id TEXT",
                    "root_job_id": "ALTER TABLE jobs ADD COLUMN root_job_id TEXT",
                    "attempt": ("ALTER TABLE jobs ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1"),
                    "artifacts_json": (
                        "ALTER TABLE jobs ADD COLUMN artifacts_json TEXT NOT NULL DEFAULT '[]'"
                    ),
                }
                for column, statement in migrations.items():
                    if column not in columns:
                        connection.execute(statement)
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS jobs_state_revision ON jobs (state, revision)"
                )
                connection.execute("CREATE INDEX IF NOT EXISTS jobs_parent ON jobs (parent_job_id)")
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The job store database could not be initialized.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            ) from exc

    def _now(self) -> datetime:
        try:
            value = self._clock()
        except Exception as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The job store clock failed.",
                stage=ErrorStage.INTERNAL,
            ) from exc
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise _error(
                "INTERNAL_ERROR",
                "The job store clock must return a timezone-aware datetime.",
                stage=ErrorStage.INTERNAL,
            )
        return value

    def _new_job_id(self) -> str:
        try:
            value = self._job_id_provider()
        except Exception as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The job id provider failed.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            ) from exc
        if not isinstance(value, str) or _JOB_ID_PATTERN.fullmatch(value) is None:
            raise _error(
                "INTERNAL_ERROR",
                "The job id provider returned an invalid identifier.",
                stage=ErrorStage.INTERNAL,
            )
        return value

    @staticmethod
    def _decode_row(row: sqlite3.Row, *, created: bool = False) -> StoredJob:
        try:
            job_id = row["job_id"]
            idempotency_key = row["idempotency_key"]
            request_fingerprint = row["request_fingerprint"]
            spec_sha256 = row["spec_sha256"]
            spec_json = row["spec_json"]
            state_value = row["state"]
            outcome_value = row["outcome"]
            revision = row["revision"]
            cancel_value = row["cancel_requested"]
            parent_job_id = row["parent_job_id"]
            root_job_id = row["root_job_id"]
            attempt = row["attempt"]
            poll_after_ms = row["poll_after_ms"]

            if not isinstance(job_id, str) or _JOB_ID_PATTERN.fullmatch(job_id) is None:
                raise _InvalidStoredJobError("persisted job id is invalid")
            if (
                not isinstance(idempotency_key, str)
                or _IDEMPOTENCY_KEY_PATTERN.fullmatch(idempotency_key) is None
            ):
                raise _InvalidStoredJobError("persisted idempotency key is invalid")
            if (
                not isinstance(request_fingerprint, str)
                or _SHA256_PATTERN.fullmatch(request_fingerprint) is None
                or not isinstance(spec_sha256, str)
                or _SHA256_PATTERN.fullmatch(spec_sha256) is None
            ):
                raise _InvalidStoredJobError("persisted request digest is invalid")
            if not isinstance(spec_json, str):
                raise _InvalidStoredJobError("persisted JobSpec has an invalid type")

            spec_document = _strict_json_loads(spec_json)
            if not isinstance(spec_document, dict) or _canonical_json(spec_document) != spec_json:
                raise _InvalidStoredJobError("persisted JobSpec is not canonical JSON")
            actual_spec_sha256 = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()
            if spec_sha256 != actual_spec_sha256 or request_fingerprint != spec_sha256:
                raise _InvalidStoredJobError("persisted JobSpec identity is inconsistent")
            spec = JobSpecV2.model_validate(spec_document)

            if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
                raise _InvalidStoredJobError("persisted retry attempt is invalid")
            if parent_job_id is None:
                if root_job_id is not None or attempt != 1:
                    raise _InvalidStoredJobError("persisted root lineage is invalid")
            elif not isinstance(parent_job_id, str) or not isinstance(root_job_id, str):
                raise _InvalidStoredJobError("persisted retry lineage is invalid")
            elif (
                _JOB_ID_PATTERN.fullmatch(parent_job_id) is None
                or _JOB_ID_PATTERN.fullmatch(root_job_id) is None
                or job_id in {parent_job_id, root_job_id}
                or attempt < 2
            ):
                raise _InvalidStoredJobError("persisted retry lineage is invalid")

            artifacts = _decode_artifacts(row["artifacts_json"], job_id=job_id)

            if not isinstance(state_value, str):
                raise _InvalidStoredJobError("persisted state has an invalid type")
            state = JobState(state_value)
            outcome = None if outcome_value is None else JobOutcome(outcome_value)
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise _InvalidStoredJobError("persisted revision is invalid")

            progress_json = row["progress_json"]
            if not isinstance(progress_json, str):
                raise _InvalidStoredJobError("persisted progress has an invalid type")
            progress_document = _strict_json_loads(progress_json)
            if (
                not isinstance(progress_document, dict)
                or _canonical_json(progress_document) != progress_json
            ):
                raise _InvalidStoredJobError("persisted progress is not canonical JSON")
            progress = JobProgress.model_validate(progress_document)
            if progress.revision != revision:
                raise _InvalidStoredJobError("progress and row revisions diverge")

            summary_json = row["summary_json"]
            if not isinstance(summary_json, str):
                raise _InvalidStoredJobError("persisted summary has an invalid type")
            summary = _strict_json_loads(summary_json)
            if not isinstance(summary, dict) or _canonical_json(summary) != summary_json:
                raise _InvalidStoredJobError("persisted summary is not canonical JSON")
            if summary != _summary_for_spec(spec):
                raise _InvalidStoredJobError("persisted summary diverges from JobSpec")

            error = _decode_optional_model(row["error_json"], ApiError)
            next_action = _decode_optional_model(row["next_action_json"], NextAction)
            if error is not None and not isinstance(error, ApiError):
                raise _InvalidStoredJobError("persisted error has the wrong model type")
            if next_action is not None and not isinstance(next_action, NextAction):
                raise _InvalidStoredJobError("persisted next action has the wrong model type")

            submitted_at = _parse_datetime(row["submitted_at"], required=True)
            started_at = _parse_datetime(row["started_at"], required=False)
            completed_at = _parse_datetime(row["completed_at"], required=False)
            cancel_requested_at = _parse_datetime(row["cancel_requested_at"], required=False)
            if submitted_at is None:  # required=True; narrows the type for Pydantic
                raise _InvalidStoredJobError("persisted submission timestamp is missing")
            if progress.updated_at is None or progress.updated_at < submitted_at:
                raise _InvalidStoredJobError(
                    "persisted progress timestamp is missing or predates submission"
                )
            if cancel_value not in {0, 1}:
                raise _InvalidStoredJobError("persisted cancellation flag is invalid")
            cancel_requested = bool(cancel_value)
            if cancel_requested != (cancel_requested_at is not None):
                raise _InvalidStoredJobError("cancellation flag and timestamp diverge")
            if cancel_requested_at is not None and cancel_requested_at < submitted_at:
                raise _InvalidStoredJobError("cancellation predates submission")

            if state in _TERMINAL_STATES:
                if completed_at is None:
                    raise _InvalidStoredJobError("terminal jobs require a completion timestamp")
                if progress.updated_at > completed_at:
                    raise _InvalidStoredJobError("persisted progress timestamp follows completion")
                if cancel_requested_at is not None and cancel_requested_at > completed_at:
                    raise _InvalidStoredJobError(
                        "persisted cancellation timestamp follows completion"
                    )
            elif completed_at is not None:
                raise _InvalidStoredJobError("non-terminal job has a completion timestamp")
            if state is JobState.RUNNING and started_at is None:
                raise _InvalidStoredJobError("running jobs require a start timestamp")
            if state is JobState.QUEUED and started_at is not None:
                raise _InvalidStoredJobError("queued jobs cannot have a start timestamp")
            if state is JobState.FAILED:
                if error is None:
                    raise _InvalidStoredJobError("failed job has no error")
            elif error is not None:
                raise _InvalidStoredJobError("only failed jobs may persist an error")

            view = AgentJobView(
                job_id=job_id,
                parent_job_id=parent_job_id,
                root_job_id=root_job_id,
                attempt=attempt,
                state=state,
                outcome=outcome,
                progress=progress,
                summary=summary,
                artifacts=list(artifacts[:32]),
                artifact_count=len(artifacts),
                error=error,
                next_action=next_action,
                cancellation_requested=cancel_requested,
                cancellable=state not in _TERMINAL_STATES,
                submitted_at=submitted_at,
                started_at=started_at,
                completed_at=completed_at,
                poll_after_ms=poll_after_ms,
            )
        except (
            _InvalidStoredJobError,
            KeyError,
            TypeError,
            ValueError,
            ValidationError,
        ) as exc:
            raise _error(
                "INTERNAL_ERROR",
                "A persisted agent job is invalid.",
                stage=ErrorStage.INTERNAL,
            ) from exc
        return StoredJob(
            spec=spec,
            view=view,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            artifacts=artifacts,
            cancel_requested=cancel_requested,
            cancel_requested_at=cancel_requested_at,
            created=created,
        )

    def create(
        self,
        spec: JobSpecV2,
        *,
        idempotency_key: str,
        request_fingerprint: str | None = None,
        parent_job_id: str | None = None,
    ) -> StoredJob:
        """Atomically create a queued job or replay an identical submission.

        Reusing ``idempotency_key`` with another canonical JobSpec v2 raises
        ``JOB_CONFLICT``.  A supplied fingerprint is checked against the full
        canonical spec instead of being trusted as an arbitrary caller label.
        """

        key = _normalize_idempotency_key(idempotency_key)
        normalized_parent_id = (
            _normalize_job_id(parent_job_id) if parent_job_id is not None else None
        )
        spec_json, detached_spec = _encode_spec(spec)
        computed_fingerprint = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()
        if request_fingerprint is not None:
            supplied_fingerprint = _normalize_fingerprint(request_fingerprint)
            if supplied_fingerprint != computed_fingerprint:
                raise _error(
                    "INVALID_PARAMETER",
                    "The request fingerprint does not match JobSpec v2.",
                    stage=ErrorStage.REQUEST,
                    details={"expected_request_fingerprint": computed_fingerprint},
                )

        submitted_at = self._now()
        progress = JobProgress(
            phase="queued",
            fraction=0.0,
            revision=0,
            updated_at=submitted_at,
        )
        progress_json = _canonical_json(progress.model_dump(mode="json"))
        summary_json = _canonical_json(_summary_for_spec(detached_spec))

        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    f"{_SELECT_JOB} WHERE idempotency_key = ?",
                    (key,),
                ).fetchone()
                if existing is not None:
                    stored = self._decode_row(existing)
                    if (
                        stored.request_fingerprint != computed_fingerprint
                        or _canonical_json(stored.spec.model_dump(mode="json")) != spec_json
                        or stored.view.parent_job_id != normalized_parent_id
                    ):
                        raise _error(
                            "JOB_CONFLICT",
                            "The idempotency key is already bound to another job request.",
                            stage=ErrorStage.ADMISSION,
                            details={"job_id": stored.job_id},
                        )
                    return stored

                root_job_id: str | None = None
                attempt = 1
                if normalized_parent_id is not None:
                    parent_row = connection.execute(
                        f"{_SELECT_JOB} WHERE job_id = ?",
                        (normalized_parent_id,),
                    ).fetchone()
                    if parent_row is None:
                        raise _error(
                            "JOB_NOT_FOUND",
                            "The requested parent job does not exist.",
                            stage=ErrorStage.ADMISSION,
                            details={"parent_job_id": normalized_parent_id},
                        )
                    parent = self._decode_row(parent_row)
                    if parent.view.state not in _TERMINAL_STATES:
                        raise _error(
                            "JOB_CONFLICT",
                            "A retry can only be created from a terminal parent job.",
                            stage=ErrorStage.ADMISSION,
                            details={
                                "parent_job_id": parent.job_id,
                                "parent_state": parent.view.state.value,
                            },
                        )
                    if (
                        parent.spec.plan_id != detached_spec.plan_id
                        or parent.spec.kind != detached_spec.kind
                    ):
                        raise _error(
                            "JOB_CONFLICT",
                            "A retry must retain its parent's plan id and job kind.",
                            stage=ErrorStage.ADMISSION,
                            details={"parent_job_id": parent.job_id},
                        )
                    if (
                        parent.view.completed_at is not None
                        and submitted_at < parent.view.completed_at
                    ):
                        raise _error(
                            "INTERNAL_ERROR",
                            "The job store clock moved backwards before retry creation.",
                            stage=ErrorStage.INTERNAL,
                        )
                    root_job_id = parent.view.root_job_id or parent.job_id
                    attempt = parent.view.attempt + 1

                inserted_job_id: str | None = None
                for _ in range(8):
                    candidate = self._new_job_id()
                    try:
                        connection.execute(
                            """
                            INSERT INTO jobs (
                                job_id,
                                idempotency_key,
                                request_fingerprint,
                                spec_sha256,
                                spec_json,
                                state,
                                outcome,
                                progress_json,
                                summary_json,
                                error_json,
                                next_action_json,
                                revision,
                                cancel_requested,
                                parent_job_id,
                                root_job_id,
                                attempt,
                                artifacts_json,
                                submitted_at,
                                started_at,
                                completed_at,
                                cancel_requested_at,
                                poll_after_ms
                            ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, NULL,
                                      0, 0, ?, ?, ?, '[]', ?, NULL, NULL, NULL, NULL)
                            """,
                            (
                                candidate,
                                key,
                                computed_fingerprint,
                                computed_fingerprint,
                                spec_json,
                                JobState.QUEUED.value,
                                progress_json,
                                summary_json,
                                normalized_parent_id,
                                root_job_id,
                                attempt,
                                submitted_at.isoformat(),
                            ),
                        )
                    except sqlite3.IntegrityError:
                        collision = connection.execute(
                            "SELECT 1 FROM jobs WHERE job_id = ?",
                            (candidate,),
                        ).fetchone()
                        if collision is not None:
                            continue
                        raise
                    inserted_job_id = candidate
                    break
                if inserted_job_id is None:
                    raise _error(
                        "INTERNAL_ERROR",
                        "A unique job id could not be allocated.",
                        stage=ErrorStage.INTERNAL,
                        retryable=True,
                    )

                row = connection.execute(
                    f"{_SELECT_JOB} WHERE job_id = ?",
                    (inserted_job_id,),
                ).fetchone()
        except JobStoreError:
            raise
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The agent job could not be persisted.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            ) from exc
        if row is None:
            raise _error(
                "INTERNAL_ERROR",
                "The agent job was unavailable after persistence.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            )
        created = self._decode_row(row, created=True)
        self._notify_change()
        return created

    def get(self, job_id: str) -> StoredJob:
        """Load one fresh, fully validated job snapshot by id."""

        normalized = _normalize_job_id(job_id)
        try:
            with self._connect() as connection:
                row = connection.execute(
                    f"{_SELECT_JOB} WHERE job_id = ?",
                    (normalized,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The job store could not be read.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            ) from exc
        if row is None:
            raise _error(
                "JOB_NOT_FOUND",
                "No agent job has the requested id.",
                details={"job_id": normalized},
            )
        return self._decode_row(row)

    def get_spec(self, job_id: str) -> JobSpecV2:
        """Load a detached copy of the immutable JobSpec v2."""

        return self.get(job_id).spec

    def get_by_idempotency_key(self, idempotency_key: str) -> StoredJob:
        """Recover a previously submitted job from its idempotency key."""

        key = _normalize_idempotency_key(idempotency_key)
        try:
            with self._connect() as connection:
                row = connection.execute(
                    f"{_SELECT_JOB} WHERE idempotency_key = ?",
                    (key,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The job store could not be read.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            ) from exc
        if row is None:
            raise _error(
                "JOB_NOT_FOUND",
                "No agent job has the requested idempotency key.",
            )
        return self._decode_row(row)

    def wait_for_revision(
        self,
        job_id: str,
        *,
        after_revision: int,
        timeout: float,
    ) -> StoredJob:
        """Wait until a job advances beyond a caller-observed revision.

        Runtime ownership guarantees one process-local writer for this store.
        Holding the condition across the durable read and wait prevents a
        revision update from being missed between those two operations.
        """

        normalized_id = _normalize_job_id(job_id)
        observed_revision = _normalize_revision(after_revision)
        wait_seconds = _normalize_wait_timeout(timeout)
        deadline = time.monotonic() + wait_seconds
        with self._changes:
            while True:
                current = self.get(normalized_id)
                if observed_revision > current.revision:
                    raise _error(
                        "INVALID_PARAMETER",
                        "after_revision must be between zero and the current revision.",
                        stage=ErrorStage.REQUEST,
                        details={"current_revision": current.revision},
                    )
                if current.view.state in _TERMINAL_STATES or current.revision > observed_revision:
                    return current
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return current
                self._changes.wait(timeout=remaining)

    def list_active(self) -> list[StoredJob]:
        """Return queued and running jobs in deterministic submission order."""

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    f"{_SELECT_JOB} WHERE state IN (?, ?) ORDER BY submitted_at ASC, job_id ASC",
                    (JobState.QUEUED.value, JobState.RUNNING.value),
                ).fetchall()
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "Active jobs could not be listed.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            ) from exc
        return [self._decode_row(row) for row in rows]

    @staticmethod
    def _state(value: JobState | str) -> JobState:
        try:
            return value if isinstance(value, JobState) else JobState(value)
        except (TypeError, ValueError) as exc:
            raise _error(
                "INVALID_PARAMETER",
                "The target job state is not supported.",
                stage=ErrorStage.REQUEST,
            ) from exc

    @staticmethod
    def _outcome(value: JobOutcome | str | None) -> JobOutcome | None:
        if value is None or isinstance(value, JobOutcome):
            return value
        try:
            return JobOutcome(value)
        except (TypeError, ValueError) as exc:
            raise _error(
                "INVALID_PARAMETER",
                "The job outcome is not supported.",
                stage=ErrorStage.REQUEST,
            ) from exc

    @staticmethod
    def _validate_terminal_fields(
        state: JobState,
        outcome: JobOutcome | None,
        error: ApiError | None,
    ) -> None:
        if state is JobState.COMPLETED:
            if outcome is None or error is not None:
                raise _error(
                    "INVALID_PARAMETER",
                    "A completed job requires an outcome and cannot include an error.",
                    stage=ErrorStage.REQUEST,
                )
            return
        if outcome is not None:
            raise _error(
                "INVALID_PARAMETER",
                "Only completed jobs may include an outcome.",
                stage=ErrorStage.REQUEST,
            )
        if state is JobState.FAILED:
            if not isinstance(error, ApiError):
                raise _error(
                    "INVALID_PARAMETER",
                    "A failed job requires a structured ApiError.",
                    stage=ErrorStage.REQUEST,
                )
        elif error is not None:
            raise _error(
                "INVALID_PARAMETER",
                "Only failed jobs may include an error.",
                stage=ErrorStage.REQUEST,
            )

    @staticmethod
    def _next_progress(
        current: JobProgress,
        supplied: JobProgress | None,
        *,
        state: JobState,
        revision: int,
        updated_at: datetime,
    ) -> JobProgress:
        if supplied is not None and not isinstance(supplied, JobProgress):
            raise _error(
                "INVALID_PARAMETER",
                "Job progress must use the JobProgress contract.",
                stage=ErrorStage.REQUEST,
            )

        if supplied is None:
            document = current.model_dump()
            document["phase"] = state.value
            if state is JobState.COMPLETED:
                document["fraction"] = 1.0
            if state in _TERMINAL_STATES:
                document["eta_seconds"] = None
        else:
            document = supplied.model_dump()

        if state is JobState.COMPLETED and document["fraction"] != 1.0:
            raise _error(
                "INVALID_PARAMETER",
                "Completed job progress must have fraction 1.0.",
                stage=ErrorStage.REQUEST,
            )
        if document["fraction"] < current.fraction:
            raise _error(
                "INVALID_JOB_TRANSITION",
                "Job progress cannot move backwards.",
                details={
                    "current_fraction": current.fraction,
                    "requested_fraction": document["fraction"],
                },
            )

        old_completed = current.completed_items
        new_completed = document["completed_items"]
        if old_completed is not None and new_completed is None:
            document["completed_items"] = old_completed
            new_completed = old_completed
        if (
            old_completed is not None
            and new_completed is not None
            and new_completed < old_completed
        ):
            raise _error(
                "INVALID_JOB_TRANSITION",
                "Completed item count cannot move backwards.",
            )
        if current.total_items is not None:
            if document["total_items"] is None:
                document["total_items"] = current.total_items
            elif document["total_items"] != current.total_items:
                raise _error(
                    "INVALID_JOB_TRANSITION",
                    "A job's total item count cannot change once known.",
                )

        document["revision"] = revision
        document["updated_at"] = updated_at
        if state in _TERMINAL_STATES:
            document["eta_seconds"] = None
        try:
            return JobProgress.model_validate(document)
        except ValidationError as exc:
            raise _error(
                "INVALID_PARAMETER",
                "The requested job progress is invalid.",
                stage=ErrorStage.REQUEST,
            ) from exc

    @staticmethod
    def _encode_optional_model(value: ApiError | NextAction | None) -> str | None:
        if value is None:
            return None
        try:
            return _canonical_json(value.model_dump(mode="json"))
        except (_InvalidStoredJobError, TypeError, ValueError) as exc:
            raise _error(
                "INVALID_PARAMETER",
                "The structured job metadata must be finite JSON.",
                stage=ErrorStage.REQUEST,
            ) from exc

    @staticmethod
    def _assert_revision(current: StoredJob, expected_revision: int) -> None:
        expected = _normalize_revision(expected_revision)
        if current.revision != expected:
            raise _error(
                "JOB_CONFLICT",
                "The job changed after the caller's last revision.",
                details={
                    "job_id": current.job_id,
                    "expected_revision": expected,
                    "current_revision": current.revision,
                },
            )

    @staticmethod
    def _updated_timestamps(
        current: StoredJob,
        target_state: JobState,
        now: datetime,
    ) -> tuple[datetime | None, datetime | None]:
        started_at = current.view.started_at
        if target_state is JobState.RUNNING and started_at is None:
            started_at = now
        completed_at = now if target_state in _TERMINAL_STATES else None
        return started_at, completed_at

    @staticmethod
    def _assert_clock_order(current: StoredJob, now: datetime) -> None:
        baseline = current.view.started_at or current.view.submitted_at
        if current.view.progress.updated_at is not None:
            baseline = max(baseline, current.view.progress.updated_at)
        if current.cancel_requested_at is not None:
            baseline = max(baseline, current.cancel_requested_at)
        if now < baseline:
            raise _error(
                "INTERNAL_ERROR",
                "The job store clock moved backwards.",
                stage=ErrorStage.INTERNAL,
            )

    @staticmethod
    def _replace_row(
        connection: sqlite3.Connection,
        current: StoredJob,
        *,
        state: JobState,
        outcome: JobOutcome | None,
        progress: JobProgress,
        error: ApiError | None,
        next_action: NextAction | None,
        started_at: datetime | None,
        completed_at: datetime | None,
        poll_after_ms: int | None,
        artifacts: Sequence[ArtifactDescriptor] | None = None,
        cancel_requested: bool | None = None,
        cancel_requested_at: datetime | None = None,
    ) -> None:
        requested = current.cancel_requested if cancel_requested is None else cancel_requested
        requested_at = (
            current.cancel_requested_at if cancel_requested is None else cancel_requested_at
        )
        stored_artifacts = current.artifacts if artifacts is None else artifacts
        cursor = connection.execute(
            """
            UPDATE jobs
            SET state = ?,
                outcome = ?,
                progress_json = ?,
                error_json = ?,
                next_action_json = ?,
                revision = ?,
                cancel_requested = ?,
                artifacts_json = ?,
                started_at = ?,
                completed_at = ?,
                cancel_requested_at = ?,
                poll_after_ms = ?
            WHERE job_id = ? AND revision = ?
            """,
            (
                state.value,
                outcome.value if outcome is not None else None,
                _canonical_json(progress.model_dump(mode="json")),
                JobStore._encode_optional_model(error),
                JobStore._encode_optional_model(next_action),
                progress.revision,
                int(requested),
                _encode_artifacts(stored_artifacts),
                started_at.isoformat() if started_at is not None else None,
                completed_at.isoformat() if completed_at is not None else None,
                requested_at.isoformat() if requested_at is not None else None,
                poll_after_ms,
                current.job_id,
                current.revision,
            ),
        )
        if cursor.rowcount != 1:
            raise _error(
                "JOB_CONFLICT",
                "The job changed during the atomic update.",
                details={"job_id": current.job_id},
            )

    def transition(
        self,
        job_id: str,
        *,
        expected_revision: int,
        state: JobState | str,
        progress: JobProgress | None = None,
        outcome: JobOutcome | str | None = None,
        error: ApiError | None = None,
        next_action: NextAction | None = None,
        poll_after_ms: int | None = None,
        artifacts: Sequence[ArtifactDescriptor] | None = None,
    ) -> StoredJob:
        """Atomically apply one legal lifecycle transition.

        The caller supplies the revision it observed.  The store owns the next
        revision and progress timestamp, preventing workers from overwriting a
        newer cancellation or terminal result.
        """

        normalized_id = _normalize_job_id(job_id)
        target_state = self._state(state)
        normalized_outcome = self._outcome(outcome)
        normalized_next_action = _normalize_next_action(next_action)
        normalized_poll_after_ms = _normalize_poll_after_ms(poll_after_ms)
        normalized_artifacts: tuple[ArtifactDescriptor, ...] = ()
        if artifacts is not None:
            if isinstance(artifacts, str | bytes) or not isinstance(artifacts, Sequence):
                normalized_artifacts = _normalize_artifacts(normalized_id, artifacts)
            elif artifacts:
                normalized_artifacts = _normalize_artifacts(normalized_id, artifacts)
        self._validate_terminal_fields(target_state, normalized_outcome, error)

        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    f"{_SELECT_JOB} WHERE job_id = ?",
                    (normalized_id,),
                ).fetchone()
                if row is None:
                    raise _error(
                        "JOB_NOT_FOUND",
                        "No agent job has the requested id.",
                        details={"job_id": normalized_id},
                    )
                current = self._decode_row(row)
                self._assert_revision(current, expected_revision)
                if target_state not in _ALLOWED_TRANSITIONS[current.view.state]:
                    raise _error(
                        "INVALID_JOB_TRANSITION",
                        "The requested job state transition is not legal.",
                        details={
                            "job_id": current.job_id,
                            "current_state": current.view.state.value,
                            "requested_state": target_state.value,
                        },
                    )
                if (
                    current.cancel_requested
                    and current.view.state is JobState.QUEUED
                    and target_state is JobState.RUNNING
                ):
                    raise _error(
                        "JOB_CONFLICT",
                        "A queued job with a cancellation request cannot be started.",
                        details={"job_id": current.job_id},
                    )

                merged_artifacts = current.artifacts
                if normalized_artifacts:
                    merged_artifacts, _ = _merge_artifacts(current, normalized_artifacts)

                now = self._now()
                self._assert_clock_order(current, now)
                next_revision = current.revision + 1
                next_progress = self._next_progress(
                    current.view.progress,
                    progress,
                    state=target_state,
                    revision=next_revision,
                    updated_at=now,
                )
                started_at, completed_at = self._updated_timestamps(current, target_state, now)
                self._replace_row(
                    connection,
                    current,
                    state=target_state,
                    outcome=normalized_outcome,
                    progress=next_progress,
                    error=error,
                    next_action=normalized_next_action,
                    started_at=started_at,
                    completed_at=completed_at,
                    poll_after_ms=normalized_poll_after_ms,
                    artifacts=merged_artifacts,
                )
                updated_row = connection.execute(
                    f"{_SELECT_JOB} WHERE job_id = ?",
                    (normalized_id,),
                ).fetchone()
        except JobStoreError:
            raise
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The job state could not be updated.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            ) from exc
        if updated_row is None:
            raise _error(
                "INTERNAL_ERROR",
                "The job disappeared after its state update.",
                stage=ErrorStage.INTERNAL,
            )
        updated = self._decode_row(updated_row)
        self._notify_change()
        return updated

    def update_progress(
        self,
        job_id: str,
        *,
        expected_revision: int,
        progress: JobProgress,
        next_action: NextAction | None = None,
        poll_after_ms: int | None = None,
    ) -> StoredJob:
        """Atomically update progress for a queued or running job."""

        normalized_id = _normalize_job_id(job_id)
        normalized_next_action = _normalize_next_action(next_action)
        normalized_poll_after_ms = _normalize_poll_after_ms(poll_after_ms)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    f"{_SELECT_JOB} WHERE job_id = ?",
                    (normalized_id,),
                ).fetchone()
                if row is None:
                    raise _error(
                        "JOB_NOT_FOUND",
                        "No agent job has the requested id.",
                        details={"job_id": normalized_id},
                    )
                current = self._decode_row(row)
                self._assert_revision(current, expected_revision)
                if current.view.state in _TERMINAL_STATES:
                    raise _error(
                        "INVALID_JOB_TRANSITION",
                        "Terminal job progress is immutable.",
                        details={"job_id": current.job_id},
                    )

                now = self._now()
                self._assert_clock_order(current, now)
                next_progress = self._next_progress(
                    current.view.progress,
                    progress,
                    state=current.view.state,
                    revision=current.revision + 1,
                    updated_at=now,
                )
                self._replace_row(
                    connection,
                    current,
                    state=current.view.state,
                    outcome=None,
                    progress=next_progress,
                    error=None,
                    next_action=normalized_next_action,
                    started_at=current.view.started_at,
                    completed_at=None,
                    poll_after_ms=normalized_poll_after_ms,
                )
                updated_row = connection.execute(
                    f"{_SELECT_JOB} WHERE job_id = ?",
                    (normalized_id,),
                ).fetchone()
        except JobStoreError:
            raise
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "Job progress could not be updated.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            ) from exc
        if updated_row is None:
            raise _error(
                "INTERNAL_ERROR",
                "The job disappeared after its progress update.",
                stage=ErrorStage.INTERNAL,
            )
        updated = self._decode_row(updated_row)
        self._notify_change()
        return updated

    def attach_artifacts(
        self,
        job_id: str,
        *,
        expected_revision: int,
        artifacts: Sequence[ArtifactDescriptor],
    ) -> StoredJob:
        """Atomically attach complete descriptors to an active job.

        Artifact ids are immutable within a job.  Reattaching byte-for-byte
        equivalent descriptors is an idempotent no-op, including a transport
        retry that still carries the revision from before the first attach.
        """

        normalized_id = _normalize_job_id(job_id)
        _normalize_revision(expected_revision)
        normalized_artifacts = _normalize_artifacts(normalized_id, artifacts)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    f"{_SELECT_JOB} WHERE job_id = ?",
                    (normalized_id,),
                ).fetchone()
                if row is None:
                    raise _error(
                        "JOB_NOT_FOUND",
                        "No agent job has the requested id.",
                        details={"job_id": normalized_id},
                    )
                current = self._decode_row(row)
                if current.view.state in _TERMINAL_STATES:
                    raise _error(
                        "INVALID_JOB_TRANSITION",
                        "Artifacts can only be attached while a job is active.",
                        details={"job_id": current.job_id},
                    )

                merged_artifacts, changed = _merge_artifacts(current, normalized_artifacts)
                if not changed:
                    return current
                self._assert_revision(current, expected_revision)

                now = self._now()
                self._assert_clock_order(current, now)
                next_progress = current.view.progress.model_copy(
                    update={"revision": current.revision + 1, "updated_at": now}
                )
                self._replace_row(
                    connection,
                    current,
                    state=current.view.state,
                    outcome=None,
                    progress=next_progress,
                    error=None,
                    next_action=current.view.next_action,
                    started_at=current.view.started_at,
                    completed_at=None,
                    poll_after_ms=current.view.poll_after_ms,
                    artifacts=merged_artifacts,
                )
                updated_row = connection.execute(
                    f"{_SELECT_JOB} WHERE job_id = ?",
                    (normalized_id,),
                ).fetchone()
        except JobStoreError:
            raise
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "Artifact descriptors could not be attached to the job.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            ) from exc
        if updated_row is None:
            raise _error(
                "INTERNAL_ERROR",
                "The job disappeared after its artifact update.",
                stage=ErrorStage.INTERNAL,
            )
        updated = self._decode_row(updated_row)
        self._notify_change()
        return updated

    def request_cancel(
        self,
        job_id: str,
        *,
        expected_revision: int | None = None,
    ) -> StoredJob:
        """Persist a cooperative cancellation request without faking success.

        Queued/running state is retained until a scheduler or worker observes
        the flag and performs a legal transition to ``cancelled``.  Repeating
        an already-recorded request is idempotent, including a retry carrying
        the pre-request revision.
        """

        normalized_id = _normalize_job_id(job_id)
        if expected_revision is not None:
            _normalize_revision(expected_revision)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    f"{_SELECT_JOB} WHERE job_id = ?",
                    (normalized_id,),
                ).fetchone()
                if row is None:
                    raise _error(
                        "JOB_NOT_FOUND",
                        "No agent job has the requested id.",
                        details={"job_id": normalized_id},
                    )
                current = self._decode_row(row)
                if current.cancel_requested or current.view.state is JobState.CANCELLED:
                    return current
                if current.view.state in {JobState.COMPLETED, JobState.FAILED}:
                    raise _error(
                        "JOB_CANCEL_UNSUPPORTED",
                        "A completed or failed job can no longer be cancelled.",
                        details={
                            "job_id": current.job_id,
                            "state": current.view.state.value,
                        },
                    )
                if expected_revision is not None:
                    self._assert_revision(current, expected_revision)

                now = self._now()
                self._assert_clock_order(current, now)
                next_progress = current.view.progress.model_copy(
                    update={"revision": current.revision + 1, "updated_at": now}
                )
                self._replace_row(
                    connection,
                    current,
                    state=current.view.state,
                    outcome=None,
                    progress=next_progress,
                    error=None,
                    next_action=current.view.next_action,
                    started_at=current.view.started_at,
                    completed_at=None,
                    poll_after_ms=current.view.poll_after_ms,
                    cancel_requested=True,
                    cancel_requested_at=now,
                )
                updated_row = connection.execute(
                    f"{_SELECT_JOB} WHERE job_id = ?",
                    (normalized_id,),
                ).fetchone()
        except JobStoreError:
            raise
        except sqlite3.Error as exc:
            raise _error(
                "INTERNAL_ERROR",
                "The cancellation request could not be persisted.",
                stage=ErrorStage.INTERNAL,
                retryable=True,
            ) from exc
        if updated_row is None:
            raise _error(
                "INTERNAL_ERROR",
                "The job disappeared after its cancellation request.",
                stage=ErrorStage.INTERNAL,
            )
        updated = self._decode_row(updated_row)
        self._notify_change()
        return updated


__all__ = [
    "JobStore",
    "JobStoreError",
    "StoredJob",
    "compute_request_fingerprint",
]
