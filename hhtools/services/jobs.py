"""Persistent, idempotent Agent job lifecycle orchestration.

``JobManager`` is the control-plane bridge between immutable retarget plans,
the shared admission scheduler, an injected executor, and managed artifacts.
It never implements IK, calibration, or robot mathematics.  A solver adapter
receives the exact JobSpec v2 plus progress/cancellation callbacks and returns
one explicit semantic outcome.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from hhtools.contracts import (
    AgentJobView,
    ApiError,
    ArtifactDescriptor,
    ErrorStage,
    EvaluationReport,
    FailureItem,
    FailureReport,
    JobManifest,
    JobOutcome,
    JobProgress,
    JobQueueView,
    JobSpecV2,
    JobState,
    NextAction,
    SchedulerMode,
)

from .admission import (
    AdmissionClosedError,
    AdmissionQueueFullError,
    AdmissionScheduler,
    ScheduledHandle,
)
from .artifacts import ArtifactStore, ArtifactStoreError, StoredArtifact
from .job_store import JobStore, JobStoreError, StoredJob
from .retarget import RetargetService, RetargetServiceError

_log = logging.getLogger(__name__)
_ACTIVE_STATES = frozenset({JobState.QUEUED, JobState.RUNNING})
_TERMINAL_STATES = frozenset({JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED})
_MAX_COMPACT_SUMMARY_BYTES = 32 * 1024
_DEFAULT_POLL_AFTER_MS = 1_500
_MUTATION_LOCK_STRIPES = 64
_MAX_ARTIFACT_PAGE_SIZE = 500


class JobManagerError(RuntimeError):
    """Expected job-service failure with a transport-neutral error body."""

    def __init__(self, error: ApiError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")

    @property
    def api_error(self) -> ApiError:
        return self.error

    @property
    def code(self) -> str:
        return self.error.code


class JobCancelledError(RuntimeError):
    """Cooperative executor signal acknowledging a cancellation request."""


class JobExecutionError(RuntimeError):
    """Structured expected failure raised by an injected executor."""

    def __init__(self, error: ApiError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")


@dataclass(frozen=True, slots=True)
class JobExecutionResult:
    """Small terminal result returned by a solver adapter.

    Large outputs must be published through ``JobExecutionContext`` and are
    represented here only by managed ``ArtifactDescriptor`` instances.
    """

    outcome: JobOutcome
    summary: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)
    evaluation_summary: str | None = None
    evaluation_metrics: Mapping[str, Any] = field(default_factory=dict)
    evaluation_checks: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    failures: Sequence[FailureItem | Mapping[str, Any]] = field(default_factory=tuple)
    execution_provenance: Mapping[str, Any] = field(default_factory=dict)
    next_action: NextAction | None = None


class JobExecutor(Protocol):
    """Injected solver adapter; implementations remain outside JobManager."""

    def __call__(
        self,
        spec: JobSpecV2,
        context: JobExecutionContext,
    ) -> JobExecutionResult: ...


def _error(
    code: str,
    message: str,
    *,
    stage: ErrorStage = ErrorStage.EXECUTION,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
    next_action: NextAction | None = None,
) -> JobManagerError:
    return JobManagerError(
        ApiError(
            code=code,
            message=message,
            retryable=retryable,
            stage=stage,
            details=dict(details or {}),
            next_action=next_action,
        )
    )


def _wrap_service_error(error: ApiError) -> JobManagerError:
    # JSON round-tripping detaches nested ``details``/``parameters`` mappings
    # before the error crosses another adapter boundary.
    return JobManagerError(ApiError.model_validate_json(error.model_dump_json()))


def _compact_summary(
    base: Mapping[str, Any],
    additions: Mapping[str, str | int | float | bool | None],
) -> dict[str, Any]:
    summary = dict(base)
    for key, value in additions.items():
        if not isinstance(key, str) or not key or len(key) > 128:
            raise _error(
                "INVALID_PARAMETER",
                "Execution summary keys must be short strings.",
                stage=ErrorStage.EXECUTION,
            )
        if isinstance(value, str) and len(value) > 4_096:
            raise _error(
                "INVALID_PARAMETER",
                "Execution summary strings are too large.",
                stage=ErrorStage.EXECUTION,
            )
        summary[key] = value
    try:
        encoded = json.dumps(
            summary,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise _error(
            "INVALID_PARAMETER",
            "Execution summary values must be finite compact JSON scalars.",
            stage=ErrorStage.EXECUTION,
        ) from exc
    if len(encoded.encode("utf-8")) > _MAX_COMPACT_SUMMARY_BYTES:
        raise _error(
            "INVALID_PARAMETER",
            "The compact execution summary is too large.",
            stage=ErrorStage.EXECUTION,
        )
    return summary


def _failure_item(value: FailureItem | Mapping[str, Any]) -> FailureItem:
    try:
        if isinstance(value, FailureItem):
            return FailureItem.model_validate_json(value.model_dump_json())
        return FailureItem.model_validate(dict(value))
    except (TypeError, ValueError, ValidationError) as exc:
        raise _error(
            "INVALID_PARAMETER",
            "The executor returned an invalid structured failure item.",
            stage=ErrorStage.EXECUTION,
        ) from exc


class JobExecutionContext:
    """Safe callbacks and managed artifact publication for one executor call."""

    def __init__(
        self,
        *,
        job_id: str,
        spec: JobSpecV2,
        artifact_store: ArtifactStore,
        cancellation_event: threading.Event,
        progress_callback: Any,
        artifact_metadata: Mapping[str, Any] | None = None,
        shared_artifacts: list[ArtifactDescriptor] | None = None,
        shared_artifact_lock: threading.Lock | None = None,
    ) -> None:
        self.job_id = job_id
        self.spec = JobSpecV2.model_validate_json(spec.model_dump_json())
        self._artifact_store = artifact_store
        self._cancellation_event = cancellation_event
        self._progress_callback = progress_callback
        self._artifact_metadata = dict(artifact_metadata or {})
        self._artifacts = shared_artifacts if shared_artifacts is not None else []
        self._artifact_lock = shared_artifact_lock or threading.Lock()

    def for_child(
        self,
        spec: JobSpecV2,
        *,
        progress_callback: Any,
        artifact_metadata: Mapping[str, Any],
    ) -> JobExecutionContext:
        """Create a child view sharing cancellation and canonical artifact membership."""

        return JobExecutionContext(
            job_id=self.job_id,
            spec=spec,
            artifact_store=self._artifact_store,
            cancellation_event=self._cancellation_event,
            progress_callback=progress_callback,
            artifact_metadata={**self._artifact_metadata, **dict(artifact_metadata)},
            shared_artifacts=self._artifacts,
            shared_artifact_lock=self._artifact_lock,
        )

    @property
    def cancellation_requested(self) -> bool:
        """Return the local cooperative cancellation signal."""

        return self._cancellation_event.is_set()

    def raise_if_cancelled(self) -> None:
        """Acknowledge cancellation at an executor-defined safe boundary."""

        if self.cancellation_requested:
            raise JobCancelledError("the job was cancelled at a safe execution boundary")

    def report_progress(
        self,
        *,
        phase: str,
        fraction: float,
        completed_items: int | None = None,
        total_items: int | None = None,
        message: str | None = None,
        eta_seconds: float | None = None,
        poll_after_ms: int | None = None,
    ) -> AgentJobView:
        """Publish one monotonic compact progress snapshot."""

        progress = JobProgress(
            phase=phase,
            fraction=fraction,
            completed_items=completed_items,
            total_items=total_items,
            message=message,
            eta_seconds=eta_seconds,
        )
        return self._progress_callback(progress, poll_after_ms)

    def publish_bytes(
        self,
        *,
        kind: str,
        payload: bytes,
        format: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactDescriptor:
        descriptor = self._artifact_store.put_bytes(
            job_id=self.job_id,
            kind=kind,
            payload=payload,
            format=format,
            media_type=media_type,
            metadata={**dict(metadata or {}), **self._artifact_metadata},
        )
        return self._remember(descriptor)

    def publish_json(
        self,
        *,
        kind: str,
        document: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactDescriptor:
        descriptor = self._artifact_store.put_json(
            job_id=self.job_id,
            kind=kind,
            document=document,
            metadata={**dict(metadata or {}), **self._artifact_metadata},
        )
        return self._remember(descriptor)

    def publish_file(
        self,
        *,
        kind: str,
        source: Path,
        format: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactDescriptor:
        descriptor = self._artifact_store.put_file(
            job_id=self.job_id,
            kind=kind,
            source=source,
            format=format,
            media_type=media_type,
            metadata={**dict(metadata or {}), **self._artifact_metadata},
        )
        return self._remember(descriptor)

    def _remember(self, descriptor: ArtifactDescriptor) -> ArtifactDescriptor:
        with self._artifact_lock:
            if all(item.artifact_id != descriptor.artifact_id for item in self._artifacts):
                self._artifacts.append(descriptor)
        return descriptor

    def published_artifacts(self) -> tuple[ArtifactDescriptor, ...]:
        with self._artifact_lock:
            return tuple(self._artifacts)

    def get_published_artifact(self, artifact_id: str) -> StoredArtifact:
        """Resolve verified bytes only for an artifact published through this context."""

        with self._artifact_lock:
            if all(item.artifact_id != artifact_id for item in self._artifacts):
                raise ArtifactStoreError(
                    ApiError(
                        code="ARTIFACT_NOT_FOUND",
                        message="The execution context did not publish the requested artifact.",
                        stage=ErrorStage.ARTIFACT,
                    )
                )
        return self._artifact_store.get(artifact_id, verify=True)


class JobManager:
    """Durable job lifecycle facade over one shared admission scheduler."""

    def __init__(
        self,
        job_store: JobStore,
        artifact_store: ArtifactStore,
        retarget_service: RetargetService,
        scheduler: AdmissionScheduler,
        *,
        executor: JobExecutor | None = None,
        recover_interrupted: bool = True,
    ) -> None:
        self._job_store = job_store
        self._artifact_store = artifact_store
        self._retarget_service = retarget_service
        self._scheduler = scheduler
        self._executor = executor
        self._submission_lock = threading.RLock()
        self._runtime_lock = threading.Lock()
        # Lifecycle writes for one job must not race between the worker,
        # progress callbacks, API cancellation, and scheduler cancellation.
        # Fixed stripes avoid an unbounded per-job lock registry; JobStore CAS
        # remains authoritative for other processes and manager instances.
        self._mutation_locks = tuple(threading.RLock() for _ in range(_MUTATION_LOCK_STRIPES))
        self._handles: dict[str, ScheduledHandle] = {}
        self._cancellation_events: dict[str, threading.Event] = {}
        if recover_interrupted:
            self._recover_interrupted()

    @property
    def execution_available(self) -> bool:
        return self._executor is not None

    @contextmanager
    def _mutating(self, job_id: str) -> Iterator[None]:
        """Serialize lifecycle mutations for one job in this process."""

        lock = self._mutation_locks[hash(job_id) % len(self._mutation_locks)]
        with lock:
            yield

    def _existing_submission(
        self,
        *,
        idempotency_key: str,
        plan_id: str,
        parent_job_id: str | None,
    ) -> StoredJob | None:
        try:
            stored = self._job_store.get_by_idempotency_key(idempotency_key)
        except JobStoreError as exc:
            if exc.code == "JOB_NOT_FOUND":
                return None
            raise _wrap_service_error(exc.api_error) from exc
        if stored.spec.plan_id != plan_id or stored.view.parent_job_id != parent_job_id:
            raise _error(
                "JOB_CONFLICT",
                "The idempotency key is already bound to another job request.",
                stage=ErrorStage.ADMISSION,
                details={"job_id": stored.job_id},
            )
        return stored

    def start_job(
        self,
        plan_id: str,
        *,
        idempotency_key: str,
        parent_job_id: str | None = None,
    ) -> AgentJobView:
        """Create at most one admitted job for one immutable workflow plan."""

        with self._submission_lock:
            existing = self._existing_submission(
                idempotency_key=idempotency_key,
                plan_id=plan_id,
                parent_job_id=parent_job_id,
            )
            if existing is not None:
                return self._project_view(existing)
            if self._executor is None:
                raise _error(
                    "BACKEND_UNAVAILABLE",
                    "No Agent retarget executor is configured in this process.",
                    stage=ErrorStage.ADMISSION,
                )
            try:
                spec = self._retarget_service.get_job_spec(plan_id)
            except RetargetServiceError as exc:
                raise _wrap_service_error(exc.api_error) from exc
            try:
                reservation = self._scheduler.reserve()
            except AdmissionQueueFullError as exc:
                raise _error(
                    "QUEUE_FULL",
                    "The configured job waiting queue is full.",
                    stage=ErrorStage.ADMISSION,
                    retryable=True,
                    next_action=NextAction(
                        actor="agent",
                        action="retry_later",
                        parameters={"poll_after_ms": 2_000},
                    ),
                ) from exc
            except AdmissionClosedError as exc:
                raise _error(
                    "SCHEDULER_UNAVAILABLE",
                    "The job scheduler is shutting down.",
                    stage=ErrorStage.ADMISSION,
                    retryable=True,
                ) from exc

            submitted = False
            created: StoredJob | None = None
            try:
                try:
                    created = self._job_store.create(
                        spec,
                        idempotency_key=idempotency_key,
                        parent_job_id=parent_job_id,
                    )
                except JobStoreError as exc:
                    raise _wrap_service_error(exc.api_error) from exc
                if not created.created:
                    reservation.cancel()
                    submitted = True
                    if (
                        created.spec.plan_id != plan_id
                        or created.view.parent_job_id != parent_job_id
                    ):
                        raise _error(
                            "JOB_CONFLICT",
                            "The idempotency key is already bound to another job request.",
                            stage=ErrorStage.ADMISSION,
                            details={"job_id": created.job_id},
                        )
                    return self._project_view(created)

                spec_artifact = self._artifact_store.put_json(
                    job_id=created.job_id,
                    kind="job_spec",
                    document=spec.model_dump(mode="json"),
                    metadata={"schema_version": "2"},
                )
                created = self._job_store.attach_artifacts(
                    created.job_id,
                    expected_revision=created.revision,
                    artifacts=[spec_artifact],
                )
                cancellation_event = threading.Event()
                with self._runtime_lock:
                    self._cancellation_events[created.job_id] = cancellation_event

                handle = reservation.submit(
                    lambda: self._run_job(created.job_id),
                    on_cancel=lambda reason: self._on_scheduler_cancel(
                        created.job_id,
                        reason,
                    ),
                )
                with self._runtime_lock:
                    latest = self._job_store.get(created.job_id)
                    if latest.view.state in _ACTIVE_STATES:
                        self._handles[created.job_id] = handle
                submitted = True
                return self._project_view(self._job_store.get(created.job_id))
            except (ArtifactStoreError, JobStoreError) as exc:
                error = exc.api_error
                if created is not None and created.created:
                    self._fail_before_execution(created.job_id, error)
                raise _wrap_service_error(error) from exc
            except AdmissionClosedError as exc:
                if created is not None and created.created:
                    self._fail_before_execution(
                        created.job_id,
                        ApiError(
                            code="SCHEDULER_UNAVAILABLE",
                            message="The scheduler closed before the job could start.",
                            retryable=True,
                            stage=ErrorStage.ADMISSION,
                        ),
                    )
                raise _error(
                    "SCHEDULER_UNAVAILABLE",
                    "The scheduler closed before the job could start.",
                    stage=ErrorStage.ADMISSION,
                    retryable=True,
                ) from exc
            finally:
                if not submitted:
                    reservation.cancel()

    def start_retarget(
        self,
        plan_id: str,
        *,
        idempotency_key: str,
        parent_job_id: str | None = None,
    ) -> AgentJobView:
        """Compatibility alias for clients created before generic job start."""

        return self.start_job(
            plan_id,
            idempotency_key=idempotency_key,
            parent_job_id=parent_job_id,
        )

    def _project_polled_job(
        self,
        stored: StoredJob,
        *,
        after_revision: int | None,
    ) -> AgentJobView:
        """Apply the shared revision contract to an already-authorized job."""

        if after_revision is not None and (
            isinstance(after_revision, bool)
            or not isinstance(after_revision, int)
            or after_revision < 0
            or after_revision > stored.revision
        ):
            raise _error(
                "INVALID_PARAMETER",
                "after_revision must be between zero and the current revision.",
                stage=ErrorStage.REQUEST,
                details={"current_revision": stored.revision},
            )
        return self._project_view(
            stored,
            unchanged=after_revision is not None and after_revision == stored.revision,
        )

    def get_job(
        self,
        job_id: str,
        *,
        after_revision: int | None = None,
    ) -> AgentJobView:
        """Return one compact view; large output bytes remain in artifacts."""

        try:
            stored = self._job_store.get(job_id)
        except JobStoreError as exc:
            raise _wrap_service_error(exc.api_error) from exc
        return self._project_polled_job(stored, after_revision=after_revision)

    def wait_job(
        self,
        job_id: str,
        *,
        after_revision: int,
        timeout: float = 30.0,
    ) -> AgentJobView:
        """Wait for a revision change or terminal state and return a compact view."""

        try:
            stored = self._job_store.wait_for_revision(
                job_id,
                after_revision=after_revision,
                timeout=timeout,
            )
        except JobStoreError as exc:
            raise _wrap_service_error(exc.api_error) from exc
        return self._project_polled_job(stored, after_revision=after_revision)

    def lookup_job(
        self,
        plan_id: str,
        *,
        idempotency_key: str,
        after_revision: int | None = None,
    ) -> AgentJobView:
        """Recover one known submission without exposing a global job listing."""

        try:
            stored = self._job_store.get_by_idempotency_key(idempotency_key)
        except JobStoreError as exc:
            raise _wrap_service_error(exc.api_error) from exc
        if stored.spec.plan_id != plan_id:
            raise _error(
                "JOB_CONFLICT",
                "The idempotency key is bound to another immutable plan.",
                stage=ErrorStage.REQUEST,
            )
        return self._project_polled_job(stored, after_revision=after_revision)

    def list_artifacts(
        self,
        job_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> list[ArtifactDescriptor]:
        """List only descriptors canonically attached to one job.

        ``ArtifactStore`` may retain immutable candidates that were written
        before a failed lifecycle CAS or a process interruption.  JobStore's
        ``artifacts_json`` is the authorization and membership boundary.
        """

        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > _MAX_ARTIFACT_PAGE_SIZE
        ):
            raise _error(
                "INVALID_PARAMETER",
                "Artifact pagination requires offset >= 0 and limit between 1 and 500.",
                stage=ErrorStage.REQUEST,
            )
        try:
            stored = self._job_store.get(job_id)
        except JobStoreError as exc:
            raise _wrap_service_error(exc.api_error) from exc
        return list(stored.artifacts[offset : offset + limit])

    def get_artifact(
        self,
        job_id: str,
        artifact_id: str,
        *,
        verify: bool = False,
    ) -> StoredArtifact:
        """Resolve one managed artifact after canonical membership checks."""

        try:
            stored_job = self._job_store.get(job_id)
        except JobStoreError as exc:
            raise _wrap_service_error(exc.api_error) from exc
        descriptor = next(
            (item for item in stored_job.artifacts if item.artifact_id == artifact_id),
            None,
        )
        if descriptor is None:
            # Do not query ArtifactStore first: doing so would reveal whether a
            # guessed id names an unbound candidate or another job's artifact.
            raise _error(
                "ARTIFACT_NOT_FOUND",
                "The job has no artifact with the requested id.",
                stage=ErrorStage.ARTIFACT,
            )
        try:
            stored_artifact = self._artifact_store.get(artifact_id, verify=verify)
        except ArtifactStoreError as exc:
            raise _wrap_service_error(exc.api_error) from exc
        if stored_artifact.descriptor != descriptor:
            raise _error(
                "INTERNAL_ERROR",
                "The managed artifact descriptor differs from its canonical job binding.",
                stage=ErrorStage.ARTIFACT,
                retryable=True,
            )
        return stored_artifact

    def cancel_job(self, job_id: str) -> AgentJobView:
        """Request truthful queued or cooperative running cancellation."""

        with self._submission_lock, self._mutating(job_id):
            try:
                requested = self._job_store.request_cancel(job_id)
            except JobStoreError as exc:
                raise _wrap_service_error(exc.api_error) from exc
            with self._runtime_lock:
                event = self._cancellation_events.get(job_id)
                handle = self._handles.get(job_id)
            if event is not None:
                event.set()
            if requested.view.state is JobState.QUEUED and (handle is None or handle.cancel()):
                self._finalize_cancelled(job_id, context=None)
            return self._project_view(self._job_store.get(job_id))

    def retry_job(self, job_id: str, *, idempotency_key: str) -> AgentJobView:
        """Create one explicit child attempt without mutating its parent."""

        try:
            parent = self._job_store.get(job_id)
        except JobStoreError as exc:
            raise _wrap_service_error(exc.api_error) from exc
        if parent.view.state not in _TERMINAL_STATES:
            raise _error(
                "INVALID_JOB_TRANSITION",
                "Only a terminal job can be retried.",
                details={"job_id": parent.job_id, "state": parent.view.state.value},
            )
        return self.start_job(
            parent.spec.plan_id,
            idempotency_key=idempotency_key,
            parent_job_id=parent.job_id,
        )

    def _run_job(self, job_id: str) -> None:
        context: JobExecutionContext | None = None
        try:
            with self._mutating(job_id):
                current = self._job_store.get(job_id)
                if current.cancel_requested:
                    self._finalize_cancelled(job_id, context=None)
                    return
                try:
                    running = self._job_store.transition(
                        job_id,
                        expected_revision=current.revision,
                        state=JobState.RUNNING,
                        progress=JobProgress(phase="starting", fraction=0.0),
                        poll_after_ms=1_000,
                    )
                except JobStoreError as exc:
                    latest = self._job_store.get(job_id)
                    if latest.cancel_requested:
                        self._finalize_cancelled(job_id, context=None)
                        return
                    raise exc
            with self._runtime_lock:
                cancellation_event = self._cancellation_events.setdefault(
                    job_id,
                    threading.Event(),
                )
            context = JobExecutionContext(
                job_id=job_id,
                spec=running.spec,
                artifact_store=self._artifact_store,
                cancellation_event=cancellation_event,
                progress_callback=lambda progress, poll_after_ms: self._report_progress(
                    job_id,
                    progress,
                    poll_after_ms,
                ),
            )
            assert self._executor is not None
            result = self._executor(running.spec, context)
            if not isinstance(result, JobExecutionResult):
                raise JobExecutionError(
                    ApiError(
                        code="INTERNAL_ERROR",
                        message="The configured executor returned an invalid result.",
                        stage=ErrorStage.INTERNAL,
                    )
                )
            self._finalize_completed(job_id, result=result, context=context)
        except JobCancelledError:
            self._finalize_cancelled(job_id, context=context)
        except JobExecutionError as exc:
            self._finalize_failed(job_id, error=exc.error, context=context)
        except JobManagerError as exc:
            self._finalize_failed(job_id, error=exc.api_error, context=context)
        except (ArtifactStoreError, JobStoreError) as exc:
            self._finalize_failed(job_id, error=exc.api_error, context=context)
        except Exception as exc:  # noqa: BLE001 - executor internals stay private
            _log.exception("unhandled Agent executor failure for %s", job_id)
            self._finalize_failed(
                job_id,
                error=ApiError(
                    code="SOLVER_FAILED",
                    message="The retarget executor stopped before producing a result.",
                    stage=ErrorStage.EXECUTION,
                    details={"exception_type": type(exc).__name__},
                ),
                context=context,
            )
        finally:
            self._forget_runtime(job_id)

    def _report_progress(
        self,
        job_id: str,
        progress: JobProgress,
        poll_after_ms: int | None,
    ) -> AgentJobView:
        with self._mutating(job_id):
            for _ in range(3):
                current = self._job_store.get(job_id)
                if current.cancel_requested:
                    with self._runtime_lock:
                        event = self._cancellation_events.get(job_id)
                    if event is not None:
                        event.set()
                if current.view.state is not JobState.RUNNING:
                    return self._project_view(current)
                try:
                    updated = self._job_store.update_progress(
                        job_id,
                        expected_revision=current.revision,
                        progress=progress,
                        poll_after_ms=poll_after_ms,
                    )
                except JobStoreError as exc:
                    if exc.code == "JOB_CONFLICT":
                        continue
                    raise
                return self._project_view(updated)
            raise _error(
                "JOB_CONFLICT",
                "Progress could not be published after concurrent job updates.",
                details={"job_id": job_id},
            )

    def _finalize_completed(
        self,
        job_id: str,
        *,
        result: JobExecutionResult,
        context: JobExecutionContext,
    ) -> None:
        with self._mutating(job_id):
            self._finalize_completed_locked(
                job_id,
                result=result,
                context=context,
            )

    def _finalize_completed_locked(
        self,
        job_id: str,
        *,
        result: JobExecutionResult,
        context: JobExecutionContext,
    ) -> None:
        try:
            outcome = (
                result.outcome
                if isinstance(result.outcome, JobOutcome)
                else JobOutcome(result.outcome)
            )
        except (TypeError, ValueError):
            self._finalize_failed(
                job_id,
                error=ApiError(
                    code="INTERNAL_ERROR",
                    message="The executor returned an unsupported outcome.",
                    stage=ErrorStage.INTERNAL,
                ),
                context=context,
            )
            return
        created_at = datetime.now(UTC)
        evaluation = EvaluationReport(
            job_id=job_id,
            outcome=outcome,
            summary=result.evaluation_summary,
            metrics=dict(result.evaluation_metrics),
            checks=[dict(item) for item in result.evaluation_checks],
            created_at=created_at,
        )
        new_artifacts = list(context.published_artifacts())
        new_artifacts.append(
            self._artifact_store.put_json(
                job_id=job_id,
                kind="evaluation_report",
                document=evaluation.model_dump(mode="json"),
            )
        )
        failures = [_failure_item(item) for item in result.failures]
        if failures:
            failure_report = FailureReport(
                job_id=job_id,
                failures=failures,
                created_at=created_at,
            )
            new_artifacts.append(
                self._artifact_store.put_json(
                    job_id=job_id,
                    kind="failure_report",
                    document=failure_report.model_dump(mode="json"),
                )
            )
        for _ in range(3):
            current = self._job_store.get(job_id)
            if current.view.state not in _ACTIVE_STATES:
                return
            summary = _compact_summary(current.view.summary, result.summary)
            manifest = self._manifest(
                current,
                state=JobState.COMPLETED,
                outcome=outcome,
                error=None,
                summary=summary,
                execution_provenance=dict(result.execution_provenance),
                artifacts=[*current.artifacts, *new_artifacts],
                completed_at=created_at,
            )
            manifest_artifact = self._artifact_store.put_json(
                job_id=job_id,
                kind="manifest",
                document=manifest.model_dump(mode="json"),
            )
            total = current.view.progress.total_items
            try:
                self._job_store.transition(
                    job_id,
                    expected_revision=current.revision,
                    state=JobState.COMPLETED,
                    outcome=outcome,
                    progress=JobProgress(
                        phase="completed",
                        fraction=1.0,
                        completed_items=total,
                        total_items=total,
                        message="Execution completed.",
                    ),
                    next_action=result.next_action,
                    artifacts=[*new_artifacts, manifest_artifact],
                )
            except JobStoreError as exc:
                if exc.code == "JOB_CONFLICT":
                    continue
                raise
            return
        raise _error(
            "JOB_CONFLICT",
            "The completed job changed while its manifest was being published.",
            details={"job_id": job_id},
        )

    def _finalize_failed(
        self,
        job_id: str,
        *,
        error: ApiError,
        context: JobExecutionContext | None,
    ) -> None:
        with self._mutating(job_id):
            self._finalize_failed_locked(
                job_id,
                error=error,
                context=context,
            )

    def _finalize_failed_locked(
        self,
        job_id: str,
        *,
        error: ApiError,
        context: JobExecutionContext | None,
    ) -> None:
        try:
            current = self._job_store.get(job_id)
            if current.view.state not in _ACTIVE_STATES:
                return
            created_at = datetime.now(UTC)
            failure = FailureReport(
                job_id=job_id,
                failures=[
                    FailureItem(
                        code=error.code,
                        message=error.message,
                        stage=error.stage,
                        retryable=error.retryable,
                        details=error.details,
                    )
                ],
                created_at=created_at,
            )
            new_artifacts = list(context.published_artifacts()) if context else []
            new_artifacts.append(
                self._artifact_store.put_json(
                    job_id=job_id,
                    kind="failure_report",
                    document=failure.model_dump(mode="json"),
                )
            )
            for _ in range(3):
                current = self._job_store.get(job_id)
                if current.view.state not in _ACTIVE_STATES:
                    return
                manifest = self._manifest(
                    current,
                    state=JobState.FAILED,
                    outcome=None,
                    error=error,
                    summary=current.view.summary,
                    execution_provenance={},
                    artifacts=[*current.artifacts, *new_artifacts],
                    completed_at=created_at,
                )
                manifest_artifact = self._artifact_store.put_json(
                    job_id=job_id,
                    kind="manifest",
                    document=manifest.model_dump(mode="json"),
                )
                try:
                    self._job_store.transition(
                        job_id,
                        expected_revision=current.revision,
                        state=JobState.FAILED,
                        error=error,
                        next_action=error.next_action,
                        artifacts=[*new_artifacts, manifest_artifact],
                    )
                except JobStoreError as exc:
                    if exc.code == "JOB_CONFLICT":
                        continue
                    raise
                return
            raise _error(
                "JOB_CONFLICT",
                "The failed job changed while its manifest was being published.",
                details={"job_id": job_id},
            )
        except (ArtifactStoreError, JobStoreError, JobManagerError):
            _log.exception("failed to publish the complete failure record for %s", job_id)
            self._fallback_terminal(job_id, state=JobState.FAILED, error=error)

    def _finalize_cancelled(
        self,
        job_id: str,
        *,
        context: JobExecutionContext | None,
    ) -> None:
        with self._mutating(job_id):
            self._finalize_cancelled_locked(job_id, context=context)

    def _finalize_cancelled_locked(
        self,
        job_id: str,
        *,
        context: JobExecutionContext | None,
    ) -> None:
        try:
            current = self._job_store.get(job_id)
            if current.view.state not in _ACTIVE_STATES:
                return
            created_at = datetime.now(UTC)
            new_artifacts = list(context.published_artifacts()) if context else []
            for _ in range(3):
                current = self._job_store.get(job_id)
                if current.view.state not in _ACTIVE_STATES:
                    return
                manifest = self._manifest(
                    current,
                    state=JobState.CANCELLED,
                    outcome=None,
                    error=None,
                    summary=current.view.summary,
                    execution_provenance={},
                    artifacts=[*current.artifacts, *new_artifacts],
                    completed_at=created_at,
                )
                manifest_artifact = self._artifact_store.put_json(
                    job_id=job_id,
                    kind="manifest",
                    document=manifest.model_dump(mode="json"),
                )
                try:
                    self._job_store.transition(
                        job_id,
                        expected_revision=current.revision,
                        state=JobState.CANCELLED,
                        artifacts=[*new_artifacts, manifest_artifact],
                    )
                except JobStoreError as exc:
                    if exc.code == "JOB_CONFLICT":
                        continue
                    raise
                return
            raise _error(
                "JOB_CONFLICT",
                "The cancelled job changed while its manifest was being published.",
                details={"job_id": job_id},
            )
        except (ArtifactStoreError, JobStoreError, JobManagerError):
            _log.exception("failed to publish the complete cancellation record for %s", job_id)
            self._fallback_terminal(job_id, state=JobState.CANCELLED, error=None)
        finally:
            self._forget_runtime(job_id)

    @staticmethod
    def _manifest(
        current: StoredJob,
        *,
        state: JobState,
        outcome: JobOutcome | None,
        error: ApiError | None,
        summary: Mapping[str, Any],
        execution_provenance: Mapping[str, Any],
        artifacts: Sequence[ArtifactDescriptor],
        completed_at: datetime,
    ) -> JobManifest:
        baseline = current.view.started_at or current.view.submitted_at
        completed_at = max(completed_at, baseline)
        return JobManifest(
            job_id=current.job_id,
            parent_job_id=current.view.parent_job_id,
            root_job_id=current.view.root_job_id,
            attempt=current.view.attempt,
            plan_id=current.spec.plan_id,
            state=state,
            outcome=outcome,
            error=error,
            cancellation_requested=current.cancel_requested,
            job_spec=current.spec,
            execution_provenance=dict(execution_provenance),
            summary=dict(summary),
            artifacts=list(artifacts),
            submitted_at=current.view.submitted_at,
            started_at=current.view.started_at,
            completed_at=completed_at,
        )

    def _on_scheduler_cancel(self, job_id: str, _reason: str) -> None:
        with self._mutating(job_id):
            try:
                current = self._job_store.get(job_id)
                if current.view.state in _ACTIVE_STATES and not current.cancel_requested:
                    self._job_store.request_cancel(
                        job_id,
                        expected_revision=current.revision,
                    )
                with self._runtime_lock:
                    event = self._cancellation_events.get(job_id)
                if event is not None:
                    event.set()
                self._finalize_cancelled(job_id, context=None)
            except JobStoreError:
                _log.exception("failed to persist scheduler cancellation for %s", job_id)

    def _fail_before_execution(self, job_id: str, error: ApiError) -> None:
        self._finalize_failed(job_id, error=error, context=None)
        self._forget_runtime(job_id)

    def _fallback_terminal(
        self,
        job_id: str,
        *,
        state: JobState,
        error: ApiError | None,
    ) -> None:
        with self._mutating(job_id):
            try:
                current = self._job_store.get(job_id)
                if current.view.state not in _ACTIVE_STATES:
                    return
                self._job_store.transition(
                    job_id,
                    expected_revision=current.revision,
                    state=state,
                    error=error,
                )
            except JobStoreError:
                _log.exception("failed to persist fallback terminal state for %s", job_id)

    def _forget_runtime(self, job_id: str) -> None:
        with self._runtime_lock:
            self._handles.pop(job_id, None)
            self._cancellation_events.pop(job_id, None)

    def _recover_interrupted(self) -> None:
        try:
            active = self._job_store.list_active()
        except JobStoreError as exc:
            raise _wrap_service_error(exc.api_error) from exc
        for stored in active:
            self._finalize_failed(
                stored.job_id,
                error=ApiError(
                    code="JOB_INTERRUPTED",
                    message=(
                        "The previous process ended before this job reached a terminal state."
                    ),
                    retryable=True,
                    stage=ErrorStage.EXECUTION,
                    next_action=NextAction(
                        actor="agent",
                        action="retry_job",
                        parameters={"job_id": stored.job_id},
                    ),
                ),
                context=None,
            )

    def _project_view(
        self,
        stored: StoredJob,
        *,
        unchanged: bool = False,
    ) -> AgentJobView:
        queue: JobQueueView | None = None
        if stored.view.state in _ACTIVE_STATES:
            snapshot = self._scheduler.snapshot()

            def value(name: str, default: int = 0) -> Any:
                if isinstance(snapshot, dict):
                    return snapshot.get(name, default)
                return getattr(snapshot, name, default)

            max_running = int(value("max_running_jobs"))
            max_queued = int(value("max_queued_jobs"))
            if max_running == 0:
                mode = SchedulerMode.UNLIMITED
            elif max_queued > 0:
                mode = SchedulerMode.LIMITED
            else:
                mode = SchedulerMode.MIXED
            with self._runtime_lock:
                handle = self._handles.get(stored.job_id)
            queue = JobQueueView(
                position=handle.queue_position() if handle is not None else None,
                max_running_jobs=max_running,
                max_queued_jobs=max_queued,
                mode=mode,
            )
        document = stored.view.model_dump(mode="json")
        document["queue"] = queue.model_dump(mode="json") if queue is not None else None
        document["poll_after_ms"] = (
            None
            if stored.view.state in _TERMINAL_STATES
            else max(
                stored.view.poll_after_ms or 0,
                _DEFAULT_POLL_AFTER_MS if unchanged else 500,
            )
        )
        return AgentJobView.model_validate(document)


__all__ = [
    "JobCancelledError",
    "JobExecutionContext",
    "JobExecutionError",
    "JobExecutionResult",
    "JobExecutor",
    "JobManager",
    "JobManagerError",
]
