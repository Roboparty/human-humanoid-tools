from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from hhtools.contracts import (
    ApiError,
    ErrorStage,
    FailureItem,
    JobManifest,
    JobOutcome,
    JobSpecInput,
    JobSpecKind,
    JobSpecProvenance,
    JobSpecRobot,
    JobSpecV2,
    JobState,
    OutputPolicy,
)
from hhtools.services.artifacts import ArtifactStore
from hhtools.services.job_store import JobStore, JobStoreError
from hhtools.services.jobs import (
    JobExecutionContext,
    JobExecutionError,
    JobExecutionResult,
    JobManager,
    JobManagerError,
)
from hhtools.web.jobs.job_scheduler import JobScheduler


def _spec(
    marker: str = "1",
    *,
    run_mode: str = "smoke",
    backend: str = "newton",
) -> JobSpecV2:
    marker = marker[0].lower()
    if marker not in "123456789abcdef":
        marker = "1"
    return JobSpecV2(
        kind=JobSpecKind.RETARGET,
        plan_id=f"plan:sha256:{marker * 64}",
        inputs=[
            JobSpecInput(
                asset_id=f"asset:sha256:{'a' * 64}",
                sha256="a" * 64,
            )
        ],
        robot=JobSpecRobot(
            robot_id="g1_29dof",
            asset_id=f"asset:sha256:{'b' * 64}",
            config_sha256="b" * 64,
        ),
        calibration=None,
        backend=backend,
        effective_parameters={
            "run_mode": run_mode,
            "limit_frames": 30 if run_mode == "smoke" else None,
            "output_format": "csv",
        },
        output_policy=OutputPolicy.CREATE_NEW,
        provenance=JobSpecProvenance(
            hhtools_git_commit="f" * 40,
            hhtools_dirty=False,
            python="3.12.9",
            pytorch="2.7.0",
            cuda=None,
            newton="1.0.0",
            device=None,
            platform="Linux-x86_64",
            dependencies={"numpy": "2.2.0"},
        ),
        created_at=datetime(2026, 8, 31, 12, 0, tzinfo=UTC),
    )


class _RetargetService:
    def __init__(self, *specs: JobSpecV2) -> None:
        self.specs = {spec.plan_id: spec for spec in specs}
        self.calls: list[str] = []

    def get_job_spec(self, plan_id: str) -> JobSpecV2:
        self.calls.append(plan_id)
        return JobSpecV2.model_validate_json(self.specs[plan_id].model_dump_json())


def _manager(
    tmp_path: Path,
    *,
    retarget: _RetargetService,
    scheduler: JobScheduler,
    executor: Any,
    recover_interrupted: bool = False,
) -> tuple[JobManager, JobStore, ArtifactStore]:
    job_store = JobStore(tmp_path / "state")
    artifact_store = ArtifactStore(tmp_path / "state")
    manager = JobManager(
        job_store,
        artifact_store,
        retarget,  # type: ignore[arg-type]
        scheduler,
        executor=executor,
        recover_interrupted=recover_interrupted,
    )
    return manager, job_store, artifact_store


def _wait_for(
    manager: JobManager,
    job_id: str,
    predicate: Any,
    *,
    timeout: float = 3.0,
):
    deadline = time.monotonic() + timeout
    last = manager.get_job(job_id)
    while time.monotonic() < deadline:
        last = manager.get_job(job_id)
        if predicate(last):
            return last
        time.sleep(0.005)
    raise AssertionError(f"job did not reach expected state; last={last.model_dump()}")


def _terminal(manager: JobManager, job_id: str):
    return _wait_for(
        manager,
        job_id,
        lambda view: (
            view.state
            in {
                JobState.COMPLETED,
                JobState.FAILED,
                JobState.CANCELLED,
            }
        ),
    )


def _artifact_document(store: ArtifactStore, descriptor) -> dict[str, Any]:
    stored = store.get(descriptor.artifact_id, verify=True)
    return json.loads(stored.path.read_text(encoding="utf-8"))


def test_success_persists_exact_spec_evaluation_outputs_and_terminal_manifest(
    tmp_path: Path,
) -> None:
    spec = _spec()
    scheduler = JobScheduler()

    def execute(_spec_value: JobSpecV2, context: JobExecutionContext):
        context.report_progress(
            phase="ik_solve",
            fraction=0.5,
            completed_items=15,
            total_items=30,
            message="Solving frame 15/30",
        )
        context.publish_bytes(
            kind="retargeted_motion",
            payload=b"frame,joint\n0,0.0\n",
            format="csv",
            media_type="text/csv",
        )
        return JobExecutionResult(
            outcome=JobOutcome.SUCCESS,
            summary={"num_frames": 30},
            evaluation_summary="All configured smoke checks passed.",
            evaluation_metrics={"mean_joint_error": 0.01},
            evaluation_checks=[{"name": "finite", "passed": True}],
            execution_provenance={"device": "cuda:0", "cuda_runtime": "12.8"},
        )

    manager, job_store, artifact_store = _manager(
        tmp_path,
        retarget=_RetargetService(spec),
        scheduler=scheduler,
        executor=execute,
    )

    submitted = manager.start_retarget(spec.plan_id, idempotency_key="success-1")
    completed = _terminal(manager, submitted.job_id)

    assert completed.state is JobState.COMPLETED
    assert completed.outcome is JobOutcome.SUCCESS
    assert completed.error is None
    assert completed.progress.fraction == 1.0
    assert completed.progress.revision >= 4
    assert completed.artifact_count == 4
    assert {artifact.kind for artifact in completed.artifacts} == {
        "job_spec",
        "retargeted_motion",
        "evaluation_report",
        "manifest",
    }
    assert "trajectory" not in completed.model_dump_json()
    assert "preview" not in completed.model_dump_json()

    stored = job_store.get(completed.job_id)
    spec_artifact = next(item for item in stored.artifacts if item.kind == "job_spec")
    assert JobSpecV2.model_validate(_artifact_document(artifact_store, spec_artifact)) == spec
    manifest_descriptor = next(item for item in stored.artifacts if item.kind == "manifest")
    manifest = JobManifest.model_validate(_artifact_document(artifact_store, manifest_descriptor))
    assert manifest.job_spec == spec
    assert manifest.state is JobState.COMPLETED
    assert manifest.outcome is JobOutcome.SUCCESS
    assert manifest.execution_provenance.device == "cuda:0"
    assert manifest.summary["num_frames"] == 30
    assert manifest_descriptor.artifact_id not in {item.artifact_id for item in manifest.artifacts}
    assert len(manifest.artifacts) == 3

    unchanged = manager.get_job(
        completed.job_id,
        after_revision=completed.progress.revision,
    )
    assert unchanged.progress.revision == completed.progress.revision
    assert unchanged.poll_after_ms is None
    with pytest.raises(JobManagerError) as future:
        manager.get_job(completed.job_id, after_revision=completed.progress.revision + 1)
    assert future.value.code == "INVALID_PARAMETER"
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_concurrent_idempotent_submission_schedules_exactly_once(tmp_path: Path) -> None:
    spec = _spec()
    scheduler = JobScheduler()
    started = threading.Event()
    release = threading.Event()
    calls = 0
    lock = threading.Lock()

    def execute(_spec_value: JobSpecV2, _context: JobExecutionContext):
        nonlocal calls
        with lock:
            calls += 1
        started.set()
        release.wait(timeout=3.0)
        return JobExecutionResult(outcome=JobOutcome.SUCCESS)

    retarget = _RetargetService(spec)
    manager, job_store, _artifact_store = _manager(
        tmp_path,
        retarget=retarget,
        scheduler=scheduler,
        executor=execute,
    )

    def submit(_index: int) -> str:
        return manager.start_retarget(
            spec.plan_id,
            idempotency_key="same-request",
        ).job_id

    with ThreadPoolExecutor(max_workers=16) as pool:
        job_ids = list(pool.map(submit, range(48)))

    assert started.wait(timeout=1.0)
    assert len(set(job_ids)) == 1
    assert calls == 1
    assert retarget.calls == [spec.plan_id]
    with job_store._connect() as connection:  # noqa: SLF001 - persistence assertion
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    release.set()
    assert _terminal(manager, job_ids[0]).state is JobState.COMPLETED
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_idempotency_key_conflict_does_not_resolve_or_schedule_another_plan(
    tmp_path: Path,
) -> None:
    first = _spec("1")
    second = _spec("2")
    scheduler = JobScheduler()
    manager, _job_store, _artifact_store = _manager(
        tmp_path,
        retarget=_RetargetService(first, second),
        scheduler=scheduler,
        executor=lambda _spec_value, _context: JobExecutionResult(outcome=JobOutcome.SUCCESS),
    )
    created = manager.start_retarget(first.plan_id, idempotency_key="one-key")

    with pytest.raises(JobManagerError) as captured:
        manager.start_retarget(second.plan_id, idempotency_key="one-key")

    assert captured.value.code == "JOB_CONFLICT"
    assert captured.value.api_error.details == {"job_id": created.job_id}
    assert _terminal(manager, created.job_id).state is JobState.COMPLETED
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_lookup_recovers_one_submission_without_enumerating_jobs(tmp_path: Path) -> None:
    first = _spec("1")
    second = _spec("2")
    scheduler = JobScheduler()
    manager, _job_store, _artifact_store = _manager(
        tmp_path,
        retarget=_RetargetService(first, second),
        scheduler=scheduler,
        executor=lambda _spec_value, _context: JobExecutionResult(outcome=JobOutcome.SUCCESS),
    )
    submitted = manager.start_retarget(first.plan_id, idempotency_key="recover-me")
    completed = _terminal(manager, submitted.job_id)

    recovered = manager.lookup_job(
        first.plan_id,
        idempotency_key="recover-me",
        after_revision=completed.progress.revision,
    )

    assert recovered.job_id == completed.job_id
    assert recovered.progress.revision == completed.progress.revision
    assert recovered.progress.message == completed.progress.message
    assert recovered.poll_after_ms is None
    with pytest.raises(JobManagerError) as conflict:
        manager.lookup_job(second.plan_id, idempotency_key="recover-me")
    assert conflict.value.code == "JOB_CONFLICT"
    with pytest.raises(JobManagerError) as missing:
        manager.lookup_job(first.plan_id, idempotency_key="not-created")
    assert missing.value.code == "JOB_NOT_FOUND"
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_queued_cancel_is_precise_releases_capacity_and_never_runs_executor(
    tmp_path: Path,
) -> None:
    first = _spec("1")
    second = _spec("2")
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=1)
    release = threading.Event()
    executed: list[str] = []

    def execute(spec: JobSpecV2, _context: JobExecutionContext):
        executed.append(spec.plan_id)
        if spec.plan_id == first.plan_id:
            release.wait(timeout=3.0)
        return JobExecutionResult(outcome=JobOutcome.SUCCESS)

    manager, store, _artifacts = _manager(
        tmp_path,
        retarget=_RetargetService(first, second),
        scheduler=scheduler,
        executor=execute,
    )
    first_job = manager.start_retarget(first.plan_id, idempotency_key="first")
    _wait_for(manager, first_job.job_id, lambda view: view.state is JobState.RUNNING)
    second_job = manager.start_retarget(second.plan_id, idempotency_key="second")
    queued = _wait_for(
        manager,
        second_job.job_id,
        lambda view: view.state is JobState.QUEUED and view.queue is not None,
    )
    assert queued.queue is not None and queued.queue.position == 1

    cancelled = manager.cancel_job(second_job.job_id)

    assert cancelled.state is JobState.CANCELLED
    assert cancelled.cancellation_requested is True
    assert cancelled.cancellable is False
    assert scheduler.snapshot().queued_jobs == 0
    assert executed == [first.plan_id]
    assert {item.kind for item in store.get(second_job.job_id).artifacts} == {
        "job_spec",
        "manifest",
    }
    release.set()
    assert _terminal(manager, first_job.job_id).state is JobState.COMPLETED
    assert scheduler.shutdown(wait=True, timeout=2.0)


@pytest.mark.parametrize("backend", ["newton", "interaction_mesh"])
def test_running_cancel_stays_truthful_until_executor_acknowledges(
    tmp_path: Path,
    backend: str,
) -> None:
    spec = _spec(backend=backend)
    scheduler = JobScheduler()
    started = threading.Event()

    def execute(_spec_value: JobSpecV2, context: JobExecutionContext):
        started.set()
        while not context.cancellation_requested:
            time.sleep(0.002)
        context.raise_if_cancelled()
        raise AssertionError("unreachable")

    manager, _store, _artifacts = _manager(
        tmp_path,
        retarget=_RetargetService(spec),
        scheduler=scheduler,
        executor=execute,
    )
    submitted = manager.start_retarget(spec.plan_id, idempotency_key="cancel-running")
    assert started.wait(timeout=1.0)

    requested = manager.cancel_job(submitted.job_id)
    assert requested.cancellation_requested is True
    assert requested.state in {JobState.RUNNING, JobState.CANCELLED}
    cancelled = _terminal(manager, submitted.job_id)
    assert cancelled.state is JobState.CANCELLED
    assert cancelled.outcome is None
    assert cancelled.error is None
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_completed_work_can_truthfully_win_a_late_cooperative_cancel(tmp_path: Path) -> None:
    spec = _spec()
    scheduler = JobScheduler()
    started = threading.Event()
    release = threading.Event()

    def execute(_spec_value: JobSpecV2, _context: JobExecutionContext):
        started.set()
        release.wait(timeout=3.0)
        return JobExecutionResult(outcome=JobOutcome.SUCCESS)

    manager, store, artifacts = _manager(
        tmp_path,
        retarget=_RetargetService(spec),
        scheduler=scheduler,
        executor=execute,
    )
    submitted = manager.start_retarget(spec.plan_id, idempotency_key="late-cancel")
    assert started.wait(timeout=1.0)
    requested = manager.cancel_job(submitted.job_id)
    assert requested.state is JobState.RUNNING
    assert requested.cancellation_requested is True
    release.set()

    completed = _terminal(manager, submitted.job_id)
    assert completed.state is JobState.COMPLETED
    assert completed.outcome is JobOutcome.SUCCESS
    assert completed.cancellation_requested is True
    manifest_descriptor = next(
        item for item in store.get(completed.job_id).artifacts if item.kind == "manifest"
    )
    manifest = JobManifest.model_validate(_artifact_document(artifacts, manifest_descriptor))
    assert manifest.cancellation_requested is True
    assert scheduler.shutdown(wait=True, timeout=2.0)


@pytest.mark.parametrize("backend", ["newton", "interaction_mesh"])
def test_structured_execution_failure_publishes_failure_report_and_manifest(
    tmp_path: Path,
    backend: str,
) -> None:
    spec = _spec(backend=backend)
    scheduler = JobScheduler()

    def execute(_spec_value: JobSpecV2, _context: JobExecutionContext):
        raise JobExecutionError(
            ApiError(
                code="CUDA_OUT_OF_MEMORY",
                message="The selected device did not have enough free memory.",
                retryable=True,
                stage=ErrorStage.EXECUTION,
                details={"device_id": "cuda:0"},
            )
        )

    manager, store, artifacts = _manager(
        tmp_path,
        retarget=_RetargetService(spec),
        scheduler=scheduler,
        executor=execute,
    )
    submitted = manager.start_retarget(spec.plan_id, idempotency_key="oom")
    failed = _terminal(manager, submitted.job_id)

    assert failed.state is JobState.FAILED
    assert failed.error is not None and failed.error.code == "CUDA_OUT_OF_MEMORY"
    assert failed.outcome is None
    stored = store.get(failed.job_id)
    assert {item.kind for item in stored.artifacts} == {
        "job_spec",
        "failure_report",
        "manifest",
    }
    report = next(item for item in stored.artifacts if item.kind == "failure_report")
    assert _artifact_document(artifacts, report)["failures"][0]["code"] == ("CUDA_OUT_OF_MEMORY")
    manifest = next(item for item in stored.artifacts if item.kind == "manifest")
    assert _artifact_document(artifacts, manifest)["state"] == "failed"
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_partial_completion_uses_outcome_and_failure_artifact_not_top_level_error(
    tmp_path: Path,
) -> None:
    spec = _spec()
    scheduler = JobScheduler()

    def execute(_spec_value: JobSpecV2, _context: JobExecutionContext):
        return JobExecutionResult(
            outcome=JobOutcome.PARTIAL,
            failures=[
                FailureItem(
                    item_id="clip-2",
                    code="SOLVER_FAILED",
                    message="One clip did not converge.",
                    stage=ErrorStage.EXECUTION,
                )
            ],
        )

    manager, store, _artifacts = _manager(
        tmp_path,
        retarget=_RetargetService(spec),
        scheduler=scheduler,
        executor=execute,
    )
    submitted = manager.start_retarget(spec.plan_id, idempotency_key="partial")
    completed = _terminal(manager, submitted.job_id)

    assert completed.state is JobState.COMPLETED
    assert completed.outcome is JobOutcome.PARTIAL
    assert completed.error is None
    assert {item.kind for item in store.get(completed.job_id).artifacts} == {
        "job_spec",
        "evaluation_report",
        "failure_report",
        "manifest",
    }
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_retry_creates_child_lineage_and_never_mutates_parent(tmp_path: Path) -> None:
    spec = _spec()
    scheduler = JobScheduler()
    calls = 0

    def execute(_spec_value: JobSpecV2, _context: JobExecutionContext):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise JobExecutionError(
                ApiError(
                    code="SOLVER_FAILED",
                    message="First attempt failed.",
                    stage=ErrorStage.EXECUTION,
                )
            )
        return JobExecutionResult(outcome=JobOutcome.SUCCESS)

    manager, store, _artifacts = _manager(
        tmp_path,
        retarget=_RetargetService(spec),
        scheduler=scheduler,
        executor=execute,
    )
    parent_submission = manager.start_retarget(spec.plan_id, idempotency_key="parent")
    parent = _terminal(manager, parent_submission.job_id)
    assert parent.state is JobState.FAILED

    child_submission = manager.retry_job(parent.job_id, idempotency_key="child")
    child = _terminal(manager, child_submission.job_id)

    assert child.state is JobState.COMPLETED
    assert child.parent_job_id == parent.job_id
    assert child.root_job_id == parent.job_id
    assert child.attempt == 2
    assert store.get(parent.job_id).view == parent
    repeated = manager.retry_job(parent.job_id, idempotency_key="child")
    assert repeated.job_id == child.job_id
    assert calls == 2
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_restart_marks_active_jobs_interrupted_without_resubmitting(
    tmp_path: Path,
) -> None:
    spec = _spec()
    data_dir = tmp_path / "state"
    store = JobStore(data_dir)
    queued = store.create(spec, idempotency_key="survives-restart")
    scheduler = JobScheduler()

    manager = JobManager(
        JobStore(data_dir),
        ArtifactStore(data_dir),
        _RetargetService(spec),  # type: ignore[arg-type]
        scheduler,
        executor=None,
        recover_interrupted=True,
    )

    recovered = manager.get_job(queued.job_id)
    assert recovered.state is JobState.FAILED
    assert recovered.error is not None and recovered.error.code == "JOB_INTERRUPTED"
    assert {item.kind for item in JobStore(data_dir).get(queued.job_id).artifacts} == {
        "failure_report",
        "manifest",
    }
    replayed = manager.start_retarget(
        spec.plan_id,
        idempotency_key="survives-restart",
    )
    assert replayed.job_id == recovered.job_id
    assert replayed.state is JobState.FAILED
    assert scheduler.snapshot().running_jobs == 0
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_bounded_queue_rejects_without_creating_a_third_job(tmp_path: Path) -> None:
    first = _spec("1")
    second = _spec("2")
    third = _spec("3")
    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=1)
    release = threading.Event()

    def execute(spec: JobSpecV2, _context: JobExecutionContext):
        if spec.plan_id == first.plan_id:
            release.wait(timeout=3.0)
        return JobExecutionResult(outcome=JobOutcome.SUCCESS)

    manager, store, _artifacts = _manager(
        tmp_path,
        retarget=_RetargetService(first, second, third),
        scheduler=scheduler,
        executor=execute,
    )
    running = manager.start_retarget(first.plan_id, idempotency_key="bounded-1")
    _wait_for(manager, running.job_id, lambda view: view.state is JobState.RUNNING)
    queued = manager.start_retarget(second.plan_id, idempotency_key="bounded-2")
    _wait_for(manager, queued.job_id, lambda view: view.state is JobState.QUEUED)

    with pytest.raises(JobManagerError) as captured:
        manager.start_retarget(third.plan_id, idempotency_key="bounded-3")

    assert captured.value.code == "QUEUE_FULL"
    with pytest.raises(JobStoreError) as missing:
        store.get_by_idempotency_key("bounded-3")
    assert missing.value.code == "JOB_NOT_FOUND"
    manager.cancel_job(queued.job_id)
    release.set()
    assert _terminal(manager, running.job_id).state is JobState.COMPLETED
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_new_submission_requires_an_executor_but_existing_history_is_readable(
    tmp_path: Path,
) -> None:
    spec = _spec()
    scheduler = JobScheduler()
    manager, _store, _artifacts = _manager(
        tmp_path,
        retarget=_RetargetService(spec),
        scheduler=scheduler,
        executor=None,
    )

    with pytest.raises(JobManagerError) as captured:
        manager.start_retarget(spec.plan_id, idempotency_key="no-executor")

    assert captured.value.code == "BACKEND_UNAVAILABLE"
    assert scheduler.snapshot().running_jobs == 0
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_terminal_manifest_publish_serializes_a_late_cancel_without_orphans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    scheduler = JobScheduler()
    executor_started = threading.Event()
    release_executor = threading.Event()
    manifest_stored = threading.Event()
    release_manifest = threading.Event()

    def execute(_spec_value: JobSpecV2, _context: JobExecutionContext):
        executor_started.set()
        assert release_executor.wait(timeout=3.0)
        return JobExecutionResult(outcome=JobOutcome.SUCCESS)

    manager, store, artifacts = _manager(
        tmp_path,
        retarget=_RetargetService(spec),
        scheduler=scheduler,
        executor=execute,
    )
    original_put_json = artifacts.put_json

    def blocking_put_json(**kwargs: Any):
        descriptor = original_put_json(**kwargs)
        if kwargs.get("kind") == "manifest":
            manifest_stored.set()
            assert release_manifest.wait(timeout=3.0)
        return descriptor

    monkeypatch.setattr(artifacts, "put_json", blocking_put_json)
    submitted = manager.start_retarget(spec.plan_id, idempotency_key="locked-terminal")
    assert executor_started.wait(timeout=1.0)
    release_executor.set()
    assert manifest_stored.wait(timeout=1.0)

    cancel_entered = threading.Event()

    def cancel_after_manifest_write():
        cancel_entered.set()
        return manager.cancel_job(submitted.job_id)

    with ThreadPoolExecutor(max_workers=1) as pool:
        cancellation = pool.submit(cancel_after_manifest_write)
        try:
            assert cancel_entered.wait(timeout=1.0)
            time.sleep(0.02)
            assert cancellation.done() is False
        finally:
            release_manifest.set()
        completed = _terminal(manager, submitted.job_id)
        with pytest.raises(JobManagerError) as captured:
            cancellation.result(timeout=1.0)
        assert captured.value.code == "JOB_CANCEL_UNSUPPORTED"

    assert completed.state is JobState.COMPLETED
    assert completed.cancellation_requested is False
    assert (
        len(
            [
                item
                for item in artifacts.list_candidates_for_job(submitted.job_id)
                if item.kind == "manifest"
            ]
        )
        == 1
    )
    canonical_manifests = [
        item for item in store.get(submitted.job_id).artifacts if item.kind == "manifest"
    ]
    assert len(canonical_manifests) == 1
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_canonical_artifact_access_hides_unbound_candidates(tmp_path: Path) -> None:
    spec = _spec()
    scheduler = JobScheduler()
    manager, store, artifacts = _manager(
        tmp_path,
        retarget=_RetargetService(spec),
        scheduler=scheduler,
        executor=lambda _spec_value, _context: JobExecutionResult(outcome=JobOutcome.SUCCESS),
    )
    submitted = manager.start_retarget(spec.plan_id, idempotency_key="canonical-only")
    completed = _terminal(manager, submitted.job_id)
    orphan = artifacts.put_bytes(
        job_id=completed.job_id,
        kind="log_tail",
        payload=b"unbound candidate",
    )

    candidates = artifacts.list_candidates_for_job(completed.job_id)
    canonical = manager.list_artifacts(completed.job_id)

    assert orphan.artifact_id in {item.artifact_id for item in candidates}
    assert orphan.artifact_id not in {item.artifact_id for item in canonical}
    assert canonical == list(store.get(completed.job_id).artifacts)
    with pytest.raises(JobManagerError) as captured:
        manager.get_artifact(completed.job_id, orphan.artifact_id)
    assert captured.value.code == "ARTIFACT_NOT_FOUND"
    resolved = manager.get_artifact(
        completed.job_id,
        canonical[0].artifact_id,
        verify=True,
    )
    assert resolved.descriptor == canonical[0]
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_canonical_artifact_access_rejects_another_jobs_descriptor(
    tmp_path: Path,
) -> None:
    first = _spec("1")
    second = _spec("2")
    scheduler = JobScheduler()
    manager, _store, artifacts = _manager(
        tmp_path,
        retarget=_RetargetService(first, second),
        scheduler=scheduler,
        executor=lambda _spec_value, _context: JobExecutionResult(outcome=JobOutcome.SUCCESS),
    )
    first_job = manager.start_retarget(first.plan_id, idempotency_key="artifact-owner-1")
    second_job = manager.start_retarget(second.plan_id, idempotency_key="artifact-owner-2")
    first_completed = _terminal(manager, first_job.job_id)
    second_completed = _terminal(manager, second_job.job_id)
    foreign = manager.list_artifacts(second_completed.job_id)[0]

    assert artifacts.get(foreign.artifact_id).descriptor == foreign
    with pytest.raises(JobManagerError) as captured:
        manager.get_artifact(first_completed.job_id, foreign.artifact_id)
    assert captured.value.code == "ARTIFACT_NOT_FOUND"
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_canonical_artifact_listing_pages_beyond_compact_job_view(
    tmp_path: Path,
) -> None:
    spec = _spec()
    scheduler = JobScheduler()

    def execute(_spec_value: JobSpecV2, context: JobExecutionContext):
        for index in range(40):
            context.publish_bytes(
                kind="chunk",
                payload=f"chunk-{index:02d}".encode(),
            )
        return JobExecutionResult(outcome=JobOutcome.SUCCESS)

    manager, store, _artifacts = _manager(
        tmp_path,
        retarget=_RetargetService(spec),
        scheduler=scheduler,
        executor=execute,
    )
    submitted = manager.start_retarget(spec.plan_id, idempotency_key="many-artifacts")
    completed = _terminal(manager, submitted.job_id)

    assert completed.artifact_count == 43
    assert len(completed.artifacts) == 32
    pages = [
        *manager.list_artifacts(completed.job_id, offset=0, limit=20),
        *manager.list_artifacts(completed.job_id, offset=20, limit=20),
        *manager.list_artifacts(completed.job_id, offset=40, limit=20),
    ]
    assert pages == list(store.get(completed.job_id).artifacts)
    assert len(manager.list_artifacts(completed.job_id)) == 43
    with pytest.raises(JobManagerError) as invalid_limit:
        manager.list_artifacts(completed.job_id, limit=501)
    assert invalid_limit.value.code == "INVALID_PARAMETER"
    assert scheduler.shutdown(wait=True, timeout=2.0)


def test_canonical_artifact_access_detects_descriptor_drift(tmp_path: Path) -> None:
    spec = _spec()
    scheduler = JobScheduler()
    manager, _store, artifacts = _manager(
        tmp_path,
        retarget=_RetargetService(spec),
        scheduler=scheduler,
        executor=lambda _spec_value, _context: JobExecutionResult(outcome=JobOutcome.SUCCESS),
    )
    submitted = manager.start_retarget(spec.plan_id, idempotency_key="descriptor-drift")
    completed = _terminal(manager, submitted.job_id)
    job_spec = next(
        item for item in manager.list_artifacts(completed.job_id) if item.kind == "job_spec"
    )
    with artifacts._connect() as connection:  # noqa: SLF001 - corruption assertion
        connection.execute(
            "UPDATE artifacts SET metadata_json = '{}' WHERE artifact_id = ?",
            (job_spec.artifact_id,),
        )

    with pytest.raises(JobManagerError) as captured:
        manager.get_artifact(completed.job_id, job_spec.artifact_id)

    assert captured.value.code == "INTERNAL_ERROR"
    assert scheduler.shutdown(wait=True, timeout=2.0)
