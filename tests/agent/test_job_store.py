from __future__ import annotations

import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hhtools.contracts import (
    ApiError,
    ArtifactDescriptor,
    ErrorStage,
    JobOutcome,
    JobProgress,
    JobSpecCalibration,
    JobSpecInput,
    JobSpecKind,
    JobSpecProvenance,
    JobSpecRobot,
    JobSpecV2,
    JobState,
    NextAction,
    OutputPolicy,
)
from hhtools.services.job_store import (
    JobStore,
    JobStoreError,
    compute_request_fingerprint,
)

NOW = datetime(2026, 8, 31, 8, 0, tzinfo=UTC)


@dataclass
class _Clock:
    value: datetime

    def __call__(self) -> datetime:
        return self.value

    def advance(self, *, seconds: int = 1) -> datetime:
        self.value += timedelta(seconds=seconds)
        return self.value


def _spec(*, run_mode: str = "smoke", created_at: datetime = NOW) -> JobSpecV2:
    return JobSpecV2(
        kind=JobSpecKind.RETARGET,
        plan_id=f"plan:sha256:{'1' * 64}",
        inputs=[
            JobSpecInput(
                asset_id=f"asset:sha256:{'2' * 64}",
                sha256="2" * 64,
            )
        ],
        robot=JobSpecRobot(
            robot_id="g1_29dof",
            asset_id=f"asset:sha256:{'3' * 64}",
            config_sha256="3" * 64,
        ),
        calibration=None,
        backend="newton",
        effective_parameters={
            "run_mode": run_mode,
            "limit_frames": 30 if run_mode == "smoke" else None,
            "solver": {"iterations": 24, "gain": 0.5},
        },
        output_policy=OutputPolicy.CREATE_NEW,
        provenance=JobSpecProvenance(
            hhtools_git_commit="f" * 40,
            hhtools_dirty=False,
            python="3.13.7",
            pytorch="2.8.0",
            cuda="12.8",
            newton="1.0.0",
            device=None,
            platform="Linux-x86_64",
            dependencies={"numpy": "2.2.0", "warp-lang": "1.9.0"},
        ),
        created_at=created_at,
    )


def _assert_code(captured: pytest.ExceptionInfo[JobStoreError], code: str) -> None:
    assert captured.value.code == code
    assert captured.value.api_error.code == code


def _failure() -> ApiError:
    return ApiError(
        code="SOLVER_FAILED",
        message="The solver stopped before producing an output.",
        stage=ErrorStage.EXECUTION,
        retryable=False,
        details={"phase": "ik_solve"},
    )


def _artifact(
    job_id: str,
    index: int,
    *,
    sha256: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ArtifactDescriptor:
    return ArtifactDescriptor(
        artifact_id=f"artifact:result:item-{index:02d}",
        job_id=job_id,
        kind="retargeted_motion",
        format="npz",
        resource_uri=f"hhtools://jobs/{job_id}/artifacts/item-{index:02d}",
        media_type="application/octet-stream",
        size_bytes=100 + index,
        sha256=sha256 or f"{index % 16:x}" * 64,
        created_at=NOW + timedelta(minutes=2, seconds=index),
        metadata=metadata or {"frame_count": 30 + index},
    )


def test_create_persists_an_immutable_detached_spec_and_enables_wal(
    tmp_path: Path,
) -> None:
    clock = _Clock(NOW + timedelta(minutes=1))
    store = JobStore(
        tmp_path / "state",
        clock=clock,
        job_id_provider=lambda: "job:first",
    )
    spec = _spec()
    fingerprint = compute_request_fingerprint(spec)

    created = store.create(
        spec,
        idempotency_key="retarget-request-1",
        request_fingerprint=fingerprint,
    )

    assert created.job_id == "job:first"
    assert created.created
    assert created.revision == 0
    assert created.request_fingerprint == fingerprint
    assert created.spec == spec
    assert created.view.state is JobState.QUEUED
    assert created.view.outcome is None
    assert created.view.progress.phase == "queued"
    assert created.view.progress.fraction == 0.0
    assert created.view.progress.updated_at == clock.value
    assert created.view.submitted_at == clock.value
    assert created.view.summary == {
        "input_count": 1,
        "robot_id": "g1_29dof",
        "backend": "newton",
        "run_mode": "smoke",
    }
    assert created.view.queue is None
    assert created.view.artifacts == []
    assert created.view.artifact_count == 0
    assert created.view.cancellation_requested is False
    assert created.view.cancellable is True
    assert not created.cancel_requested

    # Both the caller's mutable dictionaries and each returned snapshot are
    # detached from the immutable canonical JSON held by the store.
    spec.effective_parameters["run_mode"] = "changed-by-caller"
    created.spec.effective_parameters["solver"]["gain"] = 99  # type: ignore[index]
    restored = JobStore(tmp_path / "state").get(created.job_id)
    assert not restored.created
    assert restored.spec.effective_parameters["run_mode"] == "smoke"
    assert restored.spec.effective_parameters["solver"]["gain"] == 0.5  # type: ignore[index]
    assert JobStore(tmp_path / "state").get_spec(created.job_id) == restored.spec

    assert store.database_path == tmp_path / "state" / "jobs.sqlite3"
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_request_fingerprint_is_deterministic_and_covers_the_complete_spec() -> None:
    first = _spec()
    reordered = first.model_copy(deep=True)
    reordered.effective_parameters.clear()
    reordered.effective_parameters.update(
        {
            "solver": {"gain": 0.5, "iterations": 24},
            "limit_frames": 30,
            "run_mode": "smoke",
        }
    )
    changed = _spec(run_mode="full")

    assert compute_request_fingerprint(first) == compute_request_fingerprint(reordered)
    assert compute_request_fingerprint(first) != compute_request_fingerprint(changed)


def test_concurrent_retries_with_one_key_create_exactly_one_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "state", clock=_Clock(NOW + timedelta(minutes=1)))
    spec = _spec()

    def submit(_: int):
        return store.create(spec, idempotency_key="same-logical-request")

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(submit, range(48)))

    assert len({result.job_id for result in results}) == 1
    assert len({result.request_fingerprint for result in results}) == 1
    assert all(result.revision == 0 for result in results)
    assert sum(result.created for result in results) == 1
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_wait_for_revision_wakes_all_waiters_after_one_update(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "state", job_id_provider=lambda: "job:waiters")
    queued = store.create(_spec(), idempotency_key="waiters")
    ready = threading.Barrier(3)

    def wait_for_update() -> int:
        ready.wait(timeout=1.0)
        return store.wait_for_revision(
            queued.job_id,
            after_revision=queued.revision,
            timeout=1.0,
        ).revision

    with ThreadPoolExecutor(max_workers=2) as executor:
        waiters = [executor.submit(wait_for_update) for _ in range(2)]
        ready.wait(timeout=1.0)
        updated = store.transition(
            queued.job_id,
            expected_revision=queued.revision,
            state=JobState.RUNNING,
        )
        revisions = [waiter.result(timeout=1.0) for waiter in waiters]

    assert revisions == [updated.revision, updated.revision]


def test_wait_for_revision_times_out_and_terminal_jobs_return_immediately(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "state", job_id_provider=lambda: "job:wait-timeout")
    queued = store.create(_spec(), idempotency_key="wait-timeout")

    started = time.monotonic()
    unchanged = store.wait_for_revision(
        queued.job_id,
        after_revision=queued.revision,
        timeout=0.03,
    )
    elapsed = time.monotonic() - started

    assert unchanged.revision == queued.revision
    assert elapsed >= 0.02

    cancelled = store.transition(
        queued.job_id,
        expected_revision=queued.revision,
        state=JobState.CANCELLED,
    )
    started = time.monotonic()
    terminal = store.wait_for_revision(
        queued.job_id,
        after_revision=cancelled.revision,
        timeout=1.0,
    )
    assert terminal.view.state is JobState.CANCELLED
    assert time.monotonic() - started < 0.2


@pytest.mark.parametrize("timeout", [-0.1, 60.1, float("nan"), float("inf"), True])
def test_wait_for_revision_rejects_invalid_bounds(tmp_path: Path, timeout: float) -> None:
    store = JobStore(tmp_path / "state", job_id_provider=lambda: "job:wait-invalid")
    queued = store.create(_spec(), idempotency_key="wait-invalid")

    with pytest.raises(JobStoreError) as invalid_timeout:
        store.wait_for_revision(
            queued.job_id,
            after_revision=queued.revision,
            timeout=timeout,
        )
    _assert_code(invalid_timeout, "INVALID_PARAMETER")

    with pytest.raises(JobStoreError) as future_revision:
        store.wait_for_revision(
            queued.job_id,
            after_revision=queued.revision + 1,
            timeout=0,
        )
    _assert_code(future_revision, "INVALID_PARAMETER")


def test_idempotency_key_conflicts_on_changed_request_but_other_key_is_new(
    tmp_path: Path,
) -> None:
    job_ids = iter(["job:smoke", "job:full"])
    store = JobStore(
        tmp_path / "state",
        clock=_Clock(NOW + timedelta(minutes=1)),
        job_id_provider=lambda: next(job_ids),
    )
    smoke = store.create(_spec(), idempotency_key="request-key")

    with pytest.raises(JobStoreError) as captured:
        store.create(_spec(run_mode="full"), idempotency_key="request-key")

    _assert_code(captured, "JOB_CONFLICT")
    assert captured.value.error.stage is ErrorStage.ADMISSION
    assert captured.value.error.details == {"job_id": smoke.job_id}
    assert store.get(smoke.job_id).spec.effective_parameters["run_mode"] == "smoke"

    full = store.create(_spec(run_mode="full"), idempotency_key="another-key")
    assert full.job_id != smoke.job_id
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2


def test_supplied_fingerprint_is_validated_instead_of_trusted(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "state")

    with pytest.raises(JobStoreError) as malformed:
        store.create(
            _spec(),
            idempotency_key="request-key",
            request_fingerprint="not-a-digest",
        )
    _assert_code(malformed, "INVALID_PARAMETER")

    with pytest.raises(JobStoreError) as divergent:
        store.create(
            _spec(),
            idempotency_key="request-key",
            request_fingerprint="0" * 64,
        )
    _assert_code(divergent, "INVALID_PARAMETER")

    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_legal_lifecycle_transitions_own_revisions_timestamps_and_outcome(
    tmp_path: Path,
) -> None:
    clock = _Clock(NOW + timedelta(minutes=1))
    store = JobStore(
        tmp_path / "state",
        clock=clock,
        job_id_provider=lambda: "job:lifecycle",
    )
    queued = store.create(_spec(), idempotency_key="lifecycle")

    clock.advance(seconds=2)
    running = store.transition(
        queued.job_id,
        expected_revision=0,
        state=JobState.RUNNING,
        progress=JobProgress(
            phase="ik_solve",
            fraction=0.2,
            revision=999,
            completed_items=2,
            total_items=10,
            message="Solving frame 2/10",
        ),
        poll_after_ms=250,
    )
    assert running.revision == 1
    assert running.view.started_at == clock.value
    assert running.view.completed_at is None
    assert running.view.progress.revision == 1
    assert running.view.progress.updated_at == clock.value
    assert running.view.poll_after_ms == 250

    clock.advance(seconds=3)
    progressed = store.update_progress(
        running.job_id,
        expected_revision=1,
        progress=JobProgress(
            phase="ik_solve",
            fraction=0.6,
            revision=0,
            completed_items=6,
            total_items=10,
        ),
        next_action=NextAction(
            actor="system",
            action="continue_polling",
            parameters={"after_revision": 2},
        ),
        poll_after_ms=500,
    )
    assert progressed.revision == 2
    assert progressed.view.state is JobState.RUNNING
    assert progressed.view.progress.completed_items == 6
    assert progressed.view.next_action is not None
    assert progressed.view.next_action.action == "continue_polling"

    clock.advance(seconds=4)
    completed = store.transition(
        progressed.job_id,
        expected_revision=2,
        state="completed",
        outcome="success",
        progress=JobProgress(
            phase="completed",
            fraction=1.0,
            revision=0,
            completed_items=10,
            total_items=10,
        ),
    )
    assert completed.revision == 3
    assert completed.view.state is JobState.COMPLETED
    assert completed.view.outcome is JobOutcome.SUCCESS
    assert completed.view.started_at == running.view.started_at
    assert completed.view.completed_at == clock.value
    assert completed.view.progress.eta_seconds is None
    assert completed.view.error is None

    # Terminal state and its persisted JobSpec survive another store instance.
    reopened = JobStore(tmp_path / "state").get(completed.job_id)
    assert reopened.view == completed.view
    assert reopened.spec == completed.spec


def test_revision_conflict_and_illegal_transitions_leave_the_row_unchanged(
    tmp_path: Path,
) -> None:
    store = JobStore(
        tmp_path / "state",
        clock=_Clock(NOW + timedelta(minutes=1)),
        job_id_provider=lambda: "job:guarded",
    )
    queued = store.create(_spec(), idempotency_key="guarded")

    with pytest.raises(JobStoreError) as stale:
        store.transition(
            queued.job_id,
            expected_revision=1,
            state=JobState.RUNNING,
        )
    _assert_code(stale, "JOB_CONFLICT")
    assert stale.value.error.details["current_revision"] == 0

    with pytest.raises(JobStoreError) as skipped:
        store.transition(
            queued.job_id,
            expected_revision=0,
            state=JobState.COMPLETED,
            outcome=JobOutcome.SUCCESS,
        )
    _assert_code(skipped, "INVALID_JOB_TRANSITION")

    with pytest.raises(JobStoreError) as same_state:
        store.transition(
            queued.job_id,
            expected_revision=0,
            state=JobState.QUEUED,
        )
    _assert_code(same_state, "INVALID_JOB_TRANSITION")
    assert store.get(queued.job_id).view == queued.view


def test_progress_is_monotonic_and_terminal_progress_is_immutable(tmp_path: Path) -> None:
    clock = _Clock(NOW + timedelta(minutes=1))
    store = JobStore(
        tmp_path / "state",
        clock=clock,
        job_id_provider=lambda: "job:progress",
    )
    queued = store.create(_spec(), idempotency_key="progress")
    clock.advance()
    running = store.transition(
        queued.job_id,
        expected_revision=0,
        state=JobState.RUNNING,
        progress=JobProgress(
            phase="solve",
            fraction=0.5,
            completed_items=5,
            total_items=10,
        ),
    )

    clock.advance()
    with pytest.raises(JobStoreError) as backwards:
        store.update_progress(
            running.job_id,
            expected_revision=1,
            progress=JobProgress(
                phase="solve",
                fraction=0.4,
                completed_items=4,
                total_items=10,
            ),
        )
    _assert_code(backwards, "INVALID_JOB_TRANSITION")

    with pytest.raises(JobStoreError) as changed_total:
        store.update_progress(
            running.job_id,
            expected_revision=1,
            progress=JobProgress(
                phase="solve",
                fraction=0.6,
                completed_items=6,
                total_items=12,
            ),
        )
    _assert_code(changed_total, "INVALID_JOB_TRANSITION")
    assert store.get(running.job_id).revision == 1

    completed = store.transition(
        running.job_id,
        expected_revision=1,
        state=JobState.COMPLETED,
        outcome=JobOutcome.SUCCESS,
        progress=JobProgress(
            phase="completed",
            fraction=1.0,
            completed_items=10,
            total_items=10,
        ),
    )
    with pytest.raises(JobStoreError) as terminal:
        store.update_progress(
            completed.job_id,
            expected_revision=2,
            progress=JobProgress(phase="late", fraction=1.0),
        )
    _assert_code(terminal, "INVALID_JOB_TRANSITION")
    assert store.get(completed.job_id).revision == 2


def test_failed_job_requires_and_persists_structured_error(tmp_path: Path) -> None:
    clock = _Clock(NOW + timedelta(minutes=1))
    store = JobStore(
        tmp_path / "state",
        clock=clock,
        job_id_provider=lambda: "job:failed",
    )
    queued = store.create(_spec(), idempotency_key="failed")

    with pytest.raises(JobStoreError) as missing_error:
        store.transition(
            queued.job_id,
            expected_revision=0,
            state=JobState.FAILED,
        )
    _assert_code(missing_error, "INVALID_PARAMETER")

    clock.advance()
    failed = store.transition(
        queued.job_id,
        expected_revision=0,
        state=JobState.FAILED,
        error=_failure(),
    )
    assert failed.view.state is JobState.FAILED
    assert failed.view.error == _failure()
    assert failed.view.outcome is None
    assert failed.view.completed_at == clock.value

    with pytest.raises(JobStoreError) as cancel_failed:
        store.request_cancel(failed.job_id, expected_revision=1)
    _assert_code(cancel_failed, "JOB_CANCEL_UNSUPPORTED")


def test_cancel_request_is_durable_idempotent_and_blocks_queued_start(
    tmp_path: Path,
) -> None:
    clock = _Clock(NOW + timedelta(minutes=1))
    store = JobStore(
        tmp_path / "state",
        clock=clock,
        job_id_provider=lambda: "job:cancel",
    )
    queued = store.create(_spec(), idempotency_key="cancel")

    clock.advance()
    requested = store.request_cancel(queued.job_id, expected_revision=0)
    assert requested.view.state is JobState.QUEUED
    assert requested.cancel_requested
    assert requested.cancel_requested_at == clock.value
    assert requested.view.cancellation_requested is True
    assert requested.view.cancellable is True
    assert requested.revision == 1

    # A transport retry commonly still carries revision 0.  Once the same
    # cancellation fact exists it remains an idempotent success.
    repeated = store.request_cancel(queued.job_id, expected_revision=0)
    assert repeated == requested
    assert repeated.revision == 1

    with pytest.raises(JobStoreError) as start_after_cancel:
        store.transition(
            queued.job_id,
            expected_revision=1,
            state=JobState.RUNNING,
        )
    _assert_code(start_after_cancel, "JOB_CONFLICT")

    clock.advance()
    cancelled = store.transition(
        queued.job_id,
        expected_revision=1,
        state=JobState.CANCELLED,
    )
    assert cancelled.view.state is JobState.CANCELLED
    assert cancelled.view.completed_at == clock.value
    assert cancelled.cancel_requested
    assert cancelled.cancel_requested_at == requested.cancel_requested_at
    assert cancelled.revision == 2
    assert cancelled.view.cancellation_requested is True
    assert cancelled.view.cancellable is False
    assert store.request_cancel(cancelled.job_id, expected_revision=0) == cancelled


def test_only_one_concurrent_writer_can_claim_an_observed_revision(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "state", clock=_Clock(NOW + timedelta(minutes=1)))
    queued = store.create(_spec(), idempotency_key="concurrent-progress")
    running = store.transition(
        queued.job_id,
        expected_revision=0,
        state=JobState.RUNNING,
    )

    def update(index: int) -> str:
        try:
            result = store.update_progress(
                running.job_id,
                expected_revision=1,
                progress=JobProgress(
                    phase="solve",
                    fraction=0.1,
                    message=f"writer-{index}",
                ),
            )
        except JobStoreError as exc:
            return exc.code
        return f"ok:{result.revision}"

    with ThreadPoolExecutor(max_workers=12) as executor:
        outcomes = list(executor.map(update, range(24)))

    assert outcomes.count("ok:2") == 1
    assert outcomes.count("JOB_CONFLICT") == 23
    assert store.get(running.job_id).revision == 2


def test_missing_and_corrupt_jobs_have_stable_structured_errors(tmp_path: Path) -> None:
    store = JobStore(
        tmp_path / "state",
        job_id_provider=lambda: "job:corrupt",
    )

    with pytest.raises(JobStoreError) as missing:
        store.get("job:missing")
    _assert_code(missing, "JOB_NOT_FOUND")

    created = store.create(_spec(), idempotency_key="corrupt")
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE jobs SET spec_json = ? WHERE job_id = ?",
            ("{not-json", created.job_id),
        )

    with pytest.raises(JobStoreError) as corrupt:
        store.get(created.job_id)
    _assert_code(corrupt, "INTERNAL_ERROR")
    assert corrupt.value.error.stage is ErrorStage.INTERNAL

    # An idempotent retry must detect corruption, never overwrite or silently
    # return the damaged record.
    with pytest.raises(JobStoreError) as retry:
        store.create(_spec(), idempotency_key="corrupt")
    _assert_code(retry, "INTERNAL_ERROR")


def test_retry_lineage_requires_a_compatible_terminal_parent_and_survives_restart(
    tmp_path: Path,
) -> None:
    clock = _Clock(NOW + timedelta(minutes=1))
    job_ids = iter(["job:root", "job:child", "job:grandchild"])
    store = JobStore(
        tmp_path / "state",
        clock=clock,
        job_id_provider=lambda: next(job_ids),
    )
    root = store.create(_spec(), idempotency_key="root")

    with pytest.raises(JobStoreError) as active_parent:
        store.create(
            _spec(),
            idempotency_key="too-early",
            parent_job_id=root.job_id,
        )
    _assert_code(active_parent, "JOB_CONFLICT")

    clock.advance()
    root = store.transition(
        root.job_id,
        expected_revision=0,
        state=JobState.FAILED,
        error=_failure(),
    )
    child = store.create(
        _spec(),
        idempotency_key="child",
        parent_job_id=root.job_id,
    )
    assert child.created
    assert child.view.parent_job_id == root.job_id
    assert child.view.root_job_id == root.job_id
    assert child.view.attempt == 2

    clock.advance()
    child = store.transition(
        child.job_id,
        expected_revision=0,
        state=JobState.FAILED,
        error=_failure(),
    )
    grandchild = store.create(
        _spec(),
        idempotency_key="grandchild",
        parent_job_id=child.job_id,
    )
    assert grandchild.view.parent_job_id == child.job_id
    assert grandchild.view.root_job_id == root.job_id
    assert grandchild.view.attempt == 3

    reopened = JobStore(tmp_path / "state").get(grandchild.job_id)
    assert reopened.view.parent_job_id == child.job_id
    assert reopened.view.root_job_id == root.job_id
    assert reopened.view.attempt == 3
    assert not reopened.created


def test_retry_parent_identity_and_idempotency_lineage_cannot_be_changed(
    tmp_path: Path,
) -> None:
    clock = _Clock(NOW + timedelta(minutes=1))
    job_ids = iter(["job:first-parent", "job:second-parent", "job:retry"])
    store = JobStore(
        tmp_path / "state",
        clock=clock,
        job_id_provider=lambda: next(job_ids),
    )
    first = store.create(_spec(), idempotency_key="first-parent")
    clock.advance()
    first = store.transition(
        first.job_id,
        expected_revision=0,
        state=JobState.FAILED,
        error=_failure(),
    )
    second = store.create(_spec(), idempotency_key="second-parent")
    clock.advance()
    second = store.transition(
        second.job_id,
        expected_revision=0,
        state=JobState.FAILED,
        error=_failure(),
    )

    retry = store.create(
        _spec(),
        idempotency_key="retry-key",
        parent_job_id=first.job_id,
    )
    replay = store.create(
        _spec(),
        idempotency_key="retry-key",
        parent_job_id=first.job_id,
    )
    assert replay.job_id == retry.job_id
    assert not replay.created

    with pytest.raises(JobStoreError) as changed_parent:
        store.create(
            _spec(),
            idempotency_key="retry-key",
            parent_job_id=second.job_id,
        )
    _assert_code(changed_parent, "JOB_CONFLICT")

    with pytest.raises(JobStoreError) as missing_parent:
        store.create(
            _spec(),
            idempotency_key="missing-parent",
            parent_job_id="job:not-found",
        )
    _assert_code(missing_parent, "JOB_NOT_FOUND")

    changed_plan = _spec().model_copy(update={"plan_id": f"plan:sha256:{'9' * 64}"})
    with pytest.raises(JobStoreError) as plan_conflict:
        store.create(
            changed_plan,
            idempotency_key="changed-plan",
            parent_job_id=first.job_id,
        )
    _assert_code(plan_conflict, "JOB_CONFLICT")

    changed_kind = _spec().model_copy(
        update={
            "kind": JobSpecKind.R2R_RETARGET,
            "source_robot": JobSpecRobot(
                robot_id="source_robot",
                asset_id=f"asset:sha256:{'c' * 64}",
                config_sha256="c" * 64,
            ),
            "calibration": JobSpecCalibration(
                calibration_id=f"cal:sha256:{'d' * 64}",
                sha256="d" * 64,
            ),
        }
    )
    with pytest.raises(JobStoreError) as kind_conflict:
        store.create(
            changed_kind,
            idempotency_key="changed-kind",
            parent_job_id=first.job_id,
        )
    _assert_code(kind_conflict, "JOB_CONFLICT")


def test_attach_artifacts_is_cas_idempotent_complete_and_compact(tmp_path: Path) -> None:
    clock = _Clock(NOW + timedelta(minutes=10))
    store = JobStore(
        tmp_path / "state",
        clock=clock,
        job_id_provider=lambda: "job:artifacts",
    )
    queued = store.create(_spec(), idempotency_key="artifacts")
    artifacts = [_artifact(queued.job_id, index) for index in range(35)]

    clock.advance()
    attached = store.attach_artifacts(
        queued.job_id,
        expected_revision=0,
        artifacts=artifacts,
    )
    assert attached.revision == 1
    assert len(attached.artifacts) == 35
    assert len(attached.view.artifacts) == 32
    assert attached.view.artifacts == artifacts[:32]
    assert attached.view.artifact_count == 35

    # Exact retries are no-ops even when they carry the revision observed
    # before the original attach.
    replay = store.attach_artifacts(
        queued.job_id,
        expected_revision=0,
        artifacts=artifacts,
    )
    assert replay.revision == 1
    assert replay.artifacts == attached.artifacts

    divergent = _artifact(queued.job_id, 0, sha256="e" * 64)
    with pytest.raises(JobStoreError) as artifact_conflict:
        store.attach_artifacts(
            queued.job_id,
            expected_revision=1,
            artifacts=[divergent],
        )
    _assert_code(artifact_conflict, "JOB_CONFLICT")

    wrong_owner = _artifact("job:someone-else", 40)
    with pytest.raises(JobStoreError) as ownership:
        store.attach_artifacts(
            queued.job_id,
            expected_revision=1,
            artifacts=[wrong_owner],
        )
    _assert_code(ownership, "INVALID_PARAMETER")

    incomplete = _artifact(queued.job_id, 41).model_copy(update={"sha256": None})
    with pytest.raises(JobStoreError) as missing_identity:
        store.attach_artifacts(
            queued.job_id,
            expected_revision=1,
            artifacts=[incomplete],
        )
    _assert_code(missing_identity, "INVALID_PARAMETER")

    reopened = JobStore(tmp_path / "state").get(queued.job_id)
    assert len(reopened.artifacts) == 35
    assert reopened.view.artifact_count == 35
    assert len(reopened.view.artifacts) == 32


def test_terminal_transition_can_atomically_attach_final_artifacts(tmp_path: Path) -> None:
    clock = _Clock(NOW + timedelta(minutes=10))
    store = JobStore(
        tmp_path / "state",
        clock=clock,
        job_id_provider=lambda: "job:atomic-result",
    )
    queued = store.create(_spec(), idempotency_key="atomic-result")
    clock.advance()
    running = store.transition(
        queued.job_id,
        expected_revision=0,
        state=JobState.RUNNING,
    )
    final_artifact = _artifact(running.job_id, 1)

    invalid = final_artifact.model_copy(update={"size_bytes": None})
    with pytest.raises(JobStoreError) as rejected:
        store.transition(
            running.job_id,
            expected_revision=1,
            state=JobState.COMPLETED,
            outcome=JobOutcome.SUCCESS,
            artifacts=[invalid],
        )
    _assert_code(rejected, "INVALID_PARAMETER")
    unchanged = store.get(running.job_id)
    assert unchanged.view.state is JobState.RUNNING
    assert unchanged.revision == 1
    assert unchanged.artifacts == ()

    clock.advance()
    completed = store.transition(
        running.job_id,
        expected_revision=1,
        state=JobState.COMPLETED,
        outcome=JobOutcome.SUCCESS,
        artifacts=[final_artifact],
    )
    assert completed.view.state is JobState.COMPLETED
    assert completed.revision == 2
    assert completed.artifacts == (final_artifact,)
    assert completed.view.artifacts == [final_artifact]
    assert completed.view.artifact_count == 1

    with pytest.raises(JobStoreError) as terminal_attach:
        store.attach_artifacts(
            completed.job_id,
            expected_revision=2,
            artifacts=[_artifact(completed.job_id, 2)],
        )
    _assert_code(terminal_attach, "INVALID_JOB_TRANSITION")


def test_concurrent_artifact_retries_deduplicate_but_different_cas_writers_conflict(
    tmp_path: Path,
) -> None:
    store = JobStore(
        tmp_path / "state",
        clock=_Clock(NOW + timedelta(minutes=10)),
    )
    queued = store.create(_spec(), idempotency_key="artifact-concurrency")
    shared = _artifact(queued.job_id, 1)

    def attach_shared(_: int) -> tuple[str, int]:
        result = store.attach_artifacts(
            queued.job_id,
            expected_revision=0,
            artifacts=[shared],
        )
        return result.job_id, result.revision

    with ThreadPoolExecutor(max_workers=12) as executor:
        retries = list(executor.map(attach_shared, range(24)))

    assert set(retries) == {(queued.job_id, 1)}
    assert store.get(queued.job_id).artifacts == (shared,)

    def attach_different(index: int) -> str:
        try:
            result = store.attach_artifacts(
                queued.job_id,
                expected_revision=1,
                artifacts=[_artifact(queued.job_id, index + 2)],
            )
        except JobStoreError as exc:
            return exc.code
        return f"ok:{result.revision}"

    with ThreadPoolExecutor(max_workers=2) as executor:
        competing = list(executor.map(attach_different, range(2)))

    assert competing.count("ok:2") == 1
    assert competing.count("JOB_CONFLICT") == 1
    persisted = store.get(queued.job_id)
    assert persisted.revision == 2
    assert len(persisted.artifacts) == 2


def test_lookup_by_idempotency_and_list_active_survive_restart(tmp_path: Path) -> None:
    clock = _Clock(NOW + timedelta(minutes=1))
    job_ids = iter(["job:queued", "job:running", "job:terminal"])
    store = JobStore(
        tmp_path / "state",
        clock=clock,
        job_id_provider=lambda: next(job_ids),
    )
    queued = store.create(_spec(), idempotency_key="queued-key")
    clock.advance()
    running = store.create(_spec(), idempotency_key="running-key")
    clock.advance()
    running = store.transition(
        running.job_id,
        expected_revision=0,
        state=JobState.RUNNING,
    )
    clock.advance()
    terminal = store.create(_spec(), idempotency_key="terminal-key")
    clock.advance()
    store.transition(
        terminal.job_id,
        expected_revision=0,
        state=JobState.FAILED,
        error=_failure(),
    )

    reopened = JobStore(tmp_path / "state")
    recovered = reopened.get_by_idempotency_key("running-key")
    assert recovered.job_id == running.job_id
    assert recovered.view.state is JobState.RUNNING
    assert not recovered.created
    active = reopened.list_active()
    assert [job.job_id for job in active] == [queued.job_id, running.job_id]
    assert [job.view.state for job in active] == [JobState.QUEUED, JobState.RUNNING]
    assert all(job.view.cancellable for job in active)

    with pytest.raises(JobStoreError) as missing:
        reopened.get_by_idempotency_key("unknown-key")
    _assert_code(missing, "JOB_NOT_FOUND")


@pytest.mark.parametrize("host_path", ["/srv/private/output.npz", r"C:\private\output.npz"])
def test_job_spec_and_failed_error_reject_host_absolute_paths(
    tmp_path: Path,
    host_path: str,
) -> None:
    store = JobStore(
        tmp_path / "state",
        clock=_Clock(NOW + timedelta(minutes=1)),
        job_id_provider=lambda: "job:portable",
    )
    unsafe_spec = _spec().model_copy(deep=True)
    unsafe_spec.effective_parameters["cache_path"] = host_path

    with pytest.raises(JobStoreError) as unsafe_request:
        store.create(unsafe_spec, idempotency_key="unsafe-request")
    _assert_code(unsafe_request, "INVALID_PARAMETER")

    queued = store.create(_spec(), idempotency_key="portable-request")
    unsafe_error = _failure().model_copy(update={"details": {"solver_log": host_path}})
    with pytest.raises(JobStoreError) as unsafe_failure:
        store.transition(
            queued.job_id,
            expected_revision=0,
            state=JobState.FAILED,
            error=unsafe_error,
        )
    _assert_code(unsafe_failure, "INVALID_PARAMETER")
    assert store.get(queued.job_id).view.state is JobState.QUEUED

    unsafe_artifact = _artifact(
        queued.job_id,
        1,
        metadata={"local_path": host_path},
    )
    with pytest.raises(JobStoreError) as unsafe_metadata:
        store.attach_artifacts(
            queued.job_id,
            expected_revision=0,
            artifacts=[unsafe_artifact],
        )
    _assert_code(unsafe_metadata, "INVALID_PARAMETER")


def test_persisted_summary_with_a_host_path_is_detected_as_corruption(
    tmp_path: Path,
) -> None:
    store = JobStore(
        tmp_path / "state",
        job_id_provider=lambda: "job:unsafe-summary",
    )
    created = store.create(_spec(), idempotency_key="unsafe-summary")
    summary = created.view.summary | {"debug_path": "/srv/private/job.log"}
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE jobs SET summary_json = ? WHERE job_id = ?",
            (
                json.dumps(summary, sort_keys=True, separators=(",", ":")),
                created.job_id,
            ),
        )

    with pytest.raises(JobStoreError) as corrupt:
        store.get(created.job_id)
    _assert_code(corrupt, "INTERNAL_ERROR")
