from __future__ import annotations

import json
import threading
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hhtools.agent.batch_job_executor import BatchJobExecutor
from hhtools.contracts import (
    ApiError,
    ErrorStage,
    JobOutcome,
    JobProgress,
    JobSpecBatchItem,
    JobSpecInput,
    JobSpecKind,
    JobSpecProvenance,
    JobSpecRobot,
    JobSpecV2,
    JobState,
    OutputPolicy,
)
from hhtools.services.artifacts import ArtifactStore
from hhtools.services.job_store import JobStore
from hhtools.services.jobs import (
    JobCancelledError,
    JobExecutionContext,
    JobExecutionError,
    JobExecutionResult,
    JobManager,
)
from hhtools.web.jobs.job_scheduler import JobScheduler

NOW = datetime(2026, 9, 9, tzinfo=UTC)
TARGET = JobSpecRobot(
    robot_id="target_bot",
    asset_id=f"asset:sha256:{'b' * 64}",
    config_sha256="b" * 64,
)
PROVENANCE = JobSpecProvenance(
    hhtools_git_commit="test",
    hhtools_dirty=False,
    python="3.12",
)


def _child(marker: str, index: int) -> JobSpecBatchItem:
    source = JobSpecInput(
        asset_id=f"asset:sha256:{marker * 64}",
        sha256=marker * 64,
    )
    return JobSpecBatchItem(
        item_id=f"item-{index:04d}-{marker * 12}",
        plan_id=f"plan:sha256:{marker * 64}",
        kind=JobSpecKind.RETARGET,
        input=source,
        robot=TARGET,
        backend="newton",
        effective_parameters={
            "run_mode": "smoke",
            "limit_frames": 1,
            "output_format": "csv",
        },
        output_policy=OutputPolicy.CREATE_NEW,
        provenance=PROVENANCE,
        created_at=NOW,
    )


def _spec() -> JobSpecV2:
    items = [_child("1", 0), _child("2", 1)]
    return JobSpecV2(
        kind=JobSpecKind.BATCH_RETARGET,
        plan_id=f"plan:sha256:{'a' * 64}",
        inputs=[item.input for item in items],
        robot=TARGET,
        source_robot=None,
        calibration=None,
        batch_items=items,
        backend="batch",
        effective_parameters={
            "workflow": "h2r",
            "run_mode": "smoke",
            "resource_limits": {
                "max_items": 0,
                "max_total_frames": 0,
                "item_count": 2,
                "estimated_total_frames": 2,
                "execution_concurrency": 1,
                "failure_limit": 0,
            },
        },
        output_policy=OutputPolicy.CREATE_NEW,
        provenance=PROVENANCE,
        created_at=NOW,
    )


def _context(
    tmp_path: Path,
    spec: JobSpecV2,
    cancellation: threading.Event | None = None,
):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    progress: list[JobProgress] = []
    context = JobExecutionContext(
        job_id="job:batch-test",
        spec=spec,
        artifact_store=artifacts,
        cancellation_event=cancellation or threading.Event(),
        progress_callback=lambda item, _delay: progress.append(item),
    )
    return context, artifacts, progress


class _Children:
    def __init__(
        self,
        *,
        failures: set[str] | None = None,
        cancellation: threading.Event | None = None,
    ) -> None:
        self.failures = failures or set()
        self.cancellation = cancellation
        self.calls: list[str] = []

    def __call__(
        self,
        spec: JobSpecV2,
        context: JobExecutionContext,
    ) -> JobExecutionResult:
        self.calls.append(spec.plan_id)
        context.report_progress(
            phase="solving",
            fraction=0.5,
            message="halfway",
            poll_after_ms=500,
        )
        if self.cancellation is not None:
            self.cancellation.set()
            context.raise_if_cancelled()
        context.publish_json(kind="preview", document={"plan_id": spec.plan_id})
        if spec.plan_id in self.failures:
            raise JobExecutionError(
                ApiError(
                    code="SOLVER_FAILED",
                    message="This item failed.",
                    stage=ErrorStage.EXECUTION,
                )
            )
        context.publish_bytes(
            kind="retargeted_motion",
            payload=f"result:{spec.plan_id}".encode(),
            format="csv",
            media_type="text/csv",
        )
        return JobExecutionResult(
            outcome=JobOutcome.REVIEW_REQUIRED,
            summary={"num_frames": 1},
            execution_provenance={
                "executor": "fake-child",
                "backend": "newton",
                "device": "cpu",
                "device_kind": "cpu",
                "precision": "float32",
                "fallback_used": False,
            },
        )


def _artifact_json(store: ArtifactStore, context: JobExecutionContext, kind: str):
    descriptor = next(item for item in context.published_artifacts() if item.kind == kind)
    return json.loads(store.get(descriptor.artifact_id, verify=True).path.read_text())


def test_batch_executor_maps_child_progress_and_publishes_report_and_archive(
    tmp_path: Path,
) -> None:
    spec = _spec()
    context, artifacts, progress = _context(tmp_path, spec)
    children = _Children()
    validated: list[JobSpecV2] = []
    executor = BatchJobExecutor(
        children,
        validate_spec=validated.append,
        temporary_root=tmp_path / "temporary",
    )

    result = executor(spec, context)

    assert result.outcome is JobOutcome.REVIEW_REQUIRED
    assert result.summary["succeeded_items"] == 2
    assert result.summary["failed_items"] == 0
    assert validated == [spec]
    assert children.calls == [item.plan_id for item in spec.batch_items or []]
    assert progress[-1].completed_items == 2
    assert progress[-1].total_items == 2
    assert all(
        progress[index].fraction <= progress[index + 1].fraction
        for index in range(len(progress) - 1)
    )

    report = _artifact_json(artifacts, context, "batch_report")
    assert report["outcome"] == "review_required"
    assert report["succeeded_items"] == 2
    assert [item["index"] for item in report["items"]] == [0, 1]
    item_artifacts = [
        descriptor
        for descriptor in context.published_artifacts()
        if descriptor.kind in {"preview", "retargeted_motion"}
    ]
    assert all("batch_item_id" in descriptor.metadata for descriptor in item_artifacts)

    archive_descriptor = next(
        item for item in context.published_artifacts() if item.kind == "batch_archive"
    )
    archive_path = artifacts.get(archive_descriptor.artifact_id, verify=True).path
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "batch-report.json",
            f"0001_{'1' * 12}.csv",
            f"0002_{'2' * 12}.csv",
        }


@pytest.mark.parametrize(
    ("failed_plans", "expected"),
    [
        ({f"plan:sha256:{'1' * 64}"}, JobOutcome.PARTIAL),
        (
            {f"plan:sha256:{'1' * 64}", f"plan:sha256:{'2' * 64}"},
            JobOutcome.REJECTED,
        ),
    ],
)
def test_batch_executor_continues_structured_failures_and_aggregates_outcome(
    tmp_path: Path,
    failed_plans: set[str],
    expected: JobOutcome,
) -> None:
    spec = _spec()
    context, artifacts, _progress = _context(tmp_path, spec)
    children = _Children(failures=failed_plans)
    executor = BatchJobExecutor(
        children,
        validate_spec=lambda _spec_value: None,
        temporary_root=tmp_path / "temporary",
    )

    result = executor(spec, context)

    assert result.outcome is expected
    assert len(result.failures) == len(failed_plans)
    assert children.calls == [item.plan_id for item in spec.batch_items or []]
    report = _artifact_json(artifacts, context, "batch_report")
    assert report["outcome"] == expected.value
    assert report["failed_items"] == len(failed_plans)


def test_batch_cancellation_stops_current_child_and_never_starts_remaining_item(
    tmp_path: Path,
) -> None:
    spec = _spec()
    cancellation = threading.Event()
    context, _artifacts, _progress = _context(tmp_path, spec, cancellation)
    children = _Children(cancellation=cancellation)
    executor = BatchJobExecutor(
        children,
        validate_spec=lambda _spec_value: None,
        temporary_root=tmp_path / "temporary",
    )

    with pytest.raises(JobCancelledError):
        executor(spec, context)

    assert children.calls == [spec.batch_items[0].plan_id]  # type: ignore[index]
    assert all(
        artifact.kind not in {"batch_report", "batch_archive"}
        for artifact in context.published_artifacts()
    )


def test_partial_batch_persists_failure_report_and_retries_as_a_whole_attempt(
    tmp_path: Path,
) -> None:
    spec = _spec()
    failed_plan = spec.batch_items[0].plan_id  # type: ignore[index]
    children = _Children(failures={failed_plan})
    executor = BatchJobExecutor(
        children,
        validate_spec=lambda _spec_value: None,
        temporary_root=tmp_path / "temporary",
    )

    class Specs:
        def get_job_spec(self, plan_id: str) -> JobSpecV2:
            assert plan_id == spec.plan_id
            return JobSpecV2.model_validate_json(spec.model_dump_json())

    scheduler = JobScheduler(max_running_jobs=1, max_queued_jobs=1)
    store = JobStore(tmp_path / "state")
    artifacts = ArtifactStore(tmp_path / "state")
    manager = JobManager(
        store,
        artifacts,
        Specs(),  # type: ignore[arg-type]
        scheduler,
        executor=executor,
    )

    def terminal(job_id: str):
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            view = manager.get_job(job_id)
            if view.state in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
                return view
            time.sleep(0.005)
        raise AssertionError("batch job did not become terminal")

    first = manager.start_job(spec.plan_id, idempotency_key="batch-first")
    completed = terminal(first.job_id)

    assert completed.state is JobState.COMPLETED
    assert completed.outcome is JobOutcome.PARTIAL
    assert completed.progress.completed_items == 2
    assert {item.kind for item in manager.list_artifacts(completed.job_id)} >= {
        "batch_report",
        "batch_archive",
        "failure_report",
        "manifest",
    }

    retried = manager.retry_job(completed.job_id, idempotency_key="batch-retry")
    retry_completed = terminal(retried.job_id)
    assert retry_completed.parent_job_id == completed.job_id
    assert retry_completed.root_job_id == completed.job_id
    assert retry_completed.attempt == 2
    assert len(children.calls) == 4
    assert manager.get_job(completed.job_id).attempt == 1
    assert scheduler.shutdown(wait=True, timeout=2.0)
