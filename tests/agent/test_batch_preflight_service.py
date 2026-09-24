from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from hhtools.contracts import (
    AssetCategory,
    AssetInspection,
    AssetKind,
    BatchPreflightRequest,
    BatchWorkflow,
    CapabilityResponse,
    InspectionStatus,
    JobSpecCalibration,
    JobSpecInput,
    JobSpecKind,
    JobSpecProvenance,
    JobSpecRobot,
    JobSpecV2,
    OutputPolicy,
    PreflightStatus,
    SchedulerCapability,
)
from hhtools.services.batch_limits import BatchLimitSnapshot
from hhtools.services.batch_preflight import BatchPreflightService
from hhtools.services.batch_retarget import BatchRetargetService
from hhtools.services.plans import PlanStore, PlanStoreError
from hhtools.services.retarget import RetargetServiceError


def _spec(
    marker: str,
    *,
    workflow: BatchWorkflow = BatchWorkflow.H2R,
    target_marker: str = "b",
    input_marker: str | None = None,
    frames: int | None = 10,
    run_mode: str = "smoke",
) -> JobSpecV2:
    def digest(value: str) -> str:
        return value if len(value) == 64 else value * 64

    input_marker = input_marker or marker
    plan_digest = digest(marker)
    input_digest = digest(input_marker)
    target_digest = digest(target_marker)
    r2r = workflow is BatchWorkflow.R2R
    return JobSpecV2(
        kind=JobSpecKind.R2R_RETARGET if r2r else JobSpecKind.RETARGET,
        plan_id=f"plan:sha256:{plan_digest}",
        inputs=[
            JobSpecInput(
                asset_id=f"asset:sha256:{input_digest}",
                sha256=input_digest,
            )
        ],
        robot=JobSpecRobot(
            robot_id="target_bot",
            asset_id=f"asset:sha256:{target_digest}",
            config_sha256=target_digest,
        ),
        source_robot=(
            JobSpecRobot(
                robot_id="source_bot",
                asset_id=f"asset:sha256:{'c' * 64}",
                config_sha256="c" * 64,
            )
            if r2r
            else None
        ),
        calibration=(
            JobSpecCalibration(
                calibration_id=f"cal:sha256:{'d' * 64}",
                sha256="d" * 64,
            )
            if r2r
            else None
        ),
        batch_items=None,
        backend="newton",
        effective_parameters={
            "run_mode": run_mode,
            "limit_frames": frames,
            "output_format": "csv",
        },
        output_policy=OutputPolicy.CREATE_NEW,
        provenance=JobSpecProvenance(
            hhtools_git_commit="test",
            hhtools_dirty=False,
            python="3.12",
        ),
        created_at=datetime(2026, 9, 9, tzinfo=UTC),
    )


class _ChildSpecs:
    def __init__(self, *specs: JobSpecV2) -> None:
        self.specs = {spec.plan_id: spec for spec in specs}

    def get_job_spec(self, plan_id: str) -> JobSpecV2:
        try:
            spec = self.specs[plan_id]
        except KeyError as error:
            from hhtools.contracts import ApiError, ErrorStage

            raise RetargetServiceError(
                ApiError(
                    code="PLAN_NOT_FOUND",
                    message="Child plan not found.",
                    stage=ErrorStage.PREFLIGHT,
                )
            ) from error
        return JobSpecV2.model_validate_json(spec.model_dump_json())


class _Assets:
    def __init__(self, frame_counts: dict[str, int]) -> None:
        self.frame_counts = frame_counts

    def inspect(self, request) -> AssetInspection:
        frames = self.frame_counts[request.asset_id]
        return AssetInspection(
            asset_id=request.asset_id,
            status=InspectionStatus.VALID,
            kind=(
                AssetKind.ROBOT_TRAJECTORY_BUNDLE
                if request.asset_id.endswith("c" * 64)
                else AssetKind.MOTION_BUNDLE
            ),
            category=AssetCategory.PLAIN_MOTION,
            frame_count=frames,
            frame_rate_hz=30.0,
        )


def _service(
    tmp_path: Path,
    *specs: JobSpecV2,
    frame_counts: dict[str, int] | None = None,
    batch_limits: BatchLimitSnapshot | None = None,
):
    plans = PlanStore(tmp_path / "state")
    children = _ChildSpecs(*specs)
    assets = _Assets(frame_counts or {spec.inputs[0].asset_id: 100 for spec in specs})
    service = BatchPreflightService(
        plans,
        assets,  # type: ignore[arg-type]
        children,
        capabilities_provider=lambda: CapabilityResponse(
            service_version="test",
            scheduler=SchedulerCapability(
                max_running_jobs=1,
                max_queued_jobs=1,
                mode="limited",
            ),
        ),
        limits_provider=lambda: batch_limits or BatchLimitSnapshot(),
        request_id_provider=lambda: "req_batch_test",
    )
    return service, plans, children


@pytest.mark.parametrize("workflow", [BatchWorkflow.H2R, BatchWorkflow.R2R])
def test_batch_preflight_freezes_ordered_child_specs_and_resource_limits(
    tmp_path: Path,
    workflow: BatchWorkflow,
) -> None:
    first = _spec("1", workflow=workflow)
    second = _spec("2", workflow=workflow)
    service, plans, children = _service(tmp_path, first, second)

    response = service.preflight_batch(
        BatchPreflightRequest(
            workflow=workflow,
            item_plan_ids=[first.plan_id, second.plan_id],
        )
    )

    assert response.status is PreflightStatus.READY
    assert response.plan is not None
    assert [item.plan_id for item in response.plan.items] == [
        first.plan_id,
        second.plan_id,
    ]
    assert [item.index for item in response.plan.items] == [0, 1]
    assert response.plan.resource_limits.item_count == 2
    assert response.plan.resource_limits.estimated_total_frames == 20
    assert response.plan.resource_limits.execution_concurrency == 1
    assert response.plan.resource_limits.max_items == 0
    assert response.plan.resource_limits.max_total_frames == 0
    assert plans.get_payload(response.plan.plan_id)["semantics"] == "hhtools.batch.plan.v1"

    projector = BatchRetargetService(
        plans,
        children,
        provenance_provider=lambda: {
            "hhtools_git_commit": "test",
            "hhtools_dirty": False,
            "python": "3.12",
        },
    )
    spec = projector.get_job_spec(response.plan.plan_id)
    assert spec.kind is JobSpecKind.BATCH_RETARGET
    assert spec.backend == "batch"
    assert spec.batch_items is not None
    assert [item.plan_id for item in spec.batch_items] == [first.plan_id, second.plan_id]
    assert spec.inputs == [first.inputs[0], second.inputs[0]]
    assert (spec.source_robot is not None) is (workflow is BatchWorkflow.R2R)

    tampered = response.plan.model_copy(update={"target_robot_id": "another_target"})
    with pytest.raises(PlanStoreError) as conflict:
        PlanStore(tmp_path / "tampered-state").put_if_absent(
            tampered,
            plans.get_payload(response.plan.plan_id),
        )
    assert conflict.value.code in {"INVALID_PARAMETER", "PLAN_CONFLICT"}

    children.specs[first.plan_id] = first.model_copy(
        update={
            "effective_parameters": {
                **first.effective_parameters,
                "output_format": "pkl",
            }
        }
    )
    with pytest.raises(RetargetServiceError) as stale:
        projector.get_job_spec(response.plan.plan_id)
    assert stale.value.code == "PLAN_STALE"


def test_batch_preflight_rejects_mixed_targets_and_duplicate_inputs(tmp_path: Path) -> None:
    first = _spec("1")
    other_target = _spec("2", target_marker="e")
    duplicate_input = _spec("3", input_marker="1")
    service, _plans, _children = _service(tmp_path, first, other_target, duplicate_input)

    mixed = service.preflight_batch(
        BatchPreflightRequest(
            workflow="h2r",
            item_plan_ids=[first.plan_id, other_target.plan_id],
        )
    )
    duplicated = service.preflight_batch(
        BatchPreflightRequest(
            workflow="h2r",
            item_plan_ids=[first.plan_id, duplicate_input.plan_id],
        )
    )

    assert mixed.status is PreflightStatus.REJECTED
    assert mixed.error is not None and mixed.error.code == "BATCH_ROBOT_MISMATCH"
    assert duplicated.status is PreflightStatus.REJECTED
    assert duplicated.error is not None and duplicated.error.code == "BATCH_INPUT_DUPLICATE"


def test_batch_preflight_rejects_missing_or_wrong_workflow_child(tmp_path: Path) -> None:
    h2r = _spec("1")
    r2r = _spec("2", workflow=BatchWorkflow.R2R)
    service, _plans, _children = _service(tmp_path, h2r, r2r)

    wrong = service.preflight_batch(
        BatchPreflightRequest(workflow="h2r", item_plan_ids=[r2r.plan_id])
    )
    missing = service.preflight_batch(
        BatchPreflightRequest(
            workflow="h2r",
            item_plan_ids=[f"plan:sha256:{'f' * 64}"],
        )
    )

    assert wrong.status is PreflightStatus.REJECTED
    assert wrong.error is not None and wrong.error.code == "BATCH_WORKFLOW_MISMATCH"
    assert missing.status is PreflightStatus.REJECTED
    assert missing.error is not None and missing.error.code == "PLAN_NOT_FOUND"


def test_batch_preflight_rejects_mixed_modes_and_total_frame_overflow(
    tmp_path: Path,
) -> None:
    smoke = _spec("1")
    full_a = _spec("2", frames=None, run_mode="full")
    full_b = _spec("3", frames=None, run_mode="full")
    frame_counts = {
        smoke.inputs[0].asset_id: 10,
        full_a.inputs[0].asset_id: 60_000,
        full_b.inputs[0].asset_id: 60_000,
    }
    service, _plans, _children = _service(
        tmp_path,
        smoke,
        full_a,
        full_b,
        frame_counts=frame_counts,
        batch_limits=BatchLimitSnapshot(max_batch_total_frames=100_000),
    )

    mixed = service.preflight_batch(
        BatchPreflightRequest(
            workflow="h2r",
            item_plan_ids=[smoke.plan_id, full_a.plan_id],
        )
    )
    oversized = service.preflight_batch(
        BatchPreflightRequest(
            workflow="h2r",
            item_plan_ids=[full_a.plan_id, full_b.plan_id],
        )
    )

    assert mixed.status is PreflightStatus.REJECTED
    assert mixed.error is not None and mixed.error.code == "BATCH_RUN_MODE_MISMATCH"
    assert oversized.status is PreflightStatus.REJECTED
    assert oversized.error is not None and oversized.error.code == "BATCH_RESOURCE_LIMIT_EXCEEDED"

    unlimited_service, _unlimited_plans, _unlimited_children = _service(
        tmp_path / "unlimited",
        full_a,
        full_b,
        frame_counts=frame_counts,
    )
    unlimited = unlimited_service.preflight_batch(
        BatchPreflightRequest(
            workflow="h2r",
            item_plan_ids=[full_a.plan_id, full_b.plan_id],
        )
    )
    assert unlimited.status is PreflightStatus.READY
    assert unlimited.plan is not None
    assert unlimited.plan.resource_limits.max_items == 0
    assert unlimited.plan.resource_limits.max_total_frames == 0
    assert unlimited.plan.resource_limits.estimated_total_frames == 120_000

    item_limited_service, _limited_plans, _limited_children = _service(
        tmp_path / "item-limited",
        smoke,
        full_a,
        frame_counts=frame_counts,
        batch_limits=BatchLimitSnapshot(max_batch_items=1),
    )
    item_limited = item_limited_service.preflight_batch(
        BatchPreflightRequest(
            workflow="h2r",
            item_plan_ids=[smoke.plan_id, full_a.plan_id],
        )
    )
    assert item_limited.status is PreflightStatus.REJECTED
    assert (
        item_limited.error is not None
        and item_limited.error.code == "BATCH_RESOURCE_LIMIT_EXCEEDED"
    )


def test_default_batch_policy_accepts_more_than_the_legacy_item_cap(tmp_path: Path) -> None:
    specs = tuple(_spec(f"{index:064x}") for index in range(1, 41))
    service, _plans, _children = _service(tmp_path, *specs)

    response = service.preflight_batch(
        BatchPreflightRequest(
            workflow="h2r",
            item_plan_ids=[spec.plan_id for spec in specs],
        )
    )

    assert response.status is PreflightStatus.READY
    assert response.plan is not None
    assert response.plan.resource_limits.max_items == 0
    assert response.plan.resource_limits.item_count == 40
    assert response.plan.resource_limits.estimated_total_frames == 400
