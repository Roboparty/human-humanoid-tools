"""Compose ready single-item plans into one scalable immutable batch plan."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from hhtools.contracts import (
    AssetInspectionRequest,
    BatchPlan,
    BatchPlanItem,
    BatchPreflightRequest,
    BatchPreflightResponse,
    BatchResourceLimits,
    BatchWorkflow,
    CapabilityResponse,
    ErrorStage,
    InspectionStatus,
    JobSpecKind,
    JobSpecV2,
    OutputPolicy,
    PreflightCheck,
    PreflightCheckLevel,
    PreflightStatus,
)

from .asset_service import AgentAssetService
from .assets import AssetServiceError
from .batch_limits import BatchLimitSnapshot
from .plans import PlanStore, PlanStoreError, compute_plan_id
from .preflight import _check, _fail, _PreflightFailureError, _scheduler_check
from .retarget import RetargetServiceError

BATCH_PLAN_SEMANTICS = "hhtools.batch.plan.v1"


def _spec_sha256(spec: JobSpecV2) -> str:
    encoded = json.dumps(
        spec.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _child_failure(
    error: RetargetServiceError,
    *,
    index: int,
    plan_id: str,
) -> _PreflightFailureError:
    public = error.api_error.model_copy(
        update={
            "stage": ErrorStage.PREFLIGHT,
            "details": {
                **error.api_error.details,
                "item_index": index,
                "item_plan_id": plan_id,
            },
        }
    )
    return _PreflightFailureError(
        public,
        PreflightCheck(
            code=public.code,
            level=PreflightCheckLevel.ERROR,
            message=public.message,
            details=public.details,
            next_action=public.next_action,
        ),
    )


class BatchPreflightService:
    """Freeze ordered ready H2R or R2R plans under the current optional policy."""

    def __init__(
        self,
        plan_store: PlanStore,
        asset_service: AgentAssetService,
        child_specs,
        *,
        capabilities_provider: Callable[[], CapabilityResponse],
        limits_provider: Callable[[], BatchLimitSnapshot] = BatchLimitSnapshot,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        request_id_provider: Callable[[], str] = lambda: f"req_{uuid.uuid4().hex}",
    ) -> None:
        self._plan_store = plan_store
        self._asset_service = asset_service
        self._child_specs = child_specs
        self._capabilities_provider = capabilities_provider
        self._limits_provider = limits_provider
        self._clock = clock
        self._request_id_provider = request_id_provider

    def _estimated_frames(self, spec: JobSpecV2, *, index: int) -> int:
        try:
            inspection = self._asset_service.inspect(
                AssetInspectionRequest(
                    asset_id=spec.inputs[0].asset_id,
                    verify_hashes=True,
                    parse_content=True,
                )
            )
        except (AssetServiceError, TypeError, ValueError) as error:
            if isinstance(error, AssetServiceError):
                public = error.api_error
                raise _PreflightFailureError(
                    public,
                    PreflightCheck(
                        code=public.code,
                        level=PreflightCheckLevel.ERROR,
                        message=public.message,
                        details={**public.details, "item_index": index},
                    ),
                ) from error
            _fail(
                "INVALID_PARAMETER",
                "A batch input inspection request could not be constructed.",
                details={"item_index": index},
            )
        if inspection.status is InspectionStatus.INVALID or inspection.frame_count is None:
            _fail(
                "BATCH_INPUT_INVALID",
                "A batch input no longer has a verified positive frame count.",
                details={"item_index": index},
            )
        frames = int(inspection.frame_count)
        limit = spec.effective_parameters.get("limit_frames")
        if isinstance(limit, int) and not isinstance(limit, bool):
            frames = min(frames, limit)
        if frames <= 0:
            _fail(
                "BATCH_INPUT_INVALID",
                "A batch input resolves to zero executable frames.",
                details={"item_index": index},
            )
        return frames

    def preflight_batch(self, request: BatchPreflightRequest) -> BatchPreflightResponse:
        request_id = self._request_id_provider()
        checks: list[PreflightCheck] = []
        try:
            if request.output_policy is not OutputPolicy.CREATE_NEW:
                _fail(
                    "UNSUPPORTED_OUTPUT_POLICY",
                    "Managed Agent batches support only create_new output policy.",
                    details={"output_policy": request.output_policy.value},
                )
            configured_limits = self._limits_provider()
            if (
                configured_limits.max_batch_items
                and len(request.item_plan_ids) > configured_limits.max_batch_items
            ):
                _fail(
                    "BATCH_RESOURCE_LIMIT_EXCEEDED",
                    "The batch item count exceeds the configured service limit.",
                    details={"max_batch_items": configured_limits.max_batch_items},
                )
            expected_kind = (
                JobSpecKind.RETARGET
                if request.workflow is BatchWorkflow.H2R
                else JobSpecKind.R2R_RETARGET
            )
            items: list[BatchPlanItem] = []
            target_robot = None
            source_robot = None
            run_mode: str | None = None
            total_frames = 0
            seen_inputs: set[str] = set()
            for index, plan_id in enumerate(request.item_plan_ids):
                try:
                    spec = self._child_specs.get_job_spec(plan_id)
                except RetargetServiceError as error:
                    raise _child_failure(error, index=index, plan_id=plan_id) from error
                if spec.kind is not expected_kind or len(spec.inputs) != 1:
                    _fail(
                        "BATCH_WORKFLOW_MISMATCH",
                        "A child plan belongs to a different workflow.",
                        details={"item_index": index, "workflow": request.workflow.value},
                    )
                if spec.output_policy is not OutputPolicy.CREATE_NEW:
                    _fail(
                        "UNSUPPORTED_OUTPUT_POLICY",
                        "Every child plan must use create_new output policy.",
                        details={"item_index": index},
                    )
                child_run_mode = spec.effective_parameters.get("run_mode")
                if child_run_mode not in {"smoke", "full"}:
                    _fail(
                        "BATCH_CHILD_PLAN_INVALID",
                        "A child plan has no supported run mode.",
                        details={"item_index": index},
                    )
                if run_mode is None:
                    run_mode = str(child_run_mode)
                elif run_mode != child_run_mode:
                    _fail(
                        "BATCH_RUN_MODE_MISMATCH",
                        "All child plans in one batch must use the same run mode.",
                        details={"item_index": index},
                    )
                if target_robot is None:
                    target_robot = spec.robot
                elif target_robot != spec.robot:
                    _fail(
                        "BATCH_ROBOT_MISMATCH",
                        "All child plans must use the same target robot identity.",
                        details={"item_index": index},
                    )
                if request.workflow is BatchWorkflow.R2R:
                    if spec.source_robot is None:
                        _fail(
                            "BATCH_CHILD_PLAN_INVALID",
                            "An R2R child plan is missing its source robot identity.",
                            details={"item_index": index},
                        )
                    if source_robot is None:
                        source_robot = spec.source_robot
                    elif source_robot != spec.source_robot:
                        _fail(
                            "BATCH_ROBOT_MISMATCH",
                            "All R2R child plans must use the same source robot identity.",
                            details={"item_index": index},
                        )
                elif spec.source_robot is not None:
                    _fail(
                        "BATCH_CHILD_PLAN_INVALID",
                        "An H2R child plan cannot declare a source robot.",
                        details={"item_index": index},
                    )

                child_input = spec.inputs[0]
                if child_input.asset_id in seen_inputs:
                    _fail(
                        "BATCH_INPUT_DUPLICATE",
                        "A batch cannot execute the same input asset twice.",
                        details={"item_index": index},
                    )
                seen_inputs.add(child_input.asset_id)
                estimated_frames = self._estimated_frames(spec, index=index)
                total_frames += estimated_frames
                if (
                    configured_limits.max_batch_total_frames
                    and total_frames > configured_limits.max_batch_total_frames
                ):
                    _fail(
                        "BATCH_RESOURCE_LIMIT_EXCEEDED",
                        "The batch frame estimate exceeds the configured service limit.",
                        details={
                            "max_batch_total_frames": (configured_limits.max_batch_total_frames),
                            "item_index": index,
                        },
                    )
                calibration = spec.calibration
                item_digest = child_input.sha256[:12]
                items.append(
                    BatchPlanItem(
                        index=index,
                        item_id=f"item-{index:08d}-{item_digest}",
                        workflow=request.workflow,
                        plan_id=spec.plan_id,
                        input_asset_id=child_input.asset_id,
                        input_digest=child_input.sha256,
                        target_robot_id=spec.robot.robot_id,
                        target_robot_asset_id=spec.robot.asset_id,
                        target_robot_digest=spec.robot.config_sha256,
                        source_robot_id=(
                            spec.source_robot.robot_id if spec.source_robot is not None else None
                        ),
                        source_robot_asset_id=(
                            spec.source_robot.asset_id if spec.source_robot is not None else None
                        ),
                        source_robot_digest=(
                            spec.source_robot.config_sha256
                            if spec.source_robot is not None
                            else None
                        ),
                        backend=spec.backend,
                        calibration_id=(
                            calibration.calibration_id if calibration is not None else None
                        ),
                        calibration_digest=(
                            calibration.sha256 if calibration is not None else None
                        ),
                        output_format=str(spec.effective_parameters["output_format"]),
                        estimated_frames=estimated_frames,
                        job_spec_sha256=_spec_sha256(spec),
                    )
                )

            assert target_robot is not None
            assert run_mode is not None
            limits = BatchResourceLimits(
                max_items=configured_limits.max_batch_items,
                max_total_frames=configured_limits.max_batch_total_frames,
                item_count=len(items),
                estimated_total_frames=total_frames,
                failure_limit=configured_limits.max_batch_items,
            )
            checks.append(
                _check(
                    "BATCH_ITEMS_READY",
                    PreflightCheckLevel.PASS,
                    "Every child plan is current and belongs to the selected workflow.",
                    details={
                        "workflow": request.workflow.value,
                        "item_count": len(items),
                        "run_mode": run_mode,
                    },
                )
            )
            checks.append(
                _check(
                    "BATCH_RESOURCE_POLICY_ACCEPTED",
                    PreflightCheckLevel.PASS,
                    "The ordered batch satisfies the configured item and frame policy.",
                    details=limits.model_dump(mode="json"),
                )
            )
            checks.append(_scheduler_check(self._capabilities_provider().scheduler))

            canonical_payload = {
                "semantics": BATCH_PLAN_SEMANTICS,
                "workflow": request.workflow.value,
                "run_mode": run_mode,
                "items": [item.model_dump(mode="json") for item in items],
                "target_robot": {
                    "robot_id": target_robot.robot_id,
                    "asset_id": target_robot.asset_id,
                    "digest": target_robot.config_sha256,
                },
                "source_robot": (
                    {
                        "robot_id": source_robot.robot_id,
                        "asset_id": source_robot.asset_id,
                        "digest": source_robot.config_sha256,
                    }
                    if source_robot is not None
                    else None
                ),
                "output": {"policy": request.output_policy.value},
                "resource_limits": limits.model_dump(mode="json"),
            }
            plan_id = compute_plan_id(canonical_payload)
            try:
                plan = self._plan_store.get(plan_id)
            except PlanStoreError as error:
                if error.code != "PLAN_NOT_FOUND":
                    raise
                candidate = BatchPlan(
                    plan_id=plan_id,
                    created_at=self._clock(),
                    workflow=request.workflow,
                    run_mode=run_mode,
                    items=items,
                    target_robot_id=target_robot.robot_id,
                    target_robot_asset_id=target_robot.asset_id,
                    target_robot_digest=target_robot.config_sha256,
                    source_robot_id=(source_robot.robot_id if source_robot is not None else None),
                    source_robot_asset_id=(
                        source_robot.asset_id if source_robot is not None else None
                    ),
                    source_robot_digest=(
                        source_robot.config_sha256 if source_robot is not None else None
                    ),
                    output_policy=request.output_policy,
                    resource_limits=limits,
                )
                try:
                    plan = self._plan_store.put_if_absent(candidate, canonical_payload)
                except PlanStoreError as conflict:
                    if conflict.code != "PLAN_CONFLICT":
                        raise
                    plan = self._plan_store.get(plan_id)
                    if self._plan_store.get_payload(plan_id) != canonical_payload:
                        raise
            if not isinstance(plan, BatchPlan):
                _fail("PLAN_CONFLICT", "The persisted plan belongs to another workflow.")
            return BatchPreflightResponse(
                request_id=request_id,
                status=PreflightStatus.READY,
                plan=plan,
                checks=checks,
            )
        except _PreflightFailureError as failure:
            checks.append(failure.check)
            return BatchPreflightResponse(
                request_id=request_id,
                status=PreflightStatus.REJECTED,
                checks=checks,
                error=failure.error,
            )
        except PlanStoreError as failure:
            public = failure.api_error
            checks.append(
                PreflightCheck(
                    code=public.code,
                    level=PreflightCheckLevel.ERROR,
                    message=public.message,
                    details=public.details,
                    next_action=public.next_action,
                )
            )
            return BatchPreflightResponse(
                request_id=request_id,
                status=PreflightStatus.REJECTED,
                checks=checks,
                error=public,
            )


__all__ = ["BATCH_PLAN_SEMANTICS", "BatchPreflightService"]
