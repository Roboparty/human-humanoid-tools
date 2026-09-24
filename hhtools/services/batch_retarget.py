"""Project immutable batch plans into auditable parent and child JobSpecs."""

from __future__ import annotations

from hhtools.contracts import (
    BatchPlan,
    JobSpecBatchItem,
    JobSpecKind,
    JobSpecProvenance,
    JobSpecRobot,
    JobSpecV2,
)

from .batch_preflight import BATCH_PLAN_SEMANTICS, _spec_sha256
from .plans import PlanStore, PlanStoreError
from .r2r_preflight import R2R_PLAN_SEMANTICS
from .retarget import (
    ProvenanceProvider,
    RetargetServiceError,
    _default_provenance,
    _service_error,
    _snapshot_provenance,
)


def _preflight_error(plan_id: str, message: str) -> RetargetServiceError:
    from hhtools.contracts import NextAction

    return _service_error(
        "PLAN_STALE",
        message,
        details={"plan_id": plan_id},
        next_action=NextAction(
            actor="agent",
            action="preflight_batch",
            message="Preflight every child again, then create a new batch plan.",
            parameters={"plan_id": plan_id},
        ),
    )


class BatchRetargetService:
    """Rebuild every child spec before returning one immutable batch JobSpec."""

    def __init__(
        self,
        plan_store: PlanStore,
        child_specs,
        *,
        provenance_provider: ProvenanceProvider = _default_provenance,
    ) -> None:
        self._plan_store = plan_store
        self._child_specs = child_specs
        self._provenance_json = _snapshot_provenance(provenance_provider)

    def _plan(self, plan_id: str) -> BatchPlan:
        try:
            plan = self._plan_store.get(plan_id)
            payload = self._plan_store.get_payload(plan_id)
        except PlanStoreError as error:
            raise RetargetServiceError(error.api_error) from error
        if not isinstance(plan, BatchPlan) or payload.get("semantics") != BATCH_PLAN_SEMANTICS:
            raise _service_error(
                "UNSUPPORTED_PLAN_SEMANTICS",
                "The requested plan is not a supported batch plan.",
                details={"plan_id": plan_id},
            )
        return plan

    def get_job_spec(self, plan_id: str) -> JobSpecV2:
        plan = self._plan(plan_id)
        child_items: list[JobSpecBatchItem] = []
        for item in plan.items:
            try:
                child = self._child_specs.get_job_spec(item.plan_id)
            except RetargetServiceError as error:
                raise _preflight_error(
                    plan.plan_id,
                    "A child plan no longer resolves to a current JobSpec.",
                ) from error
            calibration = child.calibration
            source = child.source_robot
            projection_matches = bool(
                child.kind in {JobSpecKind.RETARGET, JobSpecKind.R2R_RETARGET}
                and (
                    (item.workflow.value == "h2r" and child.kind is JobSpecKind.RETARGET)
                    or (item.workflow.value == "r2r" and child.kind is JobSpecKind.R2R_RETARGET)
                )
                and len(child.inputs) == 1
                and child.inputs[0].asset_id == item.input_asset_id
                and child.inputs[0].sha256 == item.input_digest
                and child.robot.robot_id == item.target_robot_id
                and child.robot.asset_id == item.target_robot_asset_id
                and child.robot.config_sha256 == item.target_robot_digest
                and (source.robot_id if source is not None else None) == item.source_robot_id
                and (source.asset_id if source is not None else None) == item.source_robot_asset_id
                and (source.config_sha256 if source is not None else None)
                == item.source_robot_digest
                and child.backend == item.backend
                and (calibration.calibration_id if calibration is not None else None)
                == item.calibration_id
                and (calibration.sha256 if calibration is not None else None)
                == item.calibration_digest
                and child.effective_parameters.get("output_format") == item.output_format
                and _spec_sha256(child) == item.job_spec_sha256
            )
            if not projection_matches:
                raise _preflight_error(
                    plan.plan_id,
                    "A child JobSpec no longer matches the immutable batch projection.",
                )
            child_items.append(
                JobSpecBatchItem(
                    item_id=item.item_id,
                    plan_id=child.plan_id,
                    kind=child.kind,
                    input=child.inputs[0],
                    robot=child.robot,
                    source_robot=child.source_robot,
                    calibration=child.calibration,
                    backend=child.backend,
                    effective_parameters=child.effective_parameters,
                    output_policy=child.output_policy,
                    provenance=child.provenance,
                    created_at=child.created_at,
                )
            )

        first = child_items[0]
        source_robot = first.source_robot
        if (
            first.robot.robot_id != plan.target_robot_id
            or first.robot.asset_id != plan.target_robot_asset_id
            or first.robot.config_sha256 != plan.target_robot_digest
            or (source_robot.robot_id if source_robot is not None else None) != plan.source_robot_id
            or (source_robot.asset_id if source_robot is not None else None)
            != plan.source_robot_asset_id
            or (source_robot.config_sha256 if source_robot is not None else None)
            != plan.source_robot_digest
        ):
            raise _preflight_error(
                plan.plan_id,
                "The batch robot identities no longer match its child plans.",
            )
        provenance = JobSpecProvenance.model_validate_json(self._provenance_json)
        spec = JobSpecV2(
            kind=JobSpecKind.BATCH_RETARGET,
            plan_id=plan.plan_id,
            inputs=[item.input for item in child_items],
            robot=JobSpecRobot.model_validate(first.robot),
            source_robot=(
                JobSpecRobot.model_validate(source_robot) if source_robot is not None else None
            ),
            calibration=None,
            batch_items=child_items,
            backend="batch",
            effective_parameters={
                "workflow": plan.workflow.value,
                "run_mode": plan.run_mode,
                "resource_limits": plan.resource_limits.model_dump(mode="json"),
            },
            output_policy=plan.output_policy,
            provenance=provenance,
            created_at=plan.created_at,
        )
        return JobSpecV2.model_validate_json(spec.model_dump_json())


class ExecutionPlanService:
    """Dispatch single and batch plan semantics to their exact spec projector."""

    def __init__(self, plan_store: PlanStore, singles, batches: BatchRetargetService) -> None:
        self._plan_store = plan_store
        self._singles = singles
        self._batches = batches

    def get_job_spec(self, plan_id: str) -> JobSpecV2:
        try:
            semantics = self._plan_store.get_payload(plan_id).get("semantics")
        except PlanStoreError as error:
            raise RetargetServiceError(error.api_error) from error
        if semantics == BATCH_PLAN_SEMANTICS:
            return self._batches.get_job_spec(plan_id)
        if semantics in {"hhtools.retarget.plan.v1", R2R_PLAN_SEMANTICS}:
            return self._singles.get_job_spec(plan_id)
        raise _service_error(
            "UNSUPPORTED_PLAN_SEMANTICS",
            "The requested plan does not belong to a supported workflow.",
            details={"plan_id": plan_id},
        )


__all__ = ["BatchRetargetService", "ExecutionPlanService"]
