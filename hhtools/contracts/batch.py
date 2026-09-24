"""Scalable H2R/R2R batch preflight plans and result reports."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from .common import (
    ApiError,
    ArtifactId,
    AssetId,
    CalibrationId,
    ContractModel,
    NextAction,
    PlanId,
    SchemaVersion,
    Sha256Hex,
)
from .jobs import JobOutcome
from .preflight import OutputPolicy, PreflightCheck, PreflightStatus

DEFAULT_MAX_BATCH_ITEMS = 0
DEFAULT_MAX_BATCH_TOTAL_FRAMES = 0


class BatchWorkflow(StrEnum):
    H2R = "h2r"
    R2R = "r2r"


class BatchPreflightRequest(ContractModel):
    """Ordered ready single-item plans to freeze into one batch."""

    schema_version: SchemaVersion = SchemaVersion.V1
    workflow: BatchWorkflow
    item_plan_ids: Annotated[list[PlanId], Field(min_length=1)]
    output_policy: OutputPolicy = OutputPolicy.CREATE_NEW

    @model_validator(mode="after")
    def validate_unique_plans(self) -> BatchPreflightRequest:
        if len(self.item_plan_ids) != len(set(self.item_plan_ids)):
            raise ValueError("batch item plan ids must be unique")
        return self


class BatchResourceLimits(ContractModel):
    """Optional service caps frozen into a plan; zero means unlimited."""

    max_items: Annotated[int, Field(ge=0)] = DEFAULT_MAX_BATCH_ITEMS
    max_total_frames: Annotated[int, Field(ge=0)] = DEFAULT_MAX_BATCH_TOTAL_FRAMES
    item_count: Annotated[int, Field(ge=1)]
    estimated_total_frames: Annotated[int, Field(ge=1)]
    execution_concurrency: Literal[1] = 1
    failure_limit: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def validate_estimate(self) -> BatchResourceLimits:
        if self.max_items and self.item_count > self.max_items:
            raise ValueError("batch item count exceeds its frozen limit")
        if self.max_total_frames and self.estimated_total_frames > self.max_total_frames:
            raise ValueError("batch frame estimate exceeds its frozen limit")
        return self


class BatchPlanItem(ContractModel):
    """Public projection of one immutable child plan in execution order."""

    index: Annotated[int, Field(ge=0)]
    item_id: Annotated[
        str,
        Field(min_length=1, max_length=128, pattern=r"^item-[0-9]+-[0-9a-f]{12}$"),
    ]
    workflow: BatchWorkflow
    plan_id: PlanId
    input_asset_id: AssetId
    input_digest: Sha256Hex
    target_robot_id: Annotated[str, Field(min_length=1, max_length=256)]
    target_robot_asset_id: AssetId
    target_robot_digest: Sha256Hex
    source_robot_id: Annotated[str | None, Field(default=None, min_length=1, max_length=256)]
    source_robot_asset_id: AssetId | None = None
    source_robot_digest: Sha256Hex | None = None
    backend: Annotated[str, Field(min_length=1, max_length=128)]
    calibration_id: CalibrationId | None = None
    calibration_digest: Sha256Hex | None = None
    output_format: Annotated[str, Field(min_length=1, max_length=32)]
    estimated_frames: Annotated[int, Field(ge=1)]
    job_spec_sha256: Sha256Hex

    @model_validator(mode="after")
    def validate_optional_identities(self) -> BatchPlanItem:
        source_values = (
            self.source_robot_id,
            self.source_robot_asset_id,
            self.source_robot_digest,
        )
        if any(value is not None for value in source_values) and not all(
            value is not None for value in source_values
        ):
            raise ValueError("batch source robot identity must be complete")
        if (self.calibration_id is None) != (self.calibration_digest is None):
            raise ValueError("batch calibration identity must be complete")
        return self


class BatchPlan(ContractModel):
    """Content-addressed ordered collection of ready H2R or R2R child plans."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=True,
    )

    schema_version: SchemaVersion = SchemaVersion.V1
    plan_id: PlanId
    created_at: AwareDatetime
    expires_at: AwareDatetime | None = None
    workflow: BatchWorkflow
    run_mode: Literal["smoke", "full"]
    items: Annotated[list[BatchPlanItem], Field(min_length=1)]
    target_robot_id: Annotated[str, Field(min_length=1, max_length=256)]
    target_robot_asset_id: AssetId
    target_robot_digest: Sha256Hex
    source_robot_id: Annotated[str | None, Field(default=None, min_length=1, max_length=256)]
    source_robot_asset_id: AssetId | None = None
    source_robot_digest: Sha256Hex | None = None
    output_policy: OutputPolicy
    resource_limits: BatchResourceLimits

    @model_validator(mode="after")
    def validate_batch_projection(self) -> BatchPlan:
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at")
        if [item.index for item in self.items] != list(range(len(self.items))):
            raise ValueError("batch item indexes must be contiguous and ordered")
        if len({item.item_id for item in self.items}) != len(self.items):
            raise ValueError("batch item ids must be unique")
        if len({item.plan_id for item in self.items}) != len(self.items):
            raise ValueError("batch child plan ids must be unique")
        if len({item.input_asset_id for item in self.items}) != len(self.items):
            raise ValueError("batch input assets must be unique")
        if self.resource_limits.item_count != len(self.items):
            raise ValueError("batch resource item count must match its ordered items")
        if self.resource_limits.estimated_total_frames != sum(
            item.estimated_frames for item in self.items
        ):
            raise ValueError("batch resource frame estimate must match its ordered items")
        for item in self.items:
            if item.workflow is not self.workflow:
                raise ValueError("every batch item must use the parent workflow")
            if (
                item.target_robot_id != self.target_robot_id
                or item.target_robot_asset_id != self.target_robot_asset_id
                or item.target_robot_digest != self.target_robot_digest
            ):
                raise ValueError("every batch item must use the frozen target robot")
        if self.workflow is BatchWorkflow.H2R:
            if any(
                value is not None
                for value in (
                    self.source_robot_id,
                    self.source_robot_asset_id,
                    self.source_robot_digest,
                )
            ) or any(item.source_robot_id is not None for item in self.items):
                raise ValueError("H2R batch plans cannot declare a source robot")
        else:
            if any(
                value is None
                for value in (
                    self.source_robot_id,
                    self.source_robot_asset_id,
                    self.source_robot_digest,
                )
            ):
                raise ValueError("R2R batch plans require a complete source robot identity")
            for item in self.items:
                if (
                    item.source_robot_id != self.source_robot_id
                    or item.source_robot_asset_id != self.source_robot_asset_id
                    or item.source_robot_digest != self.source_robot_digest
                ):
                    raise ValueError("every R2R batch item must use the frozen source robot")
        return self


class BatchPreflightResponse(ContractModel):
    schema_version: SchemaVersion = SchemaVersion.V1
    request_id: Annotated[str, Field(min_length=1, max_length=256)]
    status: PreflightStatus
    plan: BatchPlan | None = None
    checks: list[PreflightCheck] = Field(default_factory=list)
    required_actions: list[NextAction] = Field(default_factory=list)
    error: ApiError | None = None

    @model_validator(mode="after")
    def validate_response_state(self) -> BatchPreflightResponse:
        if self.status is PreflightStatus.READY:
            if self.plan is None or self.error is not None or self.required_actions:
                raise ValueError("ready batch preflight requires only a plan")
        else:
            if self.plan is not None:
                raise ValueError("non-ready batch preflight cannot include a plan")
            if self.status is PreflightStatus.HUMAN_ACTION_REQUIRED and not self.required_actions:
                raise ValueError("human_action_required requires an action")
            if self.status is PreflightStatus.REJECTED and self.error is None:
                raise ValueError("rejected batch preflight requires an error")
        return self


class BatchReportItem(ContractModel):
    index: Annotated[int, Field(ge=0)]
    item_id: Annotated[str, Field(min_length=1, max_length=128)]
    plan_id: PlanId
    input_asset_id: AssetId
    state: Literal["completed", "failed"]
    outcome: JobOutcome | None = None
    error: ApiError | None = None
    summary: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    artifact_ids: list[ArtifactId] = Field(default_factory=list, max_length=8)
    execution_provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_result(self) -> BatchReportItem:
        if self.state == "completed":
            if self.outcome is None or self.error is not None:
                raise ValueError("completed batch items require an outcome and no error")
        elif self.error is None or self.outcome is not None:
            raise ValueError("failed batch items require an error and no outcome")
        return self


class BatchReport(ContractModel):
    schema_version: SchemaVersion = SchemaVersion.V1
    job_id: Annotated[str, Field(min_length=1, max_length=256)]
    workflow: BatchWorkflow
    outcome: JobOutcome
    total_items: Annotated[int, Field(ge=1)]
    completed_items: Annotated[int, Field(ge=0)]
    succeeded_items: Annotated[int, Field(ge=0)]
    failed_items: Annotated[int, Field(ge=0)]
    items: Annotated[list[BatchReportItem], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_counts(self) -> BatchReport:
        if self.total_items != len(self.items) or self.completed_items != len(self.items):
            raise ValueError("batch report counts must match its item list")
        succeeded = sum(item.state == "completed" for item in self.items)
        failed = sum(item.state == "failed" for item in self.items)
        if self.succeeded_items != succeeded or self.failed_items != failed:
            raise ValueError("batch report success/failure counts must match its items")
        return self


__all__ = [
    "BatchPlan",
    "BatchPlanItem",
    "BatchPreflightRequest",
    "BatchPreflightResponse",
    "BatchReport",
    "BatchReportItem",
    "BatchResourceLimits",
    "BatchWorkflow",
    "DEFAULT_MAX_BATCH_ITEMS",
    "DEFAULT_MAX_BATCH_TOTAL_FRAMES",
]
