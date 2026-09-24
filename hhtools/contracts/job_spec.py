"""Auditable JobSpec v2 contract.

JobSpec v1 remains implemented in :mod:`hhtools.contracts.legacy_jobs`; this module
does not reinterpret or rewrite it.  A v1 replay must register its assets and
run preflight before a truthful v2 spec can be created.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from .common import AssetId, CalibrationId, ContractModel, PlanId, Sha256Hex
from .preflight import OutputPolicy


class JobSpecKind(StrEnum):
    RETARGET = "retarget"
    R2R_RETARGET = "r2r_retarget"
    BATCH_RETARGET = "batch_retarget"


class JobSpecInput(ContractModel):
    """Content-bound input reference used by an executable job."""

    asset_id: AssetId
    sha256: Sha256Hex


class JobSpecRobot(ContractModel):
    """Robot identity and exact configuration used by the job."""

    robot_id: Annotated[str, Field(min_length=1, max_length=256)]
    asset_id: AssetId
    config_sha256: Sha256Hex


class JobSpecCalibration(ContractModel):
    """Exact calibration selected by preflight."""

    calibration_id: CalibrationId
    sha256: Sha256Hex


class JobSpecProvenance(ContractModel):
    """Code, dependency, and execution-device identity for reproduction."""

    hhtools_git_commit: Annotated[str, Field(min_length=1, max_length=128)]
    hhtools_dirty: bool
    python: Annotated[str, Field(min_length=1, max_length=128)]
    pytorch: str | None = None
    cuda: str | None = None
    newton: str | None = None
    device: str | None = None
    platform: str | None = None
    dependencies: dict[str, str] = Field(default_factory=dict)


class JobSpecBatchItem(ContractModel):
    """Exact single-workflow JobSpec projection embedded in a batch JobSpec."""

    item_id: Annotated[str, Field(min_length=1, max_length=128)]
    plan_id: PlanId
    kind: Literal[JobSpecKind.RETARGET, JobSpecKind.R2R_RETARGET]
    input: JobSpecInput
    robot: JobSpecRobot
    source_robot: JobSpecRobot | None = None
    calibration: JobSpecCalibration | None = None
    backend: Annotated[str, Field(min_length=1, max_length=128)]
    effective_parameters: dict[str, Any] = Field(default_factory=dict)
    output_policy: OutputPolicy
    provenance: JobSpecProvenance
    created_at: AwareDatetime

    @model_validator(mode="after")
    def validate_workflow_identity(self) -> JobSpecBatchItem:
        if self.kind is JobSpecKind.R2R_RETARGET:
            if self.source_robot is None or self.calibration is None:
                raise ValueError("R2R batch items require source robot and calibration identities")
            if self.source_robot.asset_id == self.robot.asset_id:
                raise ValueError("R2R batch item robot assets must differ")
        elif self.source_robot is not None:
            raise ValueError("H2R batch items cannot declare a source robot")
        return self


class JobSpecV2(ContractModel):
    """Immutable, preflight-resolved execution identity for a retarget job."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=True,
    )

    schema_version: Literal[2] = 2
    kind: JobSpecKind
    plan_id: PlanId
    inputs: Annotated[list[JobSpecInput], Field(min_length=1)]
    robot: JobSpecRobot
    source_robot: JobSpecRobot | None = None
    calibration: JobSpecCalibration | None = None
    batch_items: list[JobSpecBatchItem] | None = None
    backend: Annotated[str, Field(min_length=1, max_length=128)]
    effective_parameters: dict[str, Any] = Field(default_factory=dict)
    output_policy: OutputPolicy
    provenance: JobSpecProvenance
    created_at: AwareDatetime

    @model_validator(mode="after")
    def validate_unique_inputs(self) -> JobSpecV2:
        asset_ids = [item.asset_id for item in self.inputs]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("JobSpec v2 inputs must not contain duplicate asset ids")
        if self.kind is JobSpecKind.BATCH_RETARGET:
            if not self.batch_items:
                raise ValueError("batch JobSpec v2 requires ordered child specs")
            if self.backend != "batch":
                raise ValueError("batch JobSpec v2 backend must be batch")
            if self.calibration is not None:
                raise ValueError("batch calibration identities belong to child specs")
            if len(self.inputs) != len(self.batch_items) or any(
                source != item.input
                for source, item in zip(self.inputs, self.batch_items, strict=True)
            ):
                raise ValueError("batch inputs must match ordered child specs")
            if any(item.robot != self.robot for item in self.batch_items):
                raise ValueError("batch items must use the parent target robot")
            if any(item.source_robot != self.source_robot for item in self.batch_items):
                raise ValueError("batch source robot must match every child spec")
        elif self.batch_items is not None:
            raise ValueError("only batch JobSpec v2 may declare batch items")
        elif self.kind is JobSpecKind.R2R_RETARGET:
            if self.source_robot is None:
                raise ValueError("R2R JobSpec v2 requires a source robot identity")
            if self.calibration is None:
                raise ValueError("R2R JobSpec v2 requires pair calibration identity")
            if self.source_robot.asset_id == self.robot.asset_id:
                raise ValueError("R2R source and target robot assets must differ")
        elif self.source_robot is not None:
            raise ValueError("only R2R JobSpec v2 may declare a source robot")
        return self
