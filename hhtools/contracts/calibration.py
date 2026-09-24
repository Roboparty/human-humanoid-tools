"""Agent-facing contracts for deterministic and vision-assisted calibration."""

from __future__ import annotations

import math
import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from .common import (
    AssetId,
    CalibrationId,
    ContractModel,
    SchemaVersion,
    Sha256Hex,
)
from .portability import validate_portable_json
from .preflight import PreflightCheck

CalibrationCandidateId = Annotated[
    str,
    Field(
        pattern=r"^cal-candidate:sha256:[0-9a-f]{64}$",
        description="Content-addressed immutable calibration candidate id.",
    ),
]

_JOINT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_MAX_CALIBRATION_JOINTS = 512


class CalibrationReference(StrEnum):
    SMPLX = "smplx"
    SMPL = "smpl"
    GVHMR = "gvhmr"
    SOMA_BVH = "soma_bvh"
    LAFAN_BVH = "lafan_bvh"
    MOCAP_BVH = "mocap_bvh"
    XSENS_MOCAP = "xsens_mocap"
    GLB = "glb"


class CalibrationState(StrEnum):
    MISSING = "missing"
    VALID = "valid"
    INVALID = "invalid"
    BUNDLED = "bundled"


class CalibrationVisualVerdict(StrEnum):
    PASS = "pass"
    ADJUST = "adjust"
    REJECT = "reject"


class CalibrationJointLimit(ContractModel):
    name: Annotated[str, Field(min_length=1, max_length=256)]
    lower: Annotated[float, Field(allow_inf_nan=False)]
    upper: Annotated[float, Field(allow_inf_nan=False)]

    @model_validator(mode="after")
    def validate_interval(self) -> CalibrationJointLimit:
        if self.upper <= self.lower:
            raise ValueError("calibration joint upper limit must exceed its lower limit")
        return self


def _joint_values(value: object) -> dict[str, float]:
    if not isinstance(value, dict) or len(value) > _MAX_CALIBRATION_JOINTS:
        raise ValueError("joint values must be a bounded object")
    normalized: dict[str, float] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name)
        if _JOINT_NAME.fullmatch(name) is None:
            raise ValueError("joint values contain an invalid joint name")
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise ValueError("joint values must contain only finite numbers")
        number = float(raw_value)
        if not math.isfinite(number):
            raise ValueError("joint values must contain only finite numbers")
        normalized[name] = number
    return normalized


def _finite_metric_values(value: object) -> dict[str, float]:
    if not isinstance(value, dict) or len(value) > 32:
        raise ValueError("metric values must be a bounded object")
    normalized: dict[str, float] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name)
        if not name or len(name) > 128:
            raise ValueError("metric values contain an invalid name")
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            raise ValueError("metric values must contain only finite numbers")
        number = float(raw_value)
        if not math.isfinite(number):
            raise ValueError("metric values must contain only finite numbers")
        normalized[name] = number
    return normalized


class CalibrationStatusRequest(ContractModel):
    schema_version: SchemaVersion = SchemaVersion.V1
    robot_id: Annotated[
        str,
        Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    ]
    robot_asset_id: AssetId
    reference: CalibrationReference
    motion_asset_id: AssetId | None = None


class CalibrationProposalRequest(CalibrationStatusRequest):
    """Create or revise one immutable candidate without saving calibration."""

    base_candidate_id: CalibrationCandidateId | None = None
    joint_q_overrides: dict[str, float] = Field(default_factory=dict)
    locked_joints: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(
        default_factory=list,
        max_length=_MAX_CALIBRATION_JOINTS,
    )

    @field_validator("joint_q_overrides", mode="before")
    @classmethod
    def validate_joint_q_overrides(cls, value: object) -> dict[str, float]:
        return _joint_values(value)

    @field_validator("locked_joints")
    @classmethod
    def validate_locked_joints(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(
            _JOINT_NAME.fullmatch(name) is None for name in value
        ):
            raise ValueError("locked joints must be unique valid joint names")
        return value


class CalibrationCandidate(ContractModel):
    """Content-bound robot pose proposed for one exact robot/reference identity."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=True,
    )

    schema_version: SchemaVersion = SchemaVersion.V1
    candidate_id: CalibrationCandidateId
    robot_id: Annotated[
        str,
        Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    ]
    robot_asset_id: AssetId
    robot_digest: Sha256Hex
    reference: CalibrationReference
    motion_asset_id: AssetId | None = None
    motion_digest: Sha256Hex | None = None
    algorithm: Literal["hhtools.calibration.kinematic.v1"] = "hhtools.calibration.kinematic.v1"
    baseline: Literal["urdf_zero", "saved_calibration", "candidate"]
    baseline_calibration_id: CalibrationId | None = None
    parent_candidate_id: CalibrationCandidateId | None = None
    joint_q: dict[str, float]
    locked_joints: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(
        default_factory=list,
        max_length=_MAX_CALIBRATION_JOINTS,
    )

    @field_validator("joint_q", mode="before")
    @classmethod
    def validate_joint_q(cls, value: object) -> dict[str, float]:
        normalized = _joint_values(value)
        if not normalized:
            raise ValueError("a calibration candidate must contain robot joints")
        return normalized

    @model_validator(mode="after")
    def validate_motion_identity(self) -> CalibrationCandidate:
        if (self.motion_asset_id is None) != (self.motion_digest is None):
            raise ValueError("candidate motion identity must be complete")
        if len(self.locked_joints) != len(set(self.locked_joints)):
            raise ValueError("candidate locked joints must be unique")
        if any(name not in self.joint_q for name in self.locked_joints):
            raise ValueError("candidate locked joints must exist in joint_q")
        if self.baseline == "saved_calibration" and self.baseline_calibration_id is None:
            raise ValueError("saved-calibration candidates must bind their baseline identity")
        if self.baseline == "urdf_zero" and self.baseline_calibration_id is not None:
            raise ValueError("URDF-zero candidates cannot bind a saved calibration")
        if (self.baseline == "candidate") != (self.parent_candidate_id is not None):
            raise ValueError("candidate revisions require exactly one parent candidate")
        return self


class CalibrationValidationReport(ContractModel):
    schema_version: SchemaVersion = SchemaVersion.V1
    candidate_id: CalibrationCandidateId | None = None
    valid: bool
    score: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
    changed_joint_count: Annotated[int, Field(ge=0)]
    mapped_slots: Annotated[int, Field(ge=0, le=17)]
    edge_errors_deg: dict[str, float] = Field(default_factory=dict, max_length=32)
    near_limit_joints: list[str] = Field(default_factory=list, max_length=_MAX_CALIBRATION_JOINTS)
    checks: list[PreflightCheck] = Field(min_length=1, max_length=16)

    @field_validator("edge_errors_deg", mode="before")
    @classmethod
    def validate_edge_errors(cls, value: object) -> dict[str, float]:
        return _finite_metric_values(value)

    @model_validator(mode="after")
    def validate_verdict(self) -> CalibrationValidationReport:
        has_error = any(check.level.value == "error" for check in self.checks)
        if self.valid == has_error:
            raise ValueError("calibration validity must agree with its checks")
        return self


class CalibrationStatusResponse(ContractModel):
    schema_version: SchemaVersion = SchemaVersion.V1
    request_id: Annotated[str, Field(min_length=1, max_length=256)]
    state: CalibrationState
    robot_id: Annotated[str, Field(min_length=1, max_length=256)]
    robot_asset_id: AssetId
    robot_digest: Sha256Hex
    reference: CalibrationReference
    source: Literal["none", "user_calibration", "robot_bundle", "bundled_scaler"]
    calibration_id: CalibrationId | None = None
    calibration_digest: Sha256Hex | None = None
    joint_q: dict[str, float] = Field(default_factory=dict)
    joint_count: Annotated[int, Field(ge=0, le=_MAX_CALIBRATION_JOINTS)]
    joint_limits: list[CalibrationJointLimit] = Field(
        default_factory=list,
        max_length=_MAX_CALIBRATION_JOINTS,
    )
    mapped_slots: Annotated[int, Field(ge=0, le=17)]
    mapping_targets: dict[str, str] = Field(default_factory=dict, max_length=17)
    missing_slots: list[str] = Field(default_factory=list, max_length=17)
    can_propose: bool
    can_silent_save: bool
    current_validation: CalibrationValidationReport | None = None

    @field_validator("joint_q", mode="before")
    @classmethod
    def validate_joint_q(cls, value: object) -> dict[str, float]:
        return _joint_values(value)

    @model_validator(mode="after")
    def validate_state(self) -> CalibrationStatusResponse:
        if self.state is CalibrationState.MISSING:
            if self.source != "none" or self.calibration_id is not None:
                raise ValueError("missing calibration status cannot expose a calibration")
        elif self.state is CalibrationState.BUNDLED:
            if self.source != "bundled_scaler" or self.calibration_digest is None:
                raise ValueError("bundled calibration status requires a scaler digest")
        elif (
            self.source not in {"user_calibration", "robot_bundle"}
            or self.calibration_id is None
            or self.calibration_digest is None
            or self.current_validation is None
        ):
            raise ValueError("manual calibration status requires identity and validation")
        if any(
            _JOINT_NAME.fullmatch(str(canonical)) is None
            or _JOINT_NAME.fullmatch(str(target)) is None
            for canonical, target in self.mapping_targets.items()
        ):
            raise ValueError("calibration mapping targets must use portable names")
        if self.joint_count != len(self.joint_limits):
            raise ValueError("calibration joint count must match its limits")
        return self


class CalibrationProposalResponse(ContractModel):
    schema_version: SchemaVersion = SchemaVersion.V1
    candidate: CalibrationCandidate
    validation: CalibrationValidationReport

    @model_validator(mode="after")
    def validate_identity(self) -> CalibrationProposalResponse:
        if (
            self.validation.candidate_id is None
            or self.candidate.candidate_id != self.validation.candidate_id
        ):
            raise ValueError("proposal candidate and validation ids must match")
        return self


class CalibrationValidationRequest(ContractModel):
    schema_version: SchemaVersion = SchemaVersion.V1
    candidate_id: CalibrationCandidateId


class CalibrationPreviewRequest(CalibrationValidationRequest):
    pass


class CalibrationPreview(ContractModel):
    schema_version: SchemaVersion = SchemaVersion.V1
    candidate_id: CalibrationCandidateId
    media_type: Literal["image/png"] = "image/png"
    sha256: Sha256Hex
    width: Annotated[int, Field(ge=1, le=4096)]
    height: Annotated[int, Field(ge=1, le=4096)]
    views: list[Literal["front", "side"]] = Field(default_factory=lambda: ["front", "side"])
    validation: CalibrationValidationReport

    @model_validator(mode="after")
    def validate_identity(self) -> CalibrationPreview:
        if self.validation.candidate_id != self.candidate_id:
            raise ValueError("preview validation must describe its candidate")
        return self


class CalibrationVisualReview(ContractModel):
    reviewer: Literal["gpt_vision", "human"]
    verdict: CalibrationVisualVerdict
    model_hint: Annotated[str | None, Field(default=None, min_length=1, max_length=128)]
    summary: Annotated[str, Field(min_length=1, max_length=4_096)]
    observations: list[Annotated[str, Field(min_length=1, max_length=512)]] = Field(
        default_factory=list,
        max_length=16,
    )

    @model_validator(mode="after")
    def validate_portability(self) -> CalibrationVisualReview:
        validate_portable_json(self.model_dump(mode="json", exclude_none=True))
        return self


class CalibrationSaveRequest(CalibrationValidationRequest):
    save_mode: Literal["validated_silent", "gpt_vision_silent"]
    visual_review: CalibrationVisualReview | None = None

    @model_validator(mode="after")
    def validate_visual_authorization(self) -> CalibrationSaveRequest:
        if self.save_mode == "gpt_vision_silent":
            if (
                self.visual_review is None
                or self.visual_review.reviewer != "gpt_vision"
                or self.visual_review.verdict is not CalibrationVisualVerdict.PASS
            ):
                raise ValueError("gpt_vision_silent requires a passing GPT visual review")
        elif (
            self.visual_review is not None
            and self.visual_review.verdict is not CalibrationVisualVerdict.PASS
        ):
            raise ValueError("a rejected visual review cannot be saved")
        return self


class CalibrationSaveReceipt(ContractModel):
    schema_version: SchemaVersion = SchemaVersion.V1
    candidate_id: CalibrationCandidateId
    calibration_id: CalibrationId
    calibration_digest: Sha256Hex
    previous_calibration_id: CalibrationId | None = None
    previous_calibration_archived: bool = False
    robot_id: Annotated[str, Field(min_length=1, max_length=256)]
    reference: CalibrationReference
    save_mode: Literal["validated_silent", "gpt_vision_silent"]
    validation: CalibrationValidationReport
    visual_review: CalibrationVisualReview | None = None
    saved: Literal[True] = True

    @model_validator(mode="after")
    def validate_saved_candidate(self) -> CalibrationSaveReceipt:
        if not self.validation.valid or self.validation.candidate_id != self.candidate_id:
            raise ValueError("saved calibration requires matching valid evidence")
        if self.save_mode == "gpt_vision_silent" and (
            self.visual_review is None
            or self.visual_review.reviewer != "gpt_vision"
            or self.visual_review.verdict is not CalibrationVisualVerdict.PASS
        ):
            raise ValueError("GPT vision save receipt requires a passing visual review")
        return self


class R2RCalibrationStatusRequest(ContractModel):
    """Identify one exact source/target robot pair calibration."""

    schema_version: SchemaVersion = SchemaVersion.V1
    source_robot_id: Annotated[
        str,
        Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    ]
    source_robot_asset_id: AssetId
    target_robot_id: Annotated[
        str,
        Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    ]
    target_robot_asset_id: AssetId

    @model_validator(mode="after")
    def validate_pair(self) -> R2RCalibrationStatusRequest:
        if self.source_robot_id == self.target_robot_id:
            raise ValueError("R2R calibration requires different source and target robots")
        if self.source_robot_asset_id == self.target_robot_asset_id:
            raise ValueError("R2R calibration requires different source and target assets")
        return self


class R2RCalibrationProposalRequest(R2RCalibrationStatusRequest):
    """Create or revise one immutable target-pose candidate for a robot pair."""

    base_candidate_id: CalibrationCandidateId | None = None
    joint_q_overrides: dict[str, float] = Field(default_factory=dict)
    locked_joints: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(
        default_factory=list,
        max_length=_MAX_CALIBRATION_JOINTS,
    )

    @field_validator("joint_q_overrides", mode="before")
    @classmethod
    def validate_joint_q_overrides(cls, value: object) -> dict[str, float]:
        return _joint_values(value)

    @field_validator("locked_joints")
    @classmethod
    def validate_locked_joints(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(
            _JOINT_NAME.fullmatch(name) is None for name in value
        ):
            raise ValueError("locked joints must be unique valid joint names")
        return value


class R2RCalibrationCandidate(ContractModel):
    """Content-bound target pose for one exact source/target robot pair."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=True,
    )

    schema_version: SchemaVersion = SchemaVersion.V1
    workflow: Literal["r2r"] = "r2r"
    candidate_id: CalibrationCandidateId
    source_robot_id: Annotated[
        str,
        Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    ]
    source_robot_asset_id: AssetId
    source_robot_digest: Sha256Hex
    target_robot_id: Annotated[
        str,
        Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
    ]
    target_robot_asset_id: AssetId
    target_robot_digest: Sha256Hex
    algorithm: Literal["hhtools.r2r-calibration.kinematic.v1"] = (
        "hhtools.r2r-calibration.kinematic.v1"
    )
    baseline: Literal["urdf_zero", "saved_calibration", "candidate"]
    baseline_calibration_id: CalibrationId | None = None
    parent_candidate_id: CalibrationCandidateId | None = None
    joint_q: dict[str, float]
    locked_joints: list[Annotated[str, Field(min_length=1, max_length=256)]] = Field(
        default_factory=list,
        max_length=_MAX_CALIBRATION_JOINTS,
    )

    @field_validator("joint_q", mode="before")
    @classmethod
    def validate_joint_q(cls, value: object) -> dict[str, float]:
        normalized = _joint_values(value)
        if not normalized:
            raise ValueError("an R2R calibration candidate must contain target joints")
        return normalized

    @model_validator(mode="after")
    def validate_identity(self) -> R2RCalibrationCandidate:
        if self.source_robot_id == self.target_robot_id:
            raise ValueError("R2R candidates require different source and target robots")
        if self.source_robot_asset_id == self.target_robot_asset_id:
            raise ValueError("R2R candidates require different source and target assets")
        if len(self.locked_joints) != len(set(self.locked_joints)):
            raise ValueError("candidate locked joints must be unique")
        if any(name not in self.joint_q for name in self.locked_joints):
            raise ValueError("candidate locked joints must exist in joint_q")
        if self.baseline == "saved_calibration" and self.baseline_calibration_id is None:
            raise ValueError("saved-calibration candidates must bind their baseline identity")
        if self.baseline == "urdf_zero" and self.baseline_calibration_id is not None:
            raise ValueError("URDF-zero candidates cannot bind a saved calibration")
        if (self.baseline == "candidate") != (self.parent_candidate_id is not None):
            raise ValueError("candidate revisions require exactly one parent candidate")
        return self


class R2RCalibrationStatusResponse(ContractModel):
    schema_version: SchemaVersion = SchemaVersion.V1
    request_id: Annotated[str, Field(min_length=1, max_length=256)]
    state: CalibrationState
    source_robot_id: Annotated[str, Field(min_length=1, max_length=256)]
    source_robot_asset_id: AssetId
    source_robot_digest: Sha256Hex
    target_robot_id: Annotated[str, Field(min_length=1, max_length=256)]
    target_robot_asset_id: AssetId
    target_robot_digest: Sha256Hex
    storage: Literal["none", "user_calibration", "robot_bundle"]
    calibration_id: CalibrationId | None = None
    calibration_digest: Sha256Hex | None = None
    joint_q: dict[str, float] = Field(default_factory=dict)
    joint_count: Annotated[int, Field(ge=0, le=_MAX_CALIBRATION_JOINTS)]
    joint_limits: list[CalibrationJointLimit] = Field(
        default_factory=list,
        max_length=_MAX_CALIBRATION_JOINTS,
    )
    source_mapped_slots: Annotated[int, Field(ge=0, le=17)]
    target_mapped_slots: Annotated[int, Field(ge=0, le=17)]
    source_missing_slots: list[str] = Field(default_factory=list, max_length=17)
    target_missing_slots: list[str] = Field(default_factory=list, max_length=17)
    can_propose: bool
    can_silent_save: bool
    current_validation: CalibrationValidationReport | None = None

    @field_validator("joint_q", mode="before")
    @classmethod
    def validate_joint_q(cls, value: object) -> dict[str, float]:
        return _joint_values(value)

    @model_validator(mode="after")
    def validate_state(self) -> R2RCalibrationStatusResponse:
        if self.state is CalibrationState.MISSING:
            if (
                self.storage != "none"
                or self.calibration_id is not None
                or self.calibration_digest is not None
                or self.current_validation is not None
            ):
                raise ValueError("missing R2R calibration status cannot expose a calibration")
        elif self.state not in {CalibrationState.VALID, CalibrationState.INVALID}:
            raise ValueError("R2R calibration status does not support bundled-scaler state")
        elif (
            self.storage not in {"user_calibration", "robot_bundle"}
            or self.calibration_id is None
            or self.calibration_digest is None
            or self.current_validation is None
        ):
            raise ValueError("saved R2R calibration status requires identity and validation")
        if self.joint_count != len(self.joint_limits):
            raise ValueError("R2R target joint count must match its limits")
        return self


class R2RCalibrationProposalResponse(ContractModel):
    schema_version: SchemaVersion = SchemaVersion.V1
    candidate: R2RCalibrationCandidate
    validation: CalibrationValidationReport

    @model_validator(mode="after")
    def validate_identity(self) -> R2RCalibrationProposalResponse:
        if self.validation.candidate_id != self.candidate.candidate_id:
            raise ValueError("R2R proposal candidate and validation ids must match")
        return self


class R2RCalibrationValidationRequest(CalibrationValidationRequest):
    pass


class R2RCalibrationPreviewRequest(R2RCalibrationValidationRequest):
    pass


class R2RCalibrationPreview(CalibrationPreview):
    pass


class R2RCalibrationSaveRequest(CalibrationSaveRequest):
    pass


class R2RCalibrationSaveReceipt(ContractModel):
    schema_version: SchemaVersion = SchemaVersion.V1
    candidate_id: CalibrationCandidateId
    calibration_id: CalibrationId
    calibration_digest: Sha256Hex
    previous_calibration_id: CalibrationId | None = None
    previous_calibration_archived: bool = False
    source_robot_id: Annotated[str, Field(min_length=1, max_length=256)]
    target_robot_id: Annotated[str, Field(min_length=1, max_length=256)]
    save_mode: Literal["validated_silent", "gpt_vision_silent"]
    validation: CalibrationValidationReport
    visual_review: CalibrationVisualReview | None = None
    saved: Literal[True] = True

    @model_validator(mode="after")
    def validate_saved_candidate(self) -> R2RCalibrationSaveReceipt:
        if not self.validation.valid or self.validation.candidate_id != self.candidate_id:
            raise ValueError("saved R2R calibration requires matching valid evidence")
        if self.save_mode == "gpt_vision_silent" and (
            self.visual_review is None
            or self.visual_review.reviewer != "gpt_vision"
            or self.visual_review.verdict is not CalibrationVisualVerdict.PASS
        ):
            raise ValueError("R2R GPT vision save receipt requires a passing visual review")
        return self


__all__ = [
    "CalibrationCandidate",
    "CalibrationCandidateId",
    "CalibrationJointLimit",
    "CalibrationPreview",
    "CalibrationPreviewRequest",
    "CalibrationProposalRequest",
    "CalibrationProposalResponse",
    "CalibrationReference",
    "CalibrationSaveReceipt",
    "CalibrationSaveRequest",
    "CalibrationState",
    "CalibrationStatusRequest",
    "CalibrationStatusResponse",
    "CalibrationValidationReport",
    "CalibrationValidationRequest",
    "CalibrationVisualReview",
    "CalibrationVisualVerdict",
    "R2RCalibrationCandidate",
    "R2RCalibrationPreview",
    "R2RCalibrationPreviewRequest",
    "R2RCalibrationProposalRequest",
    "R2RCalibrationProposalResponse",
    "R2RCalibrationSaveReceipt",
    "R2RCalibrationSaveRequest",
    "R2RCalibrationStatusRequest",
    "R2RCalibrationStatusResponse",
    "R2RCalibrationValidationRequest",
]
