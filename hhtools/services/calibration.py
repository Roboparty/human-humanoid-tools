"""Transport-neutral calibration assistance and validated silent saving."""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, NoReturn

from hhtools.contracts import (
    ApiError,
    AssetInspectionRequest,
    CalibrationCandidate,
    CalibrationPreview,
    CalibrationProposalRequest,
    CalibrationProposalResponse,
    CalibrationReference,
    CalibrationSaveReceipt,
    CalibrationSaveRequest,
    CalibrationState,
    CalibrationStatusRequest,
    CalibrationStatusResponse,
    CalibrationValidationReport,
    CalibrationValidationRequest,
    ErrorStage,
    InspectionStatus,
    PreflightCheck,
    PreflightCheckLevel,
)
from hhtools.retarget.calibration.assistant import (
    CALIBRATION_ALGORITHM,
    CALIBRATION_PREVIEW_HEIGHT,
    CALIBRATION_PREVIEW_WIDTH,
    CalibrationAssessment,
    assess_calibration_pose,
    normalized_joint_q,
    propose_calibration_pose,
    render_calibration_preview_png,
)
from hhtools.robot.base import RobotPreset

from .asset_service import AgentAssetService
from .assets import AssetServiceError
from .calibration_candidates import (
    CalibrationCandidateStore,
    CalibrationCandidateStoreError,
    compute_calibration_candidate_id,
)
from .calibration_validation import calibration_validation_identity
from .preflight import (
    _bundled_scaler,
    _manual_calibration,
    _parse_stable_file,
    _PreflightFailureError,
    _robot_bundle_and_preset,
)

RobotProvider = Callable[[], Iterable[RobotPreset]]
RobotMaterializer = Callable[[str, str], Any]
RobotRelease = Callable[[Any], None]
MotionLoader = Callable[[str], Any]


class CalibrationServiceError(RuntimeError):
    """Expected calibration failure with a portable public error."""

    def __init__(self, error: ApiError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")

    @property
    def api_error(self) -> ApiError:
        return self.error

    @property
    def code(self) -> str:
        return self.error.code


def _error(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
    retryable: bool = False,
) -> CalibrationServiceError:
    return CalibrationServiceError(
        ApiError(
            code=code,
            message=message,
            retryable=retryable,
            stage=ErrorStage.CALIBRATION,
            details=dict(details or {}),
        )
    )


def _raise_preflight(error: _PreflightFailureError) -> NoReturn:
    raise CalibrationServiceError(
        error.error.model_copy(update={"stage": ErrorStage.CALIBRATION})
    ) from error


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class CalibrationService:
    """Own candidate identity, deterministic checks, previews, and writes."""

    def __init__(
        self,
        asset_service: AgentAssetService,
        candidate_store: CalibrationCandidateStore,
        *,
        robot_provider: RobotProvider,
        materialize_robot: RobotMaterializer,
        release_robot: RobotRelease,
        motion_loader: MotionLoader | None = None,
        user_robot_root: Path | None = None,
        request_id_provider: Callable[[], str] = lambda: f"req_{uuid.uuid4().hex}",
    ) -> None:
        self._asset_service = asset_service
        self._candidate_store = candidate_store
        self._robot_provider = robot_provider
        self._materialize_robot = materialize_robot
        self._release_robot = release_robot
        self._motion_loader = motion_loader
        self._user_robot_root = user_robot_root
        self._request_id_provider = request_id_provider

    def _resolve_robot(
        self,
        request: CalibrationStatusRequest,
    ) -> tuple[Any, RobotPreset, dict[str, tuple[float | None, float | None]]]:
        try:
            return _robot_bundle_and_preset(
                self._asset_service,
                robot_id=request.robot_id,
                robot_asset_id=request.robot_asset_id,
                presets=self._robot_provider(),
            )
        except _PreflightFailureError as error:
            _raise_preflight(error)

    def _motion(
        self,
        *,
        reference: CalibrationReference,
        motion_asset_id: str | None,
    ) -> tuple[Any | None, str | None]:
        if motion_asset_id is None:
            if reference is CalibrationReference.GLB:
                raise _error(
                    "CALIBRATION_MOTION_REQUIRED",
                    "GLB calibration requires the exact registered motion asset.",
                    details={"reference": reference.value},
                )
            return None, None
        try:
            bundle = self._asset_service.get(motion_asset_id)
            inspection = self._asset_service.inspect(
                AssetInspectionRequest(
                    asset_id=motion_asset_id,
                    verify_hashes=True,
                    parse_content=True,
                )
            )
        except AssetServiceError as error:
            raise CalibrationServiceError(
                error.api_error.model_copy(update={"stage": ErrorStage.CALIBRATION})
            ) from error
        if inspection.status is InspectionStatus.INVALID:
            public = (
                inspection.errors[0]
                if inspection.errors
                else ApiError(
                    code="ASSET_INSPECTION_FAILED",
                    message="The calibration motion did not pass content inspection.",
                    stage=ErrorStage.CALIBRATION,
                )
            )
            raise CalibrationServiceError(
                public.model_copy(update={"stage": ErrorStage.CALIBRATION})
            )
        from hhtools.retarget.calibration import normalize_calibration_reference

        actual = normalize_calibration_reference(str(inspection.reference_model or ""))
        if actual != reference.value:
            raise _error(
                "REFERENCE_MISMATCH",
                "The calibration motion belongs to a different reference family.",
                details={"expected_reference": reference.value, "actual_reference": actual},
            )
        motion = None
        if reference is CalibrationReference.GLB:
            if self._motion_loader is None:
                raise _error(
                    "CALIBRATION_PROPOSAL_UNAVAILABLE",
                    "This runtime cannot load a clip-specific calibration reference.",
                )
            try:
                motion = self._motion_loader(bundle.asset_id)
            except Exception as error:  # noqa: BLE001 - sanitize loader internals
                raise _error(
                    "CALIBRATION_MOTION_INVALID",
                    "The registered calibration motion could not be loaded.",
                ) from error
        return motion, bundle.asset_id.rsplit(":", 1)[-1]

    @staticmethod
    def _checks(assessment: CalibrationAssessment) -> list[PreflightCheck]:
        mapping_error = bool(assessment.missing_slots)
        mapping_warning = bool(assessment.non_distal_targets)
        limit_error = bool(
            assessment.unknown_joints or assessment.missing_joints or assessment.limit_violations
        )
        alignment_error = bool(assessment.alignment_errors)
        alignment_warning = bool(assessment.alignment_warnings)
        symmetry_warning = any(value > 20.0 for value in assessment.symmetry_errors_deg.values())
        feet_error = (
            assessment.foot_height_delta_m is not None and assessment.foot_height_delta_m > 0.08
        )
        feet_warning = (
            assessment.foot_height_delta_m is not None and assessment.foot_height_delta_m > 0.03
        )
        return [
            PreflightCheck(
                code="CALIBRATION_MAPPING_COMPLETE",
                level=(
                    PreflightCheckLevel.ERROR
                    if mapping_error
                    else PreflightCheckLevel.WARNING
                    if mapping_warning
                    else PreflightCheckLevel.PASS
                ),
                message=(
                    "Critical calibration semantic mappings are missing."
                    if mapping_error
                    else "One or more endpoint mappings have actuated descendants."
                    if mapping_warning
                    else "Critical calibration semantic mappings resolve."
                ),
                details={
                    "mapped_slots": assessment.mapped_slots,
                    "missing_slots": list(assessment.missing_slots),
                    "non_distal_targets": list(assessment.non_distal_targets),
                },
            ),
            PreflightCheck(
                code="CALIBRATION_JOINT_LIMITS_VALID",
                level=(PreflightCheckLevel.ERROR if limit_error else PreflightCheckLevel.PASS),
                message=(
                    "The candidate joint vector is incomplete or outside URDF limits."
                    if limit_error
                    else "Every candidate joint is finite and inside its URDF limit."
                ),
                details={
                    "unknown_joints": list(assessment.unknown_joints),
                    "missing_joints": list(assessment.missing_joints),
                    "limit_violations": list(assessment.limit_violations),
                    "near_limit_joints": list(assessment.near_limit_joints),
                },
            ),
            PreflightCheck(
                code="CALIBRATION_POSE_ALIGNED",
                level=(
                    PreflightCheckLevel.ERROR
                    if alignment_error
                    else PreflightCheckLevel.WARNING
                    if alignment_warning
                    else PreflightCheckLevel.PASS
                ),
                message=(
                    "One or more required limb directions remain misaligned."
                    if alignment_error
                    else "Required limb directions are within the accepted tolerance."
                ),
                details={
                    "edge_errors_deg": assessment.edge_errors_deg,
                    "unavailable_edges": list(assessment.unavailable_edges),
                    "error_edges": list(assessment.alignment_errors),
                    "warning_edges": list(assessment.alignment_warnings),
                },
            ),
            PreflightCheck(
                code="CALIBRATION_BILATERAL_SYMMETRY",
                level=(
                    PreflightCheckLevel.WARNING if symmetry_warning else PreflightCheckLevel.PASS
                ),
                message=(
                    "The proposed left and right limb directions are visibly asymmetric."
                    if symmetry_warning
                    else "The proposed bilateral limb directions are symmetric."
                ),
                details={"errors_deg": assessment.symmetry_errors_deg},
            ),
            PreflightCheck(
                code="CALIBRATION_FEET_LEVEL",
                level=(
                    PreflightCheckLevel.ERROR
                    if feet_error
                    else PreflightCheckLevel.WARNING
                    if feet_warning
                    else PreflightCheckLevel.PASS
                ),
                message=(
                    "The proposed feet are not level."
                    if feet_error or feet_warning
                    else "The proposed feet are level."
                ),
                details={"height_delta_m": assessment.foot_height_delta_m},
            ),
        ]

    @classmethod
    def _report(
        cls,
        assessment: CalibrationAssessment,
        *,
        candidate_id: str | None,
    ) -> CalibrationValidationReport:
        return CalibrationValidationReport(
            candidate_id=candidate_id,
            valid=assessment.valid,
            score=round(assessment.score, 6),
            changed_joint_count=assessment.changed_joint_count,
            mapped_slots=min(17, assessment.mapped_slots),
            edge_errors_deg=assessment.edge_errors_deg,
            near_limit_joints=list(assessment.near_limit_joints),
            checks=cls._checks(assessment),
        )

    def _manual_profile(
        self,
        preset: RobotPreset,
        reference: str,
        limits: Mapping[str, tuple[float | None, float | None]],
        robot_bundle: Any,
    ) -> tuple[Path, str, str, str] | None:
        try:
            return _manual_calibration(
                preset,
                reference,
                limits,
                robot_bundle,
                user_root=self._user_robot_root,
            )
        except _PreflightFailureError as error:
            _raise_preflight(error)

    def _record_validation(
        self,
        *,
        path: Path,
        digest: str,
        robot_id: str,
        robot_asset_id: str,
        reference: str,
        motion_asset_id: str | None,
        validation: CalibrationValidationReport,
    ) -> None:
        try:
            if _sha256_file(path) != digest:
                raise ValueError("calibration changed during validation")
            self._candidate_store.validation_store.put(
                calibration_validation_identity(
                    robot_id=robot_id,
                    robot_asset_id=robot_asset_id,
                    reference=reference,
                    calibration_digest=digest,
                    motion_asset_id=motion_asset_id,
                ),
                validation,
            )
        except (OSError, ValueError) as error:
            raise _error(
                "CALIBRATION_VALIDATION_UNAVAILABLE",
                "The current calibration validation could not be recorded; retry status.",
                retryable=True,
            ) from error

    def status(self, request: CalibrationStatusRequest) -> CalibrationStatusResponse:
        bundle, preset, limits = self._resolve_robot(request)
        manual = self._manual_profile(preset, request.reference.value, limits, bundle)
        if manual is None:
            try:
                scaler = _bundled_scaler(preset, request.reference.value, bundle)
            except _PreflightFailureError as error:
                _raise_preflight(error)
        else:
            scaler = None

        current_validation = None
        joint_q: dict[str, float] = {}
        if manual is not None:
            from hhtools.retarget.calibration import load_calibration

            path, digest, calibration_id, storage = manual
            try:
                calibration, _ = _parse_stable_file(
                    path, load_calibration, expected_digests={digest}
                )
            except (OSError, TypeError, ValueError) as error:
                raise _error(
                    "CALIBRATION_VALIDATION_UNAVAILABLE",
                    "The calibration changed before validation; retry status.",
                    retryable=True,
                ) from error
            model = self._materialize_robot(preset.name, bundle.asset_id)
            try:
                joint_q, _unknown, _missing, _violations = normalized_joint_q(
                    model,
                    calibration.calibrated_joint_q,
                    clamp=False,
                )
                assessment = assess_calibration_pose(
                    model,
                    request.reference.value,
                    joint_q,
                    reference_motion=self._motion(
                        reference=request.reference,
                        motion_asset_id=request.motion_asset_id,
                    )[0],
                )
                current_validation = self._report(assessment, candidate_id=None)
            finally:
                self._release_robot(model)
            self._record_validation(
                path=path,
                digest=digest,
                robot_id=preset.name,
                robot_asset_id=bundle.asset_id,
                reference=request.reference.value,
                motion_asset_id=request.motion_asset_id,
                validation=current_validation,
            )
            state = CalibrationState.VALID if current_validation.valid else CalibrationState.INVALID
            source = storage
        elif scaler is not None:
            _path, digest, _height = scaler
            calibration_id = None
            state = CalibrationState.BUNDLED
            source = "bundled_scaler"
        else:
            digest = None
            calibration_id = None
            state = CalibrationState.MISSING
            source = "none"

        mapped = set(preset.ik_map or {})
        from hhtools.robot.kinematics import CRITICAL_IK_SLOTS

        return CalibrationStatusResponse(
            request_id=self._request_id_provider(),
            state=state,
            robot_id=preset.name,
            robot_asset_id=bundle.asset_id,
            robot_digest=bundle.asset_id.rsplit(":", 1)[-1],
            reference=request.reference,
            source=source,
            calibration_id=calibration_id,
            calibration_digest=digest,
            joint_q=joint_q,
            joint_count=len(preset.dof_order),
            joint_limits=[
                {
                    "name": name,
                    "lower": lower if lower is not None else -math.pi,
                    "upper": upper if upper is not None else math.pi,
                }
                for name in preset.dof_order
                for lower, upper in [limits.get(name, (None, None))]
            ],
            mapped_slots=min(17, len(mapped)),
            mapping_targets={
                str(canonical): str(
                    target.get("t_body") or target.get("link") or target.get("body") or ""
                )
                if isinstance(target, Mapping)
                else str(target)
                for canonical, target in (preset.ik_map or {}).items()
            },
            missing_slots=sorted(CRITICAL_IK_SLOTS.difference(mapped)),
            can_propose=(
                request.reference is not CalibrationReference.GLB
                or request.motion_asset_id is not None
            ),
            can_silent_save=True,
            current_validation=current_validation,
        )

    def _candidate(self, candidate_id: str) -> CalibrationCandidate:
        try:
            return self._candidate_store.get(candidate_id)
        except CalibrationCandidateStoreError as error:
            code = (
                "CALIBRATION_CANDIDATE_NOT_FOUND"
                if "not found" in str(error)
                else "CALIBRATION_CANDIDATE_INVALID"
            )
            raise _error(code, "The calibration candidate is unavailable or invalid.") from error

    def _candidate_context(
        self,
        candidate: CalibrationCandidate,
    ) -> tuple[Any, RobotPreset, Any | None]:
        request = CalibrationStatusRequest(
            robot_id=candidate.robot_id,
            robot_asset_id=candidate.robot_asset_id,
            reference=candidate.reference,
            motion_asset_id=candidate.motion_asset_id,
        )
        bundle, preset, _limits = self._resolve_robot(request)
        digest = bundle.asset_id.rsplit(":", 1)[-1]
        if digest != candidate.robot_digest:
            raise _error(
                "CALIBRATION_CANDIDATE_STALE",
                "The robot bundle changed after this candidate was proposed.",
                details={"robot_id": candidate.robot_id},
            )
        motion, motion_digest = self._motion(
            reference=candidate.reference,
            motion_asset_id=candidate.motion_asset_id,
        )
        if motion_digest != candidate.motion_digest:
            raise _error(
                "CALIBRATION_CANDIDATE_STALE",
                "The calibration motion changed after this candidate was proposed.",
            )
        return bundle, preset, motion

    def propose(self, request: CalibrationProposalRequest) -> CalibrationProposalResponse:
        bundle, preset, limits = self._resolve_robot(request)
        motion, motion_digest = self._motion(
            reference=request.reference,
            motion_asset_id=request.motion_asset_id,
        )
        parent = None
        if request.base_candidate_id is not None:
            parent = self._candidate(request.base_candidate_id)
            expected = (
                parent.robot_id,
                parent.robot_asset_id,
                parent.reference,
                parent.motion_asset_id,
            )
            actual = (
                preset.name,
                bundle.asset_id,
                request.reference,
                request.motion_asset_id,
            )
            if expected != actual:
                raise _error(
                    "CALIBRATION_CANDIDATE_MISMATCH",
                    "The base candidate belongs to a different calibration identity.",
                )
            seed = dict(parent.joint_q)
            baseline = "candidate"
            baseline_calibration_id = parent.baseline_calibration_id
        else:
            manual = self._manual_profile(preset, request.reference.value, limits, bundle)
            if manual is None:
                seed = {}
                baseline = "urdf_zero"
                baseline_calibration_id = None
            else:
                from hhtools.retarget.calibration import load_calibration

                seed = dict(load_calibration(manual[0]).calibrated_joint_q)
                baseline = "saved_calibration"
                baseline_calibration_id = manual[2]
        seed.update(request.joint_q_overrides)
        locked = frozenset(request.locked_joints)

        model = self._materialize_robot(preset.name, bundle.asset_id)
        try:
            try:
                joint_q, assessment = propose_calibration_pose(
                    model,
                    request.reference.value,
                    seed,
                    locked_joints=locked,
                    reference_motion=motion,
                )
            except (TypeError, ValueError) as error:
                raise _error(
                    "CALIBRATION_PROPOSAL_FAILED",
                    "A constrained calibration candidate could not be constructed.",
                    details={"robot_id": preset.name, "reference": request.reference.value},
                ) from error
        finally:
            self._release_robot(model)

        payload = {
            "schema_version": "1.0",
            "robot_id": preset.name,
            "robot_asset_id": bundle.asset_id,
            "robot_digest": bundle.asset_id.rsplit(":", 1)[-1],
            "reference": request.reference.value,
            "motion_asset_id": request.motion_asset_id,
            "motion_digest": motion_digest,
            "algorithm": CALIBRATION_ALGORITHM,
            "baseline": baseline,
            "baseline_calibration_id": baseline_calibration_id,
            "parent_candidate_id": parent.candidate_id if parent is not None else None,
            "joint_q": dict(sorted(joint_q.items())),
            "locked_joints": sorted(locked),
        }
        candidate_id = compute_calibration_candidate_id(payload)
        candidate = CalibrationCandidate(candidate_id=candidate_id, **payload)
        try:
            candidate = self._candidate_store.put(candidate)
        except CalibrationCandidateStoreError as error:
            raise _error(
                "CALIBRATION_CANDIDATE_STORE_FAILED",
                "The calibration candidate could not be persisted.",
                retryable=True,
            ) from error
        return CalibrationProposalResponse(
            candidate=candidate,
            validation=self._report(assessment, candidate_id=candidate.candidate_id),
        )

    def validate(self, request: CalibrationValidationRequest) -> CalibrationValidationReport:
        candidate = self._candidate(request.candidate_id)
        bundle, preset, motion = self._candidate_context(candidate)
        model = self._materialize_robot(preset.name, bundle.asset_id)
        try:
            assessment = assess_calibration_pose(
                model,
                candidate.reference.value,
                candidate.joint_q,
                reference_motion=motion,
            )
        finally:
            self._release_robot(model)
        return self._report(assessment, candidate_id=candidate.candidate_id)

    def preview(
        self,
        request: CalibrationValidationRequest,
    ) -> tuple[CalibrationPreview, bytes]:
        candidate = self._candidate(request.candidate_id)
        bundle, preset, motion = self._candidate_context(candidate)
        model = self._materialize_robot(preset.name, bundle.asset_id)
        try:
            assessment = assess_calibration_pose(
                model,
                candidate.reference.value,
                candidate.joint_q,
                reference_motion=motion,
            )
            payload = render_calibration_preview_png(
                model,
                candidate.reference.value,
                candidate.joint_q,
                assessment,
                reference_motion=motion,
            )
        finally:
            self._release_robot(model)
        validation = self._report(assessment, candidate_id=candidate.candidate_id)
        return (
            CalibrationPreview(
                candidate_id=candidate.candidate_id,
                sha256=hashlib.sha256(payload).hexdigest(),
                width=CALIBRATION_PREVIEW_WIDTH,
                height=CALIBRATION_PREVIEW_HEIGHT,
                validation=validation,
            ),
            payload,
        )

    def save(self, request: CalibrationSaveRequest) -> CalibrationSaveReceipt:
        candidate = self._candidate(request.candidate_id)
        bundle, preset, motion = self._candidate_context(candidate)
        from hhtools.retarget.calibration import (
            RobotRetargetCalibration,
            derive_calibration_params,
            load_calibration,
            resolve_preset_calibration_file,
            save_calibration_for_preset,
        )

        review_note = ""
        if request.visual_review is not None:
            model_hint = request.visual_review.model_hint or "unspecified"
            review_note = (
                f"; visual_reviewer={request.visual_review.reviewer}; "
                f"model={model_hint}; visual_summary={request.visual_review.summary}"
            )
        notes = (
            f"validated agent calibration; candidate={candidate.candidate_id}; "
            f"save_mode={request.save_mode}{review_note}"
        )
        try:
            previous = resolve_preset_calibration_file(
                preset,
                candidate.reference.value,
                user_root=self._user_robot_root,
            )
            previous_payload = previous.read_bytes() if previous is not None else None
            previous_id = (
                f"cal:sha256:{hashlib.sha256(previous_payload).hexdigest()}"
                if previous_payload is not None
                else None
            )
            existing = load_calibration(previous) if previous is not None else None
        except (OSError, TypeError, ValueError) as error:
            raise _error(
                "CALIBRATION_SAVE_FAILED",
                "The previous calibration could not be verified before saving.",
                retryable=isinstance(error, OSError),
            ) from error
        idempotent_replay = bool(
            existing is not None
            and existing.robot == candidate.robot_id
            and existing.reference == candidate.reference.value
            and existing.calibrated_joint_q == candidate.joint_q
            and existing.notes == notes
        )
        if previous_id != candidate.baseline_calibration_id and not idempotent_replay:
            raise _error(
                "CALIBRATION_CANDIDATE_STALE",
                "The saved calibration changed after this candidate was proposed.",
                details={"robot_id": candidate.robot_id, "reference": candidate.reference.value},
            )
        receipt_previous_id = (
            candidate.baseline_calibration_id if idempotent_replay else previous_id
        )
        if not idempotent_replay and previous_id is not None and previous_payload is not None:
            try:
                self._candidate_store.archive_calibration(previous_id, previous_payload)
            except CalibrationCandidateStoreError as error:
                raise _error(
                    "CALIBRATION_SAVE_FAILED",
                    "The previous calibration could not be archived before saving.",
                    retryable=True,
                ) from error
        previous_archived = bool(
            receipt_previous_id is not None
            and self._candidate_store.has_calibration_archive(receipt_previous_id)
        )
        model = self._materialize_robot(preset.name, bundle.asset_id)
        try:
            assessment = assess_calibration_pose(
                model,
                candidate.reference.value,
                candidate.joint_q,
                reference_motion=motion,
            )
            validation = self._report(
                assessment,
                candidate_id=candidate.candidate_id,
            )
            if not validation.valid:
                raise _error(
                    "CALIBRATION_VALIDATION_FAILED",
                    "The candidate did not pass deterministic calibration validation.",
                    details={"candidate_id": candidate.candidate_id},
                )
            calibration = RobotRetargetCalibration(
                robot=preset.name,
                reference=candidate.reference.value,
                calibrated_joint_q=dict(candidate.joint_q),
                notes=notes,
            )
            if idempotent_replay:
                assert previous is not None
                path = previous
            else:
                derived = derive_calibration_params(
                    calibration,
                    model,
                    reference_motion=motion,
                )
                path = save_calibration_for_preset(
                    calibration,
                    preset,
                    derived=derived,
                    user_robot_root=self._user_robot_root,
                    prefer_user_overlay=True,
                )
        except CalibrationServiceError:
            raise
        except (OSError, TypeError, ValueError) as error:
            raise _error(
                "CALIBRATION_SAVE_FAILED",
                "The validated calibration could not be saved.",
                retryable=isinstance(error, OSError),
            ) from error
        finally:
            self._release_robot(model)
        try:
            saved_calibration, digest = _parse_stable_file(path, load_calibration)
            if (
                saved_calibration.robot != candidate.robot_id
                or saved_calibration.reference != candidate.reference.value
                or saved_calibration.calibrated_joint_q != candidate.joint_q
                or saved_calibration.notes != notes
            ):
                raise ValueError("saved calibration differs from validated candidate")
        except (OSError, TypeError, ValueError) as error:
            raise _error(
                "CALIBRATION_SAVE_FAILED",
                "The saved calibration could not be verified.",
                retryable=True,
            ) from error
        self._record_validation(
            path=path,
            digest=digest,
            robot_id=preset.name,
            robot_asset_id=bundle.asset_id,
            reference=candidate.reference.value,
            motion_asset_id=candidate.motion_asset_id,
            validation=validation,
        )
        return CalibrationSaveReceipt(
            candidate_id=candidate.candidate_id,
            calibration_id=f"cal:sha256:{digest}",
            calibration_digest=digest,
            previous_calibration_id=receipt_previous_id,
            previous_calibration_archived=previous_archived,
            robot_id=preset.name,
            reference=candidate.reference,
            save_mode=request.save_mode,
            validation=validation,
            visual_review=request.visual_review,
        )


__all__ = ["CalibrationService", "CalibrationServiceError"]
