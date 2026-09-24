"""Transport-neutral R2R pair-calibration assistance and silent saving."""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from hhtools.contracts import (
    CalibrationJointLimit,
    CalibrationValidationReport,
    R2RCalibrationCandidate,
    R2RCalibrationPreview,
    R2RCalibrationProposalRequest,
    R2RCalibrationProposalResponse,
    R2RCalibrationSaveReceipt,
    R2RCalibrationSaveRequest,
    R2RCalibrationStatusRequest,
    R2RCalibrationStatusResponse,
    R2RCalibrationValidationRequest,
)
from hhtools.contracts.calibration import CalibrationState
from hhtools.retarget.calibration.assistant import (
    CALIBRATION_PREVIEW_HEIGHT,
    CALIBRATION_PREVIEW_WIDTH,
    R2R_CALIBRATION_ALGORITHM,
    assess_calibration_pose,
    propose_calibration_pose,
    render_calibration_preview_png,
)
from hhtools.retarget.robot_to_robot import (
    build_source_reference_pose,
    load_r2r_calibration_record_file,
    resolve_r2r_calibration_file,
    save_r2r_calibration,
)
from hhtools.robot.base import RobotPreset
from hhtools.robot.kinematics import CRITICAL_IK_SLOTS
from hhtools.utils.paths import user_robot_dir

from .asset_service import AgentAssetService
from .calibration import CalibrationService, _error, _raise_preflight
from .calibration_candidates import (
    CalibrationCandidateStore,
    CalibrationCandidateStoreError,
    compute_calibration_candidate_id,
)
from .preflight import _parse_stable_file, _PreflightFailureError, _robot_bundle_and_preset

RobotProvider = Callable[[], Iterable[RobotPreset]]
RobotMaterializer = Callable[[str, str], Any]
RobotRelease = Callable[[Any], None]


class R2RCalibrationService:
    """Own pair identity, target-pose candidates, previews, and overlay writes."""

    def __init__(
        self,
        asset_service: AgentAssetService,
        candidate_store: CalibrationCandidateStore,
        *,
        robot_provider: RobotProvider,
        materialize_robot: RobotMaterializer,
        release_robot: RobotRelease,
        user_robot_root: Path | None = None,
        request_id_provider: Callable[[], str] = lambda: f"req_{uuid.uuid4().hex}",
    ) -> None:
        self._asset_service = asset_service
        self._candidate_store = candidate_store
        self._robot_provider = robot_provider
        self._materialize_robot = materialize_robot
        self._release_robot = release_robot
        self._user_robot_root = user_robot_root
        self._request_id_provider = request_id_provider

    def _resolve_pair(
        self,
        request: R2RCalibrationStatusRequest,
    ) -> tuple[
        Any,
        RobotPreset,
        Any,
        RobotPreset,
        dict[str, tuple[float | None, float | None]],
    ]:
        presets = tuple(self._robot_provider())
        try:
            source_bundle, source, _source_limits = _robot_bundle_and_preset(
                self._asset_service,
                robot_id=request.source_robot_id,
                robot_asset_id=request.source_robot_asset_id,
                presets=presets,
            )
            target_bundle, target, target_limits = _robot_bundle_and_preset(
                self._asset_service,
                robot_id=request.target_robot_id,
                robot_asset_id=request.target_robot_asset_id,
                presets=presets,
            )
        except _PreflightFailureError as error:
            _raise_preflight(error)
        return source_bundle, source, target_bundle, target, target_limits

    def _stored_calibration(
        self,
        source: RobotPreset,
        target: RobotPreset,
    ) -> tuple[Path, dict[str, float], str, str, str] | None:
        assert target.urdf_path is not None
        try:
            path = resolve_r2r_calibration_file(
                target.urdf_path.parent,
                source.name,
                target_robot=target.name,
                user_root=self._user_robot_root,
            )
            if path is None:
                return None
            record, digest = _parse_stable_file(
                path,
                lambda candidate: load_r2r_calibration_record_file(
                    candidate,
                    source_robot=source.name,
                    target_robot=target.name,
                ),
            )
            managed_root = Path(self._user_robot_root or user_robot_dir()).resolve(strict=False)
            storage: Literal["user_calibration", "robot_bundle"] = (
                "user_calibration"
                if path.resolve(strict=True).is_relative_to(managed_root)
                else "robot_bundle"
            )
        except (OSError, TypeError, ValueError) as error:
            raise _error(
                "R2R_CALIBRATION_INVALID",
                "The saved robot-pair calibration is malformed or changed while reading.",
                details={
                    "source_robot_id": source.name,
                    "target_robot_id": target.name,
                },
                retryable=isinstance(error, OSError),
            ) from error
        joint_q, notes = record
        return path, joint_q, digest, storage, notes

    def _with_models(
        self,
        source: RobotPreset,
        source_asset_id: str,
        target: RobotPreset,
        target_asset_id: str,
        callback: Callable[[Any, Any], Any],
    ) -> Any:
        source_model = self._materialize_robot(source.name, source_asset_id)
        target_model = None
        try:
            target_model = self._materialize_robot(target.name, target_asset_id)
            return callback(source_model, target_model)
        finally:
            if target_model is not None:
                self._release_robot(target_model)
            self._release_robot(source_model)

    @staticmethod
    def _report(assessment: Any, *, candidate_id: str | None):
        return CalibrationService._report(assessment, candidate_id=candidate_id)

    @staticmethod
    def _reference_label(source: RobotPreset) -> str:
        return f"robot:{source.name}"

    def _assess(
        self,
        source_bundle: Any,
        source: RobotPreset,
        target_bundle: Any,
        target: RobotPreset,
        joint_q: Mapping[str, float],
    ) -> Any:
        def assess(source_model: Any, target_model: Any) -> Any:
            reference_pose = build_source_reference_pose(source_model)
            return assess_calibration_pose(
                target_model,
                self._reference_label(source),
                joint_q,
                reference_pose=reference_pose,
            )

        return self._with_models(
            source,
            source_bundle.asset_id,
            target,
            target_bundle.asset_id,
            assess,
        )

    def status(self, request: R2RCalibrationStatusRequest) -> R2RCalibrationStatusResponse:
        source_bundle, source, target_bundle, target, limits = self._resolve_pair(request)
        stored = self._stored_calibration(source, target)
        current_validation = None
        joint_q: dict[str, float] = {}
        if stored is None:
            state = CalibrationState.MISSING
            storage = "none"
            calibration_id = None
            digest = None
        else:
            _path, joint_q, digest, storage, _notes = stored
            assessment = self._assess(
                source_bundle,
                source,
                target_bundle,
                target,
                joint_q,
            )
            current_validation = self._report(assessment, candidate_id=None)
            state = CalibrationState.VALID if current_validation.valid else CalibrationState.INVALID
            calibration_id = f"cal:sha256:{digest}"

        source_mapped = set(source.ik_map or {})
        target_mapped = set(target.ik_map or {})
        return R2RCalibrationStatusResponse(
            request_id=self._request_id_provider(),
            state=state,
            source_robot_id=source.name,
            source_robot_asset_id=source_bundle.asset_id,
            source_robot_digest=source_bundle.asset_id.rsplit(":", 1)[-1],
            target_robot_id=target.name,
            target_robot_asset_id=target_bundle.asset_id,
            target_robot_digest=target_bundle.asset_id.rsplit(":", 1)[-1],
            storage=storage,
            calibration_id=calibration_id,
            calibration_digest=digest,
            joint_q=joint_q,
            joint_count=len(target.dof_order),
            joint_limits=[
                CalibrationJointLimit(
                    name=name,
                    lower=lower if lower is not None else -math.pi,
                    upper=upper if upper is not None else math.pi,
                )
                for name in target.dof_order
                for lower, upper in [limits.get(name, (None, None))]
            ],
            source_mapped_slots=min(17, len(source_mapped)),
            target_mapped_slots=min(17, len(target_mapped)),
            source_missing_slots=sorted(CRITICAL_IK_SLOTS.difference(source_mapped)),
            target_missing_slots=sorted(CRITICAL_IK_SLOTS.difference(target_mapped)),
            can_propose=True,
            can_silent_save=True,
            current_validation=current_validation,
        )

    def _candidate(self, candidate_id: str) -> R2RCalibrationCandidate:
        try:
            return self._candidate_store.get_r2r(candidate_id)
        except CalibrationCandidateStoreError as error:
            code = (
                "CALIBRATION_CANDIDATE_NOT_FOUND"
                if "not found" in str(error)
                else "CALIBRATION_CANDIDATE_INVALID"
            )
            raise _error(
                code,
                "The R2R calibration candidate is unavailable or invalid.",
            ) from error

    def _candidate_context(
        self,
        candidate: R2RCalibrationCandidate,
    ) -> tuple[Any, RobotPreset, Any, RobotPreset]:
        request = R2RCalibrationStatusRequest(
            source_robot_id=candidate.source_robot_id,
            source_robot_asset_id=candidate.source_robot_asset_id,
            target_robot_id=candidate.target_robot_id,
            target_robot_asset_id=candidate.target_robot_asset_id,
        )
        source_bundle, source, target_bundle, target, _limits = self._resolve_pair(request)
        if (
            source_bundle.asset_id.rsplit(":", 1)[-1] != candidate.source_robot_digest
            or target_bundle.asset_id.rsplit(":", 1)[-1] != candidate.target_robot_digest
        ):
            raise _error(
                "CALIBRATION_CANDIDATE_STALE",
                "A robot bundle changed after this R2R candidate was proposed.",
                details={
                    "source_robot_id": candidate.source_robot_id,
                    "target_robot_id": candidate.target_robot_id,
                },
            )
        return source_bundle, source, target_bundle, target

    def propose(
        self,
        request: R2RCalibrationProposalRequest,
    ) -> R2RCalibrationProposalResponse:
        source_bundle, source, target_bundle, target, _limits = self._resolve_pair(request)
        parent = None
        if request.base_candidate_id is not None:
            parent = self._candidate(request.base_candidate_id)
            expected = (
                parent.source_robot_id,
                parent.source_robot_asset_id,
                parent.target_robot_id,
                parent.target_robot_asset_id,
            )
            actual = (
                source.name,
                source_bundle.asset_id,
                target.name,
                target_bundle.asset_id,
            )
            if expected != actual:
                raise _error(
                    "CALIBRATION_CANDIDATE_MISMATCH",
                    "The base candidate belongs to a different robot pair.",
                )
            seed = dict(parent.joint_q)
            baseline = "candidate"
            baseline_calibration_id = parent.baseline_calibration_id
        else:
            stored = self._stored_calibration(source, target)
            if stored is None:
                seed = {}
                baseline = "urdf_zero"
                baseline_calibration_id = None
            else:
                _path, seed, digest, _storage, _notes = stored
                baseline = "saved_calibration"
                baseline_calibration_id = f"cal:sha256:{digest}"
        seed.update(request.joint_q_overrides)
        locked = frozenset(request.locked_joints)

        def solve(source_model: Any, target_model: Any) -> tuple[dict[str, float], Any]:
            reference_pose = build_source_reference_pose(source_model)
            return propose_calibration_pose(
                target_model,
                self._reference_label(source),
                seed,
                locked_joints=locked,
                reference_pose=reference_pose,
            )

        try:
            joint_q, assessment = self._with_models(
                source,
                source_bundle.asset_id,
                target,
                target_bundle.asset_id,
                solve,
            )
        except (TypeError, ValueError) as error:
            raise _error(
                "CALIBRATION_PROPOSAL_FAILED",
                "A constrained R2R calibration candidate could not be constructed.",
                details={
                    "source_robot_id": source.name,
                    "target_robot_id": target.name,
                },
            ) from error

        payload = {
            "schema_version": "1.0",
            "workflow": "r2r",
            "source_robot_id": source.name,
            "source_robot_asset_id": source_bundle.asset_id,
            "source_robot_digest": source_bundle.asset_id.rsplit(":", 1)[-1],
            "target_robot_id": target.name,
            "target_robot_asset_id": target_bundle.asset_id,
            "target_robot_digest": target_bundle.asset_id.rsplit(":", 1)[-1],
            "algorithm": R2R_CALIBRATION_ALGORITHM,
            "baseline": baseline,
            "baseline_calibration_id": baseline_calibration_id,
            "parent_candidate_id": parent.candidate_id if parent is not None else None,
            "joint_q": dict(sorted(joint_q.items())),
            "locked_joints": sorted(locked),
        }
        candidate_id = compute_calibration_candidate_id(payload)
        candidate = R2RCalibrationCandidate(candidate_id=candidate_id, **payload)
        try:
            candidate = self._candidate_store.put_r2r(candidate)
        except CalibrationCandidateStoreError as error:
            raise _error(
                "CALIBRATION_CANDIDATE_STORE_FAILED",
                "The R2R calibration candidate could not be persisted.",
                retryable=True,
            ) from error
        return R2RCalibrationProposalResponse(
            candidate=candidate,
            validation=self._report(assessment, candidate_id=candidate.candidate_id),
        )

    def validate(
        self,
        request: R2RCalibrationValidationRequest,
    ) -> CalibrationValidationReport:
        candidate = self._candidate(request.candidate_id)
        source_bundle, source, target_bundle, target = self._candidate_context(candidate)
        assessment = self._assess(
            source_bundle,
            source,
            target_bundle,
            target,
            candidate.joint_q,
        )
        return self._report(assessment, candidate_id=candidate.candidate_id)

    def preview(
        self,
        request: R2RCalibrationValidationRequest,
    ) -> tuple[R2RCalibrationPreview, bytes]:
        candidate = self._candidate(request.candidate_id)
        source_bundle, source, target_bundle, target = self._candidate_context(candidate)

        def render(source_model: Any, target_model: Any) -> tuple[Any, bytes]:
            reference_pose = build_source_reference_pose(source_model)
            assessment = assess_calibration_pose(
                target_model,
                self._reference_label(source),
                candidate.joint_q,
                reference_pose=reference_pose,
            )
            payload = render_calibration_preview_png(
                target_model,
                self._reference_label(source),
                candidate.joint_q,
                assessment,
                reference_pose=reference_pose,
            )
            return assessment, payload

        assessment, payload = self._with_models(
            source,
            source_bundle.asset_id,
            target,
            target_bundle.asset_id,
            render,
        )
        validation = self._report(assessment, candidate_id=candidate.candidate_id)
        return (
            R2RCalibrationPreview(
                candidate_id=candidate.candidate_id,
                sha256=hashlib.sha256(payload).hexdigest(),
                width=CALIBRATION_PREVIEW_WIDTH,
                height=CALIBRATION_PREVIEW_HEIGHT,
                validation=validation,
            ),
            payload,
        )

    def save(self, request: R2RCalibrationSaveRequest) -> R2RCalibrationSaveReceipt:
        candidate = self._candidate(request.candidate_id)
        source_bundle, source, target_bundle, target = self._candidate_context(candidate)
        review_note = ""
        if request.visual_review is not None:
            model_hint = request.visual_review.model_hint or "unspecified"
            review_note = (
                f"; visual_reviewer={request.visual_review.reviewer}; "
                f"model={model_hint}; visual_summary={request.visual_review.summary}"
            )
        notes = (
            f"validated agent R2R calibration; candidate={candidate.candidate_id}; "
            f"save_mode={request.save_mode}{review_note}"
        )
        stored = self._stored_calibration(source, target)
        previous_path = stored[0] if stored is not None else None
        previous_joint_q = stored[1] if stored is not None else None
        previous_digest = stored[2] if stored is not None else None
        previous_storage = stored[3] if stored is not None else None
        previous_notes = stored[4] if stored is not None else None
        previous_id = f"cal:sha256:{previous_digest}" if previous_digest is not None else None
        idempotent_replay = (
            previous_storage == "user_calibration"
            and previous_joint_q == candidate.joint_q
            and previous_notes == notes
        )
        if previous_id != candidate.baseline_calibration_id and not idempotent_replay:
            raise _error(
                "CALIBRATION_CANDIDATE_STALE",
                "The saved R2R calibration changed after this candidate was proposed.",
                details={
                    "source_robot_id": source.name,
                    "target_robot_id": target.name,
                },
            )

        validation = self.validate(
            R2RCalibrationValidationRequest(candidate_id=candidate.candidate_id)
        )
        if not validation.valid:
            raise _error(
                "CALIBRATION_VALIDATION_FAILED",
                "The R2R candidate did not pass deterministic calibration validation.",
                details={"candidate_id": candidate.candidate_id},
            )

        receipt_previous_id = (
            candidate.baseline_calibration_id if idempotent_replay else previous_id
        )
        if not idempotent_replay and previous_path is not None and previous_id is not None:
            try:
                previous_payload = previous_path.read_bytes()
                if hashlib.sha256(previous_payload).hexdigest() != previous_digest:
                    raise CalibrationCandidateStoreError(
                        "previous calibration changed before archival"
                    )
                self._candidate_store.archive_calibration(previous_id, previous_payload)
            except (OSError, CalibrationCandidateStoreError) as error:
                raise _error(
                    "CALIBRATION_SAVE_FAILED",
                    "The previous R2R calibration could not be archived before saving.",
                    retryable=True,
                ) from error

        if not idempotent_replay:
            assert target.urdf_path is not None
            try:
                path = save_r2r_calibration(
                    target.urdf_path.parent,
                    target_robot=target.name,
                    source_robot=source.name,
                    calibrated_joint_q=dict(candidate.joint_q),
                    user_root=self._user_robot_root,
                    prefer_user_overlay=True,
                    notes=notes,
                )
            except (OSError, TypeError, ValueError) as error:
                raise _error(
                    "CALIBRATION_SAVE_FAILED",
                    "The validated R2R calibration could not be saved.",
                    retryable=isinstance(error, OSError),
                ) from error
        else:
            assert previous_path is not None
            path = previous_path

        try:
            saved_record, digest = _parse_stable_file(
                path,
                lambda candidate_path: load_r2r_calibration_record_file(
                    candidate_path,
                    source_robot=source.name,
                    target_robot=target.name,
                ),
            )
        except (OSError, TypeError, ValueError) as error:
            raise _error(
                "CALIBRATION_SAVE_FAILED",
                "The saved R2R calibration could not be verified.",
                retryable=isinstance(error, OSError),
            ) from error
        saved_joint_q, saved_notes = saved_record
        if saved_joint_q != candidate.joint_q or saved_notes != notes:
            raise _error(
                "CALIBRATION_SAVE_FAILED",
                "The saved R2R calibration does not match the validated candidate.",
            )

        previous_archived = bool(
            receipt_previous_id is not None
            and self._candidate_store.has_calibration_archive(receipt_previous_id)
        )
        return R2RCalibrationSaveReceipt(
            candidate_id=candidate.candidate_id,
            calibration_id=f"cal:sha256:{digest}",
            calibration_digest=digest,
            previous_calibration_id=receipt_previous_id,
            previous_calibration_archived=previous_archived,
            source_robot_id=source.name,
            target_robot_id=target.name,
            save_mode=request.save_mode,
            validation=validation,
            visual_review=request.visual_review,
        )


__all__ = ["R2RCalibrationService"]
