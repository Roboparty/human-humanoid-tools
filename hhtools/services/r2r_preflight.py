"""Read-only R2R preflight and immutable pair-plan construction."""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from hhtools.contracts import (
    AssetBundle,
    AssetCategory,
    AssetInspection,
    AssetInspectionRequest,
    AssetKind,
    CapabilityResponse,
    InspectionStatus,
    NextAction,
    OutputPolicy,
    PreflightCheck,
    PreflightCheckLevel,
    PreflightStatus,
    R2RCalibrationStatusRequest,
    R2RPlan,
    R2RPreflightRequest,
    R2RPreflightResponse,
)
from hhtools.robot.base import RobotPreset
from hhtools.services.asset_service import AgentAssetService
from hhtools.services.assets import AssetServiceError
from hhtools.services.plans import PlanStore, PlanStoreError, compute_plan_id
from hhtools.utils.paths import user_robot_dir

from .preflight import (
    _asset_digest,
    _backend_capability,
    _bundle_registration_request,
    _check,
    _contained_file,
    _fail,
    _manifest_hashes,
    _output_format,
    _parse_stable_file,
    _PreflightFailureError,
    _register_asset_action,
    _robot_bundle_and_preset,
    _scheduler_check,
)

R2R_PLAN_SEMANTICS = "hhtools.r2r.plan.v1"
_PORTABLE_ROBOT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PARAMETERS = frozenset(
    {
        "run_mode",
        "limit_frames",
        "ik_iterations",
        "source_fps",
        "retarget_fps",
    }
)
_DEFAULT_MAX_IK_ITERATIONS = 200
_DEFAULT_MAX_RETARGET_FPS = 1_000.0
_DEFAULT_MAX_RETARGET_FRAMES = 100_000


def _raise_asset_error(error: AssetServiceError) -> None:
    public = error.api_error
    raise _PreflightFailureError(
        public,
        PreflightCheck(
            code=public.code,
            level=PreflightCheckLevel.ERROR,
            message=public.message,
            details=public.details,
            next_action=public.next_action,
        ),
    ) from error


def _trajectory(
    asset_service: AgentAssetService,
    asset_id: str,
) -> tuple[AssetBundle, AssetInspection]:
    try:
        bundle = asset_service.get(asset_id)
        if bundle.kind is not AssetKind.ROBOT_TRAJECTORY_BUNDLE:
            _fail(
                "ASSET_KIND_MISMATCH",
                "trajectory_asset_id must refer to a robot_trajectory_bundle.",
                details={"asset_id": asset_id, "kind": bundle.kind.value},
            )
        inspection = asset_service.inspect(
            AssetInspectionRequest(
                asset_id=asset_id,
                verify_hashes=True,
                parse_content=True,
            )
        )
    except AssetServiceError as error:
        _raise_asset_error(error)
    if inspection.status is InspectionStatus.INVALID:
        if inspection.errors:
            public = inspection.errors[0]
            raise _PreflightFailureError(
                public,
                PreflightCheck(
                    code=public.code,
                    level=PreflightCheckLevel.ERROR,
                    message=public.message,
                    details=public.details,
                    next_action=public.next_action,
                ),
            )
        _fail(
            "ROBOT_TRAJECTORY_INVALID",
            "The registered robot trajectory did not pass safe inspection.",
        )
    if inspection.category is not AssetCategory.ROBOT_TRAJECTORY:
        _fail(
            "BUNDLE_METADATA_MISMATCH",
            "The registered trajectory has a different workflow category.",
            details={"category": inspection.category.value},
        )
    if not inspection.metadata.get("content_parsed"):
        _fail(
            str(
                inspection.metadata.get("content_validation_code")
                or "CONTENT_REQUIRES_WORKFLOW_SCHEMA"
            ),
            "The robot trajectory requires safe semantic validation before execution.",
        )
    return bundle, inspection


def _positive_float(
    parameters: Mapping[str, Any],
    name: str,
    *,
    maximum: float,
) -> float | None:
    raw = parameters.get(name)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        _fail(
            "INVALID_PARAMETER",
            f"{name} must be a finite positive number.",
            details={"parameter": name},
        )
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0 or value > maximum:
        _fail(
            "INVALID_PARAMETER",
            f"{name} must be a finite positive number within the backend limit.",
            details={"parameter": name, "maximum": maximum},
        )
    return value


def _normalize_parameters(
    request: R2RPreflightRequest,
    inspection: AssetInspection,
    *,
    backend_limits: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = request.parameters
    unknown = sorted(set(parameters).difference(_PARAMETERS))
    if unknown:
        _fail(
            "INVALID_PARAMETER",
            "The R2R request contains unsupported parameters.",
            details={"parameters": unknown},
        )
    run_mode = parameters.get("run_mode", "smoke")
    if run_mode not in {"smoke", "full"}:
        _fail(
            "INVALID_PARAMETER",
            "run_mode must be smoke or full.",
            details={"parameter": "run_mode"},
        )

    maximum_frames = int(backend_limits.get("max_retarget_frames", _DEFAULT_MAX_RETARGET_FRAMES))
    raw_limit = parameters.get("limit_frames")
    if run_mode == "full":
        if raw_limit is not None:
            _fail(
                "INVALID_PARAMETER",
                "A full R2R plan cannot declare limit_frames.",
                details={"parameter": "limit_frames"},
            )
        limit_frames = None
    else:
        limit_frames = 30 if raw_limit is None else raw_limit
        if (
            isinstance(limit_frames, bool)
            or not isinstance(limit_frames, int)
            or limit_frames <= 0
            or limit_frames > maximum_frames
        ):
            _fail(
                "INVALID_PARAMETER",
                "limit_frames must be a positive integer within the backend limit.",
                details={"parameter": "limit_frames", "maximum": maximum_frames},
            )
        if inspection.frame_count is not None:
            limit_frames = min(limit_frames, inspection.frame_count)

    maximum_iterations = int(backend_limits.get("max_ik_iterations", _DEFAULT_MAX_IK_ITERATIONS))
    ik_iterations = parameters.get("ik_iterations", 24)
    if (
        isinstance(ik_iterations, bool)
        or not isinstance(ik_iterations, int)
        or ik_iterations <= 0
        or ik_iterations > maximum_iterations
    ):
        _fail(
            "INVALID_PARAMETER",
            "ik_iterations must be a positive integer within the backend limit.",
            details={"parameter": "ik_iterations", "maximum": maximum_iterations},
        )

    maximum_fps = float(backend_limits.get("max_retarget_fps", _DEFAULT_MAX_RETARGET_FPS))
    source_fps = _positive_float(parameters, "source_fps", maximum=maximum_fps)
    if inspection.frame_rate_hz is None and source_fps is None:
        _fail(
            "SOURCE_FPS_REQUIRED",
            "The trajectory has no frame-rate metadata; source_fps is required.",
            details={"parameter": "source_fps"},
        )
    retarget_fps = _positive_float(parameters, "retarget_fps", maximum=maximum_fps)
    return {
        "run_mode": run_mode,
        "limit_frames": limit_frames,
        "ik_iterations": ik_iterations,
        "source_fps": source_fps,
        "retarget_fps": retarget_fps,
        "trajectory_profile": "mimic",
    }


def _calibration_action(
    source_robot_id: str,
    source_robot_asset_id: str,
    target_robot_id: str,
    target_robot_asset_id: str,
) -> NextAction:
    request = R2RCalibrationStatusRequest(
        source_robot_id=source_robot_id,
        source_robot_asset_id=source_robot_asset_id,
        target_robot_id=target_robot_id,
        target_robot_asset_id=target_robot_asset_id,
    )
    return NextAction(
        actor="agent",
        action="get_r2r_calibration_status",
        message="Inspect this robot pair and generate a validated target-pose candidate.",
        parameters={"request": request.model_dump(mode="json", exclude_none=True)},
    )


def _pair_calibration(
    request: R2RPreflightRequest,
    *,
    source: RobotPreset,
    source_bundle: AssetBundle,
    target: RobotPreset,
    target_bundle: AssetBundle,
    target_limits: Mapping[str, tuple[float | None, float | None]],
) -> tuple[str, str, str]:
    from hhtools.retarget.robot_to_robot import (
        load_r2r_calibration_file,
        resolve_r2r_calibration_file,
    )

    assert target.urdf_path is not None
    managed_user_root = user_robot_dir().resolve(strict=False)
    try:
        path = resolve_r2r_calibration_file(
            target.urdf_path.parent,
            source.name,
            target_robot=target.name,
            user_root=managed_user_root,
        )
    except (OSError, TypeError, ValueError):
        _fail(
            "R2R_CALIBRATION_MISMATCH",
            "The matching R2R calibration exists but is malformed.",
            details={
                "source_robot_id": source.name,
                "target_robot_id": target.name,
            },
        )
    if path is None:
        action = _calibration_action(
            source.name,
            source_bundle.asset_id,
            target.name,
            target_bundle.asset_id,
        )
        raise _PreflightFailureError(
            error := _error_for_action(
                "R2R_CALIBRATION_REQUIRED",
                "The target robot must be calibrated against this source robot.",
                action,
            ),
            PreflightCheck(
                code=error.code,
                level=PreflightCheckLevel.ERROR,
                message=error.message,
                details=error.details,
                next_action=action,
            ),
        )

    try:
        try:
            contained = _contained_file(path, managed_user_root)
            storage = "user_calibration"
            relative_path = contained.relative_to(managed_user_root).as_posix()
        except ValueError:
            contained = _contained_file(path, target.root_dir)
            storage = "robot_bundle"
            relative_path = contained.relative_to(target.root_dir).as_posix()
        calibration, digest = _parse_stable_file(
            contained,
            lambda candidate: load_r2r_calibration_file(
                candidate,
                source_robot=source.name,
                target_robot=target.name,
            ),
        )
    except (OSError, TypeError, ValueError):
        _fail(
            "R2R_CALIBRATION_MISMATCH",
            "The matching R2R calibration could not be validated.",
            details={
                "source_robot_id": source.name,
                "target_robot_id": target.name,
            },
        )

    unknown = sorted(set(calibration).difference(target.dof_order))
    if unknown:
        _fail(
            "R2R_CALIBRATION_MISMATCH",
            "The R2R calibration contains joints outside the target DOF order.",
            details={"joint_names": unknown},
        )
    for name, value in calibration.items():
        lower, upper = target_limits.get(name, (None, None))
        if (lower is not None and value < lower) or (upper is not None and value > upper):
            _fail(
                "R2R_CALIBRATION_MISMATCH",
                "The R2R calibration contains a joint value outside its URDF limit.",
                details={"joint_name": name},
            )
    if storage == "robot_bundle" and digest not in _manifest_hashes(
        target_bundle,
        role="metadata",
    ):
        action = _register_asset_action(
            _bundle_registration_request(target_bundle),
            message="Register the target robot bundle again to bind its R2R calibration.",
        )
        _fail(
            "ROBOT_BUNDLE_MISMATCH",
            "The pair calibration is not bound into the target robot bundle.",
            next_action=action,
        )
    calibration_id = f"cal:sha256:{digest}"
    if request.calibration_id is not None and request.calibration_id != calibration_id:
        _fail(
            "R2R_CALIBRATION_MISMATCH",
            "The requested calibration id does not match the selected robot pair.",
            details={"expected_calibration_id": calibration_id},
        )
    return digest, relative_path, storage


def _error_for_action(code: str, message: str, action: NextAction):
    from hhtools.contracts import ApiError, ErrorStage

    return ApiError(
        code=code,
        message=message,
        stage=ErrorStage.PREFLIGHT,
        details=dict(action.parameters),
        next_action=action,
    )


class R2RPreflightService:
    """Resolve one safe, scene-free R2R request without starting a solver."""

    def __init__(
        self,
        asset_service: AgentAssetService,
        plan_store: PlanStore,
        *,
        capabilities_provider: Callable[[], CapabilityResponse],
        robot_provider: Callable[[], Iterable[RobotPreset]],
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        request_id_provider: Callable[[], str] = lambda: f"req_{uuid.uuid4().hex}",
    ) -> None:
        self._asset_service = asset_service
        self._plan_store = plan_store
        self._capabilities_provider = capabilities_provider
        self._robot_provider = robot_provider
        self._clock = clock
        self._request_id_provider = request_id_provider

    def preflight_r2r(self, request: R2RPreflightRequest) -> R2RPreflightResponse:
        request_id = self._request_id_provider()
        checks: list[PreflightCheck] = []
        recommended_backend = "newton"
        try:
            if request.output_policy is not OutputPolicy.CREATE_NEW:
                _fail(
                    "UNSUPPORTED_OUTPUT_POLICY",
                    "Managed Agent artifacts currently support only create_new output policy.",
                    details={"output_policy": request.output_policy.value},
                )
            for parameter, robot_id in (
                ("source_robot_id", request.source_robot_id),
                ("target_robot_id", request.target_robot_id),
            ):
                if _PORTABLE_ROBOT_ID.fullmatch(robot_id) is None:
                    _fail(
                        "INVALID_PARAMETER",
                        f"{parameter} must be a portable identifier.",
                        details={"parameter": parameter},
                    )
            if request.source_robot_id == request.target_robot_id:
                _fail(
                    "R2R_ROBOT_PAIR_INVALID",
                    "Source and target robots must be different presets.",
                )

            trajectory_bundle, trajectory_inspection = _trajectory(
                self._asset_service,
                request.trajectory_asset_id,
            )
            if trajectory_inspection.has_object or trajectory_inspection.has_terrain:
                _fail(
                    "R2R_SCENE_UNSUPPORTED",
                    "The initial Agent R2R workflow accepts only scene-free trajectories.",
                    details={
                        "has_object": trajectory_inspection.has_object,
                        "has_terrain": trajectory_inspection.has_terrain,
                    },
                )
            checks.append(
                _check(
                    "TRAJECTORY_PARSEABLE",
                    PreflightCheckLevel.PASS,
                    "The robot trajectory passed safe semantic inspection.",
                    details={
                        "frame_count": trajectory_inspection.frame_count,
                        "source_format": trajectory_inspection.source_format,
                    },
                )
            )

            requested_backend = request.backend or recommended_backend
            if requested_backend != recommended_backend:
                _fail(
                    "BACKEND_INCOMPATIBLE",
                    "The initial Agent R2R workflow supports only the Newton backend.",
                    details={"backend": requested_backend},
                )
            capabilities = self._capabilities_provider()
            backend = _backend_capability(capabilities, requested_backend)
            if AssetCategory.ROBOT_TRAJECTORY not in backend.supported_categories:
                _fail(
                    "BACKEND_INCOMPATIBLE",
                    "The backend capability does not advertise R2R trajectories.",
                    details={"backend": backend.backend_id},
                )
            checks.append(
                _check(
                    "BACKEND_READY",
                    PreflightCheckLevel.PASS,
                    "The Newton backend is installed and available for R2R.",
                    details={"backend": backend.backend_id},
                )
            )

            presets = tuple(self._robot_provider())
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
            if source_bundle.asset_id == target_bundle.asset_id:
                _fail(
                    "R2R_ROBOT_PAIR_INVALID",
                    "Source and target robot assets must be different.",
                )
            checks.append(
                _check(
                    "ROBOT_PAIR_READY",
                    PreflightCheckLevel.PASS,
                    "Both robot bundles, DOF orders, and IK maps passed inspection.",
                    details={
                        "source_robot_id": source.name,
                        "target_robot_id": target.name,
                    },
                )
            )

            declared_source = trajectory_inspection.source_robot_id
            if declared_source is not None and declared_source != source.name:
                _fail(
                    "SOURCE_ROBOT_MISMATCH",
                    "The trajectory declares a different source robot.",
                    details={
                        "declared_source_robot_id": declared_source,
                        "requested_source_robot_id": source.name,
                    },
                )
            raw_dof_names = trajectory_inspection.metadata.get("dof_names")
            dof_names = (
                tuple(str(value) for value in raw_dof_names)
                if isinstance(raw_dof_names, list)
                else ()
            )
            if dof_names and dof_names != tuple(source.dof_order):
                _fail(
                    "SOURCE_ROBOT_MISMATCH",
                    "The trajectory DOF order does not match the source robot bundle.",
                    details={
                        "trajectory_dof_count": len(dof_names),
                        "source_dof_count": len(source.dof_order),
                    },
                )
            if not dof_names and declared_source is None:
                _fail(
                    "SOURCE_ROBOT_IDENTITY_UNPROVEN",
                    "The trajectory must declare its source robot or an exact DOF order.",
                )
            if (
                trajectory_inspection.joint_count is not None
                and trajectory_inspection.joint_count != len(source.dof_order)
            ):
                _fail(
                    "SOURCE_ROBOT_MISMATCH",
                    "The trajectory DOF count does not match the source robot bundle.",
                    details={
                        "trajectory_dof_count": trajectory_inspection.joint_count,
                        "source_dof_count": len(source.dof_order),
                    },
                )
            checks.append(
                _check(
                    "SOURCE_ROBOT_MATCH",
                    PreflightCheckLevel.PASS,
                    "The trajectory identity and DOF schema match the source robot.",
                    details={"source_robot_id": source.name},
                )
            )

            calibration_digest, calibration_path, calibration_storage = _pair_calibration(
                request,
                source=source,
                source_bundle=source_bundle,
                target=target,
                target_bundle=target_bundle,
                target_limits=target_limits,
            )
            calibration_id = f"cal:sha256:{calibration_digest}"
            checks.append(
                _check(
                    "PAIR_CALIBRATION_MATCH",
                    PreflightCheckLevel.PASS,
                    "The pair calibration matches both robot identities.",
                    details={
                        "source_robot_id": source.name,
                        "target_robot_id": target.name,
                        "storage": calibration_storage,
                    },
                )
            )

            output_format = _output_format(request, backend, capabilities)
            parameters = _normalize_parameters(
                request,
                trajectory_inspection,
                backend_limits=backend.limits,
            )
            checks.append(
                _check(
                    "PARAMETERS_VALID",
                    PreflightCheckLevel.PASS,
                    "R2R parameters and output policy were normalized successfully.",
                    details={
                        "run_mode": parameters["run_mode"],
                        "output_format": output_format,
                    },
                )
            )
            checks.append(_scheduler_check(capabilities.scheduler))

            canonical_payload = {
                "semantics": R2R_PLAN_SEMANTICS,
                "trajectory": {
                    "asset_id": trajectory_bundle.asset_id,
                    "digest": _asset_digest(trajectory_bundle.asset_id),
                    "category": AssetCategory.ROBOT_TRAJECTORY.value,
                    "source_robot_id": source.name,
                    "declared_source_robot_id": declared_source,
                    "profile": "mimic",
                },
                "source_robot": {
                    "asset_id": source_bundle.asset_id,
                    "digest": _asset_digest(source_bundle.asset_id),
                    "robot_id": source.name,
                },
                "target_robot": {
                    "asset_id": target_bundle.asset_id,
                    "digest": _asset_digest(target_bundle.asset_id),
                    "robot_id": target.name,
                },
                "backend": backend.backend_id,
                "pair_calibration": {
                    "source_robot_id": source.name,
                    "target_robot_id": target.name,
                    "calibration_id": calibration_id,
                    "digest": calibration_digest,
                    "storage": calibration_storage,
                    "relative_path": calibration_path,
                },
                "output": {
                    "format": output_format,
                    "policy": request.output_policy.value,
                },
                "parameters": parameters,
            }
            plan_id = compute_plan_id(canonical_payload)
            try:
                plan = self._plan_store.get(plan_id)
            except PlanStoreError as error:
                if error.code != "PLAN_NOT_FOUND":
                    raise
                candidate = R2RPlan(
                    plan_id=plan_id,
                    created_at=self._clock(),
                    trajectory_asset_id=trajectory_bundle.asset_id,
                    source_robot_id=source.name,
                    source_robot_asset_id=source_bundle.asset_id,
                    target_robot_id=target.name,
                    target_robot_asset_id=target_bundle.asset_id,
                    backend=backend.backend_id,
                    calibration_id=calibration_id,
                    output_format=output_format,
                    output_policy=request.output_policy,
                    parameters=parameters,
                    trajectory_digest=_asset_digest(trajectory_bundle.asset_id),
                    source_robot_digest=_asset_digest(source_bundle.asset_id),
                    target_robot_digest=_asset_digest(target_bundle.asset_id),
                    calibration_digest=calibration_digest,
                )
                try:
                    plan = self._plan_store.put_if_absent(candidate, canonical_payload)
                except PlanStoreError as conflict:
                    if conflict.code != "PLAN_CONFLICT":
                        raise
                    plan = self._plan_store.get(plan_id)
                    if self._plan_store.get_payload(plan_id) != canonical_payload:
                        raise
            if not isinstance(plan, R2RPlan):
                _fail("PLAN_CONFLICT", "The persisted plan has another workflow type.")
            return R2RPreflightResponse(
                request_id=request_id,
                status=PreflightStatus.READY,
                plan=plan,
                checks=checks,
                recommended_backend=recommended_backend,
            )
        except _PreflightFailureError as failure:
            checks.append(failure.check)
            if failure.error.code == "R2R_CALIBRATION_REQUIRED":
                assert failure.error.next_action is not None
                return R2RPreflightResponse(
                    request_id=request_id,
                    status=PreflightStatus.HUMAN_ACTION_REQUIRED,
                    checks=checks,
                    recommended_backend=recommended_backend,
                    required_actions=[failure.error.next_action],
                )
            return R2RPreflightResponse(
                request_id=request_id,
                status=PreflightStatus.REJECTED,
                checks=checks,
                recommended_backend=recommended_backend,
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
            return R2RPreflightResponse(
                request_id=request_id,
                status=PreflightStatus.REJECTED,
                checks=checks,
                recommended_backend=recommended_backend,
                error=public,
            )


__all__ = ["R2R_PLAN_SEMANTICS", "R2RPreflightService"]
