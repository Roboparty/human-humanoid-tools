"""Read-only retarget preflight and immutable plan construction.

The preflight boundary deliberately stops before solver construction and job
admission.  It validates content-addressed assets, a read-only robot preset,
backend compatibility, calibration/scaler readiness, effective parameters,
and the scheduler snapshot.  A successful call persists a portable plan; it
never imports a retarget pipeline, compiles MuJoCo, initializes Warp/Newton,
or reserves a queue slot.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn
from xml.etree import ElementTree

from yaml import YAMLError  # type: ignore[import-untyped]

from hhtools.contracts import (
    ApiError,
    AssetBundle,
    AssetCategory,
    AssetInspection,
    AssetInspectionRequest,
    AssetKind,
    AssetRegistrationRequest,
    AssetSourceScheme,
    BackendCapability,
    CalibrationStatusRequest,
    CapabilityResponse,
    ErrorStage,
    InspectionStatus,
    NextAction,
    OutputPolicy,
    PreflightCheck,
    PreflightCheckLevel,
    PreflightResponse,
    PreflightStatus,
    RetargetPlan,
    RetargetPreflightRequest,
    SchedulerCapability,
)
from hhtools.robot.base import RobotPreset
from hhtools.services.asset_service import AgentAssetService
from hhtools.services.assets import AssetServiceError
from hhtools.services.calibration_validation import (
    CALIBRATION_VALIDATION_VERSION,
    CalibrationValidationStore,
    calibration_validation_identity,
)
from hhtools.services.plans import PlanStore, PlanStoreError, compute_plan_id
from hhtools.utils.paths import user_robot_dir

_PLAN_SEMANTICS = "hhtools.retarget.plan.v1"
_SUPPORTED_OUTPUT_FORMATS = frozenset({"csv", "pkl"})
_PARAMETERS = frozenset(
    {
        "run_mode",
        "limit_frames",
        "ik_iterations",
        "human_height",
        "retarget_fps",
        "foot_clamp_anti_penetration",
    }
)
_ACTUATED_JOINT_TYPES = frozenset({"revolute", "continuous", "prismatic"})
_DEFAULT_MAX_IK_ITERATIONS = 200
_DEFAULT_MAX_RETARGET_FPS = 1_000.0
_DEFAULT_MAX_RETARGET_FRAMES = 100_000
_DEFAULT_MAX_HUMAN_HEIGHT = 10.0
_PORTABLE_ROBOT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class _PreflightFailureError(RuntimeError):
    """Private short-circuit carrying a safe public error and check."""

    def __init__(self, error: ApiError, check: PreflightCheck) -> None:
        self.error = error
        self.check = check
        super().__init__(f"{error.code}: {error.message}")


def _error(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
    next_action: NextAction | None = None,
    retryable: bool = False,
    stage: ErrorStage = ErrorStage.PREFLIGHT,
) -> ApiError:
    return ApiError(
        code=code,
        message=message,
        retryable=retryable,
        stage=stage,
        details=dict(details or {}),
        next_action=next_action,
    )


def _fail(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
    next_action: NextAction | None = None,
    retryable: bool = False,
) -> NoReturn:
    error = _error(
        code,
        message,
        details=details,
        next_action=next_action,
        retryable=retryable,
    )
    raise _PreflightFailureError(
        error,
        PreflightCheck(
            code=code,
            level=PreflightCheckLevel.ERROR,
            message=message,
            details=dict(details or {}),
            next_action=next_action,
        ),
    )


def _check(
    code: str,
    level: PreflightCheckLevel,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> PreflightCheck:
    return PreflightCheck(
        code=code,
        level=level,
        message=message,
        details=dict(details or {}),
    )


def _asset_digest(asset_id: str) -> str:
    return asset_id.rsplit(":", 1)[-1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _FileIdentityMismatchError(ValueError):
    """A file no longer matches the content identity selected for preflight."""


class _FileChangedError(ValueError):
    """A file changed while a parser was reading it."""


def _parse_stable_file(
    path: Path,
    parser: Callable[[Path], Any],
    *,
    expected_digests: set[str] | None = None,
) -> tuple[Any, str]:
    """Parse one file only when the bytes stay stable around the read.

    The parsers used by the existing robot/calibration modules accept paths.
    Hashing immediately before and after parsing prevents a plan from binding
    the digest of one revision to fields parsed from another revision.
    """

    before = _sha256_file(path)
    if expected_digests is not None and before not in expected_digests:
        raise _FileIdentityMismatchError("file does not match its selected identity")
    parsed = parser(path)
    after = _sha256_file(path)
    if after != before:
        raise _FileChangedError("file changed while it was being parsed")
    return parsed, before


def _safe_asset_error(error: AssetServiceError) -> ApiError:
    """Keep stable asset codes while moving the failure to preflight."""

    return error.api_error.model_copy(update={"stage": ErrorStage.PREFLIGHT})


def _raise_asset_error(error: AssetServiceError) -> NoReturn:
    public = _safe_asset_error(error)
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


def _raise_inspection_error(
    inspection: AssetInspection,
    *,
    fallback_code: str,
    fallback_message: str,
) -> NoReturn:
    """Preserve the most actionable inspector code at the preflight stage."""

    priorities = {
        "ASSET_HASH_MISMATCH": 0,
        "ASSET_OUTSIDE_ALLOWED_ROOT": 1,
        "ASSET_NOT_FOUND": 2,
        "BUNDLE_INCOMPLETE": 3,
        "BUNDLE_METADATA_MISMATCH": 4,
        "UNSUPPORTED_FORMAT": 5,
    }
    if inspection.errors:
        selected = min(
            inspection.errors,
            key=lambda item: priorities.get(item.code, 100),
        ).model_copy(update={"stage": ErrorStage.PREFLIGHT})
    else:
        selected = _error(fallback_code, fallback_message)
    raise _PreflightFailureError(
        selected,
        PreflightCheck(
            code=selected.code,
            level=PreflightCheckLevel.ERROR,
            message=selected.message,
            details=selected.details,
            next_action=selected.next_action,
        ),
    )


def _backend_for_category(category: AssetCategory) -> str:
    if category is AssetCategory.PLAIN_MOTION:
        return "newton"
    if category in {AssetCategory.OBJECT_INTERACTION, AssetCategory.TERRAIN_SCENE}:
        return "interaction_mesh"
    _fail(
        "BACKEND_INCOMPATIBLE",
        "The registered input category is not supported by retarget preflight.",
        details={"category": category.value},
    )


def _backend_capability(
    capabilities: CapabilityResponse,
    backend_id: str,
) -> BackendCapability:
    backend = next(
        (item for item in capabilities.backends if item.backend_id == backend_id),
        None,
    )
    if backend is None:
        _fail(
            "BACKEND_UNAVAILABLE",
            "The requested retarget backend is not advertised by this service.",
            details={"backend": backend_id},
        )
    if not backend.available:
        _fail(
            "BACKEND_UNAVAILABLE",
            "The requested retarget backend is unavailable in this environment.",
            details={"backend": backend_id},
        )
    return backend


def _strict_int(
    value: Any,
    *,
    name: str,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        range_text = (
            f"between {minimum} and {maximum}"
            if maximum is not None
            else f"greater than or equal to {minimum}"
        )
        _fail(
            "INVALID_PARAMETER",
            f"{name} must be an integer {range_text}.",
            details={"parameter": name, "maximum": maximum},
        )
    return value


def _strict_float(
    value: Any,
    *,
    name: str,
    minimum_exclusive: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(
            "INVALID_PARAMETER",
            f"{name} must be a finite number in the supported range.",
            details={"parameter": name, "maximum": maximum},
        )
    normalized = float(value)
    if (
        not math.isfinite(normalized)
        or normalized <= minimum_exclusive
        or (maximum is not None and normalized > maximum)
    ):
        _fail(
            "INVALID_PARAMETER",
            f"{name} must be a finite number in the supported range.",
            details={"parameter": name, "maximum": maximum},
        )
    return normalized


def _positive_capability_limit(
    backend: BackendCapability,
    name: str,
    fallback: int | float,
    *,
    integer: bool = False,
) -> int | float:
    """Read a trustworthy positive numeric limit from a backend snapshot."""

    value = backend.limits.get(name, fallback)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    if integer and not isinstance(value, int):
        return fallback
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        return fallback
    return value


def _normalize_parameters(
    request: RetargetPreflightRequest,
    inspection: AssetInspection,
    *,
    backend: BackendCapability,
    reference: str,
    profile_source: str,
    default_human_height: float,
) -> dict[str, Any]:
    raw = dict(request.parameters)
    unknown = sorted(set(raw).difference(_PARAMETERS))
    if unknown:
        _fail(
            "INVALID_PARAMETER",
            "The request contains unsupported retarget parameters.",
            details={"parameters": unknown},
        )

    run_mode = raw.get("run_mode", "smoke")
    if not isinstance(run_mode, str) or run_mode not in {"smoke", "full"}:
        _fail(
            "INVALID_PARAMETER",
            "run_mode must be either smoke or full.",
            details={"parameter": "run_mode"},
        )

    supplied_limit = raw.get("limit_frames")
    if run_mode == "full":
        if supplied_limit is not None:
            _fail(
                "INVALID_PARAMETER",
                "limit_frames cannot be set when run_mode is full.",
                details={"parameter": "limit_frames"},
            )
        limit_frames: int | None = None
    else:
        requested_limit = (
            30 if supplied_limit is None else _strict_int(supplied_limit, name="limit_frames")
        )
        limit_frames = (
            min(requested_limit, inspection.frame_count)
            if inspection.frame_count is not None and inspection.frame_count > 0
            else requested_limit
        )

    if backend.backend_id == "newton":
        maximum_iterations = int(
            _positive_capability_limit(
                backend,
                "max_ik_iterations",
                _DEFAULT_MAX_IK_ITERATIONS,
                integer=True,
            )
        )
        ik_iterations = _strict_int(
            raw.get("ik_iterations", 24),
            name="ik_iterations",
            maximum=maximum_iterations,
        )
    else:
        if "ik_iterations" in raw:
            _fail(
                "INVALID_PARAMETER",
                "ik_iterations is only supported by the Newton backend.",
                details={"parameter": "ik_iterations", "backend": backend.backend_id},
            )
        ik_iterations = None

    maximum_human_height = float(
        _positive_capability_limit(
            backend,
            "max_human_height",
            _DEFAULT_MAX_HUMAN_HEIGHT,
        )
    )
    human_height = _strict_float(
        raw.get("human_height", default_human_height),
        name="human_height",
        minimum_exclusive=0.1,
        maximum=maximum_human_height,
    )
    source_fps = inspection.frame_rate_hz
    requested_fps = raw.get("retarget_fps")
    if requested_fps is not None:
        requested_fps = _strict_float(
            requested_fps,
            name="retarget_fps",
            minimum_exclusive=0.0,
        )
    no_resample = requested_fps is None or (
        source_fps is not None and abs(requested_fps - float(source_fps)) < 1e-6
    )
    if requested_fps is not None and not no_resample:
        maximum_retarget_fps = float(
            _positive_capability_limit(
                backend,
                "max_retarget_fps",
                _DEFAULT_MAX_RETARGET_FPS,
            )
        )
        requested_fps = _strict_float(
            requested_fps,
            name="retarget_fps",
            minimum_exclusive=0.0,
            maximum=maximum_retarget_fps,
        )
        if (
            source_fps is not None
            and source_fps > 0
            and inspection.frame_count is not None
            and inspection.frame_count > 1
        ):
            predicted_frames = (
                math.floor((inspection.frame_count - 1) / float(source_fps) * requested_fps) + 1
            )
            maximum_frames = int(
                _positive_capability_limit(
                    backend,
                    "max_retarget_frames",
                    _DEFAULT_MAX_RETARGET_FRAMES,
                    integer=True,
                )
            )
            if predicted_frames > maximum_frames:
                _fail(
                    "INVALID_PARAMETER",
                    "retarget_fps would create more frames than this backend allows.",
                    details={
                        "parameter": "retarget_fps",
                        "maximum_frames": maximum_frames,
                    },
                )
    # The runtime returns the source FPS when no resampling was requested and
    # also when the requested rate is effectively equal to it.  Canonicalize
    # both forms so semantically identical requests share one plan id.
    retarget_fps: float | None
    if source_fps is not None and source_fps > 0:
        retarget_fps = float(source_fps)
        if requested_fps is not None and not no_resample:
            retarget_fps = requested_fps
    else:
        retarget_fps = requested_fps
    foot_clamp = raw.get("foot_clamp_anti_penetration", False)
    if not isinstance(foot_clamp, bool):
        _fail(
            "INVALID_PARAMETER",
            "foot_clamp_anti_penetration must be a boolean.",
            details={"parameter": "foot_clamp_anti_penetration"},
        )

    normalized: dict[str, Any] = {
        "run_mode": run_mode,
        "limit_frames": limit_frames,
        "human_height": human_height,
        "retarget_fps": retarget_fps,
        "foot_clamp_anti_penetration": foot_clamp,
        "reference": reference,
        "retarget_profile": profile_source,
    }
    if ik_iterations is not None:
        normalized["ik_iterations"] = ik_iterations
    return normalized


def _output_format(
    request: RetargetPreflightRequest,
    backend: BackendCapability,
    capabilities: CapabilityResponse,
) -> str:
    output_format = request.output_format.casefold()
    advertised = set(backend.output_formats).intersection(capabilities.supported_output_formats)
    if output_format not in _SUPPORTED_OUTPUT_FORMATS or output_format not in advertised:
        _fail(
            "INVALID_PARAMETER",
            "The requested output format is not supported by the selected backend.",
            details={"output_format": output_format, "backend": backend.backend_id},
        )
    return output_format


def _inspect_motion(
    asset_service: AgentAssetService,
    asset_id: str,
) -> tuple[AssetBundle, AssetInspection]:
    try:
        bundle = asset_service.get(asset_id)
        if bundle.kind is not AssetKind.MOTION_BUNDLE:
            _fail(
                "ASSET_KIND_MISMATCH",
                "motion_asset_id must refer to a registered motion bundle.",
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
        _raise_inspection_error(
            inspection,
            fallback_code="MOTION_PARSE_FAILED",
            fallback_message="The registered motion bundle did not pass content inspection.",
        )
    if not bool(inspection.metadata.get("content_parsed", False)):
        validation_code = str(
            inspection.metadata.get(
                "content_validation_code",
                "CONTENT_REQUIRES_ISOLATED_VALIDATION",
            )
        )
        _fail(
            validation_code,
            "The motion format requires an isolated content validator before execution.",
            details={"source_format": inspection.source_format or "unknown"},
        )
    if inspection.frame_count is None or inspection.frame_count <= 0:
        _fail(
            "MOTION_PARSE_FAILED",
            "The motion inspection did not report a positive frame count.",
        )
    if not inspection.reference_model:
        _fail(
            "REFERENCE_UNDETERMINED",
            "The motion reference model could not be determined safely.",
        )
    return bundle, inspection


def _manifest_hashes(bundle: AssetBundle, *, role: str | None = None) -> set[str]:
    return {item.sha256 for item in bundle.files if role is None or item.role.value == role}


def _contained_file(path: Path, root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("file is outside its robot preset boundary") from error
    if not resolved.is_file():
        raise ValueError("expected a regular file")
    return resolved


def _robot_joint_facts(
    urdf_path: Path,
) -> tuple[set[str], dict[str, tuple[float | None, float | None]], set[str]]:
    try:
        root = ElementTree.parse(urdf_path).getroot()
    except (ElementTree.ParseError, OSError) as error:
        raise ValueError("robot URDF is not parseable") from error
    links = {str(link.get("name")) for link in root.findall("link") if link.get("name")}
    actuated: set[str] = set()
    limits: dict[str, tuple[float | None, float | None]] = {}
    for joint in root.findall("joint"):
        name = (joint.get("name") or "").strip()
        joint_type = (joint.get("type") or "").strip().casefold()
        if not name or joint_type not in _ACTUATED_JOINT_TYPES:
            continue
        actuated.add(name)
        lower: float | None = None
        upper: float | None = None
        limit = joint.find("limit")
        if joint_type != "continuous" and limit is not None:
            try:
                lower_value = limit.get("lower")
                upper_value = limit.get("upper")
                lower = float(lower_value) if lower_value else None
                upper = float(upper_value) if upper_value else None
            except (TypeError, ValueError) as error:
                raise ValueError("robot joint limits are malformed") from error
        limits[name] = (lower, upper)
    return actuated, limits, links


def _installed_robot_preset(
    presets: Iterable[RobotPreset],
    robot_id: str,
) -> RobotPreset:
    preset = next((item for item in presets if item.name == robot_id), None)
    if preset is None:
        _fail(
            "ROBOT_NOT_FOUND",
            "The selected robot preset is not installed on this service.",
            details={"robot_id": robot_id},
        )
    return preset


def _register_asset_action(
    request: AssetRegistrationRequest,
    *,
    message: str,
) -> NextAction:
    """Map a portable registration request directly to the public MCP tool."""

    return NextAction(
        actor="agent",
        action="register_asset_bundle",
        message=message,
        parameters={"request": request.model_dump(mode="json", exclude_none=True)},
    )


def _bundle_registration_request(bundle: AssetBundle) -> AssetRegistrationRequest:
    """Reconstruct a strict registration request from one portable source."""

    source = bundle.source
    if (
        source is None
        or source.scheme is not AssetSourceScheme.MANAGED_FILE
        or source.logical_path is None
    ):
        _fail(
            "ROBOT_BUNDLE_INVALID",
            "The robot bundle has no reusable allowed-root registration source.",
        )
    try:
        return AssetRegistrationRequest(
            root_id=source.root_id,
            relative_path=source.logical_path,
            display_name=None,
            kind=bundle.kind,
            category=bundle.category,
            recursive=True,
        )
    except (TypeError, ValueError) as error:
        raise _PreflightFailureError(
            _error(
                "ROBOT_BUNDLE_INVALID",
                "The robot bundle registration source is not portable.",
            ),
            _check(
                "ROBOT_BUNDLE_INVALID",
                PreflightCheckLevel.ERROR,
                "The robot bundle registration source is not portable.",
            ),
        ) from error


def _robot_bundle_and_preset(
    asset_service: AgentAssetService,
    *,
    robot_id: str,
    robot_asset_id: str | None,
    presets: Iterable[RobotPreset],
) -> tuple[AssetBundle, RobotPreset, dict[str, tuple[float | None, float | None]]]:
    if robot_asset_id is None:
        advertised_preset = _installed_robot_preset(presets, robot_id)
        try:
            registration = asset_service.registration_hint(
                advertised_preset.root_dir,
                kind=AssetKind.ROBOT_BUNDLE,
                category=AssetCategory.ROBOT_MODEL,
            )
        except AssetServiceError as error:
            _raise_asset_error(error)
        action = _register_asset_action(
            registration,
            message="Register the installed robot directory, including YAML, URDF, and meshes.",
        )
        _fail(
            "ROBOT_ASSET_REQUIRED",
            "A content-addressed robot bundle is required for a runnable plan.",
            details={"robot_id": robot_id},
            next_action=action,
        )
    try:
        bundle = asset_service.get(robot_asset_id)
        if bundle.kind is not AssetKind.ROBOT_BUNDLE:
            _fail(
                "ASSET_KIND_MISMATCH",
                "robot_asset_id must refer to a registered robot bundle.",
                details={
                    "asset_id": robot_asset_id,
                    "kind": bundle.kind.value,
                },
            )
        inspection = asset_service.inspect(
            AssetInspectionRequest(
                asset_id=robot_asset_id,
                verify_hashes=True,
                parse_content=True,
            )
        )
    except AssetServiceError as error:
        _raise_asset_error(error)
    if inspection.status is InspectionStatus.INVALID:
        _raise_inspection_error(
            inspection,
            fallback_code="ROBOT_BUNDLE_INVALID",
            fallback_message=("The registered robot bundle did not pass structural inspection."),
        )

    advertised_preset = _installed_robot_preset(presets, robot_id)
    yaml_value = advertised_preset.meta.get("yaml_path")
    if not isinstance(yaml_value, str) or not yaml_value:
        _fail(
            "ROBOT_BUNDLE_INVALID",
            "The selected robot preset has no bound robot YAML.",
            details={"robot_id": robot_id},
        )
    try:
        yaml_path = _contained_file(Path(yaml_value), advertised_preset.root_dir)
    except (OSError, ValueError) as error:
        raise _PreflightFailureError(
            _error(
                "ROBOT_BUNDLE_INVALID",
                "The selected robot YAML is unavailable or outside the preset boundary.",
                details={"robot_id": robot_id},
            ),
            _check(
                "ROBOT_BUNDLE_INVALID",
                PreflightCheckLevel.ERROR,
                "The selected robot YAML is unavailable or outside the preset boundary.",
                details={"robot_id": robot_id},
            ),
        ) from error
    # Reload the exact manifest-bound YAML instead of trusting a mutable or
    # previously populated process cache.  Every value validated below is now
    # derived from the same bytes that participate in the robot bundle hash.
    try:
        from hhtools.robot.registry import preset_from_yaml

        preset, _yaml_digest = _parse_stable_file(
            yaml_path,
            preset_from_yaml,
            expected_digests=_manifest_hashes(bundle, role="metadata"),
        )
    except _FileIdentityMismatchError:
        _fail(
            "ROBOT_BUNDLE_MISMATCH",
            "The robot asset does not contain the YAML used by the selected preset.",
            details={"robot_id": robot_id},
        )
    except _FileChangedError:
        _fail(
            "ASSET_HASH_MISMATCH",
            "The selected robot YAML changed during preflight; register it again.",
            details={"robot_id": robot_id},
            retryable=True,
        )
    except (OSError, TypeError, ValueError):
        _fail(
            "ROBOT_BUNDLE_INVALID",
            "The manifest-bound robot YAML could not be loaded read-only.",
            details={"robot_id": robot_id},
        )
    if preset.name != robot_id:
        _fail(
            "ROBOT_BUNDLE_MISMATCH",
            "The manifest-bound robot YAML resolves to a different preset id.",
            details={"robot_id": robot_id},
        )
    if not preset.has_urdf or preset.urdf_path is None:
        _fail(
            "ROBOT_BUNDLE_INVALID",
            "The selected robot preset has no readable URDF.",
            details={"robot_id": robot_id},
        )

    primary = next(item for item in bundle.files if item.relative_path == bundle.primary_file)
    try:
        preset_urdf = _contained_file(preset.urdf_path, preset.root_dir)
    except (OSError, ValueError) as error:
        raise _PreflightFailureError(
            _error(
                "ROBOT_BUNDLE_INVALID",
                "The robot preset URDF is outside its trusted preset directory.",
                details={"robot_id": robot_id},
            ),
            _check(
                "ROBOT_BUNDLE_INVALID",
                PreflightCheckLevel.ERROR,
                "The robot preset URDF is outside its trusted preset directory.",
                details={"robot_id": robot_id},
            ),
        ) from error
    try:
        robot_facts, _preset_urdf_digest = _parse_stable_file(
            preset_urdf,
            _robot_joint_facts,
            expected_digests={primary.sha256},
        )
    except _FileIdentityMismatchError:
        _fail(
            "ROBOT_BUNDLE_MISMATCH",
            "The robot asset does not match the URDF used by the selected preset.",
            details={"robot_id": robot_id},
        )
    except _FileChangedError:
        _fail(
            "ASSET_HASH_MISMATCH",
            "The selected robot URDF changed during preflight; register it again.",
            details={"robot_id": robot_id},
            retryable=True,
        )
    except ValueError:
        _fail(
            "ROBOT_BUNDLE_INVALID",
            "The robot URDF topology or joint limits could not be validated.",
            details={"robot_id": robot_id},
        )
    actuated, limits, links = robot_facts
    if not preset.dof_order or len(set(preset.dof_order)) != len(preset.dof_order):
        _fail(
            "ROBOT_CONFIGURATION_INVALID",
            "The robot preset must declare a non-empty, unique DOF order.",
            details={"robot_id": robot_id},
        )
    invalid_dofs = sorted(set(preset.dof_order).difference(actuated))
    if invalid_dofs:
        _fail(
            "ROBOT_CONFIGURATION_INVALID",
            "The robot DOF order contains joints that are not actuated by the URDF.",
            details={"joint_names": invalid_dofs},
        )
    if not preset.ik_map:
        _fail(
            "ROBOT_CONFIGURATION_INVALID",
            "The robot preset must declare a non-empty IK mapping.",
            details={"robot_id": robot_id},
        )
    missing_links = sorted(
        {
            str(link)
            for link in preset.ik_map.values()
            if not isinstance(link, str) or not link or link not in links
        }
    )
    if missing_links:
        _fail(
            "ROBOT_CONFIGURATION_INVALID",
            "The robot IK mapping refers to links absent from the URDF.",
            details={"link_names": missing_links},
        )
    try:
        from hhtools.robot.kinematics import validate_ik_map

        issues = validate_ik_map(preset_urdf, dict(preset.ik_map))
    except (ElementTree.ParseError, OSError, RuntimeError, ValueError):
        _fail(
            "ROBOT_CONFIGURATION_INVALID",
            "The robot IK mapping could not be checked against URDF topology.",
        )
    if issues:
        _fail(
            "ROBOT_CONFIGURATION_INVALID",
            "The robot IK mapping is inconsistent with the URDF topology.",
            details={
                "slots": sorted({issue.slot for issue in issues}),
                "issue_count": len(issues),
            },
        )
    return bundle, preset, limits


def _validate_finite_document(value: Any) -> bool:
    if value is None or isinstance(value, bool | int | str):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _validate_finite_document(item) for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return all(_validate_finite_document(item) for item in value)
    return False


def _validate_scaler_semantics(scaler: Any, preset: RobotPreset) -> None:
    """Mirror cheap Scaler runtime preconditions without building a scaler."""

    joint_scales = getattr(scaler, "joint_scales", None)
    if not isinstance(joint_scales, dict) or not joint_scales:
        raise ValueError("scaler joint_scales is empty")
    if any(
        not isinstance(name, str)
        or not name
        or isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for name, value in joint_scales.items()
    ):
        raise ValueError("scaler joint scales must be finite and positive")
    root_joint = getattr(scaler, "root_joint", None)
    if not isinstance(root_joint, str) or root_joint not in joint_scales:
        raise ValueError("scaler root joint is not mapped")
    if getattr(scaler, "up_axis", None) not in {"X", "Y", "Z"}:
        raise ValueError("scaler up axis is unsupported")
    if getattr(scaler, "scale_mode", None) not in {"uniform", "height"}:
        raise ValueError("scaler scale mode is unsupported")
    if getattr(scaler, "scale_anchor", None) not in {"origin", "root"}:
        raise ValueError("scaler scale anchor is unsupported")
    missing_ik_slots = set(preset.ik_map).difference(joint_scales)
    if missing_ik_slots:
        raise ValueError("scaler does not cover the robot IK mapping")
    offsets = getattr(scaler, "joint_offsets", {})
    if not isinstance(offsets, dict) or set(offsets).difference(joint_scales):
        raise ValueError("scaler offsets refer to unmapped joints")
    for _translation, quaternion in offsets.values():
        if sum(float(component) ** 2 for component in quaternion) <= 1e-12:
            raise ValueError("scaler offset quaternion has zero norm")
    source_body_quat = getattr(scaler, "source_body_quat", ())
    if (
        len(source_body_quat) != 4
        or sum(float(component) ** 2 for component in source_body_quat) <= 1e-12
    ):
        raise ValueError("scaler source-body quaternion has zero norm")
    trajectory_scale = getattr(scaler, "root_trajectory_scale", None)
    if trajectory_scale is not None and float(trajectory_scale) <= 0.0:
        raise ValueError("scaler root trajectory scale is not positive")


def _calibration_action(
    robot_id: str,
    robot_asset_id: str,
    reference: str,
    *,
    motion_asset_id: str | None = None,
) -> NextAction:
    request = CalibrationStatusRequest(
        robot_id=robot_id,
        robot_asset_id=robot_asset_id,
        reference=reference,
        motion_asset_id=motion_asset_id,
    )
    return NextAction(
        actor="agent",
        action="get_calibration_status",
        message="Inspect this robot/reference and generate a validated calibration candidate.",
        parameters={"request": request.model_dump(mode="json", exclude_none=True)},
    )


def _manual_calibration(
    preset: RobotPreset,
    reference: str,
    limits: Mapping[str, tuple[float | None, float | None]],
    robot_bundle: AssetBundle,
    *,
    user_root: Path | None = None,
) -> tuple[Path, str, str, str] | None:
    from hhtools.retarget.calibration import (
        load_calibration,
        normalize_calibration_reference,
        resolve_preset_calibration_file,
    )

    try:
        managed_user_root = (user_root or user_robot_dir()).resolve(strict=True)
        path = resolve_preset_calibration_file(
            preset,
            reference,
            user_root=managed_user_root,
        )
    except (OSError, RuntimeError, TypeError, ValueError, YAMLError):
        _fail(
            "CALIBRATION_MISMATCH",
            "The matching robot calibration exists but is malformed.",
            details={"robot_id": preset.name, "reference": reference},
        )
    if path is None:
        return None
    try:
        try:
            contained = _contained_file(path, managed_user_root)
            storage = "user_calibration"
        except ValueError:
            contained = _contained_file(path, preset.root_dir)
            storage = "robot_bundle"
        calibration, digest = _parse_stable_file(contained, load_calibration)
    except _FileChangedError:
        _fail(
            "CALIBRATION_MISMATCH",
            "The matching robot calibration changed during preflight; retry it.",
            details={"robot_id": preset.name, "reference": reference},
            retryable=True,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, YAMLError):
        _fail(
            "CALIBRATION_MISMATCH",
            "The matching robot calibration exists but is malformed.",
            details={"robot_id": preset.name, "reference": reference},
        )
    try:
        calibration_reference = normalize_calibration_reference(calibration.reference)
    except (TypeError, ValueError):
        _fail(
            "CALIBRATION_MISMATCH",
            "The matching robot calibration declares an unsupported reference.",
            details={"robot_id": preset.name, "reference": reference},
        )
    if calibration.robot != preset.name:
        _fail(
            "CALIBRATION_MISMATCH",
            "The calibration must name this exact robot preset.",
            details={"robot_id": preset.name, "reference": reference},
        )
    if calibration_reference != reference:
        _fail(
            "CALIBRATION_MISMATCH",
            "The calibration belongs to a different motion reference.",
            details={"robot_id": preset.name, "reference": reference},
        )
    unknown = sorted(set(calibration.calibrated_joint_q).difference(preset.dof_order))
    if unknown:
        _fail(
            "CALIBRATION_MISMATCH",
            "The calibration contains joints outside the robot DOF order.",
            details={"joint_names": unknown},
        )
    for name, value in calibration.calibrated_joint_q.items():
        if not math.isfinite(value):
            _fail(
                "CALIBRATION_MISMATCH",
                "The calibration contains a non-finite joint value.",
                details={"joint_name": name},
            )
        lower, upper = limits.get(name, (None, None))
        if (lower is not None and value < lower) or (upper is not None and value > upper):
            _fail(
                "CALIBRATION_MISMATCH",
                "The calibration contains a joint value outside its URDF limit.",
                details={"joint_name": name},
            )
    if storage == "robot_bundle" and digest not in _manifest_hashes(
        robot_bundle,
        role="metadata",
    ):
        action = _register_asset_action(
            _bundle_registration_request(robot_bundle),
            message="Register the robot bundle again so the calibration is content-bound.",
        )
        _fail(
            "ROBOT_BUNDLE_MISMATCH",
            "The matching calibration is not bound into the registered robot bundle.",
            details={"robot_id": preset.name, "reference": reference},
            next_action=action,
        )
    return contained, digest, f"cal:sha256:{digest}", storage


def _bundled_scaler(
    preset: RobotPreset,
    reference: str,
    robot_bundle: AssetBundle,
) -> tuple[Path, str, float] | None:
    from hhtools.retarget.newton_basic.config import load_scaler_config
    from hhtools.robot.retarget_profile import bundled_scaler_path

    try:
        candidate = bundled_scaler_path(preset, reference)
    except (OSError, TypeError, ValueError):
        candidate = None
    if candidate is None:
        return None
    try:
        contained = _contained_file(candidate, preset.root_dir)
        scaler, digest = _parse_stable_file(
            contained,
            load_scaler_config,
            expected_digests=_manifest_hashes(robot_bundle, role="metadata"),
        )
        document = (
            asdict(scaler)
            if is_dataclass(scaler) and not isinstance(scaler, type)
            else vars(scaler)
        )
        if not _validate_finite_document(document):
            raise ValueError("scaler contains non-finite values")
        if float(scaler.human_height_assumption) <= 0.1 or float(scaler.model_height) <= 0.1:
            raise ValueError("scaler heights are invalid")
        _validate_scaler_semantics(scaler, preset)
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        _fail(
            "CALIBRATION_MISMATCH",
            "The bundled robot scaler is missing, unbound, or malformed.",
            details={"robot_id": preset.name, "reference": reference},
        )
    return contained, digest, float(scaler.human_height_assumption)


def _retarget_profile(
    request: RetargetPreflightRequest,
    *,
    backend: str,
    preset: RobotPreset,
    reference: str,
    limits: Mapping[str, tuple[float | None, float | None]],
    robot_bundle: AssetBundle,
) -> tuple[str, str, str | None, float, str, str]:
    manual = _manual_calibration(preset, reference, limits, robot_bundle)
    scaler = None
    if backend == "newton" and manual is None:
        scaler = _bundled_scaler(preset, reference, robot_bundle)
    if manual is None and scaler is None:
        if request.calibration_id is not None:
            _fail(
                "CALIBRATION_MISMATCH",
                "No installed robot calibration matches the requested id.",
                details={"robot_id": preset.name, "reference": reference},
            )
        action = _calibration_action(
            preset.name,
            robot_bundle.asset_id,
            reference,
            motion_asset_id=(request.motion_asset_id if reference == "glb" else None),
        )
        raise _PreflightFailureError(
            _error(
                "CALIBRATION_REQUIRED",
                "A matching validated robot calibration is required.",
                details={"robot_id": preset.name, "reference": reference},
                next_action=action,
            ),
            PreflightCheck(
                code="CALIBRATION_REQUIRED",
                level=PreflightCheckLevel.ERROR,
                message="A matching validated robot calibration is required.",
                details={"robot_id": preset.name, "reference": reference},
                next_action=action,
            ),
        )
    if manual is not None:
        profile_path, digest, calibration_id, storage = manual
        source = "calibration"
        from hhtools.robot.retarget_profile import default_human_height

        human_height_default = default_human_height(preset, reference)
    else:
        assert scaler is not None
        profile_path, digest, human_height_default = scaler
        calibration_id = None
        source = "bundled_scaler"
        storage = "robot_bundle"
    if request.calibration_id is not None and request.calibration_id != calibration_id:
        _fail(
            "CALIBRATION_MISMATCH",
            "The requested calibration id does not match the selected robot profile.",
            details={
                "robot_id": preset.name,
                "reference": reference,
                "expected_calibration_id": calibration_id,
            },
        )
    try:
        profile_root = (
            user_robot_dir().resolve(strict=True)
            if storage == "user_calibration"
            else preset.root_dir.resolve(strict=True)
        )
        relative_path = profile_path.relative_to(profile_root).as_posix()
    except (OSError, ValueError):
        _fail(
            "ROBOT_BUNDLE_MISMATCH",
            "The selected retarget profile is outside its managed storage boundary.",
            details={
                "robot_id": preset.name,
                "reference": reference,
                "storage": storage,
            },
        )
    return source, digest, calibration_id, human_height_default, relative_path, storage


def _scheduler_check(scheduler: SchedulerCapability) -> PreflightCheck:
    details = {
        "max_running_jobs": scheduler.max_running_jobs,
        "max_queued_jobs": scheduler.max_queued_jobs,
        "running": scheduler.running,
        "queued": scheduler.queued,
        "reserved": scheduler.reserved,
        "mode": scheduler.mode.value,
    }
    if scheduler.closed:
        _fail(
            "SCHEDULER_CLOSED",
            "The job scheduler is shutting down and cannot accept new work.",
            details=details,
            retryable=True,
        )
    if scheduler.max_running_jobs == 0:
        return _check(
            "JOB_ADMISSION",
            PreflightCheckLevel.WARNING,
            "Job concurrency is configured as unlimited.",
            details=details,
        )
    if scheduler.max_queued_jobs > 0:
        capacity = scheduler.max_running_jobs + scheduler.max_queued_jobs
        occupied = scheduler.running + scheduler.queued + scheduler.reserved
        if occupied >= capacity:
            return _check(
                "JOB_ADMISSION",
                PreflightCheckLevel.WARNING,
                "The bounded queue is currently full; start may need to be retried.",
                details=details,
            )
    return _check(
        "JOB_ADMISSION",
        PreflightCheckLevel.PASS,
        "The scheduler policy can admit work when the plan is started.",
        details=details,
    )


class PreflightService:
    """Resolve Agent retarget intent into a content-bound immutable plan."""

    def __init__(
        self,
        asset_service: AgentAssetService,
        plan_store: PlanStore,
        *,
        capabilities_provider: Callable[[], CapabilityResponse],
        robot_provider: Callable[[], Iterable[RobotPreset]] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        request_id_provider: Callable[[], str] = lambda: f"req_{uuid.uuid4().hex}",
    ) -> None:
        if robot_provider is None:
            from hhtools.robot.registry import list_presets_readonly

            robot_provider = list_presets_readonly
        self._asset_service = asset_service
        self._plan_store = plan_store
        self._calibration_validations = CalibrationValidationStore(plan_store.database_path.parent)
        self._capabilities_provider = capabilities_provider
        self._robot_provider = robot_provider
        self._clock = clock
        self._request_id_provider = request_id_provider

    def preflight_retarget(self, request: RetargetPreflightRequest) -> PreflightResponse:
        """Validate one request without starting a solver or reserving admission."""

        request_id = self._request_id_provider()
        checks: list[PreflightCheck] = []
        recommended_backend: str | None = None
        try:
            if request.output_policy is not OutputPolicy.CREATE_NEW:
                _fail(
                    "UNSUPPORTED_OUTPUT_POLICY",
                    "Managed Agent artifacts currently support only create_new output policy.",
                    details={"output_policy": request.output_policy.value},
                )
            if _PORTABLE_ROBOT_ID.fullmatch(request.robot_id) is None:
                _fail(
                    "INVALID_PARAMETER",
                    "robot_id must be a portable identifier.",
                    details={"parameter": "robot_id"},
                )
            motion_bundle, motion_inspection = _inspect_motion(
                self._asset_service,
                request.motion_asset_id,
            )
            checks.append(
                _check(
                    "INPUT_PARSEABLE",
                    (
                        PreflightCheckLevel.WARNING
                        if motion_inspection.warnings
                        else PreflightCheckLevel.PASS
                    ),
                    "The motion bundle passed safe content inspection.",
                    details={
                        "frame_count": motion_inspection.frame_count,
                        "source_format": motion_inspection.source_format,
                        "warning_count": len(motion_inspection.warnings),
                    },
                )
            )
            recommended_backend = _backend_for_category(motion_inspection.category)
            requested_backend = request.backend or recommended_backend
            if requested_backend != recommended_backend:
                _fail(
                    "BACKEND_INCOMPATIBLE",
                    "The requested backend does not support this motion category.",
                    details={
                        "backend": requested_backend,
                        "category": motion_inspection.category.value,
                        "recommended_backend": recommended_backend,
                    },
                )

            capabilities = self._capabilities_provider()
            backend = _backend_capability(capabilities, requested_backend)
            if motion_inspection.category not in backend.supported_categories:
                _fail(
                    "BACKEND_INCOMPATIBLE",
                    "The backend capability does not advertise this input category.",
                    details={
                        "backend": backend.backend_id,
                        "category": motion_inspection.category.value,
                    },
                )
            checks.append(
                _check(
                    "BACKEND_READY",
                    PreflightCheckLevel.PASS,
                    "The selected backend is installed and compatible with the input.",
                    details={"backend": backend.backend_id},
                )
            )

            robot_bundle, preset, joint_limits = _robot_bundle_and_preset(
                self._asset_service,
                robot_id=request.robot_id,
                robot_asset_id=request.robot_asset_id,
                presets=self._robot_provider(),
            )
            checks.append(
                _check(
                    "ROBOT_BUNDLE_READY",
                    PreflightCheckLevel.PASS,
                    "The robot bundle, preset, DOF order, and IK map agree.",
                    details={
                        "robot_id": preset.name,
                        "dof_count": len(preset.dof_order),
                    },
                )
            )

            from hhtools.retarget.calibration import normalize_calibration_reference

            reference = normalize_calibration_reference(str(motion_inspection.reference_model))
            (
                profile_source,
                profile_digest,
                calibration_id,
                human_height_default,
                profile_relative_path,
                profile_storage,
            ) = _retarget_profile(
                request,
                backend=backend.backend_id,
                preset=preset,
                reference=reference,
                limits=joint_limits,
                robot_bundle=robot_bundle,
            )
            if profile_source == "calibration":
                validation = self._calibration_validations.get(
                    calibration_validation_identity(
                        robot_id=preset.name,
                        robot_asset_id=robot_bundle.asset_id,
                        reference=reference,
                        calibration_digest=profile_digest,
                        motion_asset_id=request.motion_asset_id,
                    )
                )
                if validation is None or not validation.valid:
                    action = _calibration_action(
                        preset.name,
                        robot_bundle.asset_id,
                        reference,
                        motion_asset_id=request.motion_asset_id if reference == "glb" else None,
                    )
                    _fail(
                        "CALIBRATION_VALIDATION_REQUIRED"
                        if validation is None
                        else "CALIBRATION_VALIDATION_FAILED",
                        "Validate the current calibration before preflight."
                        if validation is None
                        else "The current calibration failed validation.",
                        details={
                            "robot_id": preset.name,
                            "reference": reference,
                            "calibration_digest": profile_digest,
                        },
                        next_action=action,
                    )
                checks.append(
                    _check(
                        "CALIBRATION_VALIDATED",
                        PreflightCheckLevel.PASS,
                        "The current calibration has matching valid geometric evidence.",
                        details={
                            "calibration_digest": profile_digest,
                            "validation_version": CALIBRATION_VALIDATION_VERSION,
                        },
                    )
                )
            checks.append(
                _check(
                    "CALIBRATION_MATCH",
                    PreflightCheckLevel.PASS,
                    "The selected robot profile matches the motion reference.",
                    details={
                        "robot_id": preset.name,
                        "reference": reference,
                        "profile_source": profile_source,
                        "profile_storage": profile_storage,
                    },
                )
            )

            output_format = _output_format(request, backend, capabilities)
            if (
                output_format == "pkl"
                and motion_inspection.category is AssetCategory.OBJECT_INTERACTION
            ):
                _fail(
                    "UNSUPPORTED_PORTABLE_EXPORT",
                    "Object-interaction PKL export can expose host mesh paths.",
                    details={
                        "category": motion_inspection.category.value,
                        "output_format": output_format,
                    },
                )
            parameters = _normalize_parameters(
                request,
                motion_inspection,
                backend=backend,
                reference=reference,
                profile_source=profile_source,
                default_human_height=human_height_default,
            )
            checks.append(
                _check(
                    "PARAMETERS_VALID",
                    PreflightCheckLevel.PASS,
                    "Retarget parameters and output policy were normalized successfully.",
                    details={
                        "run_mode": parameters["run_mode"],
                        "output_format": output_format,
                    },
                )
            )
            checks.append(_scheduler_check(capabilities.scheduler))

            canonical_payload = {
                "semantics": _PLAN_SEMANTICS,
                "motion": {
                    "asset_id": motion_bundle.asset_id,
                    "digest": _asset_digest(motion_bundle.asset_id),
                    "category": motion_inspection.category.value,
                    "dataset": motion_inspection.dataset,
                    "reference": reference,
                },
                "robot": {
                    "asset_id": robot_bundle.asset_id,
                    "digest": _asset_digest(robot_bundle.asset_id),
                    "robot_id": preset.name,
                },
                "backend": backend.backend_id,
                "retarget_profile": {
                    "source": profile_source,
                    "storage": profile_storage,
                    "calibration_id": calibration_id,
                    "digest": profile_digest,
                    "relative_path": profile_relative_path,
                    "validation_version": (
                        CALIBRATION_VALIDATION_VERSION if profile_source == "calibration" else None
                    ),
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
                candidate = RetargetPlan(
                    plan_id=plan_id,
                    created_at=self._clock(),
                    motion_asset_id=motion_bundle.asset_id,
                    robot_id=preset.name,
                    robot_asset_id=robot_bundle.asset_id,
                    backend=backend.backend_id,
                    calibration_id=calibration_id,
                    output_format=output_format,
                    output_policy=request.output_policy,
                    parameters=parameters,
                    input_digest=_asset_digest(motion_bundle.asset_id),
                    robot_digest=_asset_digest(robot_bundle.asset_id),
                    calibration_digest=(
                        profile_digest if profile_source == "calibration" else None
                    ),
                )
                try:
                    plan = self._plan_store.put_if_absent(
                        candidate,
                        canonical_payload,
                    )
                except PlanStoreError as conflict:
                    # A concurrent identical preflight may have won between
                    # get() and put_if_absent().  Return that immutable plan;
                    # any other store error remains a real failure.
                    if conflict.code != "PLAN_CONFLICT":
                        raise
                    plan = self._plan_store.get(plan_id)
                    if self._plan_store.get_payload(plan_id) != canonical_payload:
                        raise
            return PreflightResponse(
                request_id=request_id,
                status=PreflightStatus.READY,
                plan=plan,
                checks=checks,
                recommended_backend=recommended_backend,
            )
        except _PreflightFailureError as failure:
            checks.append(failure.check)
            if failure.error.code in {
                "CALIBRATION_REQUIRED",
                "CALIBRATION_VALIDATION_REQUIRED",
                "CALIBRATION_VALIDATION_FAILED",
            }:
                assert failure.error.next_action is not None
                return PreflightResponse(
                    request_id=request_id,
                    status=PreflightStatus.HUMAN_ACTION_REQUIRED,
                    checks=checks,
                    recommended_backend=recommended_backend,
                    required_actions=[failure.error.next_action],
                )
            return PreflightResponse(
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
            return PreflightResponse(
                request_id=request_id,
                status=PreflightStatus.REJECTED,
                checks=checks,
                recommended_backend=recommended_backend,
                error=public,
            )


__all__ = ["PreflightService"]
