"""Read-only capability discovery for humans, automation, and AI agents.

Capability discovery must stay cheap and side-effect free: in particular it
does not import either retarget pipeline or initialise Warp.  Optional package
availability is therefore probed with :mod:`importlib`, while device discovery
uses Torch only when it is already installable in the current environment.
"""

from __future__ import annotations

import importlib.util
import platform
from collections.abc import Callable, Iterable
from importlib import metadata
from typing import TYPE_CHECKING, Any

from hhtools._version import __version__
from hhtools.contracts import (
    AssetCategory,
    BackendCapability,
    CapabilityResponse,
    DeviceCapability,
    RobotCapability,
    SchedulerCapability,
    SchedulerMode,
)

from .batch_limits import BatchLimitSnapshot

if TYPE_CHECKING:
    from hhtools.robot.base import RobotPreset


_INPUT_FORMATS = (
    "bvh",
    "csv",
    "glb",
    "gltf",
    "npy",
    "npz",
    "pickle",
    "pkl",
    "pt",
    "pth",
)
_OUTPUT_FORMATS = ("csv", "pkl")
_CALIBRATION_REFERENCES = (
    "smpl",
    "smplx",
    "gvhmr",
    "soma_bvh",
    "lafan_bvh",
    "mocap_bvh",
    "xsens_mocap",
    "glb",
)

# Distribution names are not always the same as import names.  The first
# installed distribution in each tuple supplies the optional version string.
_DISTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "mujoco": ("mujoco",),
    "newton": ("newton", "newton-python"),
    "osqp": ("osqp",),
    "torch": ("torch",),
    "warp": ("warp-lang", "warp"),
}


def _module_available(module_name: str) -> bool:
    """Return whether an optional module is importable without importing it."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _module_version(module_name: str) -> str | None:
    for distribution in _DISTRIBUTIONS.get(module_name, (module_name,)):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return None


def _cpu_name() -> str:
    return platform.processor().strip() or platform.machine().strip() or "CPU"


def _detect_devices() -> list[DeviceCapability]:
    """Return compact CPU/CUDA/MPS facts without importing a retarget backend."""

    devices = [
        DeviceCapability(
            device_id="cpu",
            kind="cpu",
            display_name=_cpu_name(),
            available=True,
            metadata={"platform": platform.system().lower()},
        )
    ]
    try:
        import torch
    except (ImportError, OSError, RuntimeError):
        return devices

    torch_version = str(getattr(torch, "__version__", "unknown"))
    torch_runtime = getattr(getattr(torch, "version", None), "cuda", None)
    cuda = getattr(torch, "cuda", None)
    try:
        cuda_available = bool(cuda is not None and cuda.is_available())
    except (AttributeError, RuntimeError):
        cuda_available = False

    if cuda_available and cuda is not None:
        try:
            count = int(cuda.device_count())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            count = 0
        for index in range(count):
            try:
                props = cuda.get_device_properties(index)
                name = str(getattr(props, "name", f"CUDA device {index}"))
                total_memory = int(getattr(props, "total_memory", 0)) or None
                major = getattr(props, "major", None)
                minor = getattr(props, "minor", None)
                compute_capability = (
                    f"{int(major)}.{int(minor)}"
                    if major is not None and minor is not None
                    else None
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                name = f"CUDA device {index}"
                total_memory = None
                compute_capability = None

            free_memory: int | None = None
            try:
                free_memory = int(cuda.mem_get_info(index)[0])
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

            devices.append(
                DeviceCapability(
                    device_id=f"cuda:{index}",
                    kind="cuda",
                    display_name=name,
                    available=True,
                    total_memory_bytes=total_memory,
                    free_memory_bytes=free_memory,
                    compute_capability=compute_capability,
                    metadata={
                        "torch": torch_version,
                        "cuda_runtime": str(torch_runtime) if torch_runtime else None,
                    },
                )
            )

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    try:
        mps_available = bool(mps is not None and mps.is_available())
    except (AttributeError, RuntimeError):
        mps_available = False
    if mps_available:
        devices.append(
            DeviceCapability(
                device_id="mps",
                kind="mps",
                display_name="Apple Metal Performance Shaders",
                available=True,
                metadata={"torch": torch_version},
            )
        )
    return devices


def _scheduler_capability(snapshot: object | None) -> SchedulerCapability:
    """Normalize a Web scheduler snapshot without depending on its class."""

    def value(name: str, default: int | bool = 0) -> Any:
        if snapshot is None:
            return default
        if isinstance(snapshot, dict):
            return snapshot.get(name, default)
        return getattr(snapshot, name, default)

    max_running = int(value("max_running_jobs"))
    max_queued = int(value("max_queued_jobs"))
    # JobScheduler bypasses both running and queue admission checks whenever
    # max_running_jobs is zero.  Preserve max_queued as configured metadata,
    # but describe the effective policy rather than implying it is enforced.
    if max_running == 0:
        mode = SchedulerMode.UNLIMITED
    elif max_running > 0 and max_queued > 0:
        mode = SchedulerMode.LIMITED
    else:
        mode = SchedulerMode.MIXED
    return SchedulerCapability(
        max_running_jobs=max_running,
        max_queued_jobs=max_queued,
        running=int(value("running_jobs")),
        queued=int(value("queued_jobs")),
        reserved=int(value("reserved_jobs")),
        mode=mode,
        closed=bool(value("closed", False)),
    )


def _reference_readiness(preset: RobotPreset) -> tuple[list[str], list[str]]:
    """Return independently validated calibration and scaler references.

    A bundled Newton scaler is not equivalent to a validated robot pose
    calibration: notably, Interaction-Mesh still requires the latter.  Keep
    both facts separate so clients can make backend-specific decisions.
    """

    from hhtools.retarget.calibration import (
        load_calibration,
        normalize_calibration_reference,
        resolve_preset_calibration_file,
    )
    from hhtools.retarget.newton_basic.config import load_scaler_config
    from hhtools.robot.retarget_profile import bundled_scaler_path

    calibrated: list[str] = []
    scalers: list[str] = []
    known_joints = set(preset.dof_order)
    preset_root = preset.root_dir.resolve()
    for reference in _CALIBRATION_REFERENCES:
        try:
            calibration_path = resolve_preset_calibration_file(preset, reference)
            if calibration_path is not None:
                calibration = load_calibration(calibration_path)
                calibration_joints = set(calibration.calibrated_joint_q)
                robot_matches = calibration.robot == preset.name
                reference_matches = (
                    normalize_calibration_reference(str(calibration.reference)) == reference
                )
                joints_match = not calibration_joints.difference(known_joints)
                if robot_matches and reference_matches and joints_match:
                    calibrated.append(reference)
        except Exception:  # noqa: BLE001 - invalid optional metadata is "not ready"
            pass

        try:
            scaler_path = bundled_scaler_path(preset, reference)
            if scaler_path is not None:
                contained_scaler = scaler_path.resolve(strict=True)
                contained_scaler.relative_to(preset_root)
                load_scaler_config(contained_scaler)
                scalers.append(reference)
        except Exception:  # noqa: BLE001 - invalid optional metadata is "not ready"
            pass
    return calibrated, scalers


def _robot_capabilities(presets: Iterable[RobotPreset]) -> list[RobotCapability]:
    robots: list[RobotCapability] = []
    for preset in sorted(presets, key=lambda item: item.name):
        has_urdf = bool(preset.has_urdf)
        has_ik_mapping = bool(preset.ik_map)
        has_dof_order = bool(preset.dof_order)
        unavailable: list[str] = []
        if not has_urdf:
            unavailable.append("URDF is missing")
        if not has_ik_mapping:
            unavailable.append("IK mapping is missing")
        if not has_dof_order:
            unavailable.append("DOF order is missing")
        calibrated_references, scaler_references = _reference_readiness(preset)
        robots.append(
            RobotCapability(
                robot_id=preset.name,
                display_name=preset.display_name or preset.name,
                available=not unavailable,
                has_urdf=has_urdf,
                has_ik_mapping=has_ik_mapping,
                dof_count=len(preset.dof_order) if preset.dof_order else None,
                supported_references=list(_CALIBRATION_REFERENCES),
                calibrated_references=calibrated_references,
                scaler_references=scaler_references,
                unavailable_reason="; ".join(unavailable) if unavailable else None,
            )
        )
    return robots


def _backend_capabilities(
    devices: list[DeviceCapability],
    batch_limits: BatchLimitSnapshot,
) -> list[BackendCapability]:
    cuda_available = any(device.kind.value == "cuda" and device.available for device in devices)

    definitions = (
        (
            "newton",
            "Newton IK",
            # The solver core is Newton + Warp.  The current end-to-end robot
            # adapter imports yourdfpy and MuJoCo before constructing it, so
            # capability discovery must include those real execution-path deps.
            ("newton", "warp", "mujoco", "yourdfpy"),
            [AssetCategory.PLAIN_MOTION, AssetCategory.ROBOT_TRAJECTORY],
            {
                "batch": True,
                "scene_geometry": False,
                "r2r": True,
                "cuda_graph": cuda_available,
                "cpu_fallback": True,
            },
            {
                "requires_cuda": False,
                "recommended_linux_cuda": True,
                # These are admission-safety ceilings, not performance
                # recommendations.  Expert callers may still choose any
                # value below them, while accidental pathological values are
                # rejected before a solver or resampler is constructed.
                "max_ik_iterations": 200,
                "max_retarget_fps": 1_000.0,
                "max_retarget_frames": 100_000,
                "max_human_height": 10.0,
                "max_batch_items": batch_limits.max_batch_items,
                "max_batch_total_frames": batch_limits.max_batch_total_frames,
            },
        ),
        (
            "interaction_mesh",
            "Interaction-Mesh MPC",
            ("mujoco", "osqp", "scipy", "yourdfpy"),
            [AssetCategory.OBJECT_INTERACTION, AssetCategory.TERRAIN_SCENE],
            {
                "batch": True,
                "scene_geometry": True,
                "mpc": True,
                "cpu_fallback": True,
            },
            {
                "requires_cuda": False,
                "max_retarget_fps": 1_000.0,
                "max_retarget_frames": 100_000,
                "max_human_height": 10.0,
                "max_batch_items": batch_limits.max_batch_items,
                "max_batch_total_frames": batch_limits.max_batch_total_frames,
            },
        ),
    )
    capabilities: list[BackendCapability] = []
    for backend_id, display_name, dependencies, categories, features, limits in definitions:
        missing = [name for name in dependencies if not _module_available(name)]
        reasons: list[str] = []
        if missing:
            reasons.append(f"missing dependencies: {', '.join(missing)}")
        capabilities.append(
            BackendCapability(
                backend_id=backend_id,
                display_name=display_name,
                available=not reasons,
                version=_module_version("newton" if backend_id == "newton" else "mujoco"),
                supported_categories=categories,
                output_formats=list(_OUTPUT_FORMATS),
                unavailable_reason="; ".join(reasons) if reasons else None,
                features=features,
                limits=limits,
            )
        )
    return capabilities


class CapabilitiesService:
    """Build one truthful snapshot from optional runtime providers."""

    def __init__(
        self,
        *,
        scheduler_snapshot: Callable[[], object] | None = None,
        robot_provider: Callable[[], Iterable[RobotPreset]] | None = None,
        device_probe: Callable[[], list[DeviceCapability]] = _detect_devices,
        asset_root_provider: Callable[[], Iterable[str]] | None = None,
        batch_limits_provider: Callable[[], BatchLimitSnapshot] = BatchLimitSnapshot,
        available_asset_catalog_available: bool = False,
        preflight_available: bool = False,
        artifact_store_available: bool = False,
        job_manager_available: bool = False,
        job_execution_available: bool = False,
        mcp_available: bool = False,
        agent_rest_available: bool = True,
        json_cli_available: bool = True,
        calibration_assistance_available: bool = False,
        calibration_visual_preview_available: bool = False,
    ) -> None:
        if robot_provider is None:
            from hhtools.robot.registry import list_presets_readonly

            robot_provider = list_presets_readonly
        self._scheduler_snapshot = scheduler_snapshot
        self._robot_provider = robot_provider
        self._device_probe = device_probe
        self._asset_root_provider = asset_root_provider
        self._batch_limits_provider = batch_limits_provider
        self._available_asset_catalog_available = bool(available_asset_catalog_available)
        self._preflight_available = bool(preflight_available)
        self._artifact_store_available = bool(artifact_store_available)
        self._job_manager_available = bool(job_manager_available)
        # Execution, cancellation, and retry all require a trusted executor.
        # A durable JobStore on its own can still serve compact historical
        # queries, but it must not make a client believe new solver work can run.
        self._job_execution_available = bool(job_manager_available and job_execution_available)
        self._mcp_available = bool(mcp_available)
        self._agent_rest_available = bool(agent_rest_available)
        self._json_cli_available = bool(json_cli_available)
        self._calibration_assistance_available = bool(calibration_assistance_available)
        self._calibration_visual_preview_available = bool(
            calibration_assistance_available and calibration_visual_preview_available
        )

    def get_capabilities(self) -> CapabilityResponse:
        """Return a compact snapshot; no solver, queue slot, or asset is created."""

        snapshot = self._scheduler_snapshot() if self._scheduler_snapshot is not None else None
        devices = self._device_probe()
        asset_root_ids = (
            sorted(set(self._asset_root_provider()))
            if self._asset_root_provider is not None
            else []
        )
        return CapabilityResponse(
            service_version=__version__,
            backends=_backend_capabilities(devices, self._batch_limits_provider()),
            devices=devices,
            robots=_robot_capabilities(self._robot_provider()),
            scheduler=_scheduler_capability(snapshot),
            asset_root_ids=asset_root_ids,
            supported_input_formats=list(_INPUT_FORMATS),
            supported_output_formats=list(_OUTPUT_FORMATS),
            features={
                "agent_rest": self._agent_rest_available,
                "asset_inspection": self._asset_root_provider is not None,
                "asset_registry": self._asset_root_provider is not None,
                "available_asset_catalog": self._available_asset_catalog_available,
                "batch_execution": self._job_execution_available,
                "batch_preflight": self._preflight_available,
                "calibration_proposals": self._calibration_assistance_available,
                "calibration_silent_save": self._calibration_assistance_available,
                "calibration_status": self._calibration_assistance_available,
                "calibration_validation": self._calibration_assistance_available,
                "calibration_visual_preview": self._calibration_visual_preview_available,
                "artifact_store": self._artifact_store_available,
                "idempotent_jobs": self._job_manager_available,
                "job_cancellation": self._job_execution_available,
                "job_execution": self._job_execution_available,
                "job_retry": self._job_execution_available,
                "job_spec_v2": True,
                # Phase 4 ships the strict JSON client in the same package. It
                # delegates to this long-lived REST composition root so job
                # ownership never moves into a short-lived CLI process.
                "json_cli": self._json_cli_available,
                "mcp": self._mcp_available,
                "persistent_jobs": self._job_manager_available,
                "preflight": self._preflight_available,
                "r2r_execution": self._job_execution_available,
                "r2r_calibration_proposals": self._calibration_assistance_available,
                "r2r_calibration_silent_save": self._calibration_assistance_available,
                "r2r_calibration_status": self._calibration_assistance_available,
                "r2r_calibration_validation": self._calibration_assistance_available,
                "r2r_calibration_visual_preview": self._calibration_visual_preview_available,
                "r2r_preflight": self._preflight_available,
                "revision_polling": self._job_manager_available,
                "revision_waiting": self._job_manager_available,
            },
        )


__all__ = ["CapabilitiesService"]
