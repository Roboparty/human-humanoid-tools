"""Read-only inspection for registered human-motion asset bundles.

The registry owns path authorization and durable identity.  This module owns the
next, deliberately separate, concern: checking that one already-resolved bundle
still has the files and content needed by retargeting.  It never creates WebUI
session tokens, imports a retarget backend, or writes conversion caches.

All paths accepted here are internal server paths.  Returned contracts contain
only manifest-relative paths and compact statistics, so an Agent cannot learn a
host absolute path through either a successful result or an expected error.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import pickletools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hhtools.contracts import (
    ApiError,
    AssetBundle,
    AssetCategory,
    AssetFileRole,
    AssetInspection,
    AssetKind,
    ErrorStage,
    InspectionStatus,
)
from hhtools.io.mimic_detect import infer_mimic_dataset, path_dataset_hint

from .routing import (
    backend_for_category,
    category_for_dataset,
    reference_for_dataset,
)

SUPPORTED_MOTION_PRIMARY_EXTENSIONS = frozenset(
    {
        ".bvh",
        ".csv",
        ".glb",
        ".gltf",
        ".npy",
        ".npz",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
    }
)

_UNIFIED_NPZ_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "name",
        "framerate",
        "up_axis",
        "bone_names",
        "parent_indices",
        "positions",
        "quaternions",
    }
)
_MAX_NPZ_METADATA_MEMBER_BYTES = 64 * 1024


def _npz_member_within_limit(archive: Any, name: str) -> bool:
    """Check an NPZ member's expanded size before NumPy materializes it."""

    try:
        info = archive.zip.getinfo(f"{name}.npy")
    except (AttributeError, KeyError):
        return False
    return info.file_size <= _MAX_NPZ_METADATA_MEMBER_BYTES


@dataclass(frozen=True, slots=True)
class MotionAssetDiscovery:
    """Cheap routing facts for one unambiguous motion candidate.

    ``primary_path`` and ``sidecars`` are internal values for the registry.  A
    caller must turn them into portable paths before creating an Agent-facing
    contract.
    """

    primary_path: Path
    dataset: str
    category: AssetCategory
    reference: str
    recommended_backend: str
    sidecars: dict[AssetFileRole, tuple[Path, ...]]


class MotionAssetDiscoveryError(ValueError):
    """Expected discovery failure with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        candidates: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.candidates = candidates


@dataclass(slots=True)
class _ContentFacts:
    frame_count: int | None = None
    frame_rate_hz: float | None = None
    joint_count: int | None = None
    has_object: bool = False
    has_terrain: bool = False
    warning: str | None = None
    metadata: dict[str, Any] | None = None
    semantically_parsed: bool = True


class _ContentValidationError(ValueError):
    """An expected, path-free explanation of malformed motion content."""


def _fallback_dataset(path: Path) -> str:
    hint = path_dataset_hint(path)
    if hint:
        return hint
    return {
        ".bvh": "lafan",
        ".glb": "glb",
        ".gltf": "glb",
        ".npy": "meshmimic_holosoma",
        ".npz": "amass",
        ".pkl": "omomo",
        ".pt": "gvhmr",
        ".pth": "gvhmr",
    }.get(path.suffix.lower(), "amass")


def _safe_npz_dataset(path: Path, hint: str | None) -> str:
    """Classify NPZ keys without enabling NumPy's pickle loader."""

    dataset = hint or "amass"
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            if {"schema_version", "bone_names", "positions"}.issubset(keys):
                if hint:
                    dataset = hint
                elif (path.parent / f"{path.stem}_terrain.obj").is_file():
                    dataset = "parc_ms"
                elif "meta_json" in keys and _npz_member_within_limit(archive, "meta_json"):
                    try:
                        raw_meta = np.asarray(archive["meta_json"])
                        if raw_meta.dtype.kind in {"S", "U"}:
                            metadata = json.loads(str(raw_meta.item()))
                            declared = str(metadata.get("dataset", ""))
                            if declared:
                                dataset = declared
                            else:
                                dataset = "unified_npz"
                    except (json.JSONDecodeError, TypeError, ValueError):
                        dataset = "unified_npz"
                else:
                    dataset = "unified_npz"
            elif "poses" in keys or {"pose_body", "trans"}.issubset(keys):
                if hint in {"amass", "gvhmr", "motion_x", "phuma"}:
                    dataset = hint
                else:
                    dataset = "amass"
    except (EOFError, OSError, pickle.UnpicklingError, TypeError, ValueError):
        pass
    return dataset


def _safe_npy_dataset(path: Path, hint: str | None) -> str:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim == 2 and array.shape[1] == 322:
            return "motion_x"
        if array.ndim == 2 and array.shape[1] == 69:
            return "phuma"
    except (EOFError, OSError, TypeError, ValueError):
        pass
    if hint:
        return hint
    if path.stem == path.parent.name:
        return "meshmimic_holosoma"
    return "meshmimic_holosoma"


def _safe_pickle_dataset(path: Path, hint: str | None) -> str:
    # A named dataset ancestor is stronger evidence than a sidecar that may
    # have just gone missing.  This lets inspect() still say "OMOMO mesh is
    # missing" rather than accidentally reclassifying the clip as PARC.
    if hint in {"omomo", "parc_ms"}:
        return hint
    if any(path.parent.glob("*_cleaned_simplified.obj")):
        return "omomo"
    if (path.parent / f"{path.stem}_terrain.obj").is_file():
        return "parc_ms"
    return hint or "omomo"


def _infer_dataset(path: Path) -> str:
    suffix = path.suffix.lower()
    hint = path_dataset_hint(path)
    if suffix in {".glb", ".gltf"}:
        dataset = "glb"
    elif suffix == ".npz":
        dataset = _safe_npz_dataset(path, hint)
    elif suffix == ".npy":
        dataset = _safe_npy_dataset(path, hint)
    elif suffix in {".pickle", ".pkl"}:
        dataset = _safe_pickle_dataset(path, hint)
    elif suffix in {".pt", ".pth"}:
        dataset = hint if hint in {"gvhmr", "kungfu_athlete"} else "gvhmr"
    else:
        try:
            dataset = str(infer_mimic_dataset(path))
        except (
            EOFError,
            ImportError,
            OSError,
            pickle.UnpicklingError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            # Discovery must remain available for a damaged file so inspect()
            # can report MOTION_PARSE_FAILED rather than failing before a
            # contract is produced. Directory hints are deterministic and do
            # not parse executable dataset objects.
            dataset = _fallback_dataset(path)
    return dataset


def _logical_primary_candidates(directory: Path) -> list[Path]:
    root = directory.resolve()
    candidates: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_MOTION_PRIMARY_EXTENSIONS or not path.is_file():
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue

        # A same-stem pickle next to a conventional motion file is a terrain
        # sidecar in the current HHTools layouts, not a second logical clip.
        if resolved.suffix.lower() == ".pkl" and any(
            resolved.with_suffix(extension).is_file()
            for extension in (".npz", ".npy", ".bvh", ".glb", ".gltf")
        ):
            continue
        candidates.append(resolved)
    return candidates


def discover_motion_sidecars(
    primary_path: str | Path,
    *,
    dataset: str | None = None,
) -> dict[AssetFileRole, tuple[Path, ...]]:
    """Return deterministic sibling files that belong to ``primary_path``.

    The helper does no hashing and does not invent missing paths.  It is safe for
    an AssetRegistry to call before it constructs its own discovery record.
    """

    primary = Path(primary_path).resolve()
    dataset_name = dataset or _infer_dataset(primary)
    directory = primary.parent
    found: dict[AssetFileRole, list[Path]] = {}

    def add(role: AssetFileRole, path: Path) -> None:
        if not path.is_file() or path.resolve() == primary:
            return
        values = found.setdefault(role, [])
        resolved = path.resolve()
        if resolved not in values:
            values.append(resolved)

    if dataset_name == "omomo":
        for path in sorted(directory.glob("*_cleaned_simplified.obj")):
            add(AssetFileRole.OBJECT_MESH, path)
    elif dataset_name == "omnicontact":
        for pattern in ("prop_*.csv", "object_pose_*.csv", "object_poses_*.csv"):
            for path in sorted(directory.glob(pattern)):
                if path.name.lower() != "motion_actor.csv":
                    add(AssetFileRole.OBJECT_TRAJECTORY, path)
        add(AssetFileRole.METADATA, directory / "capture_meta.json")
    elif dataset_name == "parc_ms":
        add(AssetFileRole.TERRAIN_MESH, directory / f"{primary.stem}_terrain.obj")
        if primary.suffix.lower() != ".pkl":
            add(AssetFileRole.OTHER, primary.with_suffix(".pkl"))
    elif dataset_name == "meshmimic_holosoma":
        add(AssetFileRole.TERRAIN_MESH, directory / "terrain.obj")
        add(AssetFileRole.TERRAIN_MESH, directory / f"{primary.stem}_terrain.obj")
        add(AssetFileRole.OTHER, directory / f"{directory.name}.pkl")
        current = directory
        for _ in range(4):
            manifest = current / "source.yaml"
            if manifest.is_file():
                add(AssetFileRole.METADATA, manifest)
                break
            if current.parent == current:
                break
            current = current.parent

    return {
        role: tuple(sorted(paths))
        for role, paths in sorted(found.items(), key=lambda item: item[0].value)
    }


def discover_primary(candidate: str | Path) -> MotionAssetDiscovery:
    """Resolve a file or a directory containing exactly one logical clip.

    A directory with multiple clips is intentionally rejected.  Choosing the
    first filesystem entry would make asset identity depend on sort order and
    could silently retarget the wrong performance.
    """

    path = Path(candidate)
    if not path.exists():
        raise MotionAssetDiscoveryError("ASSET_NOT_FOUND", "The motion candidate does not exist.")
    if path.is_dir():
        candidates = _logical_primary_candidates(path)
        if not candidates:
            raise MotionAssetDiscoveryError(
                "ASSET_NOT_FOUND",
                "The directory contains no supported motion clip.",
            )
        if len(candidates) > 1:
            relative = tuple(item.relative_to(path.resolve()).as_posix() for item in candidates)
            raise MotionAssetDiscoveryError(
                "BUNDLE_AMBIGUOUS",
                "The directory contains multiple logical motion clips; register one clip.",
                candidates=relative,
            )
        primary = candidates[0]
    elif path.is_file():
        primary = path.resolve()
    else:
        raise MotionAssetDiscoveryError(
            "ASSET_NOT_FOUND",
            "The motion candidate is not a regular file or directory.",
        )

    suffix = primary.suffix.lower()
    if suffix not in SUPPORTED_MOTION_PRIMARY_EXTENSIONS:
        raise MotionAssetDiscoveryError(
            "UNSUPPORTED_FORMAT",
            f"The motion format {suffix or '(none)'} is not supported.",
        )
    dataset = _infer_dataset(primary)
    category = category_for_dataset(dataset)
    return MotionAssetDiscovery(
        primary_path=primary,
        dataset=dataset,
        category=category,
        reference=reference_for_dataset(dataset, suffix),
        recommended_backend=backend_for_category(category),
        sidecars=discover_motion_sidecars(primary, dataset=dataset),
    )


def _api_error(
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> ApiError:
    return ApiError(
        code=code,
        message=message,
        retryable=False,
        stage=ErrorStage.ASSET_INSPECTION,
        details=details or {},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_or_raise(*arrays: np.ndarray) -> None:
    for array in arrays:
        if np.issubdtype(array.dtype, np.number) and not bool(np.isfinite(array).all()):
            raise _ContentValidationError("motion arrays contain NaN or infinite values")


def _positive_fps(value: Any) -> float:
    fps = float(np.asarray(value).reshape(()))
    if not math.isfinite(fps) or fps <= 0:
        raise _ContentValidationError("frame rate must be finite and greater than zero")
    return fps


def _required_keys(keys: set[str], required: frozenset[str], label: str) -> None:
    missing = sorted(required - keys)
    if missing:
        raise _ContentValidationError(f"{label} is missing required fields: {', '.join(missing)}")


def _scalar_value(value: Any, label: str) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise _ContentValidationError(f"{label} must be a scalar value")
    return array.item()


def _numeric_array(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise _ContentValidationError(f"{label} must be a numeric array")
    _finite_or_raise(array)
    return array


def _inspect_unified_npz(archive: Any, keys: set[str]) -> _ContentFacts:
    """Mirror the structure consumed by :func:`hhtools.io.npz.load_npz`."""

    _required_keys(keys, _UNIFIED_NPZ_REQUIRED_KEYS, "Unified NPZ")
    schema = str(_scalar_value(archive["schema_version"], "schema_version"))
    if schema != "1":
        raise _ContentValidationError("Unified NPZ schema_version must be '1'")
    _scalar_value(archive["name"], "name")
    fps = _positive_fps(archive["framerate"])
    up_axis = str(_scalar_value(archive["up_axis"], "up_axis"))
    if up_axis not in {"X", "Y", "Z"}:
        raise _ContentValidationError("up_axis must be one of X, Y, or Z")

    positions = _numeric_array(archive["positions"], "positions")
    quaternions = _numeric_array(archive["quaternions"], "quaternions")
    bone_names = np.asarray(archive["bone_names"])
    parents = np.asarray(archive["parent_indices"])
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise _ContentValidationError("positions must have shape (frames, joints, 3)")
    if quaternions.ndim != 3 or quaternions.shape[-1] != 4:
        raise _ContentValidationError("quaternions must have shape (frames, joints, 4)")
    if positions.shape[:2] != quaternions.shape[:2]:
        raise _ContentValidationError("positions and quaternions have incompatible shapes")
    if positions.shape[0] == 0 or positions.shape[1] == 0:
        raise _ContentValidationError("motion must contain at least one frame and joint")
    joint_count = int(positions.shape[1])
    if bone_names.ndim != 1 or bone_names.size != joint_count:
        raise _ContentValidationError("bone_names must be a one-dimensional joint list")
    if parents.shape != (joint_count,):
        raise _ContentValidationError("parent_indices must have shape (joints,)")
    try:
        parent_indices = parents.astype(np.int64, copy=False)
    except (TypeError, ValueError) as exc:
        raise _ContentValidationError("parent_indices must contain integers") from exc
    if bool((parent_indices >= joint_count).any()):
        raise _ContentValidationError("parent_indices references an unknown joint")

    # load_npz accesses meta_json when it is present.  Touching it here with
    # allow_pickle=False rejects object arrays without executing them.
    if "meta_json" in keys:
        if not _npz_member_within_limit(archive, "meta_json"):
            raise _ContentValidationError("meta_json exceeds the supported inspection size")
        _scalar_value(archive["meta_json"], "meta_json")
    if "source_format" in keys:
        _scalar_value(archive["source_format"], "source_format")

    object_fields = {
        "objects_names",
        "objects_positions",
        "objects_quaternions",
        "objects_extents",
    }
    has_object = False
    if object_fields.issubset(keys):
        object_names = np.asarray(archive["objects_names"])
        object_positions = _numeric_array(archive["objects_positions"], "objects_positions")
        object_quaternions = _numeric_array(archive["objects_quaternions"], "objects_quaternions")
        object_extents = _numeric_array(archive["objects_extents"], "objects_extents")
        if object_names.ndim != 1:
            raise _ContentValidationError("objects_names must be one-dimensional")
        object_count = int(object_names.size)
        if object_positions.ndim != 3 or object_positions.shape[1:] != (object_count, 3):
            raise _ContentValidationError("objects_positions must have shape (frames, objects, 3)")
        if object_quaternions.ndim != 3 or object_quaternions.shape != object_positions.shape[
            :2
        ] + (4,):
            raise _ContentValidationError(
                "objects_quaternions must have shape (frames, objects, 4)"
            )
        if object_extents.shape != (object_count, 3):
            raise _ContentValidationError("objects_extents must have shape (objects, 3)")
        if "objects_mesh_paths" in keys:
            mesh_paths = np.asarray(archive["objects_mesh_paths"])
            if mesh_paths.ndim != 1:
                raise _ContentValidationError("objects_mesh_paths must be one-dimensional")
        if "objects_scales" in keys:
            scales = _numeric_array(archive["objects_scales"], "objects_scales")
            if scales.ndim != 1:
                raise _ContentValidationError("objects_scales must be one-dimensional")
        has_object = object_count > 0
    return _ContentFacts(
        frame_count=int(positions.shape[0]),
        frame_rate_hz=fps,
        joint_count=joint_count,
        has_object=has_object,
        has_terrain=bool("terrain_heightfield" in keys),
    )


def _inspect_amass_npz(archive: Any, keys: set[str]) -> _ContentFacts:
    """Validate the arrays required by ``AmassAdapter.load_params`` safely."""

    _required_keys(keys, frozenset({"trans", "betas"}), "AMASS NPZ")
    trans = _numeric_array(archive["trans"], "trans")
    betas = _numeric_array(archive["betas"], "betas")
    if trans.ndim != 2 or trans.shape[1] != 3 or trans.shape[0] == 0:
        raise _ContentValidationError("AMASS trans must have shape (frames, 3)")
    if betas.size == 0:
        raise _ContentValidationError("AMASS betas must not be empty")
    frame_count = int(trans.shape[0])

    is_stageii = "pose_body" in keys or "surface_model_type" in keys
    if is_stageii:
        _required_keys(
            keys,
            frozenset({"pose_body", "root_orient"}),
            "AMASS stage-II NPZ",
        )
        root_orient = _numeric_array(archive["root_orient"], "root_orient")
        pose_body = _numeric_array(archive["pose_body"], "pose_body")
        if root_orient.shape != (frame_count, 3):
            raise _ContentValidationError("AMASS stage-II root_orient must have shape (frames, 3)")
        if pose_body.ndim != 2 or pose_body.shape[0] != frame_count:
            raise _ContentValidationError("AMASS stage-II pose_body must be a per-frame matrix")
        for key in ("pose_hand", "pose_eye"):
            if key not in keys:
                continue
            optional_pose = _numeric_array(archive[key], key)
            if optional_pose.ndim != 2 or optional_pose.shape[0] != frame_count:
                raise _ContentValidationError(f"AMASS {key} must be a per-frame matrix")
        parameter_schema = "stageii"
    else:
        _required_keys(keys, frozenset({"poses"}), "AMASS legacy NPZ")
        poses = _numeric_array(archive["poses"], "poses")
        if poses.ndim != 2 or poses.shape[0] != frame_count or poses.shape[1] < 66:
            raise _ContentValidationError(
                "AMASS legacy poses must have shape (frames, at least 66)"
            )
        parameter_schema = "legacy"

    fps = 30.0
    for key in ("mocap_frame_rate", "mocap_framerate"):
        if key in keys:
            fps = _positive_fps(archive[key])
            break
    return _ContentFacts(
        frame_count=frame_count,
        frame_rate_hz=fps,
        metadata={
            "joint_positions_available": False,
            "parameter_schema": parameter_schema,
        },
    )


def _inspect_npz(path: Path, dataset: str) -> _ContentFacts:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        unified_markers = {"positions", "quaternions", "bone_names", "parent_indices"}
        if dataset in {"unified_npz", "parc_ms"} or bool(keys & unified_markers):
            return _inspect_unified_npz(archive, keys)
        return _inspect_amass_npz(archive, keys)


def _holosoma_manifest_facts(path: Path) -> tuple[float, int, int]:
    """Safely read the same manifest fields required by the Holosoma adapter."""

    current = path.parent
    manifest: Path | None = None
    for _ in range(4):
        candidate = current / "source.yaml"
        if candidate.is_file():
            manifest = candidate
            break
        if current.parent == current:
            break
        current = current.parent
    if manifest is None:
        raise _ContentValidationError("Holosoma source.yaml is required")
    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise _ContentValidationError("Holosoma source.yaml must contain a mapping")
        framerate = data.get("framerate", {})
        if not isinstance(framerate, dict):
            raise _ContentValidationError("Holosoma framerate must contain a mapping")
        raw = float(framerate.get("raw_hz", 120.0))
        downsample = int(framerate.get("recommended_downsample", 1))
        downsample = max(1, downsample)
        if not math.isfinite(raw) or raw <= 0:
            raise _ContentValidationError("Holosoma raw frame rate must be positive")

        skeleton = data.get("skeleton")
        if not isinstance(skeleton, dict):
            raise _ContentValidationError("Holosoma source.yaml requires a skeleton mapping")
        names = skeleton.get("joint_names", [])
        parents = skeleton.get("parent_indices", [])
        if not isinstance(names, list) or not isinstance(parents, list) or not names:
            raise _ContentValidationError(
                "Holosoma skeleton requires non-empty joint_names and parent_indices lists"
            )
        if len(names) != len(parents):
            raise _ContentValidationError(
                "Holosoma joint_names and parent_indices must have equal length"
            )
        try:
            parent_indices = np.asarray(parents, dtype=np.int64)
        except (TypeError, ValueError) as exc:
            raise _ContentValidationError("Holosoma parent_indices must contain integers") from exc
        if parent_indices.shape != (len(names),):
            raise _ContentValidationError("Holosoma parent_indices must be one-dimensional")
        if bool((parent_indices >= len(names)).any()):
            raise _ContentValidationError("Holosoma parent_indices references an unknown joint")
        for section in ("coordinate_system", "clip_layout"):
            if section in data and not isinstance(data[section], dict):
                raise _ContentValidationError(f"Holosoma {section} must contain a mapping")
        return raw, downsample, len(names)
    except _ContentValidationError:
        raise
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise _ContentValidationError("Holosoma source.yaml is malformed") from exc


def _inspect_npy(path: Path, dataset: str) -> _ContentFacts:
    array = _numeric_array(
        np.load(path, mmap_mode="r", allow_pickle=False),
        "NPY motion",
    )
    if array.ndim < 2 or array.shape[0] == 0:
        raise _ContentValidationError("NPY motion must be a non-empty array with at least 2 axes")
    _finite_or_raise(array)
    frames = int(array.shape[0])
    joints: int | None = None
    fps = 30.0 if dataset in {"motion_x", "phuma"} else None
    metadata: dict[str, Any] = {}
    if array.ndim == 3 and array.shape[-1] == 3:
        joints = int(array.shape[1])
    elif array.ndim == 2 and dataset == "motion_x" and array.shape[1] != 322:
        raise _ContentValidationError("Motion-X rows must contain 322 values")
    elif array.ndim == 2 and dataset == "phuma" and array.shape[1] != 69:
        raise _ContentValidationError("PHUMA rows must contain 69 values")

    if dataset == "meshmimic_holosoma":
        raw_fps, downsample, expected_joints = _holosoma_manifest_facts(path)
        if array.ndim != 3 or array.shape[2] != 3:
            raise _ContentValidationError(
                "Holosoma joint positions must have shape (frames, joints, 3)"
            )
        if array.shape[1] != expected_joints:
            raise _ContentValidationError("Holosoma joint count does not match source.yaml")
        joints = expected_joints
        metadata.update({"raw_frame_count": frames, "downsample": downsample})
        frames = max(1, math.ceil(frames / downsample))
        fps = raw_fps / downsample
    warning = None if fps is not None else "Frame rate could not be determined without a manifest."
    return _ContentFacts(
        frame_count=frames,
        frame_rate_hz=fps,
        joint_count=joints,
        warning=warning,
        metadata=metadata,
    )


def _inspect_pickle(path: Path) -> _ContentFacts:
    # Pickle is code-capable.  Structural validation with pickletools confirms
    # the stream is complete without executing constructors from an untrusted
    # dataset.  Full decoding remains the execution service's responsibility.
    saw_stop = False
    with path.open("rb") as handle:
        for opcode, _argument, _position in pickletools.genops(handle):
            if opcode.name == "STOP":
                saw_stop = True
    if not saw_stop:
        raise _ContentValidationError("pickle stream has no STOP opcode")
    return _ContentFacts(
        warning=("Pickle structure is valid, but semantic content requires isolated validation."),
        metadata={
            "pickle_executed": False,
            "content_validation_code": "CONTENT_REQUIRES_ISOLATED_VALIDATION",
        },
        semantically_parsed=False,
    )


def _inspect_bvh(path: Path) -> _ContentFacts:
    from hhtools.io.bvh import load_bvh

    motion = load_bvh(path)
    _finite_or_raise(motion.positions, motion.quaternions)
    return _ContentFacts(
        frame_count=motion.num_frames,
        frame_rate_hz=float(motion.framerate),
        joint_count=motion.num_bones,
    )


def _inspect_glb(path: Path) -> _ContentFacts:
    from hhtools.io.base import load_motion

    motion = load_motion(path)
    _finite_or_raise(motion.positions, motion.quaternions)
    return _ContentFacts(
        frame_count=motion.num_frames,
        frame_rate_hz=float(motion.framerate),
        joint_count=motion.num_bones,
    )


def _inspect_csv(path: Path) -> _ContentFacts:
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
    if not rows:
        raise _ContentValidationError("CSV contains no rows")
    return _ContentFacts(
        frame_count=max(0, len(rows) - 1),
        warning="CSV columns require a workflow-specific schema before joints can be identified.",
        metadata={
            "content_validation_code": "CONTENT_REQUIRES_WORKFLOW_SCHEMA",
        },
        semantically_parsed=False,
    )


def _inspect_content(path: Path, dataset: str) -> _ContentFacts:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _inspect_npy(path, dataset)
    if suffix == ".npz":
        return _inspect_npz(path, dataset)
    if suffix in {".pt", ".pth"}:
        return _ContentFacts(
            warning=("Torch checkpoint content requires isolated validation before execution."),
            metadata={
                "checkpoint_executed": False,
                "content_validation_code": "CONTENT_REQUIRES_ISOLATED_VALIDATION",
            },
            semantically_parsed=False,
        )
    inspectors = {
        ".bvh": _inspect_bvh,
        ".csv": _inspect_csv,
        ".glb": _inspect_glb,
        ".gltf": _inspect_glb,
        ".pickle": _inspect_pickle,
        ".pkl": _inspect_pickle,
    }
    inspector = inspectors.get(suffix)
    if inspector is None:
        raise _ContentValidationError(f"unsupported primary extension {suffix or '(none)'}")
    return inspector(path)


class MotionAssetInspector:
    """Validate one content-addressed motion bundle without running a solver."""

    def inspect(
        self,
        bundle: AssetBundle,
        bundle_root: str | Path,
        *,
        verify_hashes: bool = True,
        parse_content: bool = True,
    ) -> AssetInspection:
        """Inspect a resolved bundle root and return only Agent-safe facts."""

        errors: list[ApiError] = []
        warnings: list[str] = []
        root = Path(bundle_root)
        try:
            resolved_root = root.resolve(strict=True)
        except OSError:
            resolved_root = root.resolve(strict=False)
            errors.append(
                _api_error("ASSET_NOT_FOUND", "The registered bundle root is unavailable.")
            )
        if not resolved_root.is_dir():
            errors.append(
                _api_error(
                    "ASSET_NOT_FOUND",
                    "The registered bundle root is not a directory.",
                )
            )

        if bundle.kind is not AssetKind.MOTION_BUNDLE:
            errors.append(
                _api_error(
                    "UNSUPPORTED_ASSET_KIND",
                    "Motion inspection requires a motion_bundle asset.",
                    details={"kind": bundle.kind.value},
                )
            )

        resolved_files: dict[str, Path] = {}
        for manifest_file in bundle.files:
            relative = manifest_file.relative_path
            candidate = resolved_root.joinpath(*relative.split("/"))
            try:
                resolved = candidate.resolve(strict=False)
                resolved.relative_to(resolved_root)
            except (OSError, ValueError):
                errors.append(
                    _api_error(
                        "ASSET_OUTSIDE_ALLOWED_ROOT",
                        "A bundle file resolves outside its registered root.",
                        details={"relative_path": relative},
                    )
                )
                continue
            if not resolved.is_file():
                if not manifest_file.required:
                    warnings.append(
                        f"Optional bundle file is unavailable: {manifest_file.relative_path}."
                    )
                    continue
                code = "ASSET_NOT_FOUND" if relative == bundle.primary_file else "BUNDLE_INCOMPLETE"
                errors.append(
                    _api_error(
                        code,
                        "The primary motion is missing."
                        if code == "ASSET_NOT_FOUND"
                        else "A required bundle sidecar is missing.",
                        details={
                            "relative_path": relative,
                            "role": manifest_file.role.value,
                            "required": manifest_file.required,
                        },
                    )
                )
                continue
            resolved_files[relative] = resolved
            if verify_hashes:
                try:
                    actual_hash = _sha256(resolved)
                except OSError:
                    errors.append(
                        _api_error(
                            "ASSET_NOT_FOUND",
                            "A registered bundle file cannot be read.",
                            details={"relative_path": relative},
                        )
                    )
                    continue
                if actual_hash != manifest_file.sha256:
                    errors.append(
                        _api_error(
                            "ASSET_HASH_MISMATCH",
                            "A bundle file no longer matches its registered content hash.",
                            details={
                                "relative_path": relative,
                                "expected_sha256": manifest_file.sha256,
                                "actual_sha256": actual_hash,
                            },
                        )
                    )

        primary_path = resolved_files.get(bundle.primary_file)
        suffix = Path(bundle.primary_file).suffix.lower()
        source_format = suffix.removeprefix(".") or None
        declared_dataset = bundle.detected.dataset if bundle.detected is not None else None
        detected_dataset = (
            _infer_dataset(primary_path) if primary_path is not None else declared_dataset
        )
        dataset = detected_dataset or declared_dataset
        dataset = dataset or _fallback_dataset(Path(bundle.primary_file))
        category = category_for_dataset(dataset)
        reference = reference_for_dataset(dataset, suffix)
        recommended_backend = backend_for_category(category)

        routing_mismatches: dict[str, dict[str, str]] = {}
        if declared_dataset and detected_dataset and declared_dataset != detected_dataset:
            routing_mismatches["dataset"] = {
                "declared": declared_dataset,
                "detected": detected_dataset,
            }
        if bundle.detected is not None:
            if bundle.detected.reference and bundle.detected.reference != reference:
                routing_mismatches["reference"] = {
                    "declared": bundle.detected.reference,
                    "detected": reference,
                }
            if (
                bundle.detected.recommended_backend
                and bundle.detected.recommended_backend != recommended_backend
            ):
                routing_mismatches["recommended_backend"] = {
                    "declared": bundle.detected.recommended_backend,
                    "detected": recommended_backend,
                }
        if routing_mismatches:
            errors.append(
                _api_error(
                    "BUNDLE_METADATA_MISMATCH",
                    "Registered routing metadata does not match the motion content.",
                    details={"mismatches": routing_mismatches},
                )
            )
        if bundle.category is not category:
            errors.append(
                _api_error(
                    "BUNDLE_METADATA_MISMATCH",
                    "The registered category does not match the detected motion dataset.",
                    details={
                        "declared_category": bundle.category.value,
                        "detected_category": category.value,
                    },
                )
            )

        role_paths: dict[AssetFileRole, list[Path]] = {}
        for manifest_file in bundle.files:
            resolved_manifest_file = resolved_files.get(manifest_file.relative_path)
            if resolved_manifest_file is not None:
                role_paths.setdefault(manifest_file.role, []).append(resolved_manifest_file)

        has_object_sidecar = bool(
            role_paths.get(AssetFileRole.OBJECT_MESH)
            or role_paths.get(AssetFileRole.OBJECT_TRAJECTORY)
        )
        has_terrain_sidecar = bool(role_paths.get(AssetFileRole.TERRAIN_MESH))
        if dataset == "omomo" and not role_paths.get(AssetFileRole.OBJECT_MESH):
            errors.append(
                _api_error(
                    "BUNDLE_INCOMPLETE",
                    "OMOMO interaction bundles require a registered object mesh.",
                    details={"missing_roles": [AssetFileRole.OBJECT_MESH.value]},
                )
            )
        if dataset == "omnicontact" and not role_paths.get(AssetFileRole.OBJECT_TRAJECTORY):
            errors.append(
                _api_error(
                    "BUNDLE_INCOMPLETE",
                    "OmniContact bundles require a registered object trajectory CSV.",
                    details={"missing_roles": [AssetFileRole.OBJECT_TRAJECTORY.value]},
                )
            )
        if dataset == "meshmimic_holosoma":
            missing_roles: list[str] = []
            if not role_paths.get(AssetFileRole.METADATA):
                missing_roles.append(AssetFileRole.METADATA.value)
            if not has_terrain_sidecar:
                missing_roles.append(AssetFileRole.TERRAIN_MESH.value)
            if missing_roles:
                errors.append(
                    _api_error(
                        "BUNDLE_INCOMPLETE",
                        "Holosoma terrain bundles require source metadata and terrain data.",
                        details={"missing_roles": missing_roles},
                    )
                )
        if dataset == "parc_ms" and not has_terrain_sidecar:
            errors.append(
                _api_error(
                    "BUNDLE_INCOMPLETE",
                    "PARC-MS terrain bundles require a registered terrain mesh.",
                    details={"missing_roles": [AssetFileRole.TERRAIN_MESH.value]},
                )
            )

        facts = _ContentFacts()
        primary_hash_failed = any(
            error.code in {"ASSET_HASH_MISMATCH", "ASSET_NOT_FOUND"}
            and error.details.get("relative_path") == bundle.primary_file
            for error in errors
        )
        content_parsed = False
        if parse_content and primary_path is not None and not primary_hash_failed:
            try:
                facts = _inspect_content(primary_path, dataset)
                content_parsed = facts.semantically_parsed
            except _ContentValidationError as exc:
                code = (
                    "MOTION_NONFINITE_VALUES"
                    if "NaN or infinite" in str(exc)
                    else "MOTION_PARSE_FAILED"
                )
                errors.append(
                    _api_error(
                        code,
                        "The motion content contains non-finite numeric values."
                        if code == "MOTION_NONFINITE_VALUES"
                        else "The primary motion content is malformed.",
                        details={"reason": str(exc)},
                    )
                )
            except (
                EOFError,
                ImportError,
                OSError,
                pickle.UnpicklingError,
                RuntimeError,
                TypeError,
                UnicodeError,
                ValueError,
            ) as exc:
                errors.append(
                    _api_error(
                        "MOTION_PARSE_FAILED",
                        "The primary motion content could not be parsed.",
                        details={"exception_type": type(exc).__name__},
                    )
                )

        if facts.warning:
            warnings.append(facts.warning)
        has_object = category is AssetCategory.OBJECT_INTERACTION and (
            has_object_sidecar or facts.has_object or dataset == "omomo"
        )
        has_terrain = category is AssetCategory.TERRAIN_SCENE and (
            has_terrain_sidecar or facts.has_terrain
        )
        metadata: dict[str, Any] = {
            "recommended_backend": recommended_backend,
            "content_parsed": content_parsed,
        }
        if facts.metadata:
            metadata.update(facts.metadata)
        if bundle.category is not category:
            metadata["declared_category"] = bundle.category.value

        duration = None
        if facts.frame_count is not None and facts.frame_rate_hz is not None:
            duration = max(0.0, (facts.frame_count - 1) / facts.frame_rate_hz)
        if errors:
            status = InspectionStatus.INVALID
        elif warnings:
            status = InspectionStatus.VALID_WITH_WARNINGS
        else:
            status = InspectionStatus.VALID
        return AssetInspection(
            asset_id=bundle.asset_id,
            status=status,
            kind=bundle.kind,
            category=category,
            source_format=source_format,
            dataset=dataset,
            reference_model=reference,
            frame_count=facts.frame_count,
            frame_rate_hz=facts.frame_rate_hz,
            duration_seconds=duration,
            joint_count=facts.joint_count,
            has_object=has_object,
            has_terrain=has_terrain,
            warnings=warnings,
            errors=errors,
            metadata=metadata,
        )


__all__ = [
    "MotionAssetDiscovery",
    "MotionAssetDiscoveryError",
    "MotionAssetInspector",
    "SUPPORTED_MOTION_PRIMARY_EXTENSIONS",
    "discover_motion_sidecars",
    "discover_primary",
]
