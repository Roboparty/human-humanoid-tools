"""Safe discovery and inspection for robot-trajectory Agent bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickletools
import zipfile
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
from hhtools.io.robot_trajectory_detect import sniff_robot_csv

SUPPORTED_R2R_PRIMARY_EXTENSIONS = frozenset({".csv", ".npz", ".pickle", ".pkl"})
_MAX_TRAJECTORY_ARRAY_BYTES = 512 * 1024 * 1024
_MAX_METADATA_MEMBER_BYTES = 1024 * 1024
_MAX_NPZ_MEMBERS = 4_096
_MAX_NPZ_NAME_BYTES = 256 * 1024


class R2RTrajectoryDiscoveryError(ValueError):
    """Expected path-free registration failure for an R2R trajectory."""

    def __init__(self, code: str, message: str, *, candidates: tuple[str, ...] = ()) -> None:
        self.code = code
        self.candidates = candidates
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class R2RTrajectoryDiscovery:
    primary_path: Path
    source_robot_id: str | None
    profile: str
    sidecars: dict[AssetFileRole, tuple[Path, ...]]

    @property
    def has_scene(self) -> bool:
        return bool(
            self.sidecars.get(AssetFileRole.OBJECT_MESH)
            or self.sidecars.get(AssetFileRole.OBJECT_TRAJECTORY)
            or self.sidecars.get(AssetFileRole.TERRAIN_MESH)
        )


@dataclass(slots=True)
class _TrajectoryFacts:
    frame_count: int | None = None
    frame_rate_hz: float | None = None
    joint_count: int | None = None
    dof_names: tuple[str, ...] = ()
    source_robot_id: str | None = None
    warning: str | None = None
    validation_code: str | None = None
    semantically_parsed: bool = True


class _TrajectoryValidationError(ValueError):
    pass


def _api_error(
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> ApiError:
    return ApiError(
        code=code,
        message=message,
        stage=ErrorStage.ASSET_INSPECTION,
        details=details or {},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_robot_from_csv(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="strict") as stream:
            for _index, line in zip(range(256), stream, strict=False):
                stripped = line.strip()
                if not stripped:
                    continue
                if not stripped.startswith("#"):
                    break
                key, separator, value = stripped.lstrip("#").strip().partition(":")
                if separator and key.strip().casefold() in {"robot", "source_robot"}:
                    robot_id = value.strip()
                    return robot_id or None
    except (OSError, UnicodeDecodeError):
        return None
    return None


def _npz_member_names(path: Path) -> set[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if (
                len(names) > _MAX_NPZ_MEMBERS
                or sum(len(name.encode("utf-8")) for name in names) > _MAX_NPZ_NAME_BYTES
            ):
                return set()
            return {
                Path(name).name.removesuffix(".npy").casefold()
                for name in names
                if name.casefold().endswith(".npy")
            }
    except (OSError, ValueError, zipfile.BadZipFile):
        return set()


def _source_robot_from_npz(path: Path) -> str | None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in ("source_robot", "robot"):
                if key in archive.files:
                    value = _bounded_npz_array(
                        archive,
                        key,
                        max_bytes=_MAX_METADATA_MEMBER_BYTES,
                    )
                    if value.dtype.kind in {"S", "U"}:
                        robot_id = str(value.reshape(()).item()).strip()
                        if robot_id:
                            return robot_id
            if "meta_json" in archive.files:
                raw = _bounded_npz_array(
                    archive,
                    "meta_json",
                    max_bytes=_MAX_METADATA_MEMBER_BYTES,
                )
                if raw.dtype.kind in {"S", "U"}:
                    metadata = json.loads(str(raw.reshape(()).item()))
                    if isinstance(metadata, dict):
                        robot_id = str(
                            metadata.get("source_robot") or metadata.get("robot") or ""
                        ).strip()
                        return robot_id or None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return None


def _is_r2r_candidate(path: Path, *, allow_code_capable: bool) -> bool:
    suffix = path.suffix.casefold()
    if not path.is_file() or suffix not in SUPPORTED_R2R_PRIMARY_EXTENSIONS:
        return False
    if suffix == ".csv":
        return sniff_robot_csv(path)
    if suffix == ".npz":
        return bool(_npz_member_names(path).intersection({"joint_q", "qpos", "q"}))
    return allow_code_capable


def _sidecars(primary: Path) -> dict[AssetFileRole, tuple[Path, ...]]:
    parent = primary.parent
    terrain = tuple(sorted(path for path in parent.glob("*_terrain.obj") if path.is_file()))
    object_tracks = tuple(
        sorted(
            path
            for pattern in ("object_*.csv", "object_*.pkl", "object_*.npz")
            for path in parent.glob(pattern)
            if path.is_file() and path != primary
        )
    )
    objects = tuple(
        sorted(path for path in parent.glob("*.obj") if path.is_file() and path not in terrain)
    )
    discovered: dict[AssetFileRole, tuple[Path, ...]] = {}
    if terrain:
        discovered[AssetFileRole.TERRAIN_MESH] = terrain
    if object_tracks:
        discovered[AssetFileRole.OBJECT_TRAJECTORY] = object_tracks
    if objects:
        discovered[AssetFileRole.OBJECT_MESH] = objects
    return discovered


def discover_r2r_trajectory(
    candidate: str | Path,
    *,
    allow_code_capable: bool = False,
) -> R2RTrajectoryDiscovery:
    """Resolve exactly one robot trajectory without decoding pickle content."""

    path = Path(candidate)
    if not path.exists():
        raise R2RTrajectoryDiscoveryError(
            "ASSET_NOT_FOUND",
            "The robot trajectory candidate does not exist.",
        )
    if path.is_file():
        candidates = (
            [path.resolve()]
            if _is_r2r_candidate(
                path,
                allow_code_capable=allow_code_capable,
            )
            else []
        )
        relative_candidates: tuple[str, ...] = ()
    elif path.is_dir():
        root = path.resolve()
        candidates = sorted(
            item.resolve()
            for item in path.rglob("*")
            if _is_r2r_candidate(item, allow_code_capable=allow_code_capable)
        )
        relative_candidates = tuple(item.relative_to(root).as_posix() for item in candidates)
    else:
        candidates = []
        relative_candidates = ()
    if not candidates:
        raise R2RTrajectoryDiscoveryError(
            "ROBOT_TRAJECTORY_NOT_FOUND",
            "The candidate contains no safely recognizable robot trajectory.",
        )
    if len(candidates) != 1:
        raise R2RTrajectoryDiscoveryError(
            "BUNDLE_AMBIGUOUS",
            "The directory contains multiple robot trajectories; register one clip.",
            candidates=relative_candidates,
        )

    primary = candidates[0]
    sidecars = _sidecars(primary)
    if sidecars.get(AssetFileRole.TERRAIN_MESH):
        profile = "meshmimic"
    elif sidecars.get(AssetFileRole.OBJECT_MESH) or sidecars.get(AssetFileRole.OBJECT_TRAJECTORY):
        profile = "intermimic"
    else:
        profile = "mimic"
    source_robot_id = (
        _source_robot_from_csv(primary)
        if primary.suffix.casefold() == ".csv"
        else _source_robot_from_npz(primary)
        if primary.suffix.casefold() == ".npz"
        else None
    )
    return R2RTrajectoryDiscovery(
        primary_path=primary,
        source_robot_id=source_robot_id,
        profile=profile,
        sidecars=sidecars,
    )


def _normalized_column(name: str) -> str:
    value = str(name).strip().casefold()
    for character in (" ", "-", "(", ")", "[", "]", "{", "}"):
        value = value.replace(character, "_")
    while "__" in value:
        value = value.replace("__", "_")
    return value.strip("_")


def _dof_name(column: str) -> str | None:
    raw = str(column).strip()
    if not raw.casefold().startswith("dof_"):
        return None
    name = raw[4:].split("(", 1)[0].strip()
    return name or None


def _positive_float(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise _TrajectoryValidationError(f"{field} must be numeric") from error
    if not math.isfinite(result) or result <= 0.0:
        raise _TrajectoryValidationError(f"{field} must be finite and positive")
    return result


def _inspect_csv(path: Path) -> _TrajectoryFacts:
    if path.stat().st_size > _MAX_TRAJECTORY_ARRAY_BYTES:
        raise _TrajectoryValidationError("robot CSV exceeds the inspection limit")
    metadata: dict[str, str] = {}
    records: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
        for row in csv.reader(stream):
            if not row or not any(cell.strip() for cell in row):
                continue
            if row[0].lstrip().startswith("#"):
                body = ",".join(row).lstrip().lstrip("#").strip()
                key, separator, value = body.partition(":")
                if separator:
                    metadata[key.strip().casefold()] = value.strip()
                continue
            records.append(row)
    if len(records) < 2:
        raise _TrajectoryValidationError("robot CSV requires a header and at least one frame")

    header = [column.strip() for column in records[0]]
    normalized = [_normalized_column(column) for column in header]
    if normalized[0] == "time":
        root_start = 1
        time_column = True
    else:
        root_start = 0
        time_column = False
    expected_root = (
        "root_x",
        "root_y",
        "root_z",
        "root_qx",
        "root_qy",
        "root_qz",
        "root_qw",
    )
    motiondecode_root = (
        "root_pos_x_m",
        "root_pos_y_m",
        "root_pos_z_m",
        "root_rot_w",
        "root_rot_x",
        "root_rot_y",
        "root_rot_z",
    )
    root_columns = tuple(normalized[root_start : root_start + 7])
    if root_columns not in {expected_root, motiondecode_root}:
        raise _TrajectoryValidationError("robot CSV root columns do not match a supported schema")
    dof_names = tuple(
        name for column in header[root_start + 7 :] if (name := _dof_name(column)) is not None
    )
    if len(dof_names) != len(header) - root_start - 7:
        raise _TrajectoryValidationError("every robot CSV joint column must use a dof_ name")
    if not dof_names or len(set(dof_names)) != len(dof_names):
        raise _TrajectoryValidationError("robot CSV DOF names must be non-empty and unique")

    try:
        values = np.asarray(records[1:], dtype=np.float64)
    except ValueError as error:
        raise _TrajectoryValidationError("robot CSV frame values must be numeric") from error
    if values.ndim != 2 or values.shape[1] != len(header):
        raise _TrajectoryValidationError("robot CSV rows must have a consistent column count")
    if not bool(np.isfinite(values).all()):
        raise _TrajectoryValidationError("robot CSV contains NaN or infinite values")
    quaternions = values[:, root_start + 3 : root_start + 7]
    if bool((np.linalg.norm(quaternions, axis=1) <= 1e-8).any()):
        raise _TrajectoryValidationError("robot CSV contains a zero root quaternion")

    frame_rate: float | None = None
    if metadata.get("sample_rate"):
        frame_rate = _positive_float(metadata["sample_rate"], field="sample_rate")
    elif time_column and values.shape[0] > 1:
        delta = float(values[1, 0] - values[0, 0])
        frame_rate = _positive_float(1.0 / delta, field="inferred frame rate")
    warning = None
    if frame_rate is None:
        warning = "Frame rate is absent; R2R preflight must provide source_fps."
    source_robot_id = (metadata.get("source_robot") or metadata.get("robot") or "").strip()
    return _TrajectoryFacts(
        frame_count=int(values.shape[0]),
        frame_rate_hz=frame_rate,
        joint_count=len(dof_names),
        dof_names=dof_names,
        source_robot_id=source_robot_id or None,
        warning=warning,
    )


def _bounded_npz_array(archive: Any, key: str, *, max_bytes: int) -> np.ndarray:
    try:
        member = archive.zip.getinfo(f"{key}.npy")
    except (AttributeError, KeyError) as error:
        raise _TrajectoryValidationError(f"NPZ member {key} is unavailable") from error
    if member.file_size > max_bytes:
        raise _TrajectoryValidationError(f"NPZ member {key} exceeds the inspection limit")
    return np.asarray(archive[key])


def _inspect_npz(path: Path) -> _TrajectoryFacts:
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        joint_key = next((key for key in ("joint_q", "qpos", "q") if key in keys), None)
        if joint_key is None:
            raise _TrajectoryValidationError("robot NPZ has no joint_q/qpos/q array")
        joint_q = _bounded_npz_array(
            archive,
            joint_key,
            max_bytes=_MAX_TRAJECTORY_ARRAY_BYTES,
        )
        if joint_q.ndim != 2 or joint_q.shape[0] < 1 or joint_q.shape[1] < 8:
            raise _TrajectoryValidationError("robot NPZ joint_q must have shape (frames, 7+DOF)")
        if not np.issubdtype(joint_q.dtype, np.number) or not bool(np.isfinite(joint_q).all()):
            raise _TrajectoryValidationError("robot NPZ joint_q must contain finite numbers")
        if bool((np.linalg.norm(joint_q[:, 3:7], axis=1) <= 1e-8).any()):
            raise _TrajectoryValidationError("robot NPZ contains a zero root quaternion")

        dof_count = int(joint_q.shape[1] - 7)
        dof_names: tuple[str, ...] = ()
        if "dof_names" in keys:
            raw_names = _bounded_npz_array(
                archive,
                "dof_names",
                max_bytes=_MAX_METADATA_MEMBER_BYTES,
            )
            if raw_names.dtype.kind not in {"S", "U"}:
                raise _TrajectoryValidationError("robot NPZ DOF names must be strings")
            dof_names = tuple(str(value) for value in raw_names.reshape(-1).tolist())
            if len(dof_names) != dof_count or len(set(dof_names)) != len(dof_names):
                raise _TrajectoryValidationError("robot NPZ DOF names do not match joint_q")

        frame_rate: float | None = None
        for key in ("sample_rate", "fps", "framerate"):
            if key in keys:
                value = _bounded_npz_array(
                    archive,
                    key,
                    max_bytes=_MAX_METADATA_MEMBER_BYTES,
                )
                frame_rate = _positive_float(value.reshape(-1)[0], field=key)
                break

    source_robot_id = _source_robot_from_npz(path)
    warning = None
    if frame_rate is None:
        warning = "Frame rate is absent; R2R preflight must provide source_fps."
    return _TrajectoryFacts(
        frame_count=int(joint_q.shape[0]),
        frame_rate_hz=frame_rate,
        joint_count=dof_count,
        dof_names=dof_names,
        source_robot_id=source_robot_id,
        warning=warning,
    )


def _inspect_pickle(path: Path) -> _TrajectoryFacts:
    saw_stop = False
    with path.open("rb") as stream:
        for opcode, _argument, _position in pickletools.genops(stream):
            if opcode.name == "STOP":
                saw_stop = True
    if not saw_stop:
        raise _TrajectoryValidationError("pickle stream has no STOP opcode")
    return _TrajectoryFacts(
        warning="Pickle content requires isolated validation before R2R execution.",
        validation_code="CONTENT_REQUIRES_ISOLATED_VALIDATION",
        semantically_parsed=False,
    )


class R2RTrajectoryInspector:
    """Validate a registered robot trajectory without constructing either robot."""

    def inspect(
        self,
        bundle: AssetBundle,
        bundle_root: str | Path,
        *,
        verify_hashes: bool = True,
        parse_content: bool = True,
    ) -> AssetInspection:
        errors: list[ApiError] = []
        warnings: list[str] = []
        root = Path(bundle_root)
        try:
            resolved_root = root.resolve(strict=True)
        except OSError:
            resolved_root = root.resolve(strict=False)
            errors.append(_api_error("ASSET_NOT_FOUND", "The registered bundle is unavailable."))
        if not resolved_root.is_dir():
            errors.append(
                _api_error("ASSET_NOT_FOUND", "The registered bundle is not a directory.")
            )
        if bundle.kind is not AssetKind.ROBOT_TRAJECTORY_BUNDLE:
            errors.append(
                _api_error(
                    "UNSUPPORTED_ASSET_KIND",
                    "R2R inspection requires a robot_trajectory_bundle asset.",
                    details={"kind": bundle.kind.value},
                )
            )
        if bundle.category is not AssetCategory.ROBOT_TRAJECTORY:
            errors.append(
                _api_error(
                    "BUNDLE_METADATA_MISMATCH",
                    "The registered asset category is not robot_trajectory.",
                    details={"category": bundle.category.value},
                )
            )

        resolved_files: dict[str, Path] = {}
        for manifest_file in bundle.files:
            candidate = resolved_root.joinpath(*manifest_file.relative_path.split("/"))
            try:
                resolved = candidate.resolve(strict=False)
                resolved.relative_to(resolved_root)
            except (OSError, ValueError):
                errors.append(
                    _api_error(
                        "ASSET_OUTSIDE_ALLOWED_ROOT",
                        "A trajectory bundle file resolves outside its registered root.",
                        details={"relative_path": manifest_file.relative_path},
                    )
                )
                continue
            if not resolved.is_file():
                errors.append(
                    _api_error(
                        "ASSET_NOT_FOUND"
                        if manifest_file.relative_path == bundle.primary_file
                        else "BUNDLE_INCOMPLETE",
                        "A required trajectory bundle file is missing.",
                        details={"relative_path": manifest_file.relative_path},
                    )
                )
                continue
            resolved_files[manifest_file.relative_path] = resolved
            if verify_hashes:
                try:
                    digest = _sha256(resolved)
                except OSError:
                    errors.append(
                        _api_error(
                            "ASSET_NOT_FOUND",
                            "A trajectory bundle file cannot be read.",
                            details={"relative_path": manifest_file.relative_path},
                        )
                    )
                    continue
                if digest != manifest_file.sha256:
                    errors.append(
                        _api_error(
                            "ASSET_HASH_MISMATCH",
                            "A trajectory bundle file no longer matches its registered hash.",
                            details={"relative_path": manifest_file.relative_path},
                        )
                    )

        primary_manifest = next(
            (item for item in bundle.files if item.relative_path == bundle.primary_file),
            None,
        )
        if primary_manifest is None or primary_manifest.role is not AssetFileRole.ROBOT_TRAJECTORY:
            errors.append(
                _api_error(
                    "BUNDLE_METADATA_MISMATCH",
                    "The bundle primary must have the robot_trajectory role.",
                )
            )
        roles = {item.role for item in bundle.files}
        has_terrain = AssetFileRole.TERRAIN_MESH in roles
        has_object = bool(
            AssetFileRole.OBJECT_MESH in roles or AssetFileRole.OBJECT_TRAJECTORY in roles
        )
        profile = "meshmimic" if has_terrain else "intermimic" if has_object else "mimic"
        declared = bundle.detected
        if declared is not None and (
            declared.dataset != "robot_trajectory"
            or declared.trajectory_profile != profile
            or declared.recommended_backend != "newton"
        ):
            errors.append(
                _api_error(
                    "BUNDLE_METADATA_MISMATCH",
                    "Registered R2R routing metadata no longer matches the bundle.",
                )
            )

        facts = _TrajectoryFacts(
            source_robot_id=declared.source_robot_id if declared is not None else None,
            semantically_parsed=False,
        )
        primary = resolved_files.get(bundle.primary_file)
        if parse_content and primary is not None and not errors:
            try:
                suffix = primary.suffix.casefold()
                if suffix == ".csv":
                    facts = _inspect_csv(primary)
                elif suffix == ".npz":
                    facts = _inspect_npz(primary)
                elif suffix in {".pkl", ".pickle"}:
                    facts = _inspect_pickle(primary)
                else:
                    raise _TrajectoryValidationError("unsupported robot trajectory extension")
            except (
                OSError,
                TypeError,
                UnicodeDecodeError,
                ValueError,
            ):
                errors.append(
                    _api_error(
                        "ROBOT_TRAJECTORY_PARSE_FAILED",
                        "The robot trajectory could not be parsed safely.",
                    )
                )
        if facts.warning is not None:
            warnings.append(facts.warning)
        if (
            declared is not None
            and declared.source_robot_id is not None
            and facts.source_robot_id is not None
            and declared.source_robot_id != facts.source_robot_id
        ):
            errors.append(
                _api_error(
                    "BUNDLE_METADATA_MISMATCH",
                    "The trajectory source robot identity changed after registration.",
                )
            )

        metadata: dict[str, Any] = {
            "content_parsed": bool(parse_content and facts.semantically_parsed and not errors),
            "dof_names": list(facts.dof_names),
            "recommended_backend": "newton",
            "trajectory_profile": profile,
        }
        if facts.validation_code is not None:
            metadata["content_validation_code"] = facts.validation_code
        if errors:
            status = InspectionStatus.INVALID
        elif warnings:
            status = InspectionStatus.VALID_WITH_WARNINGS
        else:
            status = InspectionStatus.VALID
        source_robot_id = facts.source_robot_id or (
            declared.source_robot_id if declared is not None else None
        )
        return AssetInspection(
            asset_id=bundle.asset_id,
            status=status,
            kind=bundle.kind,
            category=AssetCategory.ROBOT_TRAJECTORY,
            source_format=Path(bundle.primary_file).suffix.casefold().lstrip(".") or None,
            dataset="robot_trajectory",
            reference_model=None,
            frame_count=facts.frame_count,
            frame_rate_hz=facts.frame_rate_hz,
            duration_seconds=(
                facts.frame_count / facts.frame_rate_hz
                if facts.frame_count is not None and facts.frame_rate_hz is not None
                else None
            ),
            joint_count=facts.joint_count,
            source_robot_id=source_robot_id,
            has_object=has_object,
            has_terrain=has_terrain,
            warnings=warnings,
            errors=errors,
            metadata=metadata,
        )


__all__ = [
    "R2RTrajectoryDiscovery",
    "R2RTrajectoryDiscoveryError",
    "R2RTrajectoryInspector",
    "SUPPORTED_R2R_PRIMARY_EXTENSIONS",
    "discover_r2r_trajectory",
]
