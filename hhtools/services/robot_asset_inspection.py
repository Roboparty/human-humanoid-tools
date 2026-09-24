"""Read-only discovery and inspection for robot URDF bundles.

This module intentionally stops at the asset boundary.  It parses XML with the
standard library, validates referenced files, and reports compact structural
facts.  It never constructs a robot model, invokes a solver, or imports MuJoCo,
Newton, Warp, or ``yourdfpy``.

Public contracts contain only bundle-relative paths.  Absolute paths returned
by discovery are an internal hand-off to :class:`AssetRegistry`; expected
errors never include those host paths.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, NoReturn
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

import yaml  # type: ignore[import-untyped]

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

_MAX_URDF_BYTES = 16 * 1024 * 1024
_MAX_ROBOT_METADATA_BYTES = 2 * 1024 * 1024
_SUPPORTED_JOINT_TYPES = frozenset(
    {"fixed", "revolute", "continuous", "prismatic", "floating", "planar"}
)
_BOUNDED_JOINT_TYPES = frozenset({"revolute", "prismatic"})
_ACTUATED_JOINT_TYPES = frozenset({"revolute", "continuous", "prismatic"})
_MESH_ROLES = frozenset({AssetFileRole.VISUAL_MESH, AssetFileRole.COLLISION_MESH})


@dataclass(frozen=True, slots=True)
class RobotAssetFile:
    """One file discovered as part of a logical robot bundle."""

    path: Path
    role: AssetFileRole
    required: bool = True


@dataclass(frozen=True, slots=True)
class RobotAssetDiscovery:
    """Internal paths and portable metadata for one unambiguous URDF bundle."""

    primary_urdf: Path
    files: tuple[RobotAssetFile, ...]
    metadata: Mapping[str, Any]

    @property
    def primary_file(self) -> Path:
        """Compatibility spelling used by registry discovery adapters."""

        return self.primary_urdf


class RobotAssetDiscoveryError(ValueError):
    """Expected discovery failure carrying an Agent-safe structured error."""

    def __init__(self, error: ApiError, *, candidates: tuple[str, ...] = ()) -> None:
        self.api_error = error
        self.candidates = candidates
        super().__init__(f"{error.code}: {error.message}")

    @property
    def code(self) -> str:
        """Stable machine code for protocol adapters and focused tests."""

        return self.api_error.code


@dataclass(frozen=True, slots=True)
class _MeshDeclaration:
    role: AssetFileRole
    reference: str
    index: int


@dataclass(slots=True)
class _UrdfFacts:
    robot_name: str | None
    link_count: int
    urdf_joint_count: int
    joint_count: int
    fixed_joint_count: int
    joint_type_counts: dict[str, int]
    visual_mesh_declaration_count: int
    collision_mesh_declaration_count: int
    mesh_declarations: tuple[_MeshDeclaration, ...]
    errors: list[ApiError]
    warnings: list[str]


def _api_error(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> ApiError:
    return ApiError(
        code=code,
        message=message,
        retryable=False,
        stage=ErrorStage.ASSET_INSPECTION,
        details=dict(details or {}),
    )


def _raise_discovery(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
    candidates: tuple[str, ...] = (),
) -> NoReturn:
    raise RobotAssetDiscoveryError(
        _api_error(code, message, details=details),
        candidates=candidates,
    )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _safe_relative(path: Path, root: Path) -> str:
    """Return a portable path after containment was already established."""

    return path.relative_to(root).as_posix()


def _contains(root: Path, path: Path, *, strict: bool) -> Path:
    try:
        resolved = path.resolve(strict=strict)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RobotAssetDiscoveryError(
            _api_error(
                "ASSET_OUTSIDE_ALLOWED_ROOT",
                "A robot bundle reference resolves outside the registered bundle.",
            )
        ) from exc
    return resolved


def _read_urdf(path: Path) -> ElementTree.Element:
    """Parse a bounded XML document without resolving external resources."""

    try:
        size = path.stat().st_size
        if size > _MAX_URDF_BYTES:
            _raise_discovery(
                "ROBOT_URDF_PARSE_FAILED",
                "The robot description exceeds the supported inspection size.",
                details={"max_bytes": _MAX_URDF_BYTES},
            )
        payload = path.read_bytes()
    except RobotAssetDiscoveryError:
        raise
    except OSError as exc:
        raise RobotAssetDiscoveryError(
            _api_error("ASSET_NOT_FOUND", "The robot description is unavailable.")
        ) from exc

    upper_payload = payload.upper()
    if b"<!DOCTYPE" in upper_payload or b"<!ENTITY" in upper_payload:
        _raise_discovery(
            "ROBOT_URDF_PARSE_FAILED",
            "DTD and entity declarations are not allowed in robot descriptions.",
        )
    try:
        return ElementTree.fromstring(payload)
    except (ElementTree.ParseError, UnicodeError, ValueError) as exc:
        raise RobotAssetDiscoveryError(
            _api_error(
                "ROBOT_URDF_PARSE_FAILED",
                "The robot description contains malformed XML.",
                details={"exception_type": type(exc).__name__},
            )
        ) from exc


def _children(element: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _descendants(element: ElementTree.Element, name: str) -> Iterable[ElementTree.Element]:
    return (child for child in element.iter() if _local_name(child.tag) == name)


def _mesh_declarations(root: ElementTree.Element) -> tuple[_MeshDeclaration, ...]:
    declarations: list[_MeshDeclaration] = []
    index = 0
    for link in _children(root, "link"):
        for container_name, role in (
            ("visual", AssetFileRole.VISUAL_MESH),
            ("collision", AssetFileRole.COLLISION_MESH),
        ):
            for container in _children(link, container_name):
                for mesh in _descendants(container, "mesh"):
                    declarations.append(
                        _MeshDeclaration(
                            role=role,
                            reference=(mesh.get("filename") or "").strip(),
                            index=index,
                        )
                    )
                    index += 1
    return tuple(declarations)


def _float_attribute(
    element: ElementTree.Element,
    name: str,
    *,
    joint_name: str,
    required: bool,
    errors: list[ApiError],
) -> float | None:
    raw = element.get(name)
    if raw is None:
        if required:
            errors.append(
                _api_error(
                    "ROBOT_URDF_INVALID",
                    "A bounded joint is missing a required numeric limit.",
                    details={"joint": joint_name, "attribute": name},
                )
            )
        return None
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        errors.append(
            _api_error(
                "ROBOT_URDF_INVALID",
                "A joint limit is not a finite number.",
                details={"joint": joint_name, "attribute": name},
            )
        )
        return None
    return value


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(name for name, count in counts.items() if name and count > 1)[:20]


def _validate_topology(
    links: set[str],
    edges: list[tuple[str, str]],
    *,
    errors: list[ApiError],
) -> None:
    valid_edges = [(parent, child) for parent, child in edges if parent in links and child in links]
    children = [child for _, child in valid_edges]
    duplicate_children = _duplicates(children)
    if duplicate_children:
        errors.append(
            _api_error(
                "ROBOT_URDF_INVALID",
                "A robot link has more than one parent joint.",
                details={"links": duplicate_children},
            )
        )

    roots = sorted(links - set(children))
    if links and len(roots) != 1:
        errors.append(
            _api_error(
                "ROBOT_URDF_INVALID",
                "A robot description must define exactly one root link.",
                details={"root_count": len(roots)},
            )
        )

    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree = {name: 0 for name in links}
    for parent, child in valid_edges:
        adjacency[parent].append(child)
        indegree[child] += 1
    queue = [name for name, degree in indegree.items() if degree == 0]
    visited = 0
    while queue:
        current = queue.pop()
        visited += 1
        for child in adjacency[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if links and visited != len(links):
        errors.append(
            _api_error(
                "ROBOT_URDF_INVALID",
                "The robot link graph contains a cycle.",
            )
        )


def _analyse_urdf(root: ElementTree.Element) -> _UrdfFacts:
    errors: list[ApiError] = []
    warnings: list[str] = []
    if _local_name(root.tag) != "robot":
        errors.append(
            _api_error(
                "ROBOT_URDF_INVALID",
                "The XML root element must be robot.",
            )
        )

    robot_name = (root.get("name") or "").strip() or None
    if robot_name is None:
        warnings.append("The robot description does not declare a robot name.")

    link_elements = _children(root, "link")
    link_names = [(element.get("name") or "").strip() for element in link_elements]
    missing_link_names = sum(not name for name in link_names)
    duplicate_links = _duplicates(link_names)
    if not link_elements:
        errors.append(_api_error("ROBOT_URDF_INVALID", "The robot defines no links."))
    if missing_link_names:
        errors.append(
            _api_error(
                "ROBOT_URDF_INVALID",
                "Every robot link must have a name.",
                details={"missing_name_count": missing_link_names},
            )
        )
    if duplicate_links:
        errors.append(
            _api_error(
                "ROBOT_URDF_INVALID",
                "Robot link names must be unique.",
                details={"duplicate_names": duplicate_links},
            )
        )
    links = {name for name in link_names if name}

    joint_elements = _children(root, "joint")
    joint_names = [(element.get("name") or "").strip() for element in joint_elements]
    missing_joint_names = sum(not name for name in joint_names)
    duplicate_joints = _duplicates(joint_names)
    if missing_joint_names:
        errors.append(
            _api_error(
                "ROBOT_URDF_INVALID",
                "Every robot joint must have a name.",
                details={"missing_name_count": missing_joint_names},
            )
        )
    if duplicate_joints:
        errors.append(
            _api_error(
                "ROBOT_URDF_INVALID",
                "Robot joint names must be unique.",
                details={"duplicate_names": duplicate_joints},
            )
        )

    joint_types: Counter[str] = Counter()
    edges: list[tuple[str, str]] = []
    effort_velocity_missing = 0
    for index, joint in enumerate(joint_elements):
        name = (joint.get("name") or "").strip() or f"joint_{index}"
        joint_type = (joint.get("type") or "").strip().lower()
        joint_types[joint_type or "(missing)"] += 1
        if joint_type not in _SUPPORTED_JOINT_TYPES:
            errors.append(
                _api_error(
                    "ROBOT_URDF_INVALID",
                    "A robot joint has an unsupported or missing type.",
                    details={"joint": name, "joint_type": joint_type or None},
                )
            )

        parent_elements = _children(joint, "parent")
        child_elements = _children(joint, "child")
        parent = (parent_elements[0].get("link") or "").strip() if len(parent_elements) == 1 else ""
        child = (child_elements[0].get("link") or "").strip() if len(child_elements) == 1 else ""
        if not parent or not child:
            errors.append(
                _api_error(
                    "ROBOT_URDF_INVALID",
                    "Every joint must declare exactly one parent and one child link.",
                    details={"joint": name},
                )
            )
        else:
            edges.append((parent, child))
            missing_references = sorted({value for value in (parent, child) if value not in links})
            if missing_references:
                errors.append(
                    _api_error(
                        "ROBOT_URDF_INVALID",
                        "A joint refers to a link that is not declared.",
                        details={"joint": name, "links": missing_references},
                    )
                )
            if parent == child:
                errors.append(
                    _api_error(
                        "ROBOT_URDF_INVALID",
                        "A joint cannot connect a link to itself.",
                        details={"joint": name},
                    )
                )

        limit_elements = _children(joint, "limit")
        if joint_type in _BOUNDED_JOINT_TYPES:
            if len(limit_elements) != 1:
                errors.append(
                    _api_error(
                        "ROBOT_URDF_INVALID",
                        "A bounded joint must declare exactly one limit element.",
                        details={"joint": name},
                    )
                )
            else:
                limit = limit_elements[0]
                lower = _float_attribute(
                    limit,
                    "lower",
                    joint_name=name,
                    required=True,
                    errors=errors,
                )
                upper = _float_attribute(
                    limit,
                    "upper",
                    joint_name=name,
                    required=True,
                    errors=errors,
                )
                if lower is not None and upper is not None and lower > upper:
                    errors.append(
                        _api_error(
                            "ROBOT_URDF_INVALID",
                            "A joint lower limit exceeds its upper limit.",
                            details={"joint": name},
                        )
                    )
                if limit.get("effort") is None or limit.get("velocity") is None:
                    effort_velocity_missing += 1
        elif joint_type == "continuous" and (
            not limit_elements
            or any(
                limit.get("effort") is None or limit.get("velocity") is None
                for limit in limit_elements
            )
        ):
            effort_velocity_missing += 1

    if effort_velocity_missing:
        warnings.append(
            f"{effort_velocity_missing} actuated joint(s) omit effort or velocity limits."
        )
    _validate_topology(links, edges, errors=errors)

    declarations = _mesh_declarations(root)
    empty_mesh_references = sum(not declaration.reference for declaration in declarations)
    if empty_mesh_references:
        errors.append(
            _api_error(
                "ROBOT_URDF_INVALID",
                "Every mesh declaration must include a filename.",
                details={"missing_filename_count": empty_mesh_references},
            )
        )
    return _UrdfFacts(
        robot_name=robot_name,
        link_count=len(link_elements),
        urdf_joint_count=len(joint_elements),
        joint_count=sum(
            joint_type in _ACTUATED_JOINT_TYPES for joint_type in joint_types.elements()
        ),
        fixed_joint_count=joint_types.get("fixed", 0),
        joint_type_counts=dict(sorted(joint_types.items())),
        visual_mesh_declaration_count=sum(
            declaration.role is AssetFileRole.VISUAL_MESH for declaration in declarations
        ),
        collision_mesh_declaration_count=sum(
            declaration.role is AssetFileRole.COLLISION_MESH for declaration in declarations
        ),
        mesh_declarations=declarations,
        errors=errors,
        warnings=warnings,
    )


def _reference_candidate(reference: str, *, urdf_path: Path, bundle_root: Path) -> Path:
    """Convert a URDF mesh reference to an internal candidate path."""

    raw = reference.strip()
    if not raw or "\x00" in raw:
        _raise_discovery(
            "BUNDLE_INCOMPLETE",
            "A robot mesh reference is empty or invalid.",
        )

    parsed = urlsplit(raw)
    scheme = parsed.scheme.casefold()
    if scheme == "package":
        if parsed.query or parsed.fragment or not parsed.netloc:
            _raise_discovery(
                "BUNDLE_INCOMPLETE",
                "A package mesh reference is malformed.",
            )
        package = unquote(parsed.netloc)
        package_path = unquote(parsed.path).replace("\\", "/").lstrip("/")
        parts = PurePosixPath(package_path).parts
        if package in {".", ".."} or "/" in package or not parts:
            _raise_discovery(
                "BUNDLE_INCOMPLETE",
                "A package mesh reference is malformed.",
            )
        return bundle_root.joinpath(*parts)

    # Absolute references are not portable manifest identities.  Even when an
    # absolute path currently lands inside the registered bundle, copying the
    # URDF into an Agent job snapshot would leave the string pointing back at
    # the mutable source directory.  Loader mesh repair could then read from or
    # write beside that source file instead of the isolated snapshot.
    decoded = unquote(raw)
    windows_path = PureWindowsPath(decoded)
    posix_path = PurePosixPath(decoded.replace("\\", "/"))
    if (
        scheme == "file"
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or posix_path.is_absolute()
    ):
        _raise_discovery(
            "ASSET_OUTSIDE_ALLOWED_ROOT",
            "Robot mesh references must be bundle-relative or use package URIs.",
        )

    # A Windows drive such as C:\\robot\\mesh.stl is parsed as a one-letter
    # URI scheme.  It was rejected above rather than treated as an unknown URI.
    if scheme and not windows_path.drive:
        _raise_discovery(
            "BUNDLE_INCOMPLETE",
            "The robot description uses an unsupported mesh URI scheme.",
        )
    normalized = decoded.replace("\\", "/")
    return urdf_path.parent.joinpath(*PurePosixPath(normalized).parts)


def _compiler_path_errors(
    root: ElementTree.Element,
    *,
    urdf_path: Path,
    bundle_root: Path,
) -> list[ApiError]:
    """Reject MuJoCo compiler directories that can escape an Agent snapshot."""

    errors: list[ApiError] = []
    for mujoco in _children(root, "mujoco"):
        for compiler in _children(mujoco, "compiler"):
            for attribute in ("meshdir", "texturedir", "assetdir"):
                raw = compiler.get(attribute)
                if raw is None or not raw.strip():
                    continue
                value = raw.strip()
                parsed = urlsplit(value)
                decoded = unquote(value)
                windows_path = PureWindowsPath(decoded)
                posix_path = PurePosixPath(decoded.replace("\\", "/"))
                if (
                    parsed.scheme
                    or parsed.netloc
                    or parsed.query
                    or parsed.fragment
                    or windows_path.is_absolute()
                    or bool(windows_path.drive)
                    or posix_path.is_absolute()
                ):
                    errors.append(
                        _api_error(
                            "ASSET_OUTSIDE_ALLOWED_ROOT",
                            "MuJoCo compiler directories must be bundle-relative.",
                            details={"attribute": attribute},
                        )
                    )
                    continue
                candidate = urdf_path.parent.joinpath(*posix_path.parts)
                try:
                    candidate.resolve(strict=False).relative_to(bundle_root)
                except (OSError, RuntimeError, ValueError):
                    errors.append(
                        _api_error(
                            "ASSET_OUTSIDE_ALLOWED_ROOT",
                            "A MuJoCo compiler directory resolves outside the robot bundle.",
                            details={"attribute": attribute},
                        )
                    )
    return errors


def _resolve_mesh(
    declaration: _MeshDeclaration,
    *,
    urdf_path: Path,
    bundle_root: Path,
) -> Path:
    candidate = _reference_candidate(
        declaration.reference,
        urdf_path=urdf_path,
        bundle_root=bundle_root,
    )
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(bundle_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RobotAssetDiscoveryError(
            _api_error(
                "ASSET_OUTSIDE_ALLOWED_ROOT",
                "A robot mesh reference resolves outside the registered bundle.",
                details={"role": declaration.role.value, "declaration_index": declaration.index},
            )
        ) from exc
    if not resolved.is_file():
        relative = _safe_relative(resolved, bundle_root)
        raise RobotAssetDiscoveryError(
            _api_error(
                "BUNDLE_INCOMPLETE",
                "A referenced robot mesh is missing or is not a regular file.",
                details={"relative_path": relative, "role": declaration.role.value},
            )
        )
    try:
        strict_resolved = candidate.resolve(strict=True)
        strict_resolved.relative_to(bundle_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RobotAssetDiscoveryError(
            _api_error(
                "ASSET_OUTSIDE_ALLOWED_ROOT",
                "A robot mesh reference resolves outside the registered bundle.",
                details={"role": declaration.role.value, "declaration_index": declaration.index},
            )
        ) from exc
    return strict_resolved


def _candidate_urdfs(candidate: Path, bundle_root: Path) -> list[Path]:
    candidates: list[Path] = []
    try:
        for current, directory_names, file_names in os.walk(
            bundle_root,
            topdown=True,
            followlinks=False,
        ):
            directory_names[:] = sorted(
                (
                    name
                    for name in directory_names
                    if not name.startswith((".", "_")) and not (Path(current) / name).is_symlink()
                ),
                key=str.casefold,
            )
            file_names.sort(key=str.casefold)
            directory = Path(current)
            for name in file_names:
                if name.startswith((".", "_")) or Path(name).suffix.casefold() != ".urdf":
                    continue
                path = directory / name
                resolved = _contains(bundle_root, path, strict=True)
                if not resolved.is_file():
                    continue
                candidates.append(resolved)
    except RobotAssetDiscoveryError:
        raise
    except OSError as exc:
        raise RobotAssetDiscoveryError(
            _api_error("ASSET_NOT_FOUND", "The robot bundle cannot be read.")
        ) from exc
    return sorted(set(candidates), key=lambda path: _safe_relative(path, bundle_root).casefold())


def _bundle_context(candidate: str | Path) -> tuple[Path, Path]:
    path = Path(candidate)
    try:
        if path.is_dir():
            bundle_root = path.resolve(strict=True)
            candidates = _candidate_urdfs(path, bundle_root)
            if not candidates:
                _raise_discovery(
                    "BUNDLE_INCOMPLETE",
                    "The robot bundle contains no URDF description.",
                )
            if len(candidates) > 1:
                relative = tuple(_safe_relative(item, bundle_root) for item in candidates)
                _raise_discovery(
                    "BUNDLE_AMBIGUOUS",
                    "The directory contains multiple robot descriptions; register one bundle.",
                    details={"candidates": list(relative)},
                    candidates=relative,
                )
            return bundle_root, candidates[0]
        if path.is_file() or path.is_symlink():
            if path.suffix.casefold() != ".urdf":
                _raise_discovery(
                    "UNSUPPORTED_FORMAT",
                    "Robot bundle discovery requires a URDF file or directory.",
                )
            bundle_root = path.parent.resolve(strict=True)
            primary = _contains(bundle_root, path, strict=True)
            if not primary.is_file():
                _raise_discovery(
                    "ASSET_NOT_FOUND",
                    "The robot description is not a regular file.",
                )
            return bundle_root, primary
    except RobotAssetDiscoveryError:
        raise
    except OSError as exc:
        raise RobotAssetDiscoveryError(
            _api_error("ASSET_NOT_FOUND", "The robot bundle is unavailable.")
        ) from exc
    _raise_discovery("ASSET_NOT_FOUND", "The robot bundle is unavailable.")


def _metadata(facts: _UrdfFacts, *, unique_meshes: int, shared_meshes: int) -> dict[str, Any]:
    return {
        "source_format": "urdf",
        "robot_name": facts.robot_name,
        "link_count": facts.link_count,
        "urdf_joint_count": facts.urdf_joint_count,
        "joint_count": facts.joint_count,
        "fixed_joint_count": facts.fixed_joint_count,
        "joint_type_counts": facts.joint_type_counts,
        "visual_mesh_declaration_count": facts.visual_mesh_declaration_count,
        "collision_mesh_declaration_count": facts.collision_mesh_declaration_count,
        "unique_mesh_count": unique_meshes,
        "shared_visual_collision_mesh_count": shared_meshes,
    }


def _robot_metadata_files(bundle_root: Path, primary: Path) -> tuple[Path, ...]:
    """Return robot YAML, calibrations, and declared scaler configs."""

    candidates = {
        bundle_root / "robot.yaml",
        bundle_root / f"robot.{primary.stem}.yaml",
        primary.parent / "robot.yaml",
        primary.parent / f"robot.{primary.stem}.yaml",
    }
    # Human- or Agent-validated calibrations are executable robot configuration, not
    # incidental workspace state.  Binding them into the robot manifest means
    # editing or adding one requires a new robot asset id before preflight can
    # produce another runnable plan.
    for directory in {bundle_root, primary.parent}:
        candidates.update(directory.glob("retarget_calibration*.yaml"))
        candidates.update(directory.glob("r2r_calibration_*.yaml"))
    files: set[Path] = set()
    for candidate in sorted(candidates, key=lambda item: item.as_posix().casefold()):
        if not candidate.exists() and not candidate.is_symlink():
            continue
        metadata_path = _contains(bundle_root, candidate, strict=True)
        if not metadata_path.is_file():
            _raise_discovery(
                "BUNDLE_INCOMPLETE",
                "Robot metadata exists but is not a regular file.",
            )
        try:
            if metadata_path.stat().st_size > _MAX_ROBOT_METADATA_BYTES:
                _raise_discovery(
                    "ROBOT_METADATA_INVALID",
                    "Robot metadata exceeds the supported inspection size.",
                    details={"max_bytes": _MAX_ROBOT_METADATA_BYTES},
                )
            payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        except RobotAssetDiscoveryError:
            raise
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise RobotAssetDiscoveryError(
                _api_error(
                    "ROBOT_METADATA_INVALID",
                    "Robot metadata could not be parsed safely.",
                    details={"exception_type": type(exc).__name__},
                )
            ) from exc
        if not isinstance(payload, dict):
            _raise_discovery(
                "ROBOT_METADATA_INVALID",
                "Robot metadata must contain a YAML mapping.",
            )
        files.add(metadata_path)
        raw_urdf = payload.get("urdf")
        if raw_urdf is not None:
            if not isinstance(raw_urdf, str) or not raw_urdf.strip() or "\x00" in raw_urdf:
                _raise_discovery(
                    "ROBOT_METADATA_INVALID",
                    "The robot metadata urdf field must be a non-empty path string.",
                )
            urdf_value = raw_urdf.strip().replace("\\", "/")
            windows_urdf = PureWindowsPath(urdf_value)
            posix_urdf = PurePosixPath(urdf_value)
            if windows_urdf.is_absolute() or windows_urdf.drive or posix_urdf.is_absolute():
                _raise_discovery(
                    "ROBOT_METADATA_INVALID",
                    "The robot metadata urdf path must be bundle-relative.",
                )
            configured_urdf = metadata_path.parent.joinpath(*posix_urdf.parts)
            try:
                resolved_urdf = configured_urdf.resolve(strict=True)
                resolved_urdf.relative_to(bundle_root)
            except (OSError, RuntimeError, ValueError) as exc:
                raise RobotAssetDiscoveryError(
                    _api_error(
                        "BUNDLE_INCOMPLETE",
                        "The URDF declared by robot metadata is missing or outside the bundle.",
                    )
                ) from exc
            if resolved_urdf != primary:
                _raise_discovery(
                    "BUNDLE_METADATA_MISMATCH",
                    "Robot metadata refers to a different URDF than the bundle primary.",
                )
        raw_search_paths = payload.get("mesh_search_paths")
        if raw_search_paths is not None:
            if not isinstance(raw_search_paths, list):
                _raise_discovery(
                    "ROBOT_METADATA_INVALID",
                    "Robot mesh_search_paths must be a list of bundle-relative directories.",
                )
            for raw_search_path in raw_search_paths:
                if (
                    not isinstance(raw_search_path, str)
                    or not raw_search_path.strip()
                    or "\x00" in raw_search_path
                ):
                    _raise_discovery(
                        "ROBOT_METADATA_INVALID",
                        "Each robot mesh search path must be a non-empty path string.",
                    )
                search_value = raw_search_path.strip().replace("\\", "/")
                windows_search = PureWindowsPath(search_value)
                posix_search = PurePosixPath(search_value)
                if (
                    windows_search.is_absolute()
                    or windows_search.drive
                    or posix_search.is_absolute()
                ):
                    _raise_discovery(
                        "ROBOT_METADATA_INVALID",
                        "Robot mesh search paths must be bundle-relative.",
                    )
                search_candidate = metadata_path.parent.joinpath(*posix_search.parts)
                try:
                    search_path = search_candidate.resolve(strict=True)
                    search_path.relative_to(bundle_root)
                except (OSError, RuntimeError, ValueError) as exc:
                    raise RobotAssetDiscoveryError(
                        _api_error(
                            "ASSET_OUTSIDE_ALLOWED_ROOT",
                            "A robot mesh search path is missing or outside the bundle.",
                        )
                    ) from exc
                if not search_path.is_dir():
                    _raise_discovery(
                        "BUNDLE_INCOMPLETE",
                        "A robot mesh search path is not a directory.",
                    )
        retarget = payload.get("retarget")
        references = retarget.get("references") if isinstance(retarget, dict) else None
        if not isinstance(references, dict):
            continue
        for reference_config in references.values():
            if not isinstance(reference_config, dict):
                continue
            raw_scaler = reference_config.get("scaler_config")
            if raw_scaler is None:
                continue
            if not isinstance(raw_scaler, str) or not raw_scaler.strip() or "\x00" in raw_scaler:
                _raise_discovery(
                    "ROBOT_METADATA_INVALID",
                    "A declared scaler_config must be a non-empty path string.",
                )
            scaler_value = raw_scaler.strip()
            windows_path = PureWindowsPath(scaler_value)
            posix_path = PurePosixPath(scaler_value.replace("\\", "/"))
            if windows_path.is_absolute() or windows_path.drive or posix_path.is_absolute():
                _raise_discovery(
                    "ROBOT_METADATA_INVALID",
                    "A declared scaler_config path must be bundle-relative.",
                )
            scaler_candidate = metadata_path.parent.joinpath(*posix_path.parts)
            try:
                scaler_path = scaler_candidate.resolve(strict=False)
                scaler_path.relative_to(bundle_root)
            except (OSError, RuntimeError, ValueError) as exc:
                raise RobotAssetDiscoveryError(
                    _api_error(
                        "ASSET_OUTSIDE_ALLOWED_ROOT",
                        "A scaler config resolves outside the registered robot bundle.",
                    )
                ) from exc
            if not scaler_path.is_file():
                _raise_discovery(
                    "BUNDLE_INCOMPLETE",
                    "A declared robot scaler config is missing.",
                    details={"relative_path": _safe_relative(scaler_path, bundle_root)},
                )
            files.add(_contains(bundle_root, scaler_candidate, strict=True))
    return tuple(sorted(files, key=lambda path: _safe_relative(path, bundle_root).casefold()))


def discover_robot_bundle(candidate: str | Path) -> RobotAssetDiscovery:
    """Discover one URDF plus every in-bundle file required by that URDF."""

    bundle_root, primary = _bundle_context(candidate)
    xml_root = _read_urdf(primary)
    facts = _analyse_urdf(xml_root)
    if facts.errors:
        first = facts.errors[0]
        raise RobotAssetDiscoveryError(first)
    compiler_errors = _compiler_path_errors(
        xml_root,
        urdf_path=primary,
        bundle_root=bundle_root,
    )
    if compiler_errors:
        raise RobotAssetDiscoveryError(compiler_errors[0])

    mesh_roles: dict[Path, set[AssetFileRole]] = defaultdict(set)
    for declaration in facts.mesh_declarations:
        if not declaration.reference:
            continue
        resolved = _resolve_mesh(
            declaration,
            urdf_path=primary,
            bundle_root=bundle_root,
        )
        mesh_roles[resolved].add(declaration.role)

    files: list[RobotAssetFile] = [RobotAssetFile(primary, AssetFileRole.ROBOT_DESCRIPTION)]
    metadata_files = _robot_metadata_files(bundle_root, primary)
    files.extend(RobotAssetFile(path, AssetFileRole.METADATA) for path in metadata_files)

    shared_meshes = 0
    for path, roles in sorted(
        mesh_roles.items(),
        key=lambda item: _safe_relative(item[0], bundle_root).casefold(),
    ):
        if len(roles) > 1:
            shared_meshes += 1
        # AssetFile has one semantic role.  A shared collision/visual mesh is
        # classified as collision because that is the stricter operational use;
        # metadata retains the fact that the file is shared.
        role = (
            AssetFileRole.COLLISION_MESH
            if AssetFileRole.COLLISION_MESH in roles
            else AssetFileRole.VISUAL_MESH
        )
        files.append(RobotAssetFile(path, role))

    deduplicated: dict[Path, RobotAssetFile] = {}
    for item in files:
        previous = deduplicated.get(item.path)
        if previous is None:
            deduplicated[item.path] = item
        elif previous.role is not item.role:
            precedence = {
                AssetFileRole.ROBOT_DESCRIPTION: 4,
                AssetFileRole.METADATA: 3,
                AssetFileRole.COLLISION_MESH: 2,
                AssetFileRole.VISUAL_MESH: 1,
            }
            if precedence.get(item.role, 0) > precedence.get(previous.role, 0):
                deduplicated[item.path] = item
    ordered = tuple(
        sorted(
            deduplicated.values(),
            key=lambda item: (
                0 if item.path == primary else 1,
                _safe_relative(item.path, bundle_root).casefold(),
            ),
        )
    )
    return RobotAssetDiscovery(
        primary_urdf=primary,
        files=ordered,
        metadata={
            **_metadata(
                facts,
                unique_meshes=len(mesh_roles),
                shared_meshes=shared_meshes,
            ),
            "metadata_file_count": len(metadata_files),
        },
    )


def _sha256(path: Path) -> tuple[str, int, bool]:
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    after = path.stat()
    before_snapshot = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_snapshot = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    stable = before_snapshot == after_snapshot and size == after.st_size
    return digest.hexdigest(), size, stable


def _inspection_path(
    bundle_root: Path,
    relative_path: str,
) -> tuple[Path | None, ApiError | None]:
    candidate = bundle_root.joinpath(*relative_path.split("/"))
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(bundle_root)
    except (OSError, RuntimeError, ValueError):
        return None, _api_error(
            "ASSET_OUTSIDE_ALLOWED_ROOT",
            "A robot bundle file resolves outside its registered root.",
            details={"relative_path": relative_path},
        )
    if not resolved.is_file():
        return None, _api_error(
            "BUNDLE_INCOMPLETE",
            "A required robot bundle file is missing.",
            details={"relative_path": relative_path},
        )
    try:
        strict_resolved = candidate.resolve(strict=True)
        strict_resolved.relative_to(bundle_root)
    except (OSError, RuntimeError, ValueError):
        return None, _api_error(
            "ASSET_OUTSIDE_ALLOWED_ROOT",
            "A robot bundle file resolves outside its registered root.",
            details={"relative_path": relative_path},
        )
    return strict_resolved, None


class RobotAssetInspector:
    """Validate one registered robot bundle without loading a robot runtime."""

    def inspect(
        self,
        bundle: AssetBundle,
        bundle_root: str | Path,
        *,
        verify_hashes: bool = True,
        parse_content: bool = True,
    ) -> AssetInspection:
        """Return compact URDF facts and structured, Agent-safe failures."""

        errors: list[ApiError] = []
        warnings: list[str] = []
        root_path = Path(bundle_root)
        try:
            root = root_path.resolve(strict=True)
            if not root.is_dir():
                raise OSError
        except (OSError, RuntimeError):
            root = root_path.resolve(strict=False)
            errors.append(
                _api_error(
                    "ASSET_NOT_FOUND",
                    "The registered robot bundle root is unavailable.",
                )
            )

        if bundle.kind is not AssetKind.ROBOT_BUNDLE:
            errors.append(
                _api_error(
                    "UNSUPPORTED_ASSET_KIND",
                    "Robot inspection requires a robot_bundle asset.",
                    details={"kind": bundle.kind.value},
                )
            )
        if bundle.category is not AssetCategory.ROBOT_MODEL:
            errors.append(
                _api_error(
                    "BUNDLE_METADATA_MISMATCH",
                    "A robot bundle must use the robot_model category.",
                    details={"category": bundle.category.value},
                )
            )

        resolved_files: dict[str, Path] = {}
        manifest_by_path = {item.relative_path: item for item in bundle.files}
        if root.is_dir():
            for item in bundle.files:
                path, path_error = _inspection_path(root, item.relative_path)
                if path_error is not None:
                    if not item.required and path_error.code == "BUNDLE_INCOMPLETE":
                        warnings.append(
                            f"Optional robot bundle file is missing: {item.relative_path}."
                        )
                    else:
                        errors.append(path_error)
                    continue
                assert path is not None
                resolved_files[item.relative_path] = path
                if verify_hashes:
                    try:
                        digest, size, stable = _sha256(path)
                    except OSError:
                        errors.append(
                            _api_error(
                                "ASSET_NOT_FOUND",
                                "A robot bundle file cannot be read.",
                                details={"relative_path": item.relative_path},
                            )
                        )
                        continue
                    if not stable or digest != item.sha256 or size != item.size_bytes:
                        errors.append(
                            _api_error(
                                "ASSET_HASH_MISMATCH",
                                "A robot bundle file no longer matches its manifest.",
                                details={
                                    "relative_path": item.relative_path,
                                    "expected_sha256": item.sha256,
                                    "actual_sha256": digest,
                                },
                            )
                        )

        primary_manifest = manifest_by_path.get(bundle.primary_file)
        if Path(bundle.primary_file).suffix.casefold() != ".urdf":
            errors.append(
                _api_error(
                    "UNSUPPORTED_FORMAT",
                    "Robot inspection requires a URDF primary file.",
                )
            )
        if primary_manifest is None:
            errors.append(
                _api_error(
                    "BUNDLE_INCOMPLETE",
                    "The primary robot description is absent from the manifest.",
                )
            )
        elif primary_manifest.role is not AssetFileRole.ROBOT_DESCRIPTION:
            errors.append(
                _api_error(
                    "BUNDLE_METADATA_MISMATCH",
                    "The primary URDF must have the robot_description role.",
                    details={"relative_path": bundle.primary_file},
                )
            )

        primary_path = resolved_files.get(bundle.primary_file)
        primary_integrity_failed = any(
            error.code in {"ASSET_HASH_MISMATCH", "ASSET_NOT_FOUND", "BUNDLE_INCOMPLETE"}
            and error.details.get("relative_path") == bundle.primary_file
            for error in errors
        )
        facts: _UrdfFacts | None = None
        unique_meshes: set[str] = set()
        shared_meshes: set[str] = set()
        content_parsed = False
        if parse_content and primary_path is not None and not primary_integrity_failed:
            try:
                xml_root = _read_urdf(primary_path)
                facts = _analyse_urdf(xml_root)
                errors.extend(facts.errors)
                warnings.extend(facts.warnings)
                errors.extend(
                    _compiler_path_errors(
                        xml_root,
                        urdf_path=primary_path,
                        bundle_root=root,
                    )
                )
                content_parsed = True

                roles_by_path: dict[str, set[AssetFileRole]] = defaultdict(set)
                for declaration in facts.mesh_declarations:
                    if not declaration.reference:
                        continue
                    try:
                        mesh_path = _resolve_mesh(
                            declaration,
                            urdf_path=primary_path,
                            bundle_root=root,
                        )
                    except RobotAssetDiscoveryError as exc:
                        errors.append(exc.api_error)
                        continue
                    relative = _safe_relative(mesh_path, root)
                    unique_meshes.add(relative)
                    roles_by_path[relative].add(declaration.role)
                    manifest_file = manifest_by_path.get(relative)
                    if manifest_file is None:
                        errors.append(
                            _api_error(
                                "BUNDLE_INCOMPLETE",
                                "A referenced robot mesh is not declared in the manifest.",
                                details={
                                    "relative_path": relative,
                                    "role": declaration.role.value,
                                },
                            )
                        )
                    elif manifest_file.role not in _MESH_ROLES:
                        errors.append(
                            _api_error(
                                "BUNDLE_METADATA_MISMATCH",
                                "A referenced robot mesh has a non-mesh manifest role.",
                                details={
                                    "relative_path": relative,
                                    "role": manifest_file.role.value,
                                },
                            )
                        )
                shared_meshes = {
                    relative for relative, roles in roles_by_path.items() if len(roles) > 1
                }
                for relative, roles in roles_by_path.items():
                    manifest_file = manifest_by_path.get(relative)
                    if manifest_file is None or manifest_file.role not in _MESH_ROLES:
                        continue
                    expected_role = (
                        AssetFileRole.COLLISION_MESH
                        if AssetFileRole.COLLISION_MESH in roles
                        else AssetFileRole.VISUAL_MESH
                    )
                    if manifest_file.role is not expected_role:
                        errors.append(
                            _api_error(
                                "BUNDLE_METADATA_MISMATCH",
                                "A robot mesh manifest role does not match its URDF use.",
                                details={
                                    "relative_path": relative,
                                    "declared_role": manifest_file.role.value,
                                    "expected_role": expected_role.value,
                                },
                            )
                        )

                declared_meshes = {
                    relative
                    for relative, item in manifest_by_path.items()
                    if item.role in _MESH_ROLES
                }
                unused = sorted(declared_meshes - unique_meshes)
                if unused:
                    warnings.append(
                        f"{len(unused)} declared mesh file(s) are not referenced by the URDF."
                    )

                expected_metadata = _robot_metadata_files(root, primary_path)
                for metadata_path in expected_metadata:
                    relative = _safe_relative(metadata_path, root)
                    manifest_file = manifest_by_path.get(relative)
                    if manifest_file is None:
                        errors.append(
                            _api_error(
                                "BUNDLE_INCOMPLETE",
                                "Robot metadata or scaler config is absent from the manifest.",
                                details={"relative_path": relative},
                            )
                        )
                    elif manifest_file.role is not AssetFileRole.METADATA:
                        errors.append(
                            _api_error(
                                "BUNDLE_METADATA_MISMATCH",
                                "Robot metadata must use the metadata manifest role.",
                                details={
                                    "relative_path": relative,
                                    "role": manifest_file.role.value,
                                },
                            )
                        )
            except RobotAssetDiscoveryError as exc:
                errors.append(exc.api_error)

        metadata: dict[str, Any] = {
            "content_parsed": content_parsed,
            "manifest_file_count": len(bundle.files),
            "mesh_manifest_count": sum(item.role in _MESH_ROLES for item in bundle.files),
        }
        joint_count = None
        if facts is not None:
            metadata.update(
                _metadata(
                    facts,
                    unique_meshes=len(unique_meshes),
                    shared_meshes=len(shared_meshes),
                )
            )
            joint_count = facts.joint_count

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
            category=AssetCategory.ROBOT_MODEL,
            source_format="urdf",
            joint_count=joint_count,
            warnings=warnings,
            errors=errors,
            metadata=metadata,
        )


__all__ = [
    "RobotAssetDiscovery",
    "RobotAssetDiscoveryError",
    "RobotAssetFile",
    "RobotAssetInspector",
    "discover_robot_bundle",
]
