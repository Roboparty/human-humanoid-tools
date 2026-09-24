# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Resolve robot-trajectory uploads for robot-to-robot batch (R2R).

Profiles mirror the human-motion basket layout:

* **mimic** — standalone ``.csv`` / ``.pkl`` / ``.npz`` robot exports (nested
  folders OK, e.g. ``dataset/clip/clip.csv``).
* **intermimic** — clip **folder** with a robot trajectory plus interaction
  sidecars: ``object_*.csv`` and/or object ``.obj`` meshes (Web export uses
  plain names like ``Box_H_1.obj``; OMOMO-style ``*_cleaned_simplified.obj``
  is also accepted).
* **meshmimic** — clip **folder** with a robot trajectory plus ``*_terrain.obj``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hhtools.io.robot_trajectory_detect import (
    _ROBOT_TRAJ_EXTS,
    _is_robot_export_trajectory,
    _joint_q_width_from_npz,
    _joint_q_width_from_pkl,
)


def _has_terrain_obj(folder: Path, stem: str) -> bool:
    if (folder / f"{stem}_terrain.obj").is_file():
        return True
    return any(folder.glob("*_terrain.obj"))


def _has_intermimic_obj(folder: Path, stem: str = "") -> bool:
    """True when the folder carries interaction meshes or object tracks.

    Accepts Web robot-export layout (``object_*.csv`` + any non-terrain
    ``.obj``) as well as OMOMO ``*_cleaned_simplified.obj``.
    """
    folder = Path(folder)
    if stem and (folder / f"{stem}_cleaned_simplified.obj").is_file():
        return True
    if any(folder.glob("*_cleaned_simplified.obj")):
        return True
    if any(folder.glob("object_*.csv")):
        return True
    for obj in folder.glob("*.obj"):
        name = obj.name.lower()
        if name.endswith("_terrain.obj") or name.endswith("_terrain"):
            continue
        if "_terrain" in name:
            continue
        return True
    return False


def _interaction_clip_dirs(drop_dir: Path) -> set[Path]:
    """Parents of interaction sidecars (object tracks / object meshes)."""
    dirs: set[Path] = set()
    for p in drop_dir.rglob("*_cleaned_simplified.obj"):
        dirs.add(p.parent)
    for p in drop_dir.rglob("object_*.csv"):
        dirs.add(p.parent)
    return dirs


def _robot_traj_rank(path: Path) -> tuple[int, int, str]:
    """Sort key: prefer more DOF columns, then longer stems (main clip name)."""
    score = 0
    ext = path.suffix.lower()
    if ext == ".csv":
        try:
            with path.open(encoding="utf-8") as fp:
                for line in fp:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    score = len(s.split(","))
                    break
        except (OSError, UnicodeDecodeError):
            pass
    elif ext == ".pkl":
        score = _joint_q_width_from_pkl(path)
    elif ext == ".npz":
        score = _joint_q_width_from_npz(path)
    return (score, len(path.stem), str(path))


def _pick_robot_primary(candidates: list[Path], parent: Path) -> Path | None:
    robots = [p for p in candidates if _is_robot_export_trajectory(p)]
    if not robots:
        return None
    for p in robots:
        if p.stem == parent.name:
            return p
    return max(robots, key=_robot_traj_rank)


def _find_meshmimic_primaries(drop_dir: Path) -> list[tuple[str, Path]]:
    """``('csv'|'pkl'|'npz', path)`` primaries inside terrain clip folders."""
    by_dir: dict[Path, tuple[str, Path]] = {}
    for parent in sorted({p.parent for p in drop_dir.rglob("*_terrain.obj")}):
        cands: list[Path] = []
        for ext in _ROBOT_TRAJ_EXTS:
            cands.extend(parent.glob(f"*{ext}"))
        picked = _pick_robot_primary(cands, parent)
        if picked is None:
            continue
        by_dir[parent] = (picked.suffix.lstrip("."), picked)
    return list(by_dir.values())


def _find_intermimic_primaries(drop_dir: Path) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for parent in sorted(_interaction_clip_dirs(drop_dir)):
        if parent in seen:
            continue
        if not _has_intermimic_obj(parent):
            continue
        cands: list[Path] = []
        for ext in _ROBOT_TRAJ_EXTS:
            cands.extend(parent.glob(f"*{ext}"))
        picked = _pick_robot_primary(cands, parent)
        if picked is None:
            continue
        seen.add(parent)
        out.append(picked)
    return out


def _find_mimic_primaries(drop_dir: Path) -> list[Path]:
    """Robot trajectory files that are not scene-folder primaries."""
    scene_dirs = {p.parent for _, p in _find_meshmimic_primaries(drop_dir)}
    scene_dirs |= {p.parent for p in _find_intermimic_primaries(drop_dir)}
    found: list[Path] = []
    for ext in _ROBOT_TRAJ_EXTS:
        for path in sorted(drop_dir.rglob(f"*{ext}")):
            if not path.is_file():
                continue
            if path.parent in scene_dirs:
                continue
            if not _is_robot_export_trajectory(path):
                continue
            found.append(path)
    found.sort(key=lambda p: (len(p.parts), str(p)))
    return found


def detect_r2r_profile(drop_dir: Path) -> str:
    if _find_meshmimic_primaries(drop_dir):
        return "meshmimic"
    if _find_intermimic_primaries(drop_dir):
        return "intermimic"
    return "mimic"


@dataclass(frozen=True)
class R2rClipRef:
    path: Path
    profile: str
    clip_kind: str = ""
    has_scene: bool = False


def r2r_clip_ref_for_path(path: Path, profile: str = "auto") -> R2rClipRef:
    """Validate one exact robot trajectory and preserve its scene semantics.

    Library selection differs from folder upload: the user has selected one
    concrete row, so resolving the first clip below its parent would be wrong.
    This helper validates that exact path and derives only the sidecars that
    belong to it.
    """

    picked = Path(path).resolve()
    if not _is_robot_export_trajectory(picked):
        raise ValueError(
            "所选资源不是机器人轨迹（需要包含 root pose 与机器人关节 DoF 的 "
            "`.csv` / `.pkl` / `.npz`）"
        )

    requested = (profile or "auto").strip().lower()
    if requested not in {"auto", "mimic", "intermimic", "meshmimic"}:
        raise ValueError(f"unsupported R2R profile: {requested}")

    if requested == "auto":
        if _has_terrain_obj(picked.parent, picked.stem):
            requested = "meshmimic"
        elif _has_intermimic_obj(picked.parent, picked.stem):
            requested = "intermimic"
        else:
            requested = "mimic"

    has_scene = requested in {"intermimic", "meshmimic"}
    if requested == "meshmimic" and not _has_terrain_obj(picked.parent, picked.stem):
        raise ValueError("meshmimic 轨迹缺少 `*_terrain.obj` 场景文件")
    if requested == "intermimic" and not _has_intermimic_obj(picked.parent, picked.stem):
        raise ValueError("intermimic 轨迹缺少 `*_cleaned_simplified.obj` 交互物体")

    return R2rClipRef(
        path=picked,
        profile=requested,
        clip_kind=picked.suffix.lstrip("."),
        has_scene=has_scene,
    )


def export_subdir_for_r2r_clip(drop_dir: Path, picked: Path) -> str:
    drop_dir = Path(drop_dir).resolve()
    picked = Path(picked).resolve()
    try:
        rel = picked.relative_to(drop_dir)
        parent = rel.parent
        return parent.as_posix() if parent != Path(".") else ""
    except ValueError:
        return ""


def enumerate_r2r_clips(drop_dir: Path, profile: str = "auto") -> list[R2rClipRef]:
    drop_dir = Path(drop_dir).resolve()
    profile = (profile or "auto").strip().lower()

    out: list[R2rClipRef] = []
    seen: set[str] = set()

    def _add(path: Path, prof: str, *, kind: str = "", scene: bool = False) -> None:
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        out.append(
            R2rClipRef(path=path, profile=prof, clip_kind=kind, has_scene=scene),
        )

    if profile == "auto":
        for kind, path in _find_meshmimic_primaries(drop_dir):
            _add(path, "meshmimic", kind=kind, scene=True)
        for path in _find_intermimic_primaries(drop_dir):
            _add(path, "intermimic", kind=path.suffix.lstrip("."), scene=True)
        for path in _find_mimic_primaries(drop_dir):
            _add(path, "mimic")
        return out

    if profile == "meshmimic":
        for kind, path in _find_meshmimic_primaries(drop_dir):
            _add(path, profile, kind=kind, scene=True)
        return out

    if profile == "intermimic":
        for path in _find_intermimic_primaries(drop_dir):
            _add(path, profile, kind=path.suffix.lstrip("."), scene=True)
        return out

    for path in _find_mimic_primaries(drop_dir):
        _add(path, profile)
    return out


def validate_r2r_upload(drop_dir: Path, profile: str) -> None:
    """Raise :class:`ValueError` when the drop does not match the profile rules."""
    profile = (profile or "auto").strip().lower()
    clips = enumerate_r2r_clips(drop_dir, profile)
    if clips:
        return
    if profile == "intermimic":
        raise ValueError(
            "未找到 intermimic 风格 clip（需要文件夹内含机器人轨迹 "
            "`.csv/.pkl/.npz`，以及 `object_*.csv` 和/或物体 `.obj`；"
            "OMOMO 的 `*_cleaned_simplified.obj` 亦可）"
        )
    if profile == "meshmimic":
        raise ValueError(
            "未找到 meshmimic 风格 clip（需要文件夹内含机器人轨迹 与 `*_terrain.obj`）"
        )
    raise ValueError("未找到机器人轨迹文件（`.csv` / `.pkl` / `.npz`，须为本工具导出格式）")
