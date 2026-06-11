# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Resolve robot-trajectory uploads for robot-to-robot batch (R2R).

Profiles mirror the human-motion basket layout:

* **mimic** — standalone ``.csv`` / ``.pkl`` / ``.npz`` robot exports (nested
  folders OK, e.g. ``dataset/clip/clip.csv``).
* **intermimic** — clip **folder** with a robot trajectory plus
  ``*_cleaned_simplified.obj`` interaction meshes (OMOMO-style layout).
* **meshmimic** — clip **folder** with a robot trajectory plus ``*_terrain.obj``.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

_ROBOT_TRAJ_EXTS = (".csv", ".pkl", ".npz")


def _has_terrain_obj(folder: Path, stem: str) -> bool:
    if (folder / f"{stem}_terrain.obj").is_file():
        return True
    return any(folder.glob("*_terrain.obj"))


def _has_intermimic_obj(folder: Path, stem: str) -> bool:
    if (folder / f"{stem}_cleaned_simplified.obj").is_file():
        return True
    return any(folder.glob("*_cleaned_simplified.obj"))


def _sniff_robot_csv(path: Path) -> bool:
    """Return True when ``path`` is a robot (not object-sidecar) trajectory CSV."""
    try:
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    low = s.lower()
                    if "frame:" in low and "retarget_robot" in low:
                        return False
                    continue
                cols = [c.strip().lower() for c in s.split(",")]
                if "pos_x" in cols and "root_x" not in cols:
                    return False
                if "root_x" in cols or any(c.startswith("dof_") for c in cols):
                    return True
                return False
    except (OSError, UnicodeDecodeError):
        return False
    return False


def _joint_q_width_from_pkl(path: Path) -> int:
    """Return ``joint_q`` column count for hhtools robot-export pkls, else ``0``."""
    if path.name.lower() in {"terrain.pkl"}:
        return 0
    try:
        with path.open("rb") as fp:
            blob = pickle.load(fp)
    except Exception:
        return 0
    if not isinstance(blob, dict):
        return 0
    # Human PARC MS clips store pickled ``motion_data`` blobs, not robot joint_q.
    if "motion_data" in blob and not blob.get("hhtools_export"):
        robot = blob.get("robot")
        if not (isinstance(robot, dict) and "joint_q" in robot):
            return 0
    robot = blob.get("robot", blob)
    if not isinstance(robot, dict) or "joint_q" not in robot:
        return 0
    try:
        jq = np.asarray(robot["joint_q"])
    except Exception:
        return 0
    if jq.ndim != 2 or jq.shape[1] < 8:
        return 0
    return int(jq.shape[1])


def _joint_q_width_from_npz(path: Path) -> int:
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return 0
    keys = set(data.files)
    jq_key = next((k for k in ("joint_q", "qpos", "q") if k in keys), None)
    if jq_key is None:
        return 0
    try:
        jq = np.asarray(data[jq_key])
    except Exception:
        return 0
    if jq.ndim != 2 or jq.shape[1] < 8:
        return 0
    return int(jq.shape[1])


def _sniff_robot_pkl(path: Path) -> bool:
    return _joint_q_width_from_pkl(path) > 0


def _sniff_robot_npz(path: Path) -> bool:
    return _joint_q_width_from_npz(path) > 0


def _is_robot_export_trajectory(path: Path) -> bool:
    if not path.is_file():
        return False
    ext = path.suffix.lower()
    if ext not in _ROBOT_TRAJ_EXTS:
        return False
    if path.name.lower().startswith("object_"):
        return False
    if ext == ".csv":
        return _sniff_robot_csv(path)
    if ext in (".pkl", ".pickle"):
        return _sniff_robot_pkl(path)
    if ext == ".npz":
        return _sniff_robot_npz(path)
    return False


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
    elif ext in (".pkl", ".pickle"):
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
    for parent in sorted({p.parent for p in drop_dir.rglob("*_cleaned_simplified.obj")}):
        if parent in seen:
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
            "`.csv/.pkl/.npz` 与 `*_cleaned_simplified.obj`）"
        )
    if profile == "meshmimic":
        raise ValueError(
            "未找到 meshmimic 风格 clip（需要文件夹内含机器人轨迹 "
            "与 `*_terrain.obj`）"
        )
    raise ValueError(
        "未找到机器人轨迹文件（`.csv` / `.pkl` / `.npz`，须为本工具导出格式）"
    )
