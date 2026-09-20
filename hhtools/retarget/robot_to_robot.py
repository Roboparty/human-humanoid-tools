# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Robot-to-robot retargeting.

Convert an *existing robot trajectory* (joint-angle sequence, e.g. an exported
Unitree G1 clip) onto **another** robot.  The idea mirrors the human→robot
pipeline, but the "source motion" is a robot rather than a human:

1. Read the source robot's exported trajectory (``csv`` / ``pkl`` / ``npz`` in
   the hhtools export schema — with or without the ``# comment`` header).
2. Run the source robot's URDF **forward kinematics** per frame to recover the
   world-space positions / orientations of the canonical key joints (driven by
   the source ``robot.yaml:ik_map``).
3. Wrap those keypoints in a :class:`~hhtools.core.motion.Motion` whose bones
   are the *canonical* hhtools joint names — i.e. exactly what the existing
   retarget scaler + Newton IK pipeline already consumes when it processes a
   human clip.
4. Calibrate the target robot once against the source robot's rest pose (the
   source FK at its zero configuration acts as the "reference" skeleton), then
   run the normal Newton IK retarget.

This module is intentionally self-contained so the feature can live as an
independent Web panel without touching the human→robot code paths.  The only
shared hook is the optional ``reference_pose`` override on
:func:`hhtools.retarget.calibration.calibration.build_scaler_config_soma_style`
and :func:`~hhtools.retarget.calibration.calibration.derive_calibration_params`.
"""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hhtools.core.grounding import (
    SOURCE_FLOOR_META_KEY,
    foot_floor_z_in_positions,
)
from hhtools.core.hierarchy import Hierarchy
from hhtools.core.math import quaternion as Q
from hhtools.core.motion import Motion
from hhtools.retarget.calibration.calibration import (
    _collect_link_transforms_at_q,
    _ik_map_pairs,
    _rotmat_to_xyzw,
)
from hhtools.retarget.calibration.reference import HumanReferencePose
from hhtools.retarget.retarget_result import RetargetedMotion
from hhtools.robot.loader import URDFRobotModel

# Fallback when a foreign robot CSV/PKL/NPZ omits ``time`` / ``sample_rate``.
DEFAULT_SOURCE_FRAMERATE = 50.0

__all__ = [
    "DEFAULT_SOURCE_FRAMERATE",
    "SourceTrajectory",
    "build_source_reference_pose",
    "load_r2r_calibration",
    "load_source_trajectory",
    "r2r_calibration_path",
    "retarget_robot_to_robot",
    "save_r2r_calibration",
    "align_retargeted_ankles_to_scaled_source",
    "source_trajectory_to_motion",
    "suggested_r2r_backend",
    "trajectory_to_retargeted_motion",
]


# Canonical upper→lower body topology used to wrap FK keypoints in a Motion.
# Subset of ``configs/skeleton_presets/canonical_human.yaml`` (feet/toes are
# left to the pipeline's endpoint augmentation).
_CANONICAL_PARENTS: dict[str, str | None] = {
    "hips": None,
    "spine": "hips",
    "chest": "spine",
    "neck": "chest",
    "head": "neck",
    "left_shoulder": "chest",
    "left_elbow": "left_shoulder",
    "left_wrist": "left_elbow",
    "right_shoulder": "chest",
    "right_elbow": "right_shoulder",
    "right_wrist": "right_elbow",
    "left_hip": "hips",
    "left_knee": "left_hip",
    "left_ankle": "left_knee",
    "right_hip": "hips",
    "right_knee": "right_hip",
    "right_ankle": "right_knee",
}
_CANONICAL_ORDER: tuple[str, ...] = tuple(_CANONICAL_PARENTS.keys())

_IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Forward kinematics → canonical keypoints
# ---------------------------------------------------------------------------


def _root_matrix(root: NDArray) -> NDArray:
    """4x4 world transform from a ``(tx,ty,tz, qx,qy,qz,qw)`` floating base."""
    T = np.eye(4, dtype=np.float64)
    q = np.asarray(root[3:7], dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        q = _IDENTITY_Q
    else:
        q = q / n
    T[:3, :3] = Q.to_matrix(q.reshape(1, 4))[0].astype(np.float64)
    T[:3, 3] = np.asarray(root[:3], dtype=np.float64)
    return T


def _canonical_keypoints(
    link_tx: dict[str, NDArray],
    ik_pairs: list[tuple[str, str]],
    *,
    T_root: NDArray | None = None,
) -> dict[str, tuple[NDArray, NDArray]]:
    """Map ik-mapped links → ``{canonical: (pos(3), quat_xyzw(4))}`` (world)."""
    out: dict[str, tuple[NDArray, NDArray]] = {}
    for canonical, link in ik_pairs:
        T = link_tx.get(link)
        if T is None:
            continue
        T = np.asarray(T, dtype=np.float64)
        if T_root is not None:
            T = T_root @ T
        pos = T[:3, 3].astype(np.float32)
        quat = _rotmat_to_xyzw(T[:3, :3]).astype(np.float32)
        out[canonical] = (pos, quat)
    return out


def _augment_upper_chain(kp: dict[str, tuple[NDArray, NDArray]]) -> None:
    """Synthesise ``spine`` / ``neck`` / ``head`` from available keypoints.

    These canonical joints are rarely present in a humanoid's ``ik_map`` (G1's
    torso is a single ``chest`` target) but a *target* robot may map them.  We
    fill them with geometrically plausible positions so the pipeline's
    ``missing canonical`` guard never trips; their tracking weight is typically
    low, so the approximation does not meaningfully degrade the solve.
    """
    def pos(name: str) -> NDArray | None:
        return kp[name][0] if name in kp else None

    chest_q = kp["chest"][1] if "chest" in kp else (
        kp["hips"][1] if "hips" in kp else _IDENTITY_Q
    )
    hips_p = pos("hips")
    chest_p = pos("chest")
    lsh = pos("left_shoulder")
    rsh = pos("right_shoulder")

    if hips_p is not None and chest_p is not None and "spine" not in kp:
        kp["spine"] = ((0.5 * (hips_p + chest_p)).astype(np.float32), chest_q)

    if chest_p is not None and "neck" not in kp:
        if lsh is not None and rsh is not None:
            neck_p = (0.5 * (lsh + rsh)).astype(np.float32)
        else:
            neck_p = (chest_p + 0.12 * _WORLD_UP).astype(np.float32)
        kp["neck"] = (neck_p, chest_q)

    if "neck" in kp and "head" not in kp:
        kp["head"] = ((kp["neck"][0] + 0.16 * _WORLD_UP).astype(np.float32), chest_q)


def _build_canonical_hierarchy(available: list[str]) -> Hierarchy:
    """Pruned canonical hierarchy keeping ``available`` joints (reparented)."""
    avail = set(available)
    if "hips" not in avail:
        raise ValueError(
            "source robot ik_map does not provide a 'hips' canonical joint; "
            "cannot build a canonical skeleton for robot-to-robot retarget."
        )
    names = [n for n in _CANONICAL_ORDER if n in avail]

    def nearest_parent(name: str) -> str | None:
        p = _CANONICAL_PARENTS[name]
        while p is not None and p not in avail:
            p = _CANONICAL_PARENTS[p]
        return p

    parent_names = [nearest_parent(n) for n in names]
    return Hierarchy.from_parent_names(names, parent_names)


# ---------------------------------------------------------------------------
# Source reference pose (zero-config FK) + trajectory → Motion
# ---------------------------------------------------------------------------


def build_source_reference_pose(source_model: URDFRobotModel) -> HumanReferencePose:
    """Reference skeleton for calibration: source robot FK at zero config.

    Joint names are *canonical* and ``source_to_canonical`` is left empty so
    the calibration math indexes directly by canonical name.  Positions are
    hips-relative, matching the human reference convention.
    """
    ik_pairs = _ik_map_pairs(source_model)
    saved = source_model.zero_configuration()
    try:
        link_tx = _collect_link_transforms_at_q(source_model, saved)
    finally:
        source_model.apply_configuration(saved)
    kp = _canonical_keypoints(link_tx, ik_pairs)
    _augment_upper_chain(kp)
    hier = _build_canonical_hierarchy(list(kp.keys()))
    names = list(hier.bone_names)

    hips_world = kp["hips"][0]
    pos = np.stack([kp[n][0] - hips_world for n in names], axis=0).astype(np.float32)
    quat = np.stack([kp[n][1] for n in names], axis=0).astype(np.float32)

    # Use the mesh-based standing height so it sits on the *same measurement
    # basis* as the target robot's ``model_height`` inside the scaler — this
    # keeps the root-trajectory scale ≈ 1.0 when source and target are similar
    # in size (and a true robot-size ratio otherwise), instead of mixing a
    # joint-extent source height with a mesh-extent target height.
    from hhtools.robot.standing_height import estimate_robot_standing_height

    try:
        height = float(estimate_robot_standing_height(source_model, saved))
    except Exception:
        height = 0.0
    if not np.isfinite(height) or height < 0.5:
        height = max(float(pos[:, 2].max() - pos[:, 2].min()), 0.5)
    return HumanReferencePose(
        name=f"robot_{source_model.preset.name}",
        root_joint="hips",
        joint_names=tuple(names),
        parent_names=tuple(hier.parent_names),
        positions=pos,
        quaternions=quat,
        source_to_canonical={},
        height_m=height,
    )


def source_trajectory_to_motion(
    source_model: URDFRobotModel,
    joint_q: NDArray,
    dof_names: tuple[str, ...],
    *,
    framerate: float,
    name: str = "robot_source",
    progress_callback=None,
) -> Motion:
    """Run source FK per frame → canonical-named :class:`Motion`.

    ``joint_q`` is ``(F, 7 + N)``: floating base ``(xyz + xyzw)`` then ``N``
    actuated DOFs aligned with ``dof_names``.
    """
    joint_q = np.asarray(joint_q, dtype=np.float32)
    if joint_q.ndim != 2 or joint_q.shape[1] < 8:
        raise ValueError(f"joint_q must be (F, 7+N); got {joint_q.shape}")
    num_frames = joint_q.shape[0]
    dof_names = tuple(dof_names)
    n_dof = len(dof_names)
    if joint_q.shape[1] != 7 + n_dof:
        raise ValueError(
            f"joint_q has {joint_q.shape[1]} columns but dof_names implies "
            f"{7 + n_dof} (7 root + {n_dof} dof)"
        )

    from hhtools.retarget.clip_ground_snap import (
        _foot_link_parts,
        _frame_min_foot_world_z,
    )

    ik_pairs = _ik_map_pairs(source_model)
    model_dof = set(source_model.dof_names())
    usable = [n for n in dof_names if n in model_dof]
    if not usable:
        raise ValueError(
            "none of the uploaded trajectory's DOF names match the source "
            f"robot {source_model.preset.name!r}; expected joints like "
            f"{sorted(model_dof)[:6]}…"
        )

    # Determine the canonical joint set / hierarchy from frame 0 (stable across
    # frames since the ik_map is fixed).
    saved = source_model.zero_configuration()
    try:
        cfg0 = {n: float(joint_q[0, 7 + i]) for i, n in enumerate(dof_names) if n in model_dof}
        link_tx0 = _collect_link_transforms_at_q(source_model, cfg0)
        kp0 = _canonical_keypoints(link_tx0, ik_pairs, T_root=_root_matrix(joint_q[0]))
        _augment_upper_chain(kp0)
        hier = _build_canonical_hierarchy(list(kp0.keys()))
        names = list(hier.bone_names)

        n_bones = len(names)
        positions = np.zeros((num_frames, n_bones, 3), dtype=np.float32)
        quaternions = np.zeros((num_frames, n_bones, 4), dtype=np.float32)
        quaternions[..., 3] = 1.0

        # The canonical skeleton stops at the ankles, so the downstream floor
        # heuristics would read the ankle joints as the contact plane and bury
        # the source robot's soles by one foot thickness.  ``ik_map``-driven FK
        # already poses the model here, so measure the real sole plane in the
        # same pass and declare it on the Motion.
        foot_parts = _foot_link_parts(source_model)
        sole_z_min: float | None = None

        for f in range(num_frames):
            cfg = {n: float(joint_q[f, 7 + i]) for i, n in enumerate(dof_names) if n in model_dof}
            link_tx = _collect_link_transforms_at_q(source_model, cfg)
            kp = _canonical_keypoints(link_tx, ik_pairs, T_root=_root_matrix(joint_q[f]))
            _augment_upper_chain(kp)
            for j, nm in enumerate(names):
                p, q = kp.get(nm, (positions[f, j], quaternions[f, j]))
                positions[f, j] = p
                quaternions[f, j] = q
            if foot_parts:
                sole_z = _frame_min_foot_world_z(
                    source_model, joint_q[f, :7], foot_parts=foot_parts,
                )
                if sole_z is not None and (sole_z_min is None or sole_z < sole_z_min):
                    sole_z_min = float(sole_z)
            if progress_callback is not None and (f == 0 or f == num_frames - 1 or f % 20 == 0):
                progress_callback(f + 1, num_frames)
    finally:
        source_model.apply_configuration(saved)

    meta: dict[str, object] = {"robot_to_robot_source": source_model.preset.name}
    if sole_z_min is not None:
        # A sole can only ever sit below the ankles that carry it, so a reading
        # above the ankle floor means the foot-mesh lookup produced garbage
        # (corrupt geometry, unresolved scene node).  Fall back to the ankle
        # heuristic rather than declaring a plane that would sink the clip.
        ankle_floor = float(foot_floor_z_in_positions(positions, tuple(names)))
        if sole_z_min <= ankle_floor + 1e-6:
            meta[SOURCE_FLOOR_META_KEY] = sole_z_min

    return Motion(
        name=name,
        hierarchy=hier,
        positions=positions,
        quaternions=quaternions,
        framerate=float(framerate),
        up_axis="Z",
        source_format="csv",
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Trajectory IO (csv / pkl / npz, with or without comment header)
# ---------------------------------------------------------------------------


@dataclass
class SourceTrajectory:
    """Parsed source robot trajectory."""

    joint_q: NDArray  # (F, 7 + N) — root (xyz + xyzw) then actuated DOFs
    dof_names: tuple[str, ...]
    framerate: float
    meta: dict


def _wxyz_to_xyzw(joint_q: NDArray) -> NDArray:
    out = np.asarray(joint_q, dtype=np.float32).copy()
    # root quat columns 3:7 stored as (w, x, y, z) → (x, y, z, w)
    w = out[:, 3].copy()
    out[:, 3:6] = out[:, 4:7]
    out[:, 6] = w
    return out


def _align_trajectory_dof_names(
    n_dof_cols: int,
    fallback_dof_names: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Map numeric CSV columns to joint names when the header is missing.

    Exports occasionally omit one trailing DOF column (or the header row).
    When ``fallback_dof_names`` comes from the source robot preset, prefer a
    prefix of that order over generic ``dof_0`` placeholders so FK / playback
    can resolve real joint names.
    """
    if not fallback_dof_names:
        return tuple(f"dof_{i}" for i in range(n_dof_cols))
    if len(fallback_dof_names) == n_dof_cols:
        return fallback_dof_names
    if len(fallback_dof_names) > n_dof_cols:
        return fallback_dof_names[:n_dof_cols]
    extra = tuple(
        f"dof_{i}" for i in range(len(fallback_dof_names), n_dof_cols)
    )
    return fallback_dof_names + extra


def _normalized_csv_column(name: str) -> str:
    text = str(name).strip().lower()
    for ch in (" ", "-", "(", ")", "[", "]", "{", "}"):
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _motiondecode_running_root_columns(header: list[str]) -> bool:
    return tuple(_normalized_csv_column(col) for col in header[:7]) == (
        "root_pos_x_m", "root_pos_y_m", "root_pos_z_m",
        "root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z",
    )


def _resolve_source_framerate(
    declared: float | None,
    source_fps: float | None,
) -> float:
    """Prefer file metadata; otherwise use ``source_fps`` or the R2R default."""
    if declared is not None and float(declared) > 0:
        return float(declared)
    if source_fps is not None and float(source_fps) > 0:
        return float(source_fps)
    return float(DEFAULT_SOURCE_FRAMERATE)


def _load_motiondecode_running_csv(
    path: Path,
    *,
    header: list[str],
    body_rows: list[list[str]],
    fallback_dof_names: tuple[str, ...] | None,
    source_fps: float | None = None,
) -> SourceTrajectory:
    """Convert MotionDecode-running CSV into the hhtools ``joint_q`` layout."""
    if body_rows:
        arr = np.asarray(body_rows, dtype=np.float64)
        joint_q = arr.astype(np.float32, copy=False)
        joint_q = _wxyz_to_xyzw(joint_q)
    else:
        joint_q = np.zeros((0, len(header)), dtype=np.float32)

    declared_dof_names = tuple(
        col[len("dof_"):].split("(", 1)[0].strip()
        for col in header[7:]
        if col.startswith("dof_")
    )
    n_dof_cols = max(joint_q.shape[1] - 7, 0)
    dof_names = (
        declared_dof_names
        if len(declared_dof_names) == n_dof_cols
        else _align_trajectory_dof_names(n_dof_cols, fallback_dof_names)
    )
    return SourceTrajectory(
        joint_q=joint_q,
        dof_names=dof_names,
        framerate=_resolve_source_framerate(None, source_fps),
        meta={"source_format": "motiondecode_running_csv", "root_quat_format": "wxyz"},
    )


def _load_header_only_robot_csv(
    path: Path,
    *,
    fallback_dof_names: tuple[str, ...] | None,
    source_fps: float | None = None,
) -> SourceTrajectory:
    """Load robot CSVs that include column names but omit ``time`` and comments."""
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        rows = [row for row in reader if row and any(str(cell).strip() for cell in row)]
    if not rows:
        raise ValueError(f"{path}: no rows found")

    header = [str(cell).strip() for cell in rows[0]]
    norm = [_normalized_csv_column(col) for col in header]
    root_aliases = (
        "root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw",
    )
    if _motiondecode_running_root_columns(header):
        return _load_motiondecode_running_csv(
            path,
            header=header,
            body_rows=rows[1:],
            fallback_dof_names=fallback_dof_names,
            source_fps=source_fps,
        )

    if tuple(norm[:7]) != root_aliases:
        raise ValueError(f"{path}: unsupported CSV header layout {header[:8]!r}")

    body_rows = rows[1:]
    if not body_rows:
        joint_q = np.zeros((0, len(header)), dtype=np.float32)
    else:
        arr = np.asarray(body_rows, dtype=np.float64)
        joint_q = arr.astype(np.float32, copy=False)

    declared_dof_names = tuple(
        col[len("dof_"):].split("(", 1)[0].strip()
        for col in header[7:]
        if col.startswith("dof_")
    )
    n_dof_cols = max(joint_q.shape[1] - 7, 0)
    dof_names = (
        declared_dof_names
        if len(declared_dof_names) == n_dof_cols
        else _align_trajectory_dof_names(n_dof_cols, fallback_dof_names)
    )

    return SourceTrajectory(
        joint_q=joint_q,
        dof_names=dof_names,
        framerate=_resolve_source_framerate(None, source_fps),
        meta={"source_format": "header_only_csv"},
    )


def _load_csv_trajectory(
    path: Path,
    *,
    fallback_dof_names: tuple[str, ...] | None,
    source_fps: float | None = None,
) -> SourceTrajectory:
    from hhtools.io.robot_csv import load_robot_csv

    try:
        csv = load_robot_csv(path)
        # hhtools CSV carries ``# sample_rate`` and/or a ``time`` column.
        return SourceTrajectory(
            joint_q=np.asarray(csv.joint_q, dtype=np.float32),
            dof_names=tuple(csv.dof_names),
            framerate=float(csv.sample_rate),
            meta=dict(csv.meta),
        )
    except ValueError:
        try:
            return _load_header_only_robot_csv(
                path,
                fallback_dof_names=fallback_dof_names,
                source_fps=source_fps,
            )
        except ValueError:
            pass
        # Header-less, comment-less numeric CSV: assume the column layout is
        # time + 7 root + N dof in the source robot's dof_order.
        rows: list[list[str]] = []
        with path.open("r", encoding="utf-8") as fp:
            for raw in fp:
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                rows.append(raw.split(","))
        if not rows:
            raise ValueError(f"{path}: no numeric rows found")
        arr = np.asarray(rows, dtype=np.float64)
        times = arr[:, 0]
        joint_q = arr[:, 1:].astype(np.float32)
        n_dof_cols = joint_q.shape[1] - 7
        dof_names = _align_trajectory_dof_names(n_dof_cols, fallback_dof_names)
        declared = (
            float(1.0 / max(times[1] - times[0], 1e-6))
            if times.shape[0] > 1
            else None
        )
        fps = _resolve_source_framerate(declared, source_fps)
        return SourceTrajectory(
            joint_q=joint_q, dof_names=dof_names, framerate=fps, meta={},
        )


def _extract_robot_trajectory_block(blob: object, *, path: Path | None = None) -> dict:
    """Pull the ``joint_q`` record from an hhtools robot-export pickle."""
    label = str(path) if path is not None else "pkl"
    if not isinstance(blob, dict):
        raise ValueError(f"{label}: expected a dict at pickle root")
    robot = blob.get("robot")
    if isinstance(robot, dict) and "joint_q" in robot:
        return robot
    if "joint_q" in blob:
        return blob
    keys = sorted(str(k) for k in blob.keys())
    raise ValueError(
        f"{label}: no robot joint_q trajectory (keys: {keys}); "
        "expected hhtools robot export with robot.joint_q"
    )


def _load_custom_robot_pkl_trajectory(
    robot: dict,
    *,
    path: Path,
    fallback_dof_names: tuple[str, ...] | None,
    source_fps: float | None = None,
) -> SourceTrajectory | None:
    """Parse lightweight robot clips with ``root_pos`` / ``root_rot`` / ``dof_pos``."""
    required = ("root_pos", "root_rot", "dof_pos")
    if not all(key in robot for key in required):
        return None
    root_pos = np.asarray(robot["root_pos"], dtype=np.float32)
    root_rot = np.asarray(robot["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(robot["dof_pos"], dtype=np.float32)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{path}: root_pos shape {root_pos.shape} is not (F, 3)")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"{path}: root_rot shape {root_rot.shape} is not (F, 4)")
    if dof_pos.ndim != 2:
        raise ValueError(f"{path}: dof_pos shape {dof_pos.shape} is not (F, N)")
    if root_pos.shape[0] != root_rot.shape[0] or root_pos.shape[0] != dof_pos.shape[0]:
        raise ValueError(
            f"{path}: root/dof frame counts differ "
            f"({root_pos.shape[0]}, {root_rot.shape[0]}, {dof_pos.shape[0]})"
        )
    dof_names = _align_trajectory_dof_names(dof_pos.shape[1], fallback_dof_names)
    joint_q = np.concatenate([root_pos, root_rot, dof_pos], axis=1).astype(np.float32, copy=False)
    declared = None
    for key in ("sample_rate", "fps", "framerate"):
        if key in robot and robot[key] is not None:
            declared = float(robot[key])
            break
    fps = _resolve_source_framerate(declared, source_fps)
    meta: dict[str, object] = {
        "source_format": "root_pos_root_rot_dof_pos",
    }
    link_body_list = robot.get("link_body_list")
    if isinstance(link_body_list, list):
        meta["link_body_list"] = [str(name) for name in link_body_list]
    return SourceTrajectory(
        joint_q=joint_q,
        dof_names=dof_names,
        framerate=fps,
        meta=meta,
    )


def _load_pkl_trajectory(
    path: Path,
    *,
    fallback_dof_names: tuple[str, ...] | None,
    source_fps: float | None = None,
) -> SourceTrajectory:
    with path.open("rb") as fp:
        blob = pickle.load(fp)
    if isinstance(blob, dict):
        custom = _load_custom_robot_pkl_trajectory(
            blob,
            path=path,
            fallback_dof_names=fallback_dof_names,
            source_fps=source_fps,
        )
        if custom is not None:
            return custom
    robot = _extract_robot_trajectory_block(blob, path=path)
    joint_q = np.asarray(robot["joint_q"], dtype=np.float32)
    dof_names = tuple(str(n) for n in robot.get("dof_names", ()))
    declared = None
    for key in ("sample_rate", "fps", "framerate"):
        if key in robot and robot[key] is not None:
            declared = float(robot[key])
            break
    fps = _resolve_source_framerate(declared, source_fps)
    if str(robot.get("root_quat_format", "xyzw")).lower() == "wxyz":
        joint_q = _wxyz_to_xyzw(joint_q)
    return SourceTrajectory(
        joint_q=joint_q, dof_names=dof_names, framerate=fps, meta=dict(robot.get("meta", {})),
    )


def _load_npz_trajectory(
    path: Path,
    *,
    fallback_dof_names: tuple[str, ...] | None,
    source_fps: float | None = None,
) -> SourceTrajectory:
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    jq_key = next((k for k in ("joint_q", "qpos", "q") if k in keys), None)
    if jq_key is not None:
        joint_q = np.asarray(data[jq_key], dtype=np.float32)
        if "dof_names" in keys:
            dof_names = tuple(str(n) for n in data["dof_names"].tolist())
        else:
            dof_names = _align_trajectory_dof_names(
                joint_q.shape[1] - 7, fallback_dof_names,
            )
        declared = None
        for k in ("sample_rate", "fps", "framerate"):
            if k in keys:
                declared = float(np.asarray(data[k]).reshape(-1)[0])
                break
        fps = _resolve_source_framerate(declared, source_fps)
        quat_fmt = "xyzw"
        if "root_quat_format" in keys:
            quat_fmt = str(data["root_quat_format"]).lower()
        if quat_fmt == "wxyz":
            joint_q = _wxyz_to_xyzw(joint_q)
        return SourceTrajectory(joint_q=joint_q, dof_names=dof_names, framerate=fps, meta={})

    isaaclab_keys = {"joint_names", "joint_pos", "base_pos_w", "base_quat_w"}
    if not isaaclab_keys.issubset(keys):
        raise ValueError(
            f"{path}: npz has neither joint_q/qpos/q nor the IsaacLab training-convention "
            f"schema (keys: {sorted(keys)})"
        )

    joint_names_raw = np.asarray(data["joint_names"])
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
    base_pos = np.asarray(data["base_pos_w"], dtype=np.float32)
    base_quat_wxyz = np.asarray(data["base_quat_w"], dtype=np.float32)
    if (
        joint_names_raw.ndim != 1
        or joint_pos.ndim != 2
        or base_pos.ndim != 2
        or base_quat_wxyz.ndim != 2
        or joint_pos.shape[0] < 1
        or joint_pos.shape[1] < 1
        or joint_names_raw.shape[0] != joint_pos.shape[1]
        or base_pos.shape != (joint_pos.shape[0], 3)
        or base_quat_wxyz.shape != (joint_pos.shape[0], 4)
    ):
        raise ValueError(
            f"{path}: invalid IsaacLab training-convention shapes: joint_names "
            f"{joint_names_raw.shape}, joint_pos {joint_pos.shape}, base_pos_w "
            f"{base_pos.shape}, base_quat_w {base_quat_wxyz.shape}"
        )

    def _joint_name(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    dof_names = tuple(_joint_name(name) for name in joint_names_raw.tolist())
    if not all(dof_names):
        raise ValueError(f"{path}: IsaacLab joint_names contains an empty name")
    # IsaacLab's training convention stores ``base_quat_w`` as wxyz; hhtools
    # uses xyzw in its 7 + N robot-trajectory layout.
    root_xyzw = np.concatenate(
        (base_pos, base_quat_wxyz[:, 1:], base_quat_wxyz[:, :1]), axis=1,
    )
    joint_q = np.concatenate((root_xyzw, joint_pos), axis=1)
    declared = None
    for k in ("sample_rate", "fps", "framerate"):
        if k in keys:
            declared = float(np.asarray(data[k]).reshape(-1)[0])
            break
    fps = _resolve_source_framerate(declared, source_fps)
    return SourceTrajectory(
        joint_q=joint_q,
        dof_names=dof_names,
        framerate=fps,
        meta={"source_format": "isaaclab_training_convention_npz"},
    )


def load_source_trajectory(
    path: str | Path,
    *,
    source_model: URDFRobotModel | None = None,
    source_fps: float | None = None,
) -> SourceTrajectory:
    """Load a robot trajectory exported in the hhtools schema.

    Supports ``.csv`` (with comment header, with header-only, or pure numeric),
    ``.pkl``, and ``.npz``.  When the file omits DOF names, the source robot's
    ``dof_order`` is used as a fallback.  Pure numeric CSV (no ``#`` comments,
    no column header) infers ``sample_rate`` from the first two ``time`` values.

    ``source_fps`` is used only when the file has no ``time`` / ``sample_rate``
    metadata (e.g. MotionDecode header-only CSV).  Defaults to
    :data:`DEFAULT_SOURCE_FRAMERATE` (50).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    fallback = tuple(source_model.dof_names()) if source_model is not None else None
    if suffix == ".csv":
        traj = _load_csv_trajectory(
            path, fallback_dof_names=fallback, source_fps=source_fps,
        )
    elif suffix in (".pkl", ".pickle"):
        traj = _load_pkl_trajectory(
            path, fallback_dof_names=fallback, source_fps=source_fps,
        )
    elif suffix == ".npz":
        traj = _load_npz_trajectory(
            path, fallback_dof_names=fallback, source_fps=source_fps,
        )
    else:
        raise ValueError(
            f"unsupported source trajectory format {suffix!r}; expected "
            f".csv / .pkl / .npz"
        )
    if traj.joint_q.ndim != 2 or traj.joint_q.shape[1] < 8:
        raise ValueError(
            f"{path}: parsed joint_q shape {traj.joint_q.shape} is not (F, 7+N)"
        )
    return traj


def trajectory_to_retargeted_motion(
    source_model: URDFRobotModel, traj: SourceTrajectory, *, name: str = "source",
) -> RetargetedMotion:
    """Wrap a parsed source trajectory as a :class:`RetargetedMotion`.

    Used purely to *visualise / play back* the uploaded source clip through the
    existing ``serialize_robot_trajectory`` path (no retarget involved).
    """
    return RetargetedMotion(
        name=name,
        joint_q=np.asarray(traj.joint_q, dtype=np.float32),
        sample_rate=float(traj.framerate),
        dof_names=tuple(traj.dof_names),
        root_coord_count=7,
        meta={"robot": source_model.preset.name},
    )


# ---------------------------------------------------------------------------
# Calibration IO (independent of the validated human-reference machinery)
# ---------------------------------------------------------------------------


def r2r_calibration_path(target_dir: str | Path, source_name: str) -> Path:
    safe = source_name.replace("/", "_").replace(":", "_")
    return Path(target_dir) / f"r2r_calibration_{safe}.yaml"


def save_r2r_calibration(
    target_dir: str | Path,
    *,
    target_robot: str,
    source_robot: str,
    calibrated_joint_q: dict[str, float],
) -> Path:
    import yaml

    path = r2r_calibration_path(target_dir, source_robot)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "robot_to_robot",
        "target_robot": target_robot,
        "source_robot": source_robot,
        "calibrated_joint_q": {
            k: float(v) for k, v in sorted(calibrated_joint_q.items())
        },
    }
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=True, default_flow_style=False)
    return path


def load_r2r_calibration(
    target_dir: str | Path, source_name: str
) -> dict[str, float] | None:
    import yaml

    path = r2r_calibration_path(target_dir, source_name)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    jq = data.get("calibrated_joint_q") or {}
    if not isinstance(jq, dict):
        return None
    return {str(k): float(v) for k, v in jq.items()}


# ---------------------------------------------------------------------------
# Scaler config + retarget
# ---------------------------------------------------------------------------


def _build_scaler_config(
    source_model: URDFRobotModel,
    target_model: URDFRobotModel,
    calibrated_joint_q: dict[str, float],
):
    from hhtools.retarget.calibration.calibration import (
        RobotRetargetCalibration,
        build_scaler_config_soma_style,
    )
    from hhtools.retarget.newton_basic.rest_pose import rest_pose_from_reference

    ref = build_source_reference_pose(source_model)
    rest_pose = rest_pose_from_reference(ref)
    identity_map = {n: n for n in ref.joint_names}
    cal = RobotRetargetCalibration(
        robot=target_model.preset.name,
        reference=f"robot_{source_model.preset.name}",  # type: ignore[arg-type]
        calibrated_joint_q={str(k): float(v) for k, v in calibrated_joint_q.items()},
        notes="robot-to-robot calibration",
    )
    cfg = build_scaler_config_soma_style(
        cal,
        target_model,
        rest_pose,
        src_to_canonical=identity_map,
        reference_pose=ref,
    )
    return cfg, ref


def _lowest_canonical_foot_z(
    bone_names: list[str] | tuple[str, ...],
    positions_frame: NDArray,
) -> float | None:
    """Lowest ankle/foot Z in one skeleton frame (same keys as the yellow overlay)."""

    def _norm(name: str) -> str:
        return str(name).lower().replace("_", "").replace(" ", "")

    name_to_i = {_norm(n): i for i, n in enumerate(bone_names)}
    zs: list[float] = []
    pos = np.asarray(positions_frame, dtype=np.float64)
    for key in (
        "leftankle",
        "rightankle",
        "leftfoot",
        "rightfoot",
        "leftleg",
        "rightleg",
    ):
        idx = name_to_i.get(key)
        if idx is not None and idx < pos.shape[0]:
            zs.append(float(pos[idx, 2]))
    return min(zs) if zs else None


def align_retargeted_ankles_to_scaled_source(
    target_model: URDFRobotModel,
    source_model: URDFRobotModel,
    source_motion: Motion,
    retargeted: RetargetedMotion,
    calibrated_joint_q: dict[str, float],
    *,
    frame_index: int = 0,
) -> RetargetedMotion:
    """Constant root-Z shift so target ankles match the scaled source overlay feet.

    Clip-floor-snap plants the **target mesh sole** on ``z=0``.  The yellow
    overlay uses source ankle keypoints after the declared sole-plane scale, so
    those sit a sole-thickness above the ground.  Planting the mesh on the
    overlay ankles (or snapping the overlay down to the mesh) leaves the robot
    floating above the yellow skeleton.  This undoes that offset without a
    second IK pass.
    """
    from dataclasses import replace

    from hhtools.viewer.anatomy import motion_has_interaction_scene

    if motion_has_interaction_scene(source_motion):
        return retargeted

    q = np.asarray(retargeted.joint_q, dtype=np.float32)
    if q.ndim != 2 or q.shape[0] == 0 or q.shape[1] < 3:
        return retargeted

    from hhtools.core.grounding import retarget_source_floor_z_world
    from hhtools.retarget.calibration.calibration import uniform_overlay_scale_for_motion

    cfg, ref = _build_scaler_config(source_model, target_model, calibrated_joint_q)
    ik_canons = (
        frozenset(target_model.preset.ik_map.keys())
        if target_model.preset.ik_map
        else frozenset()
    )
    ratio = float(
        uniform_overlay_scale_for_motion(
            cfg, float(ref.height_m), source_motion, ik_map_keys=ik_canons,
        )
    )
    names = list(source_motion.hierarchy.bone_names)
    src_pos = np.asarray(source_motion.positions, dtype=np.float64)
    f0 = int(np.clip(frame_index, 0, src_pos.shape[0] - 1))
    z_min = float(retarget_source_floor_z_world(source_motion))
    yellow_z = _lowest_canonical_foot_z(names, (src_pos[f0] - z_min) * ratio)
    if yellow_z is None:
        return retargeted

    from hhtools.web.serialize import (
        _apply_retarget_dof,
        _lowest_ankle_z,
        _quat_xyzw_to_rotmat,
    )

    f_ret = int(np.clip(f0, 0, q.shape[0] - 1))
    root = np.asarray(retargeted.root_trajectory[f_ret], dtype=np.float64)
    dof = np.asarray(retargeted.dof_trajectory[f_ret], dtype=np.float64)
    _apply_retarget_dof(target_model, list(retargeted.dof_names), dof)
    ik_map = dict(target_model.preset.ik_map) if target_model.preset.ik_map else {}
    ankle_local = _lowest_ankle_z(
        target_model, ik_map, _quat_xyzw_to_rotmat(root[3:7]),
    )
    if ankle_local is None:
        return retargeted

    delta = float(yellow_z) - (float(root[2]) + float(ankle_local))
    if abs(delta) < 1e-4:
        return retargeted

    out = q.copy()
    out[:, 2] = out[:, 2] + np.float32(delta)
    meta = dict(getattr(retargeted, "meta", {}) or {})
    meta["r2r_yellow_ankle_align_m"] = float(delta)
    return replace(retargeted, joint_q=out, meta=meta)


def suggested_r2r_backend(profile: str, *, has_scene: bool = False) -> str:
    """Default retarget backend for an R2R upload profile."""
    prof = (profile or "mimic").strip().lower()
    if prof in ("intermimic", "meshmimic") or has_scene:
        return "interaction_mesh"
    return "newton"


def retarget_robot_to_robot(
    source_model: URDFRobotModel,
    target_model: URDFRobotModel,
    *,
    calibrated_joint_q: dict[str, float],
    source_motion: Motion,
    backend: str = "newton",
    ik_iterations: int = 24,
    progress_callback=None,
) -> RetargetedMotion:
    """Retarget a canonical ``source_motion`` (from source FK) onto the target.

    ``calibrated_joint_q`` is the target robot's hand-aligned pose matching the
    source robot's rest skeleton (saved by the R2R calibration step).

    ``backend`` is ``"newton"`` (GPU IK) or ``"interaction_mesh"`` (MPC on
    terrain / interaction objects).  For the latter, attach scene data to
    ``source_motion`` before calling (see
    :func:`~hhtools.web.r2r_scene.attach_r2r_clip_scene_to_motion`).
    """
    cfg, ref = _build_scaler_config(source_model, target_model, calibrated_joint_q)
    reference_key = f"robot_{source_model.preset.name}"
    identity_map = {n: n for n in ref.joint_names}
    backend = (backend or "newton").strip().lower()

    if backend == "interaction_mesh":
        from hhtools.retarget.interaction_mesh.config import InteractionMeshPipelineConfig
        from hhtools.retarget.interaction_mesh.pipeline import InteractionMeshPipeline
        from hhtools.retarget.newton_basic.scaler import HumanToRobotScaler

        scaler = HumanToRobotScaler(
            source_motion.hierarchy, cfg, human_height=float(ref.height_m),
        )
        pipe = InteractionMeshPipeline(
            robot=target_model,
            scaler=scaler,
            cfg=InteractionMeshPipelineConfig(),
        )

        def _im_cb(stage: str, cur: int, tot: int) -> None:
            if progress_callback is None:
                return
            tot = max(1, tot)
            cur = max(0, min(cur, tot))
            if stage == "precompute":
                done = max(1, int(round(0.3 * cur)))
            else:
                done = max(1, int(round(0.3 * tot + 0.7 * cur)))
            progress_callback(done, tot)

        try:
            try:
                ret = pipe.run(source_motion, progress_callback=_im_cb)
            except TypeError:
                ret = pipe.run(source_motion)
        except ModuleNotFoundError as err:
            if "osqp" in str(err).lower():
                raise ValueError(
                    "interaction-mesh retarget needs the OSQP solver. "
                    "Install it with `uv pip install osqp` (or re-run "
                    "`uv sync --extra web`)."
                ) from err
            raise
        return align_retargeted_ankles_to_scaled_source(
            target_model,
            source_model,
            source_motion,
            ret,
            calibrated_joint_q,
        )

    from hhtools.retarget.newton_basic import NewtonBasicPipeline
    from hhtools.retarget.newton_basic._warp_config import configure as configure_warp
    from hhtools.robot.retarget_profile import (
        build_feet_stabilizer_config,
        build_pipeline_config_for_preset,
    )

    configure_warp()
    feet_cfg = build_feet_stabilizer_config(
        target_model.preset, reference_key, model=target_model,
    )
    pipeline = NewtonBasicPipeline(
        target_model,
        scaler_config=cfg,
        pipeline_config=build_pipeline_config_for_preset(
            target_model.preset, reference_key, ik_iterations=ik_iterations,
        ),
        feet_stabilizer_config=feet_cfg,
        human_height=float(ref.height_m),
        source_to_canonical=identity_map,
        configure_warp=False,
    )
    try:
        ret = pipeline.run(source_motion, progress_callback=progress_callback)
    except TypeError:
        ret = pipeline.run(source_motion)
    return align_retargeted_ankles_to_scaled_source(
        target_model,
        source_model,
        source_motion,
        ret,
        calibrated_joint_q,
    )
