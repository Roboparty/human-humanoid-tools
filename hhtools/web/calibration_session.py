"""Calibration session helpers for the web UI (parity with Viser ``_enter_calibration_mode``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from hhtools.core.grounding import foot_floor_z_in_positions
from hhtools.core.motion import Motion
from hhtools.robot.loader import URDFRobotModel


def joint_world_payload(model: URDFRobotModel) -> dict[str, dict[str, list[float]]]:
    """Per-joint pivot + rotation axis in the hhtools world frame (Z-up, metres).

    Used by the web UI to place drag handles and arcball rotation during
    interactive calibration.
    """
    out: dict[str, dict[str, list[float]]] = {}
    for j in model.actuated_joints:
        if j.joint_type in ("fixed", "floating", "planar"):
            continue
        try:
            T_parent = np.asarray(model.urdf.get_transform(j.parent_link), dtype=np.float64)
            T_child = np.asarray(model.urdf.get_transform(j.child_link), dtype=np.float64)
        except Exception:
            continue
        axis = np.asarray(j.axis, dtype=np.float64).reshape(3)
        nrm = float(np.linalg.norm(axis))
        if nrm < 1e-9:
            continue
        axis_w = (T_parent[:3, :3] @ axis) / nrm
        pivot = T_child[:3, 3]
        out[j.name] = {
            "pivot": pivot.astype(np.float32).tolist(),
            "axis": axis_w.astype(np.float32).tolist(),
        }
    return out


def _joint_limits_payload(model: URDFRobotModel) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for j in model.actuated_joints:
        if j.joint_type == "fixed":
            continue
        lo = float(j.limit_lower) if j.limit_lower is not None else -float(np.pi)
        hi = float(j.limit_upper) if j.limit_upper is not None else float(np.pi)
        if hi <= lo:
            lo, hi = -float(np.pi), float(np.pi)
        out.append({
            "name": j.name,
            "type": j.joint_type,
            "parent_link": j.parent_link,
            "child_link": j.child_link,
            "lower": lo,
            "upper": hi,
            "axis": list(j.axis),
        })
    return out


def _yaw_from_biacromial(left_pos, right_pos) -> float | None:
    shoulder = np.asarray(left_pos, dtype=np.float32) - np.asarray(right_pos, dtype=np.float32)
    fwd = np.cross(shoulder, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    fwd[2] = 0.0
    mag = float(np.linalg.norm(fwd))
    if mag < 1e-6:
        return None
    return float(np.arctan2(fwd[1] / mag, fwd[0] / mag))


def _reference_heading_rad(
    model: URDFRobotModel,
    ref,
    motion: Motion | None,
    ref_name: str,
    *,
    current_q: dict[str, float] | None = None,
) -> float:
    robot_fwd_yaw = 0.0
    robot_fwd_from_ik = False
    try:
        from hhtools.retarget.calibration.calibration import _collect_link_transforms_at_q

        q = dict(current_q) if current_q else model.zero_configuration()
        link_tx = _collect_link_transforms_at_q(model, q)
        ik_map = model.preset.ik_map or {}
        ls_link = ik_map.get("left_shoulder")
        rs_link = ik_map.get("right_shoulder")
        if isinstance(ls_link, dict):
            ls_link = ls_link.get("t_body") or ls_link.get("link")
        if isinstance(rs_link, dict):
            rs_link = rs_link.get("t_body") or rs_link.get("link")
        if ls_link and rs_link:
            T_ls = link_tx.get(str(ls_link))
            T_rs = link_tx.get(str(rs_link))
            if T_ls is not None and T_rs is not None:
                yaw = _yaw_from_biacromial(T_ls[:3, 3], T_rs[:3, 3])
                if yaw is not None:
                    robot_fwd_yaw = float(yaw)
                    robot_fwd_from_ik = True
    except Exception:
        pass
    if not robot_fwd_from_ik:
        fwd_axis = getattr(model.preset, "forward_axis", "X")
        axis_to_yaw = {"X": 0.0, "Y": np.pi / 2, "-X": np.pi, "-Y": -np.pi / 2}
        robot_fwd_yaw = axis_to_yaw.get(str(fwd_axis).upper(), 0.0)

    s2c = getattr(ref, "source_to_canonical", None) or {}
    can2native: dict[str, str] = {}
    for native, canonical in s2c.items():
        can2native.setdefault(canonical, native)
    jn = list(ref.joint_names)
    jset = frozenset(jn)

    def _ref_biacromial(c_left: str, c_right: str) -> float | None:
        ln = can2native.get(c_left)
        rn = can2native.get(c_right)
        if not ln or not rn or ln not in jset or rn not in jset:
            return None
        return _yaw_from_biacromial(ref.positions[jn.index(ln)], ref.positions[jn.index(rn)])

    ref_fwd_yaw = 0.0
    y_ref = _ref_biacromial("left_shoulder", "right_shoulder")
    if y_ref is None:
        y_ref = _ref_biacromial("left_hip", "right_hip")
    if y_ref is not None:
        ref_fwd_yaw = float(y_ref)
    return robot_fwd_yaw - ref_fwd_yaw


def serialize_reference_skeleton(
    ref,
    *,
    heading_rad: float = 0.0,
) -> dict[str, Any]:
    """Single-frame reference T-pose for three.js (blue overlay).

    Matches Viser ``ReferenceSkeletonRenderer``: yaw in XY, then foot-floor to
    z=0.  Robot mesh lift is applied only on the robot ``group`` (``ground_offset_z``),
    not baked into reference joint positions.
    """
    names = list(ref.joint_names)
    parent_names = list(ref.parent_names)
    name_to_i = {n: i for i, n in enumerate(names)}
    parents: list[int] = []
    for p in parent_names:
        parents.append(-1 if p is None else name_to_i.get(p, -1))

    pos = np.asarray(ref.positions, dtype=np.float32).reshape(-1, 3).copy()
    if pos.shape[0] != len(names):
        raise ValueError(
            f"reference positions ({pos.shape[0]} joints) != joint_names ({len(names)})"
        )

    if abs(heading_rad) > 1e-8:
        c, s = float(np.cos(heading_rad)), float(np.sin(heading_rad))
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        pos = (pos @ rot.T).astype(np.float32, copy=False)

    z_floor = float(foot_floor_z_in_positions(pos, tuple(names)))
    if abs(z_floor) > 1e-6:
        pos[:, 2] -= z_floor

    return {
        "bone_names": names,
        "parent_indices": parents,
        "positions": [pos.tolist()],
        "color": 0x5eb3ff,
    }


def _robot_ground_offset_z(model: URDFRobotModel, joint_q: dict[str, float] | None = None) -> float:
    """Lift so the lowest mesh vertex at ``joint_q`` rests on z=0."""
    from hhtools.web.serialize import _ground_offset_z

    try:
        if joint_q is not None:
            model.apply_configuration(joint_q)
        return float(_ground_offset_z(model.trimesh_scene()))
    except Exception:
        return 0.0


def _load_reference_pose_for_session(
    reference: str,
    motion: Motion | None,
):
    from hhtools.retarget.calibration.reference import build_motion_reference, load_reference_pose

    if reference == "glb":
        if motion is None or motion.num_frames == 0:
            raise ValueError("load a motion first for glb reference (frame 0)")
        return build_motion_reference(motion, "glb")
    return load_reference_pose(reference)


def build_calibration_session(
    model: URDFRobotModel,
    *,
    reference: str,
    motion: Motion | None,
) -> dict[str, Any]:
    """Payload for entering calibration mode in the browser."""
    from hhtools.retarget.calibration import load_calibration, resolve_calibration_file

    joint_order = [j.name for j in model.actuated_joints if j.joint_type != "fixed"]
    if not joint_order:
        raise ValueError("robot has no actuated joints; check URDF / upload")

    joint_q = {n: 0.0 for n in joint_order}
    urdf_parent = getattr(model.preset, "urdf_path", None)
    cal_path = None
    if urdf_parent is not None:
        cal_path = resolve_calibration_file(urdf_parent.parent, reference)
        if cal_path is not None:
            cal = load_calibration(cal_path)
            for name, value in cal.calibrated_joint_q.items():
                if name in joint_q:
                    joint_q[name] = float(value)

    ref = _load_reference_pose_for_session(reference, motion)
    if not getattr(ref, "joint_names", None):
        raise ValueError(f"reference pose {reference!r} has no joints")

    model.apply_configuration(joint_q)
    ground_z = _robot_ground_offset_z(model, joint_q)
    heading = _reference_heading_rad(
        model, ref, motion, reference, current_q=joint_q,
    )
    ref_payload = serialize_reference_skeleton(ref, heading_rad=heading)
    from hhtools.viewer.anatomy import detect_virtual_root

    exclude: list[int] = []
    if reference == "glb" and motion is not None and motion.num_bones > 0:
        if detect_virtual_root(list(motion.hierarchy.bone_names)):
            exclude.append(0)
    elif detect_virtual_root(list(ref.joint_names)):
        exclude.append(0)
    if exclude:
        ref_payload["exclude_joint_indices"] = exclude

    return {
        "joint_q": joint_q,
        "joint_limits": _joint_limits_payload(model),
        "reference": ref_payload,
        "reference_name": reference,
        "ground_offset_z": ground_z,
        "has_saved_calibration": cal_path is not None,
    }
