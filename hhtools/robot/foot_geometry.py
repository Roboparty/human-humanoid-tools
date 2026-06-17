"""Foot geometry helpers for retarget (lateral separation, contact links)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hhtools.robot.loader import URDFRobotModel

_log = logging.getLogger(__name__)

__all__ = [
    "estimate_min_lateral_foot_separation",
    "foot_mesh_lateral_inner_gap",
    "resolve_leg_abduction_joints",
    "clamp_joint_q_foot_lateral_clearance",
]

_AXIS_TO_IDX = {"X": 0, "Y": 1, "Z": 2}

# Per-robot foot mesh node lists — avoids scanning the full URDF scene each call.
_FOOT_MESH_NODE_CACHE: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
_GEOM_VERTEX_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _lateral_axis_idx(preset) -> int:
    up = str(getattr(preset, "up_axis", "Z") or "Z").upper()
    fwd = str(getattr(preset, "forward_axis", "X") or "X").upper()
    up_i = _AXIS_TO_IDX.get(up, 2)
    fwd_i = _AXIS_TO_IDX.get(fwd, 0)
    remaining = {0, 1, 2} - {up_i}
    if fwd_i in remaining:
        remaining.discard(fwd_i)
    return sorted(remaining)[0] if remaining else 1


def _foot_contact_links(model: "URDFRobotModel") -> tuple[str | None, str | None]:
    preset = model.preset
    feet = dict(getattr(preset, "feet", None) or {})
    left = feet.get("left_contact_link")
    right = feet.get("right_contact_link")
    ik = dict(preset.ik_map or {})
    if not left:
        left = ik.get("left_ankle")
    if not right:
        right = ik.get("right_ankle")
    return (
        str(left) if left else None,
        str(right) if right else None,
    )


def _root_transform(root_xyzw: np.ndarray | None) -> np.ndarray:
    from hhtools.web.serialize import _quat_xyzw_to_rotmat

    T = np.eye(4, dtype=np.float64)
    if root_xyzw is None:
        return T
    root = np.asarray(root_xyzw, dtype=np.float64).reshape(-1)
    if root.size >= 7:
        T[:3, 3] = root[:3]
        T[:3, :3] = _quat_xyzw_to_rotmat(root[3:7])
    return T


def _root_lateral_direction(preset, root_xyzw: np.ndarray | None) -> np.ndarray:
    """World-space unit vector for the robot's coronal / lateral axis.

    When a floating-base orientation is supplied, the lateral axis follows
    the root frame so foot clearance stays correct after heading alignment.
    """
    from hhtools.web.serialize import _quat_xyzw_to_rotmat

    lat_i = _lateral_axis_idx(preset)
    axis = np.zeros(3, dtype=np.float64)
    axis[lat_i] = 1.0
    root = np.asarray(root_xyzw, dtype=np.float64).reshape(-1) if root_xyzw is not None else None
    if root is not None and root.size >= 7:
        lat = _quat_xyzw_to_rotmat(root[3:7]) @ axis
        norm = float(np.linalg.norm(lat))
        if norm > 1e-9:
            return lat / norm
    return axis


def _foot_mesh_node_parts(model: "URDFRobotModel", link: str) -> tuple[tuple[str, str], ...]:
    """Cached ``(scene_node, geom_name)`` pairs belonging to ``link``."""
    key = (str(model.preset.name), link)
    cached = _FOOT_MESH_NODE_CACHE.get(key)
    if cached is not None:
        return cached

    needle = link.replace("_link", "")
    parts: list[tuple[str, str]] = []
    scene = model.urdf.scene
    for node in scene.graph.nodes_geometry:
        if link not in node and needle not in node:
            continue
        _mat, geom_name = scene.graph[node]
        if geom_name:
            parts.append((node, str(geom_name)))
    cached = tuple(parts)
    _FOOT_MESH_NODE_CACHE[key] = cached
    return cached


def _cached_geom_vertices(model: "URDFRobotModel", geom_name: str) -> np.ndarray | None:
    import trimesh

    key = (str(model.preset.name), geom_name)
    if key in _GEOM_VERTEX_CACHE:
        return _GEOM_VERTEX_CACHE[key]
    geom = model.urdf.scene.geometry.get(geom_name)
    if not isinstance(geom, trimesh.Trimesh) or geom.is_empty:
        return None
    v = np.asarray(geom.vertices, dtype=np.float64)
    _GEOM_VERTEX_CACHE[key] = v
    return v


def _link_mesh_lateral_span(
    scene,
    link: str,
    *,
    lat_i: int | None = None,
    lat_vec: np.ndarray | None = None,
    T_world: np.ndarray | None = None,
    parts: tuple[tuple[str, str], ...] | None = None,
) -> tuple[float, float] | None:
    if lat_i is None and lat_vec is None:
        raise ValueError("lat_i or lat_vec is required")

    ys: list[float] = []
    T_world = np.eye(4, dtype=np.float64) if T_world is None else np.asarray(T_world, dtype=np.float64)
    lat_vec = None if lat_vec is None else np.asarray(lat_vec, dtype=np.float64).reshape(3)
    node_iter = parts if parts is not None else (
        (node, scene.graph[node][1])
        for node in scene.graph.nodes_geometry
        if link in node or link.replace("_link", "") in node
    )
    for node, geom_name in node_iter:
        if not geom_name:
            continue
        geom = scene.geometry.get(geom_name)
        if geom is None:
            continue
        v = np.asarray(getattr(geom, "vertices", ()), dtype=np.float64)
        if v.size == 0:
            continue
        mat = scene.graph.get(node)[0]
        ones = np.ones((v.shape[0], 1), dtype=np.float64)
        w = (T_world @ mat @ np.c_[v, ones].T).T[:, :3]
        if lat_vec is not None:
            ys.extend((w @ lat_vec).tolist())
        else:
            ys.extend(w[:, int(lat_i)].tolist())
    if not ys:
        return None
    return float(min(ys)), float(max(ys))


def _link_mesh_lateral_span_fast(
    model: "URDFRobotModel",
    link: str,
    *,
    lat_i: int | None = None,
    lat_vec: np.ndarray | None = None,
    T_world: np.ndarray | None = None,
) -> tuple[float, float] | None:
    """Like :func:`_link_mesh_lateral_span` but uses cached foot mesh nodes."""
    if lat_i is None and lat_vec is None:
        raise ValueError("lat_i or lat_vec is required")

    parts = _foot_mesh_node_parts(model, link)
    if not parts:
        return None

    ys: list[float] = []
    T_world = np.eye(4, dtype=np.float64) if T_world is None else np.asarray(T_world, dtype=np.float64)
    lat_vec = None if lat_vec is None else np.asarray(lat_vec, dtype=np.float64).reshape(3)
    scene = model.urdf.scene
    for node, geom_name in parts:
        v = _cached_geom_vertices(model, geom_name)
        if v is None:
            continue
        mat = scene.graph.get(node)[0]
        ones = np.ones((v.shape[0], 1), dtype=np.float64)
        w = (T_world @ mat @ np.c_[v, ones].T).T[:, :3]
        if lat_vec is not None:
            ys.extend((w @ lat_vec).tolist())
        else:
            ys.extend(w[:, int(lat_i)].tolist())
    if not ys:
        return None
    return float(min(ys)), float(max(ys))


def foot_mesh_lateral_inner_gap(
    model: "URDFRobotModel",
    joint_q: dict[str, float],
    root_xyzw: np.ndarray | None = None,
    *,
    lat_i: int | None = None,
) -> float | None:
    """Lateral inner clearance between foot meshes (left min − right max).

    Projects foot mesh vertices onto the robot's lateral axis.  When a
    floating-base orientation is provided the axis follows the root frame,
    so clearance stays meaningful after heading alignment.

    Positive values mean the feet are separated; negative values mean the
    inner edges overlap (penetration).
    """
    left_link, right_link = _foot_contact_links(model)
    if not left_link or not right_link:
        return None

    lat_vec = None
    if lat_i is None and root_xyzw is not None:
        lat_vec = _root_lateral_direction(model.preset, root_xyzw)
    if lat_i is None and lat_vec is None:
        lat_i = _lateral_axis_idx(model.preset)
    saved = model.zero_configuration()
    try:
        model.apply_configuration(joint_q)
        T_world = _root_transform(root_xyzw)
        span_kw: dict = {"T_world": T_world}
        if lat_vec is not None:
            span_kw["lat_vec"] = lat_vec
        else:
            span_kw["lat_i"] = int(lat_i)
        l_span = _link_mesh_lateral_span_fast(model, left_link, **span_kw)
        r_span = _link_mesh_lateral_span_fast(model, right_link, **span_kw)
    except Exception as exc:
        _log.debug(
            "foot mesh inner gap failed for %r: %s",
            model.preset.name,
            exc,
        )
        return None
    finally:
        model.apply_configuration(saved)

    if l_span is None or r_span is None:
        return None
    return float(l_span[0] - r_span[1])


def _ankle_lateral_separation(
    model: "URDFRobotModel",
    joint_q: dict[str, float],
    root_xyzw: np.ndarray | None,
    lat_i: int,
) -> float | None:
    ik = dict(model.preset.ik_map or {})
    left = ik.get("left_ankle")
    right = ik.get("right_ankle")
    if not left or not right:
        return None
    saved = model.zero_configuration()
    try:
        model.apply_configuration(joint_q)
        T_root = _root_transform(root_xyzw)
        lat_vec = _root_lateral_direction(model.preset, root_xyzw)
        ys: list[float] = []
        for link in (left, right):
            T = T_root @ np.asarray(model.urdf.get_transform(link), dtype=np.float64)
            ys.append(float(T[:3, 3] @ lat_vec))
        return float(ys[0] - ys[1])
    except Exception:
        return None
    finally:
        model.apply_configuration(saved)


def estimate_min_lateral_foot_separation(
    model: "URDFRobotModel",
    *,
    joint_q: dict[str, float] | None = None,
    margin_m: float = 0.015,
) -> float | None:
    """Minimum ankle-center lateral separation so foot meshes do not overlap.

    At ``joint_q`` (defaults to zero / T-pose), measure each foot link's
    mesh inner clearance and ankle separation, then return the smallest
    center-to-center distance that yields ``margin_m`` clearance between
    the inner foot edges.
    """
    q = dict(joint_q) if joint_q is not None else model.zero_configuration()
    lat_i = _lateral_axis_idx(model.preset)
    inner_gap = foot_mesh_lateral_inner_gap(model, q, lat_i=lat_i)
    ankle_sep = _ankle_lateral_separation(model, q, None, lat_i)
    if inner_gap is None or ankle_sep is None:
        return _estimate_min_lateral_foot_separation_legacy(model, q, margin_m=margin_m)

    ankle_span = float(abs(ankle_sep))

    # Feet already have clearance at ``joint_q``: use foot-width heuristic,
    # not hip-width ankle separation (which would over-constrain pre-IK).
    if float(inner_gap) >= float(margin_m):
        legacy = _estimate_min_lateral_foot_separation_legacy(
            model, q, margin_m=margin_m,
        )
        if legacy is not None:
            return legacy
        return ankle_span

    return float(ankle_span + (float(margin_m) - float(inner_gap)))


def _estimate_min_lateral_foot_separation_legacy(
    model: "URDFRobotModel",
    joint_q: dict[str, float],
    *,
    margin_m: float,
) -> float | None:
    left_link, right_link = _foot_contact_links(model)
    if not left_link or not right_link:
        return None

    saved = model.zero_configuration()
    lat_i = _lateral_axis_idx(model.preset)
    try:
        model.apply_configuration(joint_q)
        scene = model.trimesh_scene(collision=False)
        l_span = _link_mesh_lateral_span(scene, left_link, lat_i=lat_i)
        r_span = _link_mesh_lateral_span(scene, right_link, lat_i=lat_i)
    except Exception as exc:
        _log.debug(
            "foot lateral span failed for %r: %s",
            model.preset.name,
            exc,
        )
        return None
    finally:
        model.apply_configuration(saved)

    if l_span is None or r_span is None:
        return None

    left_w = l_span[1] - l_span[0]
    right_w = r_span[1] - r_span[0]
    if left_w <= 1e-4 or right_w <= 1e-4:
        return None

    return float((left_w + right_w) * 0.5 + margin_m)


def _joint_leg_side(name: str) -> str | None:
    """Infer left/right leg side from joint name (not roll/yaw axis tokens)."""
    low = str(name).lower()
    if low.startswith("l_") or low.startswith("left"):
        return "left"
    if low.startswith("r_") or low.startswith("right"):
        return "right"
    if "left" in low and "right" not in low:
        return "left"
    if "right" in low:
        return "right"
    return None


def resolve_leg_abduction_joints(model: "URDFRobotModel") -> tuple[str, str] | None:
    """Return (left, right) hip abduction / roll DOF names when inferable."""

    def _pick(side: str) -> str | None:
        cands: list[tuple[int, int, str]] = []
        for idx, name in enumerate(model.dof_names()):
            if _joint_leg_side(name) != side:
                continue
            low = str(name).lower()
            if "hip" not in low and "leg" not in low:
                continue
            if "knee" in low or "ankle" in low or "wrist" in low:
                continue
            if "roll" in low or "_r_" in low or "hip_r" in low:
                rank = 0
            elif "yaw" in low or "_y_" in low or "hip_y" in low:
                rank = 2
            else:
                rank = 1
            cands.append((rank, idx, str(name)))
        if not cands:
            return None
        cands.sort(key=lambda t: (t[0], t[1]))
        return cands[0][2]

    left = _pick("left")
    right = _pick("right")
    if left is None or right is None or left == right:
        return None
    return left, right


def _joint_q_row_to_cfg(
    joint_q_row: np.ndarray,
    dof_names: tuple[str, ...],
    *,
    root_coord_count: int,
) -> tuple[dict[str, float], np.ndarray | None]:
    row = np.asarray(joint_q_row, dtype=np.float64).reshape(-1)
    root = row[:root_coord_count] if row.size >= root_coord_count else None
    cfg: dict[str, float] = {}
    base = int(root_coord_count)
    for i, name in enumerate(dof_names):
        if base + i >= row.size:
            break
        cfg[str(name)] = float(row[base + i])
    return cfg, root


def _apply_abduction_delta(
    cfg: dict[str, float],
    left_joint: str,
    right_joint: str,
    left_delta: float,
    right_delta: float,
) -> dict[str, float]:
    out = dict(cfg)
    out[left_joint] = float(out.get(left_joint, 0.0) + left_delta)
    out[right_joint] = float(out.get(right_joint, 0.0) + right_delta)
    return out


def _abduction_joint_limits(
    model: "URDFRobotModel",
    left_joint: str,
    right_joint: str,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return ((l_lo, l_hi), (r_lo, r_hi)) limits for the abduction joints."""
    lo_hi: dict[str, tuple[float, float]] = {}
    for j in getattr(model, "actuated_joints", ()):  # type: ignore[attr-defined]
        lo = getattr(j, "limit_lower", None)
        hi = getattr(j, "limit_upper", None)
        if lo is None or hi is None or hi <= lo:
            continue
        lo_hi[str(j.name)] = (float(lo), float(hi))
    if left_joint not in lo_hi or right_joint not in lo_hi:
        return None
    return lo_hi[left_joint], lo_hi[right_joint]


def clamp_joint_q_foot_lateral_clearance(
    model: "URDFRobotModel",
    joint_q_row: np.ndarray,
    dof_names: tuple[str, ...],
    *,
    root_coord_count: int = 7,
    min_clearance_m: float = 0.01,
    ankle_prefilter_m: float | None = None,
    max_iterations: int = 12,
    step_rad: float = 0.02,
    max_abduction_rad: float = 0.20,
) -> np.ndarray:
    """Spread hip abduction until foot meshes have ``min_clearance_m`` inner gap.

    Triggered purely by actual foot-*mesh* interpenetration (``gap <
    min_clearance_m``): poses where the feet already clear — including
    wide-stance dance frames — are left untouched, so normal gait is
    unaffected.  Unlike earlier revisions this also corrects *crossed-ankle*
    frames (e.g. a narrow-hip robot whose scaled ankle targets cross during a
    leg-crossing dance move), iteratively abducting both hips a little at a
    time, respecting joint limits and a ``max_abduction_rad`` cap per leg so
    the correction stays local and does not snap into an unnatural splay.
    """
    if min_clearance_m <= 0.0:
        return np.asarray(joint_q_row, dtype=np.float32, copy=True)

    abduction = resolve_leg_abduction_joints(model)
    if abduction is None:
        return np.asarray(joint_q_row, dtype=np.float32, copy=True)

    left_joint, right_joint = abduction
    out = np.asarray(joint_q_row, dtype=np.float32, copy=True)
    cfg, root = _joint_q_row_to_cfg(out, dof_names, root_coord_count=root_coord_count)

    if ankle_prefilter_m is not None and ankle_prefilter_m > 0.0:
        lat_i = _lateral_axis_idx(model.preset)
        ankle_gap = _ankle_lateral_separation(model, cfg, root, lat_i)
        if ankle_gap is not None and float(ankle_gap) >= float(ankle_prefilter_m):
            return out

    gap = foot_mesh_lateral_inner_gap(model, cfg, root)
    if gap is None or gap >= min_clearance_m:
        return out

    # Determine which abduction sign spreads the feet apart for each leg by
    # probing both directions once; then iterate in that direction.  This
    # handles either left/right joint-axis convention.
    limits = _abduction_joint_limits(model, left_joint, right_joint)
    l_lo, l_hi = limits[0] if limits else (-1e9, 1e9)
    r_lo, r_hi = limits[1] if limits else (-1e9, 1e9)

    l0 = float(cfg.get(left_joint, 0.0))
    r0 = float(cfg.get(right_joint, 0.0))
    step = abs(float(step_rad))
    cap = abs(float(max_abduction_rad))
    n_iter = max(1, int(max_iterations))

    best_cfg = cfg
    best_gap = float(gap)
    for _ in range(n_iter):
        improved = False
        for l_delta, r_delta in ((step, -step), (-step, step)):
            l_new = best_cfg.get(left_joint, l0) + l_delta
            r_new = best_cfg.get(right_joint, r0) + r_delta
            # respect URDF limits and the per-leg correction cap
            if not (l_lo <= l_new <= l_hi and r_lo <= r_new <= r_hi):
                continue
            if abs(l_new - l0) > cap or abs(r_new - r0) > cap:
                continue
            trial = _apply_abduction_delta(best_cfg, left_joint, right_joint, l_delta, r_delta)
            trial_gap = foot_mesh_lateral_inner_gap(model, trial, root)
            if trial_gap is not None and trial_gap > best_gap + 1e-5:
                best_gap = float(trial_gap)
                best_cfg = trial
                improved = True
                break
        if not improved or best_gap >= min_clearance_m:
            break

    if best_cfg is cfg:
        return out

    base = int(root_coord_count)
    for i, name in enumerate(dof_names):
        if base + i >= out.size:
            break
        out[base + i] = np.float32(best_cfg.get(name, float(out[base + i])))

    return out
