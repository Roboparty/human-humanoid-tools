# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""MPC-style windowing + per-frame interaction mesh / Laplacian precompute.

Full ``iterate_mpc`` + holosoma-style foot / penetration constraints will plug
into :mod:`hhtools.retarget.interaction_mesh.qp_step` once MuJoCo Jacobians are
wired.  This module already centralises **target Laplacian** construction from
scaled human + object samples so the MPC horizon can consume a pre-built list.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hhtools.retarget.interaction_mesh.laplacian_geometry import (
    calculate_laplacian_coordinates,
    create_interaction_mesh,
    get_adjacency_list,
)
from hhtools.retarget.interaction_mesh.motion_bridge import ScaledMotionScene
from hhtools.retarget.interaction_mesh.qp_step import OsqpUnreliableError

_log = logging.getLogger(__name__)

# Trust-region shrinkage applied when OSQP fails and the SQP falls
# back to a box-only L-BFGS-B solve.  At the default
# ``step_size = 0.2 rad`` this caps the per-iter |Δq| at 0.05 rad —
# small enough that ``smooth_weight`` can absorb it instead of leaving
# the multi-degree single-frame spikes the previous full-trust
# fallback was producing.  See ``sqp_step_laplacian`` docstring,
# section "Failure semantics".
OSQP_FALLBACK_TRUST_SHRINK = 0.25


def count_named_mujoco_bodies(model) -> int:
    """Count MuJoCo bodies (excluding world) that have a non-empty name."""
    import mujoco

    n = 0
    for bid in range(1, model.nbody):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid):
            n += 1
    return n


def _subsample_human_xyz_rows(h: NDArray[np.floating], nh: int) -> NDArray[np.float32]:
    """Pick ``nh`` rows from ``(J, 3)`` human joints (unique indices, roughly uniform)."""

    h = np.asarray(h, dtype=np.float32).reshape(-1, 3)
    j = int(h.shape[0])
    if j <= nh:
        return h.astype(np.float32, copy=False)
    idx = np.unique(np.linspace(0, j - 1, nh).round().astype(np.int64))
    if int(idx.size) < nh:
        idx = np.arange(min(nh, j), dtype=np.int64)
    return h[idx].astype(np.float32, copy=False)


def sample_axis_aligned_box(n: int, extents_xyz: NDArray[np.floating]) -> NDArray[np.float32]:
    """Approximately ``n`` points inside a centred box ``extents`` (full side lengths)."""
    ex = np.asarray(extents_xyz, dtype=np.float64).reshape(3)
    rng = np.random.default_rng(0)
    m = max(8, int(n))
    pts = rng.uniform(-0.5, 0.5, size=(m, 3)) * ex[None, :]
    return pts.astype(np.float32, copy=False)


@dataclass
class FrameLaplacianTarget:
    """One frame's Delaunay topology + target Laplacian coordinates."""

    adj_list: list[list[int]]
    target_laplacian: NDArray[np.float32]
    source_vertices: NDArray[np.float32]
    n_human_vertices: int
    # Source pelvis quaternion for this frame, stored as (qx, qy, qz, qw)
    # to match the rest of the codebase's xyzw convention.  Optional —
    # only populated when the precompute pipeline has access to the
    # source quaternions (the SMPL/SMPL-X path always does).  Read by
    # the SQP frame-0 base-orientation warm-start; leaving it ``None``
    # falls back to keeping whatever quaternion is in the freejoint at
    # solver entry.
    source_root_quat_xyzw: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class RobotMpcPoint:
    """One robot point used as an interaction-mesh vertex.

    ``body_name`` identifies the MuJoCo body, while ``local_offset`` is a
    body-frame point in metres.  The original coarse skeleton uses offset zero;
    contact-aware vertices use offsets derived from collision geometry.
    """

    body_name: str
    local_offset: NDArray[np.float64]
    semantic: str = ""
    source_index: int = -1
    weight: float = 1.0


def build_demo_vertices_frame(
    human_xyz: NDArray[np.floating],
    object_xyz: NDArray[np.floating] | None,
) -> NDArray[np.float32]:
    """Concatenate human joint rows (J,3) with optional object samples (No,3)."""
    h = np.asarray(human_xyz, dtype=np.float32).reshape(-1, 3)
    if object_xyz is None or object_xyz.size == 0:
        v = h
    else:
        o = np.asarray(object_xyz, dtype=np.float32).reshape(-1, 3)
        v = np.vstack([h, o])
    return v


def precompute_target_laplacians(
    scaled: ScaledMotionScene,
    *,
    object_extents: NDArray[np.floating] | list[NDArray[np.floating]] | None = None,
    object_samples: int = 24,
    max_human_vertices: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[FrameLaplacianTarget]:
    """Per-frame target Laplacian δ* from scaled human + optional object box samples.

    Args:
        scaled: Output of :func:`scale_motion_and_objects`.
        object_extents: ``(3,)`` cuboid **full side lengths** (metres). When
            ``scaled.object_positions`` is non-empty and this is ``None``, a
            ``0.6³`` placeholder is used (override with real ``SceneObject.extents``).
        object_samples: Number of random interior box samples per frame.
    max_human_vertices: When set, subsample human joint rows to this count so
        the Laplacian vertex count does not exceed the number of **named**
        MuJoCo bodies on the target robot (see :func:`iterate_mpc_rti`).
        progress_callback: Optional ``cb(frame_done, frame_total)`` for UI
        progress (throttled — at most ~40 calls per sequence).
    """
    F, _, _ = scaled.human_positions.shape
    out: list[FrameLaplacianTarget] = []

    obj_extents: list[NDArray[np.float64]] = []
    if scaled.object_positions:
        if object_extents is None:
            obj_extents = [
                np.array([0.6, 0.6, 0.6], dtype=np.float64)
                for _ in scaled.object_positions
            ]
        elif isinstance(object_extents, list):
            obj_extents = [
                np.asarray(ext, dtype=np.float64).reshape(3)
                for ext in object_extents
            ]
        else:
            ext_arr = np.asarray(object_extents, dtype=np.float64).reshape(3)
            obj_extents = [ext_arr for _ in scaled.object_positions]

    notify_stride = max(1, F // 40)

    # ----------------------------------------------------------------
    # Build vertex sets for every frame first, then tetrahedralise
    # **once** over an aggregate point cloud and reuse the resulting
    # adjacency list for every per-frame target Laplacian.
    #
    # Why a shared adjacency: ``create_interaction_mesh`` runs a
    # Delaunay tetrahedralisation whose output topology depends on
    # the precise vertex coordinates.  When the actor walks, the
    # Delaunay neighbourhood of a given joint (e.g. a hand near a
    # terrain bump in frame 100, then near a different bump in frame
    # 105) flips between frames — a single neighbour swap changes
    # the per-frame ``target_laplacian`` discontinuously, and the
    # SQP downstream reflects that as a step in ``qpos``.  In
    # quantitative terms this is observed as multi-degree per-frame
    # ``|Δq|`` spikes that no amount of ``smooth_weight`` can absorb
    # because the residual being smoothed is itself discontinuous.
    #
    # Holosoma's reference design tetrahedralises once over a
    # representative frame and locks the topology for the whole
    # clip; we go one step further and union vertices from every
    # frame so the adjacency captures every "the actor stood here at
    # some point" relationship.  The per-frame target Laplacian
    # then varies smoothly because the **same** adjacency operator
    # is applied to every frame's vertices.
    # ----------------------------------------------------------------
    per_frame_verts: list[NDArray[np.float32]] = []
    nh_per_frame: list[int] = []
    for f in range(F):
        h = scaled.human_positions[f]
        if max_human_vertices is not None and int(h.shape[0]) > int(max_human_vertices):
            h = _subsample_human_xyz_rows(h, int(max_human_vertices))
        obj_samples: list[NDArray[np.float32]] = []
        if scaled.object_points is not None:
            for pts_traj in scaled.object_points:
                if f < int(pts_traj.shape[0]) and int(pts_traj.shape[1]) > 0:
                    obj_samples.append(pts_traj[f].astype(np.float32, copy=False))
        else:
            for i, obj_traj in enumerate(scaled.object_positions):
                if i >= len(obj_extents):
                    continue
                pts = sample_axis_aligned_box(object_samples, obj_extents[i])
                obj_samples.append((pts + obj_traj[f]).astype(np.float32, copy=False))
        o_world = np.vstack(obj_samples) if obj_samples else None
        verts = build_demo_vertices_frame(h, o_world)
        if verts.shape[0] < 4:
            rng = np.random.default_rng(1000 + f)
            extra = rng.normal(scale=1e-3, size=(4 - verts.shape[0], 3)).astype(np.float32)
            verts = np.vstack([verts, extra])
        per_frame_verts.append(verts)
        nh_per_frame.append(int(h.shape[0]))

    # Use a representative frame for the shared Delaunay topology.
    # The vertex layout (role order: ``nh`` human joints followed
    # by object/terrain samples) is identical across frames, so the
    # adjacency indices computed from any one frame remain valid
    # for every other frame.  We pick the **middle** frame: the
    # actor is most likely to be in a generic configuration there,
    # which yields a tetrahedralisation whose neighbour relations
    # describe the clip as a whole better than a possibly atypical
    # T-pose at frame 0.
    V = int(per_frame_verts[0].shape[0])
    pivot_idx = F // 2
    pivot_pts = per_frame_verts[pivot_idx].astype(np.float64, copy=True)
    # Sub-millimetre isotropic perturbation eliminates any
    # accidentally-coplanar groups (terrain grid + skeletal
    # symmetries) that would otherwise produce a degenerate hull.
    rng = np.random.default_rng(0)
    pivot_pts = pivot_pts + rng.normal(scale=1e-5, size=pivot_pts.shape)
    _, tet = create_interaction_mesh(pivot_pts)
    adj = get_adjacency_list(tet, V)

    for f in range(F):
        verts = per_frame_verts[f]
        target = calculate_laplacian_coordinates(verts, adj, uniform_weight=True)
        out.append(
            FrameLaplacianTarget(
                adj_list=adj,
                target_laplacian=target,
                source_vertices=verts,
                n_human_vertices=nh_per_frame[f],
            )
        )
        if progress_callback is not None and (f % notify_stride == 0 or f == F - 1):
            try:
                progress_callback(f + 1, F)
            except Exception:
                pass
    return out


def _mj_body_names_prefix(model, nh: int) -> list[str]:
    """First ``nh`` named bodies (skip world), MuJoCo body id order."""
    import mujoco

    names: list[str] = []
    for bid in range(1, model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not nm:
            continue
        names.append(nm)
        if len(names) >= nh:
            break
    if len(names) < nh:
        raise ValueError(f"need {nh} named bodies for Laplacian, found {len(names)}")
    return names


def _stack_vertex_jacobians(
    model,
    data,
    robot_points: list[RobotMpcPoint],
    nq: int,
) -> NDArray[np.float64]:
    """``(3 * len(robot_points), nq)`` translational Jacobians at each point.

    ``mj_forward`` + ``build_T_qdot_to_qpos`` only depend on the current
    ``qpos`` (kinematics + the FREE-joint quaternion block of ``T``), so
    they need to run **once per call**, not once per point.  The previous
    implementation ran them ``len(robot_points)`` times via
    :func:`jacobian_translation_wrt_qpos`, which dominated profile in
    high-DOF SQP loops.  We now do one ``mj_forward`` and one ``T``
    construction up front, then loop over points calling ``mj_jac``
    only.  Mirrors holosoma's ``_calc_lap_foot_jacobians_batch``.
    """
    import mujoco

    from hhtools.retarget.interaction_mesh.mujoco_jacobians import (
        body_id_or_raise,
        build_T_qdot_to_qpos,
    )

    nv = model.nv
    mujoco.mj_forward(model, data)
    T = build_T_qdot_to_qpos(model, data)

    Jp = np.zeros((3, nv), dtype=np.float64, order="C")
    Jr = np.zeros((3, nv), dtype=np.float64, order="C")

    rows: list[NDArray[np.float64]] = []
    for pt in robot_points:
        bid = body_id_or_raise(model, pt.body_name)
        off = np.asarray(pt.local_offset, dtype=np.float64).reshape(3)
        R = data.xmat[bid].reshape(3, 3)
        p_w = (data.xpos[bid].astype(np.float64) + R @ off).reshape(3)
        Jp.fill(0.0)
        Jr.fill(0.0)
        mujoco.mj_jac(model, data, Jp, Jr, p_w, int(bid))
        rows.append(Jp @ T)
    return np.vstack(rows).astype(np.float64, copy=False)


def _robot_points_from_body_names(body_names: list[str]) -> list[RobotMpcPoint]:
    return [
        RobotMpcPoint(
            body_name=nm,
            local_offset=np.zeros(3, dtype=np.float64),
            semantic=nm,
        )
        for nm in body_names
    ]


def _current_robot_point_positions(
    model,
    data,
    robot_points: list[RobotMpcPoint],
) -> NDArray[np.float64]:
    from hhtools.retarget.interaction_mesh.mujoco_jacobians import body_id_or_raise

    out: list[NDArray[np.float64]] = []
    for pt in robot_points:
        bid = body_id_or_raise(model, pt.body_name)
        R = data.xmat[bid].reshape(3, 3).astype(np.float64)
        off = np.asarray(pt.local_offset, dtype=np.float64).reshape(3)
        out.append(data.xpos[bid].astype(np.float64) + R @ off)
    return np.vstack(out).astype(np.float64, copy=False)


def _normalize_free_joint_quat(model, qpos: NDArray[np.floating]) -> None:
    import mujoco

    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        return
    qadr = int(model.jnt_qposadr[0])
    qq = q[qadr + 3 : qadr + 7]
    n = float(np.linalg.norm(qq))
    if n > 1e-12:
        qq[:] = qq / n


def _build_joint_limits(model) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract per-qpos joint limits from the MuJoCo model.

    Returns ``(q_lb, q_ub)`` each of shape ``(nq,)``.  Free-joint DOFs and
    unlimited hinges get large bounds (±1e6).
    """
    import mujoco

    nq = model.nq
    q_lb = np.full(nq, -1e6, dtype=np.float64)
    q_ub = np.full(nq, 1e6, dtype=np.float64)
    for j in range(model.njnt):
        jtype = int(model.jnt_type[j])
        qadr = int(model.jnt_qposadr[j])
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            continue
        has_limit = bool(model.jnt_limited[j])
        if has_limit:
            q_lb[qadr] = float(model.jnt_range[j, 0])
            q_ub[qadr] = float(model.jnt_range[j, 1])
    return q_lb, q_ub


def _compute_hard_nonpenetration_rows(
    collision_model,
    collision_data,
    qpos: NDArray[np.float64],
    *,
    threshold: float = 0.05,
    tolerance: float = 0.002,
    fd_epsilon: float = 1e-5,
    max_pairs_per_body: int = 4,
) -> tuple[list[NDArray[np.float64]], list[float]]:
    """Mirror of holosoma's hard non-penetration linearisation.

    Wraps :func:`hhtools.retarget.interaction_mesh.collision.compute_nonpenetration_constraints`
    so that the SQP can call it with a single MuJoCo (model, data)
    pair carrying terrain meshes alongside the robot links.  The
    returned ``J_rows`` / ``rhs`` are ready to feed into OSQP as
    ``A_np · δq ≥ rhs`` rows.

    Each row corresponds to one robot↔scene geom pair whose signed
    distance (from ``mj_geomDistance``) was at most ``threshold`` at
    the linearisation point.  The constraint is **non-violable** —
    whatever the cost gradient suggests, OSQP must keep the body above
    the terrain.  This is the only mechanism strong enough to oppose a
    translation-invariant Laplacian cost driving the floating base
    into the ground.

    ``max_pairs_per_body`` caps inequality rows per
    ``(robot_body, scene_geom)`` bucket.  URDF feet that compile to many
    sub-meshes (RP1 right_ankle_roll_link → 24+ collision primitives,
    G1 ankles similar) generate near-duplicate rows whose witness points
    and normals oscillate frame-to-frame as the foot crosses heightfield
    cell boundaries; left uncapped this chatter (53 rows/frame on
    holosoma parkour_1) is the primary source of OSQP infeasibility on
    contact frames, which then triggers the SQP fallback path and
    surfaces as single-frame pose spikes.  Default ``4`` matches the
    empirical sweep noted in
    :func:`compute_nonpenetration_constraints`'s docstring.
    """
    from hhtools.retarget.interaction_mesh.collision import compute_nonpenetration_constraints

    return compute_nonpenetration_constraints(
        collision_model, collision_data, qpos,
        threshold=float(threshold),
        tolerance=float(tolerance),
        fd_epsilon=float(fd_epsilon),
        max_pairs_per_body=int(max_pairs_per_body),
    )


def resolve_foot_mpc_points(
    robot_points: list[RobotMpcPoint] | None,
    nh: int,
) -> tuple[RobotMpcPoint | None, RobotMpcPoint | None]:
    """Pick left/right foot MPC points for holosoma-style foot sticking."""

    if not robot_points:
        return None, None

    def _pick(side: str) -> RobotMpcPoint | None:
        cands: list[tuple[int, RobotMpcPoint]] = []
        for pt in robot_points:
            sem = str(getattr(pt, "semantic", "")).lower()
            if side not in sem:
                continue
            if not any(k in sem for k in ("foot", "ankle", "toe")):
                continue
            score = 1
            if ":toe" in sem:
                score = 3
            elif sem.endswith("_foot") or sem == f"{side}_foot":
                score = 2
            cands.append((score, pt))
        if not cands:
            return None
        cands.sort(key=lambda x: -x[0])
        return cands[0][1]

    return _pick("left"), _pick("right")


def _mpc_point_vertex_index(
    mpc_points: list[RobotMpcPoint],
    nh: int,
    pt: RobotMpcPoint | None,
) -> int | None:
    """Index of ``pt`` within ``mpc_points[:nh]``, or ``None`` if not in the Laplacian mesh."""
    if pt is None:
        return None
    for i, p in enumerate(mpc_points[:nh]):
        if p.body_name != pt.body_name:
            continue
        if np.allclose(p.local_offset, pt.local_offset, atol=1e-9):
            return i
    return None


def extract_foot_sticking_sequences(
    targets: list[FrameLaplacianTarget],
    robot_points: list[RobotMpcPoint] | None,
    *,
    velocity_threshold: float = 0.01,
) -> list[dict[str, bool]]:
    """Contact flags from source-foot XY velocity (holosoma ``extract_foot_sticking_sequence_velocity``)."""

    n = len(targets)
    if n == 0:
        return []

    left_vi = right_vi = None
    if robot_points:
        for i, pt in enumerate(robot_points):
            sem = str(getattr(pt, "semantic", "")).lower()
            if sem == "left_foot" or (
                left_vi is None and "left" in sem and "foot" in sem and ":" not in sem
            ):
                left_vi = i
            if sem == "right_foot" or (
                right_vi is None and "right" in sem and "foot" in sem and ":" not in sem
            ):
                right_vi = i

    left_xy = right_xy = None
    if left_vi is not None:
        left_xy = np.stack(
            [t.source_vertices[left_vi, :2] for t in targets], axis=0,
        )
    if right_vi is not None:
        right_xy = np.stack(
            [t.source_vertices[right_vi, :2] for t in targets], axis=0,
        )

    out: list[dict[str, bool]] = []
    for i in range(n):
        l_stick = False
        r_stick = False
        if left_xy is not None and i > 0:
            l_stick = bool(
                np.linalg.norm(left_xy[i] - left_xy[i - 1]) <= float(velocity_threshold)
            )
        if right_xy is not None and i > 0:
            r_stick = bool(
                np.linalg.norm(right_xy[i] - right_xy[i - 1]) <= float(velocity_threshold)
            )
        out.append({"L_Foot": l_stick, "R_Foot": r_stick})
    return out


def stabilize_foot_sticking_sequences(
    seq: list[dict[str, bool]],
    *,
    release_hysteresis: int = 0,
) -> list[dict[str, bool]]:
    """Hysteresis on foot release — avoids one-frame contact drops that jitter legs."""
    if not seq or release_hysteresis <= 0:
        return seq
    keys = list(seq[0].keys())
    out: list[dict[str, bool]] = [dict(seq[0])]
    hold = {k: 0 for k in keys}
    for i in range(1, len(seq)):
        row: dict[str, bool] = {}
        for k in keys:
            if seq[i].get(k, False):
                hold[k] = int(release_hysteresis)
                row[k] = True
            elif hold[k] > 0:
                hold[k] -= 1
                row[k] = True
            else:
                row[k] = False
        out.append(row)
    return out


def _leg_actuated_qpos_indices(model) -> NDArray[np.int64]:
    """qpos rows for leg hinges (hip / knee / ankle and common aliases)."""
    import mujoco

    keys = ("hip", "knee", "ankle", "thigh", "calf", "leg", "shank")
    idx: list[int] = []
    for j in range(model.njnt):
        jt = int(model.jnt_type[j])
        if jt not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        nl = name.lower()
        if any(k in nl for k in keys):
            idx.append(int(model.jnt_qposadr[j]))
    return np.asarray(idx, dtype=np.int64)


def causal_smooth_actuated_qpos(
    traj: NDArray[np.float64],
    actuated_idx: NDArray[np.int64],
    *,
    beta: float,
) -> NDArray[np.float64]:
    """``q[t] ← (1−β)·q[t] + β·q[t−1]`` on selected DOFs (causal leg low-pass)."""
    if traj.shape[0] < 2 or actuated_idx.size == 0 or beta <= 0.0:
        return traj
    b = float(np.clip(beta, 0.0, 0.95))
    out = np.asarray(traj, dtype=np.float64).copy()
    idx = actuated_idx
    for t in range(1, out.shape[0]):
        out[t, idx] = (1.0 - b) * out[t, idx] + b * out[t - 1, idx]
    return out


def _build_foot_sticking_rows(
    model,
    data,
    q_work: NDArray[np.float64],
    q_ref: NDArray[np.float64],
    *,
    left_foot_pt: RobotMpcPoint | None,
    right_foot_pt: RobotMpcPoint | None,
    foot_sticking: dict[str, bool] | None,
    tolerance: float,
    nq: int,
    J_r: NDArray[np.float64] | None = None,
    left_vi: int | None = None,
    right_vi: int | None = None,
) -> tuple[list[NDArray[np.float64]], list[float], list[float]]:
    """Hard foot XY rows: ``foot_lb ≤ J_xy δq ≤ foot_ub`` (holosoma foot sticking)."""
    import mujoco

    if not foot_sticking:
        return [], [], []

    left_key = right_key = None
    for key, active in foot_sticking.items():
        if not active:
            continue
        kl = key.lower()
        if kl.startswith("l"):
            left_key = key
        elif kl.startswith("r"):
            right_key = key

    entries: list[tuple[RobotMpcPoint, str, int | None]] = []
    if left_key is not None and left_foot_pt is not None and foot_sticking.get(left_key, False):
        entries.append((left_foot_pt, "left", left_vi))
    if right_key is not None and right_foot_pt is not None and foot_sticking.get(right_key, False):
        entries.append((right_foot_pt, "right", right_vi))
    if not entries:
        return [], [], []

    saved = np.asarray(data.qpos, dtype=np.float64).copy()
    tol = float(tolerance)

    data.qpos[:] = q_ref
    mujoco.mj_forward(model, data)
    ref_xy: dict[str, NDArray[np.float64]] = {}
    for pt, side, _ in entries:
        ref_xy[side] = _current_robot_point_positions(model, data, [pt])[0, :2].copy()

    data.qpos[:] = q_work
    mujoco.mj_forward(model, data)

    j_rows: list[NDArray[np.float64]] = []
    lbs: list[float] = []
    ubs: list[float] = []
    for pt, side, vi in entries:
        p_cur = _current_robot_point_positions(model, data, [pt])[0, :2]
        if J_r is not None and vi is not None:
            j_xy = J_r[3 * vi : 3 * (vi + 1), :][:2]
        else:
            j_xy = _stack_vertex_jacobians(model, data, [pt], nq).reshape(3, nq)[:2]
        delta = ref_xy[side] - p_cur
        lb = delta - tol
        ub = lb + 2.0 * tol
        for d in range(2):
            j_rows.append(j_xy[d].astype(np.float64, copy=False))
            lbs.append(float(lb[d]))
            ubs.append(float(ub[d]))

    data.qpos[:] = saved
    return j_rows, lbs, ubs


def sqp_step_laplacian(
    model,
    data,
    qpos: NDArray[np.floating],
    frame: FrameLaplacianTarget,
    body_names: list[str],
    *,
    robot_points: list[RobotMpcPoint] | None = None,
    laplacian_weight: float,
    step_size: float,
    sqp_inner_iters: int = 2,
    q_prev: NDArray[np.floating] | None = None,
    smooth_weight: float = 0.6,
    q_lb: NDArray[np.float64] | None = None,
    q_ub: NDArray[np.float64] | None = None,
    # --- absolute world-position cost (anchors global root motion) ---
    position_weight: float = 0.0,
    # --- home-pose Tikhonov on actuated joints ---
    home_pose_weight: float = 0.0,
    home_qpos: NDArray[np.float64] | None = None,
    actuated_qpos_idx: NDArray[np.int64] | None = None,
    # --- collision (holosoma-style hard non-penetration) ---
    collision_model=None,
    collision_data=None,
    collision_threshold: float = 0.05,
    penetration_tolerance: float = 0.002,
    collision_fd_epsilon: float = 1e-5,
    collision_max_pairs_per_body: int = 4,
    # --- holosoma-style foot sticking (hard XY inequalities) ---
    activate_foot_sticking: bool = False,
    foot_sticking: dict[str, bool] | None = None,
    q_foot_ref: NDArray[np.floating] | None = None,
    left_foot_pt: RobotMpcPoint | None = None,
    right_foot_pt: RobotMpcPoint | None = None,
    foot_sticking_tolerance: float = 1e-3,
    leg_actuated_qpos_idx: NDArray[np.int64] | None = None,
    leg_smooth_weight: float = 1.0,
    leg_sqp_step_scale: float = 1.0,
    # --- trust region ---
    base_step_size: float | None = None,
) -> NDArray[np.float64]:
    """One SQP frame solve with Laplacian + smoothness cost.

    Holosoma-style **hard non-penetration** is the only collision
    mechanism: when ``collision_model`` is provided, ``mj_geomDistance``
    on every robot↔scene geom pair within ``collision_threshold``
    produces an inequality row ``J · δq ≥ −φ − tol``.  These rows are
    fed straight to OSQP alongside the box trust region — exactly the
    construction in
    ``holosoma_retargeting.src.interaction_mesh_retargeter.solve_mpc_iteration``.

    The hard constraint is the only mechanism that can keep the body
    above the terrain when the cost is translation-invariant: any
    candidate ``δq`` that drops the foot below the terrain is *outside*
    the feasible set, so OSQP cannot return it regardless of how much
    the Laplacian cost would otherwise reward it.

    Failure semantics
    -----------------
    OSQP can occasionally fail (``MAX_ITER_REACHED`` /
    ``PRIMAL_INFEASIBLE`` / …) on contact-rich frames where chattery
    non-penetration rows make the KKT system stiff.  Two extreme
    fallbacks are both wrong:

    1. Silent box-only L-BFGS-B with the *same* trust region drops
       every inequality row and lets a single iter take a 0.2 rad /
       7 cm step — the multi-degree single-frame pose spikes seen on
       holosoma parkour clips.
    2. Returning ``q_prev`` unchanged risks the entire clip locking
       up: if frame 0 fails, every later frame warm-starts from the
       same qpos and can fail for the same reason, producing the
       "OMOMO / holosoma robot doesn't move" failure mode.

    The chosen middle path keeps the SQP making progress while
    bounding the per-iter step:

        OsqpUnreliableError ⇒
            box-only L-BFGS-B with trust region scaled by
            ``OSQP_FALLBACK_TRUST_SHRINK`` (currently 1/4).

    With ``step_size=0.2`` the fallback step is at most ``0.05`` rad
    (and ``base_step_size`` shrinks proportionally).  ``smooth_weight``
    pulls ``δq`` back toward zero so the realised per-frame jump is
    almost always below 0.05 rad — small enough to be invisible to
    downstream training, big enough to keep the clip moving.  The
    fallback is logged at WARNING so it shows up in the retarget log
    instead of being silent.
    """
    import mujoco

    from hhtools.retarget.interaction_mesh.laplacian_geometry import calculate_laplacian_matrix
    from hhtools.retarget.interaction_mesh.qp_step import (
        assemble_laplacian_qp,
        build_kron_laplacian_jacobian,
        solve_qp_box_lbfgsb,
    )

    nh = frame.n_human_vertices
    mpc_points = robot_points or _robot_points_from_body_names(body_names[:nh])
    nq = model.nq
    V = int(frame.source_vertices.shape[0])
    obj_pts = frame.source_vertices[nh:].astype(np.float64, copy=False)
    q_work = np.asarray(qpos, dtype=np.float64).copy()

    has_hard_np = collision_model is not None and collision_data is not None
    left_foot_vi = _mpc_point_vertex_index(mpc_points, nh, left_foot_pt)
    right_foot_vi = _mpc_point_vertex_index(mpc_points, nh, right_foot_pt)

    for _ in range(sqp_inner_iters):
        data.qpos[:] = q_work
        mujoco.mj_forward(model, data)
        pos_r = _current_robot_point_positions(model, data, mpc_points[:nh])
        verts = np.vstack([pos_r, obj_pts]).astype(np.float64, copy=False)
        if verts.shape[0] != V:
            raise RuntimeError("vertex count mismatch between robot bodies and demo mesh")
        L = calculate_laplacian_matrix(verts, frame.adj_list, uniform_weight=True)
        lap0 = (L @ verts).reshape(-1)
        target_vec = frame.target_laplacian.reshape(-1).astype(np.float64, copy=False)

        J_r = _stack_vertex_jacobians(model, data, mpc_points[:nh], nq)
        J_o = np.zeros((3 * max(0, V - nh), nq), dtype=np.float64)
        J_V = np.vstack([J_r, J_o])
        J_L = build_kron_laplacian_jacobian(L, J_V)
        qp = assemble_laplacian_qp(J_L, lap0, target_vec, laplacian_weight=laplacian_weight)

        if q_prev is not None and smooth_weight > 0:
            sw = float(smooth_weight)
            dq_smooth = np.asarray(q_prev, dtype=np.float64) - q_work
            # Holosoma smoothness applies on actuated q_a only — not the FREE
            # joint (pelvis XYZ/quat are anchored by position_weight instead).
            # Smoothing all nq DOFs was over-damping the base while under-
            # constraining hip/knee relative to q_prev, which shows up as leg
            # jitter especially with mpc_horizon > 1 window solves.
            if actuated_qpos_idx is not None and actuated_qpos_idx.size > 0:
                idx = actuated_qpos_idx
                sw_diag = np.full(int(idx.size), sw, dtype=np.float64)
                lsw = float(leg_smooth_weight)
                if (
                    lsw != 1.0
                    and leg_actuated_qpos_idx is not None
                    and leg_actuated_qpos_idx.size > 0
                ):
                    leg_mask = np.isin(idx, leg_actuated_qpos_idx)
                    sw_diag[leg_mask] *= lsw
                qp.P[np.ix_(idx, idx)] += 2.0 * np.diag(sw_diag)
                qp.q_vec[idx] += -2.0 * sw_diag * dq_smooth[idx]
            else:
                qp.P[:] += 2.0 * sw * np.eye(nq, dtype=np.float64)
                qp.q_vec[:] += -2.0 * sw * dq_smooth

        # ---- Absolute-position tracking cost ------------------
        # Adds ``½ · pw · Σ_i ‖ pos_robot_i − pos_target_i ‖²`` for
        # the ``nh`` mapped joints.  Linearised at ``q_work`` this is
        # ``½ · pw · ‖J_r δq + (pos_r − target)‖²`` which contributes
        # ``J_rᵀ J_r`` to ``P`` and ``J_rᵀ (pos_r − target)`` to
        # ``q_vec``.  Without this term the Laplacian is purely
        # translation-equivariant: an anatomy-mismatched robot whose
        # leg is longer than the scaled-source pelvis-to-foot can
        # satisfy the Laplacian cost by floating ~Δleg above the
        # source target.  The position cost ties absolute positions
        # to the source so the foot-contact pattern (relative to
        # heightfield) matches the source's.
        if position_weight > 0.0:
            pw = float(position_weight)
            target_pos = frame.source_vertices[:nh].astype(np.float64, copy=False)
            res = (pos_r - target_pos).reshape(-1)
            # Per-point relative weights let a grasping end-effector (the wrist
            # collision tip standing in for a missing hand) be prioritised so it
            # actually reaches the contact, rather than averaging out against the
            # feet / pelvis.  Defaults to 1.0 for every point (uniform = old
            # behaviour).
            w_pts = np.array(
                [float(getattr(p, "weight", 1.0)) for p in mpc_points[:nh]],
                dtype=np.float64,
            )
            if np.allclose(w_pts, 1.0):
                qp.P[:] += 2.0 * pw * (J_r.T @ J_r)
                qp.q_vec[:] += 2.0 * pw * (J_r.T @ res)
            else:
                w3 = np.repeat(w_pts, 3)  # one weight per (x, y, z) residual row
                Jw = J_r * w3[:, None]
                qp.P[:] += 2.0 * pw * (J_r.T @ Jw)
                qp.q_vec[:] += 2.0 * pw * (J_r.T @ (w3 * res))

        # ---- Home-pose Tikhonov on actuated DOFs --------------
        # ``½ · hw · Σ_{j ∈ actuated} (q_j + δq_j − q_home_j)²``.
        # Linearised gradient: ``hw · (q_j − q_home_j + δq_j)`` per
        # actuated DOF; that's a diagonal addition to ``P`` and a
        # linear addition to ``q_vec`` on those DOF rows only.
        # Free-joint quaternion / translation DOFs are deliberately
        # excluded — those are pinned by ``position_weight`` on the
        # pelvis vertex, and applying a Tikhonov to the quaternion
        # would fight per-frame yaw changes.
        if (
            home_pose_weight > 0.0
            and home_qpos is not None
            and actuated_qpos_idx is not None
            and actuated_qpos_idx.size > 0
        ):
            hw = float(home_pose_weight)
            idx = actuated_qpos_idx
            err = (q_work[idx] - home_qpos[idx]).astype(np.float64, copy=False)
            qp.P[idx, idx] += 2.0 * hw
            qp.q_vec[idx] += 2.0 * hw * err

        # --- Box trust region + joint limits ---
        lb = np.full(nq, -float(step_size), dtype=np.float64)
        ub = np.full(nq, float(step_size), dtype=np.float64)
        # Tighter cap on floating-base XYZ DOFs.  Holosoma applies its
        # ``step_size`` uniformly; we keep an extra safety margin on
        # root translation so a single OSQP solve cannot cross a 30 cm
        # step in one iteration.
        if base_step_size is not None and base_step_size > 0:
            try:
                if int(model.jnt_type[0]) == mujoco.mjtJoint.mjJNT_FREE:
                    qadr = int(model.jnt_qposadr[0])
                    bs = float(base_step_size)
                    for j in range(qadr, qadr + 3):
                        lb[j] = max(lb[j], -bs)
                        ub[j] = min(ub[j], bs)
            except Exception:
                pass
        if q_lb is not None and q_ub is not None:
            jl_lb = q_lb - q_work
            jl_ub = q_ub - q_work
            np.maximum(lb, jl_lb, out=lb)
            np.minimum(ub, jl_ub, out=ub)
        leg_scale = float(leg_sqp_step_scale)
        if (
            leg_scale > 0.0
            and leg_scale < 1.0
            and leg_actuated_qpos_idx is not None
            and leg_actuated_qpos_idx.size > 0
        ):
            cap = float(step_size) * leg_scale
            for qi in leg_actuated_qpos_idx:
                lb[int(qi)] = max(lb[int(qi)], -cap)
                ub[int(qi)] = min(ub[int(qi)], cap)

        # --- Solve QP -----------------------------------------------------
        foot_j: list[NDArray[np.float64]] = []
        foot_lb: list[float] = []
        foot_ub: list[float] = []
        if (
            activate_foot_sticking
            and foot_sticking
            and q_foot_ref is not None
            and (left_foot_pt is not None or right_foot_pt is not None)
        ):
            foot_j, foot_lb, foot_ub = _build_foot_sticking_rows(
                model,
                data,
                q_work,
                np.asarray(q_foot_ref, dtype=np.float64),
                left_foot_pt=left_foot_pt,
                right_foot_pt=right_foot_pt,
                foot_sticking=foot_sticking,
                tolerance=foot_sticking_tolerance,
                nq=nq,
                J_r=J_r,
                left_vi=left_foot_vi,
                right_vi=right_foot_vi,
            )

        if has_hard_np or foot_j:
            J_rows, rhs = ([], [])
            if has_hard_np:
                J_rows, rhs = _compute_hard_nonpenetration_rows(
                    collision_model, collision_data, q_work,
                    threshold=float(collision_threshold),
                    tolerance=float(penetration_tolerance),
                    fd_epsilon=float(collision_fd_epsilon),
                    max_pairs_per_body=int(collision_max_pairs_per_body),
                )
            try:
                dq = _solve_qp_with_inequalities(
                    qp.P, qp.q_vec, lb, ub, J_rows, rhs,
                    foot_J_rows=foot_j,
                    foot_lb=foot_lb,
                    foot_ub=foot_ub,
                ).astype(np.float64, copy=False)
            except OsqpUnreliableError as exc:
                # Bounded-step box-only fallback — see "Failure
                # semantics" in the docstring.  Trust region is
                # shrunk by OSQP_FALLBACK_TRUST_SHRINK so the solver
                # cannot take more than ~step_size/4 in any single
                # iter even though the inequality rows are dropped;
                # combined with smooth_weight this keeps the realised
                # |Δq| comparable to a healthy frame.
                _log.warning(
                    "SQP frame OSQP fallback (box-only, trust×%.2f): %s",
                    OSQP_FALLBACK_TRUST_SHRINK, exc,
                )
                lb_fb = lb * OSQP_FALLBACK_TRUST_SHRINK
                ub_fb = ub * OSQP_FALLBACK_TRUST_SHRINK
                dq = solve_qp_box_lbfgsb(qp, lb_fb, ub_fb).astype(np.float64, copy=False)
        else:
            dq = solve_qp_box_lbfgsb(qp, lb, ub).astype(np.float64, copy=False)

        q_work = q_work + dq.reshape(-1)
        _normalize_free_joint_quat(model, q_work)
    return q_work


# Quadratic penalty on the per-row non-penetration slack.
#
# Tuned on holosoma parkour_1 (rp1, 80 frames).  This is deliberately a
# *gentle* backstop, not a stiff barrier, for two reasons:
#
#  1. The base + feet are already anchored in absolute world space by the
#     position cost (``position_weight`` = 400), which tracks the scaled
#     source feet whose contact pattern relative to the terrain is correct
#     by construction — so collision only has to stop gross penetration,
#     not reproduce contact.
#  2. Heightfield ``mj_geomDistance`` witness points / normals flip
#     discontinuously as a foot crosses terrain cell boundaries.  A stiff
#     penalty turns that chatter into the whole-robot "flashing" jitter the
#     user reported (sweep: ρ=1e2 → 1.6°/frame² jerk_max; ρ=1e3 → 10.9°;
#     ρ=1e4 → 30.3°, clearly trembling).  ρ=1e2 keeps the trajectory as
#     smooth as the old (collision-dropped) path while still feasible.
#
# Override via ``HHTOOLS_NP_SLACK_PENALTY`` for experiments.
NONPENETRATION_SLACK_PENALTY = float(
    os.environ.get("HHTOOLS_NP_SLACK_PENALTY", "1.0e2")
)


def _solve_qp_with_inequalities(
    P: NDArray[np.float64],
    q_vec: NDArray[np.float64],
    lb: NDArray[np.float64],
    ub: NDArray[np.float64],
    J_rows: list[NDArray[np.float64]],
    rhs: list[float],
    *,
    foot_J_rows: list[NDArray[np.float64]] | None = None,
    foot_lb: list[float] | None = None,
    foot_ub: list[float] | None = None,
    slack_penalty: float = NONPENETRATION_SLACK_PENALTY,
) -> NDArray[np.float64]:
    """Solve QP with box bounds, optional **hard** foot sticking, and soft collision slack.

    Combines::

        min  0.5 x'Px + q'x  +  0.5·ρ·Σ sᵢ²
        s.t. lb ≤ x ≤ ub                    (box / trust region + joint limits)
             foot_lb ≤ J_foot x ≤ foot_ub   (foot sticking — hard, no slack)
             J_rows[i] · x + sᵢ ≥ rhs[i]    (non-penetration, slack sᵢ ≥ 0)

    Foot sticking rows mirror holosoma's ``Jxy @ dqa`` window around the
    previous-frame foot XY.  They are **hard** inequalities (no slack) because
    contact jitter is the primary failure mode foot sticking exists to fix.

    Non-penetration rows remain soft-slacked: heightfield witness chatter
    still makes a fully-hard stack infeasible on contact-rich frames.
    """
    from hhtools.retarget.interaction_mesh.qp_step import solve_qp_osqp

    nq = P.shape[0]
    n_ineq = len(J_rows)
    foot_j = list(foot_J_rows or [])
    foot_l = list(foot_lb or [])
    foot_u = list(foot_ub or [])
    n_foot = len(foot_j)
    if n_foot and (len(foot_l) != n_foot or len(foot_u) != n_foot):
        raise ValueError("foot_J_rows, foot_lb, foot_ub length mismatch")

    if n_ineq == 0 and n_foot == 0:
        A = np.eye(nq, dtype=np.float64)
        return solve_qp_osqp(P, q_vec, A, lb.copy(), ub.copy()).astype(
            np.float64, copy=False
        )

    if n_ineq == 0 and n_foot > 0:
        j_foot = np.vstack(foot_j)
        a_foot = np.vstack([np.eye(nq, dtype=np.float64), j_foot])
        l_full = np.concatenate([lb, np.asarray(foot_l, dtype=np.float64)])
        u_full = np.concatenate([ub, np.asarray(foot_u, dtype=np.float64)])
        z = solve_qp_osqp(P, q_vec, a_foot, l_full, u_full).astype(np.float64, copy=False)
        return z[:nq]

    # Augmented variable vector z = [δq (nq); s (n_ineq)].
    n_aug = nq + n_ineq
    rho = float(slack_penalty)

    P_aug = np.zeros((n_aug, n_aug), dtype=np.float64)
    P_aug[:nq, :nq] = P
    P_aug[nq:, nq:] = rho * np.eye(n_ineq, dtype=np.float64)
    q_aug = np.concatenate([q_vec, np.zeros(n_ineq, dtype=np.float64)])

    J_np = np.vstack(J_rows)  # (n_ineq, nq)

    blocks = [
        np.hstack([np.eye(nq), np.zeros((nq, n_ineq))]),
        np.hstack([np.zeros((n_ineq, nq)), np.eye(n_ineq)]),
        np.hstack([J_np, np.eye(n_ineq)]),
    ]
    l_parts = [lb, np.zeros(n_ineq, dtype=np.float64), np.asarray(rhs, dtype=np.float64)]
    u_parts = [ub, np.full(n_ineq, 1e20), np.full(n_ineq, 1e20)]

    if n_foot > 0:
        j_foot = np.vstack(foot_j)
        blocks.append(np.hstack([j_foot, np.zeros((n_foot, n_ineq))]))
        l_parts.append(np.asarray(foot_l, dtype=np.float64))
        u_parts.append(np.asarray(foot_u, dtype=np.float64))

    A = np.vstack(blocks)
    l_full = np.concatenate(l_parts)
    u_full = np.concatenate(u_parts)

    z = solve_qp_osqp(P_aug, q_aug, A, l_full, u_full).astype(np.float64, copy=False)
    return z[:nq]


def iterate_mpc_rti(
    model,
    data,
    targets: list[FrameLaplacianTarget],
    *,
    robot_body_names: list[str] | None = None,
    robot_points: list[RobotMpcPoint] | None = None,
    laplacian_weight: float,
    step_size: float,
    smooth_weight: float = 0.6,
    mpc_horizon: int = 1,
    sqp_inner_iters: int = 2,
    sqp_inner_iters_frame0: int = 5,
    mpc_window_sqp_iters: int = 2,
    mpc_window_warm_start: bool = True,
    mpc_collision_commit_only: bool = True,
    # --- absolute world-position cost (anchors global root motion) ---
    position_weight: float = 0.0,
    # --- home-pose Tikhonov on actuated DOFs (breaks null-space yaw drift) ---
    home_pose_weight: float = 0.0,
    # --- holosoma-style foot sticking ---
    activate_foot_sticking: bool = True,
    foot_sticking_tolerance: float = 1e-3,
    foot_sticking_velocity_threshold: float = 0.01,
    foot_sticking_release_hysteresis: int = 0,
    leg_smooth_weight: float = 1.0,
    leg_sqp_step_scale: float = 1.0,
    # --- holosoma-style hard non-penetration ---
    collision_model=None,
    collision_data=None,
    collision_threshold: float = 0.05,
    penetration_tolerance: float = 0.002,
    collision_fd_epsilon: float = 1e-5,
    collision_max_pairs_per_body: int = 4,
    # --- trust region ---
    base_step_size: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> NDArray[np.float64]:
    """Sliding-window MPC with per-frame SQP (holosoma ``iterate_mpc`` pattern).

    **Speed.**  Holosoma defaults to ``mpc_horizon=1``: foot sticking is a
    per-frame hard constraint and does not require multi-frame lookahead.
    Set ``mpc_horizon > 1`` only when you need explicit preview smoothing;
    combine with ``mpc_window_warm_start=True`` so later frames cost ~1 outer
    pass instead of ``mpc_window_sqp_iters × H`` full solves.

    Foot sticking hard inequalities apply on window index ``k == 0`` only,
    referencing the previous committed ``qpos``.
    """
    import mujoco

    if not targets:
        return np.zeros((0, model.nq), dtype=np.float64)
    nh = targets[0].n_human_vertices
    if robot_points is not None:
        body_names = [pt.body_name for pt in robot_points[:nh]]
    elif robot_body_names is not None:
        body_names = robot_body_names
    else:
        body_names = _mj_body_names_prefix(model, nh)

    H = max(1, int(mpc_horizon))
    left_foot_pt, right_foot_pt = resolve_foot_mpc_points(robot_points, nh)
    foot_sticking_seq = extract_foot_sticking_sequences(
        targets,
        robot_points,
        velocity_threshold=foot_sticking_velocity_threshold,
    )
    foot_sticking_seq = stabilize_foot_sticking_sequences(
        foot_sticking_seq,
        release_hysteresis=foot_sticking_release_hysteresis,
    )

    q_lb, q_ub = _build_joint_limits(model)
    leg_actuated_qpos_idx = _leg_actuated_qpos_indices(model)

    traj = np.zeros((len(targets), model.nq), dtype=np.float64)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    q_committed = np.asarray(data.qpos, dtype=np.float64).copy()

    home_qpos = q_committed.copy()
    _act_idx: list[int] = []
    for j in range(model.njnt):
        jt = int(model.jnt_type[j])
        if jt in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            _act_idx.append(int(model.jnt_qposadr[j]))
    actuated_qpos_idx = np.asarray(_act_idx, dtype=np.int64)

    Ftot = len(targets)

    if Ftot > 0 and int(model.jnt_type[0]) == mujoco.mjtJoint.mjJNT_FREE:
        qadr = int(model.jnt_qposadr[0])
        sv = np.asarray(targets[0].source_vertices, dtype=np.float64)
        if sv.shape[0] > 0:
            q_committed[qadr : qadr + 3] = sv[0, :3]
        sq = getattr(targets[0], "source_root_quat_xyzw", None)
        if sq is not None:
            sq = np.asarray(sq, dtype=np.float64).reshape(4)
            n = float(np.linalg.norm(sq))
            if n > 1e-9:
                sq = sq / n
                q_committed[qadr + 3] = sq[3]
                q_committed[qadr + 4] = sq[0]
                q_committed[qadr + 5] = sq[1]
                q_committed[qadr + 6] = sq[2]

    q_t_last: NDArray[np.float64] | None = None
    notify_stride = max(1, Ftot // 40)
    prev_q_window: list[NDArray[np.float64]] | None = None

    def _sqp_common_kwargs(*, use_collision: bool) -> dict:
        return dict(
            robot_points=robot_points,
            laplacian_weight=laplacian_weight,
            step_size=step_size,
            smooth_weight=smooth_weight,
            q_lb=q_lb,
            q_ub=q_ub,
            position_weight=position_weight,
            home_pose_weight=home_pose_weight,
            home_qpos=home_qpos,
            actuated_qpos_idx=actuated_qpos_idx,
            collision_model=collision_model if use_collision else None,
            collision_data=collision_data if use_collision else None,
            collision_threshold=collision_threshold,
            penetration_tolerance=penetration_tolerance,
            collision_fd_epsilon=collision_fd_epsilon,
            collision_max_pairs_per_body=collision_max_pairs_per_body,
            base_step_size=base_step_size,
            activate_foot_sticking=activate_foot_sticking,
            left_foot_pt=left_foot_pt,
            right_foot_pt=right_foot_pt,
            foot_sticking_tolerance=foot_sticking_tolerance,
            leg_actuated_qpos_idx=leg_actuated_qpos_idx,
            leg_smooth_weight=leg_smooth_weight,
            leg_sqp_step_scale=leg_sqp_step_scale,
        )

    # --- Fast path: H=1 (holosoma default) — one SQP per frame, no window loop ---
    if H == 1:
        for f in range(Ftot):
            q_ref = q_committed if q_t_last is None else q_t_last
            inner = sqp_inner_iters_frame0 if f == 0 else sqp_inner_iters
            q_committed = sqp_step_laplacian(
                model,
                data,
                q_committed,
                targets[f],
                body_names,
                sqp_inner_iters=inner,
                q_prev=q_ref,
                foot_sticking=foot_sticking_seq[f] if activate_foot_sticking else None,
                q_foot_ref=q_ref if activate_foot_sticking else None,
                **_sqp_common_kwargs(use_collision=True),
            )
            q_t_last = q_committed.copy()
            traj[f] = q_committed
            if progress_callback is not None and (f % notify_stride == 0 or f == Ftot - 1):
                try:
                    progress_callback(f + 1, Ftot)
                except Exception:
                    pass
        return traj

    # --- Multi-frame window MPC (optional preview; slower) ---
    for f in range(Ftot):
        win_len = min(H, Ftot - f)
        window_targets = targets[f : f + win_len]
        window_fs = foot_sticking_seq[f : f + win_len]

        if (
            mpc_window_warm_start
            and prev_q_window is not None
            and len(prev_q_window) >= 2
        ):
            q_window = [prev_q_window[1].copy()]
            for k in range(1, win_len):
                src = k + 1
                if src < len(prev_q_window):
                    q_window.append(prev_q_window[src].copy())
                else:
                    q_window.append(q_window[k - 1].copy())
        else:
            q_window = [q_committed.copy()]
            for k in range(1, win_len):
                q_window.append(q_window[k - 1].copy())

        q_ref = q_committed if q_t_last is None else q_t_last
        n_outer = max(1, int(mpc_window_sqp_iters))
        if f == 0:
            n_outer = max(n_outer, int(sqp_inner_iters_frame0 // 2))

        for _outer in range(n_outer):
            q_before = [q.copy() for q in q_window]
            for k in range(win_len):
                q_prev_k = q_ref if k == 0 else q_window[k - 1]
                apply_fs = (k == 0) and activate_foot_sticking
                use_coll = (k == 0) or not mpc_collision_commit_only
                if k == 0:
                    inner = sqp_inner_iters_frame0 if f == 0 else sqp_inner_iters
                else:
                    inner = 1
                q_window[k] = sqp_step_laplacian(
                    model,
                    data,
                    q_window[k],
                    window_targets[k],
                    body_names,
                    sqp_inner_iters=inner,
                    q_prev=q_prev_k,
                    foot_sticking=window_fs[k] if apply_fs else None,
                    q_foot_ref=q_ref if apply_fs else None,
                    **_sqp_common_kwargs(use_collision=use_coll),
                )
            if _outer > 0:
                max_step = max(
                    float(np.max(np.abs(q_window[k] - q_before[k])))
                    for k in range(win_len)
                )
                if max_step < 1e-3:
                    break

        prev_q_window = q_window
        q_committed = q_window[0].copy()
        q_t_last = q_committed.copy()
        traj[f] = q_committed

        if progress_callback is not None and (f % notify_stride == 0 or f == Ftot - 1):
            try:
                progress_callback(f + 1, Ftot)
            except Exception:
                pass
    return traj


__all__ = [
    "FrameLaplacianTarget",
    "RobotMpcPoint",
    "build_demo_vertices_frame",
    "causal_smooth_actuated_qpos",
    "count_named_mujoco_bodies",
    "extract_foot_sticking_sequences",
    "iterate_mpc_rti",
    "precompute_target_laplacians",
    "resolve_foot_mpc_points",
    "sample_axis_aligned_box",
    "sqp_step_laplacian",
    "stabilize_foot_sticking_sequences",
]
