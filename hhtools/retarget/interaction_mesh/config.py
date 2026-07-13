# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Configuration for :mod:`hhtools.retarget.interaction_mesh`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InteractionMeshPipelineConfig:
    """Knobs for the Laplacian MPC / SQP backend.

    Collision handling models the holosoma retargeter directly.  Terrain travels through the
    pipeline as a :class:`hhtools.core.scene.TerrainHeightfield` and is
    compiled into the SQP's MuJoCo collision model as a native
    ``<hfield>`` asset; ``mj_geomDistance`` provides per-pair signed
    distances and finite-difference Jacobians; OSQP enforces the
    resulting ``J · δq ≥ −φ − tol`` rows as **hard inequality**
    constraints alongside a box trust region on δq.

    Hard constraints are the only thing that can prevent the
    "translate-invariant Laplacian cost lets the base teleport" failure
    mode the soft penalty exhibited: the Laplacian is independent of
    rigid translation, so any soft penetration term can always be
    reduced by lifting the floating base instead of bending the foot.
    A non-violable inequality cannot be reduced — the optimum has to
    bend the foot.
    """

    mpc_horizon: int = 1
    # Weight on the Laplacian (interaction-mesh shape preservation)
    # cost.  Empirical finding (parc_ms BOXES_12 + holosoma parkour_1,
    # 80 frames each, rp1 at ``pw=200``):
    #   lap=10 → parc_ms 0.067 m / 0.231 m, holosoma 0.050 m / 0.129 m,
    #            holosoma jerk_max 5.81 °/frame²
    #   lap=2  → parc_ms 0.054 m / 0.189 m, holosoma 0.044 m / 0.088 m
    #   lap=0  → parc_ms 0.045 m / 0.162 m, holosoma 0.039 m / 0.075 m,
    #            holosoma jerk_max 3.10 °/frame²
    #
    # When the source skeleton's anatomical proportions don't match
    # the robot's exactly (always true for SMPL → rp1 / g1 / etc.),
    # the Laplacian cost — which encodes per-joint offsets from the
    # mean of its mesh neighbours, scaled by source bone lengths —
    # actively *fights* the position cost: the position cost wants
    # ``robot_joint_world = scaled_source_joint_world`` while the
    # Laplacian wants ``robot_joint - mean(neighbours) = scaled
    # (source_joint - mean(source_neighbours))``.  Both can't be
    # satisfied simultaneously when bone-length ratios differ, so
    # the QP arrives at a frame-dependent compromise that *both*
    # increases tracking residual and amplifies jitter (the
    # compromise basin shifts from frame to frame as the source
    # pose changes).  Setting the weight to zero drops the term
    # entirely; positional targets on every effector + the
    # smoothness regulariser already encode "track the yellow
    # skeleton smoothly", which is what users actually want.
    #
    # Set to a small non-zero value (e.g. 1.0) if you want a hint
    # of shape regularisation without the alignment penalty — but
    # the default is 0 because the empirical sweep above shows
    # strict alignment-better behaviour at zero on every clip we
    # tested.  Terrain interaction is unaffected: terrain anchor
    # points are still sampled and contribute positional rows to
    # the same QP, so the floating base stays anchored to the
    # heightfield in world coordinates.
    laplacian_weight: float = 0.0
    sqp_step_size: float = 0.2
    sqp_inner_iters: int = 3
    # Extra SQP inner iterations on frame 0 only (base warm-start).
    sqp_inner_iters_frame0: int = 5
    # The smoothness term ``sw · ‖q_new − q_prev‖²`` couples each
    # frame's QP back to the previous frame's solve, breaking the
    # "valley jumping" failure mode where the SQP, finding the
    # cost is locally degenerate along an under-constrained DOF
    # (e.g. ``hip_yaw`` with a foot Laplacian invariant under
    # axial-leg rotation), settles in a slightly different basin
    # frame-to-frame and produces a jittery trajectory.
    #
    # Heightfield collision constraints inject additional jitter
    # on their own — the hard non-penetration rows reformulated
    # from ``mj_geomDistance`` change witness points and normals
    # discontinuously as the foot crosses cell boundaries on the
    # heightfield, and OSQP's active set flips each frame
    # accordingly.  The user-reported "since switching to
    # heightmap, the robot trembles" failure mode is exactly
    # this: holosoma/parkour_1 ran at 11°/frame² jerk_max with
    # 51/79 frames showing > 3° per-frame ``|Δq|`` steps.
    #
    # Empirical sweep on parc_ms BOXES_12 + holosoma parkour_1
    # (rp1, 80 frames each) at ``laplacian_weight=0`` and
    # ``position_weight=400``:
    #   ``sw=8``  → holosoma 0.87 ° hinge_mean / jerk 11.1
    #   ``sw=24`` → holosoma 0.42 ° / jerk 3.10   ← default base
    #   ``sw=48`` → holosoma 0.31 ° / jerk 2.06
    #   ``sw=96`` → holosoma 0.27 ° / jerk 1.65 (lags rapid motion)
    #
    # ``sw`` was briefly dropped to 2 with a large ``leg_smooth_weight``
    # so legs stayed quiet while arms chased source wrist noise — that
    # shows up as end-of-clip left/right arm tremble once the floating
    # base orientation is locked and can no longer absorb the residual.
    # Base ``sw=24`` restores the sweep's "no visible tremble" regime
    # for arms/waist; legs get an extra multiplier below.
    smooth_weight: float = 24.0
    # Extra temporal smooth multiplier on leg actuated DOFs (hip/knee/ankle).
    # Effective leg weight ≈ ``smooth_weight * leg_smooth_weight`` (= 48
    # at the defaults — matches the original global ``sw=48`` sweet spot).
    leg_smooth_weight: float = 2.0
    # Per-iteration trust-region scale on leg hinges (< 1 → smaller |Δq| per SQP step).
    leg_sqp_step_scale: float = 0.75
    object_surface_samples: int = 32

    # Number of terrain-surface anchor points sampled from the
    # heightfield and fed into the interaction mesh as additional
    # vertices.  Holosoma's design places terrain points alongside
    # human joints in a single Delaunay tetrahedral mesh — the
    # terrain stays static in world coordinates, so every human
    # joint's Laplacian δ encodes its **position relative to the
    # terrain**, which is exactly what anchors global root motion to
    # the source trajectory.  Without these anchors the Laplacian cost
    # is translation-invariant and the floating base just smoothness-
    # damps to a near-stationary trajectory.
    terrain_surface_samples: int = 96

    # Per-mapped-joint absolute world-position cost weight.  The
    # Laplacian cost shapes *relative* limb posture but is
    # translation-equivariant on its own — an anatomy-mismatched
    # robot (longer leg than scaled source) can satisfy the
    # Laplacian by floating a fixed offset above the source target.
    # A ``position_weight`` ties each mapped joint to its
    # *absolute* world target so the heightfield contact pattern
    # matches the source: when the source ankle is 5 cm above the
    # source heightfield, the robot ankle ends up ~5 cm above the
    # robot heightfield at the same scaled XY.
    #
    # Empirical sweep on parc_ms BOXES_12 + holosoma parkour_1 (rp1):
    # ``pw=5  lap=10`` → mean 12.4 cm / max 32.9 cm (parc_ms),  7.4 cm /
    #             15.2 cm (holosoma).
    # ``pw=50 lap=10`` → mean  5.4 cm / max 11.9 cm (parc_ms),  4.4 cm /
    #              7.1 cm (holosoma).
    # ``pw=200 lap=0`` → mean  4.5 cm / max 16.2 cm (parc_ms),  3.9 cm /
    #              7.5 cm (holosoma).
    # ``pw=400 lap=0 sw=48`` → mean  4.4 cm / max 16.1 cm (parc_ms),
    #              3.8 cm /  7.6 cm (holosoma); holosoma jerk_max 2.1 °/frame²,
    #              the user-perceived "since switching to heightmap robot
    #              trembles like crazy" issue is no longer visible.
    #
    # With the Laplacian disabled (see :attr:`laplacian_weight`) the
    # position cost is the *only* thing pulling the robot toward the
    # yellow skeleton — so it has to be heavy.  ``pw=400`` puts the
    # solver in a regime where the QP's optimum closely tracks the
    # scaled-source effector positions; the heavier smoothness term
    # ``sw=48`` stops it from chasing per-frame foot-plant pops and
    # the non-penetration constraint keeps feet above terrain.
    #
    # If you re-enable the Laplacian (``laplacian_weight > 0``),
    # consider dropping this back to ~50 — at high ``pw`` and
    # high ``lap`` the two costs trample each other and the QP
    # produces visible posture distortions on rigs whose end-effector
    # targets are slightly unreachable (anatomy mismatch between
    # source bone lengths and robot link lengths after uniform
    # ``smpl_scale``).
    position_weight: float = 200.0

    # Tikhonov regularisation pulling the **actuated** joints toward
    # the model's home keyframe (``mj_resetData`` qpos0).  Without
    # this the QP cost is rank-deficient on the under-constrained
    # axes — most importantly **hip_yaw**, which only changes the
    # foot's orientation around the vertical and barely moves the
    # foot's world XYZ, so the position cost has a near-zero
    # gradient on it.  OSQP's residual rounding then injects ~0.5°
    # of noise per inner iteration on those null-direction DOFs
    # which compounds to multi-degree per-frame ``|Δq|`` jitter.
    # A small Tikhonov term breaks the symmetry and locks the null
    # directions to the home pose without measurably distorting
    # the well-constrained DOFs.  Free-joint DOFs (XYZ + quat) are
    # excluded from the regulariser — those are anchored by
    # ``position_weight`` on the pelvis vertex.
    home_pose_weight: float = 1.0

    # Position-cost weight multiplier on the wrist / ``hand_tip`` effector for
    # **grasping** clips (those carrying interaction objects, e.g. OMOMO
    # chair/box).  Hand-less robots (RP1) use the wrist link's last collision
    # tip as the de-facto hand; on grasp clips the user's priority is "first of
    # all, actually reach the object", so the hand effector is weighted heavier
    # than the locomotion effectors (feet/pelvis) which would otherwise average
    # the arm short of the contact.  ``1.0`` keeps the old uniform behaviour;
    # only applied when ``motion.objects`` is non-empty so locomotion clips are
    # unaffected.
    hand_contact_weight: float = 8.0

    # Position-cost multiplier on leg / pelvis effectors (ankle, knee, hip, foot).
    # Locomotion clips benefit from heavier leg tracking so the solver does not
    # trade leg accuracy for arm / Laplacian residuals frame-to-frame.
    leg_effector_weight: float = 3.0

    enable_collision: bool = True

    # Activation distance for the ``mj_geomDistance`` broadphase.  Pairs
    # closer than this contribute a hard inequality row; pairs farther
    # are ignored.  5 cm matches holosoma.
    collision_threshold: float = 0.05
    # Tolerance margin on the inequality: ``J·δq ≥ −φ − tol``.  A small
    # positive slack avoids chatter at the boundary.
    penetration_tolerance: float = 0.002
    # Central-difference step for the signed-distance Jacobian.
    collision_fd_epsilon: float = 1e-5

    # Per-iteration L∞ trust region on the floating-base XYZ DOFs.
    # Holosoma applies ``step_size`` uniformly; we keep a separate cap
    # on root translation so a single OSQP solve cannot cross a 30 cm
    # step in one iteration even if some objective gradient points in
    # that direction.
    sqp_base_step_size: float = 0.05

    # Holosoma-style foot sticking: when the source foot is in contact
    # (low XY velocity), pin the robot foot XY near the previous solved
    # frame via hard OSQP inequalities.
    activate_foot_sticking: bool = True
    foot_sticking_tolerance: float = 1e-3
    foot_sticking_velocity_threshold: float = 0.01
    # Keep foot sticky for this many frames after source velocity exceeds the
    # threshold — stops contact-flag flicker that injects leg micro-jitter.
    foot_sticking_release_hysteresis: int = 3
    # SQP Gauss-Seidel passes over the MPC window before committing frame 0.
    # Only used when ``mpc_horizon > 1``.  With ``mpc_horizon=1`` (holosoma
    # default, foot sticking only) this is ignored.
    mpc_window_sqp_iters: int = 2
    # When ``mpc_horizon > 1``, slide the previous window solution forward
    # and run at most one outer pass on later frames (large speed win).
    mpc_window_warm_start: bool = True
    # Skip ``mj_geomDistance`` collision rows on preview frames (k > 0) inside
    # the MPC window — only the committed frame needs terrain constraints.
    mpc_collision_commit_only: bool = True

    # One-pole low-pass on leg actuated qpos after MPC (causal).
    # ``beta=0.2`` blends 20 % of the previous frame into each leg joint;
    # increase toward 0.35 if slight tremble remains, decrease if legs feel mushy.
    post_smooth_leg_joints: bool = True
    post_smooth_leg_beta: float = 0.2
    # Same causal low-pass on arm / shoulder / elbow / wrist hinges.
    # Legs already get ``leg_smooth_weight`` inside the QP; arms only had
    # the base ``smooth_weight``, so clip-end standstill wrist noise was
    # still visible as arm tremble.  Milder beta than legs to avoid lag.
    post_smooth_arm_joints: bool = True
    post_smooth_arm_beta: float = 0.15

    # Post-MPC hip abduction spread when solved foot meshes interpenetrate or
    # ankles cross (same mesh-gated clamp as NewtonBasic).  Position-only
    # MPC has a near-zero gradient on ``hip_yaw`` / laterality when left and
    # right ankle targets nearly coincide, so the QP can settle into an
    # X-legged basin; this clamp abducts hips until the foot meshes clear.
    # Set ``≤ 0`` to disable.
    min_foot_clearance_m: float = 0.03
    # Per-leg abduction budget for the post-MPC clamp.  Ground-roll recoveries
    # can leave a deep X-leg that needs more than the Newton default (~0.2 rad).
    foot_lateral_max_abduction_rad: float = 0.55
    foot_lateral_max_iterations: int = 28
    # Pre-MPC: minimum left−right ankle/foot separation along the hip lateral
    # axis in scaled target space.  Stops roll mid-frames with coincident
    # ankles from teaching the QP an X-leg basin.  ``≤ 0`` disables.
    # Only applied on upright frames (see ``laterality_upright_min_m``) so an
    # inverted roll does not flip the hip lateral axis and yank waist yaw.
    min_target_foot_separation_m: float = 0.08
    # Hip−foot height (m, scaled) required before laterality / foot-clamp
    # corrections run.  Below this the body is treated as rolling / inverted.
    laterality_upright_min_m: float = 0.35

    # Lock FREE-joint orientation to each frame's continuous source
    # pelvis quaternion inside the SQP (δq on quat DOFs frozen to 0).
    # This is the fundamental fix for roll→stand root/leg teleports:
    # position-only MPC otherwise leaves orientation in a near-nullspace.
    lock_root_orientation_to_source: bool = True

    # Post-MPC: kill single-frame root / joint teleports after rolls.
    # Off by default — with ``lock_root_orientation_to_source`` the
    # solver already tracks source orientation; rate-limiting here only
    # papers over remaining basin hops.  Re-enable for emergency clips.
    stabilize_joints_after_mpc: bool = False
    # Floor angular speed (rad/s) for the floating-base quaternion; the
    # effective cap is ``max(floor, source_root_speed * multiplier)`` so
    # genuine flips in the mocap are preserved.
    max_root_angular_velocity: float = 2.5
    root_angular_velocity_source_multiplier: float = 1.5
    # Cap for actuated hinges whose name contains ``yaw`` / ``waist``
    # (rad/s).  Ground-roll recoveries otherwise dump a π jump into
    # ``waist_yaw`` while the feet stay planted.
    max_yaw_joint_velocity: float = 3.5
    # Cap for hip / knee / ankle hinges (rad/s).  Stand-up after a roll
    # otherwise teleports leg DOFs by tens of degrees in a single frame;
    # jumps larger than ``leg_joint_teleport_deg`` are held (not crawled).
    max_leg_joint_velocity: float = 6.0
    leg_joint_teleport_deg: float = 40.0
    # Per-frame abduction step when applying the foot-lateral clamp
    # (rad/s).  Prevents the upright-gate from dumping the full
    # ``foot_lateral_max_abduction_rad`` budget in one frame.
    foot_lateral_max_step_rad_s: float = 2.5
    # Extra causal low-pass on packed leg DOFs after clamp + rate-limit.
    # Off by default once root orientation is locked in the solver.
    post_smooth_leg_joints_after_stabilize: bool = False
    post_smooth_leg_beta_after: float = 0.35


__all__ = ["InteractionMeshPipelineConfig"]
