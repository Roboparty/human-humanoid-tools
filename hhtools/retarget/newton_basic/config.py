"""Config dataclasses for the ``newton_basic`` retargeting pipeline.

The Apache-2.0 upstream (``soma_retargeter``) stores its per-source × per-robot
tuning in a pair of JSON files loaded through
``soma_retargeter.utils.io_utils``.  We keep the same *shape* — a
``joint_scales`` map + per-joint offsets for the scaler, plus a grab-bag of
foot-planting / ground-contact thresholds for the stabilizer — but expose them
through typed dataclasses so callers can build them programmatically (unit
tests, notebooks) without touching a yaml file.

YAML support is kept optional so the pure-python stage-1 modules remain
testable in environments that don't have ``pyyaml`` installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "ArmChainConfig",
    "ScalerConfig",
    "FeetStabilizerConfig",
    "load_scaler_config",
    "load_feet_stabilizer_config",
]


# --------------------------------------------------------------------------- scaler


@dataclass
class ScalerConfig:
    """Human → robot effector-target scaler configuration.

    Attributes:
        human_height_assumption: Pelvis-to-ground height (metres) the
            ``joint_scales`` map was authored against.  At load time the
            scaler multiplies every entry by
            ``actual_human_height / human_height_assumption`` to keep the
            relative proportions constant across skeletons of different sizes.
        model_height: Nominal robot pelvis height (metres) — carried here only
            for downstream feet-stabilizer / body-ground-clearance tuning, the
            scaler itself doesn't consume it.
        joint_scales: ``canonical_joint_name -> scale_factor``.  Applied to the
            *displacement* between each joint and the root joint in world
            space.  A scale of ``1.0`` is a no-op.
        joint_offsets: ``canonical_joint_name -> ((tx, ty, tz), (qx, qy, qz, qw))``.
            ``t`` is applied in the joint's *local* frame *after* rotation;
            ``q`` is post-multiplied onto the joint's global quaternion.
            Mirrors soma's ``joint_offsets`` layout.  Missing entries default
            to identity.
        root_joint: Canonical name of the root joint whose world position
            anchors the scaling transform (defaults to ``"hips"`` — the
            canonical-human root).
        scale_mode: Either ``"uniform"`` (scale applied on all three axes
            equally — the default, matches ``scale_animation=True`` upstream)
            or ``"height"`` (scale applied on the up-axis only — matches
            ``scale_animation=False``).
        up_axis: Which world axis is "up" for the ``"height"`` mode.  hhtools
            standard is Z; change to Y for motions imported directly without
            axis conversion.
        scale_anchor: Where per-joint scales are applied *around*.

            - ``"root"`` (soma-compatible, *correct*, default produced by
              :func:`hhtools.retarget.calibration.calibration.build_scaler_config_soma_style`):
              each non-root joint's displacement from the source root is
              scaled by its own ``scale[j]``, while the root's world
              position is scaled uniformly by ``scale[root]``::

                  t_out[j] = (human_pos[j] - human_root) * scale[j]
                           + human_root * scale[root]
                           + rotate(q_out[j], t_offset[j])

              This matches upstream soma-retargeter
              (``wp_compute_scaled_effectors``) exactly.  The builder
              solves ``scale`` / ``t_offset`` / ``q_offset`` in closed
              form against this formula so frame-0 IK residual is zero
              AND motion frames stay unbiased.
            - ``"origin"`` (legacy, biased on non-rest frames): every
              joint's world position is scaled by its own ``scale[j]``::

                  t_out[j] = human_pos[j] * scale[j]
                           + rotate(q_out[j], t_offset[j])

              Matches ``"root"`` at rest (by construction of the
              builder's ``t_offset``) but introduces a per-frame bias
              ``(scale[j] - scale[root]) · (human_root[t] - Δq ⊙
              human_root_rest)`` that shows up as amplified shoulder
              roll, spurious hand reach, and pelvis drift whenever the
              source root is off the world origin — which is every real
              clip, since the source root sits at pelvis height
              (~1 m).  Retained only so pre-October-2026 configs keep
              loading; new configs should always use ``"root"``.
    """

    human_height_assumption: float = 1.7
    model_height: float = 1.3
    joint_scales: dict[str, float] = field(default_factory=dict)
    joint_offsets: dict[str, tuple[tuple[float, float, float],
                                    tuple[float, float, float, float]]] = field(
        default_factory=dict
    )
    root_joint: str = "hips"
    scale_mode: Literal["uniform", "height"] = "uniform"
    up_axis: Literal["X", "Y", "Z"] = "Z"
    scale_anchor: Literal["origin", "root"] = "root"
    # Post-scale constant vertical shift applied to every mapped joint's
    # target (not just the root).  Set by calibration to align the robot's
    # pelvis rest height with ground zero, so a source walker whose
    # pelvis sits at ~0.97 m and a robot whose pelvis sits at ~0.73 m end
    # up with feet on the ground instead of floating.  Units: metres, on
    # the chosen ``up_axis``.  ``0.0`` is the legacy / soma-compatible
    # behaviour.
    root_z_offset: float = 0.0
    # Robot pelvis-to-ground height (metres) used to recompute
    # ``root_z_offset`` at runtime when ``human_height`` differs from
    # ``human_height_assumption``.  The builder stores the pelvis height
    # it measured from the URDF; the scaler adjusts the vertical shift
    # so the scaled skeleton stays ground-aligned regardless of the
    # height ratio.  ``None`` means no runtime adjustment — the stored
    # ``root_z_offset`` is used as-is (backwards compatible).
    robot_pelvis_height: float | None = None
    # Yaw-only quaternion (xyzw) that pre-rotates every source position
    # and quaternion from the source skeleton's body-heading convention
    # into the robot's body-heading convention before scale/offset math
    # is applied.  Computed by the builder from the rest-pose yaw
    # difference between source root and robot root.  Identity (0,0,0,1)
    # means no heading correction — the source and robot already share
    # the same forward direction.  A non-identity value fixes the
    # "source walks forward but robot walks sideways" symptom caused by
    # axis convention mismatches (e.g. BVH Y-up→Z-up leaves source
    # forward along -Y, while URDF robots face +X).
    source_body_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    # Parity with ``soma_retargeter.assets.lafan1_foot_mod.append_gmr_lafan1_foot_mod``
    # (``use_toe_orientation``): when ``True`` and the source has LAFAN1-style
    # ``*Foot``/``*Toe`` naming, mapped ``*Foot`` **rotation** targets use the
    # source toe joint's **global** quaternion (positions stay on the ankle).
    # ``False`` / ``None`` (unset) leave the foot quaternion unchanged — same
    # default as upstream ``foot_mod_use_toe_orientation=False``.
    lafan_foot_mod_use_toe_orientation: bool | None = None

    # --------------------------------------------------------------- helpers

    def joint_names(self) -> list[str]:
        """Names of canonical joints that participate in the mapping.

        We keep this ordered: ``joint_scales`` insertion order is the source
        of truth, and any ``joint_offsets`` entries without a corresponding
        scale are ignored (matches upstream: no scale → no effector).
        """
        return list(self.joint_scales.keys())


def load_scaler_config(path: str | Path) -> ScalerConfig:
    """Read a :class:`ScalerConfig` from a YAML file.

    Raises:
        ImportError: If ``pyyaml`` isn't installed.
        FileNotFoundError: If ``path`` doesn't exist.
        KeyError: If required fields are missing.
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError as err:  # pragma: no cover — trivial
        raise ImportError(
            "pyyaml is required for load_scaler_config; install with "
            "`pip install pyyaml`"
        ) from err

    path = Path(path)
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    raw_offsets = data.get("joint_offsets", {}) or {}
    offsets: dict[str, tuple[tuple[float, float, float],
                              tuple[float, float, float, float]]] = {}
    for name, entry in raw_offsets.items():
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"joint_offsets[{name!r}] must be [[tx,ty,tz],[qx,qy,qz,qw]]; "
                f"got {entry!r}"
            )
        t, q = entry
        offsets[name] = (
            (float(t[0]), float(t[1]), float(t[2])),
            (float(q[0]), float(q[1]), float(q[2]), float(q[3])),
        )

    raw_sbq = data.get("source_body_quat")
    if raw_sbq is not None and len(raw_sbq) == 4:
        sbq = (float(raw_sbq[0]), float(raw_sbq[1]),
               float(raw_sbq[2]), float(raw_sbq[3]))
    else:
        sbq = (0.0, 0.0, 0.0, 1.0)

    _lafan_toe = data.get("lafan_foot_mod_use_toe_orientation")
    if _lafan_toe is None:
        lafan_toe_opt: bool | None = None
    else:
        lafan_toe_opt = bool(_lafan_toe)

    return ScalerConfig(
        human_height_assumption=float(data.get("human_height_assumption", 1.7)),
        model_height=float(data.get("model_height", 1.3)),
        joint_scales={k: float(v) for k, v in (data.get("joint_scales") or {}).items()},
        joint_offsets=offsets,
        root_joint=str(data.get("root_joint", "hips")),
        scale_mode=str(data.get("scale_mode", "uniform")),  # type: ignore[arg-type]
        up_axis=str(data.get("up_axis", "Z")),  # type: ignore[arg-type]
        scale_anchor=str(data.get("scale_anchor", "root")),  # type: ignore[arg-type]
        root_z_offset=float(data.get("root_z_offset", 0.0)),
        source_body_quat=sbq,
        lafan_foot_mod_use_toe_orientation=lafan_toe_opt,
    )


# --------------------------------------------------------------------------- feet stabilizer


@dataclass
class FeetStabilizerConfig:
    """Pre-IK effector-target constraint knobs.

    All of these map directly to fields consumed by the upstream
    ``HumanToRobotScaler`` post-processors:

    * ``ground_contact_z`` / ``min_foot_clearance``  →  ``_enforce_ground_contact``
    * ``foot_planting_velocity_threshold``           →  ``_enforce_foot_planting``
    * ``min_lateral_separation``                     →  ``_enforce_min_lateral_separation``
    * ``smoothing_max_rate``                         →  ``_smooth_corrections``

    The user only needs to set the thresholds they care about; everything else
    defaults to "no-op" so unit tests of individual constraints don't have to
    mock a full config.
    """

    up_axis: Literal["X", "Y", "Z"] = "Z"
    forward_axis: Literal["X", "Y", "Z"] = "X"

    # ---- ground contact ---------------------------------------------------
    ground_contact_z: float = 0.0
    min_foot_clearance: float = 0.0
    max_ground_correction: float = 0.05
    ground_uprightness_range: float = 0.30

    # ---- foot planting ----------------------------------------------------
    foot_planting_velocity_threshold: float = 0.0
    foot_planting_height_margin: float = 0.02
    foot_planting_release_frames: int = 3

    # ---- min-lateral separation ------------------------------------------
    min_lateral_separation: float = 0.0

    # ---- correction smoothing --------------------------------------------
    smoothing_max_rate: float = 0.008

    # ---- canonical joint names -------------------------------------------
    # Names resolved against the canonical human skeleton (hhtools convention).
    # Override for datasets that use different naming (e.g. "LeftFoot").
    left_foot_name: str = "left_ankle"
    right_foot_name: str = "right_ankle"
    # The hhtools canonical human skeleton has no toe bone — ankle is the
    # terminal foot joint — so toe defaults to None.  BVH-style rigs that do
    # expose toes (GMR / LAFAN / cranberry) set these to e.g. "left_foot".
    left_toe_name: str | None = None
    right_toe_name: str | None = None
    hips_name: str = "hips"
    # Extra Left/Right leg joint pairs we want kept symmetric for lateral
    # separation (used when ``min_lateral_separation > 0``).  Names here must
    # appear as keys in the scaler ``joint_scales`` map.
    lateral_pairs: tuple[tuple[str, str], ...] = ()

    # ---- body ground clearance (prone / crawl) ---------------------------
    # Mirrors soma ``_enforce_body_ground_clearance``: lift the whole effector
    # block when probe joints would sit below ``body_ground_plane_z + clearance``.
    enable_body_ground_clearance: bool = False
    body_ground_plane_z: float = 0.0
    body_ground_clearance: float = 0.025
    body_ground_probe_joints: tuple[str, ...] = ()
    body_ground_probe_below_meters: dict[str, float] = field(default_factory=dict)
    body_ground_default_probe_below: float = 0.0
    body_ground_lift_max_rate: float = 0.015
    body_ground_snap_on_penetration: bool = True

    # ---- hand ground contact (push-up / prone) ----------------------------
    # Mirrors soma ``_enforce_hand_ground_contact``: lower torso/arms so hand
    # effectors can reach ``hand_ground_contact_z`` when the scaled arm is too
    # short to touch the floor with hips at standing height.
    hand_ground_contact_z: float = 0.0
    chest_name: str = "Spine2"
    arm_chains: tuple["ArmChainConfig", ...] = ()


@dataclass(frozen=True)
class ArmChainConfig:
    """One arm chain for hand-ground reach budgeting (soma ``arm_chains``)."""

    shoulder: str
    chain: tuple[str, ...]
    max_reach: float


def load_feet_stabilizer_config(path: str | Path) -> FeetStabilizerConfig:
    """Read a :class:`FeetStabilizerConfig` from a YAML file."""
    try:
        import yaml  # type: ignore[import]
    except ImportError as err:  # pragma: no cover — trivial
        raise ImportError(
            "pyyaml is required for load_feet_stabilizer_config"
        ) from err

    path = Path(path)
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    lateral_raw = data.get("lateral_pairs") or []
    lateral_pairs = tuple(
        (str(pair[0]), str(pair[1])) for pair in lateral_raw
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    )

    probe_raw = data.get("body_ground_probe_joints") or ()
    probe_below_raw = data.get("body_ground_probe_below_meters") or {}
    probe_below = {
        str(k): float(v) for k, v in probe_below_raw.items()
        if isinstance(probe_below_raw, dict)
    }

    arm_chains: list[ArmChainConfig] = []
    for entry in data.get("arm_chains") or ():
        if not isinstance(entry, dict):
            continue
        shoulder = str(entry.get("shoulder", ""))
        chain_raw = entry.get("chain") or ()
        max_reach = float(entry.get("max_reach", 0.0) or 0.0)
        if shoulder and chain_raw and max_reach > 0.0:
            arm_chains.append(
                ArmChainConfig(
                    shoulder=shoulder,
                    chain=tuple(str(c) for c in chain_raw),
                    max_reach=max_reach,
                )
            )

    return FeetStabilizerConfig(
        up_axis=str(data.get("up_axis", "Z")),  # type: ignore[arg-type]
        forward_axis=str(data.get("forward_axis", "X")),  # type: ignore[arg-type]
        ground_contact_z=float(data.get("ground_contact_z", 0.0)),
        min_foot_clearance=float(data.get("min_foot_clearance", 0.0)),
        max_ground_correction=float(data.get("max_ground_correction", 0.05)),
        ground_uprightness_range=float(data.get("ground_uprightness_range", 0.30)),
        foot_planting_velocity_threshold=float(
            data.get("foot_planting_velocity_threshold", 0.0)
        ),
        foot_planting_height_margin=float(
            data.get("foot_planting_height_margin", 0.02)
        ),
        foot_planting_release_frames=int(
            data.get("foot_planting_release_frames", 3)
        ),
        min_lateral_separation=float(data.get("min_lateral_separation", 0.0)),
        smoothing_max_rate=float(data.get("smoothing_max_rate", 0.008)),
        left_foot_name=str(data.get("left_foot_name", "left_ankle")),
        right_foot_name=str(data.get("right_foot_name", "right_ankle")),
        left_toe_name=data.get("left_toe_name"),
        right_toe_name=data.get("right_toe_name"),
        hips_name=str(data.get("hips_name", "hips")),
        lateral_pairs=lateral_pairs,
        enable_body_ground_clearance=bool(
            data.get("enable_body_ground_clearance", False)
        ),
        body_ground_plane_z=float(data.get("body_ground_plane_z", 0.0)),
        body_ground_clearance=float(data.get("body_ground_clearance", 0.025)),
        body_ground_probe_joints=tuple(str(j) for j in probe_raw),
        body_ground_probe_below_meters=probe_below,
        body_ground_default_probe_below=float(
            data.get("body_ground_default_probe_below", 0.0)
        ),
        body_ground_lift_max_rate=float(
            data.get("body_ground_lift_max_rate", 0.015)
        ),
        body_ground_snap_on_penetration=bool(
            data.get("body_ground_snap_on_penetration", True)
        ),
        hand_ground_contact_z=float(data.get("hand_ground_contact_z", 0.0)),
        chest_name=str(data.get("chest_name", "Spine2")),
        arm_chains=tuple(arm_chains),
    )
