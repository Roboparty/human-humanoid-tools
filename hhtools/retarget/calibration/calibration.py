"""Robot retarget calibration — dataclass, YAML IO, closed-form derivation.

Workflow at a glance
--------------------

1. **Capture** — the viewer's calibration mode lets the user dial
   actuated joint angles so the robot, at floating-base identity,
   visually matches a chosen reference human T-pose.  The resulting
   configuration is packaged into a :class:`RobotRetargetCalibration`
   and written next to the URDF as
   ``retarget_calibration_<reference>.yaml`` (one file per robot **and**
   per reference format: ``smpl``, ``lafan_bvh``, …) via
   :func:`save_calibration`.  Legacy ``retarget_calibration.yaml`` is
   still loaded when its embedded ``reference`` matches.

2. **Use** — at retarget time, :func:`build_scaler_config_from_calibration`
   reads that yaml, runs the URDF's forward kinematics at the stored
   joint configuration, and composes per-canonical-joint:

   * scalar scale = ``|robot_link_disp_from_root| / |ref_human_disp_from_root|``
     (both measured from the respective root: ``base_link`` for the
     robot, ``hips``/``pelvis`` for the reference human).  This
     captures anthropometric size differences.
   * orientation offset that, when chained with the source motion's
     frame-0 quaternion conjugate, lands the robot in its *calibrated*
     pose at motion frame 0 and then animates relative motion from
     there (identical semantics to the auto heuristic the user
     rejected as too fragile, only now the "target pose" is whatever
     the user dialled in — no longer a guess about SMPL conventions).
   * translation offset, in the joint's local (post-rotation) frame,
     that closes the residual after the scalar scale — non-zero when
     the robot's link direction doesn't line up with the reference
     human's radial direction from the root.

3. **Root yaw** — the only per-motion quantity we still derive is a
   yaw-preserving offset for the root joint so the retargeted robot
   starts facing the same direction the source human does.  Pitch /
   roll of the source root are absorbed into the offset (so the robot
   stays upright).

The stored YAML is intentionally minimal — just
``calibrated_joint_q`` + ``reference`` + ``robot`` — so humans can
``git diff`` it, and so a re-derivation runs fresh each time the
underlying URDF or reference code changes.  The derived parameters are
not cached; the closed-form computation is <1 ms per robot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

_log = logging.getLogger(__name__)

from hhtools.core.math import quaternion as Q
from hhtools.retarget.calibration.reference import (
    HumanReferencePose,
    ReferenceName,
    build_motion_reference,
    load_reference_pose,
)

if TYPE_CHECKING:
    from hhtools.core.motion import Motion
    from hhtools.retarget.newton_basic.config import ScalerConfig
    from hhtools.retarget.newton_basic.rest_pose import SourceRestPose
    from hhtools.robot.loader import URDFRobotModel

CALIBRATION_FILENAME = "retarget_calibration.yaml"

# Old yaml files used these strings before ReferenceName stabilised.
_CALIBRATION_REFERENCE_LEGACY: dict[str, str] = {
    "canonical_human": "smpl",
    "mixamo_bvh": "lafan_bvh",
    "custom": "smpl",
    "fbx": "lafan_bvh",
}

_VALID_CALIBRATION_REFERENCES: frozenset[str] = frozenset(
    {"smplx", "smpl", "gvhmr", "soma_bvh", "lafan_bvh", "mocap_bvh", "xsens_mocap", "glb"}
)

# Canonical adult-human stature (metres) used by
# :func:`build_scaler_config_from_calibration` as
# ``ScalerConfig.human_height_assumption`` for SMPL / GLB sources.
# Pinning every clip to this value (instead of the per-clip measured Z-extent
# which jitters with frame-0 pose) gives a consistent
# ``smpl_scale = robot_height / 1.65`` for the yellow uniform-scale overlay
# and the MPC SQP solver, regardless of whether the actor is 1.55m or 1.85m
# tall.  Per-joint anatomical scales remain subject-specific; only the
# whole-body normalisation is pinned.
_CANONICAL_HUMAN_HEIGHT_M: float = 1.65


def normalize_calibration_reference(name: str) -> str:
    """Normalise a yaml ``reference`` string to a current :class:`ReferenceName`."""

    return _CALIBRATION_REFERENCE_LEGACY.get(name, str(name))


def _reference_pose_for_calibration(
    reference: str,
    *,
    motion: "Motion | None" = None,
) -> HumanReferencePose:
    """Resolve the human reference used for calibration math.

    Static references (``smpl``, ``smplx``, …) load via
    :func:`~hhtools.retarget.calibration.reference.load_reference_pose`.
    ``glb`` is clip-specific and requires the same
    :class:`~hhtools.core.motion.Motion` whose frame-0 skeleton was used
    when the user calibrated.
    """

    ref = normalize_calibration_reference(reference)
    if ref == "glb":
        if motion is None:
            raise ValueError(
                f"calibration reference {reference!r} (→ {ref!r}) requires a loaded "
                f"Motion — pass the clip whose frame-0 skeleton matches calibration."
            )
        return build_motion_reference(motion, ref)  # type: ignore[arg-type]
    return load_reference_pose(ref)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Dataclass + IO
# ---------------------------------------------------------------------------


@dataclass
class RobotRetargetCalibration:
    """Persistent record of a hand-calibrated robot↔human alignment.

    Only the three fields below are saved verbatim — everything else is
    derived from them at retarget time by running the URDF's forward
    kinematics.  That means the YAML format stays stable even if we
    later change the math in :func:`derive_calibration_params`.

    Attributes
    ----------
    robot
        Name of the robot preset this calibration is for
        (``preset.name``).  Loaded back verbatim and checked against the
        preset the user is retargeting to, so a copy-pasted YAML
        doesn't silently get misapplied to a different robot.
    reference
        Which reference human pose the calibration was done against.
        Must match one of :func:`list_reference_names`.
    calibrated_joint_q
        ``{actuated_joint_name: angle_in_radians}``.  Missing joints
        default to 0.0 at load time, so a calibration only has to store
        the joints the user actually moved.
    notes
        Optional free-form string — intended for the UI to record e.g.
        which human mesh skin was on screen when the user saved.  Not
        consumed by the derivation.
    """

    robot: str
    reference: ReferenceName
    calibrated_joint_q: dict[str, float]
    notes: str = ""


def calibration_path_for(
    robot_preset_dir: str | Path,
    *,
    reference: str | None = None,
) -> Path:
    """Filesystem path for a robot preset's calibration yaml.

    Parameters
    ----------
    robot_preset_dir
        Directory containing the robot's ``robot*.yaml`` and URDF
        (``preset.urdf_path.parent``).
    reference
        When set (e.g. ``\"smpl\"``, ``\"lafan_bvh\"``), returns
        ``retarget_calibration_<reference>.yaml`` so each human **reference
        format** keeps its own zero-pose alignment for the same URDF.

        When omitted, returns the legacy single-file name
        ``retarget_calibration.yaml`` — callers should prefer passing
        ``reference`` for new writes; :func:`resolve_calibration_file` handles
        discovery including legacy fallbacks.
    """

    base = Path(robot_preset_dir)
    if reference:
        return base / f"retarget_calibration_{reference}.yaml"
    return base / CALIBRATION_FILENAME


def resolve_calibration_file(
    robot_preset_dir: str | Path,
    reference: str,
) -> Path | None:
    """Pick an on-disk calibration file for this robot dir + reference.

    Resolution order:

    1. ``retarget_calibration_<reference>.yaml`` if present.
    2. Legacy ``retarget_calibration.yaml`` if it loads and its stored
       ``reference`` normalises to the same value as ``reference`` (so old
       repos with a single file per robot still work when the reference
       matches).

    Returns ``None`` when nothing matches — the UI / CLI should prompt for
    calibration.
    """

    want = normalize_calibration_reference(reference)
    if want not in _VALID_CALIBRATION_REFERENCES:
        raise ValueError(
            f"unknown calibration reference {reference!r} "
            f"(normalised {want!r}); expected one of "
            f"{sorted(_VALID_CALIBRATION_REFERENCES)}"
        )

    d = Path(robot_preset_dir)
    preferred = d / f"retarget_calibration_{reference}.yaml"
    if preferred.is_file():
        return preferred

    legacy = d / CALIBRATION_FILENAME
    if not legacy.is_file():
        return None
    try:
        cal = load_calibration(legacy)
    except Exception:
        return None
    if normalize_calibration_reference(cal.reference) == want:
        return legacy
    return None


def save_calibration(
    calibration: RobotRetargetCalibration,
    path: str | Path,
    *,
    derived: "_DerivedParams | None" = None,
) -> Path:
    """Write ``calibration`` to ``path`` as yaml and return the final path.

    The yaml is formatted deterministically (sorted keys, 2-space indent)
    so the diff is minimal across user edits — this is a hand-edited
    config file in practice.

    When ``derived`` is provided (the viewer populates it on save), its
    motion-independent closed-form params are mirrored into a top-level
    ``derived:`` block.  That block is purely informational — the loader
    ignores it and re-computes from ``calibrated_joint_q`` each time so
    we never drift if the robot's URDF / mesh changes under a stale
    calibration.  But it gives users a readable "what did calibration
    produce?" artefact they can diff in git, and makes it easy to spot
    obviously-wrong scales (e.g. a 3× shoulder) without re-running the
    retarget pipeline.
    """

    import yaml

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "robot": calibration.robot,
        "reference": calibration.reference,
        "calibrated_joint_q": {
            k: float(v) for k, v in sorted(calibration.calibrated_joint_q.items())
        },
        "notes": calibration.notes,
    }
    if derived is not None:
        payload["derived"] = {
            "robot_root_link": derived.robot_root_link or "",
            "reference_root": (
                derived.reference.root_joint if derived.reference else ""
            ),
            "scales": {
                k: float(v) for k, v in sorted(derived.scales.items())
            },
            "link_quaternions": {
                k: [float(x) for x in v]
                for k, v in sorted(derived.link_quaternions.items())
            },
            "translation_offsets": {
                k: [float(x) for x in v]
                for k, v in sorted(derived.translation_offsets.items())
            },
            "_comment": (
                "Informational cache of the closed-form scale + offset "
                "derivation at save time.  Loader ignores this block and "
                "re-derives from calibrated_joint_q so stale caches "
                "cannot silently misalign retarget."
            ),
        }
    with target.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, default_flow_style=False, sort_keys=False)
    return target


def load_calibration(path: str | Path) -> RobotRetargetCalibration:
    """Load a :class:`RobotRetargetCalibration` from a yaml file.

    Raises
    ------
    FileNotFoundError
        When the path doesn't exist — the caller should surface this to
        the user as "run the calibration UI first".
    ValueError
        For malformed contents (missing required keys, unknown
        reference name, non-dict joint_q, ...).
    """

    import yaml

    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(
            f"no retarget calibration at {src!s} — start the calibration "
            f"flow from the viewer's Robot tab to create one."
        )
    with src.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{src}: yaml root must be a mapping")
    missing = [k for k in ("robot", "reference", "calibrated_joint_q") if k not in data]
    if missing:
        raise ValueError(f"{src}: missing required keys {missing}")
    reference = normalize_calibration_reference(str(data["reference"]))
    if reference not in _VALID_CALIBRATION_REFERENCES:
        raise ValueError(
            f"{src}: unknown reference {data['reference']!r} → {reference!r} "
            f"(expected one of {sorted(_VALID_CALIBRATION_REFERENCES)})"
        )
    joint_q = data["calibrated_joint_q"] or {}
    if not isinstance(joint_q, dict):
        raise ValueError(f"{src}: calibrated_joint_q must be a mapping")
    return RobotRetargetCalibration(
        robot=str(data["robot"]),
        reference=reference,  # type: ignore[arg-type]
        calibrated_joint_q={str(k): float(v) for k, v in joint_q.items()},
        notes=str(data.get("notes", "")),
    )


_STRAIGHT_T_ARM_REFERENCES: tuple[str, ...] = (
    "lafan_bvh",
    "mocap_bvh",
    "xsens_mocap",
    "smpl",
    "smplx",
    "gvhmr",
)

_TPOSE_ARM_JOINTS: tuple[str, ...] = (
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_elbow_joint",
    "right_elbow_joint",
)


def repair_apose_calibration_for_straight_t_reference(
    calibration: RobotRetargetCalibration,
    robot_preset_dir: str | Path,
) -> RobotRetargetCalibration:
    """Borrow T-pose arm angles from a sibling calibration when needed.

    Straight-arm references (LAFAN / mocap bundled T-pose, SMPL T-pose, …)
    require shoulder roll ≈ ±π/2.  Legacy calibrations saved against an
    A-pose overlay (roll ≈ 0) freeze the arms; this helper copies arm
    joint values from another reference calibration in the same preset dir.
    """

    from dataclasses import replace

    ref = normalize_calibration_reference(str(calibration.reference))
    if ref == "soma_bvh" or ref not in _STRAIGHT_T_ARM_REFERENCES:
        return calibration

    roll = float(calibration.calibrated_joint_q.get("left_shoulder_roll_joint", 0.0))
    if abs(roll) > 0.5:
        return calibration

    preset_dir = Path(robot_preset_dir)
    for donor_ref in _STRAIGHT_T_ARM_REFERENCES:
        if donor_ref == ref:
            continue
        donor_path = resolve_calibration_file(preset_dir, donor_ref)
        if donor_path is None:
            continue
        try:
            donor = load_calibration(donor_path)
        except (OSError, ValueError):
            continue
        if donor.robot != calibration.robot:
            continue
        donor_roll = float(donor.calibrated_joint_q.get("left_shoulder_roll_joint", 0.0))
        if abs(donor_roll) < 0.5:
            continue
        patched = dict(calibration.calibrated_joint_q)
        for key in _TPOSE_ARM_JOINTS:
            if key in donor.calibrated_joint_q:
                patched[key] = donor.calibrated_joint_q[key]
        return replace(calibration, calibrated_joint_q=patched)

    return calibration


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


@dataclass
class _DerivedParams:
    """Internal: outputs of :func:`derive_calibration_params`.

    Stored per-canonical (not per-source-joint) because calibration is
    robot-centric; the mapping to source joint names happens later via
    :func:`build_scaler_config_from_calibration` /
    :func:`build_scaler_config_soma_style`.

    These fields are a strict function of the calibrated URDF pose —
    the robot-side half of the alignment.  They are consumed directly
    by the save-time ``derived:`` yaml cache (informational, for
    ``git diff`` legibility) and by unit tests that introspect the
    rest-closure invariant (``scale * human_disp + q_rotate(link_q,
    t_offset) == robot_disp``).  Downstream retarget math lives in the
    builders, not here.
    """

    # Per canonical joint name that appears in the robot's ik_map.
    scales: dict[str, float] = field(default_factory=dict)
    # Rotation offsets in the joint's local (post-rotation) sense — xyzw.
    # These are the robot link's world quaternion at the calibrated pose.
    link_quaternions: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    # Translation offsets in each link's local frame (applied AFTER the
    # rotation offset rebased by q_source[0] at runtime).  Non-zero when
    # the robot's link direction doesn't match the reference human's
    # displacement direction; closes the residual exactly.
    translation_offsets: dict[str, tuple[float, float, float]] = field(
        default_factory=dict
    )
    # The reference pose that was resolved (for UI display + error
    # messages); also carries the root joint name we should hand to the
    # ScalerConfig for proper root-translation handling.
    reference: HumanReferencePose | None = None
    # Name of the robot's root link as resolved from ik_map (falls back
    # to ``base_link``).  Used to anchor the robot-side displacement.
    robot_root_link: str | None = None


def _ik_map_pairs(model: URDFRobotModel) -> list[tuple[str, str]]:
    """Flatten ``robot.yaml:ik_map`` into ``(canonical, link_name)`` pairs.

    Accepts both the flat ``{canonical: link}`` form and the nested
    ``{canonical: {t_body, r_body, ...}}`` form, matching what
    :func:`hhtools.retarget.newton_basic.robot_model.resolve_ik_map`
    accepts — so the calibration derivation can be run *without*
    paying the cost of a full Newton + Warp build (that compiles warp
    kernels on first call, way too heavy for the scale-only preview
    and for unit tests).
    """

    pairs: list[tuple[str, str]] = []
    for canonical, spec in (model.preset.ik_map or {}).items():
        if isinstance(spec, dict):
            link = str(spec.get("t_body") or spec.get("link") or canonical)
        else:
            link = str(spec)
        pairs.append((str(canonical), link))
    return pairs


def _collect_link_transforms_at_q(
    model: URDFRobotModel, joint_q: dict[str, float]
) -> dict[str, NDArray]:
    """Return ``{link_name: (4,4) world transform}`` at a given configuration.

    Mutates ``urdf`` via ``apply_configuration`` — yourdfpy invalidates
    its scene-graph cache on ``update_cfg`` so reads after the call
    reflect the new pose.

    We key the result on *URDF link name* (not the visual-mesh node name
    — the latter is derived from the mesh file stem and diverges for
    robots whose meshes are named ``Left_hip_pitch.STL`` but whose link
    is ``left_hip_pitch_link``).  yourdfpy's scene graph registers each
    link as a node with the link's actual name, so iterating
    ``scene.graph.nodes_in_scene`` and intersecting with
    ``urdf.link_map`` gives exactly the link-keyed FK we want.
    """

    model.apply_configuration(joint_q)
    out: dict[str, NDArray] = {}
    try:
        scene = model.trimesh_scene(collision=False)
    except Exception:
        # No visual meshes → at least give callers the base_link at
        # identity so the derivation can still run (degenerate case).
        out[model.base_link] = np.eye(4, dtype=np.float64)
        return out

    link_names = set(getattr(model.urdf, "link_map", {}).keys())
    for node in scene.graph.nodes:
        if node not in link_names:
            continue
        try:
            mat, _geom = scene.graph[node]
        except Exception:  # noqa: BLE001 — skip unreachable nodes
            continue
        out[node] = np.asarray(mat, dtype=np.float64)
    out.setdefault(model.base_link, np.eye(4, dtype=np.float64))
    return out


def _rotmat_to_xyzw(R: NDArray) -> NDArray:
    """Wrapper around :func:`hhtools.core.math.quaternion.from_matrix`.

    Exists mostly as a naming tweak — callers here usually have a 3×3
    matrix extracted from a ``(4, 4)`` transform and Q.from_matrix
    accepts that shape directly.
    """

    return Q.from_matrix(np.asarray(R, dtype=np.float64))


def _extract_yaw_quat(
    q_xyzw: NDArray,
    forward_body: NDArray | tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> NDArray:
    """World-Z-axis yaw such that ``R(yaw) * world_Z_up = R(q) * forward_body``.

    The subject's **world-up axis is always +Z** in hhtools (every
    loader normalises to Z-up at import time).  But the body-frame
    axis that points *forward* depends on the source rig's convention:

    * hhtools canonical skeleton: ``forward_body = (+1, 0, 0)`` — body
      +X is subject-forward.  This is the default.
    * SMPL / SMPL-H / SMPL-X motions: ``forward_body = (0, 0, +1)`` —
      body +Z is subject-forward (+Y is up in SMPL's body frame).  The
      caller detects this via :func:`is_smpl_like` and overrides the
      default.

    Rather than decompose ``q`` into a body-frame twist-swing (which
    would silently misinterpret the SMPL z-component as world yaw and
    produce the 90°-off-heading bug observed on AMASS walkers), we
    rotate the body's forward-axis into the world frame and then take
    the 2-D heading of its horizontal projection.  This is convention-
    agnostic at the call site — the only knob is which body axis is
    "forward", and that is a fixed property of the source rig.

    Degenerate case: when the subject's forward projects straight onto
    world-Z (subject is looking down / up), the horizontal projection
    is near-zero and we clamp to identity — "no yaw" is the safest
    interpretation of an ambiguous orientation, and the IK will pick
    it up from the non-root joints at frame 0 anyway.
    """

    q = np.asarray(q_xyzw, dtype=np.float32).reshape(4)
    fwd_body = np.asarray(forward_body, dtype=np.float32).reshape(3)
    # Rotate the body-frame forward axis into world coordinates via the
    # quaternion rotation formula; we could pull in Q.rotate here but
    # inlining keeps this function dependency-free for test harnesses
    # that stub :mod:`hhtools.core.math.quaternion`.
    world_fwd = Q.rotate(q[None, :], fwd_body[None, :])[0]
    horiz_mag = float(np.sqrt(world_fwd[0] * world_fwd[0] + world_fwd[1] * world_fwd[1]))
    if horiz_mag < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    yaw = float(np.arctan2(float(world_fwd[1]), float(world_fwd[0])))
    half = 0.5 * yaw
    return np.array(
        [0.0, 0.0, float(np.sin(half)), float(np.cos(half))],
        dtype=np.float32,
    )


def derive_calibration_params(
    calibration: RobotRetargetCalibration,
    model: URDFRobotModel,
    *,
    reference_motion: "Motion | None" = None,
    reference_pose: "HumanReferencePose | None" = None,
) -> _DerivedParams:
    """Closed-form scale + rotation + translation offsets from a calibration.

    Runs the URDF's FK at ``calibration.calibrated_joint_q``, extracts
    each ik-mapped link's world transform, and computes per-canonical
    quantities the scaler can consume directly.

    The computation is deterministic and side-effect free except that
    it mutates ``model.urdf`` via ``apply_configuration`` (we restore
    the zero configuration on exit so callers don't end up with a
    surprise pose).

    Does **not** compose with any source motion — that happens later in
    :func:`build_scaler_config_from_calibration` which also applies the
    root yaw-preservation rule.  Splitting the two stages keeps the
    motion-independent part unit-testable.

    Parameters
    ----------
    reference_motion
        Pass the live :class:`~hhtools.core.motion.Motion` when
        ``calibration.reference`` is ``glb`` (same clip as at
        calibration time).  Ignored for static references.
    """

    if reference_pose is not None:
        reference = reference_pose
    else:
        reference = _reference_pose_for_calibration(
            str(calibration.reference),
            motion=reference_motion,
        )
    ref_names = {n: i for i, n in enumerate(reference.joint_names)}
    ref_positions = reference.positions  # (J, 3), hips-relative

    # For format-specific references, joint_names are native (e.g.
    # "pelvis", "Hips") not canonical.  Build a canonical→native lookup
    # so ik_map canonical keys can index into the reference arrays.
    _can2native: dict[str, str] = {}
    for native, canonical in reference.source_to_canonical.items():
        _can2native.setdefault(canonical, native)

    def _ref_idx(canonical_name: str) -> int | None:
        """Index into reference arrays by canonical name."""
        native = _can2native.get(canonical_name, canonical_name)
        idx = ref_names.get(native)
        if idx is not None:
            return idx
        return ref_names.get(canonical_name)

    pairs = _ik_map_pairs(model)
    link_for_canonical = {c: ln for c, ln in pairs}

    saved_q = model.zero_configuration()
    try:
        link_transforms = _collect_link_transforms_at_q(
            model, calibration.calibrated_joint_q
        )
    finally:
        model.apply_configuration(saved_q)

    # Resolve the robot's "root link" — the ik_map entry for the
    # reference's root joint (mapped to canonical) takes precedence.
    _ref_root_canonical = reference.source_to_canonical.get(
        reference.root_joint, reference.root_joint,
    )
    robot_root_link = model.base_link
    for canonical, link_name in pairs:
        if canonical == _ref_root_canonical:
            robot_root_link = link_name
            break
    robot_root_T = link_transforms.get(robot_root_link)
    if robot_root_T is None:
        robot_root_T = np.eye(4, dtype=np.float64)
    robot_root_pos = robot_root_T[:3, 3]

    out = _DerivedParams(reference=reference, robot_root_link=robot_root_link)

    # ------------------------------------------------------------------
    # Pelvis-radial derivation.  For each ik-mapped canonical joint we
    # compute ``scale = |robot_link - robot_root| / |ref_pos - ref_root|``
    # and a per-link orientation + translation offset anchored to the
    # robot's calibrated pose.  These are a strict function of the URDF
    # rest geometry and the reference pose — they're the robot-side
    # half of the closed-form solution that
    # :func:`build_scaler_config_soma_style` solves for at rest, and
    # they're written into the ``derived:`` yaml cache so calibration
    # diffs stay legible.
    # ------------------------------------------------------------------
    _root_ref_idx = _ref_idx(_ref_root_canonical)
    if _root_ref_idx is None:
        _root_ref_idx = 0

    from hhtools.retarget.newton_basic.human_aliases import is_xsens_mocap_like

    _xsens_ref = (
        str(calibration.reference).lower() == "xsens_mocap"
        or is_xsens_mocap_like(tuple(reference.joint_names))
    )
    _xsens_hip_knee: dict[str, str] = {
        "left_hip": "left_knee",
        "right_hip": "right_knee",
    }

    for canonical, link_name in pairs:
        cidx = _ref_idx(canonical)
        if cidx is None:
            out.scales[canonical] = 1.0
            out.link_quaternions[canonical] = (0.0, 0.0, 0.0, 1.0)
            out.translation_offsets[canonical] = (0.0, 0.0, 0.0)
            continue

        human_disp = ref_positions[cidx] - ref_positions[_root_ref_idx]
        link_T = link_transforms.get(link_name)
        if link_T is None:
            # URDF link named in ik_map has no visual geometry →
            # unusable for this derivation.  Same graceful fallback as
            # the "unknown canonical" branch above.
            out.scales[canonical] = 1.0
            out.link_quaternions[canonical] = (0.0, 0.0, 0.0, 1.0)
            out.translation_offsets[canonical] = (0.0, 0.0, 0.0)
            continue

        link_pos = link_T[:3, 3]
        link_R = link_T[:3, :3]
        link_q = _rotmat_to_xyzw(link_R)  # xyzw world quaternion
        robot_disp = link_pos - robot_root_pos

        # ---- scalar scale -------------------------------------------
        if canonical == _ref_root_canonical:
            # Root scale is intentionally 1.0 here — the runtime
            # ``human_height / human_height_assumption`` ratio in
            # HumanToRobotScaler applies the actual body-size ratio
            # (robot_h / motion_h) dynamically.  Keeping it at 1.0
            # lets calibrations port cleanly between motions of
            # different subject heights.
            scale = 1.0
        else:
            if _xsens_ref and canonical in _xsens_hip_knee:
                knee_canon = _xsens_hip_knee[canonical]
                knee_idx = _ref_idx(knee_canon)
                knee_link = link_for_canonical.get(knee_canon)
                knee_T = link_transforms.get(knee_link) if knee_link else None
                if knee_idx is not None and knee_T is not None:
                    human_for_scale = (
                        ref_positions[knee_idx] - ref_positions[_root_ref_idx]
                    )
                    robot_for_scale = knee_T[:3, 3] - robot_root_pos
                    h_norm = float(np.linalg.norm(human_for_scale))
                    r_norm = float(np.linalg.norm(robot_for_scale))
                else:
                    h_norm = float(np.linalg.norm(human_disp))
                    r_norm = float(np.linalg.norm(robot_disp))
            else:
                h_norm = float(np.linalg.norm(human_disp))
                r_norm = float(np.linalg.norm(robot_disp))
            if h_norm > 1e-4 and r_norm > 1e-6:
                scale = r_norm / h_norm
            else:
                # Co-located joints (e.g. canonical "spine" and "hips"
                # sometimes fall on top of each other) → bail on the
                # ratio and use unit; IK will handle the slack.
                scale = 1.0

        # ---- rotation + translation offsets -------------------------
        # Store the robot link's world quaternion at calibration as-is;
        # the Scaler-config builder will rebase it with the source
        # motion's frame-0 quaternion conjugate.
        out.link_quaternions[canonical] = (
            float(link_q[0]), float(link_q[1]),
            float(link_q[2]), float(link_q[3]),
        )

        # Translation offset in the link's local (post-rotation) frame
        # that, when added after ``scale * human_disp`` (and then
        # rotated by the link quaternion), closes the position gap
        # exactly.  Mathematically:
        #   robot_disp = scale * human_disp + q_rotate(link_q, t_offset)
        #   → t_offset = q_rotate(conj(link_q), robot_disp − scale * human_disp)
        residual = robot_disp - scale * human_disp
        t_offset = Q.rotate(Q.conjugate(link_q), residual.astype(np.float32))
        out.scales[canonical] = float(scale)
        out.translation_offsets[canonical] = (
            float(t_offset[0]), float(t_offset[1]), float(t_offset[2])
        )

    return out


# ---------------------------------------------------------------------------
# ScalerConfig builder — composes calibration with a source motion's frame-0
# ---------------------------------------------------------------------------


def _auto_source_to_canonical(bone_names: tuple[str, ...]) -> dict[str, str]:
    """Local wrapper around the SMPL→canonical alias map.

    Avoids the ``calibration → newton_basic`` import side-effects at
    module load — the alias table is a tiny dict lookup so re-importing
    it is cheap.
    """

    from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical

    return auto_source_to_canonical(bone_names)


def build_scaler_config_from_calibration(
    calibration: RobotRetargetCalibration,
    model: URDFRobotModel,
    clip,  # hhtools.core.motion.Motion — avoided here to keep this module importable when motion isn't installed
    *,
    human_height: float,
    preserve_root_yaw: bool = True,
    joint_scale_overrides: dict[str, float] | None = None,
) -> "ScalerConfig":
    """Build a :class:`ScalerConfig` that retargets ``clip`` onto ``model``.

    Thin wrapper that extracts a rest-pose snapshot from ``clip`` and
    delegates to :func:`build_scaler_config_soma_style` — the
    closed-form, origin-anchored builder that replaces the old
    hierarchical path.  At rest the scaled source effectors coincide
    exactly with the robot's calibrated link transforms (modulo the
    intentional vertical ground-align shift), so the IK solver sees
    zero residual on the first frame.

    Rest-pose selection is **source-format-dependent**:

    * **SMPL / SMPL-H / SMPL-X** (AMASS, HMR4D, Motion-X, …): uses
      :func:`~hhtools.retarget.newton_basic.rest_pose.rest_pose_from_motion_bind`
      which reconstructs the body-model's bind T-pose (all non-root
      local rotations = identity) from the clip's hierarchy.  This is
      essential because AMASS clips start in an arbitrary motion frame
      (e.g. mid-walk with arms hanging down), while the robot was
      calibrated against the SMPL T-pose reference (arms out).  Without
      this, the scaler computes large ``t_offset`` / ``q_offset`` values
      that force the scaled skeleton's arms to the T-pose level for the
      entire clip, regardless of the actual motion content.
    * **SOMA BVH**: bundled ``soma_zero_frame0.bvh`` rest
      (:func:`~hhtools.retarget.newton_basic.rest_pose.rest_pose_from_bundled_reference`)
      — the same file soma-retargeter ships.  Clip frame 0 is **not** used:
      SOMA clips often open mid-motion, and using frame 0 would bake the
      starting pose into the scaler instead of the format's canonical rest.
    * **LAFAN / GLB / other BVH**: frame-0 rest
      (:func:`~hhtools.retarget.newton_basic.rest_pose.rest_pose_from_motion`).

    ``human_height`` and ``preserve_root_yaw`` are retained for API
    compatibility.  They are not written into the returned
    :class:`~hhtools.retarget.newton_basic.config.ScalerConfig`; height
    normalisation is applied when constructing
    :class:`~hhtools.retarget.newton_basic.scaler.HumanToRobotScaler`
    from ``human_height`` vs ``human_height_assumption``.
    """

    _ = human_height  # consumed by HumanToRobotScaler, not the builder
    _ = preserve_root_yaw  # retained for API compat

    from dataclasses import replace as _dc_replace

    preset_dir = getattr(getattr(model, "preset", None), "root_dir", None)
    if preset_dir is not None:
        calibration = repair_apose_calibration_for_straight_t_reference(
            calibration, preset_dir,
        )

    from hhtools.retarget.newton_basic.human_aliases import (
        is_meshmimic_holosoma_like,
        is_smpl_like,
        is_smpl_pruned_ankle_terminated,
        is_xsens_mocap_like,
    )
    from hhtools.retarget.newton_basic.rest_pose import (
        bundled_reference_bvh_path,
        rest_pose_from_bundled_reference,
        rest_pose_from_motion,
        rest_pose_from_motion_bind,
    )

    _ref = str(calibration.reference).lower()

    from hhtools.retarget.newton_basic.rest_pose import normalize_mocap_bvh_clip

    clip = normalize_mocap_bvh_clip(clip)

    # SOMA: always use the bundled zero-frame BVH (soma-retargeter convention),
    # not the loaded clip's frame 0.
    if _ref == "soma_bvh" and bundled_reference_bvh_path(_ref) is not None:
        rest_pose = rest_pose_from_bundled_reference("soma_bvh")
    elif _ref == "xsens_mocap" and bundled_reference_bvh_path(_ref) is not None:
        rest_pose = rest_pose_from_bundled_reference("xsens_mocap")
    elif _ref == "lafan_bvh" and bundled_reference_bvh_path(_ref) is not None:
        # Clip frame 0 as a *whole* skeleton.  The blue calibration overlay
        # still uses bundled ``lafan_zero_frame0.bvh`` (T-pose), but feeding
        # that T-pose into the scaler while the clip opens mid-motion makes
        # heading / ``q_offset`` fight the trajectory (arms & legs look
        # twisted).  Frame-0 rest keeps limb directions and facing coherent
        # with the capture; wrist twist vs the T-pose overlay is a known
        # LAFAN trade-off until we have a heading-safe T-pose rest path.
        rest_pose = rest_pose_from_motion(
            clip,
            frame=0,
            source_tag="build_scaler_config_from_calibration_lafan_frame0",
        )
    elif _ref == "mocap_bvh" and bundled_reference_bvh_path(_ref) is not None:
        rest_pose = rest_pose_from_bundled_reference("mocap_bvh")
    elif _ref == "lafan_bvh":
        rest_pose = rest_pose_from_motion_bind(
            clip, source_tag="build_scaler_config_from_calibration_lafan_bind"
        )
    elif is_smpl_like(clip.hierarchy.bone_names) or is_meshmimic_holosoma_like(
        clip.hierarchy.bone_names
    ):
        # SMPL-family: bind T-pose synthesiser (AMASS frame 0 is rarely rest).
        rest_pose = rest_pose_from_motion_bind(
            clip, source_tag="build_scaler_config_from_calibration_bind"
        )
    else:
        # GLB / clip-specific BVH: frame 0 matches calibration reference.
        rest_pose = rest_pose_from_motion(
            clip, frame=0, source_tag="build_scaler_config_from_calibration"
        )

    # ---- Canonical-stature normalisation ----------------------------------
    # SMPL / GLB clips represent adult humans of varying stature
    # (subject-dependent ``betas`` for SMPL, rigging-dependent ``armature
    # scale`` for GLB).  Without normalisation the yellow uniform-scale
    # overlay is ``robot_height / measured_subject_stature``, which is
    # ``robot_height / 1.55`` for short subjects (overlay too tall) and
    # ``robot_height / 1.85`` for tall subjects (overlay too short).
    #
    # We pin ``human_height_assumption`` to a canonical adult stature so:
    #
    #   * the yellow overlay scale is ``robot_height / 1.65`` for every
    #     clip, giving a consistent visual size next to the robot;
    #   * downstream consumers reading ``ScalerConfig.human_height_assumption``
    #     (UI's "Subject height" field, retarget metadata) see one value
    #     per source family rather than per-clip jitter;
    #   * per-joint anatomical scales (computed from bone-length ratios in
    #     :func:`build_scaler_config_soma_style`, not from ``height_m``)
    #     stay subject-specific so limb proportions still get retargeted
    #     correctly.
    #
    # 1.65m is the user-chosen canonical adult Asian-male stature; tweak
    # via :data:`_CANONICAL_HUMAN_HEIGHT_M` if a future preset (children,
    # very tall reference humans) needs a different normalisation.
    if _ref in (
        "smpl", "smplx", "gvhmr", "soma_bvh", "lafan_bvh", "mocap_bvh", "xsens_mocap", "glb",
    ):
        rest_pose = _dc_replace(
            rest_pose,
            height_m=_CANONICAL_HUMAN_HEIGHT_M,
            source=f"{rest_pose.source}+canonical_height={_CANONICAL_HUMAN_HEIGHT_M}m",
        )

    # SMPL-pruned rigs (parc_ms 15-bone, ankle-terminated, no toe joint)
    # measure ``height_m`` head-to-ankle, not head-to-floor — ankle joint
    # sits ~8cm above the sole that the robot's URDF mesh extends down
    # to.  That measurement-basis mismatch makes ``smpl_scale = robot_h
    # / human_h`` collapse toward 1.0 (e.g. 1.30/1.40 ≈ 0.93), so the
    # uniform-scaled yellow overlay and the MPC SQP target barely shrink
    # toward the robot — visually identical to the source skeleton.
    #
    # SMPL was designed against a **canonical adult human stature of
    # 1.7m**, and the bone-length distributions in PARC's ``parc_ms`` /
    # AMASS / OMOMO captures all match adult anthropometry (hip→knee
    # ≈ 0.42m, knee→ankle ≈ 0.41m).  Forcing ``human_height_assumption =
    # 1.7m`` for ankle-terminated rigs yields the same scaling regime
    # holosoma already produces (smpl_scale ≈ 0.76 for a 1.30m robot)
    # and matches the user's expectation that parc_ms data represents
    # full-stature humans rather than short ones.
    if is_smpl_pruned_ankle_terminated(clip.hierarchy.bone_names):
        rest_pose = _dc_replace(
            rest_pose,
            height_m=max(float(rest_pose.height_m), 1.7),
            source=f"{rest_pose.source}+smpl_pruned_height=1.7m",
        )

    return build_scaler_config_soma_style(
        calibration,
        model,
        rest_pose,
        reference_motion=clip,
        joint_scale_overrides=joint_scale_overrides,
    )


# ---------------------------------------------------------------------------
# Soma-style ScalerConfig builder (closed-form rest alignment)
# ---------------------------------------------------------------------------


def _forward_from_shoulder_axis(
    p_left_shoulder: NDArray,
    p_right_shoulder: NDArray,
) -> NDArray | None:
    """Physical forward direction from the shoulder axis, projected to XY.

    ``forward = cross(p_left - p_right, world_up)``  where world_up = +Z.
    Returns a unit vector in XY or ``None`` when the shoulder baseline is
    degenerate (co-located shoulders, vertical-only separation).
    """
    shoulder = p_left_shoulder - p_right_shoulder
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    fwd = np.cross(shoulder.astype(np.float32), up)
    fwd[2] = 0.0
    mag = float(np.linalg.norm(fwd))
    if mag < 1e-6:
        return None
    return (fwd / mag).astype(np.float32)


def _compute_body_heading_alignment(
    p_src_rest: NDArray,
    can2src: dict[str, str],
    rest_pose: "SourceRestPose",
    link_transforms: dict[str, NDArray],
    pairs: list[tuple[str, str]],
    p_rbt_root: NDArray,
    *,
    reference_motion: "Motion | None" = None,
) -> NDArray:
    """Yaw-only quaternion that pre-rotates source positions to align
    the source skeleton's physical forward with the robot's physical forward.

    Strategy: use the shoulder axis ``cross(L-R, up)`` on both source and
    robot rest poses to determine the physical facing direction, then compute
    the Z-axis rotation that maps source forward → robot forward.

    For position-only datasets (e.g. holosoma) the rest pose is a synthesized
    T-pose whose heading is arbitrary.  When *reference_motion* is provided
    and has identity quaternions, we derive the source forward from the actual
    motion's median shoulder positions instead of the synthetic T-pose.

    Falls back to identity when geometry is unavailable.
    """
    _id = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # --- Source forward ---
    ls_src = can2src.get("left_shoulder")
    rs_src = can2src.get("right_shoulder")
    src_fwd: NDArray | None = None

    # For position-only data, prefer computing forward from the actual motion
    # rather than the synthesized T-pose (whose heading is arbitrary).
    #
    # BVH bind-open clips (PKO, Xsens exports) also need this: frame-0
    # shoulders lie on a T-pose axis (forward ≈ ±Y) unrelated to the walk
    # direction in frames 1+, which otherwise yields a ~180° heading error
    # and corrupts arm IK targets for the whole clip.
    if reference_motion is not None and ls_src and rs_src:
        _q = np.asarray(reference_motion.quaternions, dtype=np.float32)
        _identity = np.zeros(4, dtype=np.float32)
        _identity[3] = 1.0
        _is_pos_only = float(np.abs(_q[:min(5, _q.shape[0])] - _identity[None, None, :]).max()) < 0.01
        _use_motion_fwd = _is_pos_only
        if not _use_motion_fwd:
            from hhtools.retarget.newton_basic.rest_pose import motion_frame0_is_bind_pose

            _use_motion_fwd = motion_frame0_is_bind_pose(reference_motion)
        if _use_motion_fwd:
            _bn = list(reference_motion.hierarchy.bone_names)
            if ls_src in _bn and rs_src in _bn:
                _li = _bn.index(ls_src)
                _ri = _bn.index(rs_src)
                _pos = np.asarray(reference_motion.positions, dtype=np.float32)
                _n_init = min(10, _pos.shape[0])
                _start = 1 if _n_init > 1 and not _is_pos_only else 0
                _p_l = np.mean(_pos[_start:_n_init, _li, :], axis=0)
                _p_r = np.mean(_pos[_start:_n_init, _ri, :], axis=0)
                src_fwd = _forward_from_shoulder_axis(_p_l, _p_r)
                if src_fwd is not None:
                    _log.debug(
                        "using motion-derived forward (frames %d..%d): %.1f°",
                        _start,
                        _n_init - 1,
                        float(np.degrees(np.arctan2(src_fwd[1], src_fwd[0]))),
                    )

    if src_fwd is None and ls_src and rs_src:
        li = rest_pose.index(ls_src)
        ri = rest_pose.index(rs_src)
        if li >= 0 and ri >= 0:
            src_fwd = _forward_from_shoulder_axis(p_src_rest[li], p_src_rest[ri])

    if src_fwd is None:
        _log.warning(
            "cannot determine source forward from shoulder geometry — "
            "body-frame alignment disabled; retarget may walk sideways."
        )
        return _id

    # --- Robot forward ---
    link_for_can = {c: ln for c, ln in pairs}
    ls_link = link_for_can.get("left_shoulder")
    rs_link = link_for_can.get("right_shoulder")
    rbt_fwd: NDArray | None = None
    if ls_link and rs_link:
        T_ls = link_transforms.get(ls_link)
        T_rs = link_transforms.get(rs_link)
        if T_ls is not None and T_rs is not None:
            rbt_fwd = _forward_from_shoulder_axis(
                T_ls[:3, 3].astype(np.float32),
                T_rs[:3, 3].astype(np.float32),
            )
    if rbt_fwd is None:
        # Fall back to URDF convention: robot faces +X.
        rbt_fwd = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # --- Yaw correction ---
    src_yaw = float(np.arctan2(src_fwd[1], src_fwd[0]))
    rbt_yaw = float(np.arctan2(rbt_fwd[1], rbt_fwd[0]))
    correction = rbt_yaw - src_yaw
    half = 0.5 * correction
    q_align = np.array(
        [0.0, 0.0, float(np.sin(half)), float(np.cos(half))],
        dtype=np.float32,
    )

    _log.debug(
        "body heading alignment: src_fwd=%.1f° rbt_fwd=%.1f° → correction=%.1f°",
        float(np.degrees(src_yaw)),
        float(np.degrees(rbt_yaw)),
        float(np.degrees(correction)),
    )
    return q_align


def build_scaler_config_soma_style(
    calibration: RobotRetargetCalibration,
    model: URDFRobotModel,
    rest_pose: "SourceRestPose",
    *,
    reference_motion: "Motion | None" = None,
    src_to_canonical: dict[str, str] | None = None,
    reference_pose: "HumanReferencePose | None" = None,
    joint_scale_overrides: dict[str, float] | None = None,
) -> "ScalerConfig":
    """Closed-form :class:`ScalerConfig` that lines the source rest up with
    the robot's calibrated rest exactly at frame 0.

    This is the replacement for :func:`build_scaler_config_from_calibration`
    we arrived at after auditing the gaps between the hhtools pipeline
    and the ``soma-retargeter`` reference pipeline (whose RP1 output we
    use as ground truth).  The key change is **what the scaler's
    ``joint_offsets`` are solved for**:

    * Previously (hierarchical mode) the scaler silently discarded any
      ``q_offset`` and scaled source positions purely by bone-length
      ratio.  Consequence: IK rotation targets drifted free of the
      robot's frame, and the pipeline had to disable rotation
      objectives and fall back to a yaw-only pelvis constraint.
    * Here we solve the classic closed-form alignment used by
      ``soma_retargeter.robotics.human_to_robot_scaler`` — scale,
      ``q_offset``, ``t_offset`` — so that at rest::

          q_out[j, 0]  = q_src_rest[j] · q_offset[j] = q_rbt_rest[j]
          t_out[j, 0]  = scale[j] · p_src_rest[j]
                       + rotate(q_rbt_rest[j], t_offset[j])
                       = p_rbt_rest[j]

      giving zero IK residual at frame 0 *for every effector,
      including rotation targets*.  Subsequent frames ride on top of
      this rest alignment, so the IK only has to express per-frame
      deltas — exactly the regime it's most stable in.

    Parameters
    ----------
    calibration
        The hand-dialled robot↔human alignment (same object consumed by
        the hierarchical builder).
    model
        Loaded :class:`URDFRobotModel` — we run FK at
        ``calibration.calibrated_joint_q`` internally.
    rest_pose
        Source skeleton rest snapshot
        (see :class:`~hhtools.retarget.newton_basic.rest_pose.SourceRestPose`).
        For AMASS / SOMA, this is usually ``rest_pose_from_motion(clip)``
        with the clip's first frame; for unit tests it can come from
        ``rest_pose_from_reference(canonical)``.
    reference_motion
        Required when ``calibration.reference`` is ``glb`` —
        the same :class:`~hhtools.core.motion.Motion` whose frame-0
        skeleton was used during calibration.  Ignored for static references.
        :func:`build_scaler_config_from_calibration` passes ``clip`` here.
    src_to_canonical
        Optional override for the source-→-canonical alias map.  When
        not supplied we auto-detect via
        :func:`hhtools.retarget.newton_basic.human_aliases.auto_source_to_canonical`
        the same way the hierarchical builder does.

    Returns
    -------
    ScalerConfig
        In ``scale_anchor="root"`` mode (soma-compatible) with populated
        ``joint_scales`` + ``joint_offsets`` keyed by **source** bone
        names (so it drops into ``HumanToRobotScaler(hierarchy, cfg,
        human_height=rest_pose.height_m)`` without further alias work).

    Notes
    -----
    * ``scale[root] = 1.0`` — the root's position already moves with the
      subject's global translation at every frame; scaling it would
      double-compress the motion.  The rest-height mismatch between
      source and robot is picked up by ``root_z_offset`` (applied
      uniformly post-scale) so feet land on the ground.
    * ``scale[j]`` for non-root joints is the anthropometric bone-length
      ratio ``|robot_disp_from_root| / |src_disp_from_root|``.  At rest
      this, combined with the ``t_offset`` closure, makes the scaled
      target land exactly on the robot link.  At non-rest frames the
      scaled displacement preserves the same proportional body scaling
      soma-retargeter uses — tall subjects don't stretch a short robot
      past its reach.
    * Joints not in the robot's ``ik_map`` keep ``scale = 1.0`` and
      identity offsets (effectively pass-through).  They don't appear
      in the IK map, so their effector targets are never consumed, but
      we include them in ``joint_scales`` so the scaler can still be
      asked for their raw trajectories via
      :class:`~hhtools.retarget.newton_basic.scaler.ScaledEffectors`.
    """

    from hhtools.retarget.newton_basic.config import ScalerConfig
    from hhtools.retarget.newton_basic.human_aliases import is_xsens_mocap_like
    from hhtools.retarget.newton_basic.rest_pose import SourceRestPose

    if not isinstance(rest_pose, SourceRestPose):
        raise TypeError(
            "rest_pose must be a SourceRestPose; see "
            "hhtools.retarget.newton_basic.rest_pose for constructors."
        )

    # ``reference_pose`` lets callers (e.g. the robot-to-robot pipeline) inject
    # a reference skeleton that is *not* one of the bundled human references —
    # the source robot's forward-kinematics rest pose, with canonical joint
    # names.  When omitted we fall back to the registered human reference keyed
    # by ``calibration.reference`` (the original behaviour).
    if reference_pose is not None:
        reference = reference_pose
    else:
        reference = _reference_pose_for_calibration(
            str(calibration.reference),
            motion=reference_motion,
        )
    pairs = _ik_map_pairs(model)

    # Robot-side rest FK at the calibrated joint configuration.
    saved_q = model.zero_configuration()
    try:
        link_transforms = _collect_link_transforms_at_q(
            model, calibration.calibrated_joint_q
        )
    finally:
        model.apply_configuration(saved_q)

    # Resolve the robot root link.  The reference's root_joint may be a
    # native name (e.g. "Hips", "pelvis") so we first map it to canonical
    # to look it up in the ik_map.
    _ref_root_canonical = reference.source_to_canonical.get(
        reference.root_joint, reference.root_joint,
    )
    robot_root_link = model.base_link
    for canonical, link_name in pairs:
        if canonical == _ref_root_canonical:
            robot_root_link = link_name
            break
    robot_root_T = link_transforms.get(
        robot_root_link, np.eye(4, dtype=np.float64)
    )
    p_rbt_root = robot_root_T[:3, 3].astype(np.float32)

    # Source-→-canonical alias map.  Resolution priority:
    #  1. Explicit caller-supplied override (src_to_canonical parameter)
    #  2. reference.source_to_canonical (hand-authored per-format map)
    #  3. Heuristic auto_source_to_canonical fallback
    bone_names = rest_pose.bone_names
    if src_to_canonical is not None:
        src2can = dict(src_to_canonical)
    elif reference.source_to_canonical:
        # Format references (e.g. ``lafan_bvh``) list only the core 17 joints.
        # BVHs with toe chains (``LeftToeEnd``, …) must still resolve through
        # :func:`auto_source_to_canonical` so scaler rows are not left at the
        # default scale ``1.0`` while the ankle uses ~0.3× — which stretches
        # ankle→toe preview segments and drops toes below the ground plane.
        auto_merged = _auto_source_to_canonical(tuple(bone_names))
        ref_map = reference.source_to_canonical
        src2can = {
            n: (ref_map[n] if n in ref_map else auto_merged.get(n, n))
            for n in bone_names
        }
    else:
        src2can = _auto_source_to_canonical(tuple(bone_names))

    # Canonical → source, priority: exact match > first alias wins.  Same
    # rule as the hierarchical builder so the two builders route the same
    # canonical joint to the same source bone on SMPL rigs (important
    # for the shoulder-vs-clavicle case, see the unit test regression).
    can2src: dict[str, str] = {}
    for src, can in src2can.items():
        if src == can:
            can2src[can] = src
    # Collect per-canonical candidates BEFORE the setdefault pass so we
    # can emit a descriptive warning when an alias map is ambiguous —
    # e.g. a SOMA-like rig where both ``LeftShoulder`` (clavicle) and
    # ``LeftArm`` (glenohumeral) aliased onto canonical ``left_shoulder``.
    # Silent first-wins used to pick the clavicle there, freezing the
    # entire arm at rest; we'd rather surface the ambiguity so the
    # alias author can drop the anatomically-wrong entry than ship a
    # broken retarget.  The warning fires only for canonicals the
    # robot's ik_map actually consumes — collisions on unused joints
    # (e.g. ``neck`` on a headless robot) are harmless noise.
    candidates: dict[str, list[str]] = {}
    for src, can in src2can.items():
        candidates.setdefault(can, []).append(src)
    ikmap_canonicals = {canon for canon, _ in pairs}
    for src, can in src2can.items():
        can2src.setdefault(can, src)
    for canonical_name in ikmap_canonicals:
        cand = candidates.get(canonical_name, [])
        if len(cand) > 1 and canonical_name not in cand:
            # ``canonical_name in cand`` means one source already matches
            # the canonical literally (exact-match pass above picked it);
            # no ambiguity in that case.  Otherwise first-wins picked the
            # first alias and we may have picked the wrong body part.
            _log.warning(
                "source→canonical alias collision: canonical %r has multiple "
                "source candidates %r (picked %r by first-wins). Drop the "
                "anatomically-wrong alias from the source→canonical map "
                "to avoid freezing this joint at rest.",
                canonical_name, cand, can2src.get(canonical_name),
            )
    for name in bone_names:
        can2src.setdefault(name, name)

    # Preload source rest positions / quats for fast lookup.
    p_src_rest = rest_pose.positions.astype(np.float32)
    q_src_rest = rest_pose.quaternions.astype(np.float32)

    # ---- Resolve the *anatomical* source root -------------------------
    # In many BVH skeletons (SOMA, LAFAN, …) the hierarchy root
    # (``rest_pose.root_name``) is a virtual transform node sitting at
    # the world origin (e.g. SOMA's ``Root`` at (0,0,0) with the Y→Z
    # rotation baked in).  The *anatomical* pelvis (``Hips``) is one
    # level deeper.  Using the virtual root as the scaler's anchor
    # produces catastrophically wrong scales (every displacement
    # includes the full pelvis height) and wrong q_offsets (the Y→Z
    # rotation bleeds through).
    #
    # We need the source bone that corresponds to the *robot's* root
    # canonical joint (``reference.root_joint``, usually ``"hips"``).
    # This mirrors soma-retargeter's ``human_root_name: "Hips"`` field
    # which points at the anatomical pelvis, never at the BVH virtual
    # root.
    src_root_canonical = _ref_root_canonical  # e.g. "hips"
    src_root_name = can2src.get(src_root_canonical)
    if src_root_name is None:
        # Fallback: if the alias map doesn't resolve the canonical
        # root, try the rest_pose's own root_name.
        src_root_name = rest_pose.root_name
        _log.warning(
            "canonical root %r has no source alias — falling back to "
            "rest_pose.root_name %r which may be a virtual BVH node; "
            "expect degraded scaling quality.",
            src_root_canonical, src_root_name,
        )
    src_root_idx = rest_pose.index(src_root_name)
    if src_root_idx < 0:
        raise ValueError(
            f"Resolved source root {src_root_name!r} (from canonical "
            f"{src_root_canonical!r}) not in rest_pose.bone_names"
        )

    # ---- Body-frame heading alignment ----------------------------------
    # The source skeleton's "forward" direction (in hhtools Z-up world)
    # may differ from the robot's "forward" (+X by URDF convention).
    # E.g. after Y-up→Z-up BVH import the SOMA character faces -Y while
    # the robot faces +X — a ~90° yaw mismatch that makes the retarget
    # "walk sideways".
    #
    # We derive the physical forward direction from skeleton geometry
    # (shoulder-axis cross product with world up) rather than from the
    # root quaternion's body-frame axes, because the BVH Y→Z conversion
    # can map the root's local +X onto world +Z (straight up), making
    # quaternion-based yaw extraction degenerate.
    q_body_align = _compute_body_heading_alignment(
        p_src_rest, can2src, rest_pose,
        link_transforms, pairs, p_rbt_root,
        reference_motion=reference_motion,
    )

    # Pre-rotate source rest data into the robot's heading frame.
    # Yaw-only rotation preserves Z values (important for root_z_offset).
    n_bones = len(p_src_rest)
    q_bc = np.broadcast_to(q_body_align[None, :], (n_bones, 4))
    p_src_aligned = Q.rotate(q_bc, p_src_rest).astype(np.float32)
    q_src_aligned = Q.multiply(q_bc, q_src_rest).astype(np.float32)

    p_src_root = p_src_aligned[src_root_idx]

    # The soma scaler kernel (``wp_compute_scaled_effectors``) computes
    # ``scaled_root_t`` ONCE per frame using the root's scale, then each
    # non-root joint's world-position target is
    #
    #     t_out[j, t] = (p_aligned[j, t] - p_aligned_root[t]) * scale[j]
    #                 + p_aligned_root[t] * scale[root]
    #                 + rotate(q_out[j, t], t_offset[j])
    #
    # Our scaler implements this under ``scale_anchor="root"``.
    scale_root = 1.0

    # Initialise joint_scales to identity for every source bone so the
    # scaler doesn't KeyError on joints outside the ik_map.  Offsets
    # default to identity by leaving them out of joint_offsets.
    #
    # Virtual root ancestors (body_world, Armature, …) above the
    # anatomical root are excluded: they have no meaningful calibration
    # mapping and including them causes the yellow scaler skeleton to
    # draw an unwanted segment from the world origin to the pelvis.
    ancestors_of_root: set[str] = set()
    _bone_to_idx = {n: i for i, n in enumerate(bone_names)}
    _walk_name: str | None = rest_pose.parent_names[src_root_idx]
    while _walk_name is not None:
        ancestors_of_root.add(_walk_name)
        _wi = _bone_to_idx.get(_walk_name)
        _walk_name = rest_pose.parent_names[_wi] if _wi is not None else None

    joint_scales: dict[str, float] = {
        name: 1.0
        for name in bone_names
        if name not in ancestors_of_root
    }
    joint_offsets: dict[
        str, tuple[tuple[float, float, float], tuple[float, float, float, float]]
    ] = {}

    # Xsens ``LeftHip`` / ``RightHip`` sit ~8 cm lateral to ``Hips`` (hip
    # joint marker), not at the thigh root like Mixamo ``LeftUpLeg`` or SOMA
    # ``LeftLeg``.  Root-relative Hips→LeftHip scaling (~2×) mismatches the
    # knee row (~1×) and distorts the scaled thigh vector — robots squat at
    # rest even when the yellow overlay looks fine.
    _xsens_like = is_xsens_mocap_like(bone_names)
    _xsens_hip_knee: dict[str, str] = {
        "left_hip": "left_knee",
        "right_hip": "right_knee",
    }
    _link_for_canonical: dict[str, str] = {canon: link for canon, link in pairs}

    resolved_canonicals: list[str] = []
    skipped_canonicals: list[str] = []
    for canonical, link_name in pairs:
        src_name = can2src.get(canonical)
        if src_name is None or src_name not in rest_pose.bone_names:
            skipped_canonicals.append(canonical)
            continue
        resolved_canonicals.append(canonical)

        src_idx = rest_pose.index(src_name)
        p_src_j = p_src_aligned[src_idx]
        q_src_j = q_src_aligned[src_idx]

        link_T = link_transforms.get(link_name)
        if link_T is None:
            continue
        p_rbt_j = link_T[:3, 3].astype(np.float32)
        q_rbt_j = _rotmat_to_xyzw(link_T[:3, :3]).astype(np.float32)

        # ---------- scale[j] ----------
        if canonical == src_root_canonical or src_name == src_root_name:
            scale_base = scale_root
        elif _xsens_like and canonical in _xsens_hip_knee:
            knee_canon = _xsens_hip_knee[canonical]
            knee_src = can2src.get(knee_canon)
            knee_link = _link_for_canonical.get(knee_canon)
            knee_T = link_transforms.get(knee_link) if knee_link else None
            if (
                knee_src
                and knee_src in rest_pose.bone_names
                and knee_T is not None
            ):
                knee_idx = rest_pose.index(knee_src)
                src_disp = p_src_aligned[knee_idx] - p_src_root
                rbt_disp = knee_T[:3, 3].astype(np.float32) - p_rbt_root
                s_norm = float(np.linalg.norm(src_disp))
                r_norm = float(np.linalg.norm(rbt_disp))
                if s_norm < 1e-4 or r_norm < 1e-6:
                    scale_base = 1.0
                else:
                    scale_base = r_norm / s_norm
            else:
                src_disp = p_src_j - p_src_root
                rbt_disp = p_rbt_j - p_rbt_root
                s_norm = float(np.linalg.norm(src_disp))
                r_norm = float(np.linalg.norm(rbt_disp))
                if s_norm < 1e-4 or r_norm < 1e-6:
                    scale_base = 1.0
                else:
                    scale_base = r_norm / s_norm
        else:
            src_disp = p_src_j - p_src_root
            rbt_disp = p_rbt_j - p_rbt_root
            s_norm = float(np.linalg.norm(src_disp))
            r_norm = float(np.linalg.norm(rbt_disp))
            if s_norm < 1e-4 or r_norm < 1e-6:
                scale_base = 1.0
            else:
                scale_base = r_norm / s_norm

        scale = scale_base
        overrides = joint_scale_overrides or {}
        if overrides:
            override = overrides.get(canonical)
            if override is None:
                override = overrides.get(src_name)
            if override is not None:
                scale = float(override)

        # ---------- q_rest_target[j] (what q_out lands on at rest) -------
        # Standard closed-form alignment lands the runtime IK rotation
        # target on the URDF link's calibrated world quaternion at rest:
        #
        #   q_rest_target[j] = q_rbt_rest[j]
        #
        # For ankles we additionally splice in the source rest skeleton's
        # foot-yaw splay — but only when the source rig actually exposes
        # a toe / foot-tip child of the ankle (LAFAN ``LeftToe`` / SMPL
        # ``left_foot`` / Mixamo ``LeftToe_End`` etc., all aliased to
        # canonical ``left_foot`` / ``right_foot``).  Robots like RP1
        # have no toe link in the URDF, so the IK can only constrain the
        # ankle's rotation; without lifting the splay onto the ankle
        # rotation target, the robot's foot stays pointed strictly
        # forward (URDF rest baseline) and the soles overlap at the
        # midline while the yellow source overlay shows the source's
        # natural toe-out stance.
        #
        # Yaw is derived from the **position** vector (ankle → toe) on
        # the body-aligned XY plane — purely geometric, no dependency on
        # source bone-basis conventions (which is why earlier attempts
        # to extract yaw from ``q_src_aligned[ankle]`` failed: SMPL /
        # Mixamo bones carry ~90° basis rotations that contaminate any
        # quaternion-domain yaw extraction).  Body forward is +X after
        # heading alignment, so ``atan2(dy, dx)`` is exactly the
        # outward-toe angle relative to the body.
        q_rest_target = q_rbt_j
        if canonical in ("left_ankle", "right_ankle"):
            _foot_canon = (
                "left_foot" if canonical == "left_ankle" else "right_foot"
            )
            _toe_src = can2src.get(_foot_canon)
            if (
                _toe_src
                and _toe_src != src_name
                and _toe_src in rest_pose.bone_names
            ):
                _toe_idx = rest_pose.index(_toe_src)
                _v = p_src_aligned[_toe_idx] - p_src_j
                _xy_norm_sq = float(_v[0] * _v[0] + _v[1] * _v[1])
                if _xy_norm_sq > 1e-6:
                    _alpha = float(np.arctan2(float(_v[1]), float(_v[0])))
                    _half = 0.5 * _alpha
                    q_src_yaw = np.array(
                        [0.0, 0.0, float(np.sin(_half)), float(np.cos(_half))],
                        dtype=np.float32,
                    )
                    q_rest_target = Q.normalize(
                        Q.multiply(q_src_yaw[None, :], q_rbt_j[None, :])
                    )[0]

        # ---------- q_offset[j] ----------
        # q_out[j, 0] = q_src_aligned[j] * q_offset[j] =! q_rest_target[j]
        # →  q_offset[j] = conj(q_src_aligned[j]) · q_rest_target[j]
        # Source-side bone-basis conventions (SMPL +Z-forward, Mixamo
        # bone-+Y-along-bone) are absorbed cleanly by the
        # ``conj(q_src_aligned)`` factor regardless of which target we
        # land on at rest.
        q_offset = Q.normalize(
            Q.multiply(
                Q.conjugate(q_src_j[None, :]),
                q_rest_target[None, :],
            )
        )[0]

        # ---------- t_offset[j]  (root-relative residual) ----------
        # Rest closure at runtime is:
        #
        #   t_out[j, 0] = (p_src_rest[j] − p_src_root) · scale[j]
        #               + p_rbt_root
        #               + rotate(q_rest_target[j], t_offset[j])
        #               =! p_rbt_rest[j]
        #
        # so  t_offset[j] = rotate(conj(q_rest_target[j]), residual)
        # with residual = (p_rbt[j] − p_rbt_root) − scale[j] · aligned_src_disp.
        #
        # NOTE on motion-frame position behaviour: ``rotate(q_out, t_offset)``
        # at any frame equals ``rotate(motion_delta, residual)`` — the
        # ``q_rest_target`` factor cancels exactly between the q_offset
        # and t_offset construction.  So the splay yaw on ankles affects
        # only the IK rotation target, not its position target.
        rbt_disp = p_rbt_j - p_rbt_root
        # Solve t_offset against the *final* scale (including any
        # ``joint_scale_multipliers`` override) so frame-0 rest closure
        # still holds.  Proportion tweaks then affect motion-frame
        # displacements; upper-body lateral narrowing is handled post-IK.
        residual = rbt_disp - np.float32(scale) * (p_src_j - p_src_root)
        t_offset = Q.rotate(
            Q.conjugate(q_rest_target[None, :]),
            residual.astype(np.float32)[None, :],
        )[0]

        joint_scales[src_name] = float(scale)
        joint_offsets[src_name] = (
            (float(t_offset[0]), float(t_offset[1]), float(t_offset[2])),
            (
                float(q_offset[0]),
                float(q_offset[1]),
                float(q_offset[2]),
                float(q_offset[3]),
            ),
        )

    # Toe / foot-direction bones: when ``left_foot`` / ``right_foot`` are
    # absent from ``ik_map`` (typical humanoids), extra source joints keep
    # the initial ``joint_scales[·] == 1.0`` while ankles were calibrated to
    # a much smaller ratio — preview-only foot segments blow up.  Copy the
    # solved ankle scale + offset onto every source bone mapped to those
    # canonicals that still sit at the default scale (≈1).
    _foot_canon = frozenset({"left_foot", "right_foot"})
    _ankle_for: dict[str, str] = {
        "left_foot": "left_ankle",
        "right_foot": "right_ankle",
    }
    for src_name in bone_names:
        canon = src2can.get(src_name, src_name)
        if canon not in _foot_canon:
            continue
        if abs(float(joint_scales.get(src_name, 1.0)) - 1.0) > 1e-5:
            continue
        ankle_canon = _ankle_for[canon]
        src_ankle = can2src.get(ankle_canon)
        if not src_ankle or src_ankle not in joint_scales:
            continue
        joint_scales[src_name] = float(joint_scales[src_ankle])
        if src_name not in joint_offsets and src_ankle in joint_offsets:
            joint_offsets[src_name] = joint_offsets[src_ankle]

    # ---- Propagate mapped scales to unmapped descendants -------------------
    # Joints outside the ik_map still sit at the 1.0 default.  For skeletons
    # with many detail bones (fingers, spine intermediates, FootMod markers)
    # this creates a visible distortion in the interaction-mesh target: human-
    # scale extremities attached to robot-scale anchors.  Walk the parent
    # chain from every still-default joint to find the nearest ancestor that
    # was calibrated and inherit its scale.  The offset stays identity (only
    # the scale matters for the Laplacian target mesh).
    _name_to_idx: dict[str, int] = {n: i for i, n in enumerate(bone_names)}
    for src_name in bone_names:
        if abs(float(joint_scales.get(src_name, 1.0)) - 1.0) > 1e-5:
            continue
        # Walk up the parent chain until we hit a calibrated (non-default)
        # ancestor or the root.
        cur = rest_pose.parent_names[_name_to_idx[src_name]]
        while cur is not None:
            if abs(float(joint_scales.get(cur, 1.0)) - 1.0) > 1e-5:
                joint_scales[src_name] = float(joint_scales[cur])
                break
            pidx = _name_to_idx.get(cur)
            cur = rest_pose.parent_names[pidx] if pidx is not None else None

    # ---- Mapping coverage diagnostics -------------------------------------
    total_ik = len(pairs)
    n_resolved = len(resolved_canonicals)
    if skipped_canonicals:
        _log.warning(
            "ScalerConfig mapping coverage: %d/%d ik_map joints resolved. "
            "Skipped (no source bone found): %s. Source rig root=%r, "
            "%d bones. Scales for skipped joints default to 1.0.",
            n_resolved, total_ik, skipped_canonicals,
            src_root_name, len(bone_names),
        )
    else:
        _log.info(
            "ScalerConfig mapping: all %d ik_map joints resolved "
            "(source root=%r, %d bones).",
            total_ik, src_root_name, len(bone_names),
        )

    # ---- Ground-alignment shift -------------------------------------------
    # After body alignment, p_src_root[2] is unchanged (yaw preserves Z).
    # ``NewtonBasicPipeline`` floor-normalises clips (feet to z=0) before
    # scaling, so the vertical reference for the pelvis must be measured
    # from the foot plane, not the raw BVH root height (Xsens / SOMA /
    # LAFAN rest poses often float ~10 cm above the lowest foot contact).
    from hhtools.core.grounding import foot_floor_z_in_positions

    z_floor = float(
        foot_floor_z_in_positions(p_src_rest, tuple(bone_names))
    )
    p_src_root_z = float(p_src_root[2]) - z_floor
    robot_pelvis_height: float | None = None
    try:
        h_robot_pelvis_rest = _estimate_robot_pelvis_height_at_q(
            model, calibration.calibrated_joint_q, robot_root_link
        )
        root_z_offset = float(h_robot_pelvis_rest) - p_src_root_z * scale_root
        robot_pelvis_height = float(h_robot_pelvis_rest)
    except Exception:  # noqa: BLE001 — URDF without visual meshes etc.
        root_z_offset = 0.0

    from hhtools.robot.standing_height import estimate_robot_standing_height

    robot_height = estimate_robot_standing_height(
        model, calibration.calibrated_joint_q,
    )
    human_height_assumption = float(rest_pose.height_m)

    source_body_quat = (
        float(q_body_align[0]),
        float(q_body_align[1]),
        float(q_body_align[2]),
        float(q_body_align[3]),
    )

    _ref_key = str(calibration.reference).lower()
    # Ankle ``q_offset`` is solved from ``LeftFoot`` rest quats with a geometric
    # toe-yaw splice.  Runtime toe-quat injection on the ankle row breaks that
    # closure and can flip soles backward on RP1.
    _lafan_toe = False if _ref_key == "lafan_bvh" else None

    return ScalerConfig(
        human_height_assumption=human_height_assumption,
        model_height=float(robot_height),
        joint_scales=joint_scales,
        joint_offsets=joint_offsets,
        root_joint=src_root_name,
        scale_anchor="root",
        root_z_offset=root_z_offset,
        robot_pelvis_height=robot_pelvis_height,
        source_body_quat=source_body_quat,
        lafan_foot_mod_use_toe_orientation=_lafan_toe,
    )


def _estimate_robot_pelvis_height_at_q(
    model: URDFRobotModel, joint_q: dict[str, float], pelvis_link: str,
) -> float:
    """Vertical distance from the robot's pelvis link origin to its lowest
    mesh vertex when posed at ``joint_q`` with floating-base identity.

    This is the robot's natural "how high above the ground should the
    pelvis sit" distance — used by calibration to shift source pelvis
    Z so the robot stands on the floor instead of floating in mid-air.

    Implementation: iterate every visual mesh's transformed vertices
    (same routine as :func:`_estimate_robot_tpose_height` but at the
    calibrated configuration instead of zero), find the global minimum
    Z, and subtract it from the pelvis link's world Z.  Falls back to
    the robot's overall standing height if the pelvis link isn't in
    the scene graph (degenerate URDF).
    """

    model.apply_configuration(joint_q)
    try:
        scene = model.trimesh_scene(collision=False)
    except Exception:
        return 0.73

    import trimesh as _trimesh

    min_z: float | None = None
    for node_name in scene.graph.nodes_geometry:
        mat, geom_name = scene.graph[node_name]
        if geom_name is None:
            continue
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, _trimesh.Trimesh) or geom.is_empty:
            continue
        v = np.asarray(geom.vertices, dtype=np.float64)
        z = mat[2, 0] * v[:, 0] + mat[2, 1] * v[:, 1] + mat[2, 2] * v[:, 2] + mat[2, 3]
        zmin = float(z.min())
        min_z = zmin if min_z is None or zmin < min_z else min_z

    link_transforms = _collect_link_transforms_at_q(model, joint_q)
    pelvis_T = link_transforms.get(pelvis_link)
    if pelvis_T is None:
        pelvis_T = np.eye(4, dtype=np.float64)
    pelvis_z = float(pelvis_T[2, 3])

    if min_z is None:
        # No visual geometry (pure capsule URDF etc.) — fall back to the
        # aggregate standing height, treating pelvis as sitting at half.
        return max(1e-3, 0.5 * _estimate_robot_tpose_height(model))
    return max(1e-3, pelvis_z - min_z)


def _estimate_source_pelvis_height(
    clip, src_root_idx: int, foot_indices: list[int]
) -> float:
    """Vertical distance from the source clip's pelvis to its lowest
    foot position at frame 0.

    AMASS / SMPL walkers land with feet at Z ≈ 0 so this is ~ pelvis_Z
    in practice, but we compute it properly in case a subject starts
    mid-air (jump clips, stairs) — the calibration shift should zero
    out at the *resting* pelvis-to-foot height, not the current one.

    If ``foot_indices`` is empty (source rig missing ankles / toes
    altogether, very unusual), we fall back to the pelvis Z itself
    which is reasonable for humans standing on the ground plane.
    """

    frame0 = np.asarray(clip.positions[0], dtype=np.float64)
    pelvis_z = float(frame0[src_root_idx, 2])
    if not foot_indices:
        return max(1e-3, pelvis_z)
    foot_zs = frame0[np.array(foot_indices, dtype=np.int32), 2]
    lowest_foot_z = float(foot_zs.min())
    return max(1e-3, pelvis_z - lowest_foot_z)


# Canonical joints considered "upper body" for headless-robot stature matching.
# Order is highest-first so we pick the topmost joint the robot actually maps.
_UPPER_BODY_STATURE_PRIORITY: tuple[str, ...] = (
    "head",
    "neck",
    "chest",
    "spine",
)


def _robot_is_headless(pairs: list[tuple[str, str]]) -> bool:
    """True when the robot ``ik_map`` has no ``head`` target (e.g. RP1)."""
    return "head" not in {c for c, _ in pairs}


def _highest_mapped_canonical(pairs: list[tuple[str, str]]) -> str | None:
    mapped = {c for c, _ in pairs}
    for canon in _UPPER_BODY_STATURE_PRIORITY:
        if canon in mapped:
            return canon
    return None


def _estimate_source_mapped_stature(
    rest_pose: "SourceRestPose",
    can2src: dict[str, str],
    pairs: list[tuple[str, str]],
) -> float | None:
    """Ground-to-highest-mapped-joint height on the source rest skeleton.

    For headless robots (no ``head`` in ``ik_map``) the full skeleton
    ``height_m`` (pelvis→head) over-estimates the human side of
    ``robot_height / human_height``, making the yellow overlay taller than
    the robot even though IK only tracks up to ``chest``.  Measuring to
    the highest *mapped* landmark (typically ``chest``) aligns the uniform
    scale with what the retargeter actually constrains.
    """
    apex_canon = _highest_mapped_canonical(pairs)
    if apex_canon is None:
        return None
    src_apex = can2src.get(apex_canon)
    if not src_apex or src_apex not in rest_pose.bone_names:
        return None

    positions = rest_pose.positions.astype(np.float32)
    apex_z = float(positions[rest_pose.index(src_apex), 2])

    mapped_canons = {c for c, _ in pairs}
    ground_z = float(positions[:, 2].min())
    for foot_canon in ("left_ankle", "right_ankle", "left_foot", "right_foot"):
        if foot_canon not in mapped_canons:
            continue
        src_foot = can2src.get(foot_canon)
        if src_foot and src_foot in rest_pose.bone_names:
            ground_z = min(ground_z, float(positions[rest_pose.index(src_foot), 2]))

    return max(1e-3, apex_z - ground_z)


def uniform_overlay_scale(
    scaler_cfg: "ScalerConfig",
    human_height: float,
    *,
    ik_map_keys: frozenset[str] | set[str] | None = None,
) -> float:
    """Scale factor for the yellow skeleton / scene-object uniform overlay.

    Always ``model_height / human_height`` where ``human_height`` is the
    runtime subject height (UI field) falling back to
    :attr:`ScalerConfig.human_height_assumption`.  Headless robots hide
    unmapped head/neck in the preview renderer instead of shrinking the
    uniform scale to mapped-landmark height — that made the overlay too
    small relative to the full robot mesh.
    """
    _ = ik_map_keys
    robot_h = float(scaler_cfg.model_height)
    h_run = float(human_height)
    if h_run < 0.1:
        h_run = float(scaler_cfg.human_height_assumption)
    return robot_h / max(1e-3, h_run)


def _motion_overlay_stature_m(
    motion,
    *,
    headless: bool = False,
) -> float | None:
    """Robust clip stature for yellow-overlay scaling (metres, foot → apex).

    Uses the foot floor (not the global joint ``z`` minimum — hands /
    props can sit below the feet and would shrink the overlay) and the
    **median** per-frame max joint height.  A single reach / jump frame
    must not set the scale for the whole clip.
    """
    from hhtools.core.grounding import human_source_floor_z_world
    from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical

    pos = np.asarray(motion.positions, dtype=np.float32)
    if pos.size == 0:
        return None
    z_floor = float(human_source_floor_z_world(motion))
    bone_names = tuple(motion.hierarchy.bone_names)
    if headless:
        src2can = auto_source_to_canonical(bone_names)
        keep = [
            i
            for i, n in enumerate(bone_names)
            if src2can.get(n, n) not in ("head", "neck")
        ]
        if not keep:
            keep = list(range(len(bone_names)))
        rel_z = pos[:, keep, 2] - z_floor
    else:
        rel_z = pos[:, :, 2] - z_floor
    per_frame = rel_z.max(axis=1)
    good = per_frame > 0.5
    if not np.any(good):
        return None
    return float(np.median(per_frame[good]))


def uniform_overlay_scale_for_motion(
    scaler_cfg: "ScalerConfig",
    human_height: float,
    motion,
    *,
    ik_map_keys: frozenset[str] | set[str] | None = None,
) -> float:
    """Scale factor for the yellow overlay on a concrete clip.

    Uses the same ``model_height / human_height`` rule as
    :func:`uniform_overlay_scale` — ``human_height`` falls back to
    :attr:`~hhtools.retarget.newton_basic.config.ScalerConfig.human_height_assumption`
    (pinned to 1.65 m for SMPL / LAFAN / SOMA references).

    GLB clips use per-clip measured stature so overlay stride length matches
    the IK scaler's ``trajectory_scale`` (see
    :func:`effective_retarget_human_height`).
    """
    _ = ik_map_keys
    robot_h = float(scaler_cfg.model_height)
    h_run = effective_retarget_human_height(human_height, motion, scaler_cfg)
    return robot_h / max(1e-3, h_run)


def effective_retarget_human_height(
    human_height: float,
    motion,
    scaler_cfg: "ScalerConfig | None" = None,
) -> float:
    """Runtime human stature for trajectory / overlay scaling on a clip.

    GLB subjects vary in rig height; using the canonical 1.65 m assumption
    shrinks world root travel (~0.73×) while the yellow overlay already uses
    measured height (~0.86×), so the robot walks shorter than the overlay.
    """
    h_run = float(human_height)
    if h_run < 0.1 and scaler_cfg is not None:
        h_run = float(scaler_cfg.human_height_assumption)
    if motion is not None and str(getattr(motion, "source_format", "")).lower() == "glb":
        from hhtools.retarget.newton_basic.rest_pose import estimate_skeleton_height

        measured = float(estimate_skeleton_height(motion))
        if measured > 0.5:
            h_run = measured
    return h_run


def _estimate_robot_mapped_stature(
    model: URDFRobotModel,
    joint_q: dict[str, float],
    pairs: list[tuple[str, str]],
) -> float | None:
    """Ground-to-highest-mapped-link height on the robot at ``joint_q``."""
    apex_canon = _highest_mapped_canonical(pairs)
    if apex_canon is None:
        return None
    link_by_canon = dict(pairs)
    apex_link = link_by_canon.get(apex_canon)
    if not apex_link:
        return None

    saved_q = model.zero_configuration()
    try:
        link_T = _collect_link_transforms_at_q(model, joint_q)
        apex_T = link_T.get(apex_link)
        if apex_T is None:
            return None
        apex_z = float(apex_T[2, 3])

        try:
            scene = model.trimesh_scene(collision=False)
        except Exception:
            return None

        import trimesh as _trimesh

        min_z: float | None = None
        for _node_name in scene.graph.nodes_geometry:
            mat, geom_name = scene.graph[_node_name]
            if geom_name is None:
                continue
            geom = scene.geometry.get(geom_name)
            if not isinstance(geom, _trimesh.Trimesh) or geom.is_empty:
                continue
            v = np.asarray(geom.vertices, dtype=np.float64)
            z = (
                mat[2, 0] * v[:, 0]
                + mat[2, 1] * v[:, 1]
                + mat[2, 2] * v[:, 2]
                + mat[2, 3]
            )
            zmin = float(z.min())
            min_z = zmin if min_z is None or zmin < min_z else min_z
        if min_z is None:
            return None
        return max(1e-3, apex_z - min_z)
    finally:
        model.apply_configuration(saved_q)


def _estimate_robot_tpose_height(model: URDFRobotModel) -> float:
    """Standing height at zero configuration — see :func:`estimate_robot_standing_height`."""
    from hhtools.robot.standing_height import estimate_robot_standing_height

    return estimate_robot_standing_height(model, model.zero_configuration())
