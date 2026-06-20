# SPDX-License-Identifier: Apache-2.0
"""Human joint naming aliases used by the retargeting pipeline.

The hhtools :class:`~hhtools.retarget.newton_basic.scaler.HumanToRobotScaler`
emits effector targets keyed by *source* joint names — i.e. whatever the
input :class:`~hhtools.core.hierarchy.Hierarchy` uses (SMPL / SMPL-H /
SMPL-X, Mixamo, BVH, …).  Robot presets in ``configs/robots/**/robot.yaml``
reference the *canonical* hhtools skeleton (``configs/skeleton_presets
/canonical_human.yaml``) which uses names like ``hips`` / ``chest`` rather
than ``pelvis`` / ``spine3``.

This module provides the small translation tables needed to bridge the two
sides so that a pipeline user doesn't have to hand-author a rename map per
dataset.  The aliases map *source → canonical* (one direction only); we
never force canonical names back onto the source hierarchy.

Supported rig families (auto-detected in priority order):

1. **SMPL / SMPL-H / SMPL-X** — lowercase ``pelvis``, ``spine1`` … plus OMOMO-style
   abbreviations (``l_hip``, ``l_shoulder``, …) on 24-joint windows.
2. **SOMA BVH** — TitleCase ``Hips`` + ``LeftLeg``/``LeftShin`` …
3. **Xsens mocap BVH** — ``Hips`` + ``LeftHip``/``LeftKnee``/``LeftAnkle`` +
   multi-segment ``Chest`` …
4. **meshmimic / holosoma** — Mixamo-style names with ``LeftFootMod`` /
   ``RightFootMod`` sole markers (same joint rename table as Mixamo/CMU).
5. **Mixamo / CMU / LAFAN BVH** — TitleCase ``Hips`` + ``LeftUpLeg``/``LeftLeg`` …
6. **Prefix-stripped fuzzy match** — e.g. ``b_l_arm`` → ``left_shoulder``
7. **User-defined YAML overrides** in ``configs/skeleton_presets/alias_maps/``
8. **Identity** — names forwarded as-is (unknown rigs surface a clear ``KeyError``)
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# 1. SMPL / SMPL-H / SMPL-X
# ---------------------------------------------------------------------------

SMPL_BODY_TO_CANONICAL: Mapping[str, str] = {
    "pelvis": "hips",
    "spine1": "spine",
    "spine2": "spine",   # collapse mid-spine to the canonical single 'spine'
    "spine3": "chest",
    "neck": "neck",
    "head": "head",
    # ``left_collar`` / ``right_collar`` are **not** mapped onto canonical
    # ``left_shoulder`` / ``right_shoulder``: those canonical slots are the
    # glenohumeral joints (``left_shoulder`` / ``right_shoulder`` SMPL bones).
    # Collapsing collar into the same canonical as the arm root made every
    # consumer that walked ``joint_scales`` key order pick the clavicle row
    # first, producing raised-arm IK targets.  Collar bones stay identity
    # (``left_collar`` → ``left_collar``) and are ignored by typical ik_maps.
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
    "left_foot": "left_foot",
    "right_foot": "right_foot",
}

# SMPL-H / OMOMO window extracts use abbreviated names (``l_hip``, ``l_shoulder``).
# These must map like full SMPL keys so :func:`auto_source_to_canonical` stays
# consistent when :func:`is_smpl_like` widens beyond AMASS-style spellings.
SMPL_H_ABBR_TO_CANONICAL: Mapping[str, str] = {
    "l_hip": "left_hip",
    "r_hip": "right_hip",
    "l_knee": "left_knee",
    "r_knee": "right_knee",
    "l_ankle": "left_ankle",
    "r_ankle": "right_ankle",
    "l_foot": "left_foot",
    "r_foot": "right_foot",
    "l_shoulder": "left_shoulder",
    "r_shoulder": "right_shoulder",
    "l_elbow": "left_elbow",
    "r_elbow": "right_elbow",
    "l_wrist": "left_wrist",
    "r_wrist": "right_wrist",
    "l_hand": "left_wrist",
    "r_hand": "right_wrist",
}


# ---------------------------------------------------------------------------
# 2. SOMA BVH (TitleCase, uses LeftLeg=hip, LeftShin=knee)
# ---------------------------------------------------------------------------

# See original docstring in the repo history for the full rationale behind
# clavicle / mid-spine / neck omissions.
SOMA_BVH_TO_CANONICAL: Mapping[str, str] = {
    "Hips": "hips",
    "Spine1": "spine",
    "Chest": "chest",
    "Neck1": "neck",
    "Head": "head",
    "LeftArm": "left_shoulder",
    "LeftForeArm": "left_elbow",
    "LeftHand": "left_wrist",
    "RightArm": "right_shoulder",
    "RightForeArm": "right_elbow",
    "RightHand": "right_wrist",
    "LeftLeg": "left_hip",
    "LeftShin": "left_knee",
    "LeftFoot": "left_ankle",
    "LeftToeBase": "left_foot",
    "RightLeg": "right_hip",
    "RightShin": "right_knee",
    "RightFoot": "right_ankle",
    "RightToeBase": "right_foot",
    # Toe tip (when present) overwrites ``LeftToeBase`` in preview packing so
    # ankle→toe direction matches the longest segment on the rig.
    "LeftToeEnd": "left_foot",
    "RightToeEnd": "right_foot",
}


# ---------------------------------------------------------------------------
# 3. Xsens MVN / biomechanics BVH (Hip/Knee/Ankle leg chain)
# ---------------------------------------------------------------------------
# Common in Xsens MVN exports and clinical gait pipelines.  Leg joints use
# ``LeftHip``/``LeftKnee``/``LeftAnkle`` (not Mixamo ``LeftUpLeg``/``LeftLeg``
# nor SOMA ``LeftLeg``/``LeftShin``).  The spine is a multi-segment ``Chest``
# chain; ``Chest4`` parents the shoulders and is mapped to canonical ``chest``.

XSENS_MOCAP_TO_CANONICAL: Mapping[str, str] = {
    "Hips": "hips",
    "Chest": "spine",
    "Chest4": "chest",
    "Neck": "neck",
    "Head": "head",
    "LeftCollar": "left_collar",
    "LeftShoulder": "left_shoulder",
    "LeftElbow": "left_elbow",
    "LeftWrist": "left_wrist",
    "RightCollar": "right_collar",
    "RightShoulder": "right_shoulder",
    "RightElbow": "right_elbow",
    "RightWrist": "right_wrist",
    "LeftHip": "left_hip",
    "LeftKnee": "left_knee",
    "LeftAnkle": "left_ankle",
    "LeftToe": "left_foot",
    "RightHip": "right_hip",
    "RightKnee": "right_knee",
    "RightAnkle": "right_ankle",
    "RightToe": "right_foot",
}


# ---------------------------------------------------------------------------
# 4. Mixamo / CMU / LAFAN BVH convention
# ---------------------------------------------------------------------------
# Very common in motion capture databases and Mixamo auto-rigged characters.
# Key anatomical difference from SOMA:
#   LeftUpLeg = hip,  LeftLeg = knee  (vs SOMA: LeftLeg = hip, LeftShin = knee)
#
# Clavicle (``LeftShoulder``) is intentionally omitted on *plain* Mixamo exports
# because the arm chain starts at ``LeftArm`` — but meshmimic/holosoma and
# several CMU rigs expose ``LeftShoulder`` as a real clavicle bone, so we map
# it to ``left_collar`` (same convention as SMPL) rather than leaving it
# unmapped (which broke scaler / canonical packing on holosoma clips).
# Spine chain: ``Spine`` → spine root, ``Spine1`` → spine (mid), ``Spine2`` → chest.

MIXAMO_CMU_TO_CANONICAL: Mapping[str, str] = {
    "Hips": "hips",
    "Spine": "spine",
    "Spine1": "spine",
    "Spine2": "chest",
    "Neck": "neck",
    "Head": "head",
    "LeftShoulder": "left_collar",
    "RightShoulder": "right_collar",
    # Arm chain — same naming as SOMA
    "LeftArm": "left_shoulder",
    "LeftForeArm": "left_elbow",
    "LeftHand": "left_wrist",
    "RightArm": "right_shoulder",
    "RightForeArm": "right_elbow",
    "RightHand": "right_wrist",
    # Leg chain — the crucial difference from SOMA
    "LeftUpLeg": "left_hip",
    "LeftLeg": "left_knee",
    "LeftFoot": "left_ankle",
    "LeftToe": "left_foot",
    "LeftToeBase": "left_foot",      # some rigs use ToeBase instead of Toe
    "LeftToe_End": "left_foot",
    "RightUpLeg": "right_hip",
    "RightLeg": "right_knee",
    "RightFoot": "right_ankle",
    "RightToe": "right_foot",
    "RightToeBase": "right_foot",
    "RightToe_End": "right_foot",
    "LeftToeEnd": "left_foot",
    "RightToeEnd": "right_foot",
}


# ---------------------------------------------------------------------------
# 4. Fuzzy / normalized canonical fragments
# ---------------------------------------------------------------------------
# After stripping known prefixes (``mixamorig:``, ``b_``, ``Bip01 ``, etc.)
# and normalising to lowercase + underscores, we try to match against this
# table.  Entries are ordered longest-first so ``left_upleg`` matches before
# ``left``.

_NORMALISED_FRAGMENTS: dict[str, str] = {
    # Root
    "hips": "hips",
    "root": "hips",
    "pelvis": "hips",
    # Spine
    "spine": "spine",
    "spine0": "spine",
    "spine1": "spine",
    # Chest / torso
    "chest": "chest",
    "spine2": "chest",
    "spine3": "chest",
    "upper_chest": "chest",
    "upperchest": "chest",
    "torso": "chest",
    "upper_torso": "chest",
    "lower_torso": "spine",
    # Neck / head
    "neck": "neck",
    "neck0": "neck",
    "head": "head",
    # Left arm
    "l_shoulder": "left_shoulder",
    "left_shoulder": "left_shoulder",
    "l_arm": "left_shoulder",
    "left_arm": "left_shoulder",
    "l_upper_arm": "left_shoulder",
    "left_upper_arm": "left_shoulder",
    "l_forearm": "left_elbow",
    "left_forearm": "left_elbow",
    "l_lower_arm": "left_elbow",
    "left_lower_arm": "left_elbow",
    "l_elbow": "left_elbow",
    "left_elbow": "left_elbow",
    "l_wrist": "left_wrist",
    "left_wrist": "left_wrist",
    "l_hand": "left_wrist",
    "left_hand": "left_wrist",
    # Right arm
    "r_shoulder": "right_shoulder",
    "right_shoulder": "right_shoulder",
    "r_arm": "right_shoulder",
    "right_arm": "right_shoulder",
    "r_upper_arm": "right_shoulder",
    "right_upper_arm": "right_shoulder",
    "r_forearm": "right_elbow",
    "right_forearm": "right_elbow",
    "r_lower_arm": "right_elbow",
    "right_lower_arm": "right_elbow",
    "r_elbow": "right_elbow",
    "right_elbow": "right_elbow",
    "r_wrist": "right_wrist",
    "right_wrist": "right_wrist",
    "r_hand": "right_wrist",
    "right_hand": "right_wrist",
    # Left leg
    "l_upleg": "left_hip",
    "left_upleg": "left_hip",
    "l_thigh": "left_hip",
    "left_thigh": "left_hip",
    "l_hip": "left_hip",
    "left_hip": "left_hip",
    "l_leg": "left_knee",
    "left_leg": "left_knee",
    "l_shin": "left_knee",
    "left_shin": "left_knee",
    "l_knee": "left_knee",
    "left_knee": "left_knee",
    "l_foot": "left_ankle",
    "left_foot": "left_ankle",
    "l_ankle": "left_ankle",
    "left_ankle": "left_ankle",
    "l_toe": "left_foot",
    "left_toe": "left_foot",
    # Right leg
    "r_upleg": "right_hip",
    "right_upleg": "right_hip",
    "r_thigh": "right_hip",
    "right_thigh": "right_hip",
    "r_hip": "right_hip",
    "right_hip": "right_hip",
    "r_leg": "right_knee",
    "right_leg": "right_knee",
    "r_shin": "right_knee",
    "right_shin": "right_knee",
    "r_knee": "right_knee",
    "right_knee": "right_knee",
    "r_foot": "right_ankle",
    "right_foot": "right_ankle",
    "r_ankle": "right_ankle",
    "right_ankle": "right_ankle",
    "r_toe": "right_foot",
    "right_toe": "right_foot",
}

_PREFIX_RE = re.compile(
    r"^(?:mixamorig[_:]|b_|bip01[_ ]?|character1[_:]?|"
    r"sk_|skel_|bone_|jnt_|def_)",
    re.IGNORECASE,
)


def _normalise_joint_name(name: str) -> str:
    """Strip common DCC prefixes, collapse to lowercase + underscores."""
    stripped = _PREFIX_RE.sub("", name)
    return re.sub(r"[\s\-\.]+", "_", stripped).strip("_").lower()


# ---------------------------------------------------------------------------
# 5. User-defined YAML overrides
# ---------------------------------------------------------------------------

_yaml_alias_cache: dict[str, dict[str, str]] | None = None


def _load_yaml_alias_maps(
    search_dir: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Load ``configs/skeleton_presets/alias_maps/*.yaml`` files.

    Each YAML file should contain a flat ``source_name: canonical_name``
    mapping.  The file stem becomes the alias-map name.
    """
    global _yaml_alias_cache  # noqa: PLW0603
    if _yaml_alias_cache is not None:
        return _yaml_alias_cache

    if search_dir is None:
        here = Path(__file__).resolve().parent
        search_dir = here.parent.parent.parent / "configs" / "skeleton_presets" / "alias_maps"

    result: dict[str, dict[str, str]] = {}
    if search_dir.is_dir():
        import yaml  # lazy — avoids import cost if no YAML files exist

        for p in sorted(search_dir.glob("*.yaml")):
            if p.stem.startswith("_"):
                continue
            try:
                data: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    result[p.stem] = {str(k): str(v) for k, v in data.items()}
            except Exception:
                pass

    _yaml_alias_cache = result
    return result


def _try_yaml_override(joint_names: tuple[str, ...]) -> dict[str, str] | None:
    """Return a user YAML alias map if one covers enough canonical joints."""
    maps = _load_yaml_alias_maps()
    if not maps:
        return None

    name_set = set(joint_names)
    best_map: dict[str, str] | None = None
    best_coverage = 0

    for alias_map in maps.values():
        covered = sum(1 for src in alias_map if src in name_set)
        if covered > best_coverage:
            best_coverage = covered
            best_map = alias_map

    if best_map is not None and best_coverage >= 6:
        return {n: best_map.get(n, n) for n in joint_names}
    return None


# ---------------------------------------------------------------------------
# Heuristic detectors
# ---------------------------------------------------------------------------

def is_smpl_like(joint_names: Iterable[str]) -> bool:
    """Heuristic: does ``joint_names`` look like an SMPL / SMPL-H / SMPL-X rig?

    Includes OMOMO demo windows (``pelvis`` + ``l_shoulder`` / ``r_shoulder`` +
    ``spine*``) which omit the long ``left_*`` spellings used by AMASS.

    Also matches reduced skeletons (e.g. parc_ms with ``pelvis`` + ``torso`` +
    ``left_foot``/``right_foot``) that share the ``pelvis``-rooted convention
    but use different names for trunk/shoulder joints.
    """
    names = set(joint_names)
    if "pelvis" not in names:
        return False
    has_spine = any(n in names for n in ("spine1", "spine2", "spine3", "torso"))
    has_lr_shoulder = (
        ("left_shoulder" in names and "right_shoulder" in names)
        or ("l_shoulder" in names and "r_shoulder" in names)
        or ("left_upper_arm" in names and "right_upper_arm" in names)
    )
    has_lr_foot = (
        ("left_foot" in names and "right_foot" in names)
        or ("l_foot" in names and "r_foot" in names)
    )
    return has_spine and (has_lr_shoulder or has_lr_foot)


def is_smpl_pruned_ankle_terminated(joint_names: Iterable[str]) -> bool:
    """Heuristic: SMPL-derived rig whose leg chain ends at the ankle.

    Matches the ``meshmimic/parc_ms`` 15-bone skeleton
    (``pelvis → thigh(=hip) → shin(=knee) → foot(=ankle)``) — the leg chain
    has the same per-segment lengths as a 1.7m adult SMPL human, but
    the bind rest pose's lowest joint sits ~8cm above the ground (the
    sole/heel are not modelled), so the standard ``height_m = max_z −
    min_z`` measurement underestimates the real human stature by a
    constant ankle-to-floor offset.  Callers (the soma-style scaler in
    particular) treat this height as the source human's stature, so the
    underestimate yields ``smpl_scale = robot_height / underestimate``
    that is ~7% closer to 1.0 than it should be — visually the yellow
    overlay barely shrinks toward the robot.

    The detector is intentionally permissive on trunk naming
    (``torso``/``spine*``/``head`` all qualify) so future SMPL-pruned
    variants don't slip through, while requiring **strict absence** of
    any toe / sole / foot-mod joint to avoid false-positives on rigs
    like holosoma's ``LeftFoot`` (with ``LeftToeBase`` child) or
    LAFAN's ``LeftFoot`` (with ``LeftToeBase`` child).
    """
    names_lower = {str(n).lower() for n in joint_names}
    if "pelvis" not in names_lower:
        return False
    if not (
        ("left_foot" in names_lower and "right_foot" in names_lower)
        or ("l_foot" in names_lower and "r_foot" in names_lower)
    ):
        return False
    has_toe = any(
        ("toe" in n) or ("footmod" in n) or n.endswith("_sole")
        for n in names_lower
    )
    return not has_toe


def is_meshmimic_holosoma_like(joint_names: Iterable[str]) -> bool:
    """Heuristic: meshmimic/holosoma ``source.yaml`` MOCAP_DEMO_JOINTS skeleton.

    Mixamo-style names (``Hips``, ``LeftUpLeg``) plus per-foot sole markers used
    by interaction-mesh Laplacian — discriminates from plain CMU/LAFAN exports.
    """
    names = set(joint_names)
    return (
        "LeftFootMod" in names
        and "RightFootMod" in names
        and "Hips" in names
    )


def is_soma_bvh_like(joint_names: Iterable[str]) -> bool:
    """Heuristic: TitleCase ``Hips`` + ``LeftLeg``/``LeftShin`` (SOMA convention)."""
    names = set(joint_names)
    return (
        "Hips" in names
        and "LeftArm" in names
        and "RightArm" in names
        and "LeftLeg" in names
        and "RightLeg" in names
        and "LeftUpLeg" not in names  # discriminator against Mixamo/CMU
    )


def is_xsens_mocap_like(joint_names: Iterable[str]) -> bool:
    """Heuristic: Xsens MVN ``Hips`` + ``LeftHip``/``LeftKnee``/``LeftAnkle``."""
    names = set(joint_names)
    return (
        "Hips" in names
        and "LeftHip" in names
        and "RightHip" in names
        and "LeftKnee" in names
        and "LeftAnkle" in names
        and "LeftUpLeg" not in names
        and "LeftLeg" not in names
        and ("Chest" in names or "Chest2" in names)
    )


def is_mixamo_cmu_like(joint_names: Iterable[str]) -> bool:
    """Heuristic: ``Hips`` + ``LeftUpLeg`` (Mixamo / CMU / LAFAN convention).

    Also matches rigs with a ``mixamorig:`` prefix after stripping.
    """
    names = set(joint_names)
    if "LeftUpLeg" in names and "Hips" in names:
        return True
    normalised = {_normalise_joint_name(n) for n in names}
    return "hips" in normalised and "leftupleg" in normalised


# ---------------------------------------------------------------------------
# Fuzzy (prefix-stripped) matching
# ---------------------------------------------------------------------------

def _fuzzy_source_to_canonical(
    joint_names: tuple[str, ...],
) -> dict[str, str] | None:
    """Try to match joint names by stripping DCC prefixes and normalising.

    Returns ``None`` if fewer than 6 canonical joints are resolved (too
    unreliable to be useful).
    """
    result: dict[str, str] = {}
    canonical_hits: set[str] = set()

    for name in joint_names:
        norm = _normalise_joint_name(name)
        can = _NORMALISED_FRAGMENTS.get(norm)
        if can is not None:
            result[name] = can
            canonical_hits.add(can)
        else:
            result[name] = name

    if len(canonical_hits) >= 6:
        return result
    return None


# ---------------------------------------------------------------------------
# Public API — the unified dispatcher
# ---------------------------------------------------------------------------

def auto_source_to_canonical(
    joint_names: Iterable[str],
    *,
    override_map: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a best-effort source → canonical rename map.

    Resolution order:

    1. Explicit ``override_map`` (caller-supplied or from YAML config).
    2. SMPL-family (``pelvis`` root, including OMOMO abbreviations).
    3. SOMA BVH (``Hips`` + ``LeftLeg``/``LeftShin``).
    4. Xsens mocap BVH (``Hips`` + ``LeftHip``/``LeftKnee``/``LeftAnkle``).
    5. meshmimic holosoma (``LeftFootMod`` + ``RightFootMod`` + ``Hips``).
    6. Mixamo / CMU / LAFAN (``Hips`` + ``LeftUpLeg``).
    7. User YAML alias maps (``configs/skeleton_presets/alias_maps/*.yaml``).
    8. Fuzzy prefix-stripped matching (``b_l_arm`` → ``left_shoulder``).
    9. Identity (unknown rig — surfaces ``KeyError`` downstream).
    """
    names = tuple(joint_names)

    if override_map is not None:
        return {n: override_map.get(n, n) for n in names}

    if is_smpl_like(names):
        out: dict[str, str] = {}
        for n in names:
            if n in SMPL_BODY_TO_CANONICAL:
                out[n] = SMPL_BODY_TO_CANONICAL[n]
            elif n in SMPL_H_ABBR_TO_CANONICAL:
                out[n] = SMPL_H_ABBR_TO_CANONICAL[n]
            else:
                norm = _normalise_joint_name(n)
                frag = _NORMALISED_FRAGMENTS.get(norm)
                out[n] = frag if frag is not None else n
        # Reduced pelvis-rooted rigs such as parc_ms expose ``left_foot`` /
        # ``right_foot`` as the distal leg joint, with no separate ankle or toe.
        # In that topology the foot joint must drive the canonical ankle slot;
        # otherwise humanoid presets that map ``left_ankle`` / ``right_ankle``
        # lose their foot anchors entirely.
        if "left_ankle" not in names and "left_foot" in names:
            out["left_foot"] = "left_ankle"
        if "right_ankle" not in names and "right_foot" in names:
            out["right_foot"] = "right_ankle"
        return out

    if is_soma_bvh_like(names):
        return {n: SOMA_BVH_TO_CANONICAL.get(n, n) for n in names}

    if is_xsens_mocap_like(names):
        return {n: XSENS_MOCAP_TO_CANONICAL.get(n, n) for n in names}

    if is_meshmimic_holosoma_like(names):
        result = {n: MIXAMO_CMU_TO_CANONICAL.get(n, n) for n in names}
        # Holosoma has only 2 spine segments (Spine, Spine1) vs Mixamo's 3
        # (Spine, Spine1, Spine2).  Spine1 parents the shoulders, so it is
        # anatomically the chest — promote it when Spine2 is absent.
        if "Spine2" not in names and "Spine1" in names:
            result["Spine1"] = "chest"
        return result

    if is_mixamo_cmu_like(names):
        name_set = set(names)
        has_prefix = any(_PREFIX_RE.match(n) for n in names)
        if has_prefix:
            mapping: dict[str, str] = {}
            for n in names:
                stripped = _PREFIX_RE.sub("", n)
                can = MIXAMO_CMU_TO_CANONICAL.get(stripped)
                mapping[n] = can if can is not None else n
            return mapping
        return {n: MIXAMO_CMU_TO_CANONICAL.get(n, n) for n in names}

    yaml_map = _try_yaml_override(names)
    if yaml_map is not None:
        return yaml_map

    fuzzy = _fuzzy_source_to_canonical(names)
    if fuzzy is not None:
        return fuzzy

    return {n: n for n in names}


def pack_scaler_rows_to_canonical_targets(
    scaler_joint_names: Iterable[str],
    targets: NDArray[np.floating[Any]],
    *,
    rename: Mapping[str, str] | None = None,
) -> dict[str, NDArray[np.floating[Any]]]:
    """Collapse scaler rows (one per *source* bone) onto canonical effector keys.

    :class:`~hhtools.retarget.newton_basic.scaler.HumanToRobotScaler` emits
    one row per ``ScalerConfig.joint_scales`` key in **hierarchy / dict
    insertion order**.  Several source bones may rename to the same
    canonical (e.g. ``spine1`` and ``spine2`` → ``spine``).  Rules:

    1. If a source name **equals** its canonical target string, that row
       wins for that canonical (matches ``build_scaler_config_soma_style``).
    2. Otherwise first-seen wins, except
    3. ``left_foot`` / ``right_foot`` — **last** row wins (toe after toe base).

    Call :func:`effectors_to_canonical_table` from application code so
    rename resolution (auto vs override) stays in one place.

    Parameters
    ----------
    scaler_joint_names
        Same order as ``targets``' middle axis (``HumanToRobotScaler.joint_names``).
    targets
        ``(F, M, 7)`` float array ``(x,y,z, qx,qy,qz,qw)`` per mapped joint.
    rename
        Full ``source_name → canonical`` map for ``names`` (every bone listed).
    """

    names = list(scaler_joint_names)
    tgt = np.asarray(targets, dtype=np.float32)
    if rename is None:
        raise TypeError(
            "pack_scaler_rows_to_canonical_targets(..., rename=...) requires rename; "
            "call effectors_to_canonical_table() instead."
        )
    rename_map = rename
    out: dict[str, NDArray[np.floating[Any]]] = {}
    for i, name in enumerate(names):
        canon = rename_map.get(name, name)
        if name == canon:
            out[canon] = tgt[:, i, :]
    for i, name in enumerate(names):
        canon = rename_map.get(name, name)
        if canon in ("left_foot", "right_foot"):
            out[canon] = tgt[:, i, :]
        elif canon not in out:
            out[canon] = tgt[:, i, :]
    return out


def effectors_to_canonical_table(
    joint_names: Iterable[str],
    targets: NDArray[np.floating[Any]],
    *,
    source_to_canonical: Mapping[str, str] | None = None,
) -> dict[str, NDArray[np.floating[Any]]]:
    """Resolve rename + pack — **the** entry point for IK / preview pipelines.

    Parameters
    ----------
    joint_names
        Source bone names in the same order as ``targets``' middle axis.
    targets
        ``(F, M, 7)`` scaler output ``(pos, quat_xyzw)``.
    source_to_canonical
        Optional pipeline override (same contract as
        ``NewtonBasicPipeline(..., source_to_canonical=…)``).  When ``None``,
        :func:`auto_source_to_canonical` fills the map.
    """

    names = list(joint_names)
    if source_to_canonical is not None:
        ovr = dict(source_to_canonical)
        rename = {n: ovr.get(n, n) for n in names}
    else:
        rename = auto_source_to_canonical(tuple(names))
    return pack_scaler_rows_to_canonical_targets(names, targets, rename=rename)


def list_detected_rig_type(joint_names: Iterable[str]) -> str:
    """Return a human-readable label for the detected rig type."""
    names = tuple(joint_names)
    if is_smpl_like(names):
        return "SMPL/SMPL-H/SMPL-X"
    if is_soma_bvh_like(names):
        return "SOMA BVH"
    if is_xsens_mocap_like(names):
        return "Xsens mocap BVH"
    if is_meshmimic_holosoma_like(names):
        return "Holosoma / SMPL-H mocap"
    if is_mixamo_cmu_like(names):
        return "Mixamo/CMU/LAFAN"
    yaml_map = _try_yaml_override(names)
    if yaml_map is not None:
        return "User YAML alias"
    fuzzy = _fuzzy_source_to_canonical(names)
    if fuzzy is not None:
        return "Fuzzy prefix match"
    return "Unknown (identity)"


def invalidate_yaml_cache() -> None:
    """Clear the YAML alias map cache (useful after config changes)."""
    global _yaml_alias_cache  # noqa: PLW0603
    _yaml_alias_cache = None


__all__ = [
    "MIXAMO_CMU_TO_CANONICAL",
    "SMPL_BODY_TO_CANONICAL",
    "SMPL_H_ABBR_TO_CANONICAL",
    "SOMA_BVH_TO_CANONICAL",
    "XSENS_MOCAP_TO_CANONICAL",
    "auto_source_to_canonical",
    "effectors_to_canonical_table",
    "pack_scaler_rows_to_canonical_targets",
    "invalidate_yaml_cache",
    "is_meshmimic_holosoma_like",
    "is_mixamo_cmu_like",
    "is_smpl_like",
    "is_soma_bvh_like",
    "is_xsens_mocap_like",
    "list_detected_rig_type",
]
