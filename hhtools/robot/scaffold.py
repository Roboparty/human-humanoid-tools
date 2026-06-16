"""Auto-generate ``robot.yaml`` from a URDF + mesh drop.

The contract is simple: **a user adds a new robot by placing a URDF (and its
meshes) inside a directory.**  Everything else — the ``robot.yaml`` with
``dof_order`` / ``ik_map`` / ``weights`` / ``feet`` — is derived from the URDF
by this module.

Two entry points:

* :func:`scaffold_yaml_file` — pure filesystem op: given a URDF path, compute a
  plausible :class:`RobotPreset` and serialise it to ``robot.yaml`` next to the
  URDF.  Used by the registry when it finds a URDF-only directory, and by
  ``hhtools robot scaffold`` for explicit re-generation.
* :func:`scaffold_preset` — the in-memory flavour returning just a
  :class:`RobotPreset`, useful for previews/tests.

Naming policy
-------------

Presets are named after the directory they live in.  For directories that hold
multiple URDFs (e.g. ``unitree_g1/`` shipping both ``g1_29dof.urdf`` and
``g1_29dof_with_hand.urdf``), each URDF becomes its own preset with the
convention ``<dirname>__<urdf_stem>``.  The scaffolded yaml is written to
``robot.<urdf_stem>.yaml`` so all variants coexist in the same directory.  When
exactly one URDF lives in the directory the yaml is plain ``robot.yaml`` and
the preset name is the directory name — matching the long-standing
single-URDF convention (rp1, etc.).

IK map heuristic
----------------

Link-name matching is deliberately aggressive (Q3=B): we lowercase every link
name, tag it with a side (left/right/none), and score every canonical slot
against every candidate using a mix of keyword hits and token-level edit
distance.  The best-scoring candidate wins — ties broken by preferring
anatomical leaves (``_link`` end-effectors over mid-chain joints).  The result
is a best-effort starting point; the comment block in the generated yaml and
the Mapping Editor milestone both emphasise that users should double-check
this table before running a retarget.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from hhtools.robot.base import RobotPreset

_log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- canonical slot config


#: Canonical human joint names we try to fill.  Kept in sync with
#: ``configs/skeleton_presets/canonical_human.yaml``.  The order here determines
#: the order we emit in the yaml for stable diffs.
_CANONICAL_SLOTS: tuple[str, ...] = (
    "hips", "spine", "chest", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
)


#: Priority keywords for each canonical slot.  Earlier hits score higher; we
#: intentionally list multiple synonyms because URDFs in the wild use all of
#: ``pelvis``, ``base_link``, ``root``, ``hips`` for the floating base link.
#:
#: Negative keywords (prefixed ``!``) *exclude* a link when present.  This
#: prevents e.g. ``left_wrist_pitch_link`` from winning the ``left_elbow`` slot
#: just because ``wrist`` sounds close to ``elbow`` on the edit-distance axis.
_SLOT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "hips":            ("pelvis", "hips", "base_link", "root_link", "waist_root"),
    "spine":           ("spine", "waist", "torso_1", "lower_torso", "!arm", "!shoulder", "!elbow"),
    "chest":           ("chest", "torso", "upper_torso", "trunk", "thorax", "!arm", "!shoulder"),
    "neck":            ("neck",),
    "head":            ("head", "skull", "!effector", "!end_effector", "!hand", "!wrist"),
    "left_shoulder":   ("shoulder_pitch", "shoulder", "shldr", "clavicle"),
    "left_elbow":      ("elbow", "!wrist"),
    "left_wrist":      ("wrist_yaw", "wrist", "hand_palm", "palm", "hand", "!finger"),
    "right_shoulder":  ("shoulder_pitch", "shoulder", "shldr", "clavicle"),
    "right_elbow":     ("elbow", "!wrist"),
    "right_wrist":     ("wrist_yaw", "wrist", "hand_palm", "palm", "hand", "!finger"),
    "left_hip":        ("hip_pitch", "hip", "thigh", "upleg"),
    "left_knee":       ("knee", "shin", "lower_leg", "tarsus"),
    "left_ankle":      ("ankle_roll", "ankle", "foot", "toe_roll", "!toe_rod", "!toe_a"),
    "right_hip":       ("hip_pitch", "hip", "thigh", "upleg"),
    "right_knee":      ("knee", "shin", "lower_leg", "tarsus"),
    "right_ankle":     ("ankle_roll", "ankle", "foot", "toe_roll", "!toe_rod", "!toe_a"),
}


#: IK weights aligned with configs/robots/_template/robot.yaml (gimbal-safe).
_DEFAULT_WEIGHTS = {
    "t_weight": {
        "hips": 10.0,
        "chest": 0.5,
        "left_shoulder": 0.5,
        "left_elbow": 1.0,
        "left_wrist": 1.0,
        "right_shoulder": 0.5,
        "right_elbow": 1.0,
        "right_wrist": 1.0,
        "left_hip": 0.5,
        "left_knee": 1.0,
        "left_ankle": 2.0,
        "right_hip": 0.5,
        "right_knee": 1.0,
        "right_ankle": 2.0,
    },
    "r_weight": {
        "hips": 2.0,
        "chest": 0.5,
        "left_shoulder": 0.1,
        "left_elbow": 0.5,
        "left_wrist": 0.5,
        "right_shoulder": 0.1,
        "right_elbow": 0.5,
        "right_wrist": 0.5,
        "left_hip": 0.2,
        "left_knee": 0.5,
        "left_ankle": 1.0,
        "right_hip": 0.2,
        "right_knee": 0.5,
        "right_ankle": 1.0,
    },
}

#: Per-link smoother masks when these links exist on the URDF.
_GIMBAL_SMOOTH_MASK_CANDIDATES: tuple[tuple[str, float], ...] = (
    ("left_thigh_yaw_link", 0.1),
    ("right_thigh_yaw_link", 0.1),
    ("left_hip_yaw_link", 0.1),
    ("right_hip_yaw_link", 0.1),
    ("left_arm_pitch_link", 0.1),
    ("left_arm_roll_link", 1.0),
    ("left_arm_yaw_link", 0.3),
    ("right_arm_pitch_link", 0.1),
    ("right_arm_roll_link", 1.0),
    ("right_arm_yaw_link", 0.3),
    ("left_shoulder_pitch_link", 0.1),
    ("left_shoulder_roll_link", 1.0),
    ("left_shoulder_yaw_link", 0.3),
    ("right_shoulder_pitch_link", 0.1),
    ("right_shoulder_roll_link", 1.0),
    ("right_shoulder_yaw_link", 0.3),
)

_DEFAULT_RETARGET: dict[str, object] = {
    "num_initialization_frames": 10,
    "num_stabilization_frames": 5,
    "apply_feet_stabilizer": True,
    "feet_stabilizer": {
        "ground_contact_z": 0.0,
        "min_foot_clearance": 0.01,
    },
}


# --------------------------------------------------------------------------- helpers


@dataclass(frozen=True)
class _LinkCandidate:
    """Cached side-annotated link name, so we only tokenise once per URDF."""
    name: str
    lower: str
    tokens: frozenset[str]
    side: str  # "left" | "right" | ""
    depth: int  # tree depth, used only as a tiebreaker


def _side_of(lower: str) -> str:
    """Heuristic left/right detection from a link name.

    We look for whole-word ``left``/``right`` tokens (``_`` or word boundary),
    and the short forms ``l_``/``r_`` common in simulation rigs.  Anything
    else is "no side".
    """
    if re.search(r"(^|_)left(_|$)", lower) or lower.startswith("l_") or "_l_" in lower:
        return "left"
    if re.search(r"(^|_)right(_|$)", lower) or lower.startswith("r_") or "_r_" in lower:
        return "right"
    return ""


def _tokenise(lower: str) -> frozenset[str]:
    """Split a link name into tokens used for keyword/edit-distance matching."""
    return frozenset(tok for tok in re.split(r"[^a-z0-9]+", lower) if tok)


def _levenshtein(a: str, b: str) -> int:
    """Classic edit distance — good enough for short token pairs."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Two-row DP
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ca in enumerate(a, start=1):
        curr[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[len(b)]


def _edit_score(keyword: str, link: _LinkCandidate) -> float:
    """Return a 0..1 score for how close ``link`` matches ``keyword``.

    Rules:
    * Compound keywords (containing ``_``) match as a substring of the full
      lowered link name — ``"ankle_roll"`` hits ``"left_ankle_roll_link"``
      but *not* ``"left_ankle_pitch_link"``.  This is important because
      tokenising on ``_`` would otherwise discard the compound signal.
    * Simple keywords match best-of edit-distance across tokens.
    * 1.0 = exact token match / substring hit; 0.0 = no similarity.
    """
    if "_" in keyword:
        if keyword in link.lower:
            return 1.0
        return 0.0
    best = 1e9
    for tok in link.tokens:
        if tok == keyword:
            return 1.0
        d = _levenshtein(tok, keyword)
        if d < best:
            best = d
    if not keyword or best >= len(keyword):
        return 0.0
    return 1.0 - (best / len(keyword))


def _score_candidate(
    slot: str,
    keywords: tuple[str, ...],
    link: _LinkCandidate,
) -> float:
    """Score how likely ``link`` fills ``slot``.

    Negative keywords (prefixed ``!``) *disqualify* the candidate outright.
    Positive keywords add weight inversely proportional to their position
    (earliest keyword wins ties).  We also add a small bonus for exact
    prefix matches (``pelvis`` beats ``pelvis_contour_link``) and for links
    named ``*_link`` which is the URDF convention for real bodies.
    """
    side_wanted = (
        "left" if slot.startswith("left_")
        else "right" if slot.startswith("right_")
        else ""
    )
    if side_wanted and link.side and link.side != side_wanted:
        return 0.0
    if side_wanted and not link.side:
        # Mildly penalise unlocalised links so L/R keyword matches win —
        # but don't disqualify (some URDFs don't side-tag every segment).
        side_bonus = 0.4
    elif side_wanted and link.side == side_wanted:
        side_bonus = 1.0
    else:
        side_bonus = 1.0 if not link.side else 0.8

    best_pos = 0.0
    for idx, kw in enumerate(keywords):
        if kw.startswith("!"):
            negated = kw[1:]
            # Negative keywords are whole-word matches on the lowered string
            # to avoid "ankle" being killed by its substring presence in
            # "head_ankle_holder" (contrived, but the principle stands).
            if any(negated in tok for tok in link.tokens):
                return 0.0
            continue
        pos_weight = 1.0 - idx * 0.08  # 1.00, 0.92, 0.84, ...
        score = _edit_score(kw, link) * max(0.3, pos_weight)
        if score > best_pos:
            best_pos = score

    if best_pos == 0.0:
        return 0.0

    from hhtools.robot.kinematics import _is_arm_segment, _is_leg_segment

    arm_slots = {
        "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist",
    }
    leg_slots = {
        "left_hip", "left_knee", "left_ankle",
        "right_hip", "right_knee", "right_ankle",
    }
    if slot in arm_slots and _is_leg_segment(link.name):
        return 0.0
    if slot in leg_slots and _is_arm_segment(link.name):
        return 0.0

    # Prefer "_link" terminators (URDF body convention) and discourage
    # obvious helper bodies (contour/sensor/logo/imu/d435/mid360/camera).
    bonus = 0.0
    if link.lower.endswith("_link"):
        bonus += 0.05
    noise_tokens = {
        "contour", "logo", "imu", "d435", "mid360", "camera", "sensor",
        "finger", "thumb", "index", "middle", "ring", "pinky", "support",
        "rubber", "lidar", "rgb", "stereo",
    }
    if noise_tokens & link.tokens:
        bonus -= 0.2
    # Depth preference: shoulder should be shallow, wrist should be deep.
    # We express this very coarsely.
    deep_slots = {"left_wrist", "right_wrist", "left_ankle", "right_ankle", "head"}
    shallow_slots = {"hips", "spine", "chest"}
    if slot in deep_slots and link.depth >= 4:
        bonus += 0.05
    if slot in shallow_slots and link.depth <= 3:
        bonus += 0.05

    return best_pos * side_bonus + bonus


# --------------------------------------------------------------------------- public API


# Joint types that map to MuJoCo HINGE/SLIDE (scalar qpos each).
_SCALAR_JOINT_TYPES = frozenset({"revolute", "continuous", "prismatic"})


def _actuated_order_from_urdf(urdf_path: Path) -> tuple[str, ...]:
    """Walk the URDF tree from the base link and return actuated joints in DFS order.

    We don't use :mod:`yourdfpy` here to keep this module import-light — the
    scaffold runs during registry discovery and we don't want to pull in
    yourdfpy/mujoco unless the user actually loads a robot.  Plain xml.etree is
    enough because URDF is pure XML.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = []
    for joint in root.findall("joint"):
        jtype = joint.get("type", "fixed")
        if jtype == "fixed":
            continue
        if jtype not in _SCALAR_JOINT_TYPES:
            continue
        name = joint.get("name")
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if name is None or parent_el is None or child_el is None:
            continue
        joints.append(
            (
                name,
                parent_el.get("link"),
                child_el.get("link"),
                jtype,
            )
        )

    # Identify the base link: any <link> that's not a child of some joint.
    all_links = {link.get("name") for link in root.findall("link") if link.get("name")}
    child_links = {c for _, _, c, _ in joints}
    # Also include parents from fixed joints so we don't call them base.
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None and child.get("link"):
            child_links.add(child.get("link"))
    roots = [n for n in all_links if n not in child_links]
    # URDF can technically have a "world" link that floats via floating_base_joint;
    # skip that and use its child as the effective base.
    joint_children = {p: [] for p in all_links}
    for name, parent, child, _t in [
        (j.get("name"), (j.find("parent").get("link") if j.find("parent") is not None else None),
         (j.find("child").get("link") if j.find("child") is not None else None),
         j.get("type", "fixed"))
        for j in root.findall("joint")
    ]:
        if parent and child:
            joint_children.setdefault(parent, []).append((name, child))

    if not roots:
        # Probably commented-out "world" link — fall back to URDF parse order.
        return tuple(n for n, _p, _c, _t in joints)
    base = roots[0]
    for candidate in roots:
        if candidate != "world":
            base = candidate
            break

    # DFS from base collecting actuated joint names in order.
    ordered: list[str] = []
    seen: set[str] = set()

    def walk(link: str) -> None:
        for jname, child in joint_children.get(link, []):
            if jname in seen:
                continue
            seen.add(jname)
            for j_name, _p, _c, j_type in joints:
                if j_name == jname and j_type in _SCALAR_JOINT_TYPES:
                    ordered.append(jname)
                    break
            walk(child)

    walk(base)
    # Append any scalar joints not reachable from base (disconnected chains).
    for name, _p, _c, j_type in joints:
        if name not in ordered and j_type in _SCALAR_JOINT_TYPES:
            ordered.append(name)
    return tuple(ordered)


def _link_candidates(urdf_path: Path) -> list[_LinkCandidate]:
    """Build side-annotated :class:`_LinkCandidate` records for every link."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    all_links = [link.get("name") for link in root.findall("link") if link.get("name")]

    # Build parent map to compute depth.
    parent_of: dict[str, str | None] = {n: None for n in all_links}
    for joint in root.findall("joint"):
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is not None and child_el is not None:
            parent_of[child_el.get("link")] = parent_el.get("link")

    def depth(link: str) -> int:
        d = 0
        cur = link
        seen: set[str] = set()
        while parent_of.get(cur) and cur not in seen:
            seen.add(cur)
            cur = parent_of[cur]  # type: ignore[assignment]
            d += 1
        return d

    out: list[_LinkCandidate] = []
    for name in all_links:
        lower = name.lower()
        out.append(
            _LinkCandidate(
                name=name,
                lower=lower,
                tokens=_tokenise(lower),
                side=_side_of(lower),
                depth=depth(name),
            )
        )
    return out


def _auto_ik_map(urdf_path: Path) -> dict[str, str]:
    """Score URDF links against canonical slots, then refine with kinematics.

    Keyword matching alone mis-maps robots whose distal link is named
    ``left_elbow_yaw_link`` instead of ``left_wrist_link`` (e.g. RPO).
    Topology inference wins for limb endpoints and trunk; keywords fill
    any remaining canonical slots.  The result is repaired and stripped of
    invalid optional slots before return.
    """
    from hhtools.robot.kinematics import infer_ik_map_from_kinematics, infer_smooth_joint_filter_masks, prepare_ik_map

    keyword = _auto_ik_map_keywords(urdf_path)
    kinematic = infer_ik_map_from_kinematics(urdf_path)

    limb_trunk_slots = {
        "hips", "spine", "chest", "neck", "head",
        "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist",
        "left_hip", "left_knee", "left_ankle",
        "right_hip", "right_knee", "right_ankle",
    }
    merged = dict(keyword)
    for slot in limb_trunk_slots:
        if slot in kinematic:
            merged[slot] = kinematic[slot]
    prepared, _changes = prepare_ik_map(urdf_path, merged)
    return prepared


def _auto_ik_map_keywords(urdf_path: Path) -> dict[str, str]:
    """Greedy keyword/edit-distance ik_map (legacy heuristic)."""
    cands = _link_candidates(urdf_path)
    chosen: dict[str, str] = {}
    used: set[str] = set()
    # Score all (slot, candidate) pairs, then greedily assign highest-first
    # so a strong match on ``left_wrist`` isn't stolen by a marginal hit for
    # ``left_elbow``.
    scores: list[tuple[float, str, _LinkCandidate]] = []
    for slot in _CANONICAL_SLOTS:
        kws = _SLOT_KEYWORDS[slot]
        for cand in cands:
            s = _score_candidate(slot, kws, cand)
            if s > 0:
                scores.append((s, slot, cand))
    scores.sort(key=lambda x: (-x[0], x[1], x[2].name))
    MIN_SCORE = 0.4  # below this we prefer leaving the slot blank
    for score, slot, cand in scores:
        if slot in chosen:
            continue
        if cand.name in used:
            continue
        if score < MIN_SCORE:
            continue
        chosen[slot] = cand.name
        used.add(cand.name)
    return chosen


def _auto_feet(ik_map: dict[str, str]) -> dict[str, object]:
    """Populate ``feet`` from the ankle picks in ``ik_map`` when available."""
    feet: dict[str, object] = {}
    if "left_ankle" in ik_map:
        feet["left_contact_link"] = ik_map["left_ankle"]
    if "right_ankle" in ik_map:
        feet["right_contact_link"] = ik_map["right_ankle"]
    if feet:
        feet["ground_z"] = 0.0
        feet["foot_height"] = 0.02
    return feet


def _pretty_display_name(preset_name: str) -> str:
    """``unitree_g1__g1_29dof_with_hand`` → ``Unitree G1 (g1 29dof with hand)``."""
    if "__" in preset_name:
        prefix, variant = preset_name.split("__", 1)
    else:
        prefix, variant = preset_name, ""
    humanised = " ".join(tok.capitalize() for tok in prefix.split("_"))
    if variant:
        return f"{humanised} ({variant.replace('_', ' ')})"
    return humanised


def _mesh_search_paths_for(urdf_path: Path) -> list[Path]:
    """Auto-detect useful mesh search directories next to a URDF.

    We prefer the conventional ``meshes/`` sibling (flat layout) or
    ``../meshes`` (RP1-style nested ``urdf/`` layout).  If neither exists we
    fall back to the URDF's own directory — enough for URDFs that ship STLs
    directly beside them.  Including the URDF parent *alongside* an existing
    ``meshes/`` is noise in the yaml, so we skip it in that case.
    """
    sibling = urdf_path.parent / "meshes"
    nested = urdf_path.parent.parent / "meshes"
    candidates: list[Path] = []
    if sibling.is_dir():
        candidates.append(sibling)
    if nested.is_dir() and nested.resolve() != sibling.resolve():
        candidates.append(nested)
    if not candidates and urdf_path.parent.is_dir():
        candidates.append(urdf_path.parent)

    out: list[Path] = []
    seen: set[Path] = set()
    for c in candidates:
        try:
            r = c.resolve()
        except OSError:
            continue
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


@dataclass
class ScaffoldResult:
    """Return type of :func:`scaffold_preset`.

    ``yaml_path`` is where the yaml *would* live if written; it's always set
    (the caller decides whether to flush to disk via :func:`scaffold_yaml_file`).
    """
    preset: RobotPreset
    yaml_path: Path
    yaml_body: str
    created: bool = False
    actions: tuple[str, ...] = field(default_factory=tuple)


def _yaml_name_for_urdf(root_dir: Path, urdf_path: Path) -> tuple[str, str]:
    """Return ``(preset_name, yaml_filename)`` following the naming policy.

    One URDF → ``robot.yaml`` + preset name = directory name.
    Multi URDF → ``robot.<stem>.yaml`` + preset name = ``<dir>__<stem>``.
    """
    urdfs = sorted(root_dir.glob("*.urdf"))
    if len(urdfs) <= 1:
        return root_dir.name, "robot.yaml"
    stem = urdf_path.stem
    return f"{root_dir.name}__{stem}", f"robot.{stem}.yaml"


def scaffold_preset(
    root_dir: Path,
    urdf_path: Path,
) -> ScaffoldResult:
    """Compute a :class:`RobotPreset` + yaml body for a URDF drop-in.

    No filesystem side-effects beyond reading ``urdf_path`` — the yaml is
    returned as a string so callers can diff/preview before writing.
    """
    root_dir = root_dir.resolve()
    urdf_path = urdf_path.resolve()

    preset_name, yaml_filename = _yaml_name_for_urdf(root_dir, urdf_path)
    yaml_path = root_dir / yaml_filename
    display_name = _pretty_display_name(preset_name)

    dof_order = _actuated_order_from_urdf(urdf_path)
    ik_map = _auto_ik_map(urdf_path)
    feet = _auto_feet(ik_map)
    mesh_paths = _mesh_search_paths_for(urdf_path)
    mesh_paths_yaml = [
        p.relative_to(root_dir).as_posix() if p.is_relative_to(root_dir) else str(p)
        for p in mesh_paths
    ]

    urdf_rel = (
        urdf_path.relative_to(root_dir).as_posix()
        if urdf_path.is_relative_to(root_dir)
        else str(urdf_path)
    )

    # Ensure ik_map entries are emitted in canonical order for stable diffs.
    ordered_ik_map = {slot: ik_map[slot] for slot in _CANONICAL_SLOTS if slot in ik_map}

    # Only retain weights whose canonical slot actually got mapped — otherwise
    # the user opens the file and wonders why we reference joints that aren't
    # in ik_map.  This keeps the generated yaml self-consistent.
    weights = {
        "t_weight": {
            k: v for k, v in _DEFAULT_WEIGHTS["t_weight"].items() if k in ordered_ik_map
        },
        "r_weight": {
            k: v for k, v in _DEFAULT_WEIGHTS["r_weight"].items() if k in ordered_ik_map
        },
    }
    # Drop empty sub-dicts so the yaml stays clean.
    weights = {k: v for k, v in weights.items() if v}

    smooth_masks = infer_smooth_joint_filter_masks(urdf_path, ordered_ik_map)
    if not smooth_masks:
        import xml.etree.ElementTree as ET

        urdf_links = {
            el.get("name")
            for el in ET.parse(urdf_path).getroot().findall("link")
            if el.get("name")
        }
        smooth_masks = {
            link: weight
            for link, weight in _GIMBAL_SMOOTH_MASK_CANDIDATES
            if link in urdf_links
        }

    payload: dict[str, object] = {
        "name": preset_name,
        "display_name": display_name,
        "urdf": urdf_rel,
    }
    if mesh_paths_yaml:
        payload["mesh_search_paths"] = mesh_paths_yaml
    payload["length_scale"] = 1.0
    payload["up_axis"] = "Z"
    payload["forward_axis"] = "X"
    payload["dof_order"] = list(dof_order)
    if ordered_ik_map:
        payload["ik_map"] = ordered_ik_map
    if weights:
        payload["weights"] = weights
    if smooth_masks:
        payload["smooth_joint_filter_masks"] = smooth_masks
    if feet:
        payload["feet"] = feet
    payload["rest_offsets"] = {}

    retarget_cfg: dict[str, object] = dict(_DEFAULT_RETARGET)
    try:
        from hhtools.robot.base import RobotPreset
        from hhtools.robot.foot_geometry import estimate_min_lateral_foot_separation
        from hhtools.robot.loader import load_robot

        _pre = RobotPreset(
            name=preset_name,
            display_name=display_name,
            root_dir=root_dir,
            urdf_path=urdf_path,
            mesh_search_paths=tuple(mesh_paths),
            ik_map=dict(ordered_ik_map),
            weights={k: dict(v) for k, v in weights.items()} if weights else {},
            smooth_joint_filter_masks=dict(smooth_masks),
            rest_offsets={},
            feet=dict(feet),
            length_scale=1.0,
            up_axis="Z",
            forward_axis="X",
            dof_order=tuple(dof_order),
        )
        _lat = estimate_min_lateral_foot_separation(
            load_robot(_pre, compile_mjcf=False),
        )
        if _lat is not None:
            _fs = dict(retarget_cfg["feet_stabilizer"])  # type: ignore[arg-type]
            _fs["min_lateral_separation"] = round(float(_lat), 4)
            retarget_cfg["feet_stabilizer"] = _fs
    except Exception:
        pass
    payload["retarget"] = retarget_cfg

    header = _yaml_header(preset_name, urdf_rel)
    body = header + yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )

    preset = RobotPreset(
        name=preset_name,
        display_name=display_name,
        root_dir=root_dir,
        urdf_path=urdf_path,
        mesh_search_paths=tuple(mesh_paths),
        ik_map=dict(ordered_ik_map),
        weights={k: dict(v) for k, v in weights.items()},
        smooth_joint_filter_masks=dict(smooth_masks),
        rest_offsets={},
        feet=dict(feet),
        length_scale=1.0,
        up_axis="Z",
        forward_axis="X",
        dof_order=tuple(dof_order),
        meta={"auto_generated": True, "retarget": dict(retarget_cfg)},
    )
    return ScaffoldResult(preset=preset, yaml_path=yaml_path, yaml_body=body)


def scaffold_yaml_file(
    urdf_path: Path,
    *,
    overwrite: bool = False,
    root_dir: Path | None = None,
) -> ScaffoldResult:
    """Materialise a yaml next to ``urdf_path``, returning what happened.

    * ``overwrite=False`` (default): respect an existing yaml on disk.  The
      scaffold result still contains the *would-be* preset + yaml body so
      callers can diff; ``created=False`` signals "no write happened".
    * ``overwrite=True``: force-rewrite.  The previous contents are lost —
      use the CLI (``hhtools robot scaffold --force``) for intentional regen.
    * ``root_dir``: preset library folder that should own ``robot.yaml``.  When
      the URDF lives in a nested subfolder (e.g. ``<drop>/urdf/bot.urdf`` from
      a vendor zip) callers must pass ``root_dir=<drop>`` so the registry can
      discover the preset.
    """
    urdf_path = urdf_path.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"no URDF at {urdf_path}")
    enclosing = (root_dir or urdf_path.parent).resolve()
    result = scaffold_preset(enclosing, urdf_path)
    actions: list[str] = []
    if result.yaml_path.exists() and not overwrite:
        result.actions = ("skipped: yaml already exists",)
        return result
    result.yaml_path.write_text(result.yaml_body, encoding="utf-8")
    actions.append(f"wrote {result.yaml_path.name}")
    result.created = True
    result.actions = tuple(actions)
    return result


def _yaml_header(name: str, urdf_rel: str) -> str:
    return (
        f"# {name} — auto-generated by hhtools.robot.scaffold.\n"
        f"#\n"
        f"# Source URDF: {urdf_rel}\n"
        f"# You may edit this file freely; hhtools will NOT overwrite it.\n"
        f"# Run `hhtools robot scaffold {name} --force` to regenerate from the URDF.\n"
        f"#\n"
        f"# ik_map merges keyword matching with URDF topology inference\n"
        f"# (distal wrist/ankle/end-effector links).  Double-check mappings\n"
        f"# (especially shoulder/elbow and ankle/foot) before running\n"
        f"# retarget, or run `hhtools robot validate {name}`.  Canonical\n"
        f"# slot names come from configs/skeleton_presets/canonical_human.yaml.\n\n"
    )


__all__ = [
    "ScaffoldResult",
    "scaffold_preset",
    "scaffold_yaml_file",
]
