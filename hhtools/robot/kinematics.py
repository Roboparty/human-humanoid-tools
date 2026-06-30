"""URDF kinematic tree helpers for robot ik_map inference and validation.

The keyword heuristic in :mod:`hhtools.robot.scaffold` is fast but brittle on
URDFs that name links ``left_arm_yaw_link`` instead of ``left_wrist_link``.
This module walks the parent/child graph and picks **distal limb endpoints**
(elbow / wrist / knee / ankle) plus trunk links from topology, then validates
that a preset's ``ik_map`` is anatomically consistent before retarget runs.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CRITICAL_IK_SLOTS",
    "IkMapIssue",
    "KinematicModel",
    "infer_ik_map_from_kinematics",
    "infer_smooth_joint_filter_masks",
    "mujoco_body_names",
    "prepare_ik_map",
    "repair_ik_map",
    "require_valid_ik_map",
    "resolve_urdf_link_to_mujoco_body",
    "resolve_urdf_links_to_mujoco_bodies",
    "validate_ik_map",
]

# Slots that must be correct (or absent) before retarget; optional anatomy may
# be missing on headless / mini humanoids.
CRITICAL_IK_SLOTS: frozenset[str] = frozenset(
    {
        "hips",
        "left_hip", "left_knee", "left_ankle",
        "right_hip", "right_knee", "right_ankle",
        "left_wrist", "right_wrist",
    }
)


@dataclass(frozen=True)
class IkMapIssue:
    """One anatomical inconsistency in a preset ``ik_map``."""

    slot: str
    message: str

    def format(self) -> str:
        return f"ik_map[{self.slot!r}]: {self.message}"


# Tokens that indicate a link belongs to an arm/leg chain (not trunk).
_LIMB_TOKENS: frozenset[str] = frozenset(
    {
        "arm", "shoulder", "elbow", "wrist", "hand", "forearm", "shldr",
        "upper", "lower", "effector", "ee",
        "thigh", "hip", "knee", "ankle", "foot", "shin", "shank", "calf",
        "toe", "leg", "upleg", "lowleg",
    }
)
_TRUNK_TOKENS: frozenset[str] = frozenset(
    {
        "torso", "spine", "waist", "trunk", "pelvis", "base", "chest",
        "thorax", "root",
    }
)
_ARM_TOKENS: frozenset[str] = frozenset(
    {
        "arm", "shoulder", "elbow", "wrist", "hand", "forearm", "shldr",
        "upper", "lower", "effector", "ee",
    },
)
_LEG_TOKENS: frozenset[str] = frozenset(
    {
        "thigh", "hip", "knee", "ankle", "foot", "shin", "shank", "calf",
        "toe", "leg", "tarsus",
    },
)


def _tokenise(lower: str) -> frozenset[str]:
    return frozenset(tok for tok in re.split(r"[^a-z0-9]+", lower) if tok)


def _link_matches_tokens(lower: str, tokens: frozenset[str]) -> bool:
    """Match link names to anatomy tokens without substring false positives.

    Token intersection handles ``knee_pitch_l_link`` correctly; a naive
    ``"ee" in lower`` would match inside ``knee``.
    """
    if _tokenise(lower) & tokens:
        return True
    # Compound names without separators (``upperarm``, ``lowerleg``).
    for token in tokens:
        if len(token) < 4:
            continue
        if token in lower:
            return True
    return False


def _side_of(lower: str) -> str:
    if re.search(r"(^|_)left(_|$)", lower) or lower.startswith("l_") or "_l_" in lower:
        return "left"
    if re.search(r"(^|_)right(_|$)", lower) or lower.startswith("r_") or "_r_" in lower:
        return "right"
    # PND / vendor camelCase: ``toeLeft``, ``shinRight``, ``shoulderPitchLeft``.
    if re.search(r"(?<=[a-z])(?:left|right)$", lower):
        return "right" if lower.endswith("right") else "left"
    # Booster T1: AL* left arm, AR* right arm.
    if re.match(r"^al\d", lower):
        return "left"
    if re.match(r"^ar\d", lower):
        return "right"
    return ""


def is_limb_link(link_name: str) -> bool:
    """True when the link name looks like arm/leg, not trunk/pelvis."""
    tokens = _tokenise(link_name.lower())
    if tokens & _TRUNK_TOKENS:
        return False
    return bool(tokens & _LIMB_TOKENS)


def is_trunk_link(link_name: str) -> bool:
    tokens = _tokenise(link_name.lower())
    return bool(tokens & _TRUNK_TOKENS) and not is_limb_link(link_name)


@dataclass(frozen=True)
class KinematicModel:
    """Minimal URDF kinematic tree (link parent map + depths)."""

    base_link: str
    parent_of: dict[str, str | None]
    all_links: tuple[str, ...]
    joint_for_child: dict[str, str]
    children_of: dict[str, tuple[str, ...]]

    @classmethod
    def from_urdf(cls, urdf_path: Path) -> KinematicModel:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        all_links = tuple(
            link.get("name")
            for link in root.findall("link")
            if link.get("name")
        )
        parent_of: dict[str, str | None] = {name: None for name in all_links}
        joint_for_child: dict[str, str] = {}
        child_lists: dict[str, list[str]] = {name: [] for name in all_links}
        for joint in root.findall("joint"):
            parent_el = joint.find("parent")
            child_el = joint.find("child")
            jname = joint.get("name") or ""
            if parent_el is not None and child_el is not None:
                parent = parent_el.get("link")
                child = child_el.get("link")
                parent_of[child] = parent
                joint_for_child[child] = jname
                if parent in child_lists:
                    child_lists[parent].append(child)

        child_links = {c for c in parent_of if parent_of[c] is not None}
        roots = [n for n in all_links if n not in child_links]
        base = roots[0] if roots else (all_links[0] if all_links else "")
        for candidate in roots:
            if candidate != "world":
                base = candidate
                break
        children_of = {k: tuple(v) for k, v in child_lists.items()}
        return cls(
            base_link=base,
            parent_of=parent_of,
            all_links=all_links,
            joint_for_child=joint_for_child,
            children_of=children_of,
        )

    def depth(self, link: str) -> int:
        d = 0
        cur = link
        seen: set[str] = set()
        while self.parent_of.get(cur) and cur not in seen:
            seen.add(cur)
            cur = self.parent_of[cur]  # type: ignore[assignment]
            d += 1
        return d

    def side(self, link: str) -> str:
        """Side tag from link name and the joint that attaches it to the tree."""
        got = _side_of(link.lower())
        if got:
            return got
        joint = self.joint_for_child.get(link, "")
        if joint:
            return _side_of(joint.lower())
        return ""

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        cur: str | None = descendant
        seen: set[str] = set()
        while cur and cur not in seen:
            if cur == ancestor:
                return True
            seen.add(cur)
            cur = self.parent_of.get(cur)
        return False

    def lca(self, a: str, b: str) -> str | None:
        """Lowest common ancestor of two links, or ``None``."""
        if not a or not b:
            return None
        anc_a: set[str] = set()
        cur: str | None = a
        while cur:
            anc_a.add(cur)
            cur = self.parent_of.get(cur)
        cur = b
        while cur:
            if cur in anc_a:
                return cur
            cur = self.parent_of.get(cur)
        return None

    def _links_with_tokens(self, side: str, tokens: frozenset[str]) -> list[str]:
        out: list[str] = []
        for name in self.all_links:
            lower = name.lower()
            if side and self.side(name) not in ("", side):
                continue
            if _link_matches_tokens(lower, tokens):
                out.append(name)
        return out


def _is_finger_link(link_name: str) -> bool:
    lower = link_name.lower()
    if "finger" in lower or "thumb" in lower:
        return True
    tokens = _tokenise(lower)
    return bool(tokens & {"index", "middle", "ring", "pinky"})


def _is_sensor_link(link_name: str) -> bool:
    lower = link_name.lower()
    return any(
        tok in lower
        for tok in (
            "camera", "imu", "lidar", "rgb", "stereo", "sensor", "logo",
            "d435", "mid360",
        )
    )


def _is_end_effector_link(link_name: str) -> bool:
    lower = link_name.lower()
    return (
        "end_effector" in lower
        or "effector" in lower
        or lower.endswith("_ee")
        or lower.endswith("_ee_link")
    )


def _is_wrist_like(link_name: str) -> bool:
    lower = link_name.lower()
    if _is_finger_link(link_name) or _is_end_effector_link(link_name):
        return True
    tokens = _tokenise(lower)
    return bool(tokens & {"wrist", "hand", "palm"})


def _is_foot_like(link_name: str) -> bool:
    lower = link_name.lower()
    tokens = _tokenise(lower)
    return bool(tokens & {"foot", "toe"}) or lower.endswith("_foot_link")


def _is_ankle_like(link_name: str) -> bool:
    lower = link_name.lower()
    if _is_foot_like(link_name):
        return True
    return "ankle" in lower


def _is_hip_like(link_name: str) -> bool:
    lower = link_name.lower()
    if any(tok in lower for tok in ("knee", "shank", "shin", "ankle", "foot", "calf")):
        return False
    tokens = _tokenise(lower)
    return bool(tokens & {"hip", "thigh", "upleg"})


def _is_knee_like(link_name: str) -> bool:
    lower = link_name.lower()
    tokens = _tokenise(lower)
    return bool(tokens & {"knee", "shank", "shin", "calf", "tarsus"})


def _is_arm_segment(link_name: str) -> bool:
    lower = link_name.lower()
    if _is_leg_segment(link_name):
        return False
    if re.match(r"^a[lr]\d", lower):
        return True
    tokens = _tokenise(lower)
    return bool(tokens & _ARM_TOKENS)


def _is_leg_segment(link_name: str) -> bool:
    lower = link_name.lower()
    tokens = _tokenise(lower)
    if tokens & _LEG_TOKENS:
        return True
    if re.match(r"^shank", lower):
        return True
    return False


def _ancestor_chain(km: KinematicModel, link: str) -> list[str]:
    chain: list[str] = []
    cur: str | None = link
    seen: set[str] = set()
    while cur and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        cur = km.parent_of.get(cur)
    return chain


def _subtree_links(km: KinematicModel, root: str) -> set[str]:
    out: set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in out:
            continue
        out.add(node)
        stack.extend(km.children_of.get(node, ()))
    return out


def _branch_is_limb(km: KinematicModel, root: str, *, leg: bool) -> bool:
    for name in _subtree_links(km, root):
        if leg and _is_leg_segment(name):
            return True
        if not leg and _is_arm_segment(name):
            return True
    return False


def _pick_distal_on_side(
    km: KinematicModel,
    side: str,
    *,
    predicate,
) -> str | None:
    hits = [
        name for name in km.all_links
        if (not side or km.side(name) in ("", side))
        and predicate(name)
    ]
    if not hits:
        return None
    return max(hits, key=lambda name: (km.depth(name), name))


def _links_for_joint_keyword(km: KinematicModel, side: str, keyword: str) -> list[str]:
    hits: list[str] = []
    for link, joint in km.joint_for_child.items():
        if keyword not in joint.lower():
            continue
        link_side = km.side(link) or _side_of(joint.lower())
        if side and link_side and link_side != side:
            continue
        hits.append(link)
    return hits


def _arm_chain_links(km: KinematicModel, side: str) -> list[str]:
    """Arm links on one side, excluding leg/finger segments."""
    arm_links = km._links_with_tokens(side, _ARM_TOKENS)
    return [
        link for link in arm_links
        if not _is_finger_link(link) and not _is_leg_segment(link)
    ]


def _pick_distal_wrist(km: KinematicModel, side: str) -> str | None:
    arm_links = _arm_chain_links(km, side)
    if arm_links:
        return max(arm_links, key=lambda name: (km.depth(name), name))

    def is_arm_leaf(name: str) -> bool:
        if not _is_arm_segment(name):
            return False
        children = km.children_of.get(name, ())
        return not any(_is_arm_segment(c) for c in children)

    return _pick_distal_on_side(km, side, predicate=is_arm_leaf)


def _pick_elbow(km: KinematicModel, side: str, wrist: str | None) -> str | None:
    arm_links = _arm_chain_links(km, side)
    elbows = [
        l for l in arm_links
        if "elbow" in l.lower() or "lower_arm" in l.lower() or "forearm" in l.lower()
    ]
    elbows.extend(_links_for_joint_keyword(km, side, "elbow"))
    elbows = list(dict.fromkeys(elbows))
    if not elbows and wrist is not None:
        chain = _ancestor_chain(km, wrist)
        chain = [l for l in chain if _is_arm_segment(l) and not _is_wrist_like(l)]
        if len(chain) >= 2:
            return chain[1]
        if chain:
            return chain[0]
    if not elbows:
        return None
    if wrist is not None:
        ancestors = [l for l in elbows if l != wrist and km.is_ancestor(l, wrist)]
        if ancestors:
            return max(ancestors, key=lambda name: (km.depth(name), name))
    return max(elbows, key=lambda name: (km.depth(name), name))


def _pick_shoulder(km: KinematicModel, side: str, elbow: str | None) -> str | None:
    arm_links = _arm_chain_links(km, side)
    shoulders = [
        l for l in arm_links
        if "shoulder" in l.lower() or "upper_arm" in l.lower() or "clavicle" in l.lower()
    ]
    shoulders.extend(_links_for_joint_keyword(km, side, "shoulder"))
    shoulders = list(dict.fromkeys(shoulders))
    if not shoulders and elbow is not None:
        chain = _ancestor_chain(km, elbow)
        chain = [l for l in chain if _is_arm_segment(l)]
        if chain:
            # pitch–roll–yaw gimbals (Atom01 / RPO): target the middle link.
            roll_hits = [l for l in chain if "roll" in l.lower()]
            if roll_hits:
                return roll_hits[-1]
            return chain[-1]
    if not shoulders:
        return None
    if elbow is not None:
        ancestors = [l for l in shoulders if km.is_ancestor(l, elbow)]
        if ancestors:
            return min(ancestors, key=lambda name: (km.depth(name), name))
    return min(shoulders, key=lambda name: (km.depth(name), name))


def _pick_distal_ankle(km: KinematicModel, side: str) -> str | None:
    leg_links = km._links_with_tokens(side, _LEG_TOKENS)
    if not leg_links:
        return None
    # Prefer roll / foot over pitch when equally distal.
    def rank(name: str) -> tuple[int, int, str]:
        lower = name.lower()
        pref = 0
        if "ankle_roll" in lower or lower.endswith("_foot_link") or "_foot_" in lower:
            pref = 0
        elif "ankle" in lower or "foot" in lower:
            pref = 1
        else:
            pref = 2
        return (pref, -km.depth(name), name)

    return min(leg_links, key=rank)


def _pick_knee(km: KinematicModel, side: str, ankle: str | None) -> str | None:
    leg_links = km._links_with_tokens(side, _LEG_TOKENS)
    knees = [l for l in leg_links if _is_knee_like(l)]
    knees.extend(_links_for_joint_keyword(km, side, "knee"))
    knees = list(dict.fromkeys(knees))
    if not knees and ankle is not None:
        chain = _ancestor_chain(km, ankle)
        for link in chain:
            if link == ankle or _is_foot_like(link) or _is_ankle_like(link):
                continue
            if _is_hip_like(link):
                break
            if _is_leg_segment(link):
                return link
    if not knees:
        return None
    if ankle is not None:
        ancestors = [l for l in knees if l != ankle and km.is_ancestor(l, ankle)]
        if ancestors:
            return max(ancestors, key=lambda name: (km.depth(name), name))
    return max(knees, key=lambda name: (km.depth(name), name))


def _pick_hip(km: KinematicModel, side: str, knee: str | None) -> str | None:
    leg_links = km._links_with_tokens(side, _LEG_TOKENS)
    hips = [
        l for l in leg_links
        if "hip" in l.lower() or "thigh" in l.lower()
    ]
    if not hips:
        return None
    if knee is not None:
        ancestors = [l for l in hips if km.is_ancestor(l, knee)]
        if ancestors:
            roll_hits = [l for l in ancestors if "roll" in l.lower()]
            if roll_hits:
                return max(roll_hits, key=lambda name: (km.depth(name), name))
            return min(ancestors, key=lambda name: (km.depth(name), name))
    roll_hits = [l for l in hips if "roll" in l.lower()]
    if roll_hits:
        return max(roll_hits, key=lambda name: (km.depth(name), name))
    return min(hips, key=lambda name: (km.depth(name), name))


def _pick_trunk(km: KinematicModel) -> str | None:
    candidates = [
        l for l in km.all_links
        if is_trunk_link(l) and not is_limb_link(l)
    ]
    if not candidates:
        return None

    def rank(name: str) -> tuple[int, int, str]:
        lower = name.lower()
        if "torso" in lower or lower == "trunk":
            return (0, km.depth(name), name)
        if "waist" in lower or "spine" in lower:
            return (1, km.depth(name), name)
        if "pelvis" in lower:
            return (2, km.depth(name), name)
        if "base" in lower:
            return (3, km.depth(name), name)
        return (4, km.depth(name), name)

    return min(candidates, key=rank)


def _resolve_hips_chest_duplicate(
    km: KinematicModel,
    ik_map: dict[str, str],
) -> list[str]:
    """Split ``hips`` / ``chest`` when scaffold mapped both to the same link.

    Inverted humanoids (e.g. Booster T1: floating ``Trunk`` + leg root
    ``Waist``) and pelvis-root rigs with a separate arm trunk (Ultron:
    ``base_link`` legs + ``trunk_link`` arms) need distinct pelvis vs upper-
    torso targets.  When both slots collide on one link, use the LCA of the
    leg chains for ``hips`` and the LCA of the arm chains for ``chest``.
    """
    hips = ik_map.get("hips")
    chest = ik_map.get("chest")
    if not hips or not chest or hips != chest:
        return []

    lh, rh = ik_map.get("left_hip"), ik_map.get("right_hip")
    ls, rs = ik_map.get("left_shoulder"), ik_map.get("right_shoulder")
    leg_lca = km.lca(lh, rh) if lh and rh else None
    sh_lca = km.lca(ls, rs) if ls and rs else None
    changes: list[str] = []
    if leg_lca and sh_lca and leg_lca != sh_lca:
        ik_map["hips"] = leg_lca
        ik_map["chest"] = sh_lca
        changes.append(
            f"hips/chest: split duplicate {hips!r} → hips={leg_lca!r}, chest={sh_lca!r}"
        )
    elif sh_lca and sh_lca != hips:
        ik_map["chest"] = sh_lca
        changes.append(f"chest: {hips!r} → {sh_lca!r} (was duplicate of hips)")
    return changes


def _drop_chest_duplicate_of_hips(ik_map: dict[str, str]) -> str | None:
    """Drop ``chest`` when it still shares ``hips``' link (single rigid trunk).

    Mini humanoids (e.g. Booster K1) attach legs and arms to one ``Trunk``
    link; a second chest IK target on that link over-constrains the solver.
    Keep ``hips`` for root / pelvis tracking.
    """
    hips = ik_map.get("hips")
    chest = ik_map.get("chest")
    if hips and chest and hips == chest:
        ik_map.pop("chest")
        return chest
    return None


def _path_toward(km: KinematicModel, start: str, end: str) -> list[str]:
    """Links on the chain from ``start`` toward ``end`` (inclusive of ``start``)."""
    path: list[str] = []
    cur: str | None = start
    seen: set[str] = set()
    while cur and cur not in seen:
        seen.add(cur)
        path.append(cur)
        if cur == end:
            break
        nxt = None
        for child in km.children_of.get(cur, ()):
            if child == end or km.is_ancestor(child, end):
                nxt = child
                break
        if nxt is None:
            break
        cur = nxt
    return path


def _gimbal_smooth_masks_on_path(
    km: KinematicModel,
    start: str,
    end: str,
) -> dict[str, float]:
    """Per-link smoother weights for a 2–3 link gimbal segment."""
    path = _path_toward(km, start, end)
    if len(path) < 2:
        return {}
    masks: dict[str, float] = {path[0]: 0.1}
    for link in path[1:3]:
        jn = km.joint_for_child.get(link, "").lower()
        if "roll" in jn or "_r_" in jn or jn.endswith("_r"):
            masks[link] = 1.0
        elif "yaw" in jn or "_y_" in jn or jn.endswith("_y"):
            masks[link] = 0.3
        else:
            masks[link] = 0.1
    return masks


def infer_smooth_joint_filter_masks(
    urdf_path: Path,
    ik_map: dict[str, str],
) -> dict[str, float]:
    """Infer gimbal smoother masks from URDF topology + ``ik_map``."""
    km = KinematicModel.from_urdf(urdf_path)
    masks: dict[str, float] = {}
    for side in ("left", "right"):
        sh, el = ik_map.get(f"{side}_shoulder"), ik_map.get(f"{side}_elbow")
        if sh and el:
            masks.update(_gimbal_smooth_masks_on_path(km, sh, el))
        hip, kn = ik_map.get(f"{side}_hip"), ik_map.get(f"{side}_knee")
        if hip and kn:
            masks.update(_gimbal_smooth_masks_on_path(km, hip, kn))
    return masks


def _pick_head_neck(km: KinematicModel, trunk: str | None) -> tuple[str | None, str | None]:
    """Infer head/neck from a short non-limb branch off the trunk."""
    if not trunk:
        return None, None

    explicit_neck = [
        l for l in km.all_links
        if "neck" in l.lower() and not is_limb_link(l)
    ]
    explicit_head = [
        l for l in km.all_links
        if any(kw in l.lower() for kw in ("head", "skull"))
        and not is_limb_link(l)
        and not _is_end_effector_link(l)
        and not _is_sensor_link(l)
    ]
    if explicit_head:
        head = min(explicit_head, key=lambda name: (km.depth(name), name))
        neck_hits = [l for l in explicit_neck if km.is_ancestor(l, head) or l == head]
        neck = neck_hits[0] if neck_hits else None
        return neck, head

    branches: list[list[str]] = []
    for child in km.children_of.get(trunk, ()):
        if _is_sensor_link(child):
            continue
        if _branch_is_limb(km, child, leg=True) or _branch_is_limb(km, child, leg=False):
            continue
        chain: list[str] = []
        cur: str | None = child
        seen: set[str] = set()
        while cur and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            kids = km.children_of.get(cur, ())
            if len(kids) != 1:
                break
            cur = kids[0]
        if 1 <= len(chain) <= 4:
            branches.append(chain)

    if not branches:
        return None, None

    best = max(branches, key=len)
    if len(best) >= 2:
        return best[-2], best[-1]
    return None, None


def _pick_hips(km: KinematicModel) -> str | None:
    for name in km.all_links:
        lower = name.lower()
        if lower in {"pelvis", "base_link", "hips", "root_link"}:
            return name
        tokens = _tokenise(lower)
        if "pelvis" in tokens or (tokens & {"base", "root"} and "link" in tokens):
            return name
    return km.base_link or None


def infer_ik_map_from_kinematics(urdf_path: Path) -> dict[str, str]:
    """Topology-first ik_map for standard humanoid slots."""
    km = KinematicModel.from_urdf(urdf_path)
    out: dict[str, str] = {}

    hips = _pick_hips(km)
    if hips:
        out["hips"] = hips

    trunk = _pick_trunk(km)
    if trunk:
        # Map trunk pose via ``chest`` only — a second ``spine`` target on the
        # same link over-constrains single-DOF torsos (Atom01/RPO) and matches
        # upstream soma-retargeter (Hips + Chest, no Spine1).
        out["chest"] = trunk

    neck, head = _pick_head_neck(km, trunk or hips)
    if neck:
        out["neck"] = neck
    if head:
        out["head"] = head

    for side in ("left", "right"):
        wrist = _pick_distal_wrist(km, side)
        elbow = _pick_elbow(km, side, wrist)
        shoulder = _pick_shoulder(km, side, elbow)
        ankle = _pick_distal_ankle(km, side)
        knee = _pick_knee(km, side, ankle)
        hip = _pick_hip(km, side, knee)

        if wrist:
            out[f"{side}_wrist"] = wrist
        if elbow:
            out[f"{side}_elbow"] = elbow
        if shoulder:
            out[f"{side}_shoulder"] = shoulder
        if ankle:
            out[f"{side}_ankle"] = ankle
        if knee:
            out[f"{side}_knee"] = knee
        if hip:
            out[f"{side}_hip"] = hip

    _resolve_hips_chest_duplicate(km, out)
    _drop_chest_duplicate_of_hips(out)
    return out


def _validate_duplicate_trunk_slots(ik_map: dict[str, str]) -> list[IkMapIssue]:
    """Flag when multiple trunk slots share one URDF link."""
    by_link: dict[str, list[str]] = {}
    for slot, link in ik_map.items():
        if not link or slot not in ("hips", "chest", "spine"):
            continue
        by_link.setdefault(link, []).append(slot)
    issues: list[IkMapIssue] = []
    for link, slots in by_link.items():
        if len(slots) > 1:
            issues.append(
                IkMapIssue(
                    slots[0],
                    f"→ {link!r} is shared with {', '.join(slots[1:])} "
                    f"(over-constrains IK and misaligns the yellow overlay)",
                )
            )
    return issues


def _validate_slot_link(
    km: KinematicModel,
    slot: str,
    link: str,
) -> list[IkMapIssue]:
    """Anatomical checks for one ``(slot, link)`` pair."""
    issues: list[IkMapIssue] = []

    if link not in km.all_links:
        return [IkMapIssue(slot, f"→ {link!r} is not a URDF link")]

    if slot.startswith("left_") or slot.startswith("right_"):
        want_side = "left" if slot.startswith("left_") else "right"
        got_side = km.side(link)
        if got_side and got_side != want_side:
            issues.append(
                IkMapIssue(
                    slot,
                    f"→ {link!r} is on the {got_side} side, expected {want_side}",
                )
            )

    if slot in ("spine", "chest", "neck", "head") and is_limb_link(link):
        issues.append(
            IkMapIssue(slot, f"→ {link!r} looks like a limb link, not trunk")
        )

    if slot == "head":
        if _is_end_effector_link(link) or km.side(link):
            issues.append(
                IkMapIssue(slot, f"→ {link!r} is not a head link")
            )

    if slot.endswith("_knee") and _is_hip_like(link):
        issues.append(
            IkMapIssue(slot, f"→ {link!r} looks like a hip/thigh, not a knee")
        )

    if slot.endswith("_wrist") and _is_leg_segment(link):
        issues.append(
            IkMapIssue(slot, f"→ {link!r} looks like a leg link, not a wrist")
        )

    if slot.endswith("_elbow") and _is_leg_segment(link):
        issues.append(
            IkMapIssue(slot, f"→ {link!r} looks like a leg link, not an elbow")
        )

    if slot.endswith("_shoulder") and _is_leg_segment(link):
        issues.append(
            IkMapIssue(slot, f"→ {link!r} looks like a leg link, not a shoulder")
        )

    return issues


def validate_ik_map(
    urdf_path: Path,
    ik_map: dict[str, str],
) -> list[IkMapIssue]:
    """Return anatomically inconsistent ``ik_map`` entries."""
    if not ik_map:
        return []
    km = KinematicModel.from_urdf(urdf_path)
    issues: list[IkMapIssue] = _validate_duplicate_trunk_slots(ik_map)

    for slot, link in ik_map.items():
        if not link:
            continue
        issues.extend(_validate_slot_link(km, slot, link))

    for side in ("left", "right"):
        wrist = ik_map.get(f"{side}_wrist")
        elbow = ik_map.get(f"{side}_elbow")
        if wrist and elbow:
            if km.depth(wrist) < km.depth(elbow):
                issues.append(
                    IkMapIssue(
                        f"{side}_wrist",
                        f"link {wrist!r} is proximal to elbow {elbow!r} "
                        f"(depth {km.depth(wrist)} < {km.depth(elbow)})",
                    )
                )
            elif not km.is_ancestor(elbow, wrist) and elbow != wrist:
                issues.append(
                    IkMapIssue(
                        f"{side}_wrist",
                        f"elbow {elbow!r} is not an ancestor of wrist {wrist!r}",
                    )
                )

        ankle = ik_map.get(f"{side}_ankle")
        knee = ik_map.get(f"{side}_knee")
        if ankle and knee:
            if km.depth(ankle) < km.depth(knee):
                issues.append(
                    IkMapIssue(
                        f"{side}_ankle",
                        f"link {ankle!r} is proximal to knee {knee!r}",
                    )
                )
            elif not km.is_ancestor(knee, ankle) and knee != ankle:
                issues.append(
                    IkMapIssue(
                        f"{side}_ankle",
                        f"knee {knee!r} is not an ancestor of ankle {ankle!r}",
                    )
                )

    return issues


def repair_ik_map(
    urdf_path: Path,
    ik_map: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    """Replace invalid slots with topology-inferred links.

    Returns ``(repaired_map, change_log)``.  Unfixable issues are left
    untouched; run :func:`validate_ik_map` on the result to see what remains.
    """
    kinematic = infer_ik_map_from_kinematics(urdf_path)
    out = dict(ik_map)
    changes: list[str] = []

    for slot, link in kinematic.items():
        if slot not in out or not out[slot]:
            if slot == "chest" and out.get("hips") == link:
                continue
            out[slot] = link
            changes.append(f"{slot}: added {link!r}")

    for _ in range(len(out) + 1):
        issues = validate_ik_map(urdf_path, out)
        if not issues:
            break
        progressed = False
        for issue in issues:
            slot = issue.slot
            if slot not in kinematic:
                continue
            new_link = kinematic[slot]
            if out.get(slot) == new_link:
                continue
            old = out.get(slot)
            out[slot] = new_link
            if old is None:
                changes.append(f"{slot}: added {new_link!r}")
            else:
                changes.append(f"{slot}: {old!r} → {new_link!r}")
            progressed = True
            break
        if not progressed:
            break

    return out, changes


def _side_has_arm_chain(km: KinematicModel, side: str) -> bool:
    return bool(_arm_chain_links(km, side))


def _strip_arm_slots_without_chain(
    urdf_path: Path,
    ik_map: dict[str, str],
    changes: list[str],
) -> None:
    """Drop shoulder/elbow/wrist when the URDF has no arm chain on that side."""
    km = KinematicModel.from_urdf(urdf_path)
    for side in ("left", "right"):
        if _side_has_arm_chain(km, side):
            continue
        for slot in (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"):
            if slot in ik_map:
                old = ik_map.pop(slot)
                changes.append(f"{slot}: removed (no {side} arm chain, was {old!r})")


def prepare_ik_map(
    urdf_path: Path,
    ik_map: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    """Repair ``ik_map`` and drop optional slots that remain invalid.

    Returns ``(prepared_map, change_log)``.  Critical slots are always filled
    from topology when possible; unfixable critical issues remain for
    :func:`require_valid_ik_map` to surface.
    """
    repaired, changes = repair_ik_map(urdf_path, ik_map)
    _strip_arm_slots_without_chain(urdf_path, repaired, changes)
    if (
        repaired.get("spine")
        and repaired.get("chest")
        and repaired["spine"] == repaired["chest"]
    ):
        old = repaired.pop("spine")
        changes.append(f"spine: removed duplicate of chest ({old!r})")
    changes.extend(_resolve_hips_chest_duplicate(
        KinematicModel.from_urdf(urdf_path), repaired,
    ))
    dropped_chest = _drop_chest_duplicate_of_hips(repaired)
    if dropped_chest is not None:
        changes.append(
            f"chest: removed duplicate of hips ({dropped_chest!r})"
        )
    if (
        repaired.get("spine")
        and repaired.get("hips")
        and repaired["spine"] == repaired["hips"]
    ):
        old = repaired.pop("spine")
        changes.append(f"spine: removed duplicate of hips ({old!r})")
    issues = validate_ik_map(urdf_path, repaired)
    for issue in issues:
        slot = issue.slot
        if slot in CRITICAL_IK_SLOTS:
            # Arm slots on armless robots are optional (Berkeley biped, etc.).
            if slot.endswith(("_wrist", "_elbow", "_shoulder")):
                if slot in repaired:
                    old = repaired.pop(slot)
                    changes.append(f"{slot}: removed invalid {old!r}")
            continue
        if slot in repaired:
            old = repaired.pop(slot)
            changes.append(f"{slot}: removed invalid {old!r}")
    return repaired, changes


def mujoco_body_names(model) -> frozenset[str]:
    """Named MuJoCo bodies in ``model`` (excludes ``world``)."""
    import mujoco

    out: set[str] = set()
    for i in range(int(model.nbody)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name != "world":
            out.add(name)
    return frozenset(out)


def _mujoco_has_body(mj_bodies: frozenset[str], name: str) -> bool:
    if name in mj_bodies:
        return True
    short = name.removesuffix("_link")
    return short in mj_bodies and short != name


def _mujoco_pick_body(mj_bodies: frozenset[str], name: str) -> str | None:
    if name in mj_bodies:
        return name
    short = name.removesuffix("_link")
    if short in mj_bodies:
        return short
    return None


def resolve_urdf_link_to_mujoco_body(
    urdf_path: Path,
    mj_model,
    link: str,
    *,
    urdf_base: str | None = None,
    km: KinematicModel | None = None,
) -> str:
    """Map a URDF link name to a body that exists in a compiled MuJoCo model.

    MuJoCo's URDF importer merges the root link into ``worldbody`` (handled via
    ``floating_base`` after :func:`~hhtools.retarget.interaction_mesh.mujoco_scene._ensure_freejoint`)
    and **collapses fixed-joint children** (``torso_link``, ``imu_link``,
    ``*_end_effector_link``, …) into their parent.  An ``ik_map`` entry that
    points at a valid URDF link can therefore be absent from ``mj_model`` —
    the interaction-mesh backend must walk the URDF parent chain until it hits a
    body MuJoCo kept (e.g. ``torso_link`` → ``waist_yaw_link``).
    """
    mj_bodies = mujoco_body_names(mj_model)
    direct = _mujoco_pick_body(mj_bodies, link)
    if direct is not None:
        return direct

    tree = km if km is not None else KinematicModel.from_urdf(urdf_path)
    base = urdf_base or tree.base_link

    if link == base and "floating_base" in mj_bodies:
        return "floating_base"

    cur = link
    seen: set[str] = set()
    while cur and cur not in seen:
        seen.add(cur)
        parent = tree.parent_of.get(cur)
        if parent is None:
            break
        hit = _mujoco_pick_body(mj_bodies, parent)
        if hit is not None:
            return hit
        if parent == base and "floating_base" in mj_bodies:
            return "floating_base"
        cur = parent

    raise ValueError(
        f"MuJoCo body not found for URDF link {link!r} "
        f"(walked parents in {urdf_path.name}; "
        f"available bodies: {sorted(mj_bodies)})"
    )


def resolve_urdf_links_to_mujoco_bodies(
    urdf_path: Path,
    mj_model,
    links: list[str],
    *,
    urdf_base: str | None = None,
    log_remaps: bool = True,
) -> list[str]:
    """Resolve each URDF link in ``links`` to an addressable MuJoCo body."""
    import logging

    _log = logging.getLogger(__name__)
    km = KinematicModel.from_urdf(urdf_path)
    base = urdf_base or km.base_link
    out: list[str] = []
    for link in links:
        resolved = resolve_urdf_link_to_mujoco_body(
            urdf_path, mj_model, link, urdf_base=base, km=km,
        )
        if log_remaps and resolved != link:
            _log.info(
                "MuJoCo ik_map remap: %r → %r (fixed-joint / merged link)",
                link, resolved,
            )
        out.append(resolved)
    return out


def require_valid_ik_map(
    urdf_path: Path,
    ik_map: dict[str, str],
    *,
    robot_name: str,
) -> None:
    """Raise :class:`ValueError` when ``ik_map`` is not safe for retarget."""
    prepared, _changes = prepare_ik_map(urdf_path, ik_map)
    issues = validate_ik_map(urdf_path, prepared)
    km = KinematicModel.from_urdf(urdf_path)
    critical: list[IkMapIssue] = []
    for issue in issues:
        slot = issue.slot
        is_critical = (
            slot in CRITICAL_IK_SLOTS
            or slot.endswith("_knee")
            or slot == "head"
            or "is shared with" in issue.message
        )
        if slot.endswith(("_wrist", "_elbow", "_shoulder")):
            side = "left" if slot.startswith("left_") else "right"
            if not _side_has_arm_chain(km, side):
                is_critical = False
        if is_critical:
            critical.append(issue)
    if not critical:
        # Mutate caller dict in-place so pipelines see the repaired map.
        ik_map.clear()
        ik_map.update(prepared)
        return
    kinematic = infer_ik_map_from_kinematics(urdf_path)
    lines = [issue.format() for issue in critical]
    hints: list[str] = []
    for issue in critical:
        if issue.slot in kinematic:
            hints.append(
                f"  {issue.slot}: {kinematic[issue.slot]!r}"
            )
    msg = (
        f"robot {robot_name!r} has invalid ik_map — fix before retargeting:\n"
        + "\n".join(f"  • {line}" for line in lines)
    )
    if hints:
        msg += (
            "\nSuggested topology-inferred links:\n"
            + "\n".join(hints)
            + f"\nRun: hhtools robot validate {robot_name} --fix"
        )
    raise ValueError(msg)
