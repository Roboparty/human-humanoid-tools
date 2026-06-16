# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot-agnostic ground-collision capsule definitions for IK.

Capsules are derived from each preset's ``ik_map`` and URDF kinematics so
prone / floor-contact retarget works on any humanoid without per-robot JSON.
Explicit ``ground_collision_bodies`` in ``robot.yaml`` still override this.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from hhtools.robot.kinematics import KinematicModel

__all__ = ["build_ground_collision_bodies_from_ik_map", "resolve_ground_collision_bodies"]

# Canonical slots whose links may contact the floor during prone motion.
_GROUND_COLLISION_SLOTS: tuple[str, ...] = (
    "hips",
    "chest",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

# Distal ik_map slot used to orient a limb capsule along the bone axis.
_DISTAL_SLOT: dict[str, str] = {
    "hips": "chest",
    "left_hip": "left_knee",
    "right_hip": "right_knee",
    "left_knee": "left_ankle",
    "right_knee": "right_ankle",
    "left_shoulder": "left_elbow",
    "right_shoulder": "right_elbow",
    "left_elbow": "left_wrist",
    "right_elbow": "right_wrist",
}

# Default capsule geometry (link-local) when URDF joint offsets are tiny.
_FALLBACK_P1: dict[str, tuple[float, float, float]] = {
    "hips": (0.0, 0.0, 0.15),
    "chest": (0.0, 0.0, 0.20),
    "left_hip": (0.0, 0.0, -0.18),
    "right_hip": (0.0, 0.0, -0.18),
    "left_knee": (0.0, 0.0, -0.28),
    "right_knee": (0.0, 0.0, -0.28),
    "left_shoulder": (0.0, 0.0, -0.12),
    "right_shoulder": (0.0, 0.0, -0.12),
    "left_elbow": (0.10, 0.03, 0.0),
    "right_elbow": (0.10, -0.03, 0.0),
    "left_wrist": (0.08, 0.0, 0.0),
    "right_wrist": (0.08, 0.0, 0.0),
}

_RADIUS: dict[str, float] = {
    "hips": 0.10,
    "chest": 0.09,
    "left_hip": 0.06,
    "right_hip": 0.06,
    "left_knee": 0.05,
    "right_knee": 0.05,
    "left_shoulder": 0.04,
    "right_shoulder": 0.04,
    "left_elbow": 0.03,
    "right_elbow": 0.03,
    "left_wrist": 0.025,
    "right_wrist": 0.025,
}

_MARGIN: dict[str, float] = {
    "hips": 0.03,
    "chest": 0.03,
}


def resolve_ground_collision_bodies(
    body_labels: list[str],
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only ground-collision entries whose ``body`` link exists on the robot."""

    known = set(body_labels)
    out: list[dict[str, Any]] = []
    for entry in entries:
        body = str(entry.get("body", ""))
        if body not in known:
            continue
        cap = entry.get("capsule")
        if not isinstance(cap, (list, tuple)) or len(cap) < 3:
            continue
        out.append(entry)
    return out


def _joint_child_origins(urdf_path: Path) -> dict[str, tuple[float, float, float]]:
    """Map child link → joint ``origin xyz`` expressed in the parent frame."""

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    out: dict[str, tuple[float, float, float]] = {}
    for joint in root.findall("joint"):
        child_el = joint.find("child")
        if child_el is None:
            continue
        child = child_el.get("link")
        if not child:
            continue
        origin_el = joint.find("origin")
        xyz = (0.0, 0.0, 0.0)
        if origin_el is not None:
            raw = origin_el.get("xyz", "0 0 0")
            parts = raw.split()
            if len(parts) >= 3:
                xyz = (float(parts[0]), float(parts[1]), float(parts[2]))
        out[child] = xyz
    return out


def _child_toward_distal(
    km: KinematicModel,
    link: str,
    distal_link: str,
) -> str | None:
    """Direct child of ``link`` that lies on the chain toward ``distal_link``."""

    if link == distal_link:
        return None
    cur = distal_link
    while True:
        parent = km.parent_of.get(cur)
        if parent is None:
            return None
        if parent == link:
            return cur
        cur = parent


def _bone_endpoint(
    slot: str,
    link: str,
    km: KinematicModel,
    origins: dict[str, tuple[float, float, float]],
    ik_map: dict[str, str],
) -> tuple[float, float, float]:
    """Capsule end point in the link frame."""

    fallback = _FALLBACK_P1[slot]
    distal_slot = _DISTAL_SLOT.get(slot)
    if not distal_slot:
        return fallback
    distal_link = ik_map.get(distal_slot)
    if not distal_link:
        return fallback
    child = _child_toward_distal(km, link, distal_link)
    if child is None:
        return fallback
    vec = origins.get(child)
    if vec is None:
        return fallback
    norm_sq = vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2
    if norm_sq < 1e-6:
        return fallback
    norm = norm_sq ** 0.5
    fb_norm_sq = fallback[0] ** 2 + fallback[1] ** 2 + fallback[2] ** 2
    target_len = fb_norm_sq ** 0.5 if fb_norm_sq > 1e-6 else norm
    scale = target_len / norm
    return (vec[0] * scale, vec[1] * scale, vec[2] * scale)


def _capsule_entry(
    slot: str,
    link: str,
    km: KinematicModel,
    origins: dict[str, tuple[float, float, float]],
    ik_map: dict[str, str],
) -> dict[str, Any]:
    p1 = _bone_endpoint(slot, link, km, origins, ik_map)
    return {
        "body": link,
        "capsule": [[0.0, 0.0, 0.0], list(p1), _RADIUS[slot]],
        "margin": _MARGIN.get(slot, 0.02),
    }


def build_ground_collision_bodies_from_ik_map(
    ik_map: dict[str, str],
    urdf_path: Path,
) -> list[dict[str, Any]]:
    """Build soma-compatible ground-collision entries for any humanoid preset.

    Uses ``ik_map`` link names (never hard-coded vendor link strings) and
    orients limb capsules along the URDF parent→child joint offset when
    available.
    """

    if not ik_map or not urdf_path.is_file():
        return []

    km = KinematicModel.from_urdf(urdf_path)
    origins = _joint_child_origins(urdf_path)
    seen_links: set[str] = set()
    out: list[dict[str, Any]] = []

    for slot in _GROUND_COLLISION_SLOTS:
        link = ik_map.get(slot)
        if not link or link not in km.all_links or link in seen_links:
            continue
        seen_links.add(link)
        out.append(_capsule_entry(slot, link, km, origins, ik_map))

    return out
