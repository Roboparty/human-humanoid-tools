"""Core rules for compact human-skeleton visualization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class HumanHierarchy(Protocol):
    """Minimal hierarchy shape needed by compact-visualization rules."""

    bone_names: Sequence[str]
    parent_indices: object


class HumanMotion(Protocol):
    """Minimal motion shape accepted without importing concrete adapters."""

    hierarchy: HumanHierarchy
    positions: object
    num_frames: int


VIRTUAL_ROOT_NAMES: frozenset[str] = frozenset(
    {
        "root",
        "reference",
        "world",
        "armature",
        "origin",
        "root_body",
        "scene_root",
        "body_world",
        "worldroot",
        "world_root",
        "rig",
        "skeleton",
        "skeleton_root",
        "body",
        "character",
        "main",
        "rootnode",
    }
)

SMALL_BONE_KEYWORDS: tuple[str, ...] = (
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
    "finger",
    "toe",
    "eye",
    "jaw",
    "eyeball",
    "end",
    "tip",
    "headend",
)

_EXTRA_COMPACT_EXCLUDE: tuple[str, ...] = (
    "phalange",
    "metacarpal",
    "carpal",
    "metatarsal",
    "tarsal",
    "handtip",
    "footmod",
)


def detect_virtual_root(bone_names: list[str]) -> bool:
    """Return whether joint zero is a scene wrapper rather than a real pelvis."""

    if not bone_names:
        return False
    token = bone_names[0].strip().lower()
    for separator in (":", "|", "/"):
        if separator in token:
            token = token.split(separator)[-1]
    return token in VIRTUAL_ROOT_NAMES


def degenerate_auxiliary_bone_indices(
    motion: HumanMotion,
    frame: int = 0,
    *,
    eps: float = 2e-3,
) -> set[int]:
    """Return coincident FootMod markers that must not become fake segments."""

    names = tuple(motion.hierarchy.bone_names)
    parents = np.asarray(motion.hierarchy.parent_indices, dtype=np.int64)
    selected_frame = int(np.clip(frame, 0, max(0, motion.num_frames - 1)))
    positions = np.asarray(motion.positions[selected_frame], dtype=np.float64)
    excluded: set[int] = set()
    for index, name in enumerate(names):
        parent = int(parents[index])
        if parent < 0 or not name.lower().endswith("footmod"):
            continue
        if float(np.linalg.norm(positions[index] - positions[parent])) <= float(eps):
            excluded.add(index)
    return excluded


def exclude_joint_from_compact_scaled_preview(name: str) -> bool:
    """Return whether a dense face, finger, toe, or auxiliary joint is visual noise."""

    normalized = str(name).lower()
    return any(token in normalized for token in SMALL_BONE_KEYWORDS) or any(
        token in normalized for token in _EXTRA_COMPACT_EXCLUDE
    )


def compact_skeleton_exclude_indices(motion: HumanMotion) -> set[int]:
    """Return name-based exclusions for a compact body skeleton."""

    return {
        index
        for index, name in enumerate(motion.hierarchy.bone_names)
        if exclude_joint_from_compact_scaled_preview(name)
    }


def _bone_basename(name: str) -> str:
    value = str(name)
    for separator in ("|", ":", "/"):
        if separator in value:
            value = value.rsplit(separator, maxsplit=1)[-1]
    return value.strip()


def hand_foot_subtree_exclude_indices(motion: HumanMotion) -> set[int]:
    """Exclude descendants below hand and foot hubs regardless of naming scheme."""

    names = motion.hierarchy.bone_names
    parents = np.asarray(motion.hierarchy.parent_indices, dtype=np.int64)
    hubs: set[int] = set()
    for index, raw_name in enumerate(names):
        name = _bone_basename(raw_name).lower()
        if name.endswith("hand") and "forearm" not in name:
            hubs.add(index)
        elif name.endswith("foot") and not name.endswith("footmod"):
            hubs.add(index)
        elif name.endswith("feet"):
            hubs.add(index)

    excluded: set[int] = set()
    for index in range(len(names)):
        ancestor = index
        while ancestor >= 0:
            if ancestor in hubs and ancestor != index:
                excluded.add(index)
                break
            ancestor = int(parents[ancestor])
    return excluded


def dense_rig_viz_exclude_indices(motion: HumanMotion) -> set[int]:
    """Return the complete compact-body mask for dense human rigs."""

    return compact_skeleton_exclude_indices(motion) | hand_foot_subtree_exclude_indices(
        motion
    )


__all__ = [
    "SMALL_BONE_KEYWORDS",
    "VIRTUAL_ROOT_NAMES",
    "compact_skeleton_exclude_indices",
    "degenerate_auxiliary_bone_indices",
    "dense_rig_viz_exclude_indices",
    "detect_virtual_root",
    "exclude_joint_from_compact_scaled_preview",
    "hand_foot_subtree_exclude_indices",
]
