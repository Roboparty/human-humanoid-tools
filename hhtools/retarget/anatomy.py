"""Retarget-aware skeleton helpers that depend on canonical alias resolution."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hhtools.core.anatomy import deepest_mapped_canonicals
from hhtools.core.human_anatomy import hand_foot_subtree_exclude_indices
from hhtools.core.motion import Motion
from hhtools.retarget.newton_basic.human_aliases import auto_source_to_canonical


def exclude_unmapped_head_neck_from_scaled_preview(
    name: str,
    *,
    ik_map_canonicals: frozenset[str],
) -> bool:
    """Return whether an unmapped head/neck joint should be hidden in a scaled preview."""

    if "head" in ik_map_canonicals:
        return False
    canonical = auto_source_to_canonical((str(name),)).get(str(name), str(name))
    return str(canonical).lower() in ("head", "neck")


def scaled_hand_tip_positions_world(
    motion: Motion,
    scaler: object,
    side: str,
) -> NDArray[np.float32] | None:
    """Return each frame's farthest scaled hand descendant beyond the wrist."""

    names = list(motion.hierarchy.bone_names)
    parents = np.asarray(motion.hierarchy.parent_indices, dtype=np.int64)
    source_to_canonical = auto_source_to_canonical(tuple(names))

    wrist_index = next(
        (
            index
            for index, raw_name in enumerate(names)
            if str(source_to_canonical.get(raw_name, raw_name)).lower() == f"{side}_wrist"
        ),
        None,
    )
    if wrist_index is None:
        return None

    children: list[list[int]] = [[] for _ in names]
    for index, parent in enumerate(parents):
        if int(parent) >= 0:
            children[int(parent)].append(index)

    stack = [wrist_index]
    descendants: list[int] = []
    while stack:
        joint = stack.pop()
        descendants.append(joint)
        stack.extend(children[joint])
    tips = [joint for joint in descendants if joint != wrist_index] or [wrist_index]

    scaled = scaler.scale_world_points_about_root(
        motion,
        motion.positions.astype(np.float32, copy=False),
    )
    wrist_position = scaled[:, wrist_index, :]
    tip_positions = scaled[:, tips, :]
    distances = np.linalg.norm(tip_positions - wrist_position[:, None, :], axis=2)
    farthest = np.argmax(distances, axis=1)
    return tip_positions[np.arange(scaled.shape[0]), farthest, :].astype(
        np.float32,
        copy=False,
    )


def scaled_overlay_exclude_bone_indices(
    motion: Motion,
    ik_map_canonicals: frozenset[str],
) -> set[int]:
    """Return source descendants that extend beyond the robot's mapped IK terminals."""

    if not ik_map_canonicals:
        return set()
    terminals = deepest_mapped_canonicals(ik_map_canonicals)
    if not terminals:
        return set()

    names = tuple(motion.hierarchy.bone_names)
    source_to_canonical = auto_source_to_canonical(names)
    terminal_source_indices = {
        index
        for index, raw_name in enumerate(names)
        if str(source_to_canonical.get(raw_name, raw_name)).lower() in terminals
    }
    parents = np.asarray(motion.hierarchy.parent_indices, dtype=np.int64)
    excluded: set[int] = set()
    for index in range(len(names)):
        ancestor = index
        while ancestor >= 0:
            if ancestor in terminal_source_indices and ancestor != index:
                excluded.add(index)
                break
            ancestor = int(parents[ancestor])
    excluded |= hand_foot_subtree_exclude_indices(motion)
    return excluded


__all__ = [
    "exclude_unmapped_head_neck_from_scaled_preview",
    "scaled_hand_tip_positions_world",
    "scaled_overlay_exclude_bone_indices",
]
