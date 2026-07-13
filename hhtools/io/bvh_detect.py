# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Detect SOMA vs LAFAN / Mixamo BVH dialect from path hints and joint names."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from hhtools.retarget.newton_basic.human_aliases import (
    is_mixamo_cmu_like,
    is_mocap_spine3_bvh_like,
    is_soma_bvh_like,
    is_xsens_mocap_like,
)
from hhtools.viewer.library import _DIR_TO_ADAPTER, _normalise_dirname

_BVH_DATASET_HINTS = frozenset({"soma", "lafan", "mocap", "xsens_mocap"})


def iter_bvh_hierarchy_lines(path: str | Path) -> Iterator[str]:
    """Yield stripped HIERARCHY lines until ``MOTION`` (streams; skips motion data)."""
    path = Path(path)
    in_hierarchy = False
    with path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped == "HIERARCHY":
                in_hierarchy = True
                continue
            if not in_hierarchy:
                continue
            if stripped.startswith("MOTION"):
                break
            yield stripped


def bvh_hierarchy_has_joints(path: str | Path) -> bool:
    """True when the BVH has at least one ``JOINT`` (human skeleton, not object track).

    Parkour captures such as ``20260429_mocap`` ship sibling ``*_rig.bvh`` files that
    only contain a single ``ROOT`` (obstacle / prop transform).
    """
    for stripped in iter_bvh_hierarchy_lines(path):
        if stripped.startswith("JOINT"):
            return True
    return False


def read_bvh_joint_names(path: str | Path) -> tuple[str, ...]:
    """Extract ``JOINT`` / ``ROOT`` names from the HIERARCHY section (no motion parse)."""
    names: list[str] = []
    for stripped in iter_bvh_hierarchy_lines(path):
        if stripped.startswith(("ROOT", "JOINT")):
            parts = stripped.split()
            if len(parts) >= 2:
                names.append(parts[1])
    return tuple(names)


def _path_dataset_hint(path: Path) -> str | None:
    for parent in (path.parent, path.parent.parent, path.parent.parent.parent):
        adapter = _DIR_TO_ADAPTER.get(_normalise_dirname(parent.name))
        if adapter in _BVH_DATASET_HINTS:
            return adapter
    return None


def infer_bvh_dataset_from_joints(
    joint_names: tuple[str, ...] | list[str],
    *,
    path_hint: str | None = None,
) -> str | None:
    """Return ``'soma'``, ``'lafan'``, ``'mocap'``, or ``'xsens_mocap'`` from bone names."""
    names = tuple(joint_names)
    soma = is_soma_bvh_like(names)
    lafan = is_mixamo_cmu_like(names)
    mocap = is_mocap_spine3_bvh_like(names)
    xsens = is_xsens_mocap_like(names)
    if xsens and not soma and not lafan and not mocap:
        return "xsens_mocap"
    if mocap and not soma:
        return "mocap"
    if soma and not lafan and not mocap:
        return "soma"
    if lafan and not soma and not mocap:
        return "lafan"
    if soma and lafan:
        return "soma" if "LeftUpLeg" not in names else "lafan"
    if path_hint in _BVH_DATASET_HINTS:
        return path_hint
    return None


def infer_bvh_dataset(
    path: str | Path,
    *,
    bone_names: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Best-effort adapter name for a ``.bvh`` clip."""
    path = Path(path)
    hint = _path_dataset_hint(path)
    if bone_names is None:
        try:
            bone_names = read_bvh_joint_names(path)
        except OSError:
            return hint or "lafan"
    detected = infer_bvh_dataset_from_joints(bone_names, path_hint=hint)
    return detected or hint or "lafan"
