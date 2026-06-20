# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Detect SOMA vs LAFAN / Mixamo BVH dialect from path hints and joint names."""

from __future__ import annotations

from pathlib import Path

from hhtools.retarget.newton_basic.human_aliases import (
    is_mixamo_cmu_like,
    is_soma_bvh_like,
    is_xsens_mocap_like,
)
from hhtools.viewer.library import _DIR_TO_ADAPTER, _normalise_dirname

_BVH_DATASET_HINTS = frozenset({"soma", "lafan", "xsens_mocap"})


def read_bvh_joint_names(path: str | Path) -> tuple[str, ...]:
    """Extract ``JOINT`` / ``ROOT`` names from the HIERARCHY section (no motion parse)."""
    path = Path(path)
    names: list[str] = []
    in_hierarchy = False
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped == "HIERARCHY":
            in_hierarchy = True
            continue
        if not in_hierarchy:
            continue
        if stripped.startswith("MOTION"):
            break
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
    """Return ``'soma'``, ``'lafan'``, or ``'xsens_mocap'`` from bone names."""
    names = tuple(joint_names)
    soma = is_soma_bvh_like(names)
    lafan = is_mixamo_cmu_like(names)
    xsens = is_xsens_mocap_like(names)
    if xsens and not soma and not lafan:
        return "xsens_mocap"
    if soma and not lafan:
        return "soma"
    if lafan and not soma:
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
