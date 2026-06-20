# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Sniff which mimic-profile dataset adapter fits a motion file.

Combines **path hints** (parent folder names like ``SOMA/``, ``AMASS/``) with
**lightweight content inspection** (NPZ keys, NPY column width, BVH joint names,
PyTorch checkpoint layout) so a standalone drop is routed to the correct adapter
even when the parent directory name is missing.

Registered mimic datasets (see :data:`hhtools.viewer.library._DIR_TO_ADAPTER`):

* ``amass``, ``motion_x``, ``phuma`` — SMPL-family parameter files
* ``soma``, ``lafan``, ``xsens_mocap`` — BVH dialects
* ``gvhmr``, ``kungfu_athlete`` — HMR4D ``.pt`` results
* ``meshmimic_holosoma`` — holosoma ``.npy`` + ``source.yaml``
* ``unified_npz`` — hhtools internal NPZ schema
* ``glb`` — authored rig clips
"""

from __future__ import annotations

import json
from pathlib import Path

from hhtools.viewer.library import _DIR_TO_ADAPTER, _normalise_dirname

_SOURCE_MANIFEST = "source.yaml"


def path_dataset_hint(path: Path, *, depth: int = 4) -> str | None:
    """Map ancestor folder names to a registered adapter name."""
    current = path.parent
    for _ in range(depth):
        adapter = _DIR_TO_ADAPTER.get(_normalise_dirname(current.name))
        if adapter is not None:
            return adapter
        if current.parent == current:
            break
        current = current.parent
    return None


def _has_holosoma_manifest(start: Path) -> bool:
    current = start
    for _ in range(4):
        if (current / _SOURCE_MANIFEST).is_file():
            return True
        if current.parent == current:
            break
        current = current.parent
    return False


def _is_parc_ms_npz(path: Path) -> bool:
    parent = path.parent
    stem = path.stem
    if (parent / f"{stem}_terrain.obj").is_file():
        return True
    return parent.name == stem


def _is_omomo_pkl(path: Path) -> bool:
    parent = path.parent
    stem = path.stem
    if (parent / f"{stem}_cleaned_simplified.obj").is_file():
        return True
    return any(parent.glob("*_cleaned_simplified.obj"))


def _is_parc_ms_pkl(path: Path) -> bool:
    parent = path.parent
    stem = path.stem
    if (parent / f"{stem}_terrain.obj").is_file():
        return True
    return parent.name == stem


def sniff_npz_dataset(path: Path) -> str:
    """Classify an ``.npz`` without loading a full :class:`~hhtools.core.motion.Motion`."""
    import numpy as np

    hint = path_dataset_hint(path)
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)

        if {"schema_version", "bone_names", "positions"}.issubset(keys):
            if hint:
                return hint
            if _is_parc_ms_npz(path):
                return "parc_ms"
            if "meta_json" in keys:
                try:
                    meta = json.loads(str(data["meta_json"].item()))
                    declared = str(meta.get("dataset", "")).strip()
                    if declared in _DIR_TO_ADAPTER.values() or declared in {
                        "parc_ms",
                        "meshmimic_holosoma",
                        "unified_npz",
                    }:
                        return declared
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            return "unified_npz"

        if "poses" in keys or ("pose_body" in keys and "trans" in keys):
            if hint in {"motion_x", "phuma", "amass", "gvhmr"}:
                return hint
            return "amass"

    return hint or "amass"


def sniff_npy_dataset(path: Path) -> str:
    """Classify an ``.npy`` by array width and clip layout."""
    import numpy as np

    hint = path_dataset_hint(path)
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=True)
        if getattr(arr, "ndim", 0) == 2:
            width = int(arr.shape[1])
            if width == 322:
                return "motion_x"
            if width == 69:
                return "phuma"
    except Exception:
        pass

    if path.stem == path.parent.name or _has_holosoma_manifest(path.parent):
        return "meshmimic_holosoma"
    if hint:
        return hint
    return "meshmimic_holosoma"


def sniff_pt_dataset(path: Path) -> str:
    """Classify HMR4D-style ``.pt`` / ``.pth`` checkpoints."""
    hint = path_dataset_hint(path)
    if hint in {"gvhmr", "kungfu_athlete"}:
        return hint
    try:
        import torch

        data = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "smpl_params_global" in data:
            return hint or "gvhmr"
    except Exception:
        pass
    return hint or "gvhmr"


def sniff_pkl_dataset(path: Path) -> str:
    """Classify standalone ``.pkl`` when not already routed to intermimic/meshmimic."""
    if _is_omomo_pkl(path):
        return "omomo"
    if _is_parc_ms_pkl(path):
        return "parc_ms"
    hint = path_dataset_hint(path)
    return hint or "omomo"


def infer_mimic_dataset(
    path: str | Path,
    *,
    bone_names: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Return the registered adapter name for a mimic-profile clip path."""
    path = Path(path)
    suf = path.suffix.lower()
    hint = path_dataset_hint(path)

    if suf in (".glb", ".gltf"):
        return "glb"
    if suf == ".bvh":
        from hhtools.io.bvh_detect import infer_bvh_dataset

        return infer_bvh_dataset(path, bone_names=bone_names)
    if suf == ".npz":
        return sniff_npz_dataset(path)
    if suf == ".npy":
        return sniff_npy_dataset(path)
    if suf in (".pt", ".pth"):
        return sniff_pt_dataset(path)
    if suf == ".pkl":
        return sniff_pkl_dataset(path)

    return hint or "amass"
