"""Meshmimic · Holosoma adapter (human + static terrain).

Clip layout — mirrors ``intermimic/OMOMO/<clip>/<clip>.pkl`` so each clip sits
in its own folder and is self-identifying by stem:

    assets/motions/meshmimic/holosoma/
    ├── source.yaml                            # shared manifest (skeleton, fps, license)
    ├── parkour_1/
    │   ├── parkour_1.npy                      # (T, 53, 3) float32, world-frame joint positions
    │   └── terrain.obj                        # static triangle mesh, same world frame
    ├── parkour_2/{parkour_2.npy, terrain.obj}
    └── ...

Source: the holosoma climb demo bundle (``holosoma/src/holosoma_retargeting/
demo_data/climb/mocap_climb_seq_*``), cropped to the first 5 sequences and
reshuffled into the ``parkour_<n>`` naming used throughout this repo. The
``MOCAP_DEMO_JOINTS`` skeleton (53 joints) is SMPL-H-like with every finger
bone exposed and two extra sole markers (``LeftFootMod`` / ``RightFootMod``)
that the holosoma interaction-mesh Laplacian uses to form a heel-ankle-forefoot
triangle per foot.

Why a shared ``source.yaml`` rather than a per-clip metadata file:
every clip under ``meshmimic/holosoma/`` comes from the same capture rig with
the same skeleton, framerate, and coordinate conventions, so the adapter
resolves those exactly once per folder. The clip directory is then pure data
(``<clip>.npy`` + ``terrain.obj``) and stays trivially easy to add / remove /
re-import.

Rotations policy: ``joint_positions.npy`` holds global joint **positions only**,
no orientations. We follow the existing hhtools convention established by
:mod:`hhtools.io.datasets.omomo` — set ``quaternions`` to xyzw identity, flag
``meta["rotations_source"] = "none"``, and leave authoritative rotation
recovery to a future SMPL-H forward pass. Current Web rendering, analytics,
and position-only retarget paths ignore global bone quaternions, so this loses
no information for supported workflows.

Terrain wiring: the per-clip ``terrain.obj`` is surfaced as a single
:class:`SceneObject` with per-frame translations set to zero and rotations set
to identity, and ``mesh_path`` pointing at the absolute ``.obj`` path. The
The Web scene serializer loads these meshes for the Stage. The
``center_motion_root_xy`` and ``snap_motion_to_ground`` helpers shift scene
objects as well, preserving terrain/skeleton alignment.

CLI / library plumbing:

* ``hhtools/services/motion_library.py`` maps ``holosoma`` to
  "meshmimic_holosoma"``, so the ``meshmimic/holosoma/*/<clip>.npy`` files are
  auto-discovered by the viewer library scan.
* When the viewer calls the adapter it instantiates it rooted at the clip
  folder (``EphemeralCache._convert`` uses ``entry.source_path.parent``), so
  the adapter looks one level up for ``source.yaml``. When the CLI or tests
  instantiate at the dataset root directly (``meshmimic/holosoma/``) the
  same lookup finds ``source.yaml`` in place.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from hhtools.core.hierarchy import Hierarchy
from hhtools.core.motion import Motion
from hhtools.core.scene import TerrainHeightfield
from hhtools.io.datasets.base import DatasetAdapter, register_dataset

_SOURCE_MANIFEST_NAME = "source.yaml"
_DEFAULT_TERRAIN_FILENAME = "terrain.obj"
# Default heightfield resolution.  0.05 m / cell is the same value used by
# the unified npz / parc_ms loader, so heightfields built from the same
# clip — regardless of which adapter ran — share an identical grid.
_DEFAULT_HEIGHTFIELD_DX = 0.05
# Padding (m) of flat ground added around the OBJ XY bbox so the
# character has a small landing area outside the modelled terrain.
_DEFAULT_HEIGHTFIELD_PADDING = 0.5

# Minimum number of samples we'll emit after a user-forced downsample; we never
# return a completely empty Motion because downstream renderers require at
# least one frame.
_MIN_FRAMES_AFTER_DOWNSAMPLE = 1


def _find_source_yaml(start: Path) -> Path:
    """Locate ``source.yaml`` at ``start`` or any ancestor up to 3 levels.

    Walking up lets the adapter be instantiated either at the dataset root
    (``meshmimic/holosoma/``) or at any individual clip folder
    (``meshmimic/holosoma/parkour_1/``) — both find the same manifest.
    """
    current = start
    for _ in range(4):
        cand = current / _SOURCE_MANIFEST_NAME
        if cand.is_file():
            return cand
        if current.parent == current:
            break
        current = current.parent
    raise FileNotFoundError(
        f"{_SOURCE_MANIFEST_NAME} not found at or above {start}. "
        "meshmimic/holosoma clips require a shared manifest describing the "
        "skeleton topology, framerate, and coordinate conventions."
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    with open(path) as f:
        manifest = yaml.safe_load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: expected a mapping at the YAML top level")
    return manifest


def _parse_framerate(manifest: dict[str, Any]) -> tuple[float, int]:
    """Return ``(raw_fps, recommended_downsample)`` from the manifest.

    We keep both because the adapter may return either the raw capture or a
    pre-downsampled version depending on caller preference; downstream consumers
    want the *effective* framerate but may also care about the raw rate for
    noise / filtering heuristics.
    """
    fr = manifest.get("framerate", {})
    raw_hz = float(fr.get("raw_hz", 120.0))
    ds = int(fr.get("recommended_downsample", 1))
    if ds <= 0:
        ds = 1
    return raw_hz, ds


def _parse_skeleton(manifest: dict[str, Any]) -> Hierarchy:
    skel = manifest.get("skeleton")
    if not isinstance(skel, dict):
        raise ValueError("source.yaml: missing/invalid `skeleton` section")
    names = list(skel.get("joint_names", []))
    parents_raw = list(skel.get("parent_indices", []))
    if len(names) != len(parents_raw):
        raise ValueError(
            f"source.yaml: joint_names ({len(names)}) and parent_indices "
            f"({len(parents_raw)}) must be the same length"
        )
    if not names:
        raise ValueError("source.yaml: skeleton.joint_names is empty")
    parents = np.asarray(parents_raw, dtype=np.int32)
    return Hierarchy.from_parent_indices(names, parents)


def _load_terrain_heightfield(
    clip_dir: Path, terrain_filename: str
) -> TerrainHeightfield | None:
    """Load (or one-shot build + cache) the per-clip heightfield.

    Resolution order:

    1. ``<clip_dir>/<clip_dir.name>.pkl`` — PARC-format sidecar produced
       by ``scripts/build_terrain_heightfield_sidecars.py``.  Decoded
       directly via :func:`load_parc_pkl_terrain` — no rasterisation.
    2. ``<clip_dir>/<terrain_filename>`` — legacy OBJ.  Rasterised once
       via :func:`obj_to_heightfield` and the resulting grid is **also
       persisted** as a sidecar ``.pkl`` so subsequent loads hit case 1.

    Returns ``None`` only when neither input is present.  All three
    downstream consumers (Web rendering, MuJoCo MPC-SQP, PARC export) read from
    the returned :class:`TerrainHeightfield` directly — the OBJ stays
    on disk as a debugging artefact but never participates in runtime
    flow once the sidecar exists.
    """
    from hhtools.io.parc_export import load_parc_pkl_terrain, save_parc_pkl
    from hhtools.retarget.interaction_mesh.heightfield import obj_to_heightfield

    pkl_path = clip_dir / f"{clip_dir.name}.pkl"
    if pkl_path.is_file():
        terrain = load_parc_pkl_terrain(pkl_path)
        if terrain is not None:
            return terrain

    obj_path = clip_dir / terrain_filename
    if not obj_path.is_file():
        return None

    terrain = obj_to_heightfield(
        obj_path,
        dx=_DEFAULT_HEIGHTFIELD_DX,
        padding=_DEFAULT_HEIGHTFIELD_PADDING,
    )
    # Persist as a sidecar so subsequent loads / PARC training share the
    # exact same grid we just rasterised.  Motion data is left empty —
    # the holosoma adapter ships .npy without per-joint quaternions, so
    # there is no ``MSMotionData`` to fill yet; the batch sidecar tool
    # is the right place to populate motion_data when needed.
    try:
        save_parc_pkl(pkl_path, motion_data=None, terrain_data=terrain, misc_data=None)
    except OSError:
        # Read-only checkout; carry on with the in-memory heightfield.
        pass
    return terrain


def _apply_downsample(arr: NDArray, stride: int) -> NDArray:
    if stride <= 1:
        return arr
    kept = arr[::stride]
    if kept.shape[0] < _MIN_FRAMES_AFTER_DOWNSAMPLE:
        return arr[: _MIN_FRAMES_AFTER_DOWNSAMPLE]
    return kept


def _validate_positions(arr: NDArray, n_joints: int, source_path: Path) -> NDArray:
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(
            f"{source_path}: expected joint_positions shape (T, J, 3); got {arr.shape}"
        )
    if arr.shape[1] != n_joints:
        raise ValueError(
            f"{source_path}: skeleton expects {n_joints} joints but file has "
            f"{arr.shape[1]}. Did you mix meshmimic/holosoma with a different source?"
        )
    return arr.astype(np.float32)


@register_dataset
class MeshmimicHolosomaAdapter(DatasetAdapter):
    """Adapter for meshmimic/holosoma clips (human mocap + static terrain mesh).

    One :class:`DatasetAdapter` per clip folder is the natural unit because the
    viewer library already carves the tree into ``folder_label · clip`` entries;
    the adapter is therefore happy to be rooted at either the dataset directory
    (``meshmimic/holosoma/``) or a specific clip directory
    (``meshmimic/holosoma/parkour_1/``). The shared ``source.yaml`` is found by
    walking up from ``self.root`` so both cases resolve identically.
    """

    name = "meshmimic_holosoma"
    display_name = "Meshmimic · Holosoma"
    requires = "skeleton"
    file_patterns = ("*.npy",)

    def list_sequences(self) -> Iterator[str]:
        """Yield ``<clip>/<clip>.npy`` entries under ``self.root`` (or a single
        ``<clip>.npy`` when rooted at an individual clip folder).

        The same-name convention (npy stem == parent folder name) lets us
        skip *.npy files that happen to live alongside a clip but are not
        themselves the canonical motion file (e.g. derived debug dumps).
        """
        if not self.root.exists():
            return
        for p in sorted(self.root.rglob("*.npy")):
            if not p.is_file():
                continue
            if p.stem != p.parent.name:
                continue
            yield str(p.relative_to(self.root))

    def _resolve(self, sequence_id: str) -> Path:
        p = (self.root / sequence_id).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"meshmimic/holosoma clip not found: {p}")
        return p

    def load_motion(
        self,
        sequence_id: str,
        *,
        framerate: float | None = None,
        downsample: int | None = None,
        **_kwargs: Any,
    ) -> Motion:
        """Load a single clip. ``downsample=None`` uses the manifest's
        ``recommended_downsample`` (holosoma ships at 120 Hz, default 4×
        downsample → 30 Hz for retargeting). ``framerate`` overrides the
        stored framerate after downsampling; omit it to use the derived value.
        """
        npy_path = self._resolve(sequence_id)
        clip_dir = npy_path.parent

        manifest_path = _find_source_yaml(clip_dir)
        manifest = _load_manifest(manifest_path)

        hierarchy = _parse_skeleton(manifest)
        raw_hz, default_ds = _parse_framerate(manifest)
        effective_ds = int(downsample) if downsample is not None else default_ds
        if effective_ds < 1:
            effective_ds = 1

        raw_positions = np.load(npy_path)
        positions = _validate_positions(raw_positions, hierarchy.num_bones, npy_path)
        positions = _apply_downsample(positions, effective_ds)
        num_frames = int(positions.shape[0])

        # Identity global quaternions — see module docstring. Matches the
        # pattern used by the OMOMO adapter so SkeletonRenderer /
        # CapsuleMeshRenderer / analytics all behave identically across
        # position-only datasets.
        quaternions = np.zeros((num_frames, hierarchy.num_bones, 4), dtype=np.float32)
        quaternions[..., 3] = 1.0

        effective_fps = (
            float(framerate)
            if framerate is not None
            else float(raw_hz / effective_ds)
        )
        if not math.isfinite(effective_fps) or effective_fps <= 0:
            effective_fps = raw_hz

        clip_layout = manifest.get("clip_layout", {}) or {}
        terrain_filename = str(clip_layout.get("terrain_file", _DEFAULT_TERRAIN_FILENAME))
        terrain = _load_terrain_heightfield(clip_dir, terrain_filename)

        up_axis = str(manifest.get("coordinate_system", {}).get("up_axis", "Z"))
        if up_axis not in ("X", "Y", "Z"):
            up_axis = "Z"

        meta: dict[str, Any] = {
            "dataset": "meshmimic_holosoma",
            "source": manifest.get("source", "holosoma"),
            "source_url": manifest.get("source_url", ""),
            "source_license": manifest.get("source_license", ""),
            "task": manifest.get("task", ""),
            "clip_dir": str(clip_dir.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "raw_fps": raw_hz,
            "downsample_applied": effective_ds,
            "effective_fps": effective_fps,
            "rotations_source": (
                "none (position-only mocap; quaternions left at identity, "
                "matching hhtools OMOMO convention)"
            ),
            "terrain_mesh": str((clip_dir / terrain_filename).resolve())
            if (clip_dir / terrain_filename).is_file()
            else "",
            "notes": (
                "Raw mocap joint positions in world frame (Z-up, metres). "
                "Terrain carried on Motion.terrain as a heightfield (single "
                "source of truth across Web rendering, MPC-SQP, PARC export). "
                "Center/snap operations shift terrain and skeleton "
                "together so alignment is preserved."
            ),
        }

        return Motion(
            name=npy_path.stem,
            hierarchy=hierarchy,
            positions=positions,
            quaternions=quaternions,
            framerate=effective_fps,
            up_axis=up_axis,  # type: ignore[arg-type]
            source_format="smplh",
            meta=meta,
            objects=[],
            terrain=terrain,
        )


__all__ = [
    "MeshmimicHolosomaAdapter",
    "_load_terrain_heightfield",
    "_find_source_yaml",
    "_load_manifest",
    "_parse_framerate",
    "_parse_skeleton",
]
