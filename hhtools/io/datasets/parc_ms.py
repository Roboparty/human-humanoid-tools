"""Meshmimic · PARC MS adapter (OMOMO-style per-clip folders).

Each clip is a self-contained directory::

    parc_ms/<clip_name>/
        <clip_name>.pkl              # PARC MSFileData: motion_data + terrain_data
        <clip_name>_terrain.obj      # static terrain mesh (optional backup / Viser parity)

This mirrors ``intermimic/OMOMO/<clip>/<clip>.pkl`` + sibling ``.obj``.  The
unified ``.npz`` export is **not** required: skeleton poses are recovered from
``motion_data`` via FK using a shared 15-bone reference rig.

When a legacy folder still ships ``<clip>.npz`` next to ``<clip>.pkl``, the
library scanner lists only the NPZ (the pkl is a terrain sidecar).  Run
``scripts/parc_ms_to_omomo_layout.py`` to drop redundant NPZ files so clips
surface as standalone ``.pkl`` entries like OMOMO.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hhtools.core.motion import Motion
from hhtools.io.datasets.base import DatasetAdapter, register_dataset
from hhtools.io.parc_import import (
    heightfield_to_wavefront_obj,
    motion_from_ms_pickle,
    rest_offsets_local_from_reference_npz,
)
from hhtools.io.parc_ms_skeleton import default_parc_ms_skeleton_bundle

_log = logging.getLogger(__name__)


def _terrain_obj_path(clip_pkl: Path) -> Path:
    return clip_pkl.parent / f"{clip_pkl.stem}_terrain.obj"


@register_dataset
class ParcMsAdapter(DatasetAdapter):
    """PARC MS clips: one ``.pkl`` per folder (+ optional ``*_terrain.obj``)."""

    name = "parc_ms"
    display_name = "PARC MS"
    requires = "skeleton"
    file_patterns = ("*.pkl",)

    def list_sequences(self) -> Iterator[str]:
        if not self.root.exists():
            return
        for p in sorted(self.root.rglob("*.pkl")):
            if not p.is_file():
                continue
            # Skip terrain-only sidecars when a primary npz/npy exists (legacy layout).
            stem = p.stem
            parent = p.parent
            if any(
                (parent / f"{stem}{ext}").is_file()
                for ext in (".npz", ".npy", ".bvh", ".glb", ".gltf")
            ):
                continue
            yield str(p.relative_to(self.root))

    def load_motion(
        self,
        sequence_id: str,
        *,
        reference_npz: str | Path | None = None,
        **_kwargs: Any,
    ) -> Motion:
        path = (self.root / sequence_id).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"parc_ms clip not found: {path}")

        from hhtools.io.loader_progress import pop_progress_callback

        progress_callback = pop_progress_callback(_kwargs)
        npz_sidecar = path.with_suffix(".npz")
        if npz_sidecar.is_file():
            from hhtools.io.npz import load_npz

            motion = load_npz(npz_sidecar, progress_callback=progress_callback)
            motion.meta.setdefault("dataset", "parc_ms")
            motion.meta["split_terrain_grounding"] = True
            motion.meta["source_pkl"] = str(path)
            if motion.terrain is None:
                obj = _terrain_obj_path(path)
                if obj.is_file():
                    from hhtools.retarget.interaction_mesh.heightfield import (
                        obj_to_heightfield,
                    )

                    motion.terrain = obj_to_heightfield(
                        obj,
                        dx=0.05,
                        padding=0.5,
                        object_position=(0.0, 0.0, 0.0),
                        object_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                        mesh_scale=1.0,
                    )
            motion.meta["terrain_mesh"] = str(_terrain_obj_path(path))
            return motion

        if reference_npz is not None:
            bundle = rest_offsets_local_from_reference_npz(reference_npz)
        else:
            bundle = default_parc_ms_skeleton_bundle()

        motion = motion_from_ms_pickle(
            path,
            skeleton_bundle=bundle,
            attach_terrain_heightfield=True,
        )

        if motion.terrain is None:
            obj = _terrain_obj_path(path)
            if obj.is_file():
                from hhtools.io.parc_export import save_parc_pkl
                from hhtools.retarget.interaction_mesh.heightfield import obj_to_heightfield

                terrain = obj_to_heightfield(
                    obj,
                    dx=0.05,
                    padding=0.5,
                    object_position=(0.0, 0.0, 0.0),
                    object_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    mesh_scale=1.0,
                )
                motion.terrain = terrain
                sidecar = path
                try:
                    save_parc_pkl(
                        sidecar, motion_data=None, terrain_data=terrain, misc_data=None
                    )
                except OSError as exc:
                    _log.warning("could not cache terrain into %s: %s", sidecar, exc)

        obj = _terrain_obj_path(path)
        if motion.terrain is not None and not obj.is_file():
            try:
                heightfield_to_wavefront_obj(motion.terrain, obj)
            except OSError as exc:
                _log.warning("could not write terrain OBJ %s: %s", obj, exc)

        motion.meta.setdefault("dataset", "parc_ms")
        motion.meta["split_terrain_grounding"] = True
        motion.meta["terrain_mesh"] = str(obj) if obj.is_file() else None
        return motion
