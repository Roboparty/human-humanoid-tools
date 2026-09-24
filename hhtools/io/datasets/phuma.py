"""PHUMA dataset adapter.

PHUMA (Lee et al., 2025) ships curated human-motion clips as ``.npy`` arrays of shape
``(T, 69)``. The layout is documented at
https://github.com/DAVIAN-Robotics/PHUMA/blob/main/src/curation/preprocess_motionx_format.py
and mirrors the Motion-X SMPL-X convention it was derived from:

    [transl(3), global_orient(3), body_pose(63)]

* ``transl`` — global pelvis translation in metres.
* ``global_orient`` — root axis-angle rotation.
* ``body_pose`` — 21 **SMPL-X** body joints × 3 axis-angle values (NOT 23 SMPL joints).

Up-axis for PHUMA .npy clips is **Y-up** (the SMPL-X canonical frame). The README notes a
Y→Z transform only for ``.npz`` (Motion-X stageii) inputs; plain ``.npy`` clips such as
``data/human_pose/example/kick.npy`` are passed through untouched. We therefore mark the
returned :class:`SmplMotionParams` as ``up_axis="Y"`` so the viewer's automatic Y→Z
conversion applies.

Betas and framerate are not stored in the ``.npy``; we default to a neutral shape
(``betas=zeros(10)``) and 30 fps. Both are overridable via
``load_motion(..., betas=..., framerate=...)``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from hhtools.bodymodels.params import SmplMotionParams
from hhtools.core.motion import Motion
from hhtools.io.datasets._engine_cache import motion_from_params
from hhtools.io.datasets.base import DatasetAdapter, register_dataset

_DEFAULT_FRAMERATE = 30.0


@register_dataset
class PhumaAdapter(DatasetAdapter):
    name = "phuma"
    display_name = "PHUMA"
    requires = "smplx"
    file_patterns = ("*.npy",)

    def list_sequences(self) -> Iterator[str]:
        if not self.root.exists():
            return
        for p in sorted(self.root.rglob("*.npy")):
            if p.is_file():
                yield str(p.relative_to(self.root))

    def _resolve(self, sequence_id: str) -> Path:
        p = (self.root / sequence_id).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"PHUMA sequence not found: {p}")
        return p

    def load_params(
        self,
        sequence_id: str,
        *,
        framerate: float | None = None,
        gender: str = "neutral",
        betas: np.ndarray | None = None,
    ) -> SmplMotionParams:
        path = self._resolve(sequence_id)
        raw = np.load(path, allow_pickle=True).astype(np.float32)
        if raw.ndim != 2:
            raise ValueError(f"Expected 2D PHUMA array, got shape={raw.shape} for {path}")
        if raw.shape[1] != 69:
            raise ValueError(
                f"Unsupported PHUMA layout: shape[1]={raw.shape[1]} (expected 69)"
            )

        # Official PHUMA layout: [transl(3), global_orient(3), body_pose(63 = 21*3)]
        trans = raw[:, 0:3]
        root_orient = raw[:, 3:6]
        body_pose = raw[:, 6:69]  # 63 dims = SMPL-X body (21 joints)

        return SmplMotionParams(
            surface_model="smplx",
            root_orient=root_orient,
            body_pose=body_pose,
            betas=np.zeros(10, dtype=np.float32) if betas is None else betas.astype(np.float32),
            trans=trans,
            gender=gender,  # type: ignore[arg-type]
            framerate=_DEFAULT_FRAMERATE if framerate is None else framerate,
            up_axis="Y",
            meta={"dataset": "phuma", "sequence_id": sequence_id},
        )

    def load_motion(self, sequence_id: str, **kwargs: Any) -> Motion:
        with_mesh = bool(kwargs.pop("with_mesh", False))
        progress_callback = kwargs.pop("progress_callback", None)
        params = self.load_params(sequence_id, **kwargs)
        return motion_from_params(
            params,
            name=Path(sequence_id).stem,
            source_format="phuma/smpl",
            return_mesh=with_mesh,
            progress_callback=progress_callback,
        )


__all__ = ["PhumaAdapter"]
