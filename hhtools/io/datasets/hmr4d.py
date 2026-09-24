"""Adapter for GVHMR / KungFuAthlete ``hmr4d_results.pt`` files.

The official result stores 21 body-joint rotations plus SMPL-X shape, root orientation, and
translation parameters under ``smpl_params_global``.  GVHMR itself predicts with the neutral
SMPL-X model, so this adapter reuses that licensed model when present.  Existing SMPL-H and
SMPL installations remain supported as compatibility fallbacks for older imported results.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from hhtools.bodymodels.params import SmplMotionParams
from hhtools.core.motion import Motion
from hhtools.io.datasets._engine_cache import engine_for_params, motion_from_params
from hhtools.io.datasets.base import DatasetAdapter, register_dataset

_DEFAULT_FRAMERATE = 30.0


def _to_numpy(x: Any) -> np.ndarray:
    try:
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    except Exception:
        return np.asarray(x)


def _load_hmr4d(path: Path) -> SmplMotionParams:
    import torch

    from hhtools.bodymodels.compat import patch_chumpy_compat
    from hhtools.bodymodels.paths import find_body_model

    patch_chumpy_compat()
    # Imported ``.pt`` files are an untrusted boundary. The expected GVHMR
    # document is a plain dict of tensors, so PyTorch's restricted loader
    # preserves the supported format without allowing arbitrary pickle globals
    # to execute in the desktop process.
    data = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a tensor dictionary")
    block = data.get("smpl_params_global")
    if block is None:
        raise ValueError(
            f"{path} does not contain 'smpl_params_global'; is this an HMR4D results file?"
        )
    body_pose_21 = _to_numpy(block["body_pose"]).astype(np.float32)  # (T, 63)
    betas = _to_numpy(block["betas"]).astype(np.float32)
    global_orient = _to_numpy(block["global_orient"]).astype(np.float32)
    transl = _to_numpy(block["transl"]).astype(np.float32)

    if body_pose_21.ndim != 2 or body_pose_21.shape[1] != 63:
        raise ValueError(
            f"Unexpected body_pose shape {body_pose_21.shape} in {path}; expected (T, 63)."
        )
    # HMR4D fixes betas per frame; flatten to a single per-sequence shape vector for SMPL.
    if betas.ndim == 2:
        betas_flat = betas.mean(axis=0)
    else:
        betas_flat = betas.reshape(-1)

    # GVHMR itself requires SMPL-X neutral weights, so prefer that same licensed
    # file when available. Existing installations with only SMPL-H keep their
    # previous behavior; SMPL remains the final zero-padded fallback below.
    surface_model = "smplx" if find_body_model("smplx", "neutral") else "smplh"
    return SmplMotionParams(
        surface_model=surface_model,
        root_orient=global_orient,
        body_pose=body_pose_21,  # (T, 63), shared SMPL-X/SMPL-H body-joint convention
        betas=betas_flat,
        trans=transl,
        gender="neutral",
        framerate=_DEFAULT_FRAMERATE,
        hand_pose_left=None,
        hand_pose_right=None,
        up_axis="Y",  # HMR4D typically emits Y-up
        meta={"dataset": "hmr4d", "source_path": str(path)},
    )


class _Hmr4dBase(DatasetAdapter):
    requires = "smplx"
    file_patterns = ("*.pt", "*.pth")

    def list_sequences(self) -> Iterator[str]:
        if not self.root.exists():
            return
        for p in sorted(self.root.rglob("*.pt")):
            if p.is_file():
                yield str(p.relative_to(self.root))

    def _resolve(self, sequence_id: str) -> Path:
        p = (self.root / sequence_id).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"HMR4D results file not found: {p}")
        return p

    def load_params(self, sequence_id: str) -> SmplMotionParams:
        return _load_hmr4d(self._resolve(sequence_id))

    def load_motion(self, sequence_id: str, **kwargs: Any) -> Motion:
        with_mesh = bool(kwargs.pop("with_mesh", False))
        progress_callback = kwargs.pop("progress_callback", None)
        params = self.load_params(sequence_id)
        try:
            engine = engine_for_params(params)
        except FileNotFoundError:
            if params.surface_model == "smplx":
                # A separately configured SMPL-H installation can still load
                # the 21-joint body pose when SMPL-X lookup unexpectedly fails.
                params = SmplMotionParams(
                    surface_model="smplh",
                    root_orient=params.root_orient,
                    body_pose=params.body_pose,
                    betas=params.betas,
                    trans=params.trans,
                    gender=params.gender,
                    framerate=params.framerate,
                    up_axis=params.up_axis,
                    meta=params.meta,
                )
                try:
                    engine = engine_for_params(params)
                except FileNotFoundError:
                    engine = None
            else:
                engine = None
            if engine is not None:
                return engine.to_motion(
                    params,
                    name=Path(sequence_id).stem,
                    source_format=f"hmr4d/{params.surface_model}",
                    return_mesh=with_mesh,
                    progress_callback=progress_callback,
                )
            # SMPL-X / SMPL-H weights missing; fall back to SMPL by padding
            # the 21-joint body pose to SMPL's 23-joint convention.
            padded = np.zeros((params.num_frames, 69), dtype=np.float32)
            padded[:, :63] = params.body_pose
            params = SmplMotionParams(
                surface_model="smpl",
                root_orient=params.root_orient,
                body_pose=padded,
                betas=params.betas,
                trans=params.trans,
                gender=params.gender,
                framerate=params.framerate,
                up_axis=params.up_axis,
                meta=params.meta,
            )
            return motion_from_params(
                params,
                name=Path(sequence_id).stem,
                source_format=f"hmr4d/{params.surface_model}",
                return_mesh=with_mesh,
                progress_callback=progress_callback,
            )
        return engine.to_motion(
            params,
            name=Path(sequence_id).stem,
            source_format=f"hmr4d/{params.surface_model}",
            return_mesh=with_mesh,
            progress_callback=progress_callback,
        )


@register_dataset
class GvhmrAdapter(_Hmr4dBase):
    name = "gvhmr"
    display_name = "GVHMR (World-grounded Video HMR)"


@register_dataset
class KungFuAthleteAdapter(_Hmr4dBase):
    name = "kungfu_athlete"
    display_name = "KungFuAthlete (HMR4D extraction)"


__all__ = ["GvhmrAdapter", "KungFuAthleteAdapter"]
