"""AMASS dataset adapter.

AMASS ships sequences as ``.npz`` files inside per-subject sub-directories. Two major schema
versions exist:

* *Legacy* (``_poses.npz``) -- uses ``poses`` (T, 156) which concatenates SMPL+H
  ``root_orient`` (3) + ``body_pose`` (63) + ``pose_hand`` (90), plus ``trans``, ``betas`` (16)
  and ``mocap_framerate``.
* *stageii* (``_stageii.npz``) -- SMPL-X-formatted; exposes ``root_orient``, ``pose_body``,
  ``pose_hand``, ``pose_jaw``, ``pose_eye``, ``trans``, ``betas``, ``mocap_frame_rate`` and a
  ``surface_model_type`` field.

This adapter auto-detects the schema at load time.  It only depends on ``numpy``; the heavy
``smplx`` engine is loaded lazily when the caller asks for a :class:`Motion`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from hhtools.bodymodels.params import SmplMotionParams
from hhtools.core.motion import Motion
from hhtools.io.datasets._engine_cache import engine_for_params
from hhtools.io.datasets.base import DatasetAdapter, register_dataset


@register_dataset
class AmassAdapter(DatasetAdapter):
    """Adapter for the AMASS motion capture collection.

    ``root`` should point at a directory containing AMASS ``.npz`` files (possibly nested in
    sub-sub-collections like ``BMLrub/rub001/0001_walk_poses.npz``).
    """

    name = "amass"
    display_name = "AMASS"
    requires = "smplh"  # auto-escalates to smplx for stageii files
    file_patterns = ("*_poses.npz", "*_stageii.npz", "*.npz")

    def list_sequences(self) -> Iterator[str]:
        if not self.root.exists():
            return
        for npz in sorted(self.root.rglob("*.npz")):
            if not npz.is_file() or not is_amass_motion_file(npz):
                continue
            yield str(npz.relative_to(self.root))

    def _resolve(self, sequence_id: str) -> Path:
        p = (self.root / sequence_id).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"AMASS sequence not found: {p}")
        return p

    def load_params(self, sequence_id: str) -> SmplMotionParams:
        path = self._resolve(sequence_id)
        data = {k: v for k, v in np.load(path, allow_pickle=True).items()}
        return _amass_npz_to_params(data, sequence_id)

    def load_motion(self, sequence_id: str, **kwargs: Any) -> Motion:
        # ``with_mesh`` opts into baking SMPL vertices for the viewer's mesh renderer.
        # All other kwargs are ignored so callers can pass viewer-specific knobs
        # uniformly across adapters.
        with_mesh = bool(kwargs.pop("with_mesh", False))
        progress_callback = kwargs.pop("progress_callback", None)
        params = self.load_params(sequence_id)
        engine = engine_for_params(params)
        return engine.to_motion(
            params,
            name=Path(sequence_id).stem,
            source_format=f"amass/{params.surface_model}",
            return_mesh=with_mesh,
            progress_callback=progress_callback,
        )


def _amass_npz_to_params(data: dict[str, Any], sequence_id: str) -> SmplMotionParams:
    """Convert a loaded AMASS ``.npz`` dict into :class:`SmplMotionParams`.

    Detects legacy vs stageii schemas. Values are normalised to ``float32`` and gender is
    constrained to ``{male, female, neutral}``.
    """
    is_stageii = "pose_body" in data or "surface_model_type" in data
    if "trans" not in data and "poses" not in data:
        raise ValueError(
            f"AMASS 文件 {sequence_id!r} 是 stage-i 标定数据（无 trans/poses），"
            "不能作为动作 clip 加载；请选用 *_stageii.npz"
        )
    trans = np.asarray(data["trans"], dtype=np.float32)
    T = trans.shape[0]
    betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)
    # We keep the first 10 betas because ``smplx`` defaults to that count. Users can widen via
    # ``SmplxEngine(num_betas=...)`` later.
    betas = betas[:10]
    raw_gender = str(data.get("gender", "neutral"))
    gender = _normalise_gender(raw_gender)
    framerate = float(np.asarray(data.get("mocap_frame_rate", data.get("mocap_framerate", 30.0))))

    meta: dict[str, Any] = {
        "dataset": "amass",
        "sequence_id": sequence_id,
        "raw_gender": raw_gender,
    }

    if is_stageii:
        surface_model = str(data.get("surface_model_type", "smplx")).lower()
        meta["schema"] = "stageii"
        meta["surface_model_declared"] = surface_model
        pose_body = np.asarray(data["pose_body"], dtype=np.float32)
        root_orient = np.asarray(data["root_orient"], dtype=np.float32)
        pose_hand = np.asarray(data.get("pose_hand", np.zeros((T, 90), dtype=np.float32)))
        pose_hand = pose_hand.astype(np.float32, copy=False)
        hand_l = pose_hand[:, :45] if pose_hand.shape[1] >= 90 else None
        hand_r = pose_hand[:, 45:90] if pose_hand.shape[1] >= 90 else None
        jaw = np.asarray(data.get("pose_jaw"), dtype=np.float32) if "pose_jaw" in data else None
        pose_eye = data.get("pose_eye")
        leye = reye = None
        if pose_eye is not None:
            pose_eye = np.asarray(pose_eye, dtype=np.float32)
            if pose_eye.shape[1] >= 6:
                leye = pose_eye[:, :3]
                reye = pose_eye[:, 3:6]
        if surface_model in ("smplh", "mano"):
            family = "smplh"
            jaw = leye = reye = None
        else:
            family = "smplx"
        return SmplMotionParams(
            surface_model=family,  # type: ignore[arg-type]
            root_orient=root_orient,
            body_pose=pose_body,
            betas=betas,
            trans=trans,
            gender=gender,
            framerate=framerate,
            hand_pose_left=hand_l,
            hand_pose_right=hand_r,
            jaw_pose=jaw,
            leye_pose=leye,
            reye_pose=reye,
            up_axis="Z",
            meta=meta,
        )

    # Legacy schema: "poses" (T, 156) concatenates SMPL-H pose sans DMPL.
    poses = np.asarray(data["poses"], dtype=np.float32)
    if poses.shape[1] < 66:
        raise ValueError(
            f"AMASS legacy sequence {sequence_id!r} has unexpected poses shape {poses.shape}"
        )
    meta["schema"] = "legacy"
    if poses.shape[1] >= 156:
        family = "smplh"
        root_orient = poses[:, 0:3]
        pose_body = poses[:, 3:66]  # 21 body joints
        pose_hand = poses[:, 66:156]  # 15+15 hand joints
        hand_l = pose_hand[:, :45]
        hand_r = pose_hand[:, 45:90]
        jaw = leye = reye = None
    else:
        # Shorter than SMPL-H -- treat as SMPL body pose (3+69=72) or just (3+63=66).
        family = "smpl"
        root_orient = poses[:, 0:3]
        if poses.shape[1] == 72:
            pose_body = poses[:, 3:72]
        else:
            # Fallback: pad with zeros to 69 so SMPL forward is well-defined.
            body_raw = poses[:, 3:]
            pose_body = np.zeros((T, 69), dtype=np.float32)
            pose_body[:, : body_raw.shape[1]] = body_raw
        hand_l = hand_r = jaw = leye = reye = None

    return SmplMotionParams(
        surface_model=family,  # type: ignore[arg-type]
        root_orient=root_orient,
        body_pose=pose_body,
        betas=betas,
        trans=trans,
        gender=gender,
        framerate=framerate,
        hand_pose_left=hand_l,
        hand_pose_right=hand_r,
        jaw_pose=jaw,
        leye_pose=leye,
        reye_pose=reye,
        up_axis="Z",
        meta=meta,
    )


def _normalise_gender(raw: str) -> str:
    r = raw.strip().lower()
    if r.startswith("m"):
        return "male"
    if r.startswith("f"):
        return "female"
    return "neutral"


def is_amass_motion_file(path: Path | str) -> bool:
    """Return False for AMASS stage-i calibration NPZ (no pose / translation tracks)."""

    p = Path(path)
    if p.suffix.lower() != ".npz":
        return True
    stem = p.stem.lower()
    if stem.endswith("_stagei"):
        return False
    return True


__all__ = ["AmassAdapter", "is_amass_motion_file"]
