"""Motion-X dataset adapter.

Motion-X stores each sequence as a single ``.npy`` file of shape ``(T, 322)`` containing an
SMPL-X parameter vector per frame. The 322-dim layout comes directly from the official
IDEA-Research README_ and is::

    [0:3]       root_orient        (3)    axis-angle
    [3:66]      body_pose          (63)   21 body joints x 3
    [66:156]    pose_hand          (90)   15 left + 15 right hand joints x 3
    [156:159]   jaw_pose           (3)
    [159:209]   face_expression    (50)   FLAME expression coefficients
    [209:309]   face_shape         (100)  FLAME identity coefficients (skipped)
    [309:312]   trans              (3)
    [312:322]   betas              (10)

World-frame convention (important, verified empirically)
--------------------------------------------------------
Unlike mocap datasets with a globally fixed up-axis (AMASS = Z-up, HumanML3D = Y-up),
Motion-X's non-mocap clips are fit **from monocular videos with a per-clip camera world
frame** (see ``visualization/render_world_space_motion.py`` in the official repo, which
uses ``cameras_from_opencv_projection(R, T, ...)`` to project world-space meshes back onto
the source video — the world frame only needs to be consistent with that particular
video's camera, not with gravity).

As a concrete demonstration we probed two Motion-X clips by computing the world-frame
body-up vector each frame (``R(root_orient) @ [0,1,0]``, since SMPL-X canonical has the
head pointing along +Y):

* ``Aerial_Kick_Kungfu_wushu_1_clip1`` — mean body-up ≈ ``(+0.53, -0.03, +0.37)``
  (neither Y-up nor Z-up; tilted ~54° off any cardinal axis).
* ``Aerial_Kick_Kungfu_wushu_3_clip2`` — mean body-up ≈ ``(+0.07, -0.29, -0.89)``
  (close to **-Z**, i.e. gravity points toward +Z in this clip's world frame).

Two clips in the same subset, two different world-frame orientations.

Data-driven gravity alignment
-----------------------------
Because the raw per-clip world frame is not gravity-aligned, we align each clip to our
unified ``+Z``-up convention using a single rigid rotation computed **from the data**
(not from a hand-tuned heuristic):

1. For every frame, convert ``root_orient`` (axis-angle) to a rotation matrix ``R_t``.
2. Apply ``R_t`` to the SMPL-X canonical head direction ``[0, 1, 0]`` — this is the
   world-frame direction in which the body's spine points at frame ``t``.
3. Average this direction over all frames. The average is dominated by frames where
   the body is near-vertical (walking, standing, preparing…) so it tracks the clip's
   actual gravity-opposite direction even when it briefly contains inverted segments
   (flips, kicks) that cancel out.
4. Compute a single rigid rotation ``R_align`` mapping the estimated up-direction onto
   ``+Z``.
5. Apply ``R_align`` simultaneously to ``trans`` (rotate the pelvis trajectory) and to
   ``root_orient`` (pre-multiply each frame's rotation). Joint-local body/hand/jaw poses
   are unchanged because they are defined relative to the parent and therefore
   gravity-invariant.

The result preserves trajectory *shape* exactly — a straight jump stays straight, an arc
stays an arc — while rotating the entire world frame so that the estimated gravity-up is
``+Z`` in our viewer, matching AMASS/SOMA/LAFAN/PHUMA behaviour.

Skip condition: if the estimated up direction is already within 5° of ``+Z`` (some
Motion-X clips do happen to be authored Z-up-aligned because their source video's
camera was near-horizontal and right-side-up), no rotation is applied.

.. _README: https://github.com/IDEA-Research/Motion-X
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R  # noqa: N817

from hhtools.bodymodels.params import SmplMotionParams
from hhtools.core.motion import Motion
from hhtools.io.datasets._engine_cache import motion_from_params
from hhtools.io.datasets.base import DatasetAdapter, register_dataset


@dataclass(frozen=True)
class MotionXLayout:
    """Slice indices of each SMPL-X sub-vector inside a Motion-X row."""

    name: str
    root_orient: slice
    body_pose: slice
    hand_left: slice | None
    hand_right: slice | None
    jaw: slice | None
    expression: slice | None
    trans: slice
    betas: slice
    expected_dim: int


SMPLX_322_LAYOUT = MotionXLayout(
    name="smplx_322",
    root_orient=slice(0, 3),
    body_pose=slice(3, 66),
    hand_left=slice(66, 111),
    hand_right=slice(111, 156),
    jaw=slice(156, 159),
    expression=slice(159, 209),
    trans=slice(309, 312),
    betas=slice(312, 322),
    expected_dim=322,
)

# ---------------------------------------------------------------------------
# Data-driven gravity alignment
# ---------------------------------------------------------------------------

# SMPL-X canonical head direction: at rest the body stands with the head pointing along
# +Y (see SMPL-X paper, eq.~1 and the author's reference implementation). We use this to
# turn root_orient into a world-frame "body-up" vector for gravity estimation.
_SMPLX_CANONICAL_HEAD_UP: NDArray = np.array([0.0, 1.0, 0.0], dtype=np.float64)

# Target up-axis in our unified viewer world. Keep in sync with the rest of hhtools.
_TARGET_UP: NDArray = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# If the estimated world up is already within this many degrees of +Z, we consider the
# clip gravity-aligned and skip the rotation. A small value (not 0) keeps us robust
# against float noise but small enough that any meaningful tilt is still corrected.
_ALREADY_ALIGNED_THRESHOLD_DEG: float = 5.0


def _estimate_world_up_from_root_orient(root_orient: NDArray) -> NDArray:
    """Estimate the clip's gravity-opposite direction from the sequence of root orients.

    For each frame ``t``, ``R_t = Rodrigues(root_orient_t)`` rotates the SMPL-X canonical
    frame into the world frame. The world-frame spine/head direction of the body at that
    frame is therefore ``R_t @ [0, 1, 0]``.

    Averaging over all frames yields a vector dominated by the many frames where the body
    is near-upright (walking, standing, preparing). Short inversions (flips, kicks) tend
    to cancel out symmetrically. The returned vector is unit-normalised. The caller is
    responsible for handling the degenerate case where the average magnitude is tiny
    (which would indicate a clip spent mostly mid-rotation — our per-frame vote would be
    unreliable and the caller should refuse to rotate).
    """
    aa = np.asarray(root_orient, dtype=np.float64)  # (T, 3)
    rot = R.from_rotvec(aa)                          # (T,) rotations
    world_head_per_frame = rot.apply(_SMPLX_CANONICAL_HEAD_UP)  # (T, 3)
    avg = world_head_per_frame.mean(axis=0)
    norm = float(np.linalg.norm(avg))
    if norm < 1e-6:
        # Degenerate: return zero vector; alignment path will treat as "no estimate".
        return avg.astype(np.float64)
    return (avg / norm).astype(np.float64)


def _rotation_aligning(src: NDArray, dst: NDArray) -> R:
    """Return a scipy :class:`~scipy.spatial.transform.Rotation` that rotates ``src``
    onto ``dst``. Both must be unit vectors; a zero input yields the identity.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_norm = float(np.linalg.norm(src))
    dst_norm = float(np.linalg.norm(dst))
    if src_norm < 1e-9 or dst_norm < 1e-9:
        return R.identity()
    src = src / src_norm
    dst = dst / dst_norm
    cos = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if cos > 1.0 - 1e-9:
        return R.identity()
    if cos < -1.0 + 1e-9:
        # 180°: pick any orthogonal axis.
        helper = np.array([1.0, 0.0, 0.0]) if abs(src[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(src, helper)
        axis /= np.linalg.norm(axis)
        return R.from_rotvec(axis * np.pi)
    axis = np.cross(src, dst)
    axis /= np.linalg.norm(axis)
    angle = float(np.arccos(cos))
    return R.from_rotvec(axis * angle)


def _align_clip_to_world_up(
    root_orient: NDArray,
    trans: NDArray,
    *,
    target_up: NDArray = _TARGET_UP,
    already_aligned_threshold_deg: float = _ALREADY_ALIGNED_THRESHOLD_DEG,
) -> tuple[NDArray, NDArray, dict]:
    """Rotate ``root_orient`` and ``trans`` so the estimated world-up becomes ``target_up``.

    Returns a tuple ``(root_orient_new, trans_new, info)`` where ``info`` contains the
    estimated world-up, the applied rotation (as a quaternion xyzw) and the measured
    tilt (in degrees) before correction. If the clip is already near-aligned the inputs
    are returned unchanged and ``info['applied']`` is ``False``.
    """
    estimated_up = _estimate_world_up_from_root_orient(root_orient)
    norm = float(np.linalg.norm(estimated_up))
    if norm < 1e-3:
        return (
            root_orient.astype(np.float32),
            trans.astype(np.float32),
            {
                "applied": False,
                "reason": "degenerate_estimate",
                "estimated_world_up": estimated_up.tolist(),
            },
        )

    tilt = float(
        np.degrees(np.arccos(np.clip(float(np.dot(estimated_up, target_up)), -1.0, 1.0)))
    )
    if tilt < already_aligned_threshold_deg:
        return (
            root_orient.astype(np.float32),
            trans.astype(np.float32),
            {
                "applied": False,
                "reason": "already_aligned",
                "tilt_deg": tilt,
                "estimated_world_up": estimated_up.tolist(),
            },
        )

    r_align = _rotation_aligning(estimated_up, target_up)
    # Rotate trans vectors (world-space pelvis positions).
    trans_new = r_align.apply(np.asarray(trans, dtype=np.float64))
    # Compose rotation: for every frame, new R = r_align @ Rodrigues(aa_t).
    aa = np.asarray(root_orient, dtype=np.float64)
    r_orient = R.from_rotvec(aa)
    r_orient_new = r_align * r_orient  # composition applies r_align AFTER r_orient
    aa_new = r_orient_new.as_rotvec()
    return (
        aa_new.astype(np.float32),
        trans_new.astype(np.float32),
        {
            "applied": True,
            "tilt_deg": tilt,
            "estimated_world_up": estimated_up.tolist(),
            "align_quat_xyzw": r_align.as_quat().tolist(),
        },
    )


@register_dataset
class MotionXAdapter(DatasetAdapter):
    name = "motion_x"
    display_name = "Motion-X"
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
            raise FileNotFoundError(f"Motion-X sequence not found: {p}")
        return p

    def load_params(
        self,
        sequence_id: str,
        *,
        framerate: float = 30.0,
        gender: str = "neutral",
        layout: MotionXLayout | str = SMPLX_322_LAYOUT,
        align_world_up: bool = True,
    ) -> SmplMotionParams:
        path = self._resolve(sequence_id)
        raw = np.load(path, allow_pickle=True).astype(np.float32)
        if raw.ndim != 2:
            raise ValueError(f"Expected 2D Motion-X array, got shape={raw.shape}")
        lay = layout if isinstance(layout, MotionXLayout) else _named_layout(layout)
        if raw.shape[1] != lay.expected_dim:
            raise ValueError(
                f"Motion-X row width {raw.shape[1]} does not match layout "
                f"{lay.name!r} (expects {lay.expected_dim})"
            )

        betas = raw[0, lay.betas]  # betas are static within a Motion-X sequence
        root_orient_raw = raw[:, lay.root_orient]
        trans_raw = raw[:, lay.trans]

        if align_world_up:
            # Data-driven gravity alignment: bring each clip's estimated world-up to +Z.
            # See the module docstring for the full rationale and empirical motivation.
            root_orient_aligned, trans_aligned, align_info = _align_clip_to_world_up(
                root_orient_raw, trans_raw
            )
        else:
            root_orient_aligned = root_orient_raw.astype(np.float32)
            trans_aligned = trans_raw.astype(np.float32)
            align_info = {"applied": False, "reason": "align_world_up=False"}

        return SmplMotionParams(
            surface_model="smplx",
            root_orient=root_orient_aligned,
            body_pose=raw[:, lay.body_pose],
            betas=betas.astype(np.float32),
            trans=trans_aligned,
            gender=gender,  # type: ignore[arg-type]
            framerate=framerate,
            hand_pose_left=raw[:, lay.hand_left] if lay.hand_left else None,
            hand_pose_right=raw[:, lay.hand_right] if lay.hand_right else None,
            jaw_pose=raw[:, lay.jaw] if lay.jaw else None,
            leye_pose=None,
            reye_pose=None,
            expression=raw[:, lay.expression] if lay.expression else None,
            up_axis="Z",
            meta={
                "dataset": "motion_x",
                "sequence_id": sequence_id,
                "layout": lay.name,
                "world_up_alignment": align_info,
            },
        )

    def load_motion(self, sequence_id: str, **kwargs: Any) -> Motion:
        with_mesh = bool(kwargs.pop("with_mesh", False))
        progress_callback = kwargs.pop("progress_callback", None)
        params = self.load_params(sequence_id, **kwargs)
        return motion_from_params(
            params,
            name=Path(sequence_id).stem,
            source_format=f"motion_x/{params.meta['layout']}",
            return_mesh=with_mesh,
            progress_callback=progress_callback,
        )


def _named_layout(name: str) -> MotionXLayout:
    if name == "smplx_322":
        return SMPLX_322_LAYOUT
    raise ValueError(
        f"Unknown Motion-X layout {name!r}. Known: smplx_322. "
        f"Pass a MotionXLayout instance for custom packings."
    )


__all__ = [
    "MotionXAdapter",
    "MotionXLayout",
    "SMPLX_322_LAYOUT",
    "_align_clip_to_world_up",
    "_estimate_world_up_from_root_orient",
]
