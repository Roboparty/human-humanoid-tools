"""Yellow scaled overlay stays co-aligned with scaled terrain (parc_ms / OMOMO)."""

from __future__ import annotations

import numpy as np
import pytest

from hhtools.core.hierarchy import Hierarchy
from hhtools.core.motion import Motion
from hhtools.core.scene import TerrainHeightfield
from hhtools.web.analysis.scaled_preview import (
    _uniform_scaled_joint_positions,
    resolve_scaled_overlay_z_correction,
)


def _parc_like_motion(*, z_floor: float = 0.0) -> Motion:
    """Minimal 15-bone parc_ms-style clip with feet on a flat heightfield."""
    names = (
        "pelvis",
        "torso",
        "head",
        "right_upper_arm",
        "right_lower_arm",
        "right_hand",
        "left_upper_arm",
        "left_lower_arm",
        "left_hand",
        "right_thigh",
        "right_shin",
        "right_foot",
        "left_thigh",
        "left_shin",
        "left_foot",
    )
    parents = (-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13)
    parent_names: list[str | None] = [None]
    for p in parents[1:]:
        parent_names.append(names[p])
    hierarchy = Hierarchy(
        bone_names=list(names),
        parent_indices=list(parents),
        parent_names=parent_names,
    )
    pos = np.zeros((2, len(names), 3), dtype=np.float32)
    pos[:, 0, 2] = 1.0  # pelvis
    pos[:, 11, 2] = z_floor  # right_foot on ground
    pos[:, 14, 2] = z_floor  # left_foot on ground
    quat = np.zeros((2, len(names), 4), dtype=np.float32)
    quat[..., 3] = 1.0
    hf = np.full((4, 4), z_floor, dtype=np.float32)
    terrain = TerrainHeightfield(
        hf=hf,
        hf_maxmin=np.stack([hf, hf], axis=-1),
        min_point=np.array([-1.0, -1.0], dtype=np.float32),
        dx=0.5,
    )
    return Motion(
        name="parc_test",
        hierarchy=hierarchy,
        positions=pos,
        quaternions=quat,
        framerate=30.0,
        terrain=terrain,
        meta={"dataset": "parc_ms", "split_terrain_grounding": True},
    )


class _FakeScaler:
    def __init__(self, root_z: float) -> None:
        self.config = type("C", (), {"root_joint": "pelvis"})()
        self.joint_names = ("pelvis",)
        self._root_z = root_z

    def apply(self, motion: Motion):
        from hhtools.retarget.newton_basic.scaler import ScaledEffectors

        t = np.zeros((motion.num_frames, 1, 7), dtype=np.float32)
        t[:, 0, 2] = self._root_z
        t[:, 0, 6] = 1.0
        return ScaledEffectors(joint_names=("pelvis",), transforms=t)


def test_interaction_scene_skips_overlay_z_correction() -> None:
    motion = _parc_like_motion()
    scaler = _FakeScaler(root_z=0.85)
    # Would be non-zero without the terrain guard.
    assert resolve_scaled_overlay_z_correction(motion, scaler, ratio=0.85) == 0.0


def test_scaled_overlay_preserves_foot_terrain_gap() -> None:
    """Foot–terrain clearance scales uniformly (no extra pelvis correction)."""
    from hhtools.core.grounding import (
        human_source_floor_z_world,
        terrain_heightfield_z_offset_world,
    )
    from hhtools.retarget.calibration.calibration import uniform_overlay_scale_for_motion
    from hhtools.retarget.newton_basic.config import ScalerConfig

    z_floor = 0.05
    motion = _parc_like_motion(z_floor=z_floor)
    z_min = float(human_source_floor_z_world(motion))
    z_terrain = float(terrain_heightfield_z_offset_world(motion, z_min))

    foot_i = motion.hierarchy.bone_names.index("right_foot")
    src_gap = float(motion.positions[0, foot_i, 2] - motion.terrain.hf[0, 0])

    cfg = ScalerConfig(
        model_height=1.7,
        human_height_assumption=1.65,
        root_joint="pelvis",
        joint_scales={"pelvis": 1.0},
    )
    ratio = uniform_overlay_scale_for_motion(cfg, 1.65, motion)
    pos = _uniform_scaled_joint_positions(
        motion, cfg, 1.65, list(motion.hierarchy.bone_names),
        ik_canons=frozenset(), z_correction=0.0,
    )
    hf_scaled = motion.terrain.scaled(ratio, z_offset=z_terrain)
    scaled_foot = float(pos[0, foot_i, 2])
    scaled_terrain = float(hf_scaled.hf[0, 0])
    scaled_gap = scaled_foot - scaled_terrain
    assert scaled_gap == pytest.approx(src_gap * ratio, abs=1e-5)
