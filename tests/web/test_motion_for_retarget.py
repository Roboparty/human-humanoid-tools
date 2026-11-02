# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""``limit_frames`` must never truncate the session-cached source clip."""

from __future__ import annotations

import numpy as np

from hhtools.application.motions import _apply_limit_frames, _motion_for_retarget
from hhtools.core.hierarchy import Hierarchy
from hhtools.core.motion import Motion
from hhtools.core.scene import SceneObject


def _cached_clip(frames: int = 10) -> Motion:
    hierarchy = Hierarchy(
        bone_names=["hips", "left_ankle"],
        parent_indices=[-1, 0],
        parent_names=[None, "hips"],
    )
    quats = np.zeros((frames, 2, 4), dtype=np.float32)
    quats[..., 3] = 1.0
    box = SceneObject(
        name="box",
        positions=np.zeros((frames, 3), dtype=np.float32),
        quaternions=np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (frames, 1)),
    )
    return Motion(
        name="clip",
        hierarchy=hierarchy,
        positions=np.zeros((frames, 2, 3), dtype=np.float32),
        quaternions=quats,
        framerate=30.0,
        meta={"dataset": "lafan"},
        objects=[box],
    )


def test_unchanged_rate_returns_a_detached_copy() -> None:
    cached = _cached_clip()

    solver_motion, fps = _motion_for_retarget(cached, None)

    assert fps == 30.0
    assert solver_motion is not cached
    assert solver_motion.objects[0] is not cached.objects[0]
    assert solver_motion.meta == cached.meta
    assert solver_motion.meta is not cached.meta


def test_limit_frames_on_solver_motion_keeps_cached_clip_intact() -> None:
    cached = _cached_clip(frames=10)

    solver_motion, _ = _motion_for_retarget(cached, 30.0)
    truncated = _apply_limit_frames(solver_motion, 4)

    assert truncated.num_frames == 4
    assert truncated.objects[0].positions.shape[0] == 4
    assert cached.num_frames == 10
    assert cached.objects[0].positions.shape[0] == 10
