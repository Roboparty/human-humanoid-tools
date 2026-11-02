# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""The yellow overlay follows the Newton IK root instead of the world origin."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hhtools.application.previews import _anchor_scaled_preview_to_ik_root
from hhtools.retarget.retarget_result import IKTargetTrace


def _overlay() -> dict:
    # Uniform 0.5x overlay of a clip that starts 4 m from the origin, so the
    # hips sit at (2, 0) instead of the IK root target (4, 0).
    frame = [[2.0, 0.0, 0.45], [1.9, 0.1, 0.05], [2.1, -0.1, 0.05]]
    return {
        "bone_names": ["Hips", "LeftFoot", "RightFoot"],
        "frame_indices": [0, 2],
        "positions": [frame, frame],
    }


def _result(hips_xy_per_frame) -> SimpleNamespace:
    frames = len(hips_xy_per_frame)
    positions = np.zeros((frames, 2, 3), dtype=np.float32)
    positions[:, 0, :2] = hips_xy_per_frame
    positions[:, 0, 2] = 0.4
    positions[:, 1] = np.nan  # an unrelated column must not matter
    return SimpleNamespace(
        ik_targets=IKTargetTrace(
            canonical_names=("hips", "left_wrist"),
            t_body_links=("pelvis", "left_hand"),
            link_offsets=np.zeros((2, 3), dtype=np.float32),
            positions=positions,
        )
    )


def _motion(*, terrain=None) -> SimpleNamespace:
    return SimpleNamespace(terrain=terrain, objects=[])


def test_overlay_hips_land_on_the_ik_root_and_keep_their_pose() -> None:
    result = _result([[4.0, 0.0], [4.1, 0.0], [4.2, 0.3]])

    anchored = _anchor_scaled_preview_to_ik_root(_overlay(), result, _motion())

    positions = np.asarray(anchored["positions"])
    np.testing.assert_allclose(positions[0, 0, :2], [4.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(positions[1, 0, :2], [4.2, 0.3], atol=1e-4)
    # Pure XY translation: foot offsets and every Z value are unchanged.
    original = np.asarray(_overlay()["positions"])
    np.testing.assert_allclose(
        positions - positions[:, :1], original - original[:, :1], atol=1e-4
    )
    np.testing.assert_allclose(positions[..., 2], original[..., 2], atol=1e-4)


@pytest.mark.parametrize(
    "result",
    [SimpleNamespace(ik_targets=None), SimpleNamespace()],
)
def test_results_without_ik_trace_keep_the_origin_anchored_overlay(result) -> None:
    overlay = _overlay()

    assert _anchor_scaled_preview_to_ik_root(overlay, result, _motion()) is overlay


def test_interaction_scene_clips_stay_co_aligned_with_the_scaled_scene() -> None:
    overlay = _overlay()
    result = _result([[4.0, 0.0], [4.1, 0.0], [4.2, 0.3]])

    assert _anchor_scaled_preview_to_ik_root(overlay, result, _motion(terrain=object())) is overlay
