from __future__ import annotations

import numpy as np
import pytest

from hhtools.bodymodels.fallback import forward_without_weights, motion_from_fallback
from hhtools.bodymodels.layout import (
    SMPL_LAYOUT,
    SMPLH_LAYOUT,
    SMPLX_LAYOUT,
)
from hhtools.bodymodels.params import SmplMotionParams
from hhtools.io.datasets import _engine_cache
from hhtools.web.output.serialize import skeleton_exclude_joint_indices


def _params(family: str, *, up_axis: str = "Z") -> SmplMotionParams:
    body_width = 69 if family == "smpl" else 63
    return SmplMotionParams(
        surface_model=family,  # type: ignore[arg-type]
        root_orient=np.zeros((2, 3), dtype=np.float32),
        body_pose=np.zeros((2, body_width), dtype=np.float32),
        betas=np.zeros(10, dtype=np.float32),
        trans=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]], dtype=np.float32),
        up_axis=up_axis,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("family", "layout"),
    [("smpl", SMPL_LAYOUT), ("smplh", SMPLH_LAYOUT), ("smplx", SMPLX_LAYOUT)],
)
def test_weight_free_forward_preserves_layout_and_root_translation(family, layout) -> None:
    result = forward_without_weights(_params(family))

    assert result.joints.shape == (2, layout.num_joints, 3)
    assert result.quaternions_global.shape == (2, layout.num_joints, 4)
    assert np.isfinite(result.joints).all()
    assert np.isfinite(result.quaternions_global).all()
    np.testing.assert_allclose(result.joints[:, 0], [[0, 0, 0], [0.1, 0.2, 0.3]])
    np.testing.assert_allclose(
        result.quaternions_global[0],
        np.tile([0, 0, 0, 1], (layout.num_joints, 1)),
    )


def test_weight_free_forward_applies_local_pose_rotation() -> None:
    params = _params("smpl")
    params.body_pose[1, 2] = np.pi / 2.0

    result = forward_without_weights(params)

    assert not np.allclose(result.joints[0, 4], result.joints[1, 4])


def test_weight_free_forward_keeps_y_up_coordinates_for_later_grounding() -> None:
    result = forward_without_weights(_params("smplx", up_axis="Y"))

    # The proxy's feet are below the pelvis on the source Y axis. The web
    # grounding step then performs the existing Y -> Z conversion.
    assert result.joints[0, 7, 1] < result.joints[0, 0, 1]
    # SMPL's native model frame uses +X for the subject's left side.
    assert result.joints[0, 1, 0] > result.joints[0, 0, 0]
    assert result.joints[0, 2, 0] < result.joints[0, 0, 0]


def test_weight_free_forward_applies_world_up_only_through_root_orientation() -> None:
    params = _params("smplx", up_axis="Z")
    params.root_orient[:] = [np.pi / 2.0, 0.0, 0.0]

    result = forward_without_weights(params)

    pelvis = result.joints[0, 0]
    assert result.joints[0, 15, 2] > pelvis[2]
    assert result.joints[0, 7, 2] < pelvis[2]
    assert abs(float(result.joints[0, 15, 1] - pelvis[1])) < 1e-5


def test_web_visualization_omits_dense_smplx_hand_and_face_joints() -> None:
    motion = motion_from_fallback(_params("smplx"), name="preview")

    excluded = set(skeleton_exclude_joint_indices(motion))

    assert set(range(22, 55)) <= excluded
    assert 20 not in excluded
    assert 21 not in excluded


def test_dataset_engine_boundary_uses_proxy_when_weights_are_missing(monkeypatch) -> None:
    def missing(_params):
        raise FileNotFoundError("weights missing")

    monkeypatch.setattr(_engine_cache, "engine_for_params", missing)
    motion = _engine_cache.motion_from_params(
        _params("smplx"),
        name="preview",
        source_format="motion_x/smplx",
        return_mesh=True,
    )

    assert motion.meta["body_model_fallback"] is True
    assert motion.meta["baked_mesh_unavailable"] is True
    assert motion.positions.shape == (2, SMPLX_LAYOUT.num_joints, 3)
