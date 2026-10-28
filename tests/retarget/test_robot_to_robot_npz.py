"""IsaacLab NPZ compatibility across loading, upload detection and Agent inspection.

The conversion fixture follows phanghonghao's original PR #12 regression case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hhtools.contracts import AssetInspectionRequest, AssetKind, AssetRegistrationRequest
from hhtools.io.robot_trajectory_detect import is_robot_export_trajectory
from hhtools.retarget.robot_to_robot import load_source_trajectory
from hhtools.services.asset_service import AgentAssetService
from hhtools.services.assets import AssetRegistry
from hhtools.services.r2r_upload_resolve import enumerate_r2r_clips


@pytest.fixture
def training_arrays():
    return {
        "framerate": np.array(50.0),
        "joint_names": np.array(["left_hip_joint", "right_hip_joint"]),
        "joint_pos": np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float32),
        "base_pos_w": np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float32),
        "base_quat_w": np.array([[0.9, 0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0]]),
    }


def _inspect(tmp_path: Path, path: Path):
    service = AgentAssetService(AssetRegistry(tmp_path / "state", {"source": path.parent}))
    bundle = service.register(AssetRegistrationRequest(root_id="source", relative_path=path.name))
    assert bundle.kind is AssetKind.ROBOT_TRAJECTORY_BUNDLE
    return service.inspect(AssetInspectionRequest(asset_id=bundle.asset_id))


@pytest.mark.parametrize("name_dtype", ["U", "S"])
def test_training_npz_loads_and_registers_as_robot_trajectory(
    tmp_path: Path,
    training_arrays,
    name_dtype: str,
) -> None:
    training_arrays["joint_names"] = training_arrays["joint_names"].astype(name_dtype)
    path = tmp_path / "walk.npz"
    np.savez(path, **training_arrays)
    assert is_robot_export_trajectory(path)
    assert [clip.path for clip in enumerate_r2r_clips(tmp_path)] == [path]

    trajectory = load_source_trajectory(path)
    np.testing.assert_allclose(
        trajectory.joint_q,
        [
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, 0.1, -0.2],
            [1.5, 2.5, 3.5, 0.0, 0.0, 0.0, 1.0, 0.3, -0.4],
        ],
    )
    assert trajectory.dof_names == ("left_hip_joint", "right_hip_joint")
    assert trajectory.framerate == 50.0
    assert trajectory.meta["source_format"] == "isaaclab_training_convention_npz"

    inspection = _inspect(tmp_path, path)
    assert inspection.status.value == "valid"
    assert inspection.frame_count == 2
    assert inspection.joint_count == 2
    assert inspection.frame_rate_hz == 50.0
    assert inspection.metadata["dof_names"] == list(trajectory.dof_names)


@pytest.mark.parametrize("key", ["sample_rate", "fps", "framerate", None])
def test_training_npz_frame_rate_metadata_and_override(tmp_path, training_arrays, key):
    training_arrays.pop("framerate")
    if key is not None:
        training_arrays[key] = np.array(60.0)
    path = tmp_path / "timing.npz"
    np.savez(path, **training_arrays)
    assert load_source_trajectory(path, source_fps=30).framerate == (60 if key else 30)
    if key is None:
        assert load_source_trajectory(path).framerate == 50


@pytest.mark.parametrize("joint_key", ["joint_q", "qpos", "q"])
def test_native_npz_keeps_precedence_and_quaternion_convention(
    tmp_path,
    training_arrays,
    joint_key,
):
    original = np.array([[1, 2, 3, 0.9, 0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    path = tmp_path / "native.npz"
    np.savez(
        path,
        **training_arrays,
        **{
            joint_key: original,
            "dof_names": np.array(["native_joint"]),
            "root_quat_format": np.array("wxyz"),
            "sample_rate": np.array(75),
        },
    )
    result = load_source_trajectory(path)
    np.testing.assert_allclose(result.joint_q, [[1, 2, 3, 0.1, 0.2, 0.3, 0.9, 0.4]])
    assert result.dof_names == ("native_joint",)
    assert result.framerate == 75
    assert result.meta == {}
    inspection = _inspect(tmp_path, path)
    assert inspection.status.value == "valid"
    assert inspection.joint_count == 1
    assert inspection.metadata["dof_names"] == ["native_joint"]


@pytest.mark.parametrize("fps", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("metadata", [True, False], ids=["file", "override"])
def test_training_npz_rejects_nonfinite_timing(tmp_path, training_arrays, fps, metadata):
    training_arrays.pop("framerate")
    if metadata:
        training_arrays["framerate"] = np.array(fps)
    path = tmp_path / "invalid-timing.npz"
    np.savez(path, **training_arrays)
    with pytest.raises(ValueError, match="frame rate must be finite"):
        load_source_trajectory(path, source_fps=None if metadata else fps)


@pytest.mark.parametrize(
    ("key", "invalid"),
    [
        ("base_quat_w", [[1, 0, 0, 0]]),
        ("base_pos_w", [[1, 2], [3, 4]]),
        ("joint_pos", np.zeros((0, 2))),
        ("joint_pos", np.zeros((2, 0))),
        ("joint_pos", np.full((2, 2), np.nan)),
        ("base_pos_w", np.full((2, 3), np.inf)),
        ("base_quat_w", np.zeros((2, 4))),
        ("joint_names", ["left", "left"]),
        ("joint_names", ["left", " "]),
        ("joint_names", [["left", "right"]]),
        ("joint_names", [1, 2]),
        ("joint_names", np.array(["left", "right"], dtype=object)),
    ],
)
def test_invalid_training_arrays_are_rejected_across_entry_points(
    tmp_path,
    training_arrays,
    key,
    invalid,
):
    training_arrays[key] = np.asarray(invalid)
    path = tmp_path / "invalid.npz"
    np.savez(path, **training_arrays)
    assert not is_robot_export_trajectory(path)
    with pytest.raises(ValueError):
        load_source_trajectory(path)
    inspection = _inspect(tmp_path, path)
    assert inspection.status.value == "invalid"
    assert any(error.code == "ROBOT_TRAJECTORY_PARSE_FAILED" for error in inspection.errors)


def test_training_arrays_preserve_inspection_resource_limits(
    tmp_path,
    training_arrays,
    monkeypatch,
):
    from hhtools.services import r2r_asset_inspection

    path = tmp_path / "bounded.npz"
    np.savez(path, **training_arrays)
    monkeypatch.setattr(r2r_asset_inspection, "_MAX_TRAJECTORY_ARRAY_BYTES", 16)
    inspection = _inspect(tmp_path, path)
    assert inspection.status.value == "invalid"
    assert any(error.code == "ROBOT_TRAJECTORY_PARSE_FAILED" for error in inspection.errors)
