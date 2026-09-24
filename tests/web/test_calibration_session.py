from types import SimpleNamespace

import numpy as np

from hhtools.web.analysis.calibration_session import serialize_reference_skeleton


def _reference_pose():
    return SimpleNamespace(
        joint_names=("Hips", "LeftFoot", "RightFoot"),
        parent_names=(None, "Hips", "Hips"),
        positions=np.array(
            [
                [0.0, 0.0, 1.0],
                [0.25, 0.0, 0.2],
                [-0.25, 0.0, 0.2],
            ],
            dtype=np.float32,
        ),
        quaternions=np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        source_to_canonical={
            "Hips": "hips",
            "LeftFoot": "left_ankle",
            "RightFoot": "right_ankle",
        },
    )


def test_serialize_reference_skeleton_includes_semantics_and_heading() -> None:
    payload = serialize_reference_skeleton(_reference_pose(), heading_rad=np.pi / 2)

    assert payload["bone_names"] == ["Hips", "LeftFoot", "RightFoot"]
    assert payload["canonical_names"] == ["hips", "left_ankle", "right_ankle"]
    assert payload["parent_indices"] == [-1, 0, 0]

    positions = np.asarray(payload["positions"][0])
    quaternions = np.asarray(payload["quaternions"][0])
    np.testing.assert_allclose(positions[:, 2].min(), 0.0, atol=1e-6)
    np.testing.assert_allclose(positions[1, :2], [0.0, 0.25], atol=1e-6)
    np.testing.assert_allclose(quaternions[0], [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)


def test_serialize_reference_skeleton_rejects_mismatched_quaternions() -> None:
    reference = _reference_pose()
    reference.quaternions = reference.quaternions[:2]

    try:
        serialize_reference_skeleton(reference)
    except ValueError as error:
        assert "reference quaternions" in str(error)
    else:
        raise AssertionError("mismatched reference quaternions should fail")
