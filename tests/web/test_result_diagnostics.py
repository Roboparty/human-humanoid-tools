"""Result diagnostics compare the exact payloads shown by the Web renderer."""

from __future__ import annotations

import pytest

from hhtools.web.analysis.result_diagnostics import build_result_diagnostics


def _transform(x: float, y: float, z: float) -> list[float]:
    return [
        1.0,
        0.0,
        0.0,
        x,
        0.0,
        1.0,
        0.0,
        y,
        0.0,
        0.0,
        1.0,
        z,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _payloads(*, x_offset: float = 0.0, slide_per_frame: float = 0.0):
    names = ["hips", "left_ankle", "right_ankle"]
    source_frame = [[0.0, 0.0, 1.0], [-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]
    scaled = {
        "bone_names": names,
        "frame_indices": [0, 1, 2, 3],
        "positions": [source_frame for _ in range(4)],
        "framerate": 10.0,
    }
    frames = []
    for index in range(4):
        dx = x_offset + slide_per_frame * index
        frames.append(
            {
                "root": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "mesh_z_lift": 0.0,
                "links": {
                    "pelvis": _transform(dx, 0.0, 1.0),
                    "left_foot": _transform(-0.1 + dx, 0.0, 0.0),
                    "right_foot": _transform(0.1 + dx, 0.0, 0.0),
                },
            }
        )
    trajectory = {
        "frame_indices": [0, 1, 2, 3],
        "frames": frames,
        "framerate": 10.0,
    }
    ik_map = {
        "hips": "pelvis",
        "left_ankle": "left_foot",
        "right_ankle": "right_foot",
    }
    feet = {
        "left_contact_link": "left_foot",
        "right_contact_link": "right_foot",
    }
    return trajectory, scaled, ik_map, feet


def test_exact_result_has_zero_tracking_error_and_matching_contacts() -> None:
    trajectory, scaled, ik_map, feet = _payloads()

    diagnostics = build_result_diagnostics(
        trajectory,
        scaled,
        ik_map=ik_map,
        feet=feet,
    )

    assert diagnostics["available"] is True
    assert diagnostics["mapped_effectors"] == 3
    assert diagnostics["tracking"]["mean_error_m"] == pytest.approx(0.0)
    assert diagnostics["tracking"]["p95_error_m"] == pytest.approx(0.0)
    assert diagnostics["contact"]["agreement_ratio"] == pytest.approx(1.0)
    assert diagnostics["contact"]["recall_ratio"] == pytest.approx(1.0)


def test_xsens_source_names_match_canonical_robot_mappings() -> None:
    trajectory, scaled, ik_map, feet = _payloads()
    scaled["bone_names"] = [
        "Hips",
        "Chest",
        "LeftHip",
        "LeftKnee",
        "LeftAnkle",
        "RightHip",
        "RightKnee",
        "RightAnkle",
    ]
    xsens_frame = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.2],
        [-0.1, 0.0, 0.8],
        [-0.1, 0.0, 0.4],
        [-0.1, 0.0, 0.0],
        [0.1, 0.0, 0.8],
        [0.1, 0.0, 0.4],
        [0.1, 0.0, 0.0],
    ]
    scaled["positions"] = [xsens_frame for _ in range(4)]

    diagnostics = build_result_diagnostics(
        trajectory,
        scaled,
        ik_map=ik_map,
        feet=feet,
    )

    assert diagnostics["available"] is True
    assert diagnostics["mapped_effectors"] == 3
    assert diagnostics["tracking"]["mean_error_m"] == pytest.approx(0.0)
    assert diagnostics["contact"]["available"] is True
    assert diagnostics["contact"]["agreement_ratio"] == pytest.approx(1.0)


def test_tracking_error_uses_robot_root_rotation_translation_and_mesh_lift() -> None:
    trajectory = {
        "frames": [
            {
                # A 180 degree Z rotation maps local +X to world -X.
                "root": [1.0, 2.0, 0.1, 0.0, 0.0, 1.0, 0.0],
                "mesh_z_lift": 0.05,
                "links": {"hand": _transform(0.2, 0.0, 0.0)},
            }
        ],
        "framerate": 30.0,
    }
    scaled = {
        "bone_names": ["left_wrist"],
        "positions": [[[0.8, 2.0, 0.15]]],
        "framerate": 30.0,
    }

    diagnostics = build_result_diagnostics(
        trajectory,
        scaled,
        ik_map={"left_wrist": "hand"},
    )

    assert diagnostics["tracking"]["max_error_m"] == pytest.approx(0.0)


def test_known_offset_is_reported_in_metres() -> None:
    trajectory, scaled, ik_map, feet = _payloads(x_offset=0.1)

    diagnostics = build_result_diagnostics(
        trajectory,
        scaled,
        ik_map=ik_map,
        feet=feet,
    )

    assert diagnostics["tracking"]["mean_error_m"] == pytest.approx(0.1)
    assert diagnostics["tracking"]["max_error_m"] == pytest.approx(0.1)


def test_diagnostics_do_not_pair_different_source_frame_indices() -> None:
    trajectory, scaled, ik_map, feet = _payloads()
    trajectory["frame_indices"] = [10, 11, 12, 13]

    diagnostics = build_result_diagnostics(
        trajectory,
        scaled,
        ik_map=ik_map,
        feet=feet,
    )

    assert diagnostics["available"] is False
    assert diagnostics["reason"] == "no mapped target frames are available for comparison"


def test_contact_diagnostics_report_result_foot_sliding() -> None:
    trajectory, scaled, ik_map, feet = _payloads(slide_per_frame=0.01)

    diagnostics = build_result_diagnostics(
        trajectory,
        scaled,
        ik_map=ik_map,
        feet=feet,
    )

    assert diagnostics["contact"]["available"] is True
    assert diagnostics["contact"]["agreement_ratio"] == pytest.approx(1.0)
    assert diagnostics["contact"]["target_slide_mean_mps"] == pytest.approx(0.1)


def test_missing_scaled_targets_return_an_explicit_unavailable_reason() -> None:
    diagnostics = build_result_diagnostics(
        {"frames": []},
        None,
        ik_map={},
    )

    assert diagnostics["available"] is False
    assert "scaled target" in diagnostics["reason"]
