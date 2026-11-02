"""Result diagnostics compare the exact payloads shown by the Web renderer."""

from __future__ import annotations

import numpy as np
import pytest

from hhtools.retarget.retarget_result import IKTargetTrace
from hhtools.web.analysis.result_diagnostics import (
    build_result_diagnostics,
    tracking_quality_band,
)


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


def _rotation_z(transform: list[float], yaw_deg: float) -> list[float]:
    """Return ``transform`` with its rotation block set to a yaw about +Z."""
    c = float(np.cos(np.radians(yaw_deg)))
    s = float(np.sin(np.radians(yaw_deg)))
    out = list(transform)
    out[0], out[1], out[4], out[5] = c, -s, s, c
    return out


def _trace(names, links, positions, offsets=None) -> IKTargetTrace:
    return IKTargetTrace(
        canonical_names=tuple(names),
        t_body_links=tuple(links),
        link_offsets=np.zeros((len(names), 3), dtype=np.float32)
        if offsets is None
        else np.asarray(offsets, dtype=np.float32),
        positions=np.asarray(positions, dtype=np.float32),
    )


def _exact_trace(trajectory, ik_map) -> IKTargetTrace:
    """IK targets that the payload robot links reach exactly."""
    names = list(ik_map)
    links = [ik_map[name] for name in names]
    positions = [
        [frame["links"][link][3::4][:3] for link in links] for frame in trajectory["frames"]
    ]
    return _trace(names, links, positions)


def test_ik_target_trace_is_preferred_over_the_uniform_scaled_overlay() -> None:
    # The yellow overlay is a uniform rescale of the source skeleton; on robots
    # whose proportions differ it sits far from the per-joint IK targets even
    # when the robot tracks those targets perfectly.
    trajectory, scaled, ik_map, feet = _payloads(x_offset=0.5)
    trace = _trace(
        list(ik_map),
        [ik_map[name] for name in ik_map],
        [
            [[0.5, 0.0, 1.0], [0.4, 0.0, 0.0], [0.6, 0.0, 0.0]]
            for _ in range(4)
        ],
    )

    diagnostics = build_result_diagnostics(
        trajectory,
        scaled,
        ik_map=ik_map,
        feet=feet,
        ik_targets=trace,
    )

    assert diagnostics["available"] is True
    assert diagnostics["tracking"]["reference"] == "ik_targets"
    assert diagnostics["tracking"]["max_error_m"] == pytest.approx(0.0, abs=1e-6)
    assert diagnostics["contact"]["available"] is True


def test_ik_target_trace_uses_link_offsets_and_ignores_display_lift() -> None:
    trajectory = {
        "frames": [
            {
                # 180 degree yaw root, display-only grounding lift.
                "root": [1.0, 2.0, 0.1, 0.0, 0.0, 1.0, 0.0],
                "mesh_z_lift": 0.05,
                # Link yawed +90 degrees inside the root frame at local (0.2, 0, 0).
                "links": {"hand": _rotation_z(_transform(0.2, 0.0, 0.0), 90.0)},
            }
        ],
        "framerate": 30.0,
    }
    # Offset +X in the link frame -> +Y in root frame -> -Y in world.
    trace = _trace(
        ["left_wrist"],
        ["hand"],
        [[[0.8, 1.9, 0.1]]],
        offsets=[[0.1, 0.0, 0.0]],
    )

    diagnostics = build_result_diagnostics(
        trajectory,
        None,
        ik_map={"left_wrist": "hand"},
        ik_targets=trace,
    )

    assert diagnostics["available"] is True
    assert diagnostics["tracking"]["max_error_m"] == pytest.approx(0.0, abs=1e-6)
    assert diagnostics["contact"]["available"] is False


def test_ik_target_trace_reports_only_robot_ik_map_slots() -> None:
    trajectory, scaled, ik_map, feet = _payloads()
    exact = _exact_trace(trajectory, ik_map)
    # Auto-added endpoint objective (e.g. a toe tip) far from its link.
    trace = _trace(
        [*exact.canonical_names, "left_toe_tip"],
        [*exact.t_body_links, "left_foot"],
        np.concatenate(
            [exact.positions, np.full((4, 1, 3), 5.0, dtype=np.float32)],
            axis=1,
        ),
    )

    diagnostics = build_result_diagnostics(
        trajectory,
        scaled,
        ik_map=ik_map,
        feet=feet,
        ik_targets=trace,
    )

    assert diagnostics["requested_effectors"] == 3
    assert {item["canonical"] for item in diagnostics["tracking"]["effectors"]} == set(ik_map)
    assert diagnostics["tracking"]["max_error_m"] == pytest.approx(0.0, abs=1e-6)


def test_malformed_ik_target_trace_falls_back_to_scaled_overlay() -> None:
    trajectory, scaled, ik_map, feet = _payloads(x_offset=0.1)
    trace = _trace(list(ik_map), [ik_map[name] for name in ik_map], np.zeros((4, 2, 3)))

    diagnostics = build_result_diagnostics(
        trajectory,
        scaled,
        ik_map=ik_map,
        feet=feet,
        ik_targets=trace,
    )

    assert diagnostics["tracking"]["reference"] == "scaled_preview"
    assert diagnostics["tracking"]["mean_error_m"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("robot_height_m", "p95", "band", "stable_max_m"),
    [
        (1.3, 0.052, "stable", 0.052),  # ~historical 5 cm on a G1 / RP1-sized robot
        (1.8, 0.09, "review", 0.072),  # tall robot: 9 cm is no longer "large"
        (0.8, 0.07, "high_error", 0.032),  # small robot: 7 cm is a large share
        (None, 0.07, "review", 0.05),  # unknown stature keeps absolute bands
    ],
)
def test_quality_band_scales_with_robot_height(robot_height_m, p95, band, stable_max_m) -> None:
    quality = tracking_quality_band(p95, robot_height_m)

    assert quality["band"] == band
    assert quality["stable_max_m"] == pytest.approx(stable_max_m)
    assert quality["robot_height_m"] == (
        pytest.approx(robot_height_m) if robot_height_m else None
    )


def test_diagnostics_report_quality_band_from_scaled_preview_robot_height() -> None:
    trajectory, scaled, ik_map, feet = _payloads(x_offset=0.09)
    scaled["robot_height_m"] = 1.8

    diagnostics = build_result_diagnostics(trajectory, scaled, ik_map=ik_map, feet=feet)

    assert diagnostics["quality"]["band"] == "review"
    assert diagnostics["quality"]["review_max_m"] == pytest.approx(0.144)
    assert diagnostics["quality"]["p95_height_ratio"] == pytest.approx(0.05)
