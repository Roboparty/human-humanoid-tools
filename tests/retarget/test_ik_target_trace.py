# SPDX-FileCopyrightText: Copyright (c) 2026 hhtools contributors
# SPDX-License-Identifier: Apache-2.0
"""Newton results record IK targets in the same frame as the output ``joint_q``."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("newton")

from hhtools.retarget.newton_basic.pipeline import NewtonBasicPipeline  # noqa: E402
from hhtools.retarget.newton_basic.robot_model import IKMapping, IKMappingEntry  # noqa: E402


def _entry(canonical: str, link: str, offset=(0.0, 0.0, 0.0)) -> IKMappingEntry:
    return IKMappingEntry(
        canonical_name=canonical,
        t_body_link=link,
        r_body_link=link,
        t_body_index=0,
        r_body_index=0,
        t_weight=1.0,
        r_weight=1.0,
        t_offset=offset,
    )


def _pipeline_stub(source_body_quat) -> SimpleNamespace:
    stub = SimpleNamespace(
        ik_mapping=IKMapping(
            entries=(
                _entry("hips", "pelvis"),
                _entry("left_toe", "left_ankle_roll_link", (0.12, 0.0, -0.03)),
            )
        ),
        scaler_config=SimpleNamespace(source_body_quat=source_body_quat),
    )
    stub._align_preview_to_source_heading = (
        lambda transforms: NewtonBasicPipeline._align_preview_to_source_heading(stub, transforms)
    )
    return stub


def test_trace_undoes_source_heading_like_the_output_root() -> None:
    half = np.sqrt(0.5)
    stub = _pipeline_stub((0.0, 0.0, half, half))  # +90 degree yaw pre-rotation
    solver_frame = np.array(
        [[[1.0, 0.0, 0.9], [2.0, 0.5, 0.05]]],
        dtype=np.float32,
    )

    trace = NewtonBasicPipeline._ik_target_trace(stub, solver_frame)

    # conj(+90 deg yaw) maps solver +X to source -Y; Z is untouched.
    np.testing.assert_allclose(
        trace.positions,
        [[[0.0, -1.0, 0.9], [0.5, -2.0, 0.05]]],
        atol=1e-6,
    )
    assert trace.canonical_names == ("hips", "left_toe")
    assert trace.t_body_links == ("pelvis", "left_ankle_roll_link")
    np.testing.assert_allclose(trace.link_offsets, [[0.0, 0.0, 0.0], [0.12, 0.0, -0.03]])


def test_trace_is_unchanged_for_identity_heading() -> None:
    stub = _pipeline_stub((0.0, 0.0, 0.0, 1.0))
    solver_frame = np.arange(12, dtype=np.float32).reshape(2, 2, 3)

    trace = NewtonBasicPipeline._ik_target_trace(stub, solver_frame)

    np.testing.assert_allclose(trace.positions, solver_frame)
