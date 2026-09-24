from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hhtools.web.server.routes.h2r import register_h2r_routes


class _Jobs:
    def schedule(self, *_args, **_kwargs):
        raise AssertionError("calibration proposal must not schedule a job")


def test_web_calibration_proposal_is_editable_and_never_saves(monkeypatch) -> None:
    loaded = SimpleNamespace(preset=SimpleNamespace(name="test_robot"))
    fresh = object()
    state = SimpleNamespace(robots={"test_robot": loaded}, motions={})
    observed = {}

    monkeypatch.setattr("hhtools.robot.loader.load_robot", lambda *_args, **_kwargs: fresh)

    def propose(model, reference, seed, *, locked_joints, reference_motion):
        observed.update(
            {
                "model": model,
                "reference": reference,
                "seed": seed,
                "locked_joints": locked_joints,
                "reference_motion": reference_motion,
            }
        )
        return (
            {"left_shoulder_roll_joint": 1.2},
            SimpleNamespace(
                valid=True,
                score=0.94,
                changed_joint_count=1,
                edge_errors_deg={"left_upper_arm": 5.0},
                near_limit_joints=(),
                alignment_errors=(),
                alignment_warnings=(),
                foot_height_delta_m=0.0,
            ),
        )

    monkeypatch.setattr(
        "hhtools.retarget.calibration.assistant.propose_calibration_pose",
        propose,
    )
    app = FastAPI()
    register_h2r_routes(app, state=state, jobs=_Jobs())

    response = TestClient(app).post(
        "/api/calibration/propose",
        json={
            "robot": "test_robot",
            "reference": "smplx",
            "joint_q": {"left_shoulder_roll_joint": 0.0},
            "locked_joints": ["waist_joint"],
        },
    )

    assert response.status_code == 200
    assert response.json()["joint_q"] == {"left_shoulder_roll_joint": 1.2}
    assert response.json()["validation"]["valid"] is True
    assert observed == {
        "model": fresh,
        "reference": "smplx",
        "seed": {"left_shoulder_roll_joint": 0.0},
        "locked_joints": frozenset({"waist_joint"}),
        "reference_motion": None,
    }
