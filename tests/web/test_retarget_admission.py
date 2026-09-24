from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hhtools.web.server import create_app
from hhtools.web.server.routes.h2r import register_h2r_routes
from hhtools.web.server.routes.r2r import register_r2r_routes


class _RecordingJobs:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def schedule(self, kind, request, target, *, args=(), **kwargs):
        self.calls.append((kind, request))
        return SimpleNamespace(id="queued-job")

    def reserve_slot(self):
        raise AssertionError("the retarget entry point must not reserve an upload slot")


class _UnusedUploads:
    async def store(self, *args, **kwargs):
        raise AssertionError("the retarget entry point must not store uploads")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH", str(tmp_path / "library.json"))
    monkeypatch.setenv("HHTOOLS_MOTION_LIBRARY_ROOT", str(tmp_path / "library"))
    app = create_app(
        source_root=tmp_path / "source", save_dir=tmp_path / "save",
        cache_dir=tmp_path / "cache", job_history_dir=tmp_path / "history",
    )
    with TestClient(app) as browser:
        yield browser


@pytest.mark.parametrize("route", ["/api/retarget", "/api/r2r/retarget"])
@pytest.mark.parametrize("payload", [{}, {"backend": "unknown"}, {"ik_iterations": -1}])
def test_invalid_retarget_request_never_creates_job(client, route, payload):
    response = client.post(route, json=payload)
    assert response.status_code == 422
    assert not client.app.state.session_state.jobs


@pytest.mark.parametrize(("route", "payload"), [
    ("/api/retarget", {"robot": "robot", "motion_token": "expired"}),
    ("/api/r2r/retarget", {"source": "source", "target": "target", "source_token": "expired"}),
])
def test_missing_motion_is_rejected_before_admission(client, route, payload):
    assert client.post(route, json=payload).status_code == 404
    assert not client.app.state.session_state.jobs


def test_r2r_source_identity_is_rejected_before_admission(client):
    state = client.app.state.session_state
    state.r2r_sources["loaded"] = {"source_robot": "actual-source"}
    response = client.post(
        "/api/r2r/retarget",
        json={"source": "other-source", "target": "target", "source_token": "loaded"},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "source robot does not match the trajectory"
    assert not state.jobs


def test_uncalibrated_human_retarget_is_rejected(client, monkeypatch):
    state = client.app.state.session_state
    state.motions["loaded"] = {"motion": object()}
    state.robots["robot"] = SimpleNamespace(preset=object())
    monkeypatch.setattr(
        "hhtools.retarget.calibration.resolve_preset_calibration_file",
        lambda *args: None,
    )
    monkeypatch.setattr("hhtools.robot.retarget_profile.bundled_scaler_path", lambda *args: None)
    response = client.post("/api/retarget", json={"robot": "robot", "motion_token": "loaded"})
    assert response.status_code == 409
    assert not state.jobs


def test_valid_human_retarget_is_scheduled_after_preflight(tmp_path: Path, monkeypatch) -> None:
    state = SimpleNamespace(
        motions={"loaded": {"motion": object()}},
        robots={"robot": SimpleNamespace(preset=SimpleNamespace())},
    )
    jobs = _RecordingJobs()
    app = FastAPI()
    register_h2r_routes(app, state=state, jobs=jobs)
    monkeypatch.setattr(
        "hhtools.retarget.calibration.resolve_preset_calibration_file",
        lambda *args: tmp_path / "calibration.yaml",
    )

    with TestClient(app) as browser:
        response = browser.post(
            "/api/retarget",
            json={"robot": "robot", "motion_token": "loaded"},
        )

    assert response.status_code == 200
    assert response.json() == {"job_id": "queued-job"}
    assert jobs.calls == [
        (
            "retarget",
            {
                "backend": "newton",
                "ik_iterations": 24,
                "retarget_fps": None,
                "robot": "robot",
                "motion_token": "loaded",
                "reference": "smpl",
                "human_height": None,
                "limit_frames": None,
                "foot_clamp_anti_penetration": False,
            },
        )
    ]


def test_valid_robot_retarget_is_scheduled_after_preflight(tmp_path: Path, monkeypatch) -> None:
    target = SimpleNamespace(
        preset=SimpleNamespace(name="target", urdf_path=tmp_path / "target" / "robot.urdf")
    )
    state = SimpleNamespace(
        r2r_sources={"loaded": {"source_robot": "source"}},
        robots={"source": SimpleNamespace(), "target": target},
    )
    jobs = _RecordingJobs()
    app = FastAPI()
    register_r2r_routes(app, state=state, jobs=jobs, uploads=_UnusedUploads())
    monkeypatch.setattr(
        "hhtools.retarget.robot_to_robot.load_r2r_calibration",
        lambda *args, **kwargs: {"hip": 0.0},
    )

    with TestClient(app) as browser:
        response = browser.post(
            "/api/r2r/retarget",
            json={"source": "source", "target": "target", "source_token": "loaded"},
        )

    assert response.status_code == 200
    assert response.json() == {"job_id": "queued-job"}
    assert jobs.calls == [
        (
            "r2r_retarget",
            {
                "backend": "newton",
                "ik_iterations": 24,
                "retarget_fps": None,
                "target": "target",
                "source": "source",
                "source_token": "loaded",
            },
        )
    ]


@pytest.mark.parametrize(("headers", "status"), [
    ({"Origin": "https://audit.invalid"}, 403),
    ({"Origin": "null"}, 403),
    ({"Host": "audit.invalid"}, 403),
    ({"Origin": "http://127.0.0.1"}, 200),
    ({}, 200),
])
def test_web_settings_use_local_origin_and_host_boundary(client, headers, status):
    response = client.patch(
        "/api/settings/job-admission", headers=headers,
        json={"max_running_jobs": 1, "max_queued_jobs": 2},
    )
    assert response.status_code == status


@pytest.mark.parametrize("route", ["/api/retarget", "/api/r2r/retarget"])
def test_cross_origin_retarget_is_rejected_before_validation(client, route):
    response = client.post(route, headers={"Origin": "https://audit.invalid"}, json={})
    assert response.status_code == 403
    assert not client.app.state.session_state.jobs
