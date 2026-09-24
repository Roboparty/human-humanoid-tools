from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from hhtools.integrations import gvhmr
from hhtools.web import server
from hhtools.web.server import state as server_state


def _create_test_app(tmp_path: Path, monkeypatch):
    def local_tmpdir(tag: str) -> Path:
        path = tmp_path / f"runtime-{tag}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(server_state, "_tmpdir", local_tmpdir)
    monkeypatch.setattr(server_state, "_robot_library_root", lambda: tmp_path / "robots")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return server.create_app(
        source_root=tmp_path / "motions",
        save_dir=tmp_path / "save",
        cache_dir=tmp_path / "cache",
    )


def _status(*, ready: bool) -> dict:
    missing = [] if ready else ["licensed SMPL-X neutral model"]
    return {
        "ready": ready,
        "checks": {"smplx_neutral": ready},
        "missing": missing,
        "root": "C:/GVHMR",
        "body_models_root": "C:/GVHMR/inputs/checkpoints/body_models",
        "image": "hhtools-gvhmr:cu128",
        "runtime": "local",
        "uses_official_weights": True,
        "supports_custom_weights": False,
        "training_enabled": False,
    }


def test_video_to_motion_status_is_exposed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gvhmr, "gvhmr_status", lambda: _status(ready=False))
    app = _create_test_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.get("/api/video-to-motion/status")

    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["training_enabled"] is False


def test_video_upload_fails_before_storage_when_runtime_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(gvhmr, "gvhmr_status", lambda: _status(ready=False))
    app = _create_test_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/video-to-motion/upload",
            files=[("files", ("clip.mp4", b"video", "video/mp4"))],
        )

    assert response.status_code == 503
    assert "SMPL-X" in response.json()["detail"]
    assert not list(app.state.session_state.upload_root.rglob("clip.mp4"))


def test_video_upload_rejects_non_video_extension(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gvhmr, "gvhmr_status", lambda: _status(ready=True))
    app = _create_test_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/video-to-motion/upload",
            files=[("files", ("clip.txt", b"not-video", "text/plain"))],
        )

    assert response.status_code == 400
    assert not list(app.state.session_state.upload_root.rglob("clip.txt"))


def test_video_upload_records_official_weights_for_the_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(gvhmr, "gvhmr_status", lambda: _status(ready=True))
    monkeypatch.setattr(gvhmr, "run_gvhmr", lambda *_args, **_kwargs: None)
    app = _create_test_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/video-to-motion/upload",
            files=[("files", ("clip.mp4", b"video", "video/mp4"))],
        )

        assert response.status_code == 200
        job = app.state.session_state.jobs[response.json()["job_id"]]
        assert job.request["weights"] == "official"
        assert "checkpoint_name" not in job.request


def test_completed_video_publishes_only_the_final_motion(
    tmp_path: Path,
    monkeypatch,
    synthetic_terrain_motion,
) -> None:
    from hhtools.services import upload_resolve

    library_root = tmp_path / "motion-library"
    monkeypatch.setenv("HHTOOLS_MOTION_LIBRARY_ROOT", str(library_root))
    monkeypatch.setattr(gvhmr, "gvhmr_status", lambda: _status(ready=True))

    def fake_run(_video_path: Path, job_root: Path, **_kwargs) -> Path:
        output = job_root / "output" / "source"
        preprocess = output / "preprocess"
        preprocess.mkdir(parents=True)
        (preprocess / "bbx.pt").write_bytes(b"detector-cache")
        (preprocess / "vitpose.pt").write_bytes(b"pose-cache")
        result = output / "hmr4d_results.pt"
        result.write_bytes(b"motion")
        return result

    monkeypatch.setattr(gvhmr, "run_gvhmr", fake_run)
    monkeypatch.setattr(
        upload_resolve,
        "load_clip_at_path",
        lambda *_args, **_kwargs: (synthetic_terrain_motion, "gvhmr"),
    )
    app = _create_test_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/api/video-to-motion/upload",
            files=[("files", ("clip.mp4", b"video", "video/mp4"))],
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = client.get(f"/api/job/{job_id}").json()
            if job["status"] in {"done", "error"}:
                break
            time.sleep(0.01)

        assert job["status"] == "done", job.get("error")
        entries = [
            entry
            for entry in client.get("/api/library").json()["entries"]
            if entry.get("folder_label") == "gvhmr-clip"
        ]

    assert [path.name for path in (library_root / "gvhmr-clip").iterdir()] == [
        "hmr4d_results.pt"
    ]
    assert len(entries) == 1
    assert entries[0]["sequence_id"] == "hmr4d_results.pt"
