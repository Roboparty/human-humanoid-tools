from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hhtools.web import server
from hhtools.web.server import state as server_state
from hhtools.web.server.job_runtime import WebJobRuntime


def _create_test_app(tmp_path: Path, monkeypatch):
    def local_tmpdir(tag: str) -> Path:
        path = tmp_path / f"runtime-{tag}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(server_state, "_tmpdir", local_tmpdir)
    monkeypatch.setattr(server_state, "_robot_library_root", lambda: tmp_path / "robots")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return server.create_app(
        source_root=tmp_path / "motions",
        save_dir=tmp_path / "save",
        cache_dir=tmp_path / "cache",
        # Never let a developer's HHTOOLS/XDG environment select the persistent
        # user history during a test.  The explicit argument also makes restart
        # tests share only this test's temporary store.
        job_history_dir=tmp_path / "job-history",
    )


def test_job_list_returns_compact_newest_first_records(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_test_app(tmp_path, monkeypatch)
    state = app.state.session_state
    artifact = state.export_root / "batch.zip"
    artifact.write_bytes(b"result")
    wall_now = time.time()
    old_accessed = time.monotonic() - 20
    older = server_state.Job(
        id="older",
        kind="retarget",
        request={"robot": "unitree_g1", "backend": "newton"},
        status="done",
        progress=1.0,
        result={"num_frames": 120},
        created_wall_time=wall_now - 20,
        finished_wall_time=wall_now - 10,
        terminal_since=time.monotonic() - 10,
        last_accessed_at=old_accessed,
    )
    latest = server_state.Job(
        id="latest",
        kind="batch",
        request={"entries": [{"source_path": "a"}, {"source_path": "b"}]},
        status="done",
        progress=1.0,
        result={
            "written": ["a.csv", "b.csv"],
            "failures": [],
            "artifact_path": str(artifact),
            "download_name": "batch.zip",
        },
        created_wall_time=wall_now - 5,
        finished_wall_time=wall_now - 1,
        terminal_since=time.monotonic() - 1,
    )
    with state.job_lock:
        state.jobs.update({older.id: older, latest.id: latest})

    with TestClient(app) as client:
        response = client.get("/api/jobs")

        assert response.status_code == 200
        payload = response.json()
        assert payload["session_only"] is False
        assert payload["persistence"] == "disk"
        assert [record["id"] for record in payload["jobs"]] == ["latest", "older"]
        assert payload["jobs"][0]["parameters"] == {"entry_count": 2}
        assert payload["jobs"][0]["result_summary"] == {
            "download_name": "batch.zip",
            "success_count": 2,
            "failure_count": 0,
        }
        assert payload["jobs"][0]["can_download"] is True
        assert payload["jobs"][0]["scope"] == "current_session"
        assert "request" not in payload["jobs"][0]
        assert "result" not in payload["jobs"][0]
        assert older.last_accessed_at == old_accessed


@pytest.mark.parametrize(
    ("kind", "filename", "media_type"),
    [
        ("video_to_motion", "hmr4d_results.pt", "application/octet-stream"),
        ("retarget", "walk.csv", "text/csv"),
        ("r2r_retarget", "r2r.csv", "text/csv"),
        ("batch", "batch.zip", "application/zip"),
    ],
)
def test_completed_job_artifact_download_survives_restart(
    tmp_path: Path,
    monkeypatch,
    kind: str,
    filename: str,
    media_type: str,
) -> None:
    app = _create_test_app(tmp_path, monkeypatch)
    state = app.state.session_state
    artifact = state.export_root / kind / filename
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"result-bytes")
    runtime = WebJobRuntime(
        state,
        app.state.job_scheduler,
        max_retained_jobs=64,
        job_ttl_seconds=60,
    )
    job = server_state.Job(
        id=f"{kind}-artifact",
        kind=kind,
        result={"artifact_path": str(artifact), "download_name": filename},
        on_terminal=runtime.persist_terminal,
    )
    with state.job_lock:
        state.jobs[job.id] = job
    job.mark_terminal("done")

    with TestClient(app) as client:
        listed = next(
            item for item in client.get("/api/jobs").json()["jobs"] if item["id"] == job.id
        )
        response = client.get(f"/api/job/{job.id}/download")

        assert listed["can_download"] is True
        assert response.content == b"result-bytes"
        assert response.headers["content-type"].startswith(media_type)
        assert filename in response.headers["content-disposition"]

    restarted = _create_test_app(tmp_path, monkeypatch)
    with TestClient(restarted) as client:
        listed = next(
            item for item in client.get("/api/jobs").json()["jobs"] if item["id"] == job.id
        )
        response = client.get(f"/api/job/{job.id}/download")

        assert listed["scope"] == "persistent"
        assert listed["can_download"] is True
        assert response.content == b"result-bytes"
        assert response.headers["content-type"].startswith(media_type)


def test_terminal_status_waits_for_artifact_adoption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_test_app(tmp_path, monkeypatch)
    state = app.state.session_state
    artifact = state.export_root / "result.zip"
    artifact.write_bytes(b"result-bytes")
    runtime = WebJobRuntime(
        state,
        app.state.job_scheduler,
        max_retained_jobs=64,
        job_ttl_seconds=60,
    )
    adoption_started = threading.Event()
    allow_adoption = threading.Event()
    original_adopt = state.job_history.adopt_artifact

    def blocking_adopt(*args, **kwargs):
        adoption_started.set()
        if not allow_adoption.wait(timeout=2):
            raise TimeoutError("test did not release artifact adoption")
        return original_adopt(*args, **kwargs)

    monkeypatch.setattr(state.job_history, "adopt_artifact", blocking_adopt)
    job = server_state.Job(
        id="ordered-artifact",
        kind="batch",
        result={"artifact_path": str(artifact), "download_name": artifact.name},
        on_terminal=runtime.persist_terminal,
    )
    with state.job_lock:
        state.jobs[job.id] = job

    worker = threading.Thread(target=job.mark_terminal, args=("done",))
    with TestClient(app) as client:
        worker.start()
        try:
            assert adoption_started.wait(timeout=2)
            visible = client.get(f"/api/job/{job.id}").json()
            assert visible["status"] == "running"
            assert visible["can_download"] is False
            assert visible["result"]["artifact_path"] == str(artifact)
        finally:
            allow_adoption.set()
            worker.join(timeout=2)

        assert worker.is_alive() is False
        visible = client.get(f"/api/job/{job.id}").json()
        retained = Path(visible["result"]["artifact_path"])
        assert visible["status"] == "done"
        assert visible["can_download"] is True
        assert retained.is_relative_to(state.job_history.artifacts_dir)
        assert state.job_history.get(job.id)["artifact_path"] == str(retained)


def test_job_config_and_status_expose_captured_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_test_app(tmp_path, monkeypatch)
    state = app.state.session_state
    created = time.time() - 2
    job = server_state.Job(
        id="config-job",
        kind="r2r_retarget",
        request={
            "source_robot": "unitree_g1",
            "target": "booster_t1",
            "backend": "newton",
        },
        progress=0.25,
        message="solving",
        created_wall_time=created,
    )
    with state.job_lock:
        state.jobs[job.id] = job

    with TestClient(app) as client:
        config_response = client.get(f"/api/job/{job.id}/config")
        status_response = client.get(f"/api/job/{job.id}")

        assert config_response.status_code == 200
        assert config_response.json() == {
            "schema_version": 1,
            "job_id": job.id,
            "kind": "r2r_retarget",
            "status": "running",
            "created_at": created,
            "finished_at": None,
            "scope": "current_session",
            "request": job.request,
            "cli": {
                "available": False,
                "command": None,
                "reason": "该任务类型暂时没有等价的 hhtools CLI 命令。",
            },
            "spec": {
                "schema_version": 1,
                "kind": "r2r_retarget",
                "request": job.request,
            },
            "replay": {
                "available": False,
                "reason": "该任务依赖会话内对象，当前只能复制编辑配置，不能直接重跑。",
                "source_count": 0,
            },
            "parent_job_id": None,
        }
        status = status_response.json()
        assert status["progress"] == 0.25
        assert status["parameters"] == {
            "target": "booster_t1",
            "source_robot": "unitree_g1",
            "backend": "newton",
        }
        assert status["result"] is None


def test_mark_terminal_records_wall_clock_completion() -> None:
    job = server_state.Job(id="done-job", kind="test")

    job.mark_terminal("done")

    assert job.status == "done"
    assert job.terminal_since is not None
    assert job.finished_wall_time is not None
    assert job.finished_wall_time >= job.created_wall_time


def test_job_history_fixture_ignores_user_directory_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    simulated_user_history = tmp_path / "simulated-user-history"
    user_records = simulated_user_history / "records"
    user_records.mkdir(parents=True)
    user_record = user_records / "real-user-active.json"
    user_record.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "real-user-active",
                "status": "running",
                "created_at": 1.0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    original_user_bytes = user_record.read_bytes()
    monkeypatch.setenv("HHTOOLS_JOB_HISTORY_DIR", str(simulated_user_history))

    app = _create_test_app(tmp_path, monkeypatch)
    isolated_store = app.state.session_state.job_history
    assert isolated_store.root == (tmp_path / "job-history").resolve()

    with TestClient(app) as client:
        listing = client.get("/api/jobs").json()
        assert all(record["id"] != "real-user-active" for record in listing["jobs"])
        isolated_store.put(
            {
                "id": "isolated-test-job",
                "status": "done",
                "created_at": 2.0,
            }
        )

    # The listing above proves the record was not loaded. Opening the user store
    # would also recover this active record and rewrite it as an error, so byte
    # equality plus the absent test record proves there was no write-back.
    assert user_record.read_bytes() == original_user_bytes
    assert not (user_records / "isolated-test-job.json").exists()
    assert (isolated_store.records_dir / "isolated-test-job.json").is_file()


def test_job_history_survives_app_restart(tmp_path: Path, monkeypatch) -> None:
    app = _create_test_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/api/dataset/preview_robot",
            json={"source_path": str(tmp_path / "missing.csv")},
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            status = client.get(f"/api/job/{job_id}").json()
            if status["status"] == "error":
                break
            time.sleep(0.01)
        assert status["status"] == "error"

    restarted = _create_test_app(tmp_path, monkeypatch)
    with TestClient(restarted) as client:
        listing = client.get("/api/jobs").json()
        restored = next(record for record in listing["jobs"] if record["id"] == job_id)
        config = client.get(f"/api/job/{job_id}/config").json()

        assert restored["scope"] == "persistent"
        assert restored["status"] == "error"
        assert config["scope"] == "persistent"
        assert config["request"]["source_path"].endswith("missing.csv")


def test_cli_and_config_download_for_source_backed_h2r_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_test_app(tmp_path, monkeypatch)
    source = tmp_path / "walk.npz"
    source.write_bytes(b"test")
    job = server_state.Job(
        id="cli-job",
        kind="retarget",
        request={
            "source_path": str(source),
            "robot": "unitree_g1",
            "backend": "newton",
            "reference": "smpl",
            "human_height": 1.72,
            "ik_iterations": 24,
        },
    )
    with app.state.session_state.job_lock:
        app.state.session_state.jobs[job.id] = job

    with TestClient(app) as client:
        cli_response = client.get(f"/api/job/{job.id}/cli")
        config_response = client.get(f"/api/job/{job.id}/config/download")

        assert cli_response.status_code == 200
        cli = cli_response.json()
        assert cli["available"] is True
        assert "hhtools retarget run" in cli["command"]
        assert "walk.npz" in cli["command"]
        assert "--robot unitree_g1" in cli["command"]
        assert config_response.status_code == 200
        assert config_response.headers["content-type"].startswith("application/json")
        assert config_response.json()["request"] == job.request


def test_persisted_cli_rechecks_that_source_files_still_exist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_test_app(tmp_path, monkeypatch)
    source = tmp_path / "walk.npz"
    source.write_bytes(b"test")
    request = {
        "source_path": str(source),
        "robot": "unitree_g1",
        "backend": "newton",
        "reference": "smpl",
    }
    app.state.session_state.job_history.put(
        {
            "id": "persisted-cli-job",
            "kind": "retarget",
            "status": "done",
            "progress": 1.0,
            "clip_progress": 1.0,
            "message": "完成",
            "error": None,
            "created_at": time.time(),
            "finished_at": time.time(),
            "duration_seconds": 1.0,
            "parameters": {},
            "result_summary": {},
            "request": request,
            "cli": {"available": True, "command": "stale", "reason": None},
        }
    )
    source.unlink()

    with TestClient(app) as client:
        listing = client.get("/api/jobs").json()
        record = next(job for job in listing["jobs"] if job["id"] == "persisted-cli-job")
        cli = client.get("/api/job/persisted-cli-job/cli").json()
        config = client.get("/api/job/persisted-cli-job/config").json()

        assert record["can_copy_cli"] is False
        assert cli["available"] is False
        assert config["cli"]["available"] is False


def test_job_spec_validation_accepts_downloaded_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_test_app(tmp_path, monkeypatch)
    source = tmp_path / "walk.npz"
    source.write_bytes(b"test")

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs/spec/validate",
            json={
                "job_id": "downloaded-job",
                "spec": {
                    "schema_version": 1,
                    "kind": "retarget",
                    "request": {
                        "robot": "unitree_g1",
                        "source_path": str(source),
                    },
                },
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["spec"]["kind"] == "retarget"
        assert payload["replay"] == {
            "available": True,
            "reason": None,
            "source_count": 1,
        }


def test_failed_only_replay_filters_batch_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_test_app(tmp_path, monkeypatch)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    original = server_state.Job(
        id="batch-with-failure",
        kind="batch",
        status="done",
        progress=1.0,
        request={
            "robot": "missing_test_robot",
            "out_dir": "original",
            "entries": [
                {"source_path": str(first), "stem": "first"},
                {"source_path": str(second), "stem": "second"},
            ],
        },
        result={"failures": [{"source_path": str(second), "stage": "retarget", "reason": "test"}]},
    )
    with app.state.session_state.job_lock:
        app.state.session_state.jobs[original.id] = original

    with TestClient(app) as client:
        response = client.post(
            "/api/jobs/replay",
            json={"job_id": original.id, "failed_only": True},
        )

        assert response.status_code == 200
        job_id = response.json()["job_id"]
        config = client.get(f"/api/job/{job_id}/config").json()
        assert config["parent_job_id"] == original.id
        assert config["spec"]["request"]["out_dir"] == "original_failed_retry"
        assert [entry["stem"] for entry in config["spec"]["request"]["entries"]] == ["second"]
