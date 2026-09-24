from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hhtools.web import server
from hhtools.web.library.motion_library_links import motions_library_root
from hhtools.web.server import state as server_state
from hhtools.web.server.routes import motion as motion_routes


def _create_test_app(tmp_path: Path, monkeypatch, **limits):
    """Create an isolated app whose ephemeral roots are observable by tests."""

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
        **limits,
    )


def _upload_files(*items: tuple[str, bytes]):
    return [("files", (name, payload, "application/octet-stream")) for name, payload in items]


def _wait_for_job_status(
    client: TestClient,
    job_id: str,
    statuses: set[str],
    *,
    timeout: float = 2.0,
) -> dict:
    deadline = time.monotonic() + timeout
    payload: dict = {}
    while time.monotonic() < deadline:
        payload = client.get(f"/api/job/{job_id}").json()
        if payload.get("status") in statuses:
            return payload
        time.sleep(0.005)
    raise AssertionError(f"job {job_id} did not reach {sorted(statuses)}; last payload: {payload}")


def test_upload_rejects_too_many_files(tmp_path: Path, monkeypatch) -> None:
    app = _create_test_app(
        tmp_path,
        monkeypatch,
        max_upload_files=1,
        max_upload_request_bytes=1024 * 1024,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/dataset/upload",
            files=_upload_files(("one.bin", b"1"), ("two.bin", b"2")),
        )

        assert response.status_code == 413
        assert not list(app.state.session_state.upload_root.rglob("*.bin"))


def test_oversized_file_does_not_publish_partial_upload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_test_app(
        tmp_path,
        monkeypatch,
        max_upload_file_bytes=8,
        max_upload_request_bytes=1024 * 1024,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/dataset/upload",
            files=_upload_files(("small.bin", b"1234"), ("large.bin", b"123456789")),
        )

        upload_root = app.state.session_state.upload_root
        assert response.status_code == 413
        assert not list(upload_root.rglob("*.bin"))
        assert not list(upload_root.rglob("*.upload"))


def test_valid_upload_is_streamed_and_published(tmp_path: Path, monkeypatch) -> None:
    app = _create_test_app(
        tmp_path,
        monkeypatch,
        max_upload_file_bytes=32,
        max_upload_request_bytes=1024 * 1024,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/dataset/upload",
            files=_upload_files(("nested/clip.bin", b"valid-data")),
        )

        written = list(app.state.session_state.upload_root.rglob("clip.bin"))
        assert response.status_code == 200
        assert len(written) == 1
        assert written[0].read_bytes() == b"valid-data"


def test_content_length_limit_rejects_before_route_handler(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_test_app(
        tmp_path,
        monkeypatch,
        max_upload_request_bytes=128,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/dataset/upload",
            files=_upload_files(("clip.bin", b"small-payload")),
        )

        assert response.status_code == 413
        assert not list(app.state.session_state.upload_root.rglob("clip.bin"))


def test_running_limit_queues_one_job_then_returns_429(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from hhtools.web.analysis import dataset_analysis

    started = threading.Event()
    release = threading.Event()

    def blocked_analysis(*_args, **_kwargs):
        started.set()
        release.wait(timeout=2.0)
        return {"available": True}

    monkeypatch.setattr(dataset_analysis, "run_analysis", blocked_analysis)
    app = _create_test_app(
        tmp_path,
        monkeypatch,
        max_running_jobs=1,
        max_queued_jobs=1,
    )

    with TestClient(app) as client:
        first = client.post("/api/dataset/analyze", json={})
        assert first.status_code == 200
        assert started.wait(timeout=1.0)

        second = client.post("/api/dataset/analyze", json={})
        assert second.status_code == 200
        pending = client.get(f"/api/job/{second.json()['job_id']}")
        assert pending.json()["status"] == "pending"

        response = client.post("/api/dataset/analyze", json={})

        assert response.status_code == 429
        assert response.json()["detail"] == "job queue is full (waiting limit: 1)"
        release.set()


def test_zero_job_limits_enable_unlimited_defaults(tmp_path: Path, monkeypatch) -> None:
    app = _create_test_app(
        tmp_path,
        monkeypatch,
        max_running_jobs=0,
        max_queued_jobs=0,
    )

    snapshot = app.state.job_scheduler.snapshot()
    assert snapshot.max_running_jobs == 0
    assert snapshot.max_queued_jobs == 0


def test_job_admission_settings_patch_applies_without_restart_and_persists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings_path = tmp_path / "settings" / "web-settings.json"
    app = _create_test_app(
        tmp_path,
        monkeypatch,
        max_running_jobs=0,
        max_queued_jobs=0,
        job_settings_path=settings_path,
    )
    scheduler_identity = id(app.state.job_scheduler)

    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    ) as client:
        before = client.get("/api/settings/job-admission")
        response = client.patch(
            "/api/settings/job-admission",
            json={
                "max_running_jobs": 2,
                "max_queued_jobs": 32,
                "max_batch_items": 500,
                "max_batch_total_frames": 0,
            },
        )
        health = client.get("/api/health")

    assert before.status_code == 200
    assert before.json()["max_running_jobs"] == 0
    assert before.json()["max_batch_items"] == 0
    assert before.json()["max_batch_total_frames"] == 0
    assert before.json()["editable"] is True
    assert response.status_code == 200
    assert response.json()["max_running_jobs"] == 2
    assert response.json()["max_queued_jobs"] == 32
    assert response.json()["max_batch_items"] == 500
    assert response.json()["max_batch_total_frames"] == 0
    assert health.json()["job_scheduler"]["max_running_jobs"] == 2
    assert id(app.state.job_scheduler) == scheduler_identity
    assert app.state.agent_batch_limit_policy.snapshot().max_batch_items == 500
    assert app.state.agent_batch_limit_policy.snapshot().max_batch_total_frames == 0
    assert app.state.job_settings_store.load().as_payload() == {
        "max_running_jobs": 2,
        "max_queued_jobs": 32,
        "max_batch_items": 500,
        "max_batch_total_frames": 0,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"max_running_jobs": -1},
        {"max_running_jobs": 1.5},
        {"max_running_jobs": True},
        {"max_queued_jobs": "32"},
        {"max_batch_items": -1},
        {"max_batch_total_frames": 1.5},
        {"unexpected": 1},
    ],
)
def test_job_admission_settings_reject_invalid_updates(
    tmp_path: Path,
    monkeypatch,
    payload: dict,
) -> None:
    app = _create_test_app(
        tmp_path,
        monkeypatch,
        max_running_jobs=1,
        max_queued_jobs=8,
        job_settings_path=tmp_path / "web-settings.json",
    )

    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    ) as client:
        response = client.patch("/api/settings/job-admission", json=payload)

    assert response.status_code == 422
    snapshot = app.state.job_scheduler.snapshot()
    assert snapshot.max_running_jobs == 1
    assert snapshot.max_queued_jobs == 8


def test_job_admission_settings_reject_non_loopback_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_test_app(
        tmp_path,
        monkeypatch,
        job_settings_path=tmp_path / "web-settings.json",
    )

    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("192.0.2.10", 50000),
    ) as client:
        settings = client.get("/api/settings/job-admission")
        response = client.patch(
            "/api/settings/job-admission",
            json={"max_running_jobs": 1, "max_queued_jobs": 8},
        )

    assert settings.status_code == 200
    assert settings.json()["editable"] is False
    assert response.status_code == 403
    assert not (tmp_path / "web-settings.json").exists()


def test_job_admission_settings_reject_dns_rebinding_host(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_test_app(
        tmp_path,
        monkeypatch,
        job_settings_path=tmp_path / "web-settings.json",
    )

    with TestClient(
        app,
        base_url="http://attacker.example",
        client=("127.0.0.1", 50000),
    ) as client:
        response = client.patch(
            "/api/settings/job-admission",
            json={"max_running_jobs": 1, "max_queued_jobs": 8},
        )

    assert response.status_code == 403
    assert not (tmp_path / "web-settings.json").exists()


def test_full_queue_rejects_motion_before_upload_or_library_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from hhtools.web.analysis import dataset_analysis

    started = threading.Event()
    release = threading.Event()

    def blocked_analysis(*_args, **_kwargs):
        started.set()
        release.wait(timeout=2.0)
        return {"available": True}

    monkeypatch.setattr(dataset_analysis, "run_analysis", blocked_analysis)
    app = _create_test_app(
        tmp_path,
        monkeypatch,
        max_running_jobs=1,
        max_queued_jobs=1,
    )
    library = motions_library_root() / "queue-protected-library"
    library.mkdir(parents=True, exist_ok=True)
    marker = library / "keep.bvh"
    marker.write_bytes(b"existing")

    with TestClient(app) as client:
        try:
            assert client.post("/api/dataset/analyze", json={}).status_code == 200
            assert started.wait(timeout=1.0)
            assert client.post("/api/dataset/analyze", json={}).status_code == 200

            response = client.post(
                "/api/motion/upload",
                params={"library_folder_label": "queue-protected-library"},
                files={"files": ("replacement.bvh", b"candidate", "text/plain")},
            )

            assert response.status_code == 429
            assert marker.read_bytes() == b"existing"
            assert not (library / "replacement.bvh").exists()
            assert not list(app.state.session_state.upload_root.rglob("replacement.bvh"))
        finally:
            release.set()


def test_queued_same_label_motion_uploads_parse_their_immutable_drops(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A later upload must not replace the bytes an earlier queued job parses."""

    from hhtools.web.analysis import dataset_analysis
    from hhtools.web.library import motion_library_links, upload_resolve

    blocker_started = threading.Event()
    release_blocker = threading.Event()
    registration_started = threading.Event()
    release_registration = threading.Event()
    parsed_payloads: list[bytes] = []

    def blocked_analysis(*_args, **_kwargs):
        blocker_started.set()
        release_blocker.wait(timeout=2.0)
        return {"available": True}

    def recording_resolver(drop: Path, profile: str, **_kwargs):
        picked = next(Path(drop).rglob("clip.bvh"))
        parsed_payloads.append(picked.read_bytes())
        # The dummy motion intentionally fails later in registration.  This
        # test isolates admission, immutable-drop parsing, and library publish.
        return object(), "amass", {"profile": profile, "picked": str(picked)}

    def simple_library_entry(
        folder_label: str,
        _lib_dir: Path,
        picked: Path,
        dataset: str | None,
    ) -> dict:
        return {
            "dataset": dataset or "unknown",
            "folder_label": folder_label,
            "sequence_id": picked.name,
            "source_path": str(picked),
        }

    def no_resolved_source(_paths: list[str]):
        raise FileNotFoundError

    def blocked_grounding(motion):
        registration_started.set()
        release_registration.wait(timeout=2.0)
        return motion

    monkeypatch.setattr(dataset_analysis, "run_analysis", blocked_analysis)
    monkeypatch.setattr(upload_resolve, "resolve_upload_drop", recording_resolver)
    monkeypatch.setattr(motion_routes, "_library_entry_from_link", simple_library_entry)
    monkeypatch.setattr(motion_routes, "_ground_motion_for_web", blocked_grounding)
    monkeypatch.setattr(
        motion_library_links,
        "auto_resolve_source_files",
        no_resolved_source,
    )
    monkeypatch.setattr(
        motion_library_links,
        "auto_resolve_source_dir",
        no_resolved_source,
    )
    app = _create_test_app(
        tmp_path,
        monkeypatch,
        max_running_jobs=1,
        max_queued_jobs=2,
    )
    library = motions_library_root() / "用户数据集"
    first_bytes = b"first-motion"
    second_bytes = b"second-motion-is-a-different-size"

    with TestClient(app) as client:
        try:
            blocker = client.post("/api/dataset/analyze", json={})
            assert blocker.status_code == 200
            assert blocker_started.wait(timeout=1.0)

            first = client.post(
                "/api/motion/upload",
                files={"files": ("clip.bvh", first_bytes, "text/plain")},
            )
            second = client.post(
                "/api/motion/upload",
                files={"files": ("clip.bvh", second_bytes, "text/plain")},
            )
            assert first.status_code == second.status_code == 200
            assert first.json()["materialize_mode"] == "pending"
            assert second.json()["materialize_mode"] == "pending"

            first_id = first.json()["job_id"]
            second_id = second.json()["job_id"]
            assert client.get(f"/api/job/{first_id}").json()["status"] == "pending"
            assert client.get(f"/api/job/{second_id}").json()["status"] == "pending"
            assert not library.exists()

            release_blocker.set()
            assert registration_started.wait(timeout=1.0)
            published = app.state.session_state.job_history.get(first_id)
            assert published is not None
            assert published["request"]["folder_label"] == "用户数据集"
            assert published["request"]["materialize_mode"] in {
                "symlink",
                "hardlink",
                "copy",
            }
            release_registration.set()
            _wait_for_job_status(client, first_id, {"done", "error"})
            _wait_for_job_status(client, second_id, {"done", "error"})

            assert parsed_payloads == [first_bytes, second_bytes]
            assert (library / "clip.bvh").read_bytes() == second_bytes
            for job_id in (first_id, second_id):
                stored = app.state.session_state.job_history.get(job_id)
                assert stored is not None
                assert stored["request"]["folder_label"] == "用户数据集"
                assert stored["request"]["materialize_mode"] in {
                    "symlink",
                    "hardlink",
                    "copy",
                }
        finally:
            release_blocker.set()
            release_registration.set()


def test_manual_library_link_waits_for_motion_publish_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Every same-process writer must share the Motion Library namespace lock."""

    from hhtools.web.library import motion_library_links, upload_resolve

    publish_started = threading.Event()
    release_publish = threading.Event()
    link_call_started = threading.Event()
    link_entered = threading.Event()
    library_dir = motions_library_root() / "locked-publish"
    library_clip = library_dir / "clip.bvh"

    def recording_resolver(drop: Path, profile: str, **_kwargs):
        picked = next(Path(drop).rglob("clip.bvh"))
        return object(), "amass", {"profile": profile, "picked": str(picked)}

    def blocking_materialize(*_args, **_kwargs):
        library_dir.mkdir(parents=True, exist_ok=True)
        library_clip.write_bytes(b"motion")
        publish_started.set()
        release_publish.wait(timeout=2.0)
        return library_dir, library_dir.name, "copy"

    def fake_link(
        _path: str,
        *,
        folder_label: str | None = None,
        library_root: str | Path | None = None,
    ) -> Path:
        link_entered.set()
        assert library_root is not None
        destination = Path(library_root) / (folder_label or "manual-link")
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    monkeypatch.setattr(upload_resolve, "resolve_upload_drop", recording_resolver)
    monkeypatch.setattr(motion_library_links, "materialize_drop", blocking_materialize)
    monkeypatch.setattr(motion_library_links, "link_to_library", fake_link)
    monkeypatch.setattr(
        motion_library_links,
        "scan_motions_library",
        lambda _root=None: [],
    )
    monkeypatch.setattr(
        motion_routes,
        "_matching_materialized_clip",
        lambda *_args, **_kwargs: library_clip,
    )
    monkeypatch.setattr(
        motion_routes,
        "_library_entry_from_link",
        lambda label, _root, picked, dataset: {
            "dataset": dataset or "unknown",
            "folder_label": label,
            "sequence_id": picked.name,
            "source_path": str(picked),
        },
    )
    app = _create_test_app(tmp_path, monkeypatch)
    link_endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/api/library/link"
        and "POST" in getattr(route, "methods", set())
    )
    link_result: dict = {}

    def call_manual_link() -> None:
        link_call_started.set()
        link_result.update(link_endpoint({"path": str(tmp_path), "folder_label": "manual-link"}))

    with TestClient(app) as client:
        try:
            response = client.post(
                "/api/motion/upload",
                files={"files": ("clip.bvh", b"motion", "text/plain")},
            )
            assert response.status_code == 200
            assert publish_started.wait(timeout=1.0)

            link_thread = threading.Thread(target=call_manual_link, daemon=True)
            link_thread.start()
            assert link_call_started.wait(timeout=1.0)
            assert not link_entered.wait(timeout=0.05)

            release_publish.set()
            assert link_entered.wait(timeout=1.0)
            link_thread.join(timeout=1.0)
            assert not link_thread.is_alive()
            assert link_result["folder_label"] == "manual-link"
        finally:
            release_publish.set()


def test_upload_directory_failure_releases_scheduler_reservation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_test_app(
        tmp_path,
        monkeypatch,
        max_running_jobs=1,
        max_queued_jobs=1,
    )
    upload_root = app.state.session_state.upload_root
    original_mkdir = Path.mkdir

    def fail_drop_mkdir(path: Path, *args, **kwargs) -> None:
        if path.parent == upload_root:
            raise OSError("simulated upload directory failure")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_drop_mkdir)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/basket/upload",
            files={"files": ("clip.bvh", b"candidate", "text/plain")},
        )

        assert response.status_code == 500
        assert app.state.job_scheduler.snapshot().reserved_jobs == 0


def test_running_state_persistence_failure_marks_job_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_test_app(tmp_path, monkeypatch)
    store = app.state.session_state.job_history
    original_put = store.put
    calls = 0

    def fail_second_put(record: dict) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated history write failure")
        original_put(record)

    monkeypatch.setattr(store, "put", fail_second_put)
    with TestClient(app) as client:
        response = client.post("/api/dataset/analyze", json={})
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        deadline = time.monotonic() + 2.0
        payload: dict = {}
        while time.monotonic() < deadline:
            payload = client.get(f"/api/job/{job_id}").json()
            if payload["status"] == "error":
                break
            time.sleep(0.005)

        assert payload["status"] == "error"
        assert "simulated history write failure" in payload["error"]


@pytest.mark.parametrize("name", ["max_running_jobs", "max_queued_jobs"])
def test_negative_job_limit_is_rejected(
    tmp_path: Path,
    monkeypatch,
    name: str,
) -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        _create_test_app(tmp_path, monkeypatch, **{name: -1})


def test_pending_job_is_not_pruned_by_ttl(tmp_path: Path, monkeypatch) -> None:
    app = _create_test_app(tmp_path, monkeypatch, job_ttl_seconds=0.01)
    state = app.state.session_state
    pending = server_state.Job(
        id="pending-job",
        kind="test",
        status="pending",
        created_at=time.monotonic() - 10,
        last_accessed_at=time.monotonic() - 10,
    )
    with state.job_lock:
        state.jobs[pending.id] = pending

    with TestClient(app) as client:
        response = client.get(f"/api/job/{pending.id}")

        assert response.status_code == 200
        assert pending.id in state.jobs


def test_job_retention_prunes_oldest_artifact(tmp_path: Path, monkeypatch) -> None:
    app = _create_test_app(
        tmp_path,
        monkeypatch,
        max_retained_jobs=1,
        job_ttl_seconds=3600,
    )
    state = app.state.session_state
    now = time.monotonic()
    artifact = state.export_root / "old.zip"
    artifact.write_bytes(b"generated")
    old = server_state.Job(
        id="old-job",
        kind="test",
        status="done",
        result={"artifact_path": str(artifact)},
        created_at=now - 20,
        terminal_since=now - 20,
        last_accessed_at=now - 20,
    )
    latest = server_state.Job(
        id="latest-job",
        kind="test",
        status="done",
        created_at=now - 10,
        terminal_since=now - 10,
        last_accessed_at=now - 10,
    )
    with state.job_lock:
        state.jobs.update({old.id: old, latest.id: latest})

    with TestClient(app) as client:
        response = client.get(f"/api/job/{latest.id}")

        assert response.status_code == 200
        assert old.id not in state.jobs
        assert latest.id in state.jobs
        assert not artifact.exists()


def test_job_ttl_starts_when_worker_reaches_terminal_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _create_test_app(tmp_path, monkeypatch, job_ttl_seconds=0.01)
    state = app.state.session_state
    expired = server_state.Job(id="expired-job", kind="test")
    expired.mark_terminal("done")
    expired.terminal_since = time.monotonic() - 1
    expired.last_accessed_at = expired.terminal_since
    with state.job_lock:
        state.jobs[expired.id] = expired

    with TestClient(app) as client:
        response = client.get(f"/api/job/{expired.id}")

        assert response.status_code == 404
        assert expired.id not in state.jobs


def test_lifespan_removes_session_owned_temp_files(tmp_path: Path, monkeypatch) -> None:
    app = _create_test_app(tmp_path, monkeypatch)
    state = app.state.session_state
    upload_file = state.upload_root / "pending.bin"
    export_file = state.export_root / "generated.zip"
    cache_file = state.cache.cache_dir / "generated.npz"
    upload_file.write_bytes(b"upload")
    export_file.write_bytes(b"export")
    cache_file.write_bytes(b"cache")
    state.cache.written.add(cache_file.name)

    with TestClient(app):
        assert upload_file.exists()
        assert export_file.exists()
        assert cache_file.exists()

    assert not state.upload_root.exists()
    assert not state.export_root.exists()
    assert not cache_file.exists()
