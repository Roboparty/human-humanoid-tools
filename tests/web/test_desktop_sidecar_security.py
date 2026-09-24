from __future__ import annotations

from fastapi.testclient import TestClient

from hhtools.web.server import create_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH",
        str(tmp_path / "motion-library-settings.json"),
    )
    monkeypatch.setenv(
        "HHTOOLS_MOTION_LIBRARY_ROOT",
        str(tmp_path / "motion-library"),
    )
    app = create_app(
        source_root=tmp_path / "motions",
        save_dir=tmp_path / "save",
        cache_dir=tmp_path / "cache",
        desktop_session_secret="test-secret",
        desktop_allowed_host="127.0.0.1:43123",
        desktop_allowed_origin="http://127.0.0.1:43123",
    )
    return TestClient(
        app,
        base_url="http://127.0.0.1:43123",
        client=("127.0.0.1", 50_000),
    )


def test_desktop_guard_requires_session_secret(tmp_path, monkeypatch) -> None:
    with _client(tmp_path, monkeypatch) as client:
        response = client.get("/api/health")

    assert response.status_code == 401


def test_desktop_guard_accepts_matching_session(tmp_path, monkeypatch) -> None:
    with _client(tmp_path, monkeypatch) as client:
        response = client.get(
            "/api/health",
            headers={
                "X-HHTools-Session": "test-secret",
                "Origin": "http://127.0.0.1:43123",
            },
        )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_desktop_guard_rejects_wrong_origin(tmp_path, monkeypatch) -> None:
    with _client(tmp_path, monkeypatch) as client:
        response = client.get(
            "/api/health",
            headers={
                "X-HHTools-Session": "test-secret",
                "Origin": "https://example.com",
            },
        )

    assert response.status_code == 403


def test_desktop_guard_uses_versioned_errors_for_agent_failures(
    tmp_path,
    monkeypatch,
) -> None:
    with _client(tmp_path, monkeypatch) as client:
        missing_session = client.get("/api/agent/v1/capabilities")
        wrong_local_origin = client.get(
            "/api/agent/v1/capabilities",
            headers={
                "X-HHTools-Session": "test-secret",
                "Origin": "http://localhost:43123",
            },
        )
        wrong_local_host = client.get(
            "/api/agent/v1/capabilities",
            headers={
                "Host": "localhost:43123",
                "X-HHTools-Session": "test-secret",
            },
        )

    assert missing_session.status_code == 401
    assert missing_session.json()["code"] == "INVALID_DESKTOP_SESSION"
    assert wrong_local_origin.status_code == 403
    assert wrong_local_origin.json()["code"] == "INVALID_DESKTOP_ORIGIN"
    assert wrong_local_host.status_code == 400
    assert wrong_local_host.json()["code"] == "INVALID_DESKTOP_HOST"
    assert "detail" not in missing_session.json()
