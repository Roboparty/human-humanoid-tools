from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from hhtools.agent.boundary import (
    AGENT_MAX_BODY_BYTES,
    LEGACY_UPGRADE_MAX_BODY_BYTES,
    AgentBoundaryMiddleware,
)


def _app() -> FastAPI:
    app = FastAPI()

    @app.api_route("/api/agent/v1/echo", methods=["GET", "POST"])
    async def echo(request: Request) -> dict[str, int]:
        return {"size": len(await request.body())}

    @app.post("/api/agent/v1/legacy/jobspec-v1/upgrade")
    async def legacy(request: Request) -> dict[str, int]:
        return {"size": len(await request.body())}

    @app.get("/api/unrelated")
    def unrelated() -> dict[str, bool]:
        return {"ok": True}

    app.add_middleware(AgentBoundaryMiddleware)
    return app


def _local_client(app: FastAPI, *, host: str = "127.0.0.1") -> TestClient:
    return TestClient(
        app,
        base_url=f"http://{host}",
        client=("127.0.0.1", 50_000),
    )


def test_boundary_requires_literal_loopback_client_and_host() -> None:
    app = _app()

    with _local_client(app, host="localhost") as client:
        accepted = client.get("/api/agent/v1/echo")
    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("192.0.2.44", 50_000),
    ) as client:
        remote = client.get(
            "/api/agent/v1/echo",
            headers={"X-Forwarded-For": "127.0.0.1"},
        )
        unrelated = client.get("/api/unrelated", headers={"Host": "attacker.example"})
    with TestClient(
        app,
        base_url="http://attacker.example",
        client=("127.0.0.1", 50_000),
    ) as client:
        bad_host = client.get(
            "/api/agent/v1/echo",
            headers={"Forwarded": "host=127.0.0.1"},
        )
        huge_port = client.get(
            "/api/agent/v1/echo",
            headers={"Host": f"127.0.0.1:{'9' * 5_000}"},
        )
    with TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("localhost", 50_000),
    ) as client:
        named_client = client.get("/api/agent/v1/echo")

    assert accepted.status_code == 200
    assert remote.status_code == 403
    assert remote.json()["code"] == "LOOPBACK_REQUIRED"
    assert bad_host.status_code == 400
    assert bad_host.json()["code"] == "INVALID_HOST"
    assert huge_port.status_code == 400
    assert huge_port.json()["code"] == "INVALID_HOST"
    assert named_client.status_code == 403
    assert unrelated.status_code == 200
    assert "detail" not in remote.json()
    assert remote.headers["x-content-type-options"] == "nosniff"


def test_boundary_rejects_cross_site_browser_origin_but_allows_cli_and_loopback() -> None:
    app = _app()
    with _local_client(app) as client:
        cli = client.post("/api/agent/v1/echo", content=b"{}")
        local_browser = client.post(
            "/api/agent/v1/echo",
            content=b"{}",
            headers={"Origin": "http://localhost:5173"},
        )
        cross_site = client.post(
            "/api/agent/v1/echo",
            content=b"{}",
            headers={"Origin": "https://evil.example"},
        )
        null_origin = client.post(
            "/api/agent/v1/echo",
            content=b"{}",
            headers={"Origin": "null"},
        )

    assert cli.status_code == 200
    assert local_browser.status_code == 200
    assert cross_site.status_code == 403
    assert cross_site.json()["code"] == "ORIGIN_FORBIDDEN"
    assert null_origin.status_code == 403


def test_boundary_enforces_generic_body_cap_with_and_without_content_length() -> None:
    app = _app()
    at_limit = b"x" * AGENT_MAX_BODY_BYTES
    over_limit = at_limit + b"x"

    def chunked_over_limit():
        yield at_limit
        yield b"x"

    with _local_client(app) as client:
        accepted = client.post("/api/agent/v1/echo", content=at_limit)
        rejected = client.post("/api/agent/v1/echo", content=over_limit)
        chunked = client.post("/api/agent/v1/echo", content=chunked_over_limit())

    assert accepted.status_code == 200
    assert accepted.json()["size"] == AGENT_MAX_BODY_BYTES
    assert rejected.status_code == 413
    assert rejected.json()["code"] == "REQUEST_TOO_LARGE"
    assert rejected.json()["details"] == {"max_bytes": AGENT_MAX_BODY_BYTES}
    assert chunked.status_code == 413
    assert chunked.json()["code"] == "REQUEST_TOO_LARGE"


def test_boundary_applies_exact_legacy_cap_and_validates_transport_headers() -> None:
    app = _app()
    path = "/api/agent/v1/legacy/jobspec-v1/upgrade"
    at_limit = b"x" * LEGACY_UPGRADE_MAX_BODY_BYTES

    with _local_client(app) as client:
        accepted = client.post(path, content=at_limit)
        rejected = client.post(path, content=at_limit + b"x")
        invalid_length = client.post(
            path,
            content=b"{}",
            headers={"Content-Length": "two"},
        )
        mismatched_length = client.post(
            path,
            content=b"{}",
            headers={"Content-Length": "1"},
        )
        encoded = client.post(
            path,
            content=b"{}",
            headers={"Content-Encoding": "gzip"},
        )
        identity = client.post(
            path,
            content=b"{}",
            headers={"Content-Encoding": "identity"},
        )

    assert accepted.status_code == 200
    assert accepted.json()["size"] == LEGACY_UPGRADE_MAX_BODY_BYTES
    assert rejected.status_code == 413
    assert rejected.json()["details"] == {"max_bytes": LEGACY_UPGRADE_MAX_BODY_BYTES}
    assert invalid_length.status_code == 400
    assert invalid_length.json()["code"] == "INVALID_CONTENT_LENGTH"
    assert mismatched_length.status_code == 400
    assert mismatched_length.json()["code"] == "INVALID_CONTENT_LENGTH"
    assert encoded.status_code == 415
    assert encoded.json()["code"] == "UNSUPPORTED_CONTENT_ENCODING"
    assert identity.status_code == 200
