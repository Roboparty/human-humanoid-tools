"""Browser shell and request middleware."""

from __future__ import annotations

import hmac
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse

from hhtools.agent.boundary import (
    _is_loopback_literal,
    _loopback_host,
    _loopback_origin,
    agent_error_response,
    is_agent_path,
)
from hhtools.web.server.boundary import _UPLOAD_ENDPOINTS

_STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def register_shell_routes(
    app,
    *,
    static_dir: Path,
    ui_build_id: str,
    max_upload_request_bytes: int,
    desktop_session_secret: str | None,
    desktop_allowed_host: str | None,
    desktop_allowed_origin: str | None,
) -> None:
    UI_BUILD_ID = ui_build_id

    def _render_index_html() -> str:
        raw = (static_dir / "index.html").read_text(encoding="utf-8")
        return raw.replace("{{UI_BUILD}}", UI_BUILD_ID)

    @app.get("/")
    @app.get("/index.html")
    def serve_index():
        return HTMLResponse(
            _render_index_html(),
            headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"},
        )

    @app.middleware("http")
    async def _reject_oversized_upload_requests(request, call_next):  # type: ignore[no-untyped-def]
        """Reject normal browser multipart requests before Starlette parses their files."""
        if request.method == "POST" and request.url.path in _UPLOAD_ENDPOINTS:
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    request_bytes = int(content_length)
                except ValueError:
                    return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
                if request_bytes > max_upload_request_bytes:
                    return JSONResponse(
                        {"detail": (f"upload request exceeds {max_upload_request_bytes} bytes")},
                        status_code=413,
                    )
        return await call_next(request)

    def _desktop_guard_failure(
        request: Request,
        *,
        status_code: int,
        code: str,
        message: str,
        legacy_detail: str,
    ) -> JSONResponse:
        if is_agent_path(request.url.path):
            return agent_error_response(
                status_code=status_code,
                code=code,
                message=message,
            )
        return JSONResponse({"detail": legacy_detail}, status_code=status_code)

    def _local_write_boundary_failure(request: Request) -> JSONResponse | None:
        headers = list(request.scope.get("headers", []))
        if request.client is None or not _is_loopback_literal(request.client.host):
            return JSONResponse({"detail": "Loopback connection required"}, status_code=403)
        if not _loopback_host(headers):
            return JSONResponse({"detail": "Invalid localhost Host"}, status_code=403)
        if not _loopback_origin(headers):
            return JSONResponse({"detail": "Origin forbidden"}, status_code=403)
        return None

    @app.middleware("http")
    async def _desktop_request_guard(request, call_next):  # type: ignore[no-untyped-def]
        """Enforce the local trust boundary, plus Electron's per-launch session."""
        if desktop_session_secret is not None:
            host = request.headers.get("host", "")
            if desktop_allowed_host is not None and host.lower() != desktop_allowed_host.lower():
                return _desktop_guard_failure(
                    request,
                    status_code=400,
                    code="INVALID_DESKTOP_HOST",
                    message="The desktop Agent request used an unexpected Host.",
                    legacy_detail="Invalid desktop host",
                )

            supplied_secret = request.headers.get("x-hhtools-session", "")
            if not hmac.compare_digest(supplied_secret, desktop_session_secret):
                return _desktop_guard_failure(
                    request,
                    status_code=401,
                    code="INVALID_DESKTOP_SESSION",
                    message="The desktop Agent session is invalid.",
                    legacy_detail="Invalid desktop session",
                )

            origin = request.headers.get("origin")
            if (
                origin is not None
                and desktop_allowed_origin is not None
                and origin != desktop_allowed_origin
            ):
                return _desktop_guard_failure(
                    request,
                    status_code=403,
                    code="INVALID_DESKTOP_ORIGIN",
                    message="The desktop Agent origin is invalid.",
                    legacy_detail="Invalid desktop origin",
                )

        elif request.method in _STATE_CHANGING_METHODS:
            boundary_failure = _local_write_boundary_failure(request)
            if boundary_failure is not None:
                return boundary_failure

        response = await call_next(request)
        if desktop_session_secret is not None:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' blob: data:; "
                "media-src 'self' blob: data:; "
                "connect-src 'self'; "
                "worker-src 'self' blob:; "
                "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
            )
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.middleware("http")
    async def _no_cache_ui_assets(request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response
