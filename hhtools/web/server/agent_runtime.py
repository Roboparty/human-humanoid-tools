"""HTTP adapters and error boundaries for application-owned Agent services."""

from __future__ import annotations


def register_agent_error_handler(app) -> None:
    """Keep router-level Agent 404/405 responses on the versioned error wire."""
    from fastapi.exception_handlers import http_exception_handler
    from starlette.exceptions import HTTPException as StarletteHTTPException
    from starlette.routing import Match

    from hhtools.agent.boundary import agent_error_response, is_agent_path

    @app.exception_handler(StarletteHTTPException)
    async def _agent_http_exception_handler(request, exc):  # type: ignore[no-untyped-def]
        if is_agent_path(request.url.path) and exc.status_code in {404, 405}:
            if exc.status_code == 404:
                return agent_error_response(
                    status_code=404,
                    code="ENDPOINT_NOT_FOUND",
                    message="The requested Agent endpoint does not exist.",
                )
            method_headers = dict(exc.headers or {})
            if not any(name.casefold() == "allow" for name in method_headers):
                allowed_methods: set[str] = set()
                for route in request.app.routes:
                    match, _child_scope = route.matches(request.scope)
                    if match is Match.PARTIAL:
                        allowed_methods.update(getattr(route, "methods", set()) or set())
                if allowed_methods:
                    method_headers["Allow"] = ", ".join(sorted(allowed_methods))
            return agent_error_response(
                status_code=405,
                code="METHOD_NOT_ALLOWED",
                message="The HTTP method is not supported by this Agent endpoint.",
                headers=method_headers,
            )
        return await http_exception_handler(request, exc)


def install_agent_boundary(app) -> None:
    """Install the Agent guard last so it is the outermost user middleware."""
    from hhtools.agent.boundary import AgentBoundaryMiddleware

    app.add_middleware(AgentBoundaryMiddleware)
