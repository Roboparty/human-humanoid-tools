"""Startup checks for the optional WebUI runtime dependencies."""

from __future__ import annotations

from importlib.util import find_spec

_WEB_RUNTIME_IMPORTS: tuple[tuple[str, str], ...] = (
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("starlette", "starlette"),
    ("python-multipart", "multipart"),
)


class MissingWebDependenciesError(RuntimeError):
    """Raised when the browser UI or desktop sidecar cannot start."""

    def __init__(self, missing: tuple[str, ...]) -> None:
        self.missing = missing
        names = ", ".join(missing)
        super().__init__(
            "Cannot start the hhtools WebUI because required Python packages are "
            f"missing: {names}.\n"
            "Install the WebUI dependencies from the repository root with:\n"
            "    uv sync --locked --extra web\n"
            "For WebUI retargeting support, use:\n"
            "    uv sync --locked --extra web --extra retarget\n"
            "Then start it again with:\n"
            "    uv run hhtools web"
        )


def missing_web_runtime_dependencies() -> tuple[str, ...]:
    """Return install-distribution names missing from the active interpreter."""

    missing: list[str] = []
    for distribution, import_name in _WEB_RUNTIME_IMPORTS:
        try:
            available = find_spec(import_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            missing.append(distribution)
    return tuple(missing)


def require_web_runtime_dependencies() -> None:
    """Fail before server construction with an actionable installation message."""

    missing = missing_web_runtime_dependencies()
    if missing:
        raise MissingWebDependenciesError(missing)


__all__ = [
    "MissingWebDependenciesError",
    "missing_web_runtime_dependencies",
    "require_web_runtime_dependencies",
]
