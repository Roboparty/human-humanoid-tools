"""FastAPI application assembly and launch entry points."""

from __future__ import annotations

__all__ = [
    "create_app",
    "effective_job_admission_settings",
    "run_desktop_sidecar",
    "run_web",
]


def __getattr__(name: str):
    """Load the requested server entry point without importing every runtime."""

    if name == "create_app":
        from hhtools.web.server.factory import create_app

        return create_app
    if name == "effective_job_admission_settings":
        from hhtools.application.settings import effective_job_admission_settings

        return effective_job_admission_settings
    if name in {"run_desktop_sidecar", "run_web"}:
        from hhtools.web.server.launch import run_desktop_sidecar, run_web

        return {
            "run_desktop_sidecar": run_desktop_sidecar,
            "run_web": run_web,
        }[name]
    raise AttributeError(name)
