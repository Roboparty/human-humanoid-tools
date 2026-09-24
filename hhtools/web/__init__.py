"""hhtools web — Apple-styled HTML/three.js front-end + FastAPI backend.

This package owns the shared browser/Electron renderer backend. The browser
handles all 3D rendering (three.js) and interaction; the FastAPI backend
(:mod:`hhtools.web.server`) re-uses the existing ``hhtools`` pipeline for the
heavy lifting (motion IO, URDF loading, calibration, retargeting, export) that
cannot run in a browser.

Launch via the CLI::

    hhtools web            # or: uv run hhtools web

The static front-end lives under :mod:`hhtools.web.static`.
"""

from __future__ import annotations

__all__ = ["run_web", "create_app"]


def __getattr__(name: str):  # pragma: no cover - thin lazy import shim
    if name in ("run_web", "create_app"):
        from hhtools.web.server import create_app, run_web

        return {"run_web": run_web, "create_app": create_app}[name]
    raise AttributeError(name)
