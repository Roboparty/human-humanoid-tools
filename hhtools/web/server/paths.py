"""Stable filesystem locations for the Web server package."""

from __future__ import annotations

from pathlib import Path

WEB_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEB_PACKAGE_ROOT.parent.parent
STATIC_ROOT = WEB_PACKAGE_ROOT / "static"
WORKSPACE_ROBOT_ROOT = PROJECT_ROOT / "configs" / "robots"
