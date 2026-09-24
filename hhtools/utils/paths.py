"""Platform-aware cache and config paths for hhtools."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir, user_data_dir

HHTOOLS_CACHE_ENV = "HHTOOLS_CACHE_DIR"
HHTOOLS_ROBOT_DIR_ENV = "HHTOOLS_ROBOT_DIR"
HHTOOLS_JOB_HISTORY_DIR_ENV = "HHTOOLS_JOB_HISTORY_DIR"
HHTOOLS_WEB_SETTINGS_PATH_ENV = "HHTOOLS_WEB_SETTINGS_PATH"
HHTOOLS_MOTION_LIBRARY_ROOT_ENV = "HHTOOLS_MOTION_LIBRARY_ROOT"
HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH_ENV = "HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH"


def hhtools_cache_dir() -> Path:
    """Return (and create) the hhtools per-user cache directory.

    Honours the ``HHTOOLS_CACHE_DIR`` environment variable when set; otherwise falls back to the
    platform-standard user cache directory (``~/.cache/hhtools`` on Linux, ``%LOCALAPPDATA%``
    on Windows, ``~/Library/Caches/hhtools`` on macOS).
    """
    override = os.environ.get(HHTOOLS_CACHE_ENV)
    p = Path(override) if override else Path(user_cache_dir("hhtools", "hhtools"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_robot_dir() -> Path:
    """Return (and create) the per-user robot preset library.

    Web UI uploads and ``hhtools robot add``-style user installs land here so
    they survive server restarts.  Honours ``HHTOOLS_ROBOT_DIR`` when set;
    otherwise ``$XDG_CONFIG_HOME/hhtools/robots`` (typically
    ``~/.config/hhtools/robots`` on Linux).
    """
    override = os.environ.get(HHTOOLS_ROBOT_DIR_ENV)
    if override:
        p = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        user_cfg = Path(xdg).expanduser() if xdg else Path.home() / ".config"
        p = user_cfg / "hhtools" / "robots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_calibration_overlay_dir(
    robot_component: str,
    *,
    user_root: str | Path | None = None,
) -> Path:
    """Return a separate mutable layer beside the registered robot bundles.

    Callers supply their validated/encoded robot directory component. Overlay
    directories may not be symlinks: even a link within the library could point
    back into a registered bundle and invalidate its content identity on save.
    This read-only helper does not create the layer.
    """

    if (
        not robot_component
        or robot_component in {".", ".."}
        or any(character in robot_component for character in "/\\:")
    ):
        raise ValueError("unsafe calibration robot directory")
    root = (Path(user_root).expanduser() if user_root is not None else user_robot_dir()).resolve(
        strict=False
    )
    overlays = root / ".calibration-overlays"
    directory = overlays / robot_component
    if overlays.is_symlink() or directory.is_symlink():
        raise ValueError("calibration overlay directories must not be symlinks")
    return directory


def user_job_history_dir() -> Path:
    """Return the persistent per-user Web job-history directory.

    Job records are application data rather than an ephemeral compute cache. Tests and
    portable installations can redirect the directory with ``HHTOOLS_JOB_HISTORY_DIR``;
    ``XDG_STATE_HOME`` and ``XDG_CONFIG_HOME`` are also honoured before the platform
    default so isolated environments never write into the real user profile.
    """
    override = os.environ.get(HHTOOLS_JOB_HISTORY_DIR_ENV)
    if override:
        p = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_STATE_HOME") or os.environ.get("XDG_CONFIG_HOME")
        p = (
            Path(xdg).expanduser() / "hhtools" / "jobs"
            if xdg
            else Path(user_data_dir("hhtools", "hhtools")) / "jobs"
        )
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_web_settings_path() -> Path:
    """Return the cross-platform file used for persistent Web service settings."""

    override = os.environ.get(HHTOOLS_WEB_SETTINGS_PATH_ENV)
    if override:
        return Path(override).expanduser()
    return Path(user_config_dir("hhtools", "hhtools")) / "web-settings.json"


def user_motion_library_root() -> Path:
    """Return the configured or platform-standard Motion Library directory.

    ``HHTOOLS_MOTION_LIBRARY_ROOT`` is an explicit process-level override.  An
    XDG configuration root keeps existing Linux deployments and isolated test
    environments on their historical ``$XDG_CONFIG_HOME/hhtools/motions``
    path.  On hosts without XDG, an already populated legacy
    ``~/.config/hhtools/motions`` directory wins so an upgrade never makes an
    existing library appear to vanish.  New installations use the platform
    data directory (LocalAppData on Windows, Application Support on macOS, and
    the XDG data directory on Linux).

    The directory is not created here.  Callers choosing a managed storage
    root must validate/adopt it before performing writes.
    """

    override = os.environ.get(HHTOOLS_MOTION_LIBRARY_ROOT_ENV)
    if override:
        return Path(override).expanduser()

    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config).expanduser() / "hhtools" / "motions"

    legacy = Path.home() / ".config" / "hhtools" / "motions"
    if legacy.exists():
        return legacy
    return Path(user_data_dir("hhtools", "hhtools")) / "motions"


def user_motion_library_settings_path() -> Path:
    """Return the JSON file used for the persistent Motion Library setting."""

    override = os.environ.get(HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH_ENV)
    if override:
        return Path(override).expanduser()
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config).expanduser() / "hhtools" / "motion-library-settings.json"
    return Path(user_config_dir("hhtools", "hhtools")) / "motion-library-settings.json"


__all__ = [
    "HHTOOLS_CACHE_ENV",
    "HHTOOLS_JOB_HISTORY_DIR_ENV",
    "HHTOOLS_MOTION_LIBRARY_ROOT_ENV",
    "HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH_ENV",
    "HHTOOLS_ROBOT_DIR_ENV",
    "HHTOOLS_WEB_SETTINGS_PATH_ENV",
    "hhtools_cache_dir",
    "user_calibration_overlay_dir",
    "user_job_history_dir",
    "user_motion_library_root",
    "user_motion_library_settings_path",
    "user_robot_dir",
    "user_web_settings_path",
]
