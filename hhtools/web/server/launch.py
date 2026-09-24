"""Process launch entry points for Web and desktop-sidecar hosts."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from hhtools.application.settings import UI_BUILD_ID, effective_job_admission_settings
from hhtools.web.dependencies import require_web_runtime_dependencies
from hhtools.web.server.factory import create_app
from hhtools.web.server.paths import STATIC_ROOT

_log = logging.getLogger(__name__)


def run_web(
    *,
    source_root: Path,
    save_dir: Path,
    cache_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8009,
    max_running_jobs: int | None = None,
    max_queued_jobs: int | None = None,
    max_batch_items: int | None = None,
    max_batch_total_frames: int | None = None,
    job_settings_path: Path | None = None,
) -> None:
    """Launch the uvicorn server (blocking)."""
    require_web_runtime_dependencies()
    import uvicorn

    job_settings, resolved_settings_path = effective_job_admission_settings(
        max_running_jobs=max_running_jobs,
        max_queued_jobs=max_queued_jobs,
        max_batch_items=max_batch_items,
        max_batch_total_frames=max_batch_total_frames,
        job_settings_path=job_settings_path,
    )
    app = create_app(
        source_root=source_root,
        save_dir=save_dir,
        cache_dir=cache_dir,
        max_running_jobs=job_settings.max_running_jobs,
        max_queued_jobs=job_settings.max_queued_jobs,
        max_batch_items=job_settings.max_batch_items,
        max_batch_total_frames=job_settings.max_batch_total_frames,
        job_settings_path=resolved_settings_path,
    )
    url = f"http://{host}:{port}"
    static_dir = STATIC_ROOT
    print(f"\n  hhtools web  →  {url}")
    print(f"  UI build     →  {UI_BUILD_ID}")
    print(f"  static dir   →  {static_dir.resolve()}")
    print(
        "  jobs         →  "
        + (
            "unlimited concurrency"
            if job_settings.max_running_jobs == 0
            else f"{job_settings.max_running_jobs} running, "
            + (
                "unlimited waiting"
                if job_settings.max_queued_jobs == 0
                else f"{job_settings.max_queued_jobs} waiting"
            )
        )
    )
    print(
        "  侧栏应为 3 项（含「机器人 · Retarget」）；舞台左上角有「骨架|身体|机器人」。"
        "\n  git pull 后请在本仓库执行 uv sync 并用 uv run hhtools web 重启（勿用全局旧包）。"
        "\n  若界面异常：确认终端 UI build 与浏览器地址栏端口一致，再 Ctrl+Shift+R。"
        "\n  Retarget 需：uv sync --extra web --extra retarget + NVIDIA newton 包。\n"
    )
    try:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    except Exception:
        pass
    uvicorn.run(app, host=host, port=port, log_level="info")


def run_desktop_sidecar(
    *,
    source_root: Path,
    save_dir: Path,
    cache_dir: Path | None,
    host: str,
    port: int,
    session_secret: str,
    max_running_jobs: int | None = None,
    max_queued_jobs: int | None = None,
    max_batch_items: int | None = None,
    max_batch_total_frames: int | None = None,
    job_settings_path: Path | None = None,
) -> None:
    """Run the secured localhost server without opening a browser."""
    if host != "127.0.0.1":
        raise ValueError("The desktop sidecar must bind to 127.0.0.1")
    if not session_secret:
        raise ValueError("The desktop sidecar requires a session secret")

    require_web_runtime_dependencies()
    import uvicorn

    job_settings, resolved_settings_path = effective_job_admission_settings(
        max_running_jobs=max_running_jobs,
        max_queued_jobs=max_queued_jobs,
        max_batch_items=max_batch_items,
        max_batch_total_frames=max_batch_total_frames,
        job_settings_path=job_settings_path,
    )
    allowed_host = f"{host}:{port}"
    origin = f"http://{allowed_host}"
    app = create_app(
        source_root=source_root,
        save_dir=save_dir,
        cache_dir=cache_dir,
        desktop_session_secret=session_secret,
        desktop_allowed_host=allowed_host,
        desktop_allowed_origin=origin,
        max_running_jobs=job_settings.max_running_jobs,
        max_queued_jobs=job_settings.max_queued_jobs,
        max_batch_items=job_settings.max_batch_items,
        max_batch_total_frames=job_settings.max_batch_total_frames,
        job_settings_path=resolved_settings_path,
    )
    _log.info("Starting hhtools desktop sidecar on %s", origin)
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)


__all__ = ["UI_BUILD_ID", "run_desktop_sidecar", "run_web"]
