"""FastAPI backend for the hhtools web UI.

Single-user, localhost-first.  All heavy lifting (motion IO, URDF loading,
calibration, retargeting) re-uses the existing ``hhtools`` pipeline; the
browser only renders and drives interaction.

Run via ``hhtools web`` (see :mod:`hhtools.cli.web`) or::

    uv run hhtools web
"""

from __future__ import annotations

import logging
from pathlib import Path

from hhtools.application.runtime import (
    ApplicationPaths,
    ApplicationRuntime,
    build_application_runtime,
)
from hhtools.application.settings import (
    _DEFAULT_JOB_TTL_SECONDS,
    _DEFAULT_MAX_QUEUED_JOBS,
    _DEFAULT_MAX_RETAINED_JOBS,
    _DEFAULT_MAX_RUNNING_JOBS,
    _DEFAULT_MAX_UPLOAD_FILE_BYTES,
    _DEFAULT_MAX_UPLOAD_FILES,
    _DEFAULT_MAX_UPLOAD_REQUEST_BYTES,
    UI_BUILD_ID,
)
from hhtools.web.server.job_runtime import WebJobRuntime
from hhtools.web.server.paths import STATIC_ROOT
from hhtools.web.server.upload_runtime import UploadStore

_log = logging.getLogger(__name__)


def _mount_web_app(
    *,
    source_root: Path,
    save_dir: Path,
    cache_dir: Path | None = None,
    desktop_session_secret: str | None = None,
    desktop_allowed_host: str | None = None,
    desktop_allowed_origin: str | None = None,
    max_upload_files: int = _DEFAULT_MAX_UPLOAD_FILES,
    max_upload_file_bytes: int = _DEFAULT_MAX_UPLOAD_FILE_BYTES,
    max_upload_request_bytes: int = _DEFAULT_MAX_UPLOAD_REQUEST_BYTES,
    max_running_jobs: int = _DEFAULT_MAX_RUNNING_JOBS,
    max_queued_jobs: int = _DEFAULT_MAX_QUEUED_JOBS,
    max_batch_items: int = 0,
    max_batch_total_frames: int = 0,
    max_retained_jobs: int = _DEFAULT_MAX_RETAINED_JOBS,
    job_ttl_seconds: float = _DEFAULT_JOB_TTL_SECONDS,
    job_history_dir: Path | None = None,
    job_settings_path: Path | None = None,
    agent_mcp_available: bool = False,
    agent_rest_available: bool = True,
    agent_json_cli_available: bool = True,
    runtime: ApplicationRuntime,
):
    """Mount HTTP adapters onto an application-owned runtime."""
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    static_dir = STATIC_ROOT

    positive_limits = {
        "max_upload_files": max_upload_files,
        "max_upload_file_bytes": max_upload_file_bytes,
        "max_upload_request_bytes": max_upload_request_bytes,
        "max_retained_jobs": max_retained_jobs,
    }
    invalid_limits = [name for name, value in positive_limits.items() if int(value) <= 0]
    if float(job_ttl_seconds) <= 0:
        invalid_limits.append("job_ttl_seconds")
    if invalid_limits:
        names = ", ".join(invalid_limits)
        raise ValueError(f"resource limits must be positive: {names}")
    non_negative_limits = {
        "max_running_jobs": max_running_jobs,
        "max_queued_jobs": max_queued_jobs,
        "max_batch_items": max_batch_items,
        "max_batch_total_frames": max_batch_total_frames,
    }
    invalid_limits = [name for name, value in non_negative_limits.items() if int(value) < 0]
    if invalid_limits:
        names = ", ".join(invalid_limits)
        raise ValueError(f"job and batch limits must be non-negative: {names}")

    state = runtime.state
    scheduler = runtime.scheduler
    job_settings_store = runtime.job_settings_store
    motion_library_publish_lock = runtime.motion_library_publish_lock
    job_settings_update_lock = runtime.job_settings_update_lock
    app = FastAPI(title="hhtools web", version="0.1", lifespan=runtime.lifecycle.lifespan)
    app.state.application_runtime = runtime
    app.state.session_state = state
    app.state.job_scheduler = scheduler
    app.state.job_settings_store = job_settings_store
    app.state.agent_runtime_lease = runtime.lifecycle.agent_lease
    for name, service in vars(runtime.services).items():
        setattr(app.state, name, service)

    uploads = UploadStore(
        max_files=max_upload_files,
        max_file_bytes=max_upload_file_bytes,
        max_request_bytes=max_upload_request_bytes,
    )

    jobs = WebJobRuntime(
        state=state,
        scheduler=scheduler,
        max_retained_jobs=max_retained_jobs,
        job_ttl_seconds=job_ttl_seconds,
    )

    motion_library_settings_store = runtime.motion_library_settings_store
    app.state.motion_library_settings_store = motion_library_settings_store
    from hhtools.agent.api import router as agent_router
    from hhtools.web.server.agent_runtime import (
        install_agent_boundary,
        register_agent_error_handler,
    )

    register_agent_error_handler(app)
    app.include_router(agent_router)

    from hhtools.web.server.routes.batch import register_batch_routes
    from hhtools.web.server.routes.dataset import register_dataset_routes
    from hhtools.web.server.routes.h2r import register_h2r_routes
    from hhtools.web.server.routes.jobs import register_job_routes
    from hhtools.web.server.routes.library import register_library_routes
    from hhtools.web.server.routes.motion import register_motion_routes
    from hhtools.web.server.routes.r2r import register_r2r_routes
    from hhtools.web.server.routes.robot import register_robot_routes
    from hhtools.web.server.routes.shell import register_shell_routes
    from hhtools.web.server.routes.system import register_system_routes

    register_shell_routes(
        app,
        static_dir=static_dir,
        ui_build_id=UI_BUILD_ID,
        max_upload_request_bytes=max_upload_request_bytes,
        desktop_session_secret=desktop_session_secret,
        desktop_allowed_host=desktop_allowed_host,
        desktop_allowed_origin=desktop_allowed_origin,
    )
    register_system_routes(
        app,
        state=state,
        static_dir=static_dir,
        ui_build_id=UI_BUILD_ID,
        scheduler=scheduler,
        batch_limit_policy=runtime.services.agent_batch_limit_policy,
        jobs=jobs,
        job_settings_store=job_settings_store,
        job_settings_update_lock=job_settings_update_lock,
        motion_library_settings_store=motion_library_settings_store,
        motion_library_publish_lock=motion_library_publish_lock,
    )
    register_library_routes(
        app,
        state=state,
        motion_library_publish_lock=motion_library_publish_lock,
    )
    register_dataset_routes(app, state=state, jobs=jobs, uploads=uploads)
    motion_operations = register_motion_routes(
        app,
        state=state,
        jobs=jobs,
        uploads=uploads,
        motion_library_publish_lock=motion_library_publish_lock,
    )
    robot_operations = register_robot_routes(app, state=state, uploads=uploads)
    h2r_operations = register_h2r_routes(app, state=state, jobs=jobs)
    batch_operations = register_batch_routes(app, state=state, jobs=jobs)
    register_job_routes(
        app,
        state=state,
        jobs=jobs,
        motion_operations=motion_operations,
        robot_operations=robot_operations,
        h2r_operations=h2r_operations,
        batch_operations=batch_operations,
    )
    register_r2r_routes(app, state=state, jobs=jobs, uploads=uploads)

    # ----------------------------------------------------------------- static

    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    # Added last so it is the outermost user middleware in Starlette's stack:
    # malformed or oversized Agent requests are rejected before desktop/UI
    # middleware, routing, JSON parsing, or any service is reached.
    install_agent_boundary(app)

    return app


def create_app(
    *,
    source_root: Path,
    save_dir: Path,
    cache_dir: Path | None = None,
    desktop_session_secret: str | None = None,
    desktop_allowed_host: str | None = None,
    desktop_allowed_origin: str | None = None,
    max_upload_files: int = _DEFAULT_MAX_UPLOAD_FILES,
    max_upload_file_bytes: int = _DEFAULT_MAX_UPLOAD_FILE_BYTES,
    max_upload_request_bytes: int = _DEFAULT_MAX_UPLOAD_REQUEST_BYTES,
    max_running_jobs: int = _DEFAULT_MAX_RUNNING_JOBS,
    max_queued_jobs: int = _DEFAULT_MAX_QUEUED_JOBS,
    max_batch_items: int = 0,
    max_batch_total_frames: int = 0,
    max_retained_jobs: int = _DEFAULT_MAX_RETAINED_JOBS,
    job_ttl_seconds: float = _DEFAULT_JOB_TTL_SECONDS,
    job_history_dir: Path | None = None,
    job_settings_path: Path | None = None,
    agent_mcp_available: bool = False,
    agent_rest_available: bool = True,
    agent_json_cli_available: bool = True,
):
    """Build one local runtime with exclusive ownership of its Agent jobs.

    Job recovery and GPU scheduling are process-local.  The operating-system
    lease is therefore acquired before any Agent store or ``JobManager`` is
    constructed, and is transferred to the application lifespan on success.
    """

    runtime = build_application_runtime(
        ApplicationPaths(
            source_root=source_root,
            save_dir=save_dir,
            cache_dir=cache_dir,
            job_history_dir=job_history_dir,
            job_settings_path=job_settings_path,
        ),
        max_running_jobs=max_running_jobs,
        max_queued_jobs=max_queued_jobs,
        max_batch_items=max_batch_items,
        max_batch_total_frames=max_batch_total_frames,
        max_retained_jobs=max_retained_jobs,
        agent_mcp_available=agent_mcp_available,
        agent_rest_available=agent_rest_available,
        agent_json_cli_available=agent_json_cli_available,
    )
    try:
        return _mount_web_app(
            source_root=source_root,
            save_dir=save_dir,
            cache_dir=cache_dir,
            desktop_session_secret=desktop_session_secret,
            desktop_allowed_host=desktop_allowed_host,
            desktop_allowed_origin=desktop_allowed_origin,
            max_upload_files=max_upload_files,
            max_upload_file_bytes=max_upload_file_bytes,
            max_upload_request_bytes=max_upload_request_bytes,
            max_running_jobs=max_running_jobs,
            max_queued_jobs=max_queued_jobs,
            max_batch_items=max_batch_items,
            max_batch_total_frames=max_batch_total_frames,
            max_retained_jobs=max_retained_jobs,
            job_ttl_seconds=job_ttl_seconds,
            job_history_dir=job_history_dir,
            job_settings_path=job_settings_path,
            agent_mcp_available=agent_mcp_available,
            agent_rest_available=agent_rest_available,
            agent_json_cli_available=agent_json_cli_available,
            runtime=runtime,
        )
    except BaseException:
        runtime.scheduler.shutdown(wait=True)
        runtime.lifecycle._cleanup_once()
        runtime.lifecycle.agent_lease.release()
        raise
