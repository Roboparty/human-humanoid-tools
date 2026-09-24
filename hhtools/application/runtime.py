"""One service owner shared by HTTP, desktop sidecars, and stdio adapters."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from hhtools.application.lifecycle import RuntimeLifecycle
from hhtools.application.state import SessionState, _cleanup_session_state
from hhtools.services.job_scheduler import JobScheduler
from hhtools.services.job_settings import JobAdmissionSettingsStore
from hhtools.services.motion_cache import EphemeralCache
from hhtools.services.motion_library_links import ensure_motions_library, motions_library_root
from hhtools.services.motion_library_settings import MotionLibrarySettingsStore
from hhtools.services.runtime_lease import AgentRuntimeLease
from hhtools.services.web_job_history import JobHistoryStore


@dataclass(frozen=True)
class ApplicationPaths:
    source_root: Path
    save_dir: Path
    cache_dir: Path | None = None
    robot_root: Path | None = None
    motion_library_root: Path | None = None
    motion_library_settings_path: Path | None = None
    job_history_dir: Path | None = None
    job_settings_path: Path | None = None
    workspace_robot_root: Path | None = None

    @classmethod
    def isolated(cls, root: Path, *, source_root: Path | None = None) -> ApplicationPaths:
        root = Path(root)
        return cls(
            source_root=source_root or root / "source",
            save_dir=root / "save",
            cache_dir=root / "cache",
            robot_root=root / "robots",
            motion_library_root=root / "motion-library",
            motion_library_settings_path=root / "settings/motion-library.json",
            job_history_dir=root / "history",
            job_settings_path=root / "settings/jobs.json",
            workspace_robot_root=root / "workspace-robots",
        )


@dataclass
class ApplicationRuntime:
    paths: ApplicationPaths
    state: SessionState
    scheduler: JobScheduler
    lifecycle: RuntimeLifecycle
    services: SimpleNamespace
    job_settings_store: JobAdmissionSettingsStore | None
    motion_library_settings_store: MotionLibrarySettingsStore
    job_settings_update_lock: threading.Lock
    motion_library_publish_lock: threading.Lock

    @asynccontextmanager
    async def lifespan(self):
        async with self.lifecycle.lifespan(self):
            yield self


def build_application_runtime(
    paths: ApplicationPaths,
    *,
    max_running_jobs: int = 0,
    max_queued_jobs: int = 0,
    max_batch_items: int = 0,
    max_batch_total_frames: int = 0,
    max_retained_jobs: int = 64,
    agent_mcp_available: bool = False,
    agent_rest_available: bool = True,
    agent_json_cli_available: bool = True,
    agent_runtime_lease: AgentRuntimeLease | None = None,
) -> ApplicationRuntime:
    from hhtools.application.agent_services import assemble_agent_services
    from hhtools.utils.paths import user_job_history_dir, user_motion_library_settings_path

    if any(
        value < 0
        for value in (
            max_running_jobs,
            max_queued_jobs,
            max_batch_items,
            max_batch_total_frames,
        )
    ):
        raise ValueError("job and batch limits must be non-negative")
    if max_retained_jobs <= 0:
        raise ValueError("resource limits must be positive: max_retained_jobs")
    lease = agent_runtime_lease or AgentRuntimeLease.acquire(
        Path(paths.save_dir) / ".hhtools-agent"
    )
    state = None
    scheduler = None
    lifecycle = None
    try:
        robot_options = (
            {"robot_root": Path(paths.robot_root)} if paths.robot_root is not None else {}
        )
        state = SessionState(
            source_root=Path(paths.source_root),
            save_dir=Path(paths.save_dir),
            **robot_options,
        )
        state.cache = EphemeralCache.create(cache_dir=paths.cache_dir, save_dir=paths.save_dir)
        state.job_history = JobHistoryStore(
            paths.job_history_dir if paths.job_history_dir is not None else user_job_history_dir(),
            max_records=max_retained_jobs,
        )
        scheduler = JobScheduler(max_running_jobs=max_running_jobs, max_queued_jobs=max_queued_jobs)
        settings_lock = threading.Lock()
        library_lock = threading.Lock()
        settings_store = (
            JobAdmissionSettingsStore(paths.job_settings_path)
            if paths.job_settings_path is not None
            else None
        )
        library_settings = MotionLibrarySettingsStore(
            paths.motion_library_settings_path
            if paths.motion_library_settings_path is not None
            else user_motion_library_settings_path(),
        )
        if paths.motion_library_root is not None:
            library_root = ensure_motions_library(paths.motion_library_root)

            def library_provider() -> Path:
                return library_root

        elif paths.motion_library_settings_path is not None:
            from hhtools.services.motion_library_settings import effective_motion_library_root

            def library_provider() -> Path:
                return effective_motion_library_root(library_settings.load())

            ensure_motions_library(library_provider())
        else:
            library_provider = motions_library_root
            ensure_motions_library()
        services = assemble_agent_services(
            state=state,
            scheduler=scheduler,
            motion_library_root_provider=library_provider,
            workspace_robot_root=(
                paths.workspace_robot_root
                if paths.workspace_robot_root is not None
                else Path(__file__).resolve().parents[2] / "configs/robots"
            ),
            agent_mcp_available=agent_mcp_available,
            agent_rest_available=agent_rest_available,
            agent_json_cli_available=agent_json_cli_available,
            max_batch_items=max_batch_items,
            max_batch_total_frames=max_batch_total_frames,
        )
        lifecycle = RuntimeLifecycle(state, scheduler, settings_lock, lease)
        return ApplicationRuntime(
            paths,
            state,
            scheduler,
            lifecycle,
            services,
            settings_store,
            library_settings,
            settings_lock,
            library_lock,
        )
    except BaseException:
        if scheduler is not None:
            scheduler.shutdown(wait=True)
        if state is not None:
            _cleanup_session_state(state)
        lease.release()
        raise
