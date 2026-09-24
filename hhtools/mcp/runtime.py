"""Stdio projection of the transport-neutral HHTools application runtime."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hhtools.application.runtime import ApplicationPaths
from hhtools.services import (
    AgentAssetService,
    ArtifactExportService,
    AvailableAssetCatalogService,
    BatchPreflightService,
    CalibrationService,
    CapabilitiesService,
    JobManager,
    PlanStore,
    PreflightService,
    R2RCalibrationService,
    R2RPreflightService,
)


@dataclass(frozen=True)
class LocalRuntimeConfig:
    """Host-only configuration used to assemble one stdio-owned runtime."""

    source_root: Path = Path("assets/motions")
    save_dir: Path = Path("assets/save_npz")
    cache_dir: Path | None = None
    max_running_jobs: int | None = None
    max_queued_jobs: int | None = None
    max_batch_items: int | None = None
    max_batch_total_frames: int | None = None
    job_settings_path: Path | None = None
    web_ui_url: str = "http://127.0.0.1:8009"
    paths: ApplicationPaths | None = None


@dataclass(frozen=True)
class AgentRuntime:
    """Only the transport-neutral services exposed to MCP handlers."""

    capabilities: CapabilitiesService
    assets: AgentAssetService
    available_assets: AvailableAssetCatalogService
    preflight: PreflightService
    r2r_preflight: R2RPreflightService
    batch_preflight: BatchPreflightService
    plans: PlanStore
    jobs: JobManager
    exports: ArtifactExportService
    calibration: CalibrationService | None = None
    r2r_calibration: R2RCalibrationService | None = None

    @classmethod
    def from_application(cls, app: Any) -> AgentRuntime:
        """Project the service surface from a fully assembled local app."""

        return cls.from_services(app.state)

    @classmethod
    def from_services(cls, services: Any) -> AgentRuntime:
        return cls(
            capabilities=services.agent_capabilities_service,
            assets=services.agent_asset_service,
            available_assets=services.agent_available_asset_catalog_service,
            preflight=services.agent_preflight_service,
            r2r_preflight=services.agent_r2r_preflight_service,
            batch_preflight=services.agent_batch_preflight_service,
            calibration=services.agent_calibration_service,
            r2r_calibration=services.agent_r2r_calibration_service,
            plans=services.agent_plan_store,
            jobs=services.agent_job_manager,
            exports=services.agent_artifact_export_service,
        )


@asynccontextmanager
async def local_agent_runtime(
    config: LocalRuntimeConfig,
) -> AsyncIterator[AgentRuntime]:
    """Create one service owner and drain its scheduler when stdio closes."""

    from dataclasses import replace

    from hhtools.application.settings import effective_job_admission_settings
    from hhtools.services.runtime_lease import AgentRuntimeLease

    paths = config.paths or ApplicationPaths(
        source_root=config.source_root,
        save_dir=config.save_dir,
        cache_dir=config.cache_dir,
        job_settings_path=config.job_settings_path,
    )
    lease = AgentRuntimeLease.acquire(Path(paths.save_dir) / ".hhtools-agent")
    try:
        # Warp prints its device banner to stdout on first initialization.
        # Acquire ownership first so a conflicting process fails before any
        # heavyweight import, then quiet Warp before application assembly.
        from hhtools.retarget.newton_basic._warp_config import (
            configure as configure_warp_cache,
        )

        configure_warp_cache(quiet=True)

        from hhtools.application.runtime import build_application_runtime

        settings, settings_path = effective_job_admission_settings(
            max_running_jobs=config.max_running_jobs,
            max_queued_jobs=config.max_queued_jobs,
            max_batch_items=config.max_batch_items,
            max_batch_total_frames=config.max_batch_total_frames,
            job_settings_path=paths.job_settings_path,
        )
        runtime = build_application_runtime(
            replace(paths, job_settings_path=settings_path),
            max_running_jobs=settings.max_running_jobs,
            max_queued_jobs=settings.max_queued_jobs,
            max_batch_items=settings.max_batch_items,
            max_batch_total_frames=settings.max_batch_total_frames,
            agent_mcp_available=True,
            agent_rest_available=False,
            agent_json_cli_available=False,
            agent_runtime_lease=lease,
        )
    except BaseException:
        lease.release()
        raise
    async with runtime.lifespan():
        yield AgentRuntime.from_services(runtime.services)


__all__ = ["AgentRuntime", "LocalRuntimeConfig", "local_agent_runtime"]
