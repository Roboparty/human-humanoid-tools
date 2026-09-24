"""Transport-neutral application services for HHTools clients.

The Web UI, JSON CLI, REST API, and MCP adapter must call this layer rather
than importing one another.  Solver and calibration algorithms stay in their
existing modules; services only discover capabilities and orchestrate them.
"""

from .artifact_exports import (
    AGENT_EXPORT_ROOT_ID,
    ArtifactExportError,
    ArtifactExportService,
)
from .artifacts import ArtifactStore, ArtifactStoreError, StoredArtifact
from .asset_service import AgentAssetService
from .assets import AssetRegistry, AssetServiceError
from .available_assets import (
    MAX_AVAILABLE_ASSET_CANDIDATES,
    MAX_AVAILABLE_ASSET_SCAN_ENTRIES,
    AvailableAssetCandidate,
    AvailableAssetCatalogLimitError,
    AvailableAssetCatalogService,
    AvailableAssetProvider,
    classify_catalog_robot_trajectory,
    is_catalog_motion_sidecar,
    iter_bounded_catalog_files,
    require_bounded_catalog_root,
)
from .batch_limits import BatchLimitPolicy, BatchLimitSnapshot
from .batch_preflight import BatchPreflightService
from .batch_retarget import BatchRetargetService, ExecutionPlanService
from .calibration import CalibrationService, CalibrationServiceError
from .calibration_candidates import CalibrationCandidateStore
from .capabilities import CapabilitiesService
from .job_store import JobStore, JobStoreError, StoredJob, compute_request_fingerprint
from .jobs import (
    JobCancelledError,
    JobExecutionContext,
    JobExecutionError,
    JobExecutionResult,
    JobExecutor,
    JobManager,
    JobManagerError,
)
from .legacy_job_upgrade import (
    DynamicRootLocator,
    LegacyJobUpgradeError,
    LegacyJobUpgradeResult,
    LegacyJobUpgradeService,
    LegacyMigrationReceipt,
)
from .plans import PlanStore, PlanStoreError, compute_plan_id
from .preflight import PreflightService
from .r2r_asset_inspection import R2RTrajectoryInspector
from .r2r_calibration import R2RCalibrationService
from .r2r_preflight import R2RPreflightService
from .r2r_retarget import R2RRetargetService, WorkflowRetargetService
from .retarget import RetargetService, RetargetServiceError
from .runtime_lease import AgentRuntimeLease, RuntimeLeaseError

__all__ = [
    "AGENT_EXPORT_ROOT_ID",
    "AgentAssetService",
    "AgentRuntimeLease",
    "AssetRegistry",
    "AssetServiceError",
    "ArtifactStore",
    "ArtifactStoreError",
    "ArtifactExportError",
    "ArtifactExportService",
    "AvailableAssetCandidate",
    "AvailableAssetCatalogLimitError",
    "AvailableAssetCatalogService",
    "AvailableAssetProvider",
    "BatchPreflightService",
    "BatchLimitPolicy",
    "BatchLimitSnapshot",
    "BatchRetargetService",
    "CalibrationCandidateStore",
    "CalibrationService",
    "CalibrationServiceError",
    "CapabilitiesService",
    "JobCancelledError",
    "JobExecutionContext",
    "JobExecutionError",
    "JobExecutionResult",
    "JobExecutor",
    "JobManager",
    "JobManagerError",
    "JobStore",
    "JobStoreError",
    "DynamicRootLocator",
    "ExecutionPlanService",
    "LegacyJobUpgradeError",
    "LegacyJobUpgradeResult",
    "LegacyJobUpgradeService",
    "LegacyMigrationReceipt",
    "PlanStore",
    "PlanStoreError",
    "PreflightService",
    "R2RPreflightService",
    "R2RCalibrationService",
    "R2RRetargetService",
    "R2RTrajectoryInspector",
    "RetargetService",
    "RetargetServiceError",
    "RuntimeLeaseError",
    "WorkflowRetargetService",
    "MAX_AVAILABLE_ASSET_CANDIDATES",
    "MAX_AVAILABLE_ASSET_SCAN_ENTRIES",
    "StoredArtifact",
    "StoredJob",
    "compute_plan_id",
    "compute_request_fingerprint",
    "classify_catalog_robot_trajectory",
    "is_catalog_motion_sidecar",
    "iter_bounded_catalog_files",
    "require_bounded_catalog_root",
]
