"""Canonical registry for the public Agent JSON Schema surface.

Schema export and MCP resource discovery both use this registry so a new
contract cannot silently appear in one transport but not the other.
"""

from __future__ import annotations

from types import MappingProxyType

from pydantic import BaseModel

from .artifact_exports import ArtifactExportReceipt
from .artifacts import EvaluationReport, FailureReport, JobManifest
from .assets import (
    AssetBundle,
    AssetInspection,
    AssetRegistrationRequest,
    AssetSearchResponse,
    AvailableAssetCatalogRequest,
    AvailableAssetCatalogResponse,
)
from .batch import BatchPlan, BatchPreflightRequest, BatchPreflightResponse, BatchReport
from .calibration import (
    CalibrationCandidate,
    CalibrationPreview,
    CalibrationPreviewRequest,
    CalibrationProposalRequest,
    CalibrationProposalResponse,
    CalibrationSaveReceipt,
    CalibrationSaveRequest,
    CalibrationStatusRequest,
    CalibrationStatusResponse,
    CalibrationValidationReport,
    CalibrationValidationRequest,
    R2RCalibrationCandidate,
    R2RCalibrationPreview,
    R2RCalibrationPreviewRequest,
    R2RCalibrationProposalRequest,
    R2RCalibrationProposalResponse,
    R2RCalibrationSaveReceipt,
    R2RCalibrationSaveRequest,
    R2RCalibrationStatusRequest,
    R2RCalibrationStatusResponse,
    R2RCalibrationValidationRequest,
)
from .capabilities import CapabilityResponse, RobotListResponse
from .common import ApiError
from .job_spec import JobSpecV2
from .jobs import (
    AgentJobView,
    ArtifactDescriptor,
    ArtifactListResponse,
    JobLookupRequest,
    JobRetryRequest,
    JobStartRequest,
)
from .migration import (
    LegacyJobUpgradeRequest,
    LegacyJobUpgradeResponse,
    LegacyMigrationReceipt,
)
from .preflight import (
    PreflightResponse,
    R2RPlan,
    R2RPreflightRequest,
    R2RPreflightResponse,
    RetargetPreflightRequest,
)

PUBLIC_AGENT_SCHEMAS: MappingProxyType[str, type[BaseModel]] = MappingProxyType(
    {
        "agent-job-view": AgentJobView,
        "api-error": ApiError,
        "artifact": ArtifactDescriptor,
        "artifact-export-receipt": ArtifactExportReceipt,
        "artifact-list-response": ArtifactListResponse,
        "asset-bundle": AssetBundle,
        "asset-inspection": AssetInspection,
        "asset-registration-request": AssetRegistrationRequest,
        "asset-search-response": AssetSearchResponse,
        "available-asset-catalog-request": AvailableAssetCatalogRequest,
        "available-asset-catalog-response": AvailableAssetCatalogResponse,
        "batch-plan": BatchPlan,
        "batch-preflight-request": BatchPreflightRequest,
        "batch-preflight-response": BatchPreflightResponse,
        "batch-report": BatchReport,
        "calibration-candidate": CalibrationCandidate,
        "calibration-preview": CalibrationPreview,
        "calibration-preview-request": CalibrationPreviewRequest,
        "calibration-proposal-request": CalibrationProposalRequest,
        "calibration-proposal-response": CalibrationProposalResponse,
        "calibration-save-receipt": CalibrationSaveReceipt,
        "calibration-save-request": CalibrationSaveRequest,
        "calibration-status-request": CalibrationStatusRequest,
        "calibration-status-response": CalibrationStatusResponse,
        "calibration-validation-report": CalibrationValidationReport,
        "calibration-validation-request": CalibrationValidationRequest,
        "r2r-calibration-candidate": R2RCalibrationCandidate,
        "r2r-calibration-preview": R2RCalibrationPreview,
        "r2r-calibration-preview-request": R2RCalibrationPreviewRequest,
        "r2r-calibration-proposal-request": R2RCalibrationProposalRequest,
        "r2r-calibration-proposal-response": R2RCalibrationProposalResponse,
        "r2r-calibration-save-receipt": R2RCalibrationSaveReceipt,
        "r2r-calibration-save-request": R2RCalibrationSaveRequest,
        "r2r-calibration-status-request": R2RCalibrationStatusRequest,
        "r2r-calibration-status-response": R2RCalibrationStatusResponse,
        "r2r-calibration-validation-request": R2RCalibrationValidationRequest,
        "capabilities": CapabilityResponse,
        "evaluation-report": EvaluationReport,
        "failure-report": FailureReport,
        "job-manifest": JobManifest,
        "job-lookup-request": JobLookupRequest,
        "job-retry-request": JobRetryRequest,
        "job-start-request": JobStartRequest,
        "job-spec-v2": JobSpecV2,
        "legacy-job-upgrade-request": LegacyJobUpgradeRequest,
        "legacy-job-upgrade-response": LegacyJobUpgradeResponse,
        "legacy-migration-receipt": LegacyMigrationReceipt,
        "preflight-response": PreflightResponse,
        "r2r-plan": R2RPlan,
        "r2r-preflight-request": R2RPreflightRequest,
        "r2r-preflight-response": R2RPreflightResponse,
        "retarget-preflight-request": RetargetPreflightRequest,
        "robot-list-response": RobotListResponse,
    }
)


__all__ = ["PUBLIC_AGENT_SCHEMAS"]
