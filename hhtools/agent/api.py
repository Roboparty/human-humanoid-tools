"""Versioned REST adapter for HHTools' transport-neutral agent services.

This module intentionally contains no capability probing or solver logic.  It
only retrieves the service instance assembled by :func:`hhtools.web.server.create_app`
and serializes its contracts through FastAPI.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Protocol, cast

from fastapi import APIRouter, Body, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute

from hhtools.agent.artifact_response import verified_artifact_response
from hhtools.contracts import (
    AgentJobView,
    ApiError,
    ArtifactDescriptor,
    ArtifactId,
    ArtifactListResponse,
    AssetBundle,
    AssetCategory,
    AssetId,
    AssetInspection,
    AssetInspectionRequest,
    AssetKind,
    AssetRegistrationRequest,
    AssetSearchResponse,
    AvailableAssetCatalogRequest,
    AvailableAssetCatalogResponse,
    BatchPreflightRequest,
    BatchPreflightResponse,
    CalibrationPreviewRequest,
    CalibrationProposalRequest,
    CalibrationProposalResponse,
    CalibrationSaveReceipt,
    CalibrationSaveRequest,
    CalibrationStatusRequest,
    CalibrationStatusResponse,
    CalibrationValidationReport,
    CalibrationValidationRequest,
    CapabilityResponse,
    ErrorStage,
    JobLookupRequest,
    JobRetryRequest,
    JobStartRequest,
    LegacyJobUpgradeRequest,
    LegacyJobUpgradeResponse,
    PreflightResponse,
    R2RCalibrationPreviewRequest,
    R2RCalibrationProposalRequest,
    R2RCalibrationProposalResponse,
    R2RCalibrationSaveReceipt,
    R2RCalibrationSaveRequest,
    R2RCalibrationStatusRequest,
    R2RCalibrationStatusResponse,
    R2RCalibrationValidationRequest,
    R2RPreflightRequest,
    R2RPreflightResponse,
    RetargetPreflightRequest,
)
from hhtools.contracts.portability import (
    MAX_PORTABLE_DEPTH,
)
from hhtools.contracts.portability import (
    PortableJsonError as ContractPortableJsonError,
)
from hhtools.contracts.portability import (
    looks_like_host_path as contract_looks_like_host_path,
)
from hhtools.contracts.portability import (
    validate_portable_json as validate_contract_portable_json,
)
from hhtools.services.artifacts import StoredArtifact
from hhtools.services.assets import AssetServiceError
from hhtools.services.calibration import CalibrationServiceError
from hhtools.services.jobs import JobManagerError
from hhtools.services.legacy_job_upgrade import LegacyJobUpgradeError

_log = logging.getLogger(__name__)

_ERROR_STATUS_BY_CODE = {
    "ALLOWED_ROOT_UNAVAILABLE": 503,
    "AMBIGUOUS_ALLOWED_ROOT": 409,
    "ARTIFACT_HASH_MISMATCH": 409,
    "ARTIFACT_NOT_FOUND": 404,
    "ASSET_CHANGED_DURING_UPGRADE": 409,
    "ASSET_HASH_MISMATCH": 409,
    "ASSET_NOT_FOUND": 404,
    "ASSET_OUTSIDE_ALLOWED_ROOT": 403,
    "ASSET_REGISTRATION_MISMATCH": 409,
    "AVAILABLE_ASSET_CATALOG_UNAVAILABLE": 503,
    "AVAILABLE_ASSET_CATALOG_LIMIT_EXCEEDED": 422,
    "BACKEND_UNAVAILABLE": 503,
    "BUNDLE_AMBIGUOUS": 409,
    "CALIBRATION_CANDIDATE_MISMATCH": 409,
    "CALIBRATION_CANDIDATE_NOT_FOUND": 404,
    "CALIBRATION_CANDIDATE_STALE": 409,
    "INTERNAL_ERROR": 500,
    "INVALID_JOB_TRANSITION": 409,
    "INVALID_JOB_SPEC": 400,
    "INVALID_JSON": 400,
    "INVALID_PARAMETER": 400,
    "JOB_CANCEL_UNSUPPORTED": 409,
    "JOB_CONFLICT": 409,
    "JOB_NOT_FOUND": 404,
    "LEGACY_METADATA_MISMATCH": 409,
    "PLAN_CONFLICT": 409,
    "PLAN_NOT_FOUND": 404,
    "PLAN_STALE": 409,
    "QUEUE_FULL": 429,
    "ROBOT_AMBIGUOUS": 409,
    "ROBOT_NOT_FOUND": 404,
    "ROBOT_REGISTRY_UNAVAILABLE": 503,
    "RANGE_NOT_SUPPORTED": 416,
    "REFERENCE_MISMATCH": 409,
    "SCHEDULER_UNAVAILABLE": 503,
}


class _InvalidAgentJsonError(ValueError):
    """The JSON representation is ambiguous or unsafe for the public wire."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidAgentJsonError("duplicate object key")
        result[key] = value
    return result


def _reject_nonfinite_constant(_value: str) -> None:
    raise _InvalidAgentJsonError("non-finite number")


def _strict_json_int(value: str) -> int:
    if len(value) > 128:
        raise _InvalidAgentJsonError("integer token is too long")
    return int(value)


def _strict_json_float(value: str) -> float:
    if len(value) > 128:
        raise _InvalidAgentJsonError("float token is too long")
    result = float(value)
    if not math.isfinite(result):
        raise _InvalidAgentJsonError("non-finite number")
    return result


def _reject_excessive_json_depth(value: Any) -> None:
    """Reject documents deeper than the shared public-contract limit."""

    pending: list[tuple[Any, int]] = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > MAX_PORTABLE_DEPTH:
            raise _InvalidAgentJsonError("document nesting too deep")
        if isinstance(current, dict):
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)


def _strict_json_loads(payload: bytes) -> Any:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_float=_strict_json_float,
            parse_int=_strict_json_int,
            parse_constant=_reject_nonfinite_constant,
        )
        _reject_excessive_json_depth(value)
        return value
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _InvalidAgentJsonError,
        RecursionError,
    ) as exc:
        raise _InvalidAgentJsonError("invalid strict JSON") from exc


def _looks_like_host_path(
    value: str,
    *,
    allow_same_origin_ui_url: bool = False,
) -> bool:
    return contract_looks_like_host_path(
        value,
        allow_same_origin_ui_url=allow_same_origin_ui_url,
    )


def _validate_portable_json(
    value: Any,
    *,
    allow_same_origin_ui_url: bool = False,
) -> None:
    try:
        if allow_same_origin_ui_url and isinstance(value, str):
            if contract_looks_like_host_path(
                value,
                allow_same_origin_ui_url=True,
            ):
                raise ContractPortableJsonError("host path")
            return
        validate_contract_portable_json(value)
    except ContractPortableJsonError as error:
        raise _InvalidAgentJsonError(str(error)) from error


def _error_status(error: ApiError) -> int:
    status = _ERROR_STATUS_BY_CODE.get(error.code)
    if status is not None:
        return status
    return 500 if error.stage is ErrorStage.INTERNAL else 422


def _error_response(error: ApiError, *, status_code: int | None = None) -> JSONResponse:
    headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    if error.code == "QUEUE_FULL" and error.next_action is not None:
        raw_poll_ms = error.next_action.parameters.get("poll_after_ms")
        if (
            isinstance(raw_poll_ms, int | float)
            and not isinstance(raw_poll_ms, bool)
            and math.isfinite(raw_poll_ms)
            and 0 <= raw_poll_ms <= 300_000
        ):
            headers["Retry-After"] = str(max(1, math.ceil(raw_poll_ms / 1000)))
    return JSONResponse(
        status_code=status_code or _error_status(error),
        content=error.model_dump(mode="json", exclude_none=True),
        headers=headers,
    )


def _portable_response(response: Response) -> Response:
    content_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
    if content_type != "application/json" and not content_type.endswith("+json"):
        return response
    body = getattr(response, "body", None)
    if not isinstance(body, bytes | bytearray | memoryview):
        raise _InvalidAgentJsonError("JSON response is not buffered")
    value = _strict_json_loads(bytes(body))
    _validate_portable_json(value)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def _portable_or_internal_error(response: Response) -> Response:
    try:
        return _portable_response(response)
    except _InvalidAgentJsonError:
        _log.error("blocked a non-portable Agent JSON response")
        return _error_response(
            ApiError(
                code="INTERNAL_ERROR",
                message="The Agent service produced an unsafe response.",
                retryable=False,
                stage=ErrorStage.INTERNAL,
            )
        )


class _AgentRoute(APIRoute):
    """Keep expected Agent failures on the same versioned error contract."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        route_handler = super().get_route_handler()

        async def agent_route_handler(request: Request) -> Response:
            try:
                body = await request.body()
                if body:
                    try:
                        _strict_json_loads(body)
                    except _InvalidAgentJsonError:
                        return _portable_or_internal_error(
                            _error_response(
                                ApiError(
                                    code="INVALID_JSON",
                                    message=(
                                        "The Agent request body must be strict UTF-8 JSON "
                                        "without duplicate keys or non-finite numbers."
                                    ),
                                    stage=ErrorStage.REQUEST,
                                ),
                                status_code=400,
                            )
                        )
                response = await route_handler(request)
            except RequestValidationError as exc:
                issues = [
                    {
                        "location": ".".join(str(part) for part in issue.get("loc", ())),
                        "type": str(issue.get("type", "validation_error")),
                    }
                    for issue in exc.errors()
                ]
                response = _error_response(
                    ApiError(
                        code="INVALID_PARAMETER",
                        message="The Agent request does not match the versioned contract.",
                        stage=ErrorStage.REQUEST,
                        details={"issues": issues},
                    ),
                    status_code=422,
                )
            except AssetServiceError as exc:
                response = _error_response(exc.api_error)
            except (CalibrationServiceError, JobManagerError, LegacyJobUpgradeError) as exc:
                response = _error_response(exc.api_error)
            except Exception:  # noqa: BLE001 - REST must not expose internals
                _log.exception("unexpected Agent REST failure")
                response = _error_response(
                    ApiError(
                        code="INTERNAL_ERROR",
                        message="The Agent service could not complete the request.",
                        retryable=True,
                        stage=ErrorStage.INTERNAL,
                    )
                )
            return _portable_or_internal_error(response)

        return agent_route_handler


router = APIRouter(
    prefix="/api/agent/v1",
    tags=["agent"],
    route_class=_AgentRoute,
)


class _CapabilityProvider(Protocol):
    def get_capabilities(self) -> CapabilityResponse: ...


class _AssetProvider(Protocol):
    @property
    def allowed_root_ids(self) -> tuple[str, ...]: ...

    def register(self, request: AssetRegistrationRequest) -> AssetBundle: ...

    def get(self, asset_id: str) -> AssetBundle: ...

    def search(self, **filters: Any) -> AssetSearchResponse: ...

    def inspect(self, request: AssetInspectionRequest) -> AssetInspection: ...


class _AvailableAssetCatalogProvider(Protocol):
    def list_available(
        self,
        request: AvailableAssetCatalogRequest,
    ) -> AvailableAssetCatalogResponse: ...


class _PreflightProvider(Protocol):
    def preflight_retarget(
        self,
        request: RetargetPreflightRequest,
    ) -> PreflightResponse: ...


class _R2RPreflightProvider(Protocol):
    def preflight_r2r(
        self,
        request: R2RPreflightRequest,
    ) -> R2RPreflightResponse: ...


class _BatchPreflightProvider(Protocol):
    def preflight_batch(
        self,
        request: BatchPreflightRequest,
    ) -> BatchPreflightResponse: ...


class _CalibrationProvider(Protocol):
    def status(self, request: CalibrationStatusRequest) -> CalibrationStatusResponse: ...

    def propose(self, request: CalibrationProposalRequest) -> CalibrationProposalResponse: ...

    def validate(self, request: CalibrationValidationRequest) -> CalibrationValidationReport: ...

    def preview(self, request: CalibrationPreviewRequest) -> tuple[Any, bytes]: ...

    def save(self, request: CalibrationSaveRequest) -> CalibrationSaveReceipt: ...


class _R2RCalibrationProvider(Protocol):
    def status(
        self,
        request: R2RCalibrationStatusRequest,
    ) -> R2RCalibrationStatusResponse: ...

    def propose(
        self,
        request: R2RCalibrationProposalRequest,
    ) -> R2RCalibrationProposalResponse: ...

    def validate(
        self,
        request: R2RCalibrationValidationRequest,
    ) -> CalibrationValidationReport: ...

    def preview(self, request: R2RCalibrationPreviewRequest) -> tuple[Any, bytes]: ...

    def save(self, request: R2RCalibrationSaveRequest) -> R2RCalibrationSaveReceipt: ...


class _JobProvider(Protocol):
    def start_job(
        self,
        plan_id: str,
        *,
        idempotency_key: str,
        parent_job_id: str | None = None,
    ) -> AgentJobView: ...

    def start_retarget(
        self,
        plan_id: str,
        *,
        idempotency_key: str,
        parent_job_id: str | None = None,
    ) -> AgentJobView: ...

    def get_job(
        self,
        job_id: str,
        *,
        after_revision: int | None = None,
    ) -> AgentJobView: ...

    def wait_job(
        self,
        job_id: str,
        *,
        after_revision: int,
        timeout: float = 30.0,
    ) -> AgentJobView: ...

    def lookup_job(
        self,
        plan_id: str,
        *,
        idempotency_key: str,
        after_revision: int | None = None,
    ) -> AgentJobView: ...

    def cancel_job(self, job_id: str) -> AgentJobView: ...

    def retry_job(self, job_id: str, *, idempotency_key: str) -> AgentJobView: ...

    def list_artifacts(
        self,
        job_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> list[ArtifactDescriptor]: ...

    def get_artifact(
        self,
        job_id: str,
        artifact_id: str,
        *,
        verify: bool = False,
    ) -> StoredArtifact: ...


class _LegacyUpgradeProvider(Protocol):
    def upgrade(self, payload: Any) -> LegacyJobUpgradeResponse: ...


def _capabilities_service(request: Request) -> _CapabilityProvider:
    service = getattr(request.app.state, "agent_capabilities_service", None)
    if service is None or not callable(getattr(service, "get_capabilities", None)):
        raise RuntimeError("agent capabilities service is not configured")
    return cast("_CapabilityProvider", service)


def _asset_service(request: Request) -> _AssetProvider:
    service = getattr(request.app.state, "agent_asset_service", None)
    required = ("register", "get", "search", "inspect")
    if service is None or any(not callable(getattr(service, name, None)) for name in required):
        raise RuntimeError("agent asset service is not configured")
    return cast("_AssetProvider", service)


def _available_asset_catalog_service(request: Request) -> _AvailableAssetCatalogProvider:
    service = getattr(request.app.state, "agent_available_asset_catalog_service", None)
    if service is None or not callable(getattr(service, "list_available", None)):
        raise RuntimeError("agent available asset catalog service is not configured")
    return cast("_AvailableAssetCatalogProvider", service)


def _preflight_service(request: Request) -> _PreflightProvider:
    service = getattr(request.app.state, "agent_preflight_service", None)
    if service is None or not callable(getattr(service, "preflight_retarget", None)):
        raise RuntimeError("agent preflight service is not configured")
    return cast("_PreflightProvider", service)


def _r2r_preflight_service(request: Request) -> _R2RPreflightProvider:
    service = getattr(request.app.state, "agent_r2r_preflight_service", None)
    if service is None or not callable(getattr(service, "preflight_r2r", None)):
        raise RuntimeError("agent R2R preflight service is not configured")
    return cast("_R2RPreflightProvider", service)


def _batch_preflight_service(request: Request) -> _BatchPreflightProvider:
    service = getattr(request.app.state, "agent_batch_preflight_service", None)
    if service is None or not callable(getattr(service, "preflight_batch", None)):
        raise RuntimeError("agent batch preflight service is not configured")
    return cast("_BatchPreflightProvider", service)


def _calibration_service(request: Request) -> _CalibrationProvider:
    service = getattr(request.app.state, "agent_calibration_service", None)
    required = ("status", "propose", "validate", "preview", "save")
    if service is None or any(not callable(getattr(service, name, None)) for name in required):
        raise RuntimeError("agent calibration service is not configured")
    return cast("_CalibrationProvider", service)


def _r2r_calibration_service(request: Request) -> _R2RCalibrationProvider:
    service = getattr(request.app.state, "agent_r2r_calibration_service", None)
    required = ("status", "propose", "validate", "preview", "save")
    if service is None or any(not callable(getattr(service, name, None)) for name in required):
        raise RuntimeError("agent R2R calibration service is not configured")
    return cast("_R2RCalibrationProvider", service)


def _job_manager(request: Request) -> _JobProvider:
    service = getattr(request.app.state, "agent_job_manager", None)
    required = (
        "start_retarget",
        "get_job",
        "wait_job",
        "lookup_job",
        "cancel_job",
        "retry_job",
        "list_artifacts",
        "get_artifact",
    )
    if service is None or any(not callable(getattr(service, name, None)) for name in required):
        raise RuntimeError("agent job manager is not configured")
    return cast("_JobProvider", service)


def _legacy_upgrade_service(request: Request) -> _LegacyUpgradeProvider:
    service = getattr(request.app.state, "agent_legacy_job_upgrade_service", None)
    if service is None or not callable(getattr(service, "upgrade", None)):
        raise RuntimeError("agent legacy job upgrade service is not configured")
    return cast("_LegacyUpgradeProvider", service)


@router.get(
    "/capabilities",
    response_model=CapabilityResponse,
    response_model_exclude_none=True,
)
def get_capabilities(request: Request) -> CapabilityResponse:
    """Describe backends, devices, robots, formats, and live admission state."""

    return _capabilities_service(request).get_capabilities()


@router.post(
    "/assets",
    response_model=AssetBundle,
    response_model_exclude_none=True,
    status_code=201,
)
def register_asset(
    request: Request,
    registration: AssetRegistrationRequest,
) -> AssetBundle:
    """Register one content-addressed bundle below a configured server root."""

    return _asset_service(request).register(registration)


@router.get(
    "/assets",
    response_model=AssetSearchResponse,
    response_model_exclude_none=True,
)
def search_assets(
    request: Request,
    query: str | None = Query(default=None, max_length=256),
    kind: AssetKind | None = None,
    category: AssetCategory | None = None,
    dataset: str | None = Query(default=None, max_length=128),
    reference: str | None = Query(default=None, max_length=128),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> AssetSearchResponse:
    """Search immutable asset manifests using compact, bounded filters."""

    return _asset_service(request).search(
        query=query,
        kind=kind,
        category=category,
        dataset=dataset,
        reference=reference,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/assets/available",
    response_model=AvailableAssetCatalogResponse,
    response_model_exclude_none=True,
)
def list_available_assets(
    request: Request,
    root_id: str | None = Query(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
    query: str | None = Query(default=None, min_length=1, max_length=256),
    kind: AssetKind | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> AvailableAssetCatalogResponse:
    """List portable registration candidates below configured allowlisted roots."""

    catalog_request = AvailableAssetCatalogRequest(
        root_id=root_id,
        query=query,
        kind=kind,
        limit=limit,
        offset=offset,
    )
    return _available_asset_catalog_service(request).list_available(catalog_request)


@router.get(
    "/assets/{asset_id}",
    response_model=AssetBundle,
    response_model_exclude_none=True,
)
def get_asset(request: Request, asset_id: AssetId) -> AssetBundle:
    """Return one portable content manifest without exposing a host path."""

    return _asset_service(request).get(asset_id)


@router.get(
    "/assets/{asset_id}/inspect",
    response_model=AssetInspection,
    response_model_exclude_none=True,
)
def inspect_asset(
    request: Request,
    asset_id: AssetId,
    verify_hashes: bool = True,
    parse_content: bool = True,
) -> AssetInspection:
    """Validate bundle integrity and content without starting a solver job."""

    return _asset_service(request).inspect(
        AssetInspectionRequest(
            asset_id=asset_id,
            verify_hashes=verify_hashes,
            parse_content=parse_content,
        )
    )


@router.post(
    "/preflight/retarget",
    response_model=PreflightResponse,
    response_model_exclude_none=True,
)
def preflight_retarget(
    request: Request,
    preflight: RetargetPreflightRequest,
) -> PreflightResponse:
    """Resolve retarget intent without loading a solver or reserving a job."""

    return _preflight_service(request).preflight_retarget(preflight)


@router.post(
    "/preflight/r2r",
    response_model=R2RPreflightResponse,
    response_model_exclude_none=True,
)
def preflight_r2r(
    request: Request,
    preflight: R2RPreflightRequest,
) -> R2RPreflightResponse:
    """Resolve R2R intent while binding the trajectory and both robots."""

    return _r2r_preflight_service(request).preflight_r2r(preflight)


@router.post(
    "/preflight/batch",
    response_model=BatchPreflightResponse,
    response_model_exclude_none=True,
)
def preflight_batch(
    request: Request,
    preflight: BatchPreflightRequest,
) -> BatchPreflightResponse:
    """Freeze an ordered list of ready child plans into one batch."""

    return _batch_preflight_service(request).preflight_batch(preflight)


@router.post(
    "/calibrations/status",
    response_model=CalibrationStatusResponse,
    response_model_exclude_none=True,
)
def calibration_status(
    request: Request,
    calibration: CalibrationStatusRequest,
) -> CalibrationStatusResponse:
    """Inspect one exact robot/reference calibration and deterministic quality report."""

    return _calibration_service(request).status(calibration)


@router.post(
    "/calibrations/proposals",
    response_model=CalibrationProposalResponse,
    response_model_exclude_none=True,
)
def propose_calibration(
    request: Request,
    calibration: CalibrationProposalRequest,
) -> CalibrationProposalResponse:
    """Create or revise an immutable, limit-constrained calibration candidate."""

    return _calibration_service(request).propose(calibration)


@router.post(
    "/calibrations/validate",
    response_model=CalibrationValidationReport,
    response_model_exclude_none=True,
)
def validate_calibration(
    request: Request,
    calibration: CalibrationValidationRequest,
) -> CalibrationValidationReport:
    """Revalidate a stored candidate against the current content-bound robot."""

    return _calibration_service(request).validate(calibration)


@router.post("/calibrations/preview")
def preview_calibration(
    request: Request,
    calibration: CalibrationPreviewRequest,
) -> Response:
    """Render a deterministic front/side PNG for a vision-capable Agent."""

    preview, payload = _calibration_service(request).preview(calibration)
    return Response(
        content=payload,
        media_type=preview.media_type,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-HHTools-Calibration-Candidate": preview.candidate_id,
            "X-HHTools-Content-SHA256": preview.sha256,
        },
    )


@router.post(
    "/calibrations/save",
    response_model=CalibrationSaveReceipt,
    response_model_exclude_none=True,
)
def save_calibration(
    request: Request,
    calibration: CalibrationSaveRequest,
) -> CalibrationSaveReceipt:
    """Persist only a valid candidate under an explicit silent-save mode."""

    return _calibration_service(request).save(calibration)


@router.post(
    "/calibrations/r2r/status",
    response_model=R2RCalibrationStatusResponse,
    response_model_exclude_none=True,
)
def r2r_calibration_status(
    request: Request,
    calibration: R2RCalibrationStatusRequest,
) -> R2RCalibrationStatusResponse:
    """Inspect one exact source/target robot pair calibration."""

    return _r2r_calibration_service(request).status(calibration)


@router.post(
    "/calibrations/r2r/proposals",
    response_model=R2RCalibrationProposalResponse,
    response_model_exclude_none=True,
)
def propose_r2r_calibration(
    request: Request,
    calibration: R2RCalibrationProposalRequest,
) -> R2RCalibrationProposalResponse:
    """Create or revise an immutable target-pose candidate for a robot pair."""

    return _r2r_calibration_service(request).propose(calibration)


@router.post(
    "/calibrations/r2r/validate",
    response_model=CalibrationValidationReport,
    response_model_exclude_none=True,
)
def validate_r2r_calibration(
    request: Request,
    calibration: R2RCalibrationValidationRequest,
) -> CalibrationValidationReport:
    """Revalidate an R2R candidate against both current robot bundles."""

    return _r2r_calibration_service(request).validate(calibration)


@router.post("/calibrations/r2r/preview")
def preview_r2r_calibration(
    request: Request,
    calibration: R2RCalibrationPreviewRequest,
) -> Response:
    """Render source-reference and target-pose front/side overlays."""

    preview, payload = _r2r_calibration_service(request).preview(calibration)
    return Response(
        content=payload,
        media_type=preview.media_type,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-HHTools-Calibration-Candidate": preview.candidate_id,
            "X-HHTools-Content-SHA256": preview.sha256,
        },
    )


@router.post(
    "/calibrations/r2r/save",
    response_model=R2RCalibrationSaveReceipt,
    response_model_exclude_none=True,
)
def save_r2r_calibration(
    request: Request,
    calibration: R2RCalibrationSaveRequest,
) -> R2RCalibrationSaveReceipt:
    """Persist only a valid pair candidate under an explicit silent-save mode."""

    return _r2r_calibration_service(request).save(calibration)


@router.post(
    "/jobs",
    response_model=AgentJobView,
    response_model_exclude_none=True,
    status_code=202,
)
def start_retarget_job(
    request: Request,
    submission: Annotated[
        JobStartRequest,
        Body(
            openapi_examples={
                "smoke": {
                    "summary": "Submit a preflighted smoke plan",
                    "value": {
                        "schema_version": "1.0",
                        "plan_id": f"plan:sha256:{'1' * 64}",
                        "idempotency_key": "agent-smoke-001",
                    },
                }
            }
        ),
    ],
) -> AgentJobView:
    """Submit one immutable preflight plan with caller-owned idempotency."""

    jobs = _job_manager(request)
    start = getattr(jobs, "start_job", jobs.start_retarget)
    return start(
        submission.plan_id,
        idempotency_key=submission.idempotency_key,
    )


@router.post(
    "/jobs/lookup",
    response_model=AgentJobView,
    response_model_exclude_none=True,
)
def lookup_retarget_job(
    request: Request,
    lookup: Annotated[
        JobLookupRequest,
        Body(
            openapi_examples={
                "recover": {
                    "summary": "Recover one caller-owned submission",
                    "value": {
                        "schema_version": "1.0",
                        "plan_id": f"plan:sha256:{'1' * 64}",
                        "idempotency_key": "agent-smoke-001",
                        "after_revision": 4,
                    },
                }
            }
        ),
    ],
) -> AgentJobView:
    """Recover a known submission without exposing a global job listing."""

    return _job_manager(request).lookup_job(
        lookup.plan_id,
        idempotency_key=lookup.idempotency_key,
        after_revision=lookup.after_revision,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=AgentJobView,
    response_model_exclude_none=True,
)
def get_retarget_job(
    request: Request,
    job_id: str,
    after_revision: int | None = Query(default=None, ge=0),
) -> AgentJobView:
    """Return a compact job snapshot suitable for revision-aware polling."""

    return _job_manager(request).get_job(
        job_id,
        after_revision=after_revision,
    )


@router.get(
    "/jobs/{job_id}/wait",
    response_model=AgentJobView,
    response_model_exclude_none=True,
)
def wait_for_retarget_job(
    request: Request,
    job_id: str,
    after_revision: int = Query(ge=0),
    timeout: float = Query(default=30.0, ge=0.0, le=60.0, allow_inf_nan=False),
) -> AgentJobView:
    """Wait for one job revision change without cancelling the underlying job."""

    return _job_manager(request).wait_job(
        job_id,
        after_revision=after_revision,
        timeout=timeout,
    )


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=AgentJobView,
    response_model_exclude_none=True,
)
def cancel_retarget_job(request: Request, job_id: str) -> AgentJobView:
    """Persist a queued or cooperative-running cancellation request."""

    return _job_manager(request).cancel_job(job_id)


@router.post(
    "/jobs/{job_id}/retry",
    response_model=AgentJobView,
    response_model_exclude_none=True,
    status_code=202,
)
def retry_retarget_job(
    request: Request,
    job_id: str,
    retry: Annotated[
        JobRetryRequest,
        Body(
            openapi_examples={
                "retry": {
                    "summary": "Create one child attempt",
                    "value": {
                        "schema_version": "1.0",
                        "idempotency_key": "agent-retry-001",
                    },
                }
            }
        ),
    ],
) -> AgentJobView:
    """Create an idempotent child attempt for one terminal parent job."""

    return _job_manager(request).retry_job(
        job_id,
        idempotency_key=retry.idempotency_key,
    )


@router.get(
    "/jobs/{job_id}/artifacts",
    response_model=ArtifactListResponse,
    response_model_exclude_none=True,
)
def list_job_artifacts(
    request: Request,
    job_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> ArtifactListResponse:
    """List a bounded page of canonical artifacts attached to one job."""

    manager = _job_manager(request)
    # List first, then read the compact view.  Canonical membership only grows,
    # so a concurrently completed job cannot make ``total`` smaller than this
    # returned page.
    artifacts = manager.list_artifacts(job_id, offset=offset, limit=limit)
    view = manager.get_job(job_id)
    if view.artifact_count is None:
        raise JobManagerError(
            ApiError(
                code="INTERNAL_ERROR",
                message="The canonical artifact count is unavailable.",
                retryable=True,
                stage=ErrorStage.INTERNAL,
            )
        )
    return ArtifactListResponse(
        job_id=job_id,
        artifacts=artifacts,
        total=view.artifact_count,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/jobs/{job_id}/artifacts/{artifact_id}",
    response_model=ArtifactDescriptor,
    response_model_exclude_none=True,
)
def get_job_artifact_descriptor(
    request: Request,
    job_id: str,
    artifact_id: ArtifactId,
    verify: bool = False,
) -> ArtifactDescriptor:
    """Return metadata only after canonical job-membership authorization."""

    return (
        _job_manager(request)
        .get_artifact(
            job_id,
            artifact_id,
            verify=verify,
        )
        .descriptor
    )


@router.get(
    "/jobs/{job_id}/artifacts/{artifact_id}/content",
    response_class=Response,
)
def download_job_artifact(
    request: Request,
    job_id: str,
    artifact_id: ArtifactId,
) -> Response:
    """Download managed bytes for a canonically attached artifact."""

    stored = _job_manager(request).get_artifact(
        job_id,
        artifact_id,
        verify=False,
    )
    if request.headers.get("range") is not None:
        raise JobManagerError(
            ApiError(
                code="RANGE_NOT_SUPPORTED",
                message="Range requests are not supported for Agent artifacts.",
                retryable=False,
                stage=ErrorStage.ARTIFACT,
            )
        )
    return verified_artifact_response(stored)


@router.post(
    "/legacy/jobspec-v1/upgrade",
    response_model=LegacyJobUpgradeResponse,
    response_model_exclude_none=True,
)
def upgrade_legacy_jobspec(
    request: Request,
    upgrade: Annotated[
        LegacyJobUpgradeRequest,
        Body(
            openapi_examples={
                "single_h2r": {
                    "summary": "Upgrade one allowlisted H2R JobSpec v1",
                    "value": {
                        "schema_version": "1.0",
                        "payload": {
                            "schema_version": 1,
                            "kind": "retarget",
                            "request": {
                                "source_path": "/srv/hhtools/motions/walk.bvh",
                                "robot": "g1_29dof",
                            },
                        },
                    },
                }
            }
        ),
    ],
) -> LegacyJobUpgradeResponse:
    """Safely re-register and preflight one legacy path-based JobSpec v1."""

    return _legacy_upgrade_service(request).upgrade(upgrade.payload)


__all__ = ["router"]
