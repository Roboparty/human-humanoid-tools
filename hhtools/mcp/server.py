"""HHTools MCP Python SDK v2 server over local stdio.

Tools in this module are deliberately thin calls into application services.
They neither invoke the CLI/REST adapter nor duplicate loader, IK, calibration,
export, scheduler, or artifact-membership logic.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, cast
from urllib.parse import urlsplit

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.types import (
    CallToolResult,
    ImageContent,
    InputRequiredResult,
    TextContent,
    ToolAnnotations,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hhtools._version import __version__
from hhtools.contracts import (
    AgentJobView,
    ApiError,
    ArtifactDescriptor,
    ArtifactExportReceipt,
    ArtifactListResponse,
    AssetBundle,
    AssetCategory,
    AssetInspection,
    AssetInspectionRequest,
    AssetKind,
    AssetRegistrationRequest,
    AssetSearchResponse,
    AvailableAssetCatalogRequest,
    AvailableAssetCatalogResponse,
    BatchPreflightRequest,
    BatchPreflightResponse,
    BatchReport,
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
    CapabilityResponse,
    ErrorStage,
    EvaluationReport,
    FailureReport,
    JobLookupRequest,
    JobManifest,
    JobRetryRequest,
    JobStartRequest,
    PreflightResponse,
    R2RCalibrationPreview,
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
    RobotListResponse,
)
from hhtools.contracts.portability import validate_portable_json
from hhtools.contracts.schema_registry import PUBLIC_AGENT_SCHEMAS
from hhtools.services.jobs import JobManagerError
from hhtools.services.runtime_lease import RuntimeLeaseError

from .runtime import AgentRuntime, LocalRuntimeConfig, local_agent_runtime

_log = logging.getLogger(__name__)
_REPORT_LIMIT_BYTES = 2 * 1024 * 1024
RuntimeFactory = Callable[[], AbstractAsyncContextManager[AgentRuntime]]

_READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
_SAFE_WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
_CANCEL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)
_CALIBRATION_SAVE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


def _harden_tool_argument_models(server: MCPServer[Any]) -> None:
    """Reject unknown MCP arguments without echoing their values.

    The MCP SDK currently derives a dynamic Pydantic model for every function
    signature with Pydantic's default ``extra='ignore'`` behavior.  That makes
    a misspelled or wrongly wrapped request silently fall back to defaults.
    Harden the registered models at this composition boundary so the live
    validator and the advertised JSON Schema enforce the same closed shape.

    ``hide_input_in_errors`` is equally important: validation failures are
    returned to the MCP client, so rejected paths, tokens, or other caller data
    must not be copied into the error prose.
    """

    # MCPServer does not yet expose a public hook for configuring its generated
    # argument models.  Keep the SDK-specific access isolated here and cover it
    # with contract tests so an SDK upgrade fails loudly rather than weakening
    # validation unnoticed.
    manager = cast(Any, server)._tool_manager
    for tool in manager.list_tools():
        argument_model = tool.fn_metadata.arg_model
        argument_model.model_config = ConfigDict(
            **argument_model.model_config,
            extra="forbid",
            hide_input_in_errors=True,
        )
        argument_model.model_rebuild(force=True)
        tool.parameters = argument_model.model_json_schema(by_alias=True)


def _model_document(model: BaseModel) -> dict[str, Any]:
    document = model.model_dump(mode="json", exclude_none=True)
    validate_portable_json(document)
    return document


def _internal_error() -> ApiError:
    return ApiError(
        code="INTERNAL_ERROR",
        message="The HHTools MCP service could not complete the request.",
        retryable=True,
        stage=ErrorStage.INTERNAL,
    )


def _public_error(exception: Exception) -> ApiError:
    error = getattr(exception, "api_error", None)
    return error if isinstance(error, ApiError) else _internal_error()


def _safe_error_document(exception: Exception) -> tuple[ApiError, dict[str, Any]]:
    """Return one portable public error without echoing rejected service data."""

    error = _public_error(exception)
    try:
        return error, _model_document(error)
    except Exception:  # noqa: BLE001 - the protocol boundary must never leak it
        # Do not log the rejected error or exception: either may contain the
        # host path (or other unsafe value) that this boundary is removing.
        _log.error("discarded a non-portable HHTools MCP service error")
        error = _internal_error()
        return error, _model_document(error)


def _error_result(document: dict[str, Any]) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(document, ensure_ascii=False, separators=(",", ":")),
            )
        ],
        structuredContent=document,
        isError=True,
    )


def _invalid_tool_arguments_result() -> CallToolResult:
    """Return one path-free contract for every advertised-schema violation."""

    error = ApiError(
        code="INVALID_PARAMETER",
        message="The MCP tool arguments do not match the advertised schema.",
        retryable=False,
        stage=ErrorStage.REQUEST,
    )
    return _error_result(_model_document(error))


class _HHToolsMCPServer(MCPServer[AgentRuntime]):
    """Translate SDK argument validation at HHTools' composition boundary."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Context[AgentRuntime, Any] | None = None,
    ) -> CallToolResult | InputRequiredResult:
        try:
            return await super().call_tool(name, arguments, context)
        except ToolError as exception:
            # Tool.run wraps only its argument-model ValidationError directly.
            # Handler failures have a different cause chain and retain the
            # existing service-error path in MCPServer._handle_call_tool.
            if isinstance(exception.__cause__, ValidationError):
                return _invalid_tool_arguments_result()
            raise


def _tool_call[T](call: Callable[[], T]) -> T:
    """Keep success schemas while returning expected failures as MCP tool errors."""

    try:
        result = call()
        if isinstance(result, BaseModel):
            # Return a detached model rebuilt from the exact portable snapshot
            # that was checked.  Returning the service-owned instance would let
            # a concurrently mutated nested dict/list diverge before the SDK's
            # later serialization step.
            return cast(T, type(result).model_validate(_model_document(result)))
        return result
    except Exception as exception:  # noqa: BLE001 - protocol boundary
        error, document = _safe_error_document(exception)
        if error.code == "INTERNAL_ERROR":
            # Exception text can itself contain a host path.  Keep diagnostics
            # useful without copying untrusted service data to stderr.
            _log.error(
                "unexpected HHTools MCP tool failure (%s)",
                type(exception).__name__,
            )
        # MCPServer recognises a direct CallToolResult before validating the
        # declared success model, retaining both outputSchema and ApiError.
        return cast(T, _error_result(document))


def _calibration_preview_call[T: CalibrationPreview](
    call: Callable[[], tuple[T, bytes]],
) -> T:
    """Return typed metadata plus an actual image block for vision-capable clients."""

    try:
        preview, payload = call()
        document = _model_document(preview)
        return cast(
            T,
            CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(document, ensure_ascii=False, separators=(",", ":")),
                    ),
                    ImageContent(
                        type="image",
                        data=base64.b64encode(payload).decode("ascii"),
                        mimeType="image/png",
                    ),
                ],
                structuredContent=document,
            ),
        )
    except Exception as exception:  # noqa: BLE001 - protocol boundary
        _error_value, document = _safe_error_document(exception)
        return cast(T, _error_result(document))


def _resource_call[T](call: Callable[[], T]) -> T:
    try:
        return call()
    except Exception as exception:  # noqa: BLE001 - protocol boundary
        error, document = _safe_error_document(exception)
        if error.code == "INTERNAL_ERROR":
            _log.error(
                "unexpected HHTools MCP resource failure (%s)",
                type(exception).__name__,
            )
        raise ResourceError(
            json.dumps(
                document,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ) from None


def _runtime(context: Context[AgentRuntime, Any]) -> AgentRuntime:
    return context.request_context.lifespan_context


def _calibration_runtime(context: Context[AgentRuntime, Any]):
    service = _runtime(context).calibration
    if service is None:
        raise RuntimeError("the calibration service is not configured")
    return service


def _r2r_calibration_runtime(context: Context[AgentRuntime, Any]):
    service = _runtime(context).r2r_calibration
    if service is None:
        raise RuntimeError("the R2R calibration service is not configured")
    return service


def _job_error(code: str, message: str, *, job_id: str) -> JobManagerError:
    return JobManagerError(
        ApiError(
            code=code,
            message=message,
            stage=ErrorStage.ARTIFACT,
            details={"job_id": job_id},
        )
    )


def _find_report(
    runtime: AgentRuntime,
    job_id: str,
    kind: str,
) -> ArtifactDescriptor:
    view = runtime.jobs.get_job(job_id)
    total = view.artifact_count or 0
    offset = 0
    matches: list[ArtifactDescriptor] = []
    while offset < total:
        page = runtime.jobs.list_artifacts(job_id, offset=offset, limit=500)
        if not page:
            break
        matches.extend(item for item in page if item.kind == kind)
        offset += len(page)
    if len(matches) != 1:
        raise _job_error(
            "ARTIFACT_NOT_FOUND",
            f"The job does not expose one canonical {kind} artifact.",
            job_id=job_id,
        )
    return matches[0]


def _read_report[T](
    runtime: AgentRuntime,
    job_id: str,
    kind: str,
    model: type[T],
) -> T:
    descriptor = _find_report(runtime, job_id, kind)
    if descriptor.size_bytes is not None and descriptor.size_bytes > _REPORT_LIMIT_BYTES:
        raise _job_error(
            "REPORT_TOO_LARGE",
            "The verified report is too large for inline model context; export its artifact.",
            job_id=job_id,
        )
    stored = runtime.jobs.get_artifact(job_id, descriptor.artifact_id, verify=False)
    try:
        with stored.path.open("rb") as stream:
            payload = stream.read(_REPORT_LIMIT_BYTES + 1)
    except OSError as exception:
        raise _job_error(
            "ARTIFACT_HASH_MISMATCH",
            "The managed report no longer matches its descriptor.",
            job_id=job_id,
        ) from exception
    # Hash exactly the immutable byte string that will be parsed and returned.
    # A second read of the file would permit a concurrent writer to make the
    # digest describe different bytes (a classic check/use race).
    if (
        len(payload) > _REPORT_LIMIT_BYTES
        or len(payload) != descriptor.size_bytes
        or hashlib.sha256(payload).hexdigest() != descriptor.sha256
    ):
        raise _job_error(
            "ARTIFACT_HASH_MISMATCH",
            "The managed report no longer matches its descriptor.",
            job_id=job_id,
        )
    try:
        validator = cast(Any, model)
        return cast(T, validator.model_validate_json(payload))
    except Exception as exception:  # noqa: BLE001 - invalid managed artifact
        raise _job_error(
            "ARTIFACT_HASH_MISMATCH",
            "The managed report is not a valid versioned contract.",
            job_id=job_id,
        ) from exception


def _server_instructions(web_ui_url: str) -> str:
    return (
        "For every new H2R, R2R, or batch run: get capabilities, register/search and inspect "
        "assets. Before H2R preflight, check the exact robot/reference calibration status; before "
        "R2R preflight, check the exact source/target pair calibration status. Replace missing or "
        "invalid calibration through the matching validated proposal flow. Then "
        "preflight a smoke plan, start only a ready plan, wait by revision, then read "
        "evaluation and manifest for human review. Persist each plan_id plus idempotency "
        "key before start; use lookup_job to recover an ambiguous submission without job "
        "enumeration. On CALIBRATION_REQUIRED or R2R_CALIBRATION_REQUIRED, use the matching "
        "calibration status, "
        "proposal, deterministic validation, and preview tools; stop and present every other "
        "human action. A "
        "vision-capable GPT client should inspect the preview image before using "
        "gpt_vision_silent save; the review declaration is audit metadata, not authentication. "
        "Validated silent calibration writes are allowed and must be followed by fresh preflight. "
        "run_mode is frozen at preflight, "
        "and batch preflight accepts only ordered ready child plans from one workflow and "
        "run mode. Batch retry always retries the whole plan. "
        "Full execution requires a new full preflight plus explicit user approval. Completed "
        "does not mean quality-approved. Never use host paths, put Base64 in tool arguments or "
        "text, or deploy to a real robot. preview_calibration and preview_r2r_calibration are the "
        "only image-content exceptions. For user-requested files, export only by job_id and "
        "artifact_id and return the portable agent-exports receipt. Cancellation is "
        "cooperative while native code runs. "
        "Only one local runtime may own a save directory. The WebUI fallback remains available "
        f"at {web_ui_url}, but never run it concurrently with this stdio owner or request its "
        "session token."
    )


def _validate_web_ui_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("web_ui_url must be an unauthenticated loopback HTTP URL")
    return value.rstrip("/")


def create_mcp_server(
    config: LocalRuntimeConfig | None = None,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> MCPServer[AgentRuntime]:
    """Build the stdio server; tests may inject the same service protocols."""

    config = config or LocalRuntimeConfig()
    web_ui_url = _validate_web_ui_url(config.web_ui_url)
    factory = runtime_factory or (lambda: local_agent_runtime(config))

    runtime_slot: AgentRuntime | None = None

    def active_runtime() -> AgentRuntime:
        if runtime_slot is None:
            raise RuntimeError("the MCP runtime is not active")
        return runtime_slot

    @asynccontextmanager
    async def lifespan(_server: MCPServer[AgentRuntime]) -> AsyncIterator[AgentRuntime]:
        nonlocal runtime_slot
        async with factory() as runtime:
            if runtime_slot is not None:
                raise RuntimeError("the MCP runtime is already active")
            runtime_slot = runtime
            try:
                yield runtime
            finally:
                runtime_slot = None

    server: MCPServer[AgentRuntime] = _HHToolsMCPServer(
        "hhtools",
        title="HHTools Agent",
        description=("Safe local H2R, scene-free R2R, batch, and validated calibration services."),
        instructions=_server_instructions(web_ui_url),
        version=__version__,
        lifespan=lifespan,
        # stdio reserves stdout for protocol frames.  Keep routine SDK
        # diagnostics on stderr quiet while retaining warnings and failures.
        log_level="WARNING",
    )

    @server.tool(annotations=_READ_ONLY)
    def get_capabilities(context: Context[AgentRuntime, Any]) -> CapabilityResponse:
        """Return backends, devices, robots, allowlisted roots, and live admission state."""

        return _tool_call(_runtime(context).capabilities.get_capabilities)

    @server.tool(annotations=_SAFE_WRITE)
    def register_asset_bundle(
        request: AssetRegistrationRequest,
        context: Context[AgentRuntime, Any],
    ) -> AssetBundle:
        """Register a content-addressed bundle using root_id plus a relative path."""

        return _tool_call(lambda: _runtime(context).assets.register(request))

    @server.tool(annotations=_READ_ONLY)
    def search_assets(
        context: Context[AgentRuntime, Any],
        query: str | None = None,
        kind: AssetKind | None = None,
        category: AssetCategory | None = None,
        dataset: str | None = None,
        reference: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> AssetSearchResponse:
        """Search immutable asset manifests with bounded portable filters."""

        return _tool_call(
            lambda: _runtime(context).assets.search(
                query=query,
                kind=kind,
                category=category,
                dataset=dataset,
                reference=reference,
                limit=limit,
                offset=offset,
            )
        )

    @server.tool(annotations=_READ_ONLY)
    def list_available_assets(
        context: Context[AgentRuntime, Any],
        root_id: Annotated[
            str | None,
            Field(
                min_length=1,
                max_length=128,
                pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
            ),
        ] = None,
        query: Annotated[str | None, Field(min_length=1, max_length=256)] = None,
        kind: AssetKind | None = None,
        limit: Annotated[int, Field(ge=1, le=500)] = 100,
        offset: Annotated[int, Field(ge=0)] = 0,
    ) -> AvailableAssetCatalogResponse:
        """List registerable assets below configured allowlisted roots."""

        request = AvailableAssetCatalogRequest(
            root_id=root_id,
            query=query,
            kind=kind,
            limit=limit,
            offset=offset,
        )
        return _tool_call(lambda: _runtime(context).available_assets.list_available(request))

    @server.tool(annotations=_READ_ONLY)
    def inspect_asset_bundle(
        request: AssetInspectionRequest,
        context: Context[AgentRuntime, Any],
    ) -> AssetInspection:
        """Verify manifest hashes and parse bundle content without starting a job."""

        return _tool_call(lambda: _runtime(context).assets.inspect(request))

    @server.tool(annotations=_READ_ONLY)
    def list_robots(context: Context[AgentRuntime, Any]) -> RobotListResponse:
        """List robot availability, references, IK-map facts, and calibration readiness."""

        return _tool_call(
            lambda: RobotListResponse(
                robots=_runtime(context).capabilities.get_capabilities().robots
            )
        )

    @server.tool(annotations=_READ_ONLY)
    def get_calibration_status(
        request: CalibrationStatusRequest,
        context: Context[AgentRuntime, Any],
    ) -> CalibrationStatusResponse:
        """Inspect one content-bound robot/reference calibration and its current quality."""

        return _tool_call(lambda: _calibration_runtime(context).status(request))

    @server.tool(annotations=_SAFE_WRITE)
    def propose_calibration(
        request: CalibrationProposalRequest,
        context: Context[AgentRuntime, Any],
    ) -> CalibrationProposalResponse:
        """Generate or revise a persisted, limit-constrained calibration candidate."""

        return _tool_call(lambda: _calibration_runtime(context).propose(request))

    @server.tool(annotations=_READ_ONLY)
    def validate_calibration(
        request: CalibrationValidationRequest,
        context: Context[AgentRuntime, Any],
    ) -> CalibrationValidationReport:
        """Recompute deterministic mapping, limit, alignment, symmetry, and foot checks."""

        return _tool_call(lambda: _calibration_runtime(context).validate(request))

    @server.tool(annotations=_READ_ONLY)
    def preview_calibration(
        request: CalibrationPreviewRequest,
        context: Context[AgentRuntime, Any],
    ) -> CalibrationPreview:
        """Return front/side PNG overlays for a vision-capable GPT calibration review."""

        return _calibration_preview_call(lambda: _calibration_runtime(context).preview(request))

    @server.tool(annotations=_CALIBRATION_SAVE)
    def save_calibration(
        request: CalibrationSaveRequest,
        context: Context[AgentRuntime, Any],
    ) -> CalibrationSaveReceipt:
        """Silently save only a currently valid candidate under an explicit save mode."""

        return _tool_call(lambda: _calibration_runtime(context).save(request))

    @server.tool(annotations=_READ_ONLY)
    def get_r2r_calibration_status(
        request: R2RCalibrationStatusRequest,
        context: Context[AgentRuntime, Any],
    ) -> R2RCalibrationStatusResponse:
        """Inspect one content-bound source/target robot pair calibration."""

        return _tool_call(lambda: _r2r_calibration_runtime(context).status(request))

    @server.tool(annotations=_SAFE_WRITE)
    def propose_r2r_calibration(
        request: R2RCalibrationProposalRequest,
        context: Context[AgentRuntime, Any],
    ) -> R2RCalibrationProposalResponse:
        """Generate or revise a target-pose candidate against the source robot rest pose."""

        return _tool_call(lambda: _r2r_calibration_runtime(context).propose(request))

    @server.tool(annotations=_READ_ONLY)
    def validate_r2r_calibration(
        request: R2RCalibrationValidationRequest,
        context: Context[AgentRuntime, Any],
    ) -> CalibrationValidationReport:
        """Recompute deterministic pair mapping, limits, alignment, symmetry, and foot checks."""

        return _tool_call(lambda: _r2r_calibration_runtime(context).validate(request))

    @server.tool(annotations=_READ_ONLY)
    def preview_r2r_calibration(
        request: R2RCalibrationPreviewRequest,
        context: Context[AgentRuntime, Any],
    ) -> R2RCalibrationPreview:
        """Return source-reference and target-pose front/side PNG overlays for visual review."""

        return _calibration_preview_call(lambda: _r2r_calibration_runtime(context).preview(request))

    @server.tool(annotations=_CALIBRATION_SAVE)
    def save_r2r_calibration(
        request: R2RCalibrationSaveRequest,
        context: Context[AgentRuntime, Any],
    ) -> R2RCalibrationSaveReceipt:
        """Silently save a valid pair candidate to the target robot's user overlay."""

        return _tool_call(lambda: _r2r_calibration_runtime(context).save(request))

    @server.tool(annotations=_SAFE_WRITE)
    def preflight_retarget(
        request: RetargetPreflightRequest,
        context: Context[AgentRuntime, Any],
    ) -> PreflightResponse:
        """Validate an H2R intent and freeze an immutable smoke or full plan."""

        return _tool_call(lambda: _runtime(context).preflight.preflight_retarget(request))

    @server.tool(annotations=_SAFE_WRITE)
    def preflight_r2r(
        request: R2RPreflightRequest,
        context: Context[AgentRuntime, Any],
    ) -> R2RPreflightResponse:
        """Validate one scene-free R2R intent and freeze both robot identities."""

        return _tool_call(lambda: _runtime(context).r2r_preflight.preflight_r2r(request))

    @server.tool(annotations=_SAFE_WRITE)
    def preflight_batch(
        request: BatchPreflightRequest,
        context: Context[AgentRuntime, Any],
    ) -> BatchPreflightResponse:
        """Freeze an ordered list of ready H2R or R2R plans into one batch."""

        return _tool_call(lambda: _runtime(context).batch_preflight.preflight_batch(request))

    @server.tool(annotations=_SAFE_WRITE)
    def start_job(
        request: JobStartRequest,
        context: Context[AgentRuntime, Any],
    ) -> AgentJobView:
        """Submit one H2R or R2R immutable plan through the shared lifecycle."""

        def start() -> AgentJobView:
            jobs = _runtime(context).jobs
            submit = getattr(jobs, "start_job", jobs.start_retarget)
            return submit(
                request.plan_id,
                idempotency_key=request.idempotency_key,
            )

        return _tool_call(start)

    @server.tool(annotations=_SAFE_WRITE)
    def start_retarget(
        request: JobStartRequest,
        context: Context[AgentRuntime, Any],
    ) -> AgentJobView:
        """Compatibility alias for submitting an immutable preflighted plan."""

        return _tool_call(
            lambda: _runtime(context).jobs.start_retarget(
                request.plan_id,
                idempotency_key=request.idempotency_key,
            )
        )

    @server.tool(annotations=_READ_ONLY)
    def get_job(
        job_id: str,
        context: Context[AgentRuntime, Any],
        after_revision: int | None = None,
    ) -> AgentJobView:
        """Read a compact job snapshot, optionally marking an unchanged revision."""

        return _tool_call(
            lambda: _runtime(context).jobs.get_job(
                job_id,
                after_revision=after_revision,
            )
        )

    @server.tool(annotations=_READ_ONLY)
    def wait_job(
        job_id: str,
        after_revision: Annotated[int, Field(ge=0)],
        context: Context[AgentRuntime, Any],
        timeout: Annotated[float, Field(ge=0.0, le=60.0, allow_inf_nan=False)] = 30.0,
    ) -> AgentJobView:
        """Wait until one job advances beyond a known revision or becomes terminal."""

        return _tool_call(
            lambda: _runtime(context).jobs.wait_job(
                job_id,
                after_revision=after_revision,
                timeout=timeout,
            )
        )

    @server.tool(annotations=_READ_ONLY)
    def lookup_job(
        request: JobLookupRequest,
        context: Context[AgentRuntime, Any],
    ) -> AgentJobView:
        """Recover one caller-owned submission by its immutable plan and key."""

        return _tool_call(
            lambda: _runtime(context).jobs.lookup_job(
                request.plan_id,
                idempotency_key=request.idempotency_key,
                after_revision=request.after_revision,
            )
        )

    @server.tool(annotations=_CANCEL)
    def cancel_job(
        job_id: str,
        context: Context[AgentRuntime, Any],
    ) -> AgentJobView:
        """Request exact queued cancellation or cooperative running cancellation."""

        return _tool_call(lambda: _runtime(context).jobs.cancel_job(job_id))

    @server.tool(annotations=_SAFE_WRITE)
    def retry_job(
        job_id: str,
        request: JobRetryRequest,
        context: Context[AgentRuntime, Any],
    ) -> AgentJobView:
        """Create an idempotent whole-plan child attempt for a terminal workflow job."""

        return _tool_call(
            lambda: _runtime(context).jobs.retry_job(
                job_id,
                idempotency_key=request.idempotency_key,
            )
        )

    @server.tool(annotations=_READ_ONLY)
    def list_job_artifacts(
        job_id: str,
        context: Context[AgentRuntime, Any],
        limit: int = 100,
        offset: int = 0,
    ) -> ArtifactListResponse:
        """List a bounded page of canonical descriptors attached to one job."""

        def list_page() -> ArtifactListResponse:
            runtime = _runtime(context)
            artifacts = runtime.jobs.list_artifacts(job_id, limit=limit, offset=offset)
            view = runtime.jobs.get_job(job_id)
            if view.artifact_count is None:
                raise _job_error(
                    "INTERNAL_ERROR",
                    "The canonical artifact count is unavailable.",
                    job_id=job_id,
                )
            return ArtifactListResponse(
                job_id=job_id,
                artifacts=artifacts,
                total=view.artifact_count,
                limit=limit,
                offset=offset,
            )

        return _tool_call(list_page)

    @server.tool(annotations=_SAFE_WRITE)
    def export_artifact(
        job_id: str,
        artifact_id: str,
        context: Context[AgentRuntime, Any],
    ) -> ArtifactExportReceipt:
        """Materialize verified bytes below the fixed agent-exports root.

        The tool never returns bytes or a host path.  Its receipt identifies a
        deterministic path below ``<save-dir>/agent-exports`` so the caller can
        hand the result to the human without inspecting HHTools' private store.
        """

        return _tool_call(lambda: _runtime(context).exports.export(job_id, artifact_id))

    @server.resource(
        "hhtools://capabilities",
        name="hhtools-capabilities",
        description="Current HHTools capability snapshot.",
        mime_type="application/json",
    )
    def capabilities_resource() -> dict[str, Any]:
        return _resource_call(
            lambda: _model_document(active_runtime().capabilities.get_capabilities())
        )

    @server.resource(
        "hhtools://schemas/agent/v1/{schema_name}",
        name="hhtools-agent-schema",
        description="One public Agent v1 JSON Schema by canonical slug.",
        mime_type="application/schema+json",
    )
    def schema_resource(schema_name: str) -> dict[str, Any]:
        def schema() -> dict[str, Any]:
            model = PUBLIC_AGENT_SCHEMAS.get(schema_name)
            if model is None:
                raise JobManagerError(
                    ApiError(
                        code="SCHEMA_NOT_FOUND",
                        message="No public Agent schema has the requested name.",
                        stage=ErrorStage.REQUEST,
                        details={"schema_name": schema_name},
                    )
                )
            return model.model_json_schema()

        return _resource_call(schema)

    @server.resource(
        "hhtools://robots/{robot_id}",
        name="hhtools-robot",
        description="One robot capability and calibration-readiness record.",
        mime_type="application/json",
    )
    async def robot_resource(
        robot_id: str,
        context: Context,
    ) -> dict[str, Any]:
        def robot() -> dict[str, Any]:
            robots = _runtime(context).capabilities.get_capabilities().robots
            match = next((item for item in robots if item.robot_id == robot_id), None)
            if match is None:
                raise JobManagerError(
                    ApiError(
                        code="ROBOT_NOT_FOUND",
                        message="No robot has the requested id.",
                        stage=ErrorStage.REQUEST,
                        details={"robot_id": robot_id},
                    )
                )
            return _model_document(match)

        return _resource_call(robot)

    @server.resource(
        "hhtools://assets/{asset_id}/manifest",
        name="hhtools-asset-manifest",
        description="Portable immutable manifest for one registered asset.",
        mime_type="application/json",
    )
    async def asset_resource(
        asset_id: str,
        context: Context,
    ) -> dict[str, Any]:
        return _resource_call(lambda: _model_document(_runtime(context).assets.get(asset_id)))

    @server.resource(
        "hhtools://plans/{plan_id}",
        name="hhtools-retarget-plan",
        description="One immutable preflighted retarget plan.",
        mime_type="application/json",
    )
    async def plan_resource(
        plan_id: str,
        context: Context,
    ) -> dict[str, Any]:
        return _resource_call(lambda: _model_document(_runtime(context).plans.get(plan_id)))

    @server.resource(
        "hhtools://jobs/{job_id}/status",
        name="hhtools-job-status",
        description="Compact current state for one H2R or R2R job.",
        mime_type="application/json",
    )
    async def job_resource(
        job_id: str,
        context: Context,
    ) -> dict[str, Any]:
        return _resource_call(lambda: _model_document(_runtime(context).jobs.get_job(job_id)))

    @server.resource(
        "hhtools://jobs/{job_id}/manifest",
        name="hhtools-job-manifest",
        description="Verified terminal audit manifest for one H2R or R2R job.",
        mime_type="application/json",
    )
    async def manifest_resource(
        job_id: str,
        context: Context,
    ) -> dict[str, Any]:
        return _resource_call(
            lambda: _model_document(
                _read_report(_runtime(context), job_id, "manifest", JobManifest)
            )
        )

    @server.resource(
        "hhtools://jobs/{job_id}/evaluation",
        name="hhtools-job-evaluation",
        description="Verified quality report; completion alone is not approval.",
        mime_type="application/json",
    )
    async def evaluation_resource(
        job_id: str,
        context: Context,
    ) -> dict[str, Any]:
        return _resource_call(
            lambda: _model_document(
                _read_report(
                    _runtime(context),
                    job_id,
                    "evaluation_report",
                    EvaluationReport,
                )
            )
        )

    @server.resource(
        "hhtools://jobs/{job_id}/batch",
        name="hhtools-job-batch-report",
        description="Verified per-item result report for one batch job.",
        mime_type="application/json",
    )
    async def batch_resource(
        job_id: str,
        context: Context,
    ) -> dict[str, Any]:
        return _resource_call(
            lambda: _model_document(
                _read_report(_runtime(context), job_id, "batch_report", BatchReport)
            )
        )

    @server.resource(
        "hhtools://jobs/{job_id}/failures",
        name="hhtools-job-failures",
        description="Verified structured failure report for a failed or partial job.",
        mime_type="application/json",
    )
    async def failures_resource(
        job_id: str,
        context: Context,
    ) -> dict[str, Any]:
        return _resource_call(
            lambda: _model_document(
                _read_report(
                    _runtime(context),
                    job_id,
                    "failure_report",
                    FailureReport,
                )
            )
        )

    @server.resource(
        "hhtools://jobs/{job_id}/artifacts/{artifact_id}",
        name="hhtools-artifact-descriptor",
        description="Job-scoped descriptor only; binary bytes are never embedded.",
        mime_type="application/json",
    )
    async def artifact_resource(
        job_id: str,
        artifact_id: str,
        context: Context,
    ) -> dict[str, Any]:
        return _resource_call(
            lambda: _model_document(
                _runtime(context).jobs.get_artifact(job_id, artifact_id, verify=True).descriptor
            )
        )

    _harden_tool_argument_models(server)
    return server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hhtools-mcp",
        description="Run the local HHTools MCP server over stdio.",
    )
    parser.add_argument("--source", type=Path, default=Path("assets/motions"))
    parser.add_argument("--save-dir", type=Path, default=Path("assets/save_npz"))
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--job-settings", type=Path, default=None)
    parser.add_argument("--max-running-jobs", type=int, default=None)
    parser.add_argument("--max-queued-jobs", type=int, default=None)
    parser.add_argument("--max-batch-items", type=int, default=None)
    parser.add_argument("--max-batch-total-frames", type=int, default=None)
    parser.add_argument("--web-ui-url", default="http://127.0.0.1:8009")
    return parser


_RUNTIME_LEASE_EXIT_CODE = 3


def _runtime_lease_failure(exception: BaseException) -> RuntimeLeaseError | None:
    """Unwrap an expected lease failure from the MCP SDK's task group."""

    if isinstance(exception, RuntimeLeaseError):
        return exception
    if not isinstance(exception, BaseExceptionGroup):
        return None

    matched, remainder = exception.split(RuntimeLeaseError)
    # Do not hide an unrelated sibling failure.  The MCP/AnyIO startup path
    # currently wraps the single lifespan exception in an ExceptionGroup.
    if matched is None or remainder is not None:
        return None
    pending: list[BaseException] = [matched]
    while pending:
        item = pending.pop()
        if isinstance(item, RuntimeLeaseError):
            return item
        if isinstance(item, BaseExceptionGroup):
            pending.extend(item.exceptions)
    return None


def _runtime_lease_message(error: RuntimeLeaseError) -> str:
    """Return one actionable stderr line without echoing a host path."""

    if error.code == "RUNTIME_ALREADY_ACTIVE":
        return (
            "ERROR RUNTIME_ALREADY_ACTIVE: Another HHTools runtime owns this Agent "
            "data directory. Close the existing WebUI or other HHTools runtime, then "
            "reconnect hhtools-mcp with the same --save-dir."
        )
    return (
        "ERROR RUNTIME_LEASE_UNAVAILABLE: HHTools could not establish exclusive runtime "
        "ownership. Check the configured --save-dir permissions and retry."
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Console entry point. stdout remains exclusively owned by MCP framing."""

    arguments = _parser().parse_args(argv)
    for name in (
        "max_running_jobs",
        "max_queued_jobs",
        "max_batch_items",
        "max_batch_total_frames",
    ):
        value = getattr(arguments, name)
        if value is not None and value < 0:
            _parser().error(f"--{name.replace('_', '-')} must be non-negative")
    config = LocalRuntimeConfig(
        source_root=arguments.source,
        save_dir=arguments.save_dir,
        cache_dir=arguments.cache,
        max_running_jobs=arguments.max_running_jobs,
        max_queued_jobs=arguments.max_queued_jobs,
        max_batch_items=arguments.max_batch_items,
        max_batch_total_frames=arguments.max_batch_total_frames,
        job_settings_path=arguments.job_settings,
        web_ui_url=arguments.web_ui_url,
    )
    try:
        create_mcp_server(config).run("stdio")
    except BaseException as exception:
        lease_error = _runtime_lease_failure(exception)
        if lease_error is None:
            raise
        print(_runtime_lease_message(lease_error), file=sys.stderr)
        raise SystemExit(_RUNTIME_LEASE_EXIT_CODE) from None


if __name__ == "__main__":
    main()


__all__ = ["RuntimeFactory", "create_mcp_server", "main"]
