"""Strict, versioned JSON client for the HHTools Agent REST API.

Every invocation writes exactly one JSON document to stdout.  Human-oriented
diagnostics belong on stderr, while large artifact bytes require an explicit
``--output`` file and never appear in JSON or Base64 form.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never, TextIO, cast
from urllib.parse import quote

import typer
from pydantic import BaseModel, ValidationError

from hhtools.contracts import (
    AgentCliArgumentDiagnostic,
    AgentCliHelp,
    AgentCliHelpArgument,
    AgentCliHelpSubcommand,
    AgentJobView,
    ApiError,
    ArtifactDescriptor,
    ArtifactListResponse,
    AssetBundle,
    AssetInspection,
    AssetRegistrationRequest,
    AssetSearchResponse,
    AvailableAssetCatalogRequest,
    AvailableAssetCatalogResponse,
    BatchPreflightRequest,
    BatchPreflightResponse,
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
    PreflightStatus,
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
from hhtools.contracts.cli import (
    AgentCliCommandName,
    AgentCliDiagnosticArgument,
    AgentCliReasonCode,
)

from .agent_transport import (
    AgentTransport,
    AgentTransportError,
    HttpAgentTransport,
    PortableJsonError,
    StrictJsonError,
    ensure_portable_json,
    loads_strict_json,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8009/api/agent/v1"
EXIT_SUCCESS = 0
EXIT_PARAMETER_ERROR = 2
EXIT_PREFLIGHT_ERROR = 3
EXIT_JOB_ERROR = 4
EXIT_INTERNAL_ERROR = 5
_MAX_REQUEST_BYTES = 8 * 1024 * 1024

TransportFactory = Callable[[str, float], AgentTransport]


@dataclass(frozen=True, slots=True)
class _CliArgumentSpec:
    name: AgentCliDiagnosticArgument
    description: str
    value_name: str | None = None
    required: bool = False
    expected: str | None = None
    value_kind: str = "string"


@dataclass(frozen=True, slots=True)
class _CliCommandSpec:
    path: tuple[str, ...]
    summary: str
    positionals: tuple[_CliArgumentSpec, ...] = ()
    options: tuple[_CliArgumentSpec, ...] = ()


_REQUEST_ARGUMENT = _CliArgumentSpec(
    "--request",
    "Read one strict UTF-8 JSON request from this file, or from stdin when the value is '-'.",
    value_name="JSON_FILE_OR_DASH",
    required=True,
    expected="A readable UTF-8 JSON file, or '-' for stdin.",
)
_PLAN_ARGUMENT = _CliArgumentSpec(
    "--plan",
    "Use the immutable plan id returned by retarget preflight.",
    value_name="PLAN_ID",
    required=True,
    expected="A plan id matching 'plan:sha256:' followed by 64 lowercase hex characters.",
)
_IDEMPOTENCY_ARGUMENT = _CliArgumentSpec(
    "--idempotency-key",
    "Bind this caller-generated key to one logical submission.",
    value_name="KEY",
    required=True,
    expected=(
        "A 1-256 character key beginning with a letter or digit and containing only "
        "letters, digits, '.', '_', '~', ':', or '-'."
    ),
)
_JOB_ID_ARGUMENT = _CliArgumentSpec(
    "JOB_ID",
    "Use a job id returned by job start or retry.",
    required=True,
    expected="A job id returned by job start or retry.",
)
_ASSET_ID_ARGUMENT = _CliArgumentSpec(
    "ASSET_ID",
    "Use a content-addressed asset id returned by asset register or search.",
    required=True,
    expected="An asset id returned by asset register or search.",
)
_ARTIFACT_ID_ARGUMENT = _CliArgumentSpec(
    "ARTIFACT_ID",
    "Use an artifact id returned by artifact list.",
    required=True,
    expected="An artifact id returned by artifact list.",
)
_AFTER_REVISION_ARGUMENT = _CliArgumentSpec(
    "--after-revision",
    "Mark a response unchanged when this revision is still current.",
    value_name="INTEGER",
    expected="A non-negative integer revision.",
    value_kind="int",
)
_WAIT_AFTER_REVISION_ARGUMENT = _CliArgumentSpec(
    "--after-revision",
    "Wait until the job advances beyond this observed revision.",
    value_name="INTEGER",
    required=True,
    expected="A non-negative integer revision returned by start, get, lookup, or wait.",
    value_kind="int",
)
_WAIT_TIMEOUT_ARGUMENT = _CliArgumentSpec(
    "--wait-timeout",
    "Wait this many seconds for a job revision change.",
    value_name="SECONDS",
    expected="A finite number from 0 through 60.",
    value_kind="float",
)

_COMMAND_SPECS: dict[tuple[str, ...], _CliCommandSpec] = {
    (): _CliCommandSpec((), "Call the versioned Agent API with strict JSON input and output."),
    ("capabilities",): _CliCommandSpec(
        ("capabilities",), "Return the live Agent capability document."
    ),
    ("asset",): _CliCommandSpec(("asset",), "Catalog, register, inspect, get, or search assets."),
    ("asset", "catalog"): _CliCommandSpec(
        ("asset", "catalog"),
        "List registerable assets below configured allowlisted roots.",
        options=(
            _CliArgumentSpec("--root-id", "Filter by allowlisted root id.", value_name="ROOT_ID"),
            _CliArgumentSpec("--query", "Filter by text query.", value_name="TEXT"),
            _CliArgumentSpec("--kind", "Filter by asset kind.", value_name="KIND"),
            _CliArgumentSpec(
                "--limit",
                "Limit the returned page.",
                value_name="INTEGER",
                expected="An integer from 1 through 500.",
                value_kind="int",
            ),
            _CliArgumentSpec(
                "--offset",
                "Start the returned page at this offset.",
                value_name="INTEGER",
                expected="A non-negative integer offset.",
                value_kind="int",
            ),
        ),
    ),
    ("asset", "register"): _CliCommandSpec(
        ("asset", "register"),
        "Register one allowlisted content-addressed asset bundle.",
        options=(_REQUEST_ARGUMENT,),
    ),
    ("asset", "get"): _CliCommandSpec(
        ("asset", "get"),
        "Return one registered asset manifest.",
        positionals=(_ASSET_ID_ARGUMENT,),
    ),
    ("asset", "inspect"): _CliCommandSpec(
        ("asset", "inspect"),
        "Verify and inspect one registered asset bundle.",
        positionals=(_ASSET_ID_ARGUMENT,),
        options=(
            _CliArgumentSpec("--no-verify-hashes", "Skip content hash verification."),
            _CliArgumentSpec("--no-parse-content", "Skip bounded content parsing."),
        ),
    ),
    ("asset", "search"): _CliCommandSpec(
        ("asset", "search"),
        "Search registered assets with bounded scalar filters.",
        options=(
            _CliArgumentSpec("--query", "Filter by text query.", value_name="TEXT"),
            _CliArgumentSpec("--kind", "Filter by asset kind.", value_name="KIND"),
            _CliArgumentSpec("--category", "Filter by workflow category.", value_name="CATEGORY"),
            _CliArgumentSpec("--dataset", "Filter by dataset identity.", value_name="DATASET"),
            _CliArgumentSpec("--reference", "Filter by reference model.", value_name="REFERENCE"),
            _CliArgumentSpec(
                "--limit",
                "Limit the returned page.",
                value_name="INTEGER",
                expected="An integer page limit.",
                value_kind="int",
            ),
            _CliArgumentSpec(
                "--offset",
                "Start the returned page at this offset.",
                value_name="INTEGER",
                expected="A non-negative integer offset.",
                value_kind="int",
            ),
        ),
    ),
    ("preflight",): _CliCommandSpec(("preflight",), "Validate and freeze an execution plan."),
    ("preflight", "retarget"): _CliCommandSpec(
        ("preflight", "retarget"),
        "Validate one retarget request and freeze an immutable plan.",
        options=(_REQUEST_ARGUMENT,),
    ),
    ("preflight", "r2r"): _CliCommandSpec(
        ("preflight", "r2r"),
        "Validate one robot-to-robot request and freeze both robot identities.",
        options=(_REQUEST_ARGUMENT,),
    ),
    ("preflight", "batch"): _CliCommandSpec(
        ("preflight", "batch"),
        "Freeze ordered ready child plans into one H2R or R2R batch.",
        options=(_REQUEST_ARGUMENT,),
    ),
    ("calibration",): _CliCommandSpec(
        ("calibration",),
        "Inspect, propose, validate, or save robot calibration candidates.",
    ),
    ("calibration", "status"): _CliCommandSpec(
        ("calibration", "status"),
        "Inspect one content-bound robot/reference calibration.",
        options=(_REQUEST_ARGUMENT,),
    ),
    ("calibration", "propose"): _CliCommandSpec(
        ("calibration", "propose"),
        "Generate or revise a constrained calibration candidate.",
        options=(_REQUEST_ARGUMENT,),
    ),
    ("calibration", "validate"): _CliCommandSpec(
        ("calibration", "validate"),
        "Revalidate a persisted calibration candidate.",
        options=(_REQUEST_ARGUMENT,),
    ),
    ("calibration", "save"): _CliCommandSpec(
        ("calibration", "save"),
        "Silently save a candidate only after deterministic validation.",
        options=(_REQUEST_ARGUMENT,),
    ),
    ("calibration", "r2r"): _CliCommandSpec(
        ("calibration", "r2r"),
        "Inspect, propose, validate, or save source/target pair calibration.",
    ),
    ("calibration", "r2r", "status"): _CliCommandSpec(
        ("calibration", "r2r", "status"),
        "Inspect one content-bound source/target pair calibration.",
        options=(_REQUEST_ARGUMENT,),
    ),
    ("calibration", "r2r", "propose"): _CliCommandSpec(
        ("calibration", "r2r", "propose"),
        "Generate or revise a constrained R2R target-pose candidate.",
        options=(_REQUEST_ARGUMENT,),
    ),
    ("calibration", "r2r", "validate"): _CliCommandSpec(
        ("calibration", "r2r", "validate"),
        "Revalidate a persisted R2R calibration candidate.",
        options=(_REQUEST_ARGUMENT,),
    ),
    ("calibration", "r2r", "save"): _CliCommandSpec(
        ("calibration", "r2r", "save"),
        "Silently save a valid pair candidate to the target user overlay.",
        options=(_REQUEST_ARGUMENT,),
    ),
    ("job",): _CliCommandSpec(
        ("job",),
        "Start, wait for, recover, inspect, cancel, or retry jobs.",
    ),
    ("job", "start"): _CliCommandSpec(
        ("job", "start"),
        "Submit one immutable preflight plan.",
        options=(_PLAN_ARGUMENT, _IDEMPOTENCY_ARGUMENT),
    ),
    ("job", "get"): _CliCommandSpec(
        ("job", "get"),
        "Return one compact revision-aware job snapshot.",
        positionals=(_JOB_ID_ARGUMENT,),
        options=(_AFTER_REVISION_ARGUMENT,),
    ),
    ("job", "wait"): _CliCommandSpec(
        ("job", "wait"),
        "Wait for one job to advance beyond a known revision.",
        positionals=(_JOB_ID_ARGUMENT,),
        options=(_WAIT_AFTER_REVISION_ARGUMENT, _WAIT_TIMEOUT_ARGUMENT),
    ),
    ("job", "lookup"): _CliCommandSpec(
        ("job", "lookup"),
        "Recover one caller-owned submission without enumerating jobs.",
        options=(_PLAN_ARGUMENT, _IDEMPOTENCY_ARGUMENT, _AFTER_REVISION_ARGUMENT),
    ),
    ("job", "cancel"): _CliCommandSpec(
        ("job", "cancel"),
        "Request queued or cooperative-running cancellation.",
        positionals=(_JOB_ID_ARGUMENT,),
    ),
    ("job", "retry"): _CliCommandSpec(
        ("job", "retry"),
        "Create one idempotent child attempt for a terminal job.",
        positionals=(_JOB_ID_ARGUMENT,),
        options=(_IDEMPOTENCY_ARGUMENT,),
    ),
    ("artifact",): _CliCommandSpec(("artifact",), "List, verify, or download job artifacts."),
    ("artifact", "list"): _CliCommandSpec(
        ("artifact", "list"),
        "List a bounded page of artifacts attached to one job.",
        positionals=(_JOB_ID_ARGUMENT,),
        options=(
            _CliArgumentSpec(
                "--limit",
                "Limit the returned page.",
                value_name="INTEGER",
                expected="An integer page limit.",
                value_kind="int",
            ),
            _CliArgumentSpec(
                "--offset",
                "Start the returned page at this offset.",
                value_name="INTEGER",
                expected="A non-negative integer offset.",
                value_kind="int",
            ),
        ),
    ),
    ("artifact", "get"): _CliCommandSpec(
        ("artifact", "get"),
        "Return one descriptor and optionally download its verified bytes.",
        positionals=(_JOB_ID_ARGUMENT, _ARTIFACT_ID_ARGUMENT),
        options=(
            _CliArgumentSpec("--verify", "Verify managed bytes before returning the descriptor."),
            _CliArgumentSpec(
                "--output",
                "Download artifact bytes to this destination.",
                value_name="PATH",
                expected="A destination path supplied by the caller.",
            ),
            _CliArgumentSpec("--force", "Replace an existing --output destination."),
        ),
    ),
    ("legacy",): _CliCommandSpec(("legacy",), "Upgrade supported legacy Agent documents."),
    ("legacy", "upgrade"): _CliCommandSpec(
        ("legacy", "upgrade"),
        "Upgrade one legacy JobSpec v1 document through preflight.",
        options=(_REQUEST_ARGUMENT,),
    ),
}

_GLOBAL_HELP_ARGUMENTS = (
    _CliArgumentSpec("--json", "Compatibility flag; Agent CLI output is always strict JSON."),
    _CliArgumentSpec(
        "--base-url",
        "Override the resident Agent REST base URL.",
        value_name="URL",
        expected="A configured Agent REST base URL.",
    ),
    _CliArgumentSpec(
        "--timeout",
        "Set the request timeout in seconds.",
        value_name="SECONDS",
        expected="A finite number from 0.1 through 3600.",
        value_kind="float",
    ),
    _CliArgumentSpec("--help", "Return machine-readable JSON help without contacting the service."),
    _CliArgumentSpec("-h", "Alias for --help."),
)


def _command_name(path: tuple[str, ...]) -> AgentCliCommandName:
    return cast(AgentCliCommandName, " ".join(("hhtools", "agent", *path)))


def _child_specs(spec: _CliCommandSpec) -> list[_CliCommandSpec]:
    child_length = len(spec.path) + 1
    return [
        candidate
        for path, candidate in _COMMAND_SPECS.items()
        if len(path) == child_length and path[: len(spec.path)] == spec.path
    ]


def _usage(spec: _CliCommandSpec) -> str:
    parts: list[str] = [_command_name(spec.path)]
    if _child_specs(spec):
        parts.append("COMMAND")
    for argument in spec.positionals:
        parts.append(argument.name if argument.required else f"[{argument.name}]")
    for argument in spec.options:
        rendered: str = argument.name
        if argument.value_name is not None:
            rendered = f"{rendered} {argument.value_name}"
        parts.append(rendered if argument.required else f"[{rendered}]")
    parts.extend(("[--json]", "[--base-url URL]", "[--timeout SECONDS]", "[--help]"))
    return " ".join(parts)


def _help_argument(argument: _CliArgumentSpec) -> AgentCliHelpArgument:
    return AgentCliHelpArgument(
        name=argument.name,
        value_name=argument.value_name,
        required=argument.required,
        description=argument.description,
    )


def _command_spec_from_argv(arguments: Sequence[str]) -> _CliCommandSpec:
    """Resolve only static command tokens, skipping global option values."""

    path: tuple[str, ...] = ()
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value in {"--json", "--help", "-h"}:
            index += 1
            continue
        if value in {"--base-url", "--timeout"}:
            index += 2
            continue
        if value.startswith(("--base-url=", "--timeout=")):
            index += 1
            continue
        children = {
            candidate.path[-1]: candidate for candidate in _child_specs(_COMMAND_SPECS[path])
        }
        child = children.get(value)
        if child is None:
            break
        path = child.path
        index += 1
        if not _child_specs(child):
            break
    return _COMMAND_SPECS[path]


def _help_document(arguments: Sequence[str]) -> AgentCliHelp | None:
    if not any(value in {"--help", "-h"} for value in arguments):
        return None
    spec = _command_spec_from_argv(arguments)
    return AgentCliHelp(
        command=_command_name(spec.path),
        summary=spec.summary,
        usage=_usage(spec),
        positionals=[_help_argument(argument) for argument in spec.positionals],
        options=[_help_argument(argument) for argument in (*spec.options, *_GLOBAL_HELP_ARGUMENTS)],
        subcommands=[
            AgentCliHelpSubcommand(name=child.path[-1], summary=child.summary)
            for child in _child_specs(spec)
        ],
    )


class _ArgumentError(RuntimeError):
    def __init__(
        self,
        reason_code: AgentCliReasonCode = "INVALID_VALUE",
        *,
        argument: AgentCliDiagnosticArgument | None = None,
        expected: str | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.argument = argument
        self.expected = expected
        super().__init__(reason_code)


class _JsonArgumentParser(argparse.ArgumentParser):
    """Raise parse failures so stdout can remain one structured document."""

    def error(self, message: str) -> Never:
        del message
        raise _ArgumentError("INVALID_VALUE")

    def exit(self, status: int = 0, message: str | None = None) -> Never:
        del status, message
        raise _ArgumentError("INVALID_VALUE")


def _expected_subcommands(spec: _CliCommandSpec) -> str:
    names = ", ".join(child.path[-1] for child in _child_specs(spec))
    return f"One of: {names}."


def _diagnostic(
    spec: _CliCommandSpec,
    reason_code: AgentCliReasonCode,
    *,
    argument: AgentCliDiagnosticArgument | None = None,
    expected: str | None = None,
) -> AgentCliArgumentDiagnostic:
    return AgentCliArgumentDiagnostic(
        reason_code=reason_code,
        command=_command_name(spec.path),
        argument=argument,
        expected=expected,
        usage=_usage(spec),
    )


def _diagnose_parse_failure(  # noqa: PLR0911 - one safe branch per parser failure
    arguments: Sequence[str],
) -> AgentCliArgumentDiagnostic:
    """Classify argparse rejection without copying any caller-supplied token."""

    spec = _COMMAND_SPECS[()]
    index = 0
    while children := _child_specs(spec):
        if index >= len(arguments):
            return _diagnostic(
                spec,
                "MISSING_COMMAND",
                argument="COMMAND",
                expected=_expected_subcommands(spec),
            )
        by_name = {child.path[-1]: child for child in children}
        child = by_name.get(arguments[index])
        if child is None:
            if arguments[index].startswith("-"):
                return _diagnostic(
                    spec,
                    "UNKNOWN_ARGUMENT",
                    argument="<unrecognized>",
                    expected="Only the documented global options are accepted here.",
                )
            return _diagnostic(
                spec,
                "UNKNOWN_COMMAND",
                argument="COMMAND",
                expected=_expected_subcommands(spec),
            )
        spec = child
        index += 1

    option_by_name: dict[str, _CliArgumentSpec] = {
        argument.name: argument for argument in spec.options
    }
    seen_options: set[str] = set()
    positional_count = 0
    positional_only = False
    while index < len(arguments):
        raw = arguments[index]
        if raw == "--" and not positional_only:
            positional_only = True
            index += 1
            continue
        if not positional_only and raw.startswith("-"):
            name, separator, inline_value = raw.partition("=")
            argument = option_by_name.get(name)
            if argument is None:
                return _diagnostic(
                    spec,
                    "UNKNOWN_ARGUMENT",
                    argument="<unrecognized>",
                    expected="Only the documented options and positional arguments are accepted.",
                )
            seen_options.add(name)
            if argument.value_name is None:
                if separator:
                    return _diagnostic(
                        spec,
                        "INVALID_VALUE",
                        argument=argument.name,
                        expected=f"{argument.name} is a flag and does not take a value.",
                    )
                index += 1
                continue
            if separator:
                if not inline_value:
                    return _diagnostic(
                        spec,
                        "MISSING_VALUE",
                        argument=argument.name,
                        expected=argument.expected or f"A value for {argument.name}.",
                    )
                value = inline_value
                index += 1
            else:
                if index + 1 >= len(arguments) or arguments[index + 1].startswith("-"):
                    return _diagnostic(
                        spec,
                        "MISSING_VALUE",
                        argument=argument.name,
                        expected=argument.expected or f"A value for {argument.name}.",
                    )
                value = arguments[index + 1]
                index += 2
            try:
                if argument.value_kind == "int":
                    int(value)
                elif argument.value_kind == "float":
                    parsed = float(value)
                    if not math.isfinite(parsed):
                        raise ValueError
            except ValueError:
                return _diagnostic(
                    spec,
                    "INVALID_VALUE",
                    argument=argument.name,
                    expected=argument.expected or f"A valid value for {argument.name}.",
                )
            continue
        positional_count += 1
        index += 1

    for argument in spec.options:
        if argument.required and argument.name not in seen_options:
            return _diagnostic(
                spec,
                "MISSING_ARGUMENT",
                argument=argument.name,
                expected=argument.expected or f"The required {argument.name} option.",
            )
    if positional_count < len(spec.positionals):
        argument = spec.positionals[positional_count]
        return _diagnostic(
            spec,
            "MISSING_ARGUMENT",
            argument=argument.name,
            expected=argument.expected,
        )
    if positional_count > len(spec.positionals):
        return _diagnostic(
            spec,
            "UNEXPECTED_ARGUMENT",
            argument="<unrecognized>",
            expected="No additional positional arguments.",
        )
    return _diagnostic(
        spec,
        "INVALID_VALUE",
        argument="<unrecognized>",
        expected="Arguments matching the documented command usage.",
    )


def _argument_api_error(
    error: _ArgumentError,
    arguments: Sequence[str],
) -> ApiError:
    if error.reason_code == "INVALID_VALUE" and error.argument is None:
        diagnostic = _diagnose_parse_failure(arguments)
    else:
        diagnostic = _diagnostic(
            _command_spec_from_argv(arguments),
            error.reason_code,
            argument=error.argument,
            expected=error.expected,
        )
    return ApiError(
        code="INVALID_PARAMETER",
        message="The Agent command arguments are invalid.",
        stage=ErrorStage.REQUEST,
        details=diagnostic.model_dump(mode="json", exclude_none=True),
    )


def _parser() -> _JsonArgumentParser:
    parser = _JsonArgumentParser(prog="hhtools agent", add_help=False)
    commands = parser.add_subparsers(dest="group", required=True)

    capabilities = commands.add_parser("capabilities", add_help=False)
    capabilities.set_defaults(operation="capabilities")

    asset = commands.add_parser("asset", add_help=False)
    asset_commands = asset.add_subparsers(dest="asset_command", required=True)
    register = asset_commands.add_parser("register", add_help=False)
    register.add_argument("--request", required=True)
    register.set_defaults(operation="asset_register")
    get_asset = asset_commands.add_parser("get", add_help=False)
    get_asset.add_argument("asset_id")
    get_asset.set_defaults(operation="asset_get")
    inspect = asset_commands.add_parser("inspect", add_help=False)
    inspect.add_argument("asset_id")
    inspect.add_argument("--no-verify-hashes", action="store_false", dest="verify_hashes")
    inspect.add_argument("--no-parse-content", action="store_false", dest="parse_content")
    inspect.set_defaults(operation="asset_inspect")
    search = asset_commands.add_parser("search", add_help=False)
    search.add_argument("--query")
    search.add_argument("--kind")
    search.add_argument("--category")
    search.add_argument("--dataset")
    search.add_argument("--reference")
    search.add_argument("--limit", type=int, default=100)
    search.add_argument("--offset", type=int, default=0)
    search.set_defaults(operation="asset_search")
    catalog = asset_commands.add_parser("catalog", add_help=False)
    catalog.add_argument("--root-id")
    catalog.add_argument("--query")
    catalog.add_argument("--kind")
    catalog.add_argument("--limit", type=int, default=100)
    catalog.add_argument("--offset", type=int, default=0)
    catalog.set_defaults(operation="asset_catalog")

    preflight = commands.add_parser("preflight", add_help=False)
    preflight_commands = preflight.add_subparsers(dest="preflight_command", required=True)
    retarget = preflight_commands.add_parser("retarget", add_help=False)
    retarget.add_argument("--request", required=True)
    retarget.set_defaults(operation="preflight_retarget")
    r2r = preflight_commands.add_parser("r2r", add_help=False)
    r2r.add_argument("--request", required=True)
    r2r.set_defaults(operation="preflight_r2r")
    batch = preflight_commands.add_parser("batch", add_help=False)
    batch.add_argument("--request", required=True)
    batch.set_defaults(operation="preflight_batch")

    calibration = commands.add_parser("calibration", add_help=False)
    calibration_commands = calibration.add_subparsers(
        dest="calibration_command",
        required=True,
    )
    calibration_status = calibration_commands.add_parser("status", add_help=False)
    calibration_status.add_argument("--request", required=True)
    calibration_status.set_defaults(operation="calibration_status")
    calibration_propose = calibration_commands.add_parser("propose", add_help=False)
    calibration_propose.add_argument("--request", required=True)
    calibration_propose.set_defaults(operation="calibration_propose")
    calibration_validate = calibration_commands.add_parser("validate", add_help=False)
    calibration_validate.add_argument("--request", required=True)
    calibration_validate.set_defaults(operation="calibration_validate")
    calibration_save = calibration_commands.add_parser("save", add_help=False)
    calibration_save.add_argument("--request", required=True)
    calibration_save.set_defaults(operation="calibration_save")
    calibration_r2r = calibration_commands.add_parser("r2r", add_help=False)
    calibration_r2r_commands = calibration_r2r.add_subparsers(
        dest="r2r_calibration_command",
        required=True,
    )
    for command in ("status", "propose", "validate", "save"):
        command_parser = calibration_r2r_commands.add_parser(command, add_help=False)
        command_parser.add_argument("--request", required=True)
        command_parser.set_defaults(operation=f"r2r_calibration_{command}")

    job = commands.add_parser("job", add_help=False)
    job_commands = job.add_subparsers(dest="job_command", required=True)
    start = job_commands.add_parser("start", add_help=False)
    start.add_argument("--plan", required=True)
    start.add_argument("--idempotency-key", required=True)
    start.set_defaults(operation="job_start")
    get_job = job_commands.add_parser("get", add_help=False)
    get_job.add_argument("job_id")
    get_job.add_argument("--after-revision", type=int)
    get_job.set_defaults(operation="job_get")
    wait_job = job_commands.add_parser("wait", add_help=False)
    wait_job.add_argument("job_id")
    wait_job.add_argument("--after-revision", type=int, required=True)
    wait_job.add_argument("--wait-timeout", type=float, default=20.0)
    wait_job.set_defaults(operation="job_wait")
    lookup_job = job_commands.add_parser("lookup", add_help=False)
    lookup_job.add_argument("--plan", required=True)
    lookup_job.add_argument("--idempotency-key", required=True)
    lookup_job.add_argument("--after-revision", type=int)
    lookup_job.set_defaults(operation="job_lookup")
    cancel = job_commands.add_parser("cancel", add_help=False)
    cancel.add_argument("job_id")
    cancel.set_defaults(operation="job_cancel")
    retry = job_commands.add_parser("retry", add_help=False)
    retry.add_argument("job_id")
    retry.add_argument("--idempotency-key", required=True)
    retry.set_defaults(operation="job_retry")

    artifact = commands.add_parser("artifact", add_help=False)
    artifact_commands = artifact.add_subparsers(dest="artifact_command", required=True)
    list_artifacts = artifact_commands.add_parser("list", add_help=False)
    list_artifacts.add_argument("job_id")
    list_artifacts.add_argument("--limit", type=int, default=100)
    list_artifacts.add_argument("--offset", type=int, default=0)
    list_artifacts.set_defaults(operation="artifact_list")
    get_artifact = artifact_commands.add_parser("get", add_help=False)
    get_artifact.add_argument("job_id")
    get_artifact.add_argument("artifact_id")
    get_artifact.add_argument("--verify", action="store_true")
    get_artifact.add_argument("--output", type=Path)
    get_artifact.add_argument("--force", action="store_true")
    get_artifact.set_defaults(operation="artifact_get")

    legacy = commands.add_parser("legacy", add_help=False)
    legacy_commands = legacy.add_subparsers(dest="legacy_command", required=True)
    upgrade = legacy_commands.add_parser("upgrade", add_help=False)
    upgrade.add_argument("--request", required=True)
    upgrade.set_defaults(operation="legacy_upgrade")
    return parser


def _extract_global_option(
    arguments: list[str],
    name: str,
    *,
    default: str,
) -> tuple[list[str], str]:
    """Allow connection options before or after nested command names."""

    remaining: list[str] = []
    selected: str | None = None
    expected = next(
        argument.expected for argument in _GLOBAL_HELP_ARGUMENTS if argument.name == name
    )
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == name:
            if selected is not None:
                raise _ArgumentError(
                    "DUPLICATE_ARGUMENT",
                    argument=cast(AgentCliDiagnosticArgument, name),
                    expected=f"Provide {name} at most once.",
                )
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise _ArgumentError(
                    "MISSING_VALUE",
                    argument=cast(AgentCliDiagnosticArgument, name),
                    expected=expected,
                )
            selected = arguments[index + 1]
            index += 2
            continue
        prefix = f"{name}="
        if value.startswith(prefix):
            if selected is not None:
                raise _ArgumentError(
                    "DUPLICATE_ARGUMENT",
                    argument=cast(AgentCliDiagnosticArgument, name),
                    expected=f"Provide {name} at most once.",
                )
            selected = value[len(prefix) :]
            if not selected:
                raise _ArgumentError(
                    "MISSING_VALUE",
                    argument=cast(AgentCliDiagnosticArgument, name),
                    expected=expected,
                )
            index += 1
            continue
        remaining.append(value)
        index += 1
    return remaining, selected if selected is not None else default


def _normalize_argv(argv: Sequence[str]) -> tuple[list[str], str, float]:
    # ``--json`` is accepted in any position for parity with the documented
    # examples.  Agent commands are always strict JSON even when it is omitted.
    arguments = [value for value in argv if value != "--json"]
    arguments, base_url = _extract_global_option(
        arguments,
        "--base-url",
        default=os.environ.get("HHTOOLS_AGENT_BASE_URL", DEFAULT_BASE_URL),
    )
    arguments, raw_timeout = _extract_global_option(
        arguments,
        "--timeout",
        default=os.environ.get("HHTOOLS_AGENT_TIMEOUT", "30"),
    )
    try:
        timeout = float(raw_timeout)
    except ValueError as error:
        raise _ArgumentError(
            "INVALID_VALUE",
            argument="--timeout",
            expected="A finite number from 0.1 through 3600.",
        ) from error
    if not math.isfinite(timeout) or not 0.1 <= timeout <= 3_600:
        raise _ArgumentError(
            "INVALID_VALUE",
            argument="--timeout",
            expected="A finite number from 0.1 through 3600.",
        )
    return arguments, base_url, timeout


def _request_document(location: str, stdin: TextIO) -> Any:
    if location == "-":
        raw = stdin.read(_MAX_REQUEST_BYTES + 1)
    else:
        try:
            with Path(location).open("r", encoding="utf-8") as stream:
                raw = stream.read(_MAX_REQUEST_BYTES + 1)
        except UnicodeError as error:
            raise _ArgumentError(
                "REQUEST_ENCODING_INVALID",
                argument="--request",
                expected="A request encoded as UTF-8 JSON.",
            ) from error
        except OSError as error:
            raise _ArgumentError(
                "REQUEST_FILE_UNREADABLE",
                argument="--request",
                expected="A readable UTF-8 JSON file, or '-' for stdin.",
            ) from error
    try:
        encoded_size = len(raw.encode("utf-8"))
    except UnicodeError as error:
        raise _ArgumentError(
            "REQUEST_ENCODING_INVALID",
            argument="--request",
            expected="A request encoded as UTF-8 JSON.",
        ) from error
    if encoded_size > _MAX_REQUEST_BYTES:
        raise _ArgumentError(
            "REQUEST_TOO_LARGE",
            argument="--request",
            expected="A UTF-8 JSON request no larger than 8 MiB.",
        )
    try:
        return loads_strict_json(raw)
    except StrictJsonError as error:
        raise _ArgumentError(
            "REQUEST_JSON_INVALID",
            argument="--request",
            expected="Exactly one strict UTF-8 JSON document.",
        ) from error


def _validated_request[ContractT: BaseModel](
    model: type[ContractT], location: str, stdin: TextIO
) -> ContractT:
    try:
        return model.model_validate(_request_document(location, stdin))
    except ValidationError as error:
        raise _ArgumentError(
            "REQUEST_CONTRACT_INVALID",
            argument="--request",
            expected=f"A JSON document matching the {model.__name__} contract.",
        ) from error


def _response[ContractT: BaseModel](model: type[ContractT], payload: Any) -> ContractT:
    try:
        return model.model_validate(payload)
    except ValidationError as error:
        raise AgentTransportError(
            ApiError(
                code="REMOTE_PROTOCOL_ERROR",
                message="The Agent service response does not match the versioned contract.",
                retryable=True,
                stage=ErrorStage.INTERNAL,
            )
        ) from error


def _path_segment(value: str) -> str:
    return quote(value, safe="")


def _execute(  # noqa: PLR0911 - one explicit branch per public CLI operation
    namespace: argparse.Namespace,
    transport: AgentTransport,
    *,
    stdin: TextIO,
) -> BaseModel:
    operation = namespace.operation
    if operation == "capabilities":
        return _response(CapabilityResponse, transport.request_json("GET", "/capabilities"))

    if operation == "asset_register":
        registration_request = _validated_request(
            AssetRegistrationRequest, namespace.request, stdin
        )
        return _response(
            AssetBundle,
            transport.request_json(
                "POST",
                "/assets",
                document=registration_request.model_dump(mode="json", exclude_none=True),
            ),
        )
    if operation == "asset_get":
        path = f"/assets/{_path_segment(namespace.asset_id)}"
        return _response(AssetBundle, transport.request_json("GET", path))
    if operation == "asset_inspect":
        path = f"/assets/{_path_segment(namespace.asset_id)}/inspect"
        return _response(
            AssetInspection,
            transport.request_json(
                "GET",
                path,
                query={
                    "verify_hashes": namespace.verify_hashes,
                    "parse_content": namespace.parse_content,
                },
            ),
        )
    if operation == "asset_search":
        return _response(
            AssetSearchResponse,
            transport.request_json(
                "GET",
                "/assets",
                query={
                    "query": namespace.query,
                    "kind": namespace.kind,
                    "category": namespace.category,
                    "dataset": namespace.dataset,
                    "reference": namespace.reference,
                    "limit": namespace.limit,
                    "offset": namespace.offset,
                },
            ),
        )

    if operation == "asset_catalog":
        try:
            request = AvailableAssetCatalogRequest(
                root_id=namespace.root_id,
                query=namespace.query,
                kind=namespace.kind,
                limit=namespace.limit,
                offset=namespace.offset,
            )
        except ValidationError as error:
            invalid_fields = {issue["loc"][0] for issue in error.errors() if issue["loc"]}
            argument_by_field: dict[str, AgentCliDiagnosticArgument] = {
                "root_id": "--root-id",
                "query": "--query",
                "kind": "--kind",
                "limit": "--limit",
                "offset": "--offset",
            }
            field = next(
                (name for name in argument_by_field if name in invalid_fields),
                "limit",
            )
            raise _ArgumentError(
                "INVALID_VALUE",
                argument=argument_by_field[field],
                expected={
                    "root_id": "A capability-advertised portable root id.",
                    "query": "A non-empty query no longer than 256 characters.",
                    "kind": "A supported asset kind.",
                    "limit": "An integer from 1 through 500.",
                    "offset": "A non-negative integer offset.",
                }[field],
            ) from error
        return _response(
            AvailableAssetCatalogResponse,
            transport.request_json(
                "GET",
                "/assets/available",
                query=request.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude={"schema_version"},
                ),
            ),
        )

    if operation == "preflight_retarget":
        preflight_request = _validated_request(RetargetPreflightRequest, namespace.request, stdin)
        return _response(
            PreflightResponse,
            transport.request_json(
                "POST",
                "/preflight/retarget",
                document=preflight_request.model_dump(mode="json", exclude_none=True),
            ),
        )

    if operation == "preflight_r2r":
        preflight_request = _validated_request(R2RPreflightRequest, namespace.request, stdin)
        return _response(
            R2RPreflightResponse,
            transport.request_json(
                "POST",
                "/preflight/r2r",
                document=preflight_request.model_dump(mode="json", exclude_none=True),
            ),
        )

    if operation == "preflight_batch":
        preflight_request = _validated_request(BatchPreflightRequest, namespace.request, stdin)
        return _response(
            BatchPreflightResponse,
            transport.request_json(
                "POST",
                "/preflight/batch",
                document=preflight_request.model_dump(mode="json", exclude_none=True),
            ),
        )

    if operation == "calibration_status":
        calibration_request = _validated_request(
            CalibrationStatusRequest,
            namespace.request,
            stdin,
        )
        return _response(
            CalibrationStatusResponse,
            transport.request_json(
                "POST",
                "/calibrations/status",
                document=calibration_request.model_dump(mode="json", exclude_none=True),
            ),
        )
    if operation == "calibration_propose":
        calibration_request = _validated_request(
            CalibrationProposalRequest,
            namespace.request,
            stdin,
        )
        return _response(
            CalibrationProposalResponse,
            transport.request_json(
                "POST",
                "/calibrations/proposals",
                document=calibration_request.model_dump(mode="json", exclude_none=True),
            ),
        )
    if operation == "calibration_validate":
        calibration_request = _validated_request(
            CalibrationValidationRequest,
            namespace.request,
            stdin,
        )
        return _response(
            CalibrationValidationReport,
            transport.request_json(
                "POST",
                "/calibrations/validate",
                document=calibration_request.model_dump(mode="json", exclude_none=True),
            ),
        )
    if operation == "calibration_save":
        calibration_request = _validated_request(
            CalibrationSaveRequest,
            namespace.request,
            stdin,
        )
        return _response(
            CalibrationSaveReceipt,
            transport.request_json(
                "POST",
                "/calibrations/save",
                document=calibration_request.model_dump(mode="json", exclude_none=True),
            ),
        )

    if operation == "r2r_calibration_status":
        r2r_calibration_request = _validated_request(
            R2RCalibrationStatusRequest,
            namespace.request,
            stdin,
        )
        return _response(
            R2RCalibrationStatusResponse,
            transport.request_json(
                "POST",
                "/calibrations/r2r/status",
                document=r2r_calibration_request.model_dump(mode="json", exclude_none=True),
            ),
        )
    if operation == "r2r_calibration_propose":
        r2r_calibration_request = _validated_request(
            R2RCalibrationProposalRequest,
            namespace.request,
            stdin,
        )
        return _response(
            R2RCalibrationProposalResponse,
            transport.request_json(
                "POST",
                "/calibrations/r2r/proposals",
                document=r2r_calibration_request.model_dump(mode="json", exclude_none=True),
            ),
        )
    if operation == "r2r_calibration_validate":
        r2r_calibration_request = _validated_request(
            R2RCalibrationValidationRequest,
            namespace.request,
            stdin,
        )
        return _response(
            CalibrationValidationReport,
            transport.request_json(
                "POST",
                "/calibrations/r2r/validate",
                document=r2r_calibration_request.model_dump(mode="json", exclude_none=True),
            ),
        )
    if operation == "r2r_calibration_save":
        r2r_calibration_request = _validated_request(
            R2RCalibrationSaveRequest,
            namespace.request,
            stdin,
        )
        return _response(
            R2RCalibrationSaveReceipt,
            transport.request_json(
                "POST",
                "/calibrations/r2r/save",
                document=r2r_calibration_request.model_dump(mode="json", exclude_none=True),
            ),
        )
    if operation == "job_start":
        try:
            start_request = JobStartRequest(
                plan_id=namespace.plan,
                idempotency_key=namespace.idempotency_key,
            )
        except ValidationError as error:
            invalid_fields = {issue["loc"][0] for issue in error.errors() if issue["loc"]}
            argument = _PLAN_ARGUMENT if "plan_id" in invalid_fields else _IDEMPOTENCY_ARGUMENT
            raise _ArgumentError(
                "INVALID_VALUE",
                argument=argument.name,
                expected=argument.expected,
            ) from error
        return _response(
            AgentJobView,
            transport.request_json(
                "POST",
                "/jobs",
                document=start_request.model_dump(mode="json", exclude_none=True),
            ),
        )
    if operation == "job_get":
        path = f"/jobs/{_path_segment(namespace.job_id)}"
        return _response(
            AgentJobView,
            transport.request_json("GET", path, query={"after_revision": namespace.after_revision}),
        )
    if operation == "job_wait":
        if (
            isinstance(namespace.wait_timeout, bool)
            or not math.isfinite(namespace.wait_timeout)
            or not 0 <= namespace.wait_timeout <= 60
        ):
            raise _ArgumentError(
                "INVALID_VALUE",
                argument=_WAIT_TIMEOUT_ARGUMENT.name,
                expected=_WAIT_TIMEOUT_ARGUMENT.expected,
            )
        path = f"/jobs/{_path_segment(namespace.job_id)}/wait"
        return _response(
            AgentJobView,
            transport.request_json(
                "GET",
                path,
                query={
                    "after_revision": namespace.after_revision,
                    "timeout": namespace.wait_timeout,
                },
            ),
        )
    if operation == "job_lookup":
        try:
            lookup_request = JobLookupRequest(
                plan_id=namespace.plan,
                idempotency_key=namespace.idempotency_key,
                after_revision=namespace.after_revision,
            )
        except ValidationError as error:
            invalid_fields = {issue["loc"][0] for issue in error.errors() if issue["loc"]}
            if "plan_id" in invalid_fields:
                argument = _PLAN_ARGUMENT
            elif "idempotency_key" in invalid_fields:
                argument = _IDEMPOTENCY_ARGUMENT
            else:
                argument = _AFTER_REVISION_ARGUMENT
            raise _ArgumentError(
                "INVALID_VALUE",
                argument=argument.name,
                expected=argument.expected,
            ) from error
        return _response(
            AgentJobView,
            transport.request_json(
                "POST",
                "/jobs/lookup",
                document=lookup_request.model_dump(mode="json", exclude_none=True),
            ),
        )
    if operation == "job_cancel":
        path = f"/jobs/{_path_segment(namespace.job_id)}/cancel"
        return _response(AgentJobView, transport.request_json("POST", path, document={}))
    if operation == "job_retry":
        try:
            retry_request = JobRetryRequest(idempotency_key=namespace.idempotency_key)
        except ValidationError as error:
            raise _ArgumentError(
                "INVALID_VALUE",
                argument=_IDEMPOTENCY_ARGUMENT.name,
                expected=_IDEMPOTENCY_ARGUMENT.expected,
            ) from error
        path = f"/jobs/{_path_segment(namespace.job_id)}/retry"
        return _response(
            AgentJobView,
            transport.request_json(
                "POST",
                path,
                document=retry_request.model_dump(mode="json", exclude_none=True),
            ),
        )

    if operation == "artifact_list":
        path = f"/jobs/{_path_segment(namespace.job_id)}/artifacts"
        return _response(
            ArtifactListResponse,
            transport.request_json(
                "GET", path, query={"limit": namespace.limit, "offset": namespace.offset}
            ),
        )
    if operation == "artifact_get":
        path = (
            f"/jobs/{_path_segment(namespace.job_id)}/artifacts/"
            f"{_path_segment(namespace.artifact_id)}"
        )
        descriptor = _response(
            ArtifactDescriptor,
            transport.request_json("GET", path, query={"verify": namespace.verify}),
        )
        if descriptor.job_id != namespace.job_id or descriptor.artifact_id != namespace.artifact_id:
            raise AgentTransportError(
                ApiError(
                    code="REMOTE_PROTOCOL_ERROR",
                    message="The Agent service returned a descriptor for another job or artifact.",
                    retryable=True,
                    stage=ErrorStage.INTERNAL,
                )
            )
        if namespace.output is not None:
            transport.download_artifact(
                f"{path}/content",
                destination=namespace.output,
                descriptor=descriptor,
                overwrite=namespace.force,
            )
        elif namespace.force:
            raise _ArgumentError(
                "INVALID_COMBINATION",
                argument="--force",
                expected="Use --force only together with --output PATH.",
            )
        return descriptor

    if operation == "legacy_upgrade":
        # The file is the historical v1 document (or its historical download
        # wrapper), not a second transport wrapper users must manufacture.  The
        # CLI adds the versioned request envelope before crossing REST.
        try:
            upgrade_request = LegacyJobUpgradeRequest(
                payload=_request_document(namespace.request, stdin)
            )
        except ValidationError as error:
            raise AgentTransportError(
                ApiError(
                    code="INVALID_PARAMETER",
                    message="The legacy request must contain one JSON object.",
                    stage=ErrorStage.REQUEST,
                )
            ) from error
        return _response(
            LegacyJobUpgradeResponse,
            transport.request_json(
                "POST",
                "/legacy/jobspec-v1/upgrade",
                document=upgrade_request.model_dump(mode="json", exclude_none=True),
            ),
        )
    raise RuntimeError("unknown Agent CLI operation")


def _error_exit_code(error: ApiError) -> int:
    if error.stage in {
        ErrorStage.REQUEST,
        ErrorStage.ASSET_REGISTRATION,
        ErrorStage.ASSET_INSPECTION,
    }:
        return EXIT_PARAMETER_ERROR
    if error.stage in {ErrorStage.CALIBRATION, ErrorStage.PREFLIGHT}:
        return EXIT_PREFLIGHT_ERROR
    if error.stage in {
        ErrorStage.ADMISSION,
        ErrorStage.EXECUTION,
        ErrorStage.EVALUATION,
        ErrorStage.ARTIFACT,
    }:
        return EXIT_JOB_ERROR
    return EXIT_INTERNAL_ERROR


def _result_exit_code(result: BaseModel) -> int:
    if isinstance(result, PreflightResponse | R2RPreflightResponse | BatchPreflightResponse):
        return EXIT_SUCCESS if result.status is PreflightStatus.READY else EXIT_PREFLIGHT_ERROR
    if isinstance(result, CalibrationProposalResponse):
        return EXIT_SUCCESS if result.validation.valid else EXIT_PREFLIGHT_ERROR
    if isinstance(result, CalibrationValidationReport):
        return EXIT_SUCCESS if result.valid else EXIT_PREFLIGHT_ERROR
    if isinstance(result, LegacyJobUpgradeResponse):
        return (
            EXIT_SUCCESS
            if result.preflight.status is PreflightStatus.READY
            else EXIT_PREFLIGHT_ERROR
        )
    return EXIT_SUCCESS


def _write_document(document: BaseModel, stdout: TextIO) -> bool:
    """Write one portable document, replacing unsafe responses as a whole."""

    safe = True
    try:
        payload = document.model_dump(mode="json", exclude_none=True)
        ensure_portable_json(payload)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (PortableJsonError, TypeError, ValueError, OverflowError):
        safe = False
        fallback = ApiError(
            code="INTERNAL_ERROR",
            message="The Agent response was not safe for portable JSON output.",
            retryable=False,
            stage=ErrorStage.INTERNAL,
        )
        encoded = fallback.model_dump_json(exclude_none=True)
    stdout.write(encoded)
    stdout.write("\n")
    stdout.flush()
    return safe


def _default_transport_factory(base_url: str, timeout: float) -> AgentTransport:
    return HttpAgentTransport(base_url, timeout_seconds=timeout)


def run(
    argv: Sequence[str] | None = None,
    *,
    transport_factory: TransportFactory | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run one command without allowing parser or protocol prose on stdout."""

    input_stream = stdin or sys.stdin
    output_stream = stdout or sys.stdout
    # Retained as an explicit dependency boundary for future progress logging;
    # no current command emits human prose during a successful invocation.
    _ = stderr or sys.stderr
    raw_arguments = list(argv or ())
    arguments: list[str] | None = None
    try:
        help_document = _help_document(raw_arguments)
        if help_document is not None:
            safe = _write_document(help_document, output_stream)
            return EXIT_SUCCESS if safe else EXIT_INTERNAL_ERROR
        arguments, base_url, timeout = _normalize_argv(raw_arguments)
        namespace = _parser().parse_args(arguments)
        if namespace.operation == "job_wait":
            if not math.isfinite(namespace.wait_timeout) or not 0 <= namespace.wait_timeout <= 60:
                raise _ArgumentError(
                    "INVALID_VALUE",
                    argument=_WAIT_TIMEOUT_ARGUMENT.name,
                    expected=_WAIT_TIMEOUT_ARGUMENT.expected,
                )
            if timeout <= namespace.wait_timeout:
                raise _ArgumentError(
                    "INVALID_COMBINATION",
                    argument="--timeout",
                    expected=(
                        "Set the Agent request --timeout higher than --wait-timeout "
                        "so the server can return its wait response."
                    ),
                )
        factory = transport_factory or _default_transport_factory
        transport = factory(base_url, timeout)
        result = _execute(namespace, transport, stdin=input_stream)
        safe = _write_document(result, output_stream)
        return _result_exit_code(result) if safe else EXIT_INTERNAL_ERROR
    except _ArgumentError as error:
        api_error = _argument_api_error(
            error,
            arguments if arguments is not None else raw_arguments,
        )
    except AgentTransportError as error:
        api_error = error.error
    except Exception:  # noqa: BLE001 - stdout must remain a JSON contract on failure
        api_error = ApiError(
            code="INTERNAL_ERROR",
            message="The JSON CLI could not complete the request.",
            retryable=True,
            stage=ErrorStage.INTERNAL,
        )
    safe = _write_document(api_error, output_stream)
    return _error_exit_code(api_error) if safe else EXIT_INTERNAL_ERROR


_PASSTHROUGH_CONTEXT = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}

app = typer.Typer(
    help="Call the versioned Agent API with strict JSON input and output.",
    add_completion=False,
    no_args_is_help=False,
    context_settings=_PASSTHROUGH_CONTEXT,
)


@app.callback(invoke_without_command=True)
def launch(ctx: typer.Context) -> None:
    """Return a JSON argument error when no operation was selected."""

    if ctx.invoked_subcommand is None:
        raise typer.Exit(code=run(ctx.args))


def _passthrough(prefix: Sequence[str], ctx: typer.Context) -> None:
    raise typer.Exit(code=run([*prefix, *ctx.args]))


@app.command("capabilities", context_settings=_PASSTHROUGH_CONTEXT)
def capabilities_command(ctx: typer.Context) -> None:
    _passthrough(["capabilities"], ctx)


asset_app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    context_settings=_PASSTHROUGH_CONTEXT,
)
app.add_typer(asset_app, name="asset")


@asset_app.callback(invoke_without_command=True)
def asset_group(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _passthrough(["asset"], ctx)


@asset_app.command("register", context_settings=_PASSTHROUGH_CONTEXT)
def asset_register_command(ctx: typer.Context) -> None:
    _passthrough(["asset", "register"], ctx)


@asset_app.command("get", context_settings=_PASSTHROUGH_CONTEXT)
def asset_get_command(ctx: typer.Context) -> None:
    _passthrough(["asset", "get"], ctx)


@asset_app.command("inspect", context_settings=_PASSTHROUGH_CONTEXT)
def asset_inspect_command(ctx: typer.Context) -> None:
    _passthrough(["asset", "inspect"], ctx)


@asset_app.command("search", context_settings=_PASSTHROUGH_CONTEXT)
def asset_search_command(ctx: typer.Context) -> None:
    _passthrough(["asset", "search"], ctx)


@asset_app.command("catalog", context_settings=_PASSTHROUGH_CONTEXT)
def asset_catalog_command(ctx: typer.Context) -> None:
    _passthrough(["asset", "catalog"], ctx)


preflight_app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    context_settings=_PASSTHROUGH_CONTEXT,
)
app.add_typer(preflight_app, name="preflight")


@preflight_app.callback(invoke_without_command=True)
def preflight_group(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _passthrough(["preflight"], ctx)


@preflight_app.command("retarget", context_settings=_PASSTHROUGH_CONTEXT)
def preflight_retarget_command(ctx: typer.Context) -> None:
    _passthrough(["preflight", "retarget"], ctx)


@preflight_app.command("r2r", context_settings=_PASSTHROUGH_CONTEXT)
def preflight_r2r_command(ctx: typer.Context) -> None:
    _passthrough(["preflight", "r2r"], ctx)


@preflight_app.command("batch", context_settings=_PASSTHROUGH_CONTEXT)
def preflight_batch_command(ctx: typer.Context) -> None:
    _passthrough(["preflight", "batch"], ctx)


calibration_app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    context_settings=_PASSTHROUGH_CONTEXT,
)
app.add_typer(calibration_app, name="calibration")


@calibration_app.callback(invoke_without_command=True)
def calibration_group(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _passthrough(["calibration"], ctx)


@calibration_app.command("status", context_settings=_PASSTHROUGH_CONTEXT)
def calibration_status_command(ctx: typer.Context) -> None:
    _passthrough(["calibration", "status"], ctx)


@calibration_app.command("propose", context_settings=_PASSTHROUGH_CONTEXT)
def calibration_propose_command(ctx: typer.Context) -> None:
    _passthrough(["calibration", "propose"], ctx)


@calibration_app.command("validate", context_settings=_PASSTHROUGH_CONTEXT)
def calibration_validate_command(ctx: typer.Context) -> None:
    _passthrough(["calibration", "validate"], ctx)


@calibration_app.command("save", context_settings=_PASSTHROUGH_CONTEXT)
def calibration_save_command(ctx: typer.Context) -> None:
    _passthrough(["calibration", "save"], ctx)


r2r_calibration_app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=False,
    context_settings=_PASSTHROUGH_CONTEXT,
)
calibration_app.add_typer(r2r_calibration_app, name="r2r")


@r2r_calibration_app.callback(invoke_without_command=True)
def r2r_calibration_group(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _passthrough(["calibration", "r2r"], ctx)


@r2r_calibration_app.command("status", context_settings=_PASSTHROUGH_CONTEXT)
def r2r_calibration_status_command(ctx: typer.Context) -> None:
    _passthrough(["calibration", "r2r", "status"], ctx)


@r2r_calibration_app.command("propose", context_settings=_PASSTHROUGH_CONTEXT)
def r2r_calibration_propose_command(ctx: typer.Context) -> None:
    _passthrough(["calibration", "r2r", "propose"], ctx)


@r2r_calibration_app.command("validate", context_settings=_PASSTHROUGH_CONTEXT)
def r2r_calibration_validate_command(ctx: typer.Context) -> None:
    _passthrough(["calibration", "r2r", "validate"], ctx)


@r2r_calibration_app.command("save", context_settings=_PASSTHROUGH_CONTEXT)
def r2r_calibration_save_command(ctx: typer.Context) -> None:
    _passthrough(["calibration", "r2r", "save"], ctx)


job_app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    context_settings=_PASSTHROUGH_CONTEXT,
)
app.add_typer(job_app, name="job")


@job_app.callback(invoke_without_command=True)
def job_group(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _passthrough(["job"], ctx)


@job_app.command("start", context_settings=_PASSTHROUGH_CONTEXT)
def job_start_command(ctx: typer.Context) -> None:
    _passthrough(["job", "start"], ctx)


@job_app.command("get", context_settings=_PASSTHROUGH_CONTEXT)
def job_get_command(ctx: typer.Context) -> None:
    _passthrough(["job", "get"], ctx)


@job_app.command("wait", context_settings=_PASSTHROUGH_CONTEXT)
def job_wait_command(ctx: typer.Context) -> None:
    _passthrough(["job", "wait"], ctx)


@job_app.command("lookup", context_settings=_PASSTHROUGH_CONTEXT)
def job_lookup_command(ctx: typer.Context) -> None:
    _passthrough(["job", "lookup"], ctx)


@job_app.command("cancel", context_settings=_PASSTHROUGH_CONTEXT)
def job_cancel_command(ctx: typer.Context) -> None:
    _passthrough(["job", "cancel"], ctx)


@job_app.command("retry", context_settings=_PASSTHROUGH_CONTEXT)
def job_retry_command(ctx: typer.Context) -> None:
    _passthrough(["job", "retry"], ctx)


artifact_app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    context_settings=_PASSTHROUGH_CONTEXT,
)
app.add_typer(artifact_app, name="artifact")


@artifact_app.callback(invoke_without_command=True)
def artifact_group(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _passthrough(["artifact"], ctx)


@artifact_app.command("list", context_settings=_PASSTHROUGH_CONTEXT)
def artifact_list_command(ctx: typer.Context) -> None:
    _passthrough(["artifact", "list"], ctx)


@artifact_app.command("get", context_settings=_PASSTHROUGH_CONTEXT)
def artifact_get_command(ctx: typer.Context) -> None:
    _passthrough(["artifact", "get"], ctx)


legacy_app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    context_settings=_PASSTHROUGH_CONTEXT,
)
app.add_typer(legacy_app, name="legacy")


@legacy_app.callback(invoke_without_command=True)
def legacy_group(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _passthrough(["legacy"], ctx)


@legacy_app.command("upgrade", context_settings=_PASSTHROUGH_CONTEXT)
def legacy_upgrade_command(ctx: typer.Context) -> None:
    _passthrough(["legacy", "upgrade"], ctx)


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone entry point used by tests and embedders."""

    return run(sys.argv[1:] if argv is None else argv)


__all__ = [
    "DEFAULT_BASE_URL",
    "EXIT_INTERNAL_ERROR",
    "EXIT_JOB_ERROR",
    "EXIT_PARAMETER_ERROR",
    "EXIT_PREFLIGHT_ERROR",
    "EXIT_SUCCESS",
    "app",
    "main",
    "run",
]
