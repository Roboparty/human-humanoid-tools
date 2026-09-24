"""Versioned, transport-safe documents emitted by the strict JSON CLI."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .common import ContractModel, SchemaVersion

AgentCliCommandName = Literal[
    "hhtools agent",
    "hhtools agent capabilities",
    "hhtools agent asset",
    "hhtools agent asset catalog",
    "hhtools agent asset register",
    "hhtools agent asset get",
    "hhtools agent asset inspect",
    "hhtools agent asset search",
    "hhtools agent calibration",
    "hhtools agent calibration propose",
    "hhtools agent calibration r2r",
    "hhtools agent calibration r2r propose",
    "hhtools agent calibration r2r save",
    "hhtools agent calibration r2r status",
    "hhtools agent calibration r2r validate",
    "hhtools agent calibration save",
    "hhtools agent calibration status",
    "hhtools agent calibration validate",
    "hhtools agent preflight",
    "hhtools agent preflight batch",
    "hhtools agent preflight r2r",
    "hhtools agent preflight retarget",
    "hhtools agent job",
    "hhtools agent job start",
    "hhtools agent job get",
    "hhtools agent job wait",
    "hhtools agent job lookup",
    "hhtools agent job cancel",
    "hhtools agent job retry",
    "hhtools agent artifact",
    "hhtools agent artifact list",
    "hhtools agent artifact get",
    "hhtools agent legacy",
    "hhtools agent legacy upgrade",
]
AgentCliDiagnosticArgument = Literal[
    "COMMAND",
    "ASSET_ID",
    "JOB_ID",
    "ARTIFACT_ID",
    "<unrecognized>",
    "--base-url",
    "--timeout",
    "--request",
    "--plan",
    "--idempotency-key",
    "--query",
    "--root-id",
    "--kind",
    "--category",
    "--dataset",
    "--reference",
    "--no-verify-hashes",
    "--no-parse-content",
    "--after-revision",
    "--wait-timeout",
    "--limit",
    "--offset",
    "--verify",
    "--output",
    "--force",
    "--json",
    "--help",
    "-h",
]
AgentCliReasonCode = Literal[
    "MISSING_COMMAND",
    "UNKNOWN_COMMAND",
    "MISSING_ARGUMENT",
    "MISSING_VALUE",
    "UNKNOWN_ARGUMENT",
    "UNEXPECTED_ARGUMENT",
    "DUPLICATE_ARGUMENT",
    "INVALID_VALUE",
    "INVALID_COMBINATION",
    "REQUEST_FILE_UNREADABLE",
    "REQUEST_ENCODING_INVALID",
    "REQUEST_TOO_LARGE",
    "REQUEST_JSON_INVALID",
    "REQUEST_CONTRACT_INVALID",
]


class AgentCliArgumentDiagnostic(ContractModel):
    """Sanitized CLI failure detail assembled only from static command metadata."""

    reason_code: AgentCliReasonCode
    command: AgentCliCommandName
    argument: AgentCliDiagnosticArgument | None = None
    expected: Annotated[str | None, Field(default=None, max_length=512)]
    usage: Annotated[str, Field(min_length=1, max_length=1_024)]


class AgentCliHelpArgument(ContractModel):
    """One documented positional or option in a JSON help response."""

    name: Annotated[
        str,
        Field(min_length=1, max_length=64, pattern=r"^(?:--?[a-z][a-z0-9-]*|[A-Z][A-Z0-9_]*)$"),
    ]
    value_name: Annotated[
        str | None,
        Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$"),
    ]
    required: bool = False
    description: Annotated[str, Field(min_length=1, max_length=512)]


class AgentCliHelpSubcommand(ContractModel):
    """One immediate child command in a JSON help response."""

    name: Annotated[
        str,
        Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9-]*$"),
    ]
    summary: Annotated[str, Field(min_length=1, max_length=512)]


class AgentCliHelp(ContractModel):
    """Machine-readable help that preserves the CLI's one-JSON-document invariant."""

    schema_version: SchemaVersion = SchemaVersion.V1
    kind: Literal["agent_cli_help"] = "agent_cli_help"
    command: AgentCliCommandName
    summary: Annotated[str, Field(min_length=1, max_length=512)]
    usage: Annotated[str, Field(min_length=1, max_length=1_024)]
    positionals: list[AgentCliHelpArgument] = Field(default_factory=list, max_length=8)
    options: list[AgentCliHelpArgument] = Field(default_factory=list, max_length=32)
    subcommands: list[AgentCliHelpSubcommand] = Field(default_factory=list, max_length=16)


__all__ = [
    "AgentCliArgumentDiagnostic",
    "AgentCliCommandName",
    "AgentCliDiagnosticArgument",
    "AgentCliHelp",
    "AgentCliHelpArgument",
    "AgentCliHelpSubcommand",
    "AgentCliReasonCode",
]
