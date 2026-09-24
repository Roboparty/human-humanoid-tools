"""Shared primitives for HHTools' public agent contracts.

The models in :mod:`hhtools.contracts` are transport-neutral.  REST, JSON CLI,
OpenAPI, and MCP adapters should serialize these models instead of defining
their own wire formats.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

from .portability import (
    PORTABLE_URI_HOST_PATTERN,
    PORTABLE_URI_PORT_PATTERN,
    PORTABLE_URI_TAIL_PATTERN,
    is_portable_next_action_url,
    is_portable_resource_uri,
)


class ContractModel(BaseModel):
    """Base class that rejects misspelled or unsupported wire fields."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class SchemaVersion(StrEnum):
    """Version of the transport-neutral HHTools agent schema."""

    V1 = "1.0"


MachineCode = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Z][A-Z0-9_]*$",
        description="Stable, English, machine-readable code.",
    ),
]

Sha256Hex = Annotated[
    str,
    Field(pattern=r"^[0-9a-f]{64}$", description="Lower-case SHA-256 content digest."),
]
AssetId = Annotated[
    str,
    Field(pattern=r"^asset:sha256:[0-9a-f]{64}$", description="Content-addressed asset id."),
]
PlanId = Annotated[
    str,
    Field(pattern=r"^plan:sha256:[0-9a-f]{64}$", description="Content-addressed plan id."),
]
CalibrationId = Annotated[
    str,
    Field(pattern=r"^cal:sha256:[0-9a-f]{64}$", description="Content-addressed calibration id."),
]
ArtifactId = Annotated[
    str,
    Field(
        pattern=r"^artifact:[a-z][a-z0-9_-]*:[A-Za-z0-9._~-]+$",
        description="Artifact id with a stable kind namespace.",
    ),
]

_HTTP_URI = (
    rf"https?://{PORTABLE_URI_HOST_PATTERN}"
    rf"(?::{PORTABLE_URI_PORT_PATTERN})?{PORTABLE_URI_TAIL_PATTERN}"
)
_HTTPS_URI = (
    rf"https://{PORTABLE_URI_HOST_PATTERN}"
    rf"(?::{PORTABLE_URI_PORT_PATTERN})?{PORTABLE_URI_TAIL_PATTERN}"
)
_HHTOOLS_ARTIFACT_URI = r"hhtools://jobs/[A-Za-z0-9._~:-]+/artifacts/[A-Za-z0-9._~:-]+"
_UI_QUERY_PAIR = r"(?:calibrate|panel|robot|view)=[^&#\s]{0,256}"
_UI_QUERY = rf"\?(?:{_UI_QUERY_PAIR}(?:&{_UI_QUERY_PAIR})*)?"
_LOCAL_UI_URL = (
    rf"(?:/(?:{_UI_QUERY})?|http://(?:127\.0\.0\.1|localhost|\[::1\]):"
    rf"{PORTABLE_URI_PORT_PATTERN}/(?:{_UI_QUERY})?)"
)
_RESOURCE_URI_PATTERN = rf"^(?:{_HTTP_URI}|{_HHTOOLS_ARTIFACT_URI})$"
_NEXT_ACTION_URL_PATTERN = rf"^(?:{_LOCAL_UI_URL}|{_HTTPS_URI})$"


def _validate_resource_uri(value: str) -> str:
    if not is_portable_resource_uri(value):
        raise ValueError("resource URI must be canonical and host independent")
    return value


ResourceUri = Annotated[
    str,
    Field(
        min_length=1,
        pattern=_RESOURCE_URI_PATTERN,
        description=("Canonical job-scoped HHTools artifact URI or portable HTTP(S) URI."),
    ),
    AfterValidator(_validate_resource_uri),
]


class NextAction(ContractModel):
    """An explicit recovery or continuation step for an agent or human."""

    actor: Literal["agent", "human", "system"]
    action: Annotated[
        str,
        Field(
            min_length=1,
            max_length=128,
            pattern=r"^[a-z][a-z0-9_]*$",
            description="Stable, English action identifier.",
        ),
    ]
    message: str | None = Field(
        default=None,
        description="Optional human-readable instruction.",
    )
    url: str | None = Field(
        default=None,
        pattern=_NEXT_ACTION_URL_PATTERN,
        description=(
            "Optional allowlisted local calibration UI route or portable HTTPS documentation URL."
        ),
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured parameters needed to perform the action.",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is not None and not is_portable_next_action_url(value):
            raise ValueError(
                "url must be an allowlisted local UI route or portable HTTPS documentation"
            )
        return value


class ErrorStage(StrEnum):
    """Stable stage in which an API error occurred."""

    REQUEST = "request"
    ASSET_REGISTRATION = "asset_registration"
    ASSET_INSPECTION = "asset_inspection"
    CALIBRATION = "calibration"
    PREFLIGHT = "preflight"
    ADMISSION = "admission"
    EXECUTION = "execution"
    EVALUATION = "evaluation"
    ARTIFACT = "artifact"
    INTERNAL = "internal"


class ApiError(ContractModel):
    """Structured failure that an agent can inspect without parsing prose."""

    schema_version: SchemaVersion = SchemaVersion.V1
    code: MachineCode
    message: Annotated[
        str,
        Field(min_length=1, description="Human-readable, potentially localized explanation."),
    ]
    retryable: bool = False
    stage: ErrorStage
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Small structured context; large payloads belong in artifacts.",
    )
    next_action: NextAction | None = None
