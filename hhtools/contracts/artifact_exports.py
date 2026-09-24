"""Portable receipt for an explicitly exported managed artifact."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .common import ArtifactId, ContractModel, SchemaVersion, Sha256Hex

_FORMAT_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,31}$"
_KIND_PATTERN = r"^[a-z][a-z0-9_-]{0,127}$"
_MEDIA_TYPE_PATTERN = (
    r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+/[!#$%&'*+.^_`|~0-9A-Za-z-]+"
    r"(?:[ \t]*;[^\r\n\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+)*$"
)
_RELATIVE_PATH_PATTERN = r"^jobs/[0-9a-f]{64}/[0-9a-f]{64}\.[a-z0-9][a-z0-9._+-]{0,31}$"


class ArtifactExportReceipt(ContractModel):
    """Host-independent identity for one file copied to the configured export root."""

    schema_version: SchemaVersion = SchemaVersion.V1
    root_id: Literal["agent-exports"] = "agent-exports"
    relative_path: Annotated[
        str,
        Field(
            min_length=1,
            max_length=1024,
            pattern=_RELATIVE_PATH_PATTERN,
            description="Portable path below the server-configured agent export root.",
        ),
    ]
    job_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=256,
            pattern=r"^job:[A-Za-z0-9][A-Za-z0-9._~-]{0,251}$",
        ),
    ]
    artifact_id: ArtifactId
    kind: Annotated[
        str,
        Field(min_length=1, max_length=128, pattern=_KIND_PATTERN),
    ]
    format: Annotated[
        str | None,
        Field(default=None, min_length=1, max_length=32, pattern=_FORMAT_PATTERN),
    ]
    media_type: Annotated[
        str | None,
        Field(default=None, max_length=255, pattern=_MEDIA_TYPE_PATTERN),
    ]
    size_bytes: Annotated[int, Field(ge=0)]
    sha256: Sha256Hex

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("export paths must use forward slashes")
        posix = PurePosixPath(value)
        windows = PureWindowsPath(value)
        if posix.is_absolute() or windows.is_absolute() or windows.drive:
            raise ValueError("export path must be relative")
        if any(part in {"", ".", ".."} for part in posix.parts):
            raise ValueError("export path must be normalized and cannot traverse parents")
        return posix.as_posix()

    @model_validator(mode="after")
    def validate_identity_path(self) -> ArtifactExportReceipt:
        job_token = hashlib.sha256(self.job_id.encode("utf-8")).hexdigest()
        artifact_token = hashlib.sha256(self.artifact_id.encode("utf-8")).hexdigest()
        extension = self.format.casefold() if self.format is not None else "bin"
        expected = f"jobs/{job_token}/{artifact_token}.{extension}"
        if self.relative_path != expected:
            raise ValueError("export path must match the job and artifact identity")
        return self


__all__ = ["ArtifactExportReceipt"]
