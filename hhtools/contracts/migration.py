"""Public contracts for safe, non-executing legacy JobSpec upgrades."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from .common import (
    AssetId,
    ContractModel,
    MachineCode,
    PlanId,
    SchemaVersion,
    Sha256Hex,
)
from .job_spec import JobSpecV2
from .preflight import PreflightResponse, PreflightStatus


class LegacyJobUpgradeRequest(ContractModel):
    """One bounded JSON object containing a JobSpec v1 or download wrapper.

    The migration service remains responsible for the stricter v1 shape,
    depth, node-count, byte-size, allowlisted-root, and content checks.  This
    wrapper only gives REST and JSON CLI a stable, versioned transport shape.
    """

    schema_version: SchemaVersion = SchemaVersion.V1
    payload: dict[str, Any] = Field(
        description="Raw JobSpec v1 document or an existing single-job download wrapper."
    )


class LegacyMigrationReceipt(ContractModel):
    """Portable proof of how one canonical v1 document became JobSpec v2."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    semantics: Literal["hhtools.legacy-job-upgrade.v1"] = "hhtools.legacy-job-upgrade.v1"
    source_schema_version: Literal[1] = 1
    canonical_v1_sha256: Sha256Hex
    motion_asset_id: AssetId
    robot_asset_id: AssetId
    plan_id: PlanId
    job_spec_sha256: Sha256Hex
    output_format: Literal["csv"] = "csv"
    output_policy: Literal["create_new"] = "create_new"
    warnings: tuple[MachineCode, ...] = ()

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("migration warning codes must be unique and sorted")
        return value


class LegacyJobUpgradeResponse(ContractModel):
    """Non-executing upgrade result tied to its authoritative preflight."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SchemaVersion = SchemaVersion.V1
    preflight: PreflightResponse
    job_spec: JobSpecV2 | None = None
    receipt: LegacyMigrationReceipt | None = None

    @model_validator(mode="after")
    def validate_preflight_result(self) -> LegacyJobUpgradeResponse:
        ready = self.preflight.status is PreflightStatus.READY
        complete = self.job_spec is not None and self.receipt is not None
        empty = self.job_spec is None and self.receipt is None
        if (ready and not complete) or (not ready and not empty):
            raise ValueError("upgrade result must match its preflight status")
        return self


__all__ = [
    "LegacyJobUpgradeRequest",
    "LegacyJobUpgradeResponse",
    "LegacyMigrationReceipt",
]
