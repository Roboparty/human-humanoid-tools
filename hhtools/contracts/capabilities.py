"""Service, backend, and compute-device capability contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import Field, model_validator

from .assets import AssetCategory
from .common import ContractModel, SchemaVersion


class DeviceKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class SchedulerMode(StrEnum):
    """How the running and queued admission limits are configured."""

    UNLIMITED = "unlimited"
    LIMITED = "limited"
    MIXED = "mixed"


class SchedulerCapability(ContractModel):
    """Current admission policy and occupancy.

    ``max_running_jobs == 0`` disables admission control entirely in the Web
    scheduler.  In that state ``max_queued_jobs`` is retained as configured
    metadata but is not enforced, so the effective mode remains ``unlimited``.
    """

    max_running_jobs: Annotated[int, Field(ge=0)] = 0
    max_queued_jobs: Annotated[int, Field(ge=0)] = 0
    running: Annotated[int, Field(ge=0)] = 0
    queued: Annotated[int, Field(ge=0)] = 0
    reserved: Annotated[int, Field(ge=0)] = 0
    mode: SchedulerMode
    closed: bool = False

    @model_validator(mode="after")
    def validate_mode(self) -> SchedulerCapability:
        expected = SchedulerMode.MIXED
        if self.max_running_jobs == 0:
            expected = SchedulerMode.UNLIMITED
        elif self.max_running_jobs > 0 and self.max_queued_jobs > 0:
            expected = SchedulerMode.LIMITED
        if self.mode is not expected:
            raise ValueError(f"mode must be {expected.value} for the configured limits")
        return self


class DeviceCapability(ContractModel):
    """One execution device visible to the HHTools service."""

    device_id: Annotated[str, Field(min_length=1, max_length=128)]
    kind: DeviceKind
    display_name: Annotated[str, Field(min_length=1, max_length=256)]
    available: bool
    total_memory_bytes: Annotated[int | None, Field(default=None, ge=0)]
    free_memory_bytes: Annotated[int | None, Field(default=None, ge=0)]
    compute_capability: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BackendCapability(ContractModel):
    """A retargeting backend and the inputs/outputs it can handle."""

    backend_id: Annotated[
        str,
        Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_-]*$"),
    ]
    display_name: Annotated[str, Field(min_length=1, max_length=256)]
    available: bool
    version: str | None = None
    supported_categories: list[AssetCategory] = Field(default_factory=list)
    output_formats: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None
    features: dict[str, bool] = Field(default_factory=dict)
    limits: dict[str, Any] = Field(default_factory=dict)


class RobotCapability(ContractModel):
    """Agent-facing robot availability and calibration summary."""

    robot_id: Annotated[str, Field(min_length=1, max_length=256)]
    display_name: Annotated[str, Field(min_length=1, max_length=256)]
    available: bool
    has_urdf: bool
    has_ik_mapping: bool
    dof_count: Annotated[int | None, Field(default=None, ge=0)]
    supported_references: list[str] = Field(default_factory=list)
    calibrated_references: list[str] = Field(default_factory=list)
    scaler_references: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None


class RobotListResponse(ContractModel):
    """Stable envelope for MCP robot discovery without repeating all capabilities."""

    schema_version: SchemaVersion = SchemaVersion.V1
    robots: list[RobotCapability] = Field(default_factory=list)


class CapabilityResponse(ContractModel):
    """Compact discovery response used before an agent builds a plan."""

    schema_version: SchemaVersion = SchemaVersion.V1
    service_name: str = "hhtools"
    service_version: Annotated[str, Field(min_length=1)]
    agent_api_version: str = "v1"
    backends: list[BackendCapability] = Field(default_factory=list)
    devices: list[DeviceCapability] = Field(default_factory=list)
    robots: list[RobotCapability] = Field(default_factory=list)
    scheduler: SchedulerCapability
    asset_root_ids: list[
        Annotated[
            str,
            Field(
                min_length=1,
                max_length=128,
                pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
            ),
        ]
    ] = Field(default_factory=list)
    supported_input_formats: list[str] = Field(default_factory=list)
    supported_output_formats: list[str] = Field(default_factory=list)
    features: dict[str, bool] = Field(default_factory=dict)
