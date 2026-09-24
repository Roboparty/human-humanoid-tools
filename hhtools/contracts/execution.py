"""Shared retarget execution options and runtime provenance."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .common import AssetId, ContractModel, Sha256Hex

ExecutionIdentifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveExecutionFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
PositiveExecutionInt = Annotated[int, Field(gt=0, strict=True)]


class SolverExecutionOptions(ContractModel):
    """Parameters shared by browser and Agent solver entry points."""

    backend: Literal["newton", "interaction_mesh"] = "newton"
    ik_iterations: PositiveExecutionInt = 24
    retarget_fps: PositiveExecutionFloat | None = None


class HumanRetargetExecutionOptions(SolverExecutionOptions):
    """Stable H2R options independent of a browser motion token."""

    reference: ExecutionIdentifier = "smpl"
    human_height: Annotated[float, Field(gt=0.1, allow_inf_nan=False)] | None = None
    limit_frames: PositiveExecutionInt | None = None
    foot_clamp_anti_penetration: Annotated[bool, Field(strict=True)] = False


class AgentH2RExecutionParameters(HumanRetargetExecutionOptions):
    """Fully resolved H2R parameters persisted in an immutable Agent JobSpec."""

    run_mode: Literal["smoke", "full"]
    human_height: Annotated[float, Field(gt=0.1, allow_inf_nan=False)]
    foot_clamp_anti_penetration: Annotated[bool, Field(strict=True)]
    retarget_profile: ExecutionIdentifier
    output_format: Literal["csv", "pkl"]

    @model_validator(mode="after")
    def validate_run_scope(self) -> AgentH2RExecutionParameters:
        if self.run_mode == "full" and self.limit_frames is not None:
            raise ValueError("full execution cannot declare limit_frames")
        if self.backend == "interaction_mesh" and "ik_iterations" in self.model_fields_set:
            raise ValueError("interaction_mesh execution cannot declare ik_iterations")
        return self


class AgentR2RExecutionParameters(SolverExecutionOptions):
    """Fully resolved parameters for one plain robot-to-robot trajectory."""

    run_mode: Literal["smoke", "full"]
    limit_frames: PositiveExecutionInt | None = None
    source_fps: PositiveExecutionFloat | None = None
    output_format: Literal["csv", "pkl"]
    trajectory_profile: Literal["mimic"] = "mimic"

    @model_validator(mode="after")
    def validate_run_scope(self) -> AgentR2RExecutionParameters:
        if self.backend != "newton":
            raise ValueError("the initial R2R Agent workflow supports only newton")
        if self.run_mode == "full" and self.limit_frames is not None:
            raise ValueError("full execution cannot declare limit_frames")
        return self


class ExecutionProvenance(ContractModel):
    """Observed execution environment recorded after a solver actually runs."""

    executor: Annotated[str, Field(min_length=1, max_length=128)] = "unknown"
    backend: Annotated[str, Field(min_length=1, max_length=128)] = "unknown"
    device: Annotated[str, Field(min_length=1, max_length=256)] = "unknown"
    device_kind: Literal["cpu", "cuda", "mps", "unknown"] = "unknown"
    precision: Literal["float32", "float64", "mixed", "unknown"] = "unknown"
    runtime: Annotated[str | None, Field(default=None, max_length=128)]
    runtime_version: Annotated[str | None, Field(default=None, max_length=128)]
    solver: Annotated[str | None, Field(default=None, max_length=128)]
    cuda_runtime: Annotated[str | None, Field(default=None, max_length=128)]
    cuda_graph_requested: bool | None = None
    cuda_graph_used: bool | None = None
    fallback_used: bool = False
    fallback_reason: Annotated[str | None, Field(default=None, max_length=512)]
    dataset: Annotated[str | None, Field(default=None, max_length=128)]
    motion_asset_id: AssetId | None = None
    motion_sha256: Sha256Hex | None = None
    robot_asset_id: AssetId | None = None
    robot_config_sha256: Sha256Hex | None = None
    source_robot_asset_id: AssetId | None = None
    source_robot_config_sha256: Sha256Hex | None = None
    reference: Annotated[str | None, Field(default=None, max_length=128)]

    @model_validator(mode="after")
    def validate_fallback(self) -> ExecutionProvenance:
        if self.fallback_reason is not None and not self.fallback_used:
            raise ValueError("fallback_reason requires fallback_used")
        if self.cuda_graph_used and not self.cuda_graph_requested:
            raise ValueError("cuda_graph_used requires cuda_graph_requested")
        return self


__all__ = [
    "AgentH2RExecutionParameters",
    "AgentR2RExecutionParameters",
    "ExecutionIdentifier",
    "ExecutionProvenance",
    "HumanRetargetExecutionOptions",
    "PositiveExecutionFloat",
    "PositiveExecutionInt",
    "SolverExecutionOptions",
]
