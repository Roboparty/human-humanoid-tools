"""Versioned JSON reports stored as immutable job artifacts."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import AwareDatetime, Field, model_validator

from .common import ApiError, ContractModel, ErrorStage, MachineCode, PlanId, SchemaVersion
from .execution import ExecutionProvenance
from .job_spec import JobSpecV2
from .jobs import ArtifactDescriptor, JobOutcome, JobState


class EvaluationReport(ContractModel):
    """Compact quality verdict; large plots and previews remain separate artifacts."""

    schema_version: SchemaVersion = SchemaVersion.V1
    job_id: Annotated[str, Field(min_length=1, max_length=256)]
    outcome: JobOutcome
    summary: Annotated[str | None, Field(default=None, max_length=4_096)]
    metrics: dict[str, Any] = Field(default_factory=dict)
    checks: list[dict[str, Any]] = Field(default_factory=list, max_length=256)
    created_at: AwareDatetime


class FailureItem(ContractModel):
    """One structured failed input or execution stage."""

    item_id: Annotated[str | None, Field(default=None, min_length=1, max_length=256)]
    code: MachineCode
    message: Annotated[str, Field(min_length=1, max_length=8_192)]
    stage: ErrorStage
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class FailureReport(ContractModel):
    """Structured failures for a failed or partially completed job."""

    schema_version: SchemaVersion = SchemaVersion.V1
    job_id: Annotated[str, Field(min_length=1, max_length=256)]
    failures: Annotated[list[FailureItem], Field(min_length=1)]
    created_at: AwareDatetime


class JobManifest(ContractModel):
    """Terminal audit record.

    ``artifacts`` lists every artifact published before the manifest itself;
    self-inclusion would make a content hash recursively impossible.
    """

    schema_version: SchemaVersion = SchemaVersion.V1
    job_id: Annotated[str, Field(min_length=1, max_length=256)]
    parent_job_id: Annotated[str | None, Field(default=None, min_length=1, max_length=256)]
    root_job_id: Annotated[str | None, Field(default=None, min_length=1, max_length=256)]
    attempt: Annotated[int, Field(ge=1)] = 1
    plan_id: PlanId
    state: JobState
    outcome: JobOutcome | None = None
    error: ApiError | None = None
    cancellation_requested: bool = False
    job_spec: JobSpecV2
    execution_provenance: ExecutionProvenance = Field(default_factory=ExecutionProvenance)
    summary: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactDescriptor] = Field(default_factory=list)
    submitted_at: AwareDatetime
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime

    @model_validator(mode="after")
    def validate_terminal_result(self) -> JobManifest:
        if self.state not in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}:
            raise ValueError("a manifest can only describe a terminal job")
        if self.state is JobState.COMPLETED:
            if self.outcome is None or self.error is not None:
                raise ValueError("completed manifests require an outcome and no error")
        elif self.outcome is not None:
            raise ValueError("only completed manifests may include an outcome")
        if self.state is JobState.FAILED:
            if self.error is None:
                raise ValueError("failed manifests require an error")
        elif self.error is not None:
            raise ValueError("only failed manifests may include an error")
        if self.started_at is not None and self.started_at < self.submitted_at:
            raise ValueError("started_at cannot precede submitted_at")
        if self.completed_at < (self.started_at or self.submitted_at):
            raise ValueError("completed_at cannot precede job execution")
        if self.parent_job_id is None:
            if self.root_job_id is not None or self.attempt != 1:
                raise ValueError("root manifests cannot declare retry lineage")
        elif self.root_job_id is None or self.attempt < 2:
            raise ValueError("retry manifests require complete lineage")
        return self


__all__ = [
    "EvaluationReport",
    "FailureItem",
    "FailureReport",
    "JobManifest",
]
