"""Compact job and artifact contracts for polling agents."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any

from pydantic import AliasChoices, AwareDatetime, Field, field_validator, model_validator

from .capabilities import SchedulerMode
from .common import (
    ApiError,
    ArtifactId,
    ContractModel,
    NextAction,
    PlanId,
    ResourceUri,
    SchemaVersion,
    Sha256Hex,
)

IdempotencyKey = Annotated[
    str,
    Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._~:-]{0,255}$",
        description="Caller-generated key binding one logical job submission.",
    ),
]
_MEDIA_TYPE = re.compile(
    r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+/[!#$%&'*+.^_`|~0-9A-Za-z-]+"
    r"(?:[ \t]*;[^\r\n\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+)*$"
)


class JobStartRequest(ContractModel):
    """Submit one already-preflighted immutable retarget plan."""

    schema_version: SchemaVersion = SchemaVersion.V1
    plan_id: PlanId
    idempotency_key: IdempotencyKey


class JobRetryRequest(ContractModel):
    """Create a child attempt without mutating the terminal parent job."""

    schema_version: SchemaVersion = SchemaVersion.V1
    idempotency_key: IdempotencyKey


class JobLookupRequest(ContractModel):
    """Recover one caller-owned submission without enumerating other jobs."""

    schema_version: SchemaVersion = SchemaVersion.V1
    plan_id: PlanId
    idempotency_key: IdempotencyKey
    after_revision: Annotated[int | None, Field(default=None, ge=0)]


class JobState(StrEnum):
    """Execution lifecycle, independent from output quality."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobOutcome(StrEnum):
    """Semantic result of a completed job."""

    SUCCESS = "success"
    PARTIAL = "partial"
    REVIEW_REQUIRED = "review_required"
    REJECTED = "rejected"


class JobProgress(ContractModel):
    """Small monotonic progress snapshot suitable for frequent polling."""

    phase: Annotated[str, Field(min_length=1, max_length=128)] = "queued"
    fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    revision: Annotated[int, Field(ge=0)] = 0
    completed_items: Annotated[int | None, Field(default=None, ge=0)]
    total_items: Annotated[int | None, Field(default=None, ge=0)]
    message: Annotated[str | None, Field(default=None, max_length=2_048)]
    updated_at: AwareDatetime | None = None
    eta_seconds: Annotated[float | None, Field(default=None, ge=0)]

    @model_validator(mode="after")
    def validate_item_counts(self) -> JobProgress:
        if self.completed_items is not None and self.total_items is None:
            raise ValueError("total_items is required when completed_items is provided")
        if (
            self.completed_items is not None
            and self.total_items is not None
            and self.completed_items > self.total_items
        ):
            raise ValueError("completed_items cannot exceed total_items")
        return self


class ArtifactDescriptor(ContractModel):
    """Metadata and URI for a job output; binary data is never embedded."""

    schema_version: SchemaVersion = SchemaVersion.V1
    artifact_id: ArtifactId
    job_id: Annotated[str, Field(min_length=1, max_length=256)]
    kind: Annotated[
        str,
        Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_-]{0,127}$"),
    ]
    format: Annotated[
        str | None,
        Field(
            default=None,
            min_length=1,
            max_length=32,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,31}$",
        ),
    ]
    resource_uri: Annotated[
        ResourceUri,
        Field(
            min_length=1,
            validation_alias=AliasChoices("resource_uri", "uri"),
            description=(
                "Resolvable canonical job-scoped HHTools artifact URI or portable HTTP(S) URI."
            ),
        ),
    ]
    media_type: Annotated[str | None, Field(default=None, max_length=255)]
    size_bytes: Annotated[int | None, Field(default=None, ge=0)]
    sha256: Sha256Hex | None = None
    created_at: AwareDatetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("media_type")
    @classmethod
    def validate_media_type(cls, value: str | None) -> str | None:
        if value is not None and _MEDIA_TYPE.fullmatch(value) is None:
            raise ValueError("media_type must be a safe MIME type without control characters")
        return value

    @property
    def uri(self) -> str:
        """Read-only compatibility accessor; JSON always uses ``resource_uri``."""

        return self.resource_uri


class ArtifactListResponse(ContractModel):
    """Bounded page of canonical artifacts attached to one job."""

    schema_version: SchemaVersion = SchemaVersion.V1
    job_id: Annotated[str, Field(min_length=1, max_length=256)]
    artifacts: list[ArtifactDescriptor] = Field(default_factory=list, max_length=500)
    total: Annotated[int, Field(ge=0)]
    limit: Annotated[int, Field(ge=1, le=500)] = 100
    offset: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def validate_page(self) -> ArtifactListResponse:
        if any(item.job_id != self.job_id for item in self.artifacts):
            raise ValueError("every artifact must belong to the requested job")
        if len(self.artifacts) > self.limit:
            raise ValueError("the returned artifact page cannot exceed limit")
        if self.artifacts and self.offset + len(self.artifacts) > self.total:
            raise ValueError("the returned artifact page cannot extend beyond total")
        return self


class JobQueueView(ContractModel):
    """Queue position and admission settings captured with a job snapshot."""

    position: Annotated[int | None, Field(default=None, ge=1)]
    max_running_jobs: Annotated[int, Field(ge=0)] = 0
    max_queued_jobs: Annotated[int, Field(ge=0)] = 0
    mode: SchedulerMode


class AgentJobView(ContractModel):
    """Compact default job view; large arrays live behind artifact URIs."""

    schema_version: SchemaVersion = SchemaVersion.V1
    job_id: Annotated[str, Field(min_length=1, max_length=256)]
    parent_job_id: Annotated[str | None, Field(default=None, min_length=1, max_length=256)]
    root_job_id: Annotated[str | None, Field(default=None, min_length=1, max_length=256)]
    attempt: Annotated[int, Field(ge=1)] = 1
    state: JobState
    outcome: JobOutcome | None = None
    progress: JobProgress
    summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Small input/backend/robot summary, never trajectory arrays.",
    )
    queue: JobQueueView | None = None
    artifacts: list[ArtifactDescriptor] = Field(default_factory=list, max_length=32)
    artifact_count: Annotated[int | None, Field(default=None, ge=0)]
    error: ApiError | None = None
    next_action: NextAction | None = None
    cancellation_requested: bool = False
    cancellable: bool = False
    submitted_at: AwareDatetime
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None
    poll_after_ms: Annotated[int | None, Field(default=None, ge=0, le=300_000)]

    @model_validator(mode="after")
    def validate_lifecycle(self) -> AgentJobView:
        terminal = {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}
        if self.outcome is not None and self.state is not JobState.COMPLETED:
            raise ValueError("outcome is only valid for completed jobs")
        if self.state is JobState.COMPLETED and self.outcome is None:
            raise ValueError("completed jobs must include an outcome")
        if self.state is JobState.FAILED and self.error is None:
            raise ValueError("failed jobs must include an error")
        if self.state is not JobState.FAILED and self.error is not None:
            raise ValueError("only failed jobs may include an error")
        if self.state is JobState.QUEUED and self.started_at is not None:
            raise ValueError("queued jobs cannot include started_at")
        if self.state is JobState.RUNNING and self.started_at is None:
            raise ValueError("running jobs must include started_at")
        if self.state in terminal and self.completed_at is None:
            raise ValueError("terminal jobs must include completed_at")
        if self.state not in terminal and self.completed_at is not None:
            raise ValueError("non-terminal jobs cannot include completed_at")
        if self.state in terminal and self.cancellable:
            raise ValueError("terminal jobs cannot be cancellable")
        if self.artifact_count is not None and self.artifact_count < len(self.artifacts):
            raise ValueError("artifact_count cannot be smaller than returned artifacts")
        if self.parent_job_id is None:
            if self.root_job_id is not None or self.attempt != 1:
                raise ValueError("root jobs cannot declare retry lineage")
        elif self.parent_job_id == self.job_id or self.root_job_id is None or self.attempt < 2:
            raise ValueError("retry jobs require valid parent/root lineage and attempt")
        if self.started_at is not None and self.started_at < self.submitted_at:
            raise ValueError("started_at cannot precede submitted_at")
        if self.completed_at is not None:
            baseline = self.started_at or self.submitted_at
            if self.completed_at < baseline:
                raise ValueError("completed_at cannot precede the job start")
        return self
