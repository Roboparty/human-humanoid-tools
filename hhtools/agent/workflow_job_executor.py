"""Dispatch immutable Agent jobs to their workflow-specific executor."""

from __future__ import annotations

from typing import Protocol

from hhtools.contracts import ApiError, ErrorStage, JobSpecKind, JobSpecV2
from hhtools.services.jobs import (
    JobExecutionContext,
    JobExecutionError,
    JobExecutionResult,
)

from .h2r_job_executor import H2RJobExecutor
from .r2r_job_executor import R2RJobExecutor


class BatchExecutor(Protocol):
    def __call__(
        self,
        spec: JobSpecV2,
        context: JobExecutionContext,
    ) -> JobExecutionResult: ...


class WorkflowJobExecutor:
    """One JobManager executor that preserves separate H2R and R2R adapters."""

    def __init__(
        self,
        h2r: H2RJobExecutor,
        r2r: R2RJobExecutor,
        batch: BatchExecutor | None = None,
    ) -> None:
        self.h2r = h2r
        self.r2r = r2r
        self.batch = batch

    def __call__(
        self,
        spec: JobSpecV2,
        context: JobExecutionContext,
    ) -> JobExecutionResult:
        if spec.kind is JobSpecKind.BATCH_RETARGET:
            if self.batch is None:
                raise JobExecutionError(
                    ApiError(
                        code="BACKEND_UNAVAILABLE",
                        message="No Agent batch executor is configured.",
                        stage=ErrorStage.ADMISSION,
                    )
                )
            return self.batch(spec, context)
        if spec.kind is JobSpecKind.R2R_RETARGET:
            return self.r2r(spec, context)
        return self.h2r(spec, context)


__all__ = ["WorkflowJobExecutor"]
