"""Sequential scalable batch orchestration over existing single-item executors."""

from __future__ import annotations

import json
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from hhtools.contracts import (
    ApiError,
    BatchReport,
    BatchReportItem,
    BatchResourceLimits,
    BatchWorkflow,
    ErrorStage,
    FailureItem,
    JobOutcome,
    JobProgress,
    JobSpecBatchItem,
    JobSpecKind,
    JobSpecV2,
    OutputPolicy,
)
from hhtools.services.jobs import (
    JobCancelledError,
    JobExecutionContext,
    JobExecutionError,
    JobExecutionResult,
)


class ChildExecutor(Protocol):
    def __call__(
        self,
        spec: JobSpecV2,
        context: JobExecutionContext,
    ) -> JobExecutionResult: ...


def _execution_error(code: str, message: str) -> JobExecutionError:
    return JobExecutionError(ApiError(code=code, message=message, stage=ErrorStage.EXECUTION))


def _child_spec(item: JobSpecBatchItem) -> JobSpecV2:
    return JobSpecV2(
        kind=item.kind,
        plan_id=item.plan_id,
        inputs=[item.input],
        robot=item.robot,
        source_robot=item.source_robot,
        calibration=item.calibration,
        batch_items=None,
        backend=item.backend,
        effective_parameters=item.effective_parameters,
        output_policy=item.output_policy,
        provenance=item.provenance,
        created_at=item.created_at,
    )


def _aggregate_outcome(items: list[BatchReportItem]) -> JobOutcome:
    failures = sum(item.state == "failed" for item in items)
    if failures == len(items):
        return JobOutcome.REJECTED
    if failures:
        return JobOutcome.PARTIAL
    outcomes = {item.outcome for item in items}
    if JobOutcome.REJECTED in outcomes:
        return JobOutcome.REJECTED
    if JobOutcome.PARTIAL in outcomes:
        return JobOutcome.PARTIAL
    if JobOutcome.REVIEW_REQUIRED in outcomes:
        return JobOutcome.REVIEW_REQUIRED
    return JobOutcome.SUCCESS


def _report_bytes(report: BatchReport) -> bytes:
    document = report.model_dump(mode="json", exclude_none=True)
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


class BatchJobExecutor:
    """Run child plans in order, continuing only across structured item failures."""

    def __init__(
        self,
        child_executor: ChildExecutor,
        *,
        validate_spec: Callable[[JobSpecV2], None],
        temporary_root: Path,
    ) -> None:
        self._child_executor = child_executor
        self._validate_spec = validate_spec
        self._temporary_root = Path(temporary_root)

    def _parameters(
        self,
        spec: JobSpecV2,
    ) -> tuple[BatchWorkflow, str, BatchResourceLimits]:
        if (
            spec.kind is not JobSpecKind.BATCH_RETARGET
            or not spec.batch_items
            or spec.backend != "batch"
            or spec.output_policy is not OutputPolicy.CREATE_NEW
        ):
            raise _execution_error(
                "INVALID_PARAMETER",
                "The batch executor requires one validated batch JobSpec.",
            )
        try:
            workflow = BatchWorkflow(spec.effective_parameters["workflow"])
            run_mode = str(spec.effective_parameters["run_mode"])
            limits = BatchResourceLimits.model_validate(
                spec.effective_parameters["resource_limits"]
            )
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            raise _execution_error(
                "INVALID_PARAMETER",
                "The batch JobSpec contains invalid resource parameters.",
            ) from error
        if run_mode not in {"smoke", "full"} or limits.item_count != len(spec.batch_items):
            raise _execution_error(
                "INVALID_PARAMETER",
                "The batch JobSpec run mode or item count is inconsistent.",
            )
        return workflow, run_mode, limits

    def _publish_archive(
        self,
        context: JobExecutionContext,
        report: BatchReport,
    ) -> None:
        descriptors = {
            descriptor.artifact_id: descriptor
            for descriptor in context.published_artifacts()
            if descriptor.kind == "retargeted_motion"
        }
        self._temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="agent-batch-",
            dir=self._temporary_root,
        ) as temporary:
            archive_path = Path(temporary) / "batch-results.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("batch-report.json", _report_bytes(report))
                for item in report.items:
                    for artifact_id in item.artifact_ids:
                        descriptor = descriptors.get(artifact_id)
                        if descriptor is None:
                            continue
                        stored = context.get_published_artifact(artifact_id)
                        extension = descriptor.format or "bin"
                        name = f"{item.index + 1:04d}_{item.input_asset_id[-12:]}.{extension}"
                        archive.write(stored.path, name)
            context.publish_file(
                kind="batch_archive",
                source=archive_path,
                format="zip",
                media_type="application/zip",
                metadata={
                    "workflow": report.workflow.value,
                    "total_items": report.total_items,
                    "succeeded_items": report.succeeded_items,
                    "failed_items": report.failed_items,
                },
            )

    def __call__(
        self,
        spec: JobSpecV2,
        context: JobExecutionContext,
    ) -> JobExecutionResult:
        workflow, run_mode, limits = self._parameters(spec)
        assert spec.batch_items is not None
        self._validate_spec(spec)
        total = len(spec.batch_items)
        reports: list[BatchReportItem] = []
        failures: list[FailureItem] = []
        context.report_progress(
            phase="batch",
            fraction=0.0,
            completed_items=0,
            total_items=total,
            message=f"Starting {workflow.value.upper()} batch 0/{total}.",
            poll_after_ms=1_000,
        )

        for index, item in enumerate(spec.batch_items):
            context.raise_if_cancelled()
            child = _child_spec(item)
            before = {artifact.artifact_id for artifact in context.published_artifacts()}

            def child_progress(
                progress: JobProgress,
                poll_after_ms: int | None,
                _index: int = index,
            ):
                return context.report_progress(
                    phase=f"batch:{progress.phase}",
                    fraction=min(0.99, (_index + progress.fraction) / total),
                    completed_items=_index,
                    total_items=total,
                    message=f"Item {_index + 1}/{total}: {progress.message or progress.phase}",
                    eta_seconds=progress.eta_seconds,
                    poll_after_ms=poll_after_ms,
                )

            child_context = context.for_child(
                child,
                progress_callback=child_progress,
                artifact_metadata={
                    "batch_item_id": item.item_id,
                    "batch_item_index": index,
                    "batch_plan_id": spec.plan_id,
                },
            )
            try:
                result = self._child_executor(child, child_context)
            except JobCancelledError:
                raise
            except JobExecutionError as error:
                public = error.error
                failures.append(
                    FailureItem(
                        item_id=item.item_id,
                        code=public.code,
                        message=public.message,
                        stage=public.stage,
                        retryable=public.retryable,
                        details={**public.details, "item_index": index},
                    )
                )
                artifact_ids = [
                    artifact.artifact_id
                    for artifact in context.published_artifacts()
                    if artifact.artifact_id not in before
                ]
                reports.append(
                    BatchReportItem(
                        index=index,
                        item_id=item.item_id,
                        plan_id=item.plan_id,
                        input_asset_id=item.input.asset_id,
                        state="failed",
                        error=public,
                        artifact_ids=artifact_ids,
                    )
                )
            else:
                for failure in result.failures:
                    normalized = FailureItem.model_validate(failure)
                    failures.append(normalized.model_copy(update={"item_id": item.item_id}))
                artifact_ids = [
                    artifact.artifact_id
                    for artifact in context.published_artifacts()
                    if artifact.artifact_id not in before
                ]
                reports.append(
                    BatchReportItem(
                        index=index,
                        item_id=item.item_id,
                        plan_id=item.plan_id,
                        input_asset_id=item.input.asset_id,
                        state="completed",
                        outcome=result.outcome,
                        summary=dict(result.summary),
                        artifact_ids=artifact_ids,
                        execution_provenance=dict(result.execution_provenance),
                    )
                )
            context.raise_if_cancelled()
            context.report_progress(
                phase="batch",
                fraction=min(0.99, (index + 1) / total),
                completed_items=index + 1,
                total_items=total,
                message=f"Completed batch item {index + 1}/{total}.",
                poll_after_ms=1_000,
            )

        outcome = _aggregate_outcome(reports)
        succeeded = sum(item.state == "completed" for item in reports)
        failed = total - succeeded
        report = BatchReport(
            job_id=context.job_id,
            workflow=workflow,
            outcome=outcome,
            total_items=total,
            completed_items=total,
            succeeded_items=succeeded,
            failed_items=failed,
            items=reports,
        )
        context.publish_json(
            kind="batch_report",
            document=report.model_dump(mode="json", exclude_none=True),
            metadata={
                "workflow": workflow.value,
                "total_items": total,
                "succeeded_items": succeeded,
                "failed_items": failed,
            },
        )
        self._publish_archive(context, report)
        return JobExecutionResult(
            outcome=outcome,
            summary={
                "workflow": f"batch_{workflow.value}",
                "run_mode": run_mode,
                "total_items": total,
                "completed_items": total,
                "succeeded_items": succeeded,
                "failed_items": failed,
                "estimated_total_frames": limits.estimated_total_frames,
            },
            evaluation_summary=(
                f"Batch completed {succeeded}/{total} items; every successful motion "
                "still requires review."
            ),
            evaluation_metrics={
                "total_items": total,
                "succeeded_items": succeeded,
                "failed_items": failed,
            },
            evaluation_checks=[
                {
                    "code": "BATCH_ITEM_RESULTS",
                    "status": outcome.value,
                    "succeeded_items": succeeded,
                    "failed_items": failed,
                }
            ],
            failures=failures,
            execution_provenance={
                "executor": "agent_bounded_batch_v1",
                "backend": "batch",
                "device": "unknown",
                "device_kind": "unknown",
                "precision": "unknown",
                "fallback_used": False,
            },
        )


__all__ = ["BatchJobExecutor"]
