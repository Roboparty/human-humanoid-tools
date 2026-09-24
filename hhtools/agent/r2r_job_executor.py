"""Thin Agent executor adapter over the existing robot-to-robot runtime."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from hhtools.contracts import (
    AgentR2RExecutionParameters,
    ApiError,
    ErrorStage,
    JobSpecKind,
    JobSpecV2,
    OutputPolicy,
)
from hhtools.services.artifacts import ArtifactStoreError
from hhtools.services.assets import AssetServiceError
from hhtools.services.execution import build_execution_provenance
from hhtools.services.jobs import (
    JobCancelledError,
    JobExecutionContext,
    JobExecutionError,
    JobExecutionResult,
)
from hhtools.services.retarget import RetargetServiceError

from .h2r_job_executor import _is_cuda_oom, _quality_verdict

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResolvedR2RTrajectory:
    asset_id: str
    source_robot_id: str | None
    source_path: Path
    stem: str
    profile: str
    has_scene: bool


@dataclass(frozen=True, slots=True)
class PreparedR2RSource:
    motion: Any
    playback: Any
    source_fps: float
    num_frames: int


@dataclass(frozen=True, slots=True)
class R2RPreview:
    document: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    yellow_foot_z: float | None = None


class ValidateSpec(Protocol):
    def __call__(self, spec: JobSpecV2) -> None: ...


class ResolveTrajectory(Protocol):
    def __call__(self, asset_id: str) -> ResolvedR2RTrajectory: ...


class GetRobotModel(Protocol):
    def __call__(self, spec: JobSpecV2) -> Any: ...


class PrepareSource(Protocol):
    def __call__(
        self,
        resolved: ResolvedR2RTrajectory,
        source_model: Any,
        *,
        source_fps: float | None,
        limit_frames: int | None,
        progress_callback: Callable[[int, int], None],
    ) -> PreparedR2RSource: ...


class LoadPairCalibration(Protocol):
    def __call__(self, source_model: Any, target_model: Any) -> dict[str, float]: ...


class PrepareMotion(Protocol):
    def __call__(self, motion: Any, retarget_fps: float | None) -> tuple[Any, float]: ...


class RunR2R(Protocol):
    def __call__(
        self,
        source_model: Any,
        target_model: Any,
        motion: Any,
        calibration: dict[str, float],
        *,
        backend: str,
        ik_iterations: int,
        progress_callback: Callable[[int, int], None],
    ) -> Any: ...


class BuildPreview(Protocol):
    def __call__(
        self,
        source_model: Any,
        target_model: Any,
        source: PreparedR2RSource,
        motion: Any,
        calibration: dict[str, float],
        retargeted: Any,
    ) -> R2RPreview: ...


class WriteExport(Protocol):
    def __call__(
        self,
        retargeted: Any,
        source_model: Any,
        target_model: Any,
        source_motion: Any,
        calibration: dict[str, float],
        resolved: ResolvedR2RTrajectory,
        output_root: Path,
        *,
        output_format: str,
        backend: str,
        yellow_foot_z: float | None,
    ) -> Path: ...


def _noop_release_robot_model(_model: Any) -> None:
    pass


@dataclass(frozen=True, slots=True)
class R2RExecutorBindings:
    validate_spec: ValidateSpec
    resolve_trajectory: ResolveTrajectory
    get_source_model: GetRobotModel
    get_target_model: GetRobotModel
    prepare_source: PrepareSource
    load_pair_calibration: LoadPairCalibration
    prepare_motion: PrepareMotion
    run_r2r: RunR2R
    build_preview: BuildPreview
    write_export: WriteExport
    release_robot_model: Callable[[Any], None] = _noop_release_robot_model


def _execution_error(
    code: str,
    message: str,
    *,
    stage: ErrorStage = ErrorStage.EXECUTION,
    retryable: bool = False,
    details: Mapping[str, Any] | None = None,
) -> JobExecutionError:
    return JobExecutionError(
        ApiError(
            code=code,
            message=message,
            stage=stage,
            retryable=retryable,
            details=dict(details or {}),
        )
    )


def _parameters(spec: JobSpecV2) -> AgentR2RExecutionParameters:
    if (
        spec.kind is not JobSpecKind.R2R_RETARGET
        or len(spec.inputs) != 1
        or spec.source_robot is None
        or spec.calibration is None
    ):
        raise _execution_error(
            "INVALID_PARAMETER",
            "The R2R executor requires one trajectory, two robots, and pair calibration.",
        )
    if spec.output_policy is not OutputPolicy.CREATE_NEW:
        raise _execution_error(
            "UNSUPPORTED_OUTPUT_POLICY",
            "Managed Agent artifacts currently support only create_new output policy.",
            stage=ErrorStage.PREFLIGHT,
        )
    try:
        return AgentR2RExecutionParameters.model_validate(
            {**spec.effective_parameters, "backend": spec.backend}
        )
    except ValidationError as error:
        first = error.errors(include_url=False)[0] if error.error_count() else {}
        location = first.get("loc") if isinstance(first, dict) else None
        parameter = str(location[-1]) if isinstance(location, tuple | list) and location else None
        raise _execution_error(
            "INVALID_PARAMETER",
            "The persisted R2R JobSpec contains invalid execution parameters.",
            details={**({"parameter": parameter} if parameter else {})},
        ) from error


class _ProgressBridge:
    def __init__(self, context: JobExecutionContext, *, start: float, span: float) -> None:
        self._context = context
        self._start = start
        self._span = span

    def __call__(self, done: int, total: int) -> None:
        self._context.raise_if_cancelled()
        total = max(1, int(total))
        done = max(0, min(int(done), total))
        self._context.report_progress(
            phase="solving" if self._start >= 0.2 else "preparing",
            fraction=min(0.98, self._start + self._span * done / total),
            message=f"R2R {'solve' if self._start >= 0.2 else 'FK'} {done}/{total}",
            poll_after_ms=1_000,
        )


class R2RJobExecutor:
    """Map an exact R2R JobSpec onto existing FK, IK, preview, and export helpers."""

    def __init__(self, bindings: R2RExecutorBindings, *, temporary_root: Path) -> None:
        self._bindings = bindings
        self._temporary_root = Path(temporary_root)

    def __call__(
        self,
        spec: JobSpecV2,
        context: JobExecutionContext,
    ) -> JobExecutionResult:
        parameters = _parameters(spec)
        assert spec.source_robot is not None
        assert spec.calibration is not None
        stage = "validate_spec"
        source_model: Any | None = None
        target_model: Any | None = None
        try:
            context.raise_if_cancelled()
            self._bindings.validate_spec(spec)
            stage = "resolve_input"
            resolved = self._bindings.resolve_trajectory(spec.inputs[0].asset_id)
            if resolved.asset_id != spec.inputs[0].asset_id:
                raise ValueError("resolved trajectory identity differs from the JobSpec")
            if resolved.profile != "mimic" or resolved.has_scene:
                raise _execution_error(
                    "R2R_SCENE_UNSUPPORTED",
                    "The initial Agent R2R executor accepts only scene-free trajectories.",
                    stage=ErrorStage.PREFLIGHT,
                )
            if (
                resolved.source_robot_id is not None
                and resolved.source_robot_id != spec.source_robot.robot_id
            ):
                raise _execution_error(
                    "SOURCE_ROBOT_MISMATCH",
                    "The trajectory source identity differs from the immutable plan.",
                    stage=ErrorStage.ASSET_INSPECTION,
                )

            context.report_progress(
                phase="preparing",
                fraction=0.01,
                message="Materializing the verified R2R robot pair.",
                poll_after_ms=1_000,
            )
            stage = "load_source_robot"
            source_model = self._bindings.get_source_model(spec)
            context.raise_if_cancelled()
            stage = "load_target_robot"
            target_model = self._bindings.get_target_model(spec)
            context.raise_if_cancelled()
            stage = "prepare_input"
            source = self._bindings.prepare_source(
                resolved,
                source_model,
                source_fps=parameters.source_fps,
                limit_frames=parameters.limit_frames,
                progress_callback=_ProgressBridge(context, start=0.02, span=0.16),
            )
            context.raise_if_cancelled()
            motion, effective_fps = self._bindings.prepare_motion(
                source.motion,
                parameters.retarget_fps,
            )
            stage = "load_calibration"
            calibration = self._bindings.load_pair_calibration(source_model, target_model)
            context.raise_if_cancelled()

            # Re-hash all original assets and calibration after loaders have
            # consumed them, immediately before entering native solver code.
            stage = "validate_spec"
            self._bindings.validate_spec(spec)
            context.raise_if_cancelled()
            stage = "solve"
            retargeted = self._bindings.run_r2r(
                source_model,
                target_model,
                motion,
                calibration,
                backend=spec.backend,
                ik_iterations=parameters.ik_iterations,
                progress_callback=_ProgressBridge(context, start=0.2, span=0.77),
            )
            context.raise_if_cancelled()
            num_frames = int(retargeted.num_frames)
            sample_rate = float(retargeted.sample_rate)
            provenance = build_execution_provenance(
                retargeted,
                executor="existing_web_r2r_adapter_v1",
                backend=spec.backend,
                dataset="robot_trajectory",
                motion_asset_id=spec.inputs[0].asset_id,
                motion_sha256=spec.inputs[0].sha256,
                source_robot_asset_id=spec.source_robot.asset_id,
                source_robot_config_sha256=spec.source_robot.config_sha256,
                robot_asset_id=spec.robot.asset_id,
                robot_config_sha256=spec.robot.config_sha256,
                reference=f"robot_{spec.source_robot.robot_id}",
            )

            stage = "preview"
            context.report_progress(
                phase="evaluating",
                fraction=0.98,
                message="Building bounded R2R preview evidence.",
                poll_after_ms=1_000,
            )
            preview = self._bindings.build_preview(
                source_model,
                target_model,
                source,
                motion,
                calibration,
                retargeted,
            )
            context.raise_if_cancelled()
            context.publish_json(
                kind="preview",
                document=dict(preview.document),
                metadata={
                    "evidence_level": "kinematic_preview_heuristic",
                    "trajectory_asset_id": spec.inputs[0].asset_id,
                    "trajectory_sha256": spec.inputs[0].sha256,
                    "source_robot_asset_id": spec.source_robot.asset_id,
                    "source_robot_config_sha256": spec.source_robot.config_sha256,
                    "target_robot_asset_id": spec.robot.asset_id,
                    "target_robot_config_sha256": spec.robot.config_sha256,
                    "pair_calibration_id": spec.calibration.calibration_id,
                    "pair_calibration_sha256": spec.calibration.sha256,
                    "backend": spec.backend,
                    "num_frames": num_frames,
                    "sample_rate_hz": sample_rate,
                },
            )

            stage = "export"
            self._temporary_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="agent-r2r-",
                dir=self._temporary_root,
            ) as temporary:
                temporary_path = Path(temporary).resolve(strict=True)
                exported = self._bindings.write_export(
                    retargeted,
                    source_model,
                    target_model,
                    motion,
                    calibration,
                    resolved,
                    temporary_path,
                    output_format=parameters.output_format,
                    backend=spec.backend,
                    yellow_foot_z=preview.yellow_foot_z,
                )
                context.raise_if_cancelled()
                try:
                    exported = Path(exported).resolve(strict=True)
                    exported.relative_to(temporary_path)
                except (OSError, RuntimeError, ValueError) as error:
                    raise _execution_error(
                        "OUTPUT_PATH_ESCAPE",
                        "The existing R2R exporter returned a file outside its managed root.",
                        stage=ErrorStage.ARTIFACT,
                    ) from error
                if not exported.is_file():
                    raise _execution_error(
                        "OUTPUT_WRITE_FAILED",
                        "The existing R2R exporter did not produce a regular file.",
                        stage=ErrorStage.ARTIFACT,
                        retryable=True,
                    )
                suffix = exported.suffix.casefold().lstrip(".") or None
                media_type = {
                    "csv": "text/csv",
                    "pkl": "application/octet-stream",
                    "zip": "application/zip",
                }.get(suffix or "", "application/octet-stream")
                context.publish_file(
                    kind="retargeted_motion",
                    source=exported,
                    format=suffix,
                    media_type=media_type,
                    metadata={
                        "filename": exported.name,
                        "requested_format": parameters.output_format,
                        "content_format": suffix,
                        "trajectory_asset_id": spec.inputs[0].asset_id,
                        "source_robot_asset_id": spec.source_robot.asset_id,
                        "target_robot_asset_id": spec.robot.asset_id,
                        "pair_calibration_id": spec.calibration.calibration_id,
                        "backend": spec.backend,
                        "num_frames": num_frames,
                        "sample_rate_hz": sample_rate,
                    },
                )
            context.raise_if_cancelled()

            outcome, evaluation_summary, metrics, checks = _quality_verdict(preview.diagnostics)
            return JobExecutionResult(
                outcome=outcome,
                summary={
                    "workflow": "r2r",
                    "run_mode": parameters.run_mode,
                    "stem": resolved.stem,
                    "num_frames": num_frames,
                    "trajectory_source_fps": source.source_fps,
                    "retarget_fps": float(effective_fps),
                    "source_fps": sample_rate,
                    "output_format": parameters.output_format,
                    "has_scene": False,
                },
                evaluation_summary=evaluation_summary,
                evaluation_metrics=metrics,
                evaluation_checks=checks,
                execution_provenance=provenance.model_dump(mode="json", exclude_none=True),
            )
        except (JobCancelledError, JobExecutionError, ArtifactStoreError):
            raise
        except AssetServiceError as error:
            raise JobExecutionError(error.api_error) from error
        except RetargetServiceError as error:
            raise JobExecutionError(error.api_error) from error
        except Exception as error:
            if isinstance(error, (ImportError, ModuleNotFoundError)):
                dependency = getattr(error, "name", None)
                raise _execution_error(
                    "BACKEND_UNAVAILABLE",
                    "A dependency required by the R2R execution path is unavailable.",
                    details={
                        "operation": stage,
                        **({"dependency": dependency} if dependency else {}),
                    },
                ) from error
            if _is_cuda_oom(error):
                raise _execution_error(
                    "CUDA_OUT_OF_MEMORY",
                    "The R2R backend exhausted the available GPU memory.",
                    retryable=True,
                    details={"operation": stage},
                ) from error
            code, message, error_stage, retryable = {
                "resolve_input": (
                    "ASSET_INVALID",
                    "The verified robot trajectory could not be resolved.",
                    ErrorStage.ASSET_INSPECTION,
                    False,
                ),
                "prepare_input": (
                    "ROBOT_TRAJECTORY_PARSE_FAILED",
                    "The source trajectory could not be prepared for R2R.",
                    ErrorStage.ASSET_INSPECTION,
                    False,
                ),
                "load_source_robot": (
                    "ROBOT_LOAD_FAILED",
                    "The source robot bundle could not be materialized.",
                    ErrorStage.EXECUTION,
                    False,
                ),
                "load_target_robot": (
                    "ROBOT_LOAD_FAILED",
                    "The target robot bundle could not be materialized.",
                    ErrorStage.EXECUTION,
                    False,
                ),
                "load_calibration": (
                    "R2R_CALIBRATION_MISMATCH",
                    "The pair calibration could not be loaded for execution.",
                    ErrorStage.PREFLIGHT,
                    False,
                ),
                "preview": (
                    "OUTPUT_VALIDATION_FAILED",
                    "The R2R result could not be converted into preview evidence.",
                    ErrorStage.ARTIFACT,
                    False,
                ),
                "export": (
                    "OUTPUT_WRITE_FAILED",
                    "The existing R2R exporter could not produce the requested artifact.",
                    ErrorStage.ARTIFACT,
                    True,
                ),
            }.get(
                stage,
                (
                    "SOLVER_FAILED",
                    "The existing R2R execution path stopped before producing a result.",
                    ErrorStage.EXECUTION,
                    False,
                ),
            )
            raise _execution_error(
                code,
                message,
                stage=error_stage,
                retryable=retryable,
                details={"operation": stage, "exception_type": type(error).__name__},
            ) from error
        finally:
            for model in (target_model, source_model):
                if model is None:
                    continue
                try:
                    self._bindings.release_robot_model(model)
                except Exception:  # noqa: BLE001 - cleanup must not mask job truth
                    _log.warning("failed to release Agent robot workspace", exc_info=True)


__all__ = [
    "PreparedR2RSource",
    "R2RExecutorBindings",
    "R2RJobExecutor",
    "R2RPreview",
    "ResolvedR2RTrajectory",
]
