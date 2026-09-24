"""Thin Agent ``JobExecutor`` adapter over the existing H2R Web runtime.

The adapter owns orchestration only.  Motion loading, robot loading, retarget
math, preview construction, and export remain injected legacy callables so this
module cannot silently fork IK, calibration, or output semantics.
"""

from __future__ import annotations

import logging
import math
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from hhtools.contracts import (
    AgentH2RExecutionParameters,
    ApiError,
    AssetCategory,
    ErrorStage,
    JobOutcome,
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

_TRACKING_STABLE_P95_M = 0.05
_TRACKING_REVIEW_P95_M = 0.10

_log = logging.getLogger(__name__)


class ValidateSpec(Protocol):
    """Rebuild and exact-match a persisted spec against current asset bytes."""

    def __call__(self, spec: JobSpecV2) -> None: ...


@dataclass(frozen=True, slots=True)
class ResolvedMotion:
    """Trusted in-process motion routing facts; never serialized to clients."""

    asset_id: str
    category: AssetCategory
    dataset: str
    source_path: Path
    stem: str


class ResolveMotion(Protocol):
    """Resolve one verified bundle to its trusted primary and routing facts."""

    def __call__(self, asset_id: str) -> ResolvedMotion: ...


class LoadMotion(Protocol):
    def __call__(self, resolved: ResolvedMotion) -> Any: ...


class PrepareMotion(Protocol):
    def __call__(self, motion: Any, retarget_fps: float | None) -> tuple[Any, float]: ...


class GetRobotModel(Protocol):
    def __call__(self, spec: JobSpecV2) -> Any: ...


class RunRetarget(Protocol):
    def __call__(
        self,
        model: Any,
        robot_id: str,
        motion: Any,
        reference: str,
        backend: str,
        ik_iterations: int,
        human_height: float,
        limit_frames: int | None,
        progress_job: Any,
        *,
        foot_clamp_anti_penetration: bool,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class H2RPreview:
    """Existing Web preview/diagnostic payload prepared for artifact storage."""

    document: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    yellow_foot_z: float | None = None


class BuildPreview(Protocol):
    def __call__(
        self,
        model: Any,
        robot_id: str,
        motion: Any,
        reference: str,
        human_height: float,
        retargeted: Any,
    ) -> H2RPreview: ...


class WriteExport(Protocol):
    def __call__(
        self,
        retargeted: Any,
        model: Any,
        source_motion: Any,
        output_root: Path,
        *,
        stem: str,
        output_format: str,
        backend: str,
        source_path: Path,
        yellow_foot_z: float | None,
    ) -> Path: ...


def _noop_release_robot_model(_model: Any) -> None:
    """Default release hook for bindings that do not own a model workspace."""


@dataclass(frozen=True, slots=True)
class H2RExecutorBindings:
    """Calls into the existing H2R implementation without owning its math."""

    validate_spec: ValidateSpec
    resolve_motion: ResolveMotion
    load_motion: LoadMotion
    ground_motion: Callable[[Any], Any]
    prepare_motion: PrepareMotion
    get_robot_model: GetRobotModel
    run_retarget: RunRetarget
    build_preview: BuildPreview
    write_export: WriteExport
    release_robot_model: Callable[[Any], None] = _noop_release_robot_model


class _LegacyProgressBridge:
    """Present the tiny mutable shape expected by the existing solver helpers."""

    kind = "retarget"

    def __init__(self, context: JobExecutionContext) -> None:
        self._context = context
        self._fraction = 0.0
        self._message: str | None = None

    @property
    def progress(self) -> float:
        return self._fraction

    @progress.setter
    def progress(self, value: float) -> None:
        self._context.raise_if_cancelled()
        # Interaction-Mesh may report a stage-local zero after the legacy
        # wrapper has already emitted its setup fraction.  Agent progress is a
        # monotonic job fact, so retain the larger value without changing the
        # solver callback itself.
        self._fraction = max(self._fraction, min(0.99, max(0.0, float(value))))

    @property
    def clip_progress(self) -> float:
        return self._fraction

    @clip_progress.setter
    def clip_progress(self, value: float) -> None:
        self.progress = value

    @property
    def message(self) -> str | None:
        return self._message

    @message.setter
    def message(self, value: str) -> None:
        self._context.raise_if_cancelled()
        self._message = str(value)
        self._context.report_progress(
            phase="solving",
            fraction=self._fraction,
            message=self._message,
            poll_after_ms=1_000,
        )


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


def _parameters(spec: JobSpecV2) -> AgentH2RExecutionParameters:
    if spec.kind is not JobSpecKind.RETARGET or len(spec.inputs) != 1:
        raise _execution_error(
            "INVALID_PARAMETER",
            "The H2R executor requires one single-retarget JobSpec.",
        )
    if spec.output_policy is not OutputPolicy.CREATE_NEW:
        raise _execution_error(
            "UNSUPPORTED_OUTPUT_POLICY",
            "Managed Agent artifacts currently support only create_new output policy.",
            stage=ErrorStage.PREFLIGHT,
            details={"output_policy": spec.output_policy.value},
        )
    try:
        return AgentH2RExecutionParameters.model_validate(
            {**spec.effective_parameters, "backend": spec.backend}
        )
    except ValidationError as error:
        first = error.errors(include_url=False)[0] if error.error_count() else {}
        location = first.get("loc") if isinstance(first, dict) else None
        parameter = str(location[-1]) if isinstance(location, tuple | list) and location else None
        raise _execution_error(
            "INVALID_PARAMETER",
            "The persisted JobSpec contains invalid execution parameters.",
            details={**({"parameter": parameter} if parameter else {})},
        ) from error


def _quality_verdict(
    diagnostics: Mapping[str, Any],
) -> tuple[JobOutcome, str, dict[str, Any], list[dict[str, Any]]]:
    """Report UI diagnostic bands without treating them as acceptance criteria."""

    evidence = "kinematic_preview_heuristic"
    tracking = diagnostics.get("tracking")
    p95_value = tracking.get("p95_error_m") if isinstance(tracking, Mapping) else None
    if (
        not diagnostics.get("available")
        or isinstance(p95_value, bool)
        or not isinstance(p95_value, int | float)
        or not math.isfinite(float(p95_value))
        or float(p95_value) < 0.0
    ):
        return (
            JobOutcome.REVIEW_REQUIRED,
            "The existing Web diagnostics could not produce a tracking verdict.",
            {
                "evidence_level": evidence,
                "diagnostics_available": False,
                "quality_band": "unavailable",
            },
            [
                {
                    "code": "TRACKING_DIAGNOSTICS_UNAVAILABLE",
                    "status": "review_required",
                    "evidence_level": evidence,
                }
            ],
        )

    p95 = float(p95_value)
    metrics: dict[str, Any] = {
        "evidence_level": evidence,
        "diagnostics_available": True,
        "tracking_p95_error_m": p95,
    }
    if isinstance(tracking, Mapping):
        for source, target in (
            ("mean_error_m", "tracking_mean_error_m"),
            ("max_error_m", "tracking_max_error_m"),
        ):
            value = tracking.get(source)
            if isinstance(value, int | float) and not isinstance(value, bool):
                normalized = float(value)
                if math.isfinite(normalized):
                    metrics[target] = normalized

    if p95 <= _TRACKING_STABLE_P95_M:
        quality_band = "stable"
    elif p95 <= _TRACKING_REVIEW_P95_M:
        quality_band = "review"
    else:
        quality_band = "high_error"
    metrics["quality_band"] = quality_band
    return (
        JobOutcome.REVIEW_REQUIRED,
        (
            f"The kinematic preview is in the {quality_band} display band; "
            "a validated evaluator or human must decide acceptance."
        ),
        metrics,
        [
            {
                "code": "TRACKING_P95_ERROR",
                "status": "review_required",
                "observed_band": quality_band,
                "value_m": p95,
                "stable_max_m": _TRACKING_STABLE_P95_M,
                "review_max_m": _TRACKING_REVIEW_P95_M,
                "evidence_level": evidence,
            }
        ],
    )


_WINDOWS_RESERVED_STEMS = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_PORTABLE_STEM_FORBIDDEN = frozenset('<>:"/\\|?*')


def _validate_resolved_motion(
    resolved: ResolvedMotion,
    *,
    expected_asset_id: str,
) -> ResolvedMotion:
    """Defend the exporter boundary even when a custom binding is injected."""

    stem = resolved.stem
    if (
        resolved.asset_id != expected_asset_id
        or not isinstance(resolved.dataset, str)
        or not resolved.dataset
        or len(resolved.dataset) > 128
        or not isinstance(stem, str)
        or not stem
        or len(stem) > 128
        or stem in {".", ".."}
        or stem != stem.strip()
        or stem.endswith(".")
        or any(character in _PORTABLE_STEM_FORBIDDEN for character in stem)
        or any(ord(character) < 32 for character in stem)
        or stem.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_STEMS
    ):
        raise _execution_error(
            "BUNDLE_METADATA_MISMATCH",
            "The verified motion bundle has unsafe or inconsistent routing metadata.",
            stage=ErrorStage.ASSET_INSPECTION,
        )
    return resolved


def _is_cuda_oom(error: BaseException) -> bool:
    message = str(error).casefold()
    return "out of memory" in message and ("cuda" in message or "gpu" in message)


class H2RJobExecutor:
    """Map an exact JobSpec v2 onto the already-working H2R call chain."""

    def __init__(self, bindings: H2RExecutorBindings, *, temporary_root: Path) -> None:
        self._bindings = bindings
        self._temporary_root = Path(temporary_root)

    def __call__(
        self,
        spec: JobSpecV2,
        context: JobExecutionContext,
    ) -> JobExecutionResult:
        parameters = _parameters(spec)
        stage = "validate_spec"
        model: Any | None = None
        try:
            context.raise_if_cancelled()
            self._bindings.validate_spec(spec)
            context.raise_if_cancelled()
            context.report_progress(
                phase="preparing",
                fraction=0.01,
                message="Resolving the verified motion bundle.",
                poll_after_ms=1_000,
            )
            stage = "resolve_input"
            resolved = _validate_resolved_motion(
                self._bindings.resolve_motion(spec.inputs[0].asset_id),
                expected_asset_id=spec.inputs[0].asset_id,
            )
            source_path = Path(resolved.source_path)
            context.raise_if_cancelled()
            if (
                parameters.output_format == "pkl"
                and resolved.category is AssetCategory.OBJECT_INTERACTION
            ):
                raise _execution_error(
                    "UNSUPPORTED_PORTABLE_EXPORT",
                    "Object-interaction PKL export is not portable in the existing H2R path.",
                    stage=ErrorStage.PREFLIGHT,
                    details={
                        "category": resolved.category.value,
                        "output_format": parameters.output_format,
                    },
                )

            stage = "load_input"
            motion = self._bindings.load_motion(resolved)
            context.raise_if_cancelled()
            stage = "ground_input"
            motion = self._bindings.ground_motion(motion)
            context.raise_if_cancelled()
            motion_source_fps = float(motion.framerate)
            stage = "prepare_input"
            motion, effective_fps = self._bindings.prepare_motion(
                motion,
                parameters.retarget_fps,
            )
            context.raise_if_cancelled()

            stage = "load_robot"
            model = self._bindings.get_robot_model(spec)
            context.raise_if_cancelled()
            context.report_progress(
                phase="preparing",
                fraction=0.02,
                message="The verified motion and robot are ready.",
                poll_after_ms=1_000,
            )

            # Model construction and dataset loaders can read robot profiles and
            # motion sidecars.  Rebuild the immutable spec immediately before
            # solving so a queued job never silently consumes changed bytes.
            stage = "validate_spec"
            self._bindings.validate_spec(spec)
            context.raise_if_cancelled()

            stage = "solve"
            progress = _LegacyProgressBridge(context)
            retargeted = self._bindings.run_retarget(
                model,
                spec.robot.robot_id,
                motion,
                parameters.reference,
                spec.backend,
                parameters.ik_iterations,
                parameters.human_height,
                parameters.limit_frames,
                progress,
                foot_clamp_anti_penetration=(parameters.foot_clamp_anti_penetration),
            )
            context.raise_if_cancelled()
            num_frames = int(retargeted.num_frames)
            sample_rate = float(retargeted.sample_rate)
            execution_provenance = build_execution_provenance(
                retargeted,
                executor="existing_web_h2r_adapter_v1",
                backend=spec.backend,
                dataset=resolved.dataset,
                motion_asset_id=spec.inputs[0].asset_id,
                motion_sha256=spec.inputs[0].sha256,
                robot_asset_id=spec.robot.asset_id,
                robot_config_sha256=spec.robot.config_sha256,
                reference=parameters.reference,
            )

            stage = "preview"
            context.report_progress(
                phase="evaluating",
                fraction=0.985,
                message="Building the existing Web preview and diagnostics.",
                poll_after_ms=1_000,
            )
            preview = self._bindings.build_preview(
                model,
                spec.robot.robot_id,
                motion,
                parameters.reference,
                parameters.human_height,
                retargeted,
            )
            context.raise_if_cancelled()
            context.publish_json(
                kind="preview",
                document=dict(preview.document),
                metadata={
                    "evidence_level": "kinematic_preview_heuristic",
                    "motion_asset_id": spec.inputs[0].asset_id,
                    "motion_sha256": spec.inputs[0].sha256,
                    "robot_asset_id": spec.robot.asset_id,
                    "robot_config_sha256": spec.robot.config_sha256,
                    "reference": parameters.reference,
                    "backend": spec.backend,
                    "num_frames": num_frames,
                    "sample_rate_hz": sample_rate,
                },
            )
            context.raise_if_cancelled()

            stage = "export"
            context.raise_if_cancelled()
            self._temporary_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="agent-h2r-",
                dir=self._temporary_root,
            ) as temporary:
                temporary_path = Path(temporary).resolve(strict=True)
                exported = self._bindings.write_export(
                    retargeted,
                    model,
                    motion,
                    temporary_path,
                    stem=resolved.stem,
                    output_format=parameters.output_format,
                    backend=spec.backend,
                    source_path=source_path,
                    yellow_foot_z=preview.yellow_foot_z,
                )
                context.raise_if_cancelled()
                try:
                    exported = Path(exported).resolve(strict=True)
                    exported.relative_to(temporary_path)
                except (OSError, RuntimeError, ValueError) as error:
                    raise _execution_error(
                        "OUTPUT_PATH_ESCAPE",
                        "The existing H2R exporter returned a file outside its managed root.",
                        stage=ErrorStage.ARTIFACT,
                    ) from error
                if not exported.is_file():
                    raise _execution_error(
                        "OUTPUT_WRITE_FAILED",
                        "The existing H2R exporter did not produce a regular file.",
                        stage=ErrorStage.ARTIFACT,
                        retryable=True,
                    )
                context.raise_if_cancelled()
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
                        "motion_asset_id": spec.inputs[0].asset_id,
                        "motion_sha256": spec.inputs[0].sha256,
                        "robot_asset_id": spec.robot.asset_id,
                        "robot_config_sha256": spec.robot.config_sha256,
                        "reference": parameters.reference,
                        "backend": spec.backend,
                        "num_frames": num_frames,
                        "sample_rate_hz": sample_rate,
                    },
                )
            context.raise_if_cancelled()

            outcome, evaluation_summary, metrics, checks = _quality_verdict(preview.diagnostics)
            has_scene = bool(
                getattr(motion, "terrain", None) is not None or getattr(motion, "objects", ())
            )
            return JobExecutionResult(
                outcome=outcome,
                summary={
                    "run_mode": parameters.run_mode,
                    "stem": resolved.stem,
                    "num_frames": num_frames,
                    "motion_source_fps": motion_source_fps,
                    "retarget_fps": float(effective_fps),
                    "source_fps": sample_rate,
                    "output_format": parameters.output_format,
                    "has_scene": has_scene,
                },
                evaluation_summary=evaluation_summary,
                evaluation_metrics=metrics,
                evaluation_checks=checks,
                execution_provenance=execution_provenance.model_dump(
                    mode="json",
                    exclude_none=True,
                ),
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
                    "A dependency required by the selected H2R execution path is unavailable.",
                    details={
                        "operation": stage,
                        **({"dependency": dependency} if dependency else {}),
                    },
                ) from error
            if _is_cuda_oom(error):
                raise _execution_error(
                    "CUDA_OUT_OF_MEMORY",
                    "The H2R backend exhausted the available GPU memory.",
                    retryable=True,
                    details={"operation": stage},
                ) from error
            code, message, error_stage, retryable = {
                "resolve_input": (
                    "ASSET_INVALID",
                    "The verified motion asset could not be resolved for execution.",
                    ErrorStage.ASSET_INSPECTION,
                    False,
                ),
                "load_input": (
                    "MOTION_PARSE_FAILED",
                    "The verified motion could not be loaded by its dataset adapter.",
                    ErrorStage.ASSET_INSPECTION,
                    False,
                ),
                "ground_input": (
                    "MOTION_PARSE_FAILED",
                    "The verified motion could not be prepared for retargeting.",
                    ErrorStage.ASSET_INSPECTION,
                    False,
                ),
                "prepare_input": (
                    "MOTION_PARSE_FAILED",
                    "The verified motion could not be prepared for retargeting.",
                    ErrorStage.ASSET_INSPECTION,
                    False,
                ),
                "load_robot": (
                    "ROBOT_LOAD_FAILED",
                    "The verified robot bundle could not be materialized.",
                    ErrorStage.EXECUTION,
                    False,
                ),
                "preview": (
                    "OUTPUT_VALIDATION_FAILED",
                    "The H2R result could not be converted into bounded preview evidence.",
                    ErrorStage.ARTIFACT,
                    False,
                ),
                "export": (
                    "OUTPUT_WRITE_FAILED",
                    "The existing H2R exporter could not produce the requested artifact.",
                    ErrorStage.ARTIFACT,
                    True,
                ),
            }.get(
                stage,
                (
                    "SOLVER_FAILED",
                    "The existing H2R execution path stopped before producing a result.",
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
            if model is not None:
                try:
                    self._bindings.release_robot_model(model)
                except Exception:  # noqa: BLE001 - cleanup must not mask job truth
                    _log.warning("failed to release Agent robot workspace", exc_info=True)


__all__ = [
    "H2RExecutorBindings",
    "H2RJobExecutor",
    "H2RPreview",
    "ResolvedMotion",
]
