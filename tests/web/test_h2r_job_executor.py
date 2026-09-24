from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from hhtools.agent.h2r_job_executor import (
    H2RExecutorBindings,
    H2RJobExecutor,
    H2RPreview,
    ResolvedMotion,
)
from hhtools.contracts import (
    ApiError,
    AssetCategory,
    ErrorStage,
    JobOutcome,
    JobProgress,
    JobSpecInput,
    JobSpecKind,
    JobSpecProvenance,
    JobSpecRobot,
    JobSpecV2,
    OutputPolicy,
)
from hhtools.services.artifacts import ArtifactStore
from hhtools.services.jobs import JobCancelledError, JobExecutionContext, JobExecutionError
from hhtools.services.retarget import RetargetServiceError

SHA_A = "a" * 64
SHA_B = "b" * 64
PLAN_ID = f"plan:sha256:{'c' * 64}"
MOTION_ID = f"asset:sha256:{SHA_A}"
ROBOT_ID = f"asset:sha256:{SHA_B}"


@dataclass
class _Motion:
    name: str = "walk"
    framerate: float = 60.0
    terrain: Any | None = None
    objects: list[Any] = field(default_factory=list)


@dataclass
class _Retargeted:
    num_frames: int = 30
    sample_rate: float = 30.0
    meta: dict[str, Any] = field(
        default_factory=lambda: {
            "execution_provenance": {
                "backend": "newton",
                "device": "cuda:0",
                "device_kind": "cuda",
                "precision": "float32",
                "runtime": "warp",
                "runtime_version": "1.12.1",
                "solver": "newton-1.1.0",
                "cuda_graph_requested": True,
                "cuda_graph_used": True,
                "fallback_used": False,
            }
        }
    )


def _spec(*, backend: str = "newton", **parameter_updates: Any) -> JobSpecV2:
    parameters: dict[str, Any] = {
        "run_mode": "smoke",
        "limit_frames": 30,
        "ik_iterations": 24,
        "human_height": 1.72,
        "retarget_fps": 30.0,
        "foot_clamp_anti_penetration": True,
        "reference": "smpl",
        "retarget_profile": "bundled_scaler",
        "output_format": "csv",
    }
    parameters.update(parameter_updates)
    if backend == "interaction_mesh":
        parameters.pop("ik_iterations", None)
    return JobSpecV2(
        kind=JobSpecKind.RETARGET,
        plan_id=PLAN_ID,
        inputs=[JobSpecInput(asset_id=MOTION_ID, sha256=SHA_A)],
        robot=JobSpecRobot(
            robot_id="g1_29dof",
            asset_id=ROBOT_ID,
            config_sha256=SHA_B,
        ),
        calibration=None,
        backend=backend,
        effective_parameters=parameters,
        output_policy=OutputPolicy.CREATE_NEW,
        provenance=JobSpecProvenance(
            hhtools_git_commit="test",
            hhtools_dirty=False,
            python="3.12",
        ),
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )


def _context(
    tmp_path: Path,
    spec: JobSpecV2,
    *,
    cancellation_event: threading.Event | None = None,
) -> tuple[JobExecutionContext, ArtifactStore, list[tuple[JobProgress, int | None]]]:
    artifacts = ArtifactStore(tmp_path / "agent")
    progress: list[tuple[JobProgress, int | None]] = []
    context = JobExecutionContext(
        job_id="job:test-h2r",
        spec=spec,
        artifact_store=artifacts,
        cancellation_event=cancellation_event or threading.Event(),
        progress_callback=lambda item, delay: progress.append((item, delay)),
    )
    return context, artifacts, progress


def _bindings(
    tmp_path: Path,
    calls: dict[str, Any],
    *,
    p95_error_m: float | None = 0.04,
    fail_export: bool = False,
    category: AssetCategory = AssetCategory.PLAIN_MOTION,
    dataset: str = "amass",
    motion_name: str = "walk",
    outside_export: bool = False,
    has_scene: bool = False,
) -> H2RExecutorBindings:
    source_path = tmp_path / "walk.npz"
    source_path.write_bytes(b"source")
    motion = _Motion(name=motion_name, objects=[object()] if has_scene else [])
    model = object()
    retargeted = _Retargeted()

    def validate_spec(spec: JobSpecV2) -> None:
        calls.setdefault("validated_specs", []).append(spec)

    def resolve_motion(asset_id: str) -> ResolvedMotion:
        calls["resolved_asset_id"] = asset_id
        return ResolvedMotion(
            asset_id=asset_id,
            category=category,
            dataset=dataset,
            source_path=source_path,
            stem="walk",
        )

    def load_motion(resolved: ResolvedMotion) -> _Motion:
        calls["load"] = resolved
        return motion

    def ground_motion(value: _Motion) -> _Motion:
        calls["grounded"] = value
        return value

    def prepare_motion(value: _Motion, fps: float | None) -> tuple[_Motion, float]:
        calls["prepare"] = (value, fps)
        return value, 30.0

    def get_robot_model(spec: JobSpecV2) -> object:
        calls["robot_spec"] = spec
        return model

    def release_robot_model(model_value: object) -> None:
        calls.setdefault("released_models", []).append(model_value)

    def run_retarget(
        model_value: object,
        robot_id: str,
        motion_value: _Motion,
        reference: str,
        backend: str,
        ik_iterations: int,
        human_height: float,
        limit_frames: int | None,
        progress_job: Any,
        *,
        foot_clamp_anti_penetration: bool,
    ) -> _Retargeted:
        calls["run"] = {
            "model": model_value,
            "robot_id": robot_id,
            "motion": motion_value,
            "reference": reference,
            "backend": backend,
            "ik_iterations": ik_iterations,
            "human_height": human_height,
            "limit_frames": limit_frames,
            "foot_clamp_anti_penetration": foot_clamp_anti_penetration,
        }
        progress_job.progress = 0.5
        progress_job.message = "IK 15/30"
        return retargeted

    diagnostics: dict[str, Any]
    if p95_error_m is None:
        diagnostics = {"schema_version": 1, "available": False, "reason": "missing"}
    else:
        diagnostics = {
            "schema_version": 1,
            "available": True,
            "tracking": {
                "mean_error_m": p95_error_m / 2,
                "p95_error_m": p95_error_m,
                "max_error_m": p95_error_m * 2,
                "series": [{"frame": 0}],
            },
        }

    def build_preview(*args: Any) -> H2RPreview:
        calls["preview_args"] = args
        return H2RPreview(
            document={"trajectory": [[0.0]], "diagnostics": diagnostics},
            diagnostics=diagnostics,
            yellow_foot_z=0.012,
        )

    def write_export(
        retargeted_value: _Retargeted,
        model_value: object,
        source_motion: _Motion,
        output_root: Path,
        **kwargs: Any,
    ) -> Path:
        calls["export"] = {
            "retargeted": retargeted_value,
            "model": model_value,
            "motion": source_motion,
            "output_root": output_root,
            **kwargs,
        }
        if fail_export:
            raise OSError("disk unavailable")
        if outside_export:
            output = tmp_path / "outside.csv"
            output.write_bytes(b"outside")
            return output
        output = output_root / "walk.csv"
        output.write_bytes(b"root_x,root_y\n0,0\n")
        return output

    return H2RExecutorBindings(
        validate_spec=validate_spec,
        resolve_motion=resolve_motion,
        load_motion=load_motion,
        ground_motion=ground_motion,
        prepare_motion=prepare_motion,
        get_robot_model=get_robot_model,
        run_retarget=run_retarget,
        build_preview=build_preview,
        write_export=write_export,
        release_robot_model=release_robot_model,
    )


def test_executor_maps_exact_job_spec_to_existing_h2r_chain_and_managed_artifacts(
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}
    spec = _spec()
    context, artifact_store, progress = _context(tmp_path, spec)
    executor = H2RJobExecutor(
        _bindings(tmp_path, calls),
        temporary_root=tmp_path / "temporary",
    )

    result = executor(spec, context)

    assert result.outcome is JobOutcome.REVIEW_REQUIRED
    assert result.summary == {
        "run_mode": "smoke",
        "stem": "walk",
        "num_frames": 30,
        "motion_source_fps": 60.0,
        "retarget_fps": 30.0,
        "source_fps": 30.0,
        "output_format": "csv",
        "has_scene": False,
    }
    assert result.execution_provenance == {
        "executor": "existing_web_h2r_adapter_v1",
        "backend": "newton",
        "device": "cuda:0",
        "device_kind": "cuda",
        "precision": "float32",
        "runtime": "warp",
        "runtime_version": "1.12.1",
        "solver": "newton-1.1.0",
        "cuda_graph_requested": True,
        "cuda_graph_used": True,
        "fallback_used": False,
        "dataset": "amass",
        "motion_asset_id": MOTION_ID,
        "motion_sha256": SHA_A,
        "robot_asset_id": ROBOT_ID,
        "robot_config_sha256": SHA_B,
        "reference": "smpl",
    }
    assert calls["validated_specs"] == [spec, spec]
    assert calls["resolved_asset_id"] == MOTION_ID
    assert calls["load"].dataset == "amass"
    assert calls["robot_spec"] is spec
    assert calls["released_models"] == [calls["preview_args"][0]]
    assert calls["prepare"][1] == 30.0
    assert calls["run"] == {
        "model": calls["preview_args"][0],
        "robot_id": "g1_29dof",
        "motion": calls["prepare"][0],
        "reference": "smpl",
        "backend": "newton",
        "ik_iterations": 24,
        "human_height": 1.72,
        "limit_frames": 30,
        "foot_clamp_anti_penetration": True,
    }
    assert calls["export"]["output_format"] == "csv"
    assert calls["export"]["yellow_foot_z"] == 0.012
    assert any(item.phase == "solving" and item.fraction == 0.5 for item, _ in progress)

    published = context.published_artifacts()
    assert [item.kind for item in published] == ["preview", "retargeted_motion"]
    preview = artifact_store.get(published[0].artifact_id, verify=True)
    exported = artifact_store.get(published[1].artifact_id, verify=True)
    assert b'"trajectory":[[0.0]]' in preview.path.read_bytes()
    assert exported.path.read_bytes() == b"root_x,root_y\n0,0\n"
    assert exported.descriptor.metadata["filename"] == "walk.csv"
    assert exported.descriptor.metadata["requested_format"] == "csv"
    assert exported.descriptor.metadata["content_format"] == "csv"
    assert exported.descriptor.metadata["motion_asset_id"] == MOTION_ID
    assert exported.descriptor.metadata["robot_asset_id"] == ROBOT_ID
    assert exported.descriptor.metadata["num_frames"] == 30
    assert preview.descriptor.metadata["evidence_level"] == "kinematic_preview_heuristic"


def test_executor_preserves_preflighted_interaction_mesh_identity_and_scene(
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}
    spec = _spec(backend="interaction_mesh")
    context, _artifact_store, _progress = _context(tmp_path, spec)
    executor = H2RJobExecutor(
        _bindings(
            tmp_path,
            calls,
            category=AssetCategory.OBJECT_INTERACTION,
            dataset="omomo",
            has_scene=True,
        ),
        temporary_root=tmp_path / "temporary",
    )

    result = executor(spec, context)

    assert result.outcome is JobOutcome.REVIEW_REQUIRED
    assert result.summary["has_scene"] is True
    assert result.execution_provenance["backend"] == "interaction_mesh"
    assert result.execution_provenance["dataset"] == "omomo"
    assert calls["run"]["backend"] == "interaction_mesh"
    assert calls["run"]["ik_iterations"] == 24
    assert calls["export"]["backend"] == "interaction_mesh"
    assert {item.kind for item in context.published_artifacts()} == {
        "preview",
        "retargeted_motion",
    }


@pytest.mark.parametrize(
    ("p95", "expected_band"),
    [
        (0.05, "stable"),
        (0.05001, "review"),
        (0.10, "review"),
        (0.10001, "high_error"),
        (None, "unavailable"),
    ],
)
def test_executor_reports_web_tracking_band_without_claiming_quality_acceptance(
    tmp_path: Path,
    p95: float | None,
    expected_band: str,
) -> None:
    calls: dict[str, Any] = {}
    spec = _spec()
    context, _artifact_store, _progress = _context(tmp_path, spec)
    executor = H2RJobExecutor(
        _bindings(tmp_path, calls, p95_error_m=p95),
        temporary_root=tmp_path / "temporary",
    )

    result = executor(spec, context)

    assert result.outcome is JobOutcome.REVIEW_REQUIRED
    assert result.evaluation_metrics["evidence_level"] == "kinematic_preview_heuristic"
    assert result.evaluation_metrics["quality_band"] == expected_band
    assert result.evaluation_checks[0]["code"].startswith("TRACKING_")
    assert result.evaluation_checks[0]["status"] == "review_required"


@pytest.mark.parametrize("backend", ["newton", "interaction_mesh"])
def test_executor_acknowledges_running_cancellation_at_legacy_progress_boundary(
    tmp_path: Path,
    backend: str,
) -> None:
    calls: dict[str, Any] = {}
    spec = _spec(backend=backend)
    cancelled = threading.Event()
    context, _artifact_store, _progress = _context(
        tmp_path,
        spec,
        cancellation_event=cancelled,
    )
    bindings = _bindings(
        tmp_path,
        calls,
        category=(
            AssetCategory.OBJECT_INTERACTION
            if backend == "interaction_mesh"
            else AssetCategory.PLAIN_MOTION
        ),
        dataset="omomo" if backend == "interaction_mesh" else "amass",
        has_scene=backend == "interaction_mesh",
    )

    original_run = bindings.run_retarget

    def cancel_then_run(*args: Any, **kwargs: Any) -> Any:
        cancelled.set()
        return original_run(*args, **kwargs)

    executor = H2RJobExecutor(
        replace(bindings, run_retarget=cancel_then_run),
        temporary_root=tmp_path / "temporary",
    )

    with pytest.raises(JobCancelledError):
        executor(spec, context)

    assert "preview_args" not in calls
    assert "export" not in calls
    assert context.published_artifacts() == ()
    assert calls["released_models"] == [calls["run"]["model"]]


@pytest.mark.parametrize("backend", ["newton", "interaction_mesh"])
def test_executor_maps_export_failure_without_leaking_host_path(
    tmp_path: Path,
    backend: str,
) -> None:
    calls: dict[str, Any] = {}
    spec = _spec(backend=backend)
    context, _artifact_store, _progress = _context(tmp_path, spec)
    executor = H2RJobExecutor(
        _bindings(
            tmp_path,
            calls,
            fail_export=True,
            category=(
                AssetCategory.OBJECT_INTERACTION
                if backend == "interaction_mesh"
                else AssetCategory.PLAIN_MOTION
            ),
            dataset="omomo" if backend == "interaction_mesh" else "amass",
            has_scene=backend == "interaction_mesh",
        ),
        temporary_root=tmp_path / "temporary",
    )

    with pytest.raises(JobExecutionError) as raised:
        executor(spec, context)

    assert raised.value.error.code == "OUTPUT_WRITE_FAILED"
    assert raised.value.error.stage.value == "artifact"
    assert raised.value.error.retryable is True
    assert raised.value.error.details == {
        "operation": "export",
        "exception_type": "OSError",
    }
    assert str(tmp_path) not in raised.value.error.model_dump_json()
    assert [item.kind for item in context.published_artifacts()] == ["preview"]
    assert calls["released_models"] == [calls["preview_args"][0]]


def test_executor_rejects_non_preflight_parameter_shape_before_runtime_calls(
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}
    spec = _spec(foot_clamp_anti_penetration="yes")
    context, _artifact_store, _progress = _context(tmp_path, spec)
    executor = H2RJobExecutor(
        _bindings(tmp_path, calls),
        temporary_root=tmp_path / "temporary",
    )

    with pytest.raises(JobExecutionError) as raised:
        executor(spec, context)

    assert raised.value.error.code == "INVALID_PARAMETER"
    assert calls == {}


def test_executor_maps_cuda_oom_to_stable_retryable_error(tmp_path: Path) -> None:
    calls: dict[str, Any] = {}
    spec = _spec()
    context, _artifact_store, _progress = _context(tmp_path, spec)
    bindings = _bindings(tmp_path, calls)

    def oom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("CUDA out of memory while allocating")

    executor = H2RJobExecutor(
        replace(bindings, run_retarget=oom),
        temporary_root=tmp_path / "temporary",
    )

    with pytest.raises(JobExecutionError) as raised:
        executor(spec, context)

    assert raised.value.error.code == "CUDA_OUT_OF_MEMORY"
    assert raised.value.error.retryable is True
    assert raised.value.error.details == {"operation": "solve"}


def test_executor_uses_manifest_stem_instead_of_untrusted_motion_name(
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}
    spec = _spec()
    context, _artifact_store, _progress = _context(tmp_path, spec)
    executor = H2RJobExecutor(
        _bindings(tmp_path, calls, motion_name="../../escape"),
        temporary_root=tmp_path / "temporary",
    )

    result = executor(spec, context)

    assert result.summary["stem"] == "walk"
    assert calls["export"]["stem"] == "walk"


def test_executor_rejects_exporter_path_escape(tmp_path: Path) -> None:
    calls: dict[str, Any] = {}
    spec = _spec()
    context, _artifact_store, _progress = _context(tmp_path, spec)
    executor = H2RJobExecutor(
        _bindings(tmp_path, calls, outside_export=True),
        temporary_root=tmp_path / "temporary",
    )

    with pytest.raises(JobExecutionError) as raised:
        executor(spec, context)

    assert raised.value.error.code == "OUTPUT_PATH_ESCAPE"
    assert raised.value.error.stage is ErrorStage.ARTIFACT
    assert (tmp_path / "outside.csv").read_bytes() == b"outside"
    assert [item.kind for item in context.published_artifacts()] == ["preview"]


def test_executor_rejects_output_policy_without_a_managed_alias(tmp_path: Path) -> None:
    calls: dict[str, Any] = {}
    spec = _spec().model_copy(update={"output_policy": OutputPolicy.OVERWRITE})
    context, _artifact_store, _progress = _context(tmp_path, spec)
    executor = H2RJobExecutor(
        _bindings(tmp_path, calls),
        temporary_root=tmp_path / "temporary",
    )

    with pytest.raises(JobExecutionError) as raised:
        executor(spec, context)

    assert raised.value.error.code == "UNSUPPORTED_OUTPUT_POLICY"
    assert raised.value.error.stage is ErrorStage.PREFLIGHT
    assert calls == {}


def test_executor_rejects_nonportable_object_pkl_before_loading(tmp_path: Path) -> None:
    calls: dict[str, Any] = {}
    spec = _spec(backend="interaction_mesh", output_format="pkl")
    context, _artifact_store, _progress = _context(tmp_path, spec)
    executor = H2RJobExecutor(
        _bindings(
            tmp_path,
            calls,
            category=AssetCategory.OBJECT_INTERACTION,
            dataset="omomo",
        ),
        temporary_root=tmp_path / "temporary",
    )

    with pytest.raises(JobExecutionError) as raised:
        executor(spec, context)

    assert raised.value.error.code == "UNSUPPORTED_PORTABLE_EXPORT"
    assert "load" not in calls
    assert "robot_spec" not in calls


def test_executor_preserves_structured_plan_drift_error(tmp_path: Path) -> None:
    calls: dict[str, Any] = {}
    spec = _spec()
    context, _artifact_store, _progress = _context(tmp_path, spec)
    bindings = _bindings(tmp_path, calls)
    expected = ApiError(
        code="PLAN_STALE",
        message="The plan changed.",
        stage=ErrorStage.PREFLIGHT,
        details={"plan_id": PLAN_ID},
    )

    def stale(_spec_value: JobSpecV2) -> None:
        raise RetargetServiceError(expected)

    executor = H2RJobExecutor(
        replace(bindings, validate_spec=stale),
        temporary_root=tmp_path / "temporary",
    )

    with pytest.raises(JobExecutionError) as raised:
        executor(spec, context)

    assert raised.value.error == expected
    assert calls == {}


def test_executor_revalidates_exact_spec_after_robot_materialization(
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}
    spec = _spec()
    context, _artifact_store, _progress = _context(tmp_path, spec)
    bindings = _bindings(tmp_path, calls)
    validations = 0

    def drift_on_second_check(_spec_value: JobSpecV2) -> None:
        nonlocal validations
        validations += 1
        if validations == 2:
            raise RetargetServiceError(
                ApiError(
                    code="PLAN_STALE",
                    message="A bundle changed while the job was preparing.",
                    stage=ErrorStage.PREFLIGHT,
                )
            )

    executor = H2RJobExecutor(
        replace(bindings, validate_spec=drift_on_second_check),
        temporary_root=tmp_path / "temporary",
    )

    with pytest.raises(JobExecutionError) as raised:
        executor(spec, context)

    assert raised.value.error.code == "PLAN_STALE"
    assert validations == 2
    assert "run" not in calls
    assert "preview_args" not in calls
    assert len(calls["released_models"]) == 1
