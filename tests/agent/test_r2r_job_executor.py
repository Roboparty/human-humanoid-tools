from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from hhtools.agent.r2r_job_executor import (
    PreparedR2RSource,
    R2RExecutorBindings,
    R2RJobExecutor,
    R2RPreview,
    ResolvedR2RTrajectory,
)
from hhtools.contracts import (
    JobOutcome,
    JobProgress,
    JobSpecCalibration,
    JobSpecInput,
    JobSpecKind,
    JobSpecProvenance,
    JobSpecRobot,
    JobSpecV2,
    OutputPolicy,
)
from hhtools.services.artifacts import ArtifactStore
from hhtools.services.jobs import JobCancelledError, JobExecutionContext, JobExecutionError

SHA_TRAJECTORY = "a" * 64
SHA_SOURCE = "b" * 64
SHA_TARGET = "c" * 64
SHA_CALIBRATION = "d" * 64
TRAJECTORY_ID = f"asset:sha256:{SHA_TRAJECTORY}"
SOURCE_ID = f"asset:sha256:{SHA_SOURCE}"
TARGET_ID = f"asset:sha256:{SHA_TARGET}"


@dataclass
class _Motion:
    framerate: float = 50.0


@dataclass
class _Retargeted:
    num_frames: int = 2
    sample_rate: float = 50.0
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {
                "execution_provenance": {
                    "device": "cpu",
                    "device_kind": "cpu",
                    "precision": "float32",
                    "runtime": "warp",
                    "solver": "newton",
                    "fallback_used": False,
                }
            }


def _spec() -> JobSpecV2:
    return JobSpecV2(
        kind=JobSpecKind.R2R_RETARGET,
        plan_id=f"plan:sha256:{'e' * 64}",
        inputs=[JobSpecInput(asset_id=TRAJECTORY_ID, sha256=SHA_TRAJECTORY)],
        source_robot=JobSpecRobot(
            robot_id="source_bot",
            asset_id=SOURCE_ID,
            config_sha256=SHA_SOURCE,
        ),
        robot=JobSpecRobot(
            robot_id="target_bot",
            asset_id=TARGET_ID,
            config_sha256=SHA_TARGET,
        ),
        calibration=JobSpecCalibration(
            calibration_id=f"cal:sha256:{SHA_CALIBRATION}",
            sha256=SHA_CALIBRATION,
        ),
        backend="newton",
        effective_parameters={
            "run_mode": "smoke",
            "limit_frames": 2,
            "ik_iterations": 24,
            "source_fps": None,
            "retarget_fps": None,
            "trajectory_profile": "mimic",
            "output_format": "csv",
        },
        output_policy=OutputPolicy.CREATE_NEW,
        provenance=JobSpecProvenance(
            hhtools_git_commit="test",
            hhtools_dirty=False,
            python="3.12",
        ),
        created_at=datetime(2026, 9, 8, tzinfo=UTC),
    )


def _context(
    tmp_path: Path,
    spec: JobSpecV2,
    *,
    cancellation: threading.Event | None = None,
):
    artifacts = ArtifactStore(tmp_path / "artifacts")
    progress: list[tuple[JobProgress, int | None]] = []
    context = JobExecutionContext(
        job_id="job:r2r-test",
        spec=spec,
        artifact_store=artifacts,
        cancellation_event=cancellation or threading.Event(),
        progress_callback=lambda item, delay: progress.append((item, delay)),
    )
    return context, artifacts, progress


def _bindings(
    tmp_path: Path,
    calls: dict[str, Any],
    *,
    has_scene: bool = False,
    fail_export: bool = False,
) -> R2RExecutorBindings:
    source_model = object()
    target_model = object()
    motion = _Motion()
    playback = object()
    retargeted = _Retargeted()
    resolved = ResolvedR2RTrajectory(
        asset_id=TRAJECTORY_ID,
        source_robot_id="source_bot",
        source_path=tmp_path / "walk.csv",
        stem="walk",
        profile="intermimic" if has_scene else "mimic",
        has_scene=has_scene,
    )
    resolved.source_path.write_text("trajectory", encoding="utf-8")

    def validate_spec(spec: JobSpecV2) -> None:
        calls.setdefault("validated", []).append(spec)

    def resolve_trajectory(asset_id: str) -> ResolvedR2RTrajectory:
        calls["resolved"] = asset_id
        return resolved

    def get_source_model(spec: JobSpecV2) -> object:
        calls["source_spec"] = spec
        return source_model

    def get_target_model(spec: JobSpecV2) -> object:
        calls["target_spec"] = spec
        return target_model

    def prepare_source(
        _resolved: ResolvedR2RTrajectory,
        model: object,
        *,
        source_fps: float | None,
        limit_frames: int | None,
        progress_callback,
    ) -> PreparedR2RSource:
        calls["prepare_source"] = (model, source_fps, limit_frames)
        progress_callback(1, 2)
        progress_callback(2, 2)
        return PreparedR2RSource(
            motion=motion,
            playback=playback,
            source_fps=50.0,
            num_frames=2,
        )

    def load_pair_calibration(source: object, target: object) -> dict[str, float]:
        calls["calibration_models"] = (source, target)
        return {"hip": 0.0}

    def prepare_motion(value: _Motion, fps: float | None):
        calls["prepare_motion"] = (value, fps)
        return value, 50.0

    def run_r2r(
        source: object,
        target: object,
        value: _Motion,
        calibration: dict[str, float],
        *,
        backend: str,
        ik_iterations: int,
        progress_callback,
    ) -> _Retargeted:
        calls["run"] = {
            "source": source,
            "target": target,
            "motion": value,
            "calibration": calibration,
            "backend": backend,
            "ik_iterations": ik_iterations,
        }
        progress_callback(1, 2)
        progress_callback(2, 2)
        return retargeted

    def build_preview(*args: Any) -> R2RPreview:
        calls["preview"] = args
        return R2RPreview(
            document={"source_trajectory": [], "target_trajectory": []},
            diagnostics={
                "available": True,
                "tracking": {
                    "mean_error_m": 0.01,
                    "p95_error_m": 0.02,
                    "max_error_m": 0.03,
                },
            },
            yellow_foot_z=0.04,
        )

    def write_export(*args: Any, **kwargs: Any) -> Path:
        calls["export"] = (args, kwargs)
        if fail_export:
            raise OSError("disk unavailable")
        output_root = args[6]
        output = output_root / "walk.csv"
        output.write_text("time,root_x\n0,0\n", encoding="utf-8")
        return output

    def release(model: object) -> None:
        calls.setdefault("released", []).append(model)

    return R2RExecutorBindings(
        validate_spec=validate_spec,
        resolve_trajectory=resolve_trajectory,
        get_source_model=get_source_model,
        get_target_model=get_target_model,
        prepare_source=prepare_source,
        load_pair_calibration=load_pair_calibration,
        prepare_motion=prepare_motion,
        run_r2r=run_r2r,
        build_preview=build_preview,
        write_export=write_export,
        release_robot_model=release,
    )


def test_r2r_executor_preserves_pair_identity_and_publishes_managed_artifacts(
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}
    spec = _spec()
    context, artifacts, progress = _context(tmp_path, spec)
    executor = R2RJobExecutor(
        _bindings(tmp_path, calls),
        temporary_root=tmp_path / "temporary",
    )

    result = executor(spec, context)

    assert result.outcome is JobOutcome.REVIEW_REQUIRED
    assert result.summary == {
        "workflow": "r2r",
        "run_mode": "smoke",
        "stem": "walk",
        "num_frames": 2,
        "trajectory_source_fps": 50.0,
        "retarget_fps": 50.0,
        "source_fps": 50.0,
        "output_format": "csv",
        "has_scene": False,
    }
    assert result.execution_provenance["source_robot_asset_id"] == SOURCE_ID
    assert result.execution_provenance["robot_asset_id"] == TARGET_ID
    assert result.execution_provenance["motion_asset_id"] == TRAJECTORY_ID
    assert result.execution_provenance["reference"] == "robot_source_bot"
    assert calls["validated"] == [spec, spec]
    assert calls["run"]["backend"] == "newton"
    assert calls["run"]["ik_iterations"] == 24
    assert calls["released"] == [calls["run"]["target"], calls["run"]["source"]]
    assert any(item.phase == "preparing" for item, _delay in progress)
    assert any(item.phase == "solving" for item, _delay in progress)

    published = context.published_artifacts()
    assert [item.kind for item in published] == ["preview", "retargeted_motion"]
    exported = artifacts.get(published[1].artifact_id, verify=True)
    assert exported.path.read_text(encoding="utf-8") == "time,root_x\n0,0\n"
    assert exported.descriptor.metadata["source_robot_asset_id"] == SOURCE_ID
    assert exported.descriptor.metadata["target_robot_asset_id"] == TARGET_ID


def test_r2r_executor_defensively_rejects_scene_trajectory(tmp_path: Path) -> None:
    calls: dict[str, Any] = {}
    spec = _spec()
    context, _artifacts, _progress = _context(tmp_path, spec)
    executor = R2RJobExecutor(
        _bindings(tmp_path, calls, has_scene=True),
        temporary_root=tmp_path / "temporary",
    )

    with pytest.raises(JobExecutionError) as captured:
        executor(spec, context)

    assert captured.value.error.code == "R2R_SCENE_UNSUPPORTED"
    assert "source_spec" not in calls
    assert context.published_artifacts() == ()


def test_r2r_executor_cancels_at_existing_fk_progress_boundary(tmp_path: Path) -> None:
    calls: dict[str, Any] = {}
    spec = _spec()
    cancellation = threading.Event()
    context, _artifacts, _progress = _context(
        tmp_path,
        spec,
        cancellation=cancellation,
    )
    bindings = _bindings(tmp_path, calls)
    original_prepare = bindings.prepare_source

    def cancel_then_prepare(*args: Any, **kwargs: Any) -> PreparedR2RSource:
        cancellation.set()
        return original_prepare(*args, **kwargs)

    executor = R2RJobExecutor(
        replace(bindings, prepare_source=cancel_then_prepare),
        temporary_root=tmp_path / "temporary",
    )

    with pytest.raises(JobCancelledError):
        executor(spec, context)

    assert "run" not in calls
    assert context.published_artifacts() == ()
    assert len(calls["released"]) == 2


def test_r2r_export_failure_is_structured_and_path_free(tmp_path: Path) -> None:
    calls: dict[str, Any] = {}
    spec = _spec()
    context, _artifacts, _progress = _context(tmp_path, spec)
    executor = R2RJobExecutor(
        _bindings(tmp_path, calls, fail_export=True),
        temporary_root=tmp_path / "temporary",
    )

    with pytest.raises(JobExecutionError) as captured:
        executor(spec, context)

    assert captured.value.error.code == "OUTPUT_WRITE_FAILED"
    assert captured.value.error.retryable is True
    assert str(tmp_path) not in captured.value.error.model_dump_json()
    assert [item.kind for item in context.published_artifacts()] == ["preview"]
