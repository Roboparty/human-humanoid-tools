"""Human-to-robot retarget runtime helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hhtools.application.progress import _set_retarget_job_clip_progress
from hhtools.application.robots import (
    _join_robot_prewarm,
    _require_newton_package,
)

if TYPE_CHECKING:
    from hhtools.application.state import SessionState


def _request_human_height(body: dict, preset, reference: str) -> float:
    """Resolve source-human height from the request, with a scaler-aware default."""
    from hhtools.robot.retarget_profile import default_human_height

    raw = body.get("human_height")
    if raw is not None:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = 0.0
        if val > 0.1:
            return val
    return default_human_height(preset, reference)


def _retarget_single(
    model,
    robot_name,
    motion,
    reference,
    backend,
    ik_iters,
    human_height,
    limit_frames,
    job,
    *,
    state: SessionState | None = None,
    foot_clamp_anti_penetration: bool | None = None,
    preset=None,
):
    """Run one clip through the requested backend, returning RetargetedMotion."""
    from hhtools.retarget.calibration import resolve_preset_calibration_file
    from hhtools.robot.retarget_profile import bundled_scaler_path

    if preset is None:
        from hhtools.robot.registry import get as get_preset

        preset = get_preset(robot_name)
    elif preset.name != robot_name:
        raise ValueError("the injected robot preset does not match the requested robot")
    cal_path = resolve_preset_calibration_file(preset, reference)
    if cal_path is None and bundled_scaler_path(preset, reference) is None:
        raise ValueError(
            f"robot {robot_name!r} not calibrated for reference {reference!r}; calibrate first"
        )

    if limit_frames:
        lf = int(limit_frames)
        if motion.num_frames > lf:
            motion.positions = motion.positions[:lf]
            motion.quaternions = motion.quaternions[:lf]
            for obj in motion.objects:
                obj.positions = obj.positions[:lf]
                obj.quaternions = obj.quaternions[:lf]

    if backend == "interaction_mesh":
        from hhtools.retarget.interaction_mesh.config import InteractionMeshPipelineConfig
        from hhtools.retarget.interaction_mesh.pipeline import InteractionMeshPipeline

        if job is not None:
            _set_retarget_job_clip_progress(
                job,
                0.04,
                "正在构建 Interaction-Mesh 场景（新机器人首次较慢）…",
            )
        im_cfg = InteractionMeshPipelineConfig()
        if foot_clamp_anti_penetration is not None:
            im_cfg.post_mpc_foot_clamps = bool(foot_clamp_anti_penetration)
            if not im_cfg.post_mpc_foot_clamps:
                im_cfg.min_foot_clearance_m = 0.0
        pipe = InteractionMeshPipeline.from_calibration(
            model,
            motion,
            str(cal_path),
            human_height=human_height,
            cfg=im_cfg,
        )

        def _im_cb(stage: str, cur: int, tot: int) -> None:
            if job is None:
                return
            if stage == "precompute":
                frac = 0.3 * (cur / max(1, tot))
                _set_retarget_job_clip_progress(job, frac, f"预处理 {cur}/{tot}")
            else:
                frac = 0.3 + 0.68 * (cur / max(1, tot))
                _set_retarget_job_clip_progress(job, frac, f"MPC 求解 {cur}/{tot}")

        try:
            try:
                return pipe.run(motion, progress_callback=_im_cb)
            except TypeError:
                return pipe.run(motion)
        except ModuleNotFoundError as err:
            if "osqp" in str(err).lower():
                raise ValueError(
                    "interaction-mesh retarget on terrain needs the OSQP solver. "
                    "Install it with `uv pip install osqp` (or re-run "
                    "`uv sync --extra web`)."
                ) from err
            raise

    if backend != "interaction_mesh":
        _require_newton_package()

    from hhtools.retarget.calibration import load_calibration
    from hhtools.retarget.newton_basic import NewtonBasicPipeline
    from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache
    from hhtools.robot.retarget_profile import (
        build_feet_stabilizer_config,
        build_pipeline_config_for_preset,
        resolve_retarget_scaler_config,
    )

    if job is not None:
        _set_retarget_job_clip_progress(job, 0.03, "正在加载标定与缩放参数…")
    if state is not None:
        _join_robot_prewarm(state, robot_name, job)

    configure_warp_cache()
    calibration = load_calibration(cal_path) if cal_path is not None else None
    scaler_cfg = resolve_retarget_scaler_config(
        preset,
        reference,
        calibration=calibration,
        model=model,
        motion=motion,
        human_height=human_height,
    )
    feet_cfg = build_feet_stabilizer_config(preset, reference, model=model)
    if job is not None:
        try:
            from hhtools.retarget.newton_basic.pipeline import is_newton_ik_prewarmed

            prewarmed = is_newton_ik_prewarmed(robot_name)
        except Exception:
            prewarmed = False
        _set_retarget_job_clip_progress(
            job,
            0.06,
            (
                "正在初始化 Newton IK…"
                if prewarmed
                else "正在初始化 Newton IK（首次会编译 GPU 内核，之后会复用缓存）…"
            ),
        )
    pipeline = NewtonBasicPipeline(
        model,
        scaler_config=scaler_cfg,
        pipeline_config=build_pipeline_config_for_preset(
            preset,
            reference,
            ik_iterations=ik_iters,
            foot_clamp_anti_penetration=foot_clamp_anti_penetration,
        ),
        feet_stabilizer_config=feet_cfg,
        human_height=human_height,
        configure_warp=False,
    )

    def _cb(done: int, total: int) -> None:
        if job is None:
            return
        if done <= 0:
            _set_retarget_job_clip_progress(
                job,
                0.08,
                "正在捕获 CUDA 图 / 准备逐帧 IK（首次较慢，请耐心等待）…",
            )
        else:
            _set_retarget_job_clip_progress(
                job,
                min(0.98, 0.1 + 0.88 * (done / max(1, total))),
                f"IK 求解 {done}/{total}",
            )

    try:
        return pipeline.run(motion, progress_callback=_cb)
    except TypeError:
        return pipeline.run(motion)


__all__ = ["_request_human_height", "_retarget_single"]
