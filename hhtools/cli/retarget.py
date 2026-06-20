"""``hhtools retarget`` — retarget NPZ human motion to a humanoid robot."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(no_args_is_help=True, help="Retarget an NPZ motion to a humanoid robot.")

_log = logging.getLogger(__name__)


def _load_input_motion(path: Path):
    """Load a :class:`~hhtools.core.motion.Motion` from ``path``.

    Tries the internal hhtools NPZ format first (``load_motion`` dispatch),
    then falls back to :class:`~hhtools.io.datasets.amass.AmassAdapter` for
    AMASS-style ``*_poses.npz`` / ``*_stageii.npz`` files.  Other external
    formats can be added here as we grow adapter coverage.
    """
    from hhtools.io import load_motion

    try:
        return load_motion(path)
    except Exception as internal_err:  # noqa: BLE001 — fall through to adapters
        _log.debug("load_motion failed for %s: %s; trying AMASS adapter", path, internal_err)

    if path.suffix.lower() == ".npz":
        from hhtools.io.datasets.amass import AmassAdapter

        try:
            adapter = AmassAdapter(root=path.parent)
            return adapter.load_motion(path.name)
        except Exception as amass_err:  # noqa: BLE001 — surface both paths
            raise typer.BadParameter(
                f"failed to load {path}: not an hhtools NPZ, and AMASS "
                f"adapter raised: {amass_err}"
            ) from amass_err

    raise typer.BadParameter(
        f"unsupported input format for {path}; expected hhtools NPZ or "
        f"AMASS *_poses.npz / *_stageii.npz"
    )


def _load_motion_any(path: Path):
    """Load :class:`~hhtools.core.motion.Motion` from NPZ/BVH/OMOMO ``.pkl`` / meshmimic ``.npy``."""
    from hhtools.io import load_motion

    suf = path.suffix.lower()
    if suf in (".npz", ".bvh", ".csv"):
        return load_motion(path)
    if suf == ".pkl":
        from hhtools.io.datasets.omomo import OmomoAdapter

        return OmomoAdapter(root=path.parent).load_motion(path.name)
    if suf == ".npy":
        from hhtools.io.datasets.meshmimic_holosoma import MeshmimicHolosomaAdapter

        parent = path.parent
        if parent.name != path.stem:
            raise typer.BadParameter(
                f"meshmimic holosoma clip must be clip/<clip>.npy; got {path}"
            )
        holosoma_root = parent.parent
        seq = f"{parent.name}/{path.name}"
        return MeshmimicHolosomaAdapter(root=holosoma_root).load_motion(seq)
    raise typer.BadParameter(
        f"unsupported extension for interaction-mesh loader: {path.suffix}"
    )


def _expand_inputs(inputs: list[Path]) -> list[Path]:
    """Expand directories into their NPZ children (non-recursive)."""
    out: list[Path] = []
    for raw in inputs:
        if raw.is_dir():
            npzs = sorted(raw.glob("*.npz"))
            if not npzs:
                _log.warning("no *.npz in directory %s", raw)
            out.extend(npzs)
        else:
            out.append(raw)
    return out


@app.command("run")
def retarget(
    inputs: list[Path] = typer.Argument(..., help="NPZ files or directories to retarget."),
    robot: str = typer.Option(..., "--robot", help="Registered robot name (e.g. unitree_g1__g1_29dof)."),
    output: Path = typer.Option(
        ..., "--output", "-o",
        help="Output directory or single .csv path (when a single input is given).",
    ),
    ik_iterations: int = typer.Option(24, "--ik-iterations", help="Newton IK LM iterations per frame."),
    human_height: float = typer.Option(
        1.7, "--human-height", help="Subject height in metres (drives the scaler's ratio correction).",
    ),
    joint_limit_weight: float = typer.Option(
        10.0, "--joint-limit-weight",
        help="Hard IKObjectiveJointLimit weight; 0 disables.",
    ),
    smooth_joint_filter_weight: float = typer.Option(
        0.0, "--smooth-joint-filter-weight",
        help="Soft pull-to-midpoint joint-limit weight; 0 disables (upstream default).",
    ),
    max_joint_velocity: float = typer.Option(
        0.0, "--max-joint-velocity",
        help="Rate-limit non-root DOFs (rad/s); 0 disables.",
    ),
    apply_feet_stabilizer: bool = typer.Option(
        False, "--feet-stabilizer/--no-feet-stabilizer",
        help="Enable the foot-plant/ground-contact constraints before IK.",
    ),
    limit_frames: int | None = typer.Option(
        None, "--limit-frames",
        help="Optional per-clip frame cap — useful for smoke-testing.",
    ),
    calibration_reference: str = typer.Option(
        "smpl",
        "--calibration-reference",
        help=(
            "Which saved calibration to load (matches the viewer's Reference "
            "pose): smpl, smplx, soma_bvh, lafan_bvh, xsens_mocap, glb. "
            "Looks for retarget_calibration_<ref>.yaml beside the URDF, then "
            "legacy retarget_calibration.yaml if its embedded reference matches."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Retarget one or more NPZ motion clips to ``robot`` and save CSVs."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Configure Warp's kernel cache *before* anything that imports warp, so
    # the pipeline's first kernel compile lands in a writable directory.
    from hhtools.retarget.newton_basic._warp_config import configure as configure_warp_cache

    configure_warp_cache()

    from hhtools.core.motion import Motion
    from hhtools.io.robot_csv import save_robot_csv
    from hhtools.retarget.calibration import (
        load_calibration,
        resolve_calibration_file,
    )
    from hhtools.robot.retarget_profile import (
        build_feet_stabilizer_config,
        build_pipeline_config_for_preset,
        build_scaler_config_for_robot,
    )
    from hhtools.retarget.newton_basic import (
        NewtonBasicPipeline,
        PipelineConfig,
    )
    from hhtools.robot.loader import load_robot
    from hhtools.robot.registry import get as get_preset

    files = _expand_inputs(inputs)
    if not files:
        typer.echo("No NPZ inputs to process.", err=True)
        raise typer.Exit(code=2)

    # Robot loading.
    try:
        preset = get_preset(robot)
    except KeyError as err:
        raise typer.BadParameter(str(err)) from err
    robot_model = load_robot(preset)

    # Require a retarget calibration yaml next to the URDF (per reference
    # format, or legacy single file when its embedded reference matches).
    if preset.urdf_path is None:
        raise typer.BadParameter(
            f"robot preset {robot!r} has no URDF on disk; calibration "
            "cannot be resolved."
        )
    preset_dir = preset.urdf_path.parent
    cal_path = resolve_calibration_file(preset_dir, calibration_reference)
    if cal_path is None:
        raise typer.BadParameter(
            f"no retarget calibration for robot {robot!r} with reference "
            f"{calibration_reference!r} under {preset_dir}.\n"
            "Expected e.g. "
            f"`retarget_calibration_{calibration_reference}.yaml`, or a "
            "legacy `retarget_calibration.yaml` whose `reference` field "
            "matches after normalisation.  Calibrate in the viewer (Robot "
            "tab) or pass a different --calibration-reference."
        )
    calibration = load_calibration(cal_path)

    output_is_file = (
        output.suffix.lower() == ".csv"
        or (len(files) == 1 and not output.is_dir())
    )
    if output_is_file and len(files) != 1:
        raise typer.BadParameter(
            "--output is a .csv path but multiple inputs were given; "
            "pass a directory instead."
        )
    if not output_is_file:
        output.mkdir(parents=True, exist_ok=True)

    from dataclasses import replace

    pipeline_cfg = build_pipeline_config_for_preset(
        preset,
        calibration_reference,
        ik_iterations=ik_iterations,
    )
    pipeline_cfg = replace(
        pipeline_cfg,
        joint_limit_weight=joint_limit_weight,
        smooth_joint_filter_weight=smooth_joint_filter_weight,
        max_joint_velocity=(
            max_joint_velocity
            if max_joint_velocity > 0.0
            else pipeline_cfg.max_joint_velocity
        ),
        apply_feet_stabilizer=(
            apply_feet_stabilizer or pipeline_cfg.apply_feet_stabilizer
        ),
    )
    feet_stabilizer_config = None
    if pipeline_cfg.apply_feet_stabilizer:
        feet_stabilizer_config = build_feet_stabilizer_config(
            preset,
            calibration_reference,
            model=robot_model,
        )

    pipeline: NewtonBasicPipeline | None = None
    errors: list[tuple[Path, str]] = []
    written: list[Path] = []

    for src in files:
        typer.echo(f"[retarget] {src}")
        try:
            motion: Motion = _load_input_motion(src)
        except Exception as err:  # noqa: BLE001
            errors.append((src, f"load: {err}"))
            continue
        if limit_frames is not None and motion.num_frames > limit_frames:
            motion.positions = motion.positions[:limit_frames]
            motion.quaternions = motion.quaternions[:limit_frames]

        # Build the scaler config from the robot's saved calibration +
        # this motion's frame-0 pose (the latter drives the yaw-
        # preserving root offset).  All per-limb scales are derived
        # from the URDF's FK at the calibrated joint configuration;
        # see :mod:`hhtools.retarget.calibration`.
        try:
            scaler_cfg = build_scaler_config_for_robot(
                calibration, robot_model, motion, human_height=human_height,
            )
        except Exception as err:  # noqa: BLE001
            errors.append((src, f"calibration: {err}"))
            continue

        # One pipeline per run; we don't reset between clips because the
        # Newton model / IK solver only cares about the robot, not the
        # source hierarchy.  The scaler is re-created per motion internally.
        if pipeline is None:
            pipeline = NewtonBasicPipeline(
                robot_model,
                scaler_config=scaler_cfg,
                feet_stabilizer_config=feet_stabilizer_config,
                pipeline_config=pipeline_cfg,
                human_height=human_height,
                configure_warp=False,
            )
        else:
            pipeline.scaler_config = scaler_cfg

        try:
            retargeted = pipeline.run(motion)
        except Exception as err:  # noqa: BLE001
            errors.append((src, f"solve: {err}"))
            continue

        if output_is_file:
            out_path = output
        else:
            out_path = output / f"{motion.name or src.stem}.csv"

        save_robot_csv(
            out_path,
            robot=robot_model,
            joint_q=retargeted.joint_q,
            sample_rate=retargeted.sample_rate,
            meta={
                "source": str(src.resolve()),
                "source_format": motion.meta.get("source_format", ""),
                "ik_iterations": str(ik_iterations),
                **{k: str(v) for k, v in retargeted.meta.items()},
            },
        )
        typer.echo(f"  → {out_path}  ({retargeted.num_frames} frames)")
        written.append(out_path)

    if errors:
        typer.echo("", err=True)
        typer.echo(f"{len(errors)} file(s) failed:", err=True)
        for path, reason in errors:
            typer.echo(f"  {path}: {reason}", err=True)

    if not written:
        raise typer.Exit(code=1)

    typer.echo(f"Done. {len(written)} CSV(s) written to {output}.", file=sys.stderr)


_im = typer.Typer(
    no_args_is_help=True,
    help="Interaction-mesh (Laplacian) backend — same calibration/scaler as ``run``.",
)


@_im.command("precompute-laplacian")
def interaction_mesh_precompute_laplacian(
    input_path: Path = typer.Argument(..., help="NPZ, OMOMO .pkl, or meshmimic holosoma clip .npy"),
    robot: str = typer.Option(..., "--robot", help="Registered robot preset name."),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output .npz with stacked target Laplacians.",
    ),
    human_height: float = typer.Option(1.7, "--human-height"),
    calibration_reference: str = typer.Option("smpl", "--calibration-reference"),
    limit_frames: int | None = typer.Option(
        None, "--limit-frames", help="Optional frame cap for smoke tests.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Scale scene (mimic-style) and precompute per-frame Laplacian targets."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from pathlib import Path as _P

    from hhtools.retarget.calibration import resolve_calibration_file
    from hhtools.retarget.interaction_mesh.pipeline import InteractionMeshPipeline
    from hhtools.robot.loader import load_robot
    from hhtools.robot.registry import get as get_preset

    src = input_path.resolve()
    try:
        preset = get_preset(robot)
    except KeyError as err:
        raise typer.BadParameter(str(err)) from err
    robot_model = load_robot(preset)
    if preset.urdf_path is None:
        raise typer.BadParameter(f"preset {robot!r} has no URDF")
    cal_path = resolve_calibration_file(preset.urdf_path.parent, calibration_reference)
    if cal_path is None:
        raise typer.BadParameter(f"no calibration for {robot!r} ref={calibration_reference!r}")

    motion = _load_motion_any(src)
    if limit_frames is not None and motion.num_frames > limit_frames:
        motion.positions = motion.positions[:limit_frames]
        motion.quaternions = motion.quaternions[:limit_frames]
        if motion.objects:
            for o in motion.objects:
                o.positions = o.positions[:limit_frames]
                o.quaternions = o.quaternions[:limit_frames]

    pipe = InteractionMeshPipeline.from_calibration(
        robot_model,
        motion,
        str(cal_path),
        human_height=human_height,
    )
    targets, _robot_links, _z_min, _smpl_scale, _robot_points = pipe.precompute_laplacian_targets(motion)
    stacked = np.stack([t.target_laplacian for t in targets], axis=0)
    output = _P(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        target_laplacian=stacked.astype(np.float32),
        meta_source=np.array(str(src)),
        meta_robot=np.array(robot),
        meta_frames=np.array(len(targets), dtype=np.int32),
        meta_backend=np.array("interaction_mesh_precompute"),
    )
    typer.echo(f"Wrote {output} shape={stacked.shape}", file=sys.stderr)


@_im.command("run")
def interaction_mesh_run(
    inputs: list[Path] = typer.Argument(..., help="NPZ, OMOMO .pkl, or meshmimic clip .npy paths."),
    robot: str = typer.Option(..., "--robot", help="Registered robot preset name."),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory or single .csv path."),
    human_height: float = typer.Option(1.7, "--human-height"),
    calibration_reference: str = typer.Option("smpl", "--calibration-reference"),
    limit_frames: int | None = typer.Option(None, "--limit-frames"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Retarget with interaction-mesh (Laplacian SQP / RTI MPC) and write robot CSV."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from hhtools.io.robot_csv import save_robot_csv
    from hhtools.retarget.calibration import resolve_calibration_file
    from hhtools.retarget.interaction_mesh.pipeline import InteractionMeshPipeline
    from hhtools.robot.loader import load_robot
    from hhtools.robot.registry import get as get_preset

    files: list[Path] = []
    for raw in inputs:
        files.append(raw.resolve() if raw.is_file() else raw)
    if not files:
        raise typer.Exit(code=2)
    preset = get_preset(robot)
    robot_model = load_robot(preset)
    if preset.urdf_path is None:
        raise typer.BadParameter(f"preset {robot!r} has no URDF")
    cal_path = resolve_calibration_file(preset.urdf_path.parent, calibration_reference)
    if cal_path is None:
        raise typer.BadParameter(f"no calibration for {robot!r} ref={calibration_reference!r}")

    output_is_file = output.suffix.lower() == ".csv" or (len(files) == 1 and not output.is_dir())
    if output_is_file and len(files) != 1:
        raise typer.BadParameter("single .csv output requires exactly one input")
    if not output_is_file:
        output.mkdir(parents=True, exist_ok=True)

    for src in files:
        motion = _load_motion_any(src)
        if limit_frames is not None and motion.num_frames > limit_frames:
            motion.positions = motion.positions[:limit_frames]
            motion.quaternions = motion.quaternions[:limit_frames]
            if motion.objects:
                for o in motion.objects:
                    o.positions = o.positions[:limit_frames]
                    o.quaternions = o.quaternions[:limit_frames]
        pipe = InteractionMeshPipeline.from_calibration(
            robot_model, motion, str(cal_path), human_height=human_height,
        )
        ret = pipe.run(motion)
        out_path = output if output_is_file else output / f"{motion.name or src.stem}.csv"
        save_robot_csv(
            out_path,
            robot=robot_model,
            joint_q=ret.joint_q,
            sample_rate=ret.sample_rate,
            meta={
                "source": str(src),
                "retarget_backend": "interaction_mesh",
                **{k: str(v) for k, v in ret.meta.items()},
            },
        )
        typer.echo(f"[interaction-mesh] → {out_path} ({ret.num_frames} frames)", file=sys.stderr)


app.add_typer(_im, name="interaction-mesh")


__all__ = ["app"]
