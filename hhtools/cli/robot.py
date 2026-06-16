"""``hhtools robot`` — list / inspect / export-schema for robot presets.

Subcommands:
* ``hhtools robot list`` — discover presets on disk, show URDF availability.
* ``hhtools robot info <name>`` — load a preset, show link/joint counts,
  base link, ``dof_order`` snapshot.
* ``hhtools robot schema <name> [--out PATH]`` — write the CSV column header
  for retargeted trajectories.
* ``hhtools robot add`` — wizard scaffold (placeholder until milestone M9).

All commands route through :mod:`hhtools.robot.registry` so the CLI and the
viewer see exactly the same preset list.
"""

from __future__ import annotations

from pathlib import Path

import shutil

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hhtools.robot import (
    header_columns,
    list_presets,
    load_robot,
    refresh,
    scaffold_yaml_file,
    write_empty_csv,
)

app = typer.Typer(no_args_is_help=True, help="List or inspect humanoid robot presets.")
_console = Console()


@app.command("list")
def list_robots() -> None:
    """List every robot preset discovered by the registry.

    Discovery roots (in order, first-wins):
    1. workspace ``configs/robots/``
    2. ``~/.config/hhtools/robots/``
    3. ``$HHTOOLS_ROBOT_PATH`` (colon-separated)
    """
    refresh()  # always re-scan; CLI is short-lived, caching buys nothing
    presets = list_presets()
    if not presets:
        _console.print(
            "[yellow]No robot presets found.  Drop a robot.yaml under "
            "configs/robots/<name>/ or see configs/robots/_template/.[/]"
        )
        raise typer.Exit()

    table = Table(title="registered robots")
    table.add_column("name")
    table.add_column("display")
    table.add_column("urdf")
    table.add_column("dof_order")
    table.add_column("ik_map")
    table.add_column("source")
    table.add_column("root dir")
    for p in presets:
        urdf_cell = (
            f"[green]{p.urdf_path.name}[/]" if p.has_urdf
            else f"[yellow]missing[/]  (expect at {p.urdf_path})"
        )
        dof_cell = (
            f"{len(p.dof_order)} declared" if p.dof_order
            else "[dim]unpinned[/]"
        )
        source_cell = (
            "[cyan]auto[/]" if p.meta.get("auto_generated") else "[green]hand[/]"
        )
        table.add_row(
            p.name,
            p.display_name,
            urdf_cell,
            dof_cell,
            str(len(p.ik_map)),
            source_cell,
            str(p.root_dir),
        )
    _console.print(table)
    autos = [p for p in presets if p.meta.get("auto_generated")]
    if autos:
        _console.print(
            f"[dim]{len(autos)} preset(s) auto-generated from URDF drops.  "
            f"Run `hhtools robot info <name>` to review the scaffolded ik_map.[/]"
        )


@app.command("info")
def info(
    name: str = typer.Argument(..., help="Preset name (see `hhtools robot list`)."),
    compile_mjcf: bool = typer.Option(
        True, "--mjcf/--no-mjcf",
        help="Also compile URDF→MJCF to validate the model.",
    ),
) -> None:
    """Load a preset and print topology + DOF summary."""
    refresh()
    try:
        model = load_robot(
            _get_preset(name), compile_mjcf=compile_mjcf,
        )
    except FileNotFoundError as err:
        _console.print(f"[red]{err}[/]")
        raise typer.Exit(code=1) from err
    except Exception as err:
        _console.print(f"[red]Failed to load {name}:[/] {err}")
        raise typer.Exit(code=1) from err

    preset = model.preset
    sections = [
        f"[bold]name[/]           {preset.name}",
        f"[bold]display_name[/]   {preset.display_name}",
        f"[bold]urdf[/]           {preset.urdf_path}",
        f"[bold]base_link[/]      {model.base_link}",
        f"[bold]links[/]          {len(model.links)}",
        f"[bold]joints[/]         {len(model.joints)} ({len(model.actuated_joints)} actuated)",
        f"[bold]up / forward[/]   {preset.up_axis}-up / +{preset.forward_axis}",
        f"[bold]ik_map[/]         {len(preset.ik_map)} entries",
        f"[bold]MJCF[/]           "
        + (
            f"compiled ({len(model.mjcf_xml)} chars)"
            if model.mjcf_xml else "[yellow]unavailable[/]"
        ),
    ]
    _console.print(Panel("\n".join(sections), title=f"robot: {name}", expand=False))

    # DOF table — this is the most commonly-asked-for bit, so make it legible.
    dof_table = Table(title=f"DOFs ({len(model.actuated_joints)})")
    dof_table.add_column("idx", justify="right")
    dof_table.add_column("joint")
    dof_table.add_column("type")
    dof_table.add_column("parent → child")
    dof_table.add_column("limits [rad]")
    for i, joint in enumerate(model.actuated_joints):
        limits = (
            f"[{joint.limit_lower:+.2f}, {joint.limit_upper:+.2f}]"
            if joint.limit_lower is not None and joint.limit_upper is not None
            else "[dim]unbounded[/]"
        )
        dof_table.add_row(
            str(i), joint.name, joint.joint_type,
            f"{joint.parent_link} → {joint.child_link}", limits,
        )
    _console.print(dof_table)

    if not preset.dof_order:
        _console.print(
            "[yellow]dof_order is unpinned in robot.yaml.[/]  "
            "Paste the above joint names under `dof_order:` to pin the CSV layout."
        )

    if preset.ik_map and preset.urdf_path is not None and preset.urdf_path.is_file():
        from hhtools.robot.kinematics import validate_ik_map

        ik_issues = validate_ik_map(preset.urdf_path, preset.ik_map)
        if ik_issues:
            _console.print(
                Panel(
                    "\n".join(f"• {issue.format()}" for issue in ik_issues),
                    title="ik_map warnings",
                    border_style="yellow",
                    expand=False,
                )
            )


@app.command("schema")
def schema(
    name: str = typer.Argument(..., help="Preset name."),
    out: Path | None = typer.Option(
        None, "--out", "-o",
        help="Output CSV path; if omitted the header is written to stdout.",
    ),
) -> None:
    """Export the CSV column header for retargeted trajectories."""
    refresh()
    try:
        model = load_robot(_get_preset(name), compile_mjcf=False)
    except FileNotFoundError as err:
        _console.print(f"[red]{err}[/]")
        raise typer.Exit(code=1) from err

    cols = header_columns(model)
    if out is None:
        print(",".join(cols))
        _console.print(
            f"[dim]{len(cols)} columns: "
            f"time, 7 root (xyz+xyzw), {len(cols) - 8} DOF[/]"
        )
        return
    write_empty_csv(model, out)
    _console.print(
        f"[green]Wrote {out}[/]  "
        f"[dim]({len(cols)} columns: time, 7 root, {len(cols) - 8} DOF)[/]"
    )


@app.command("add")
def add_robot(
    source: Path = typer.Argument(
        ...,
        help="Either a .urdf file or a directory containing *.urdf + optional meshes/.",
    ),
    dest_name: str | None = typer.Option(
        None, "--name",
        help="Preset directory name under configs/robots/ (default: source dir name).",
    ),
    configs_dir: Path = typer.Option(
        Path("configs/robots"), "--configs",
        help="Robots config root (relative paths resolve against the workspace).",
    ),
    copy_meshes: bool = typer.Option(
        True, "--copy-meshes/--link-meshes",
        help="Copy meshes into the preset directory (default) or symlink them.",
    ),
    force: bool = typer.Option(
        False, "--force/--no-force",
        help="Overwrite existing files at the destination.",
    ),
) -> None:
    """Ingest a user-supplied URDF (+ meshes) into ``configs/robots/<name>/``.

    The philosophy is that *as a user you only drop a URDF + mesh files*; this
    command handles everything else:

    1. Resolves ``source`` to one or more URDF files (single file or directory).
    2. Creates ``configs/robots/<name>/`` (``name`` defaults to the source
       directory's basename, or the URDF file stem for single-file inputs).
    3. Copies / symlinks meshes next to the URDF so the preset is
       self-contained.
    4. Runs :func:`hhtools.robot.scaffold.scaffold_yaml_file` on every URDF so
       ``robot.yaml`` (or ``robot.<stem>.yaml`` per variant) is populated with
       a best-effort ``dof_order`` + ``ik_map``.
    5. Prints a ``hhtools robot list`` / ``info`` hint so the user can verify.

    No user editing is required to make the preset loadable.  Tune
    ``ik_map`` / weights afterwards via the Mapping Editor (or by editing the
    yaml directly — we never auto-overwrite a hand-edited yaml).
    """
    src = source.resolve()
    if not src.exists():
        _console.print(f"[red]source {src} does not exist[/]")
        raise typer.Exit(code=1)

    # Determine URDF list + a candidate source mesh dir.
    if src.is_file() and src.suffix.lower() == ".urdf":
        urdfs = [src]
        default_name = src.stem
        src_root = src.parent
    elif src.is_dir():
        urdfs = sorted(src.glob("*.urdf"))
        default_name = src.name
        src_root = src
    else:
        _console.print(
            f"[red]{src} is neither a .urdf file nor a directory containing URDFs[/]"
        )
        raise typer.Exit(code=1)

    if not urdfs:
        _console.print(f"[red]no .urdf files found at {src}[/]")
        raise typer.Exit(code=1)

    name = dest_name or default_name
    dest = (configs_dir / name).resolve()
    if dest.exists() and not force and any(dest.iterdir()):
        _console.print(
            f"[red]destination {dest} exists and is non-empty.  "
            f"Re-run with --force to overwrite.[/]"
        )
        raise typer.Exit(code=1)
    dest.mkdir(parents=True, exist_ok=True)

    # Copy URDFs (normalise mesh paths when vendor URDFs double meshdir).
    from hhtools.robot.urdf_normalize import ensure_urdf_meshes_resolvable

    for urdf in urdfs:
        target = dest / urdf.name
        if target.exists() and force:
            target.unlink()
        shutil.copy2(urdf, target)
        _console.print(f"[green]copied[/] {urdf.name} -> {target}")

    # Copy / symlink meshes (one dir is enough — we look for the conventional
    # ``meshes/`` sibling of the URDFs).
    src_mesh_dirs = [src_root / "meshes"]
    # Also try the URDF's parent/../meshes for URDFs nested in urdf/ subdirs.
    for urdf in urdfs:
        src_mesh_dirs.append(urdf.parent / "meshes")
        src_mesh_dirs.append(urdf.parent.parent / "meshes")
    seen: set[Path] = set()
    for mesh_dir in src_mesh_dirs:
        try:
            rp = mesh_dir.resolve()
        except OSError:
            continue
        if rp in seen or not rp.is_dir():
            continue
        seen.add(rp)
        target = dest / "meshes"
        if target.exists() and force:
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
        if target.exists():
            _console.print(f"[yellow]meshes/ already present, skipping {rp}[/]")
            continue
        if copy_meshes:
            shutil.copytree(rp, target)
            _console.print(f"[green]copied mesh dir[/] {rp} -> {target}")
        else:
            target.symlink_to(rp, target_is_directory=True)
            _console.print(f"[green]symlinked mesh dir[/] {rp} -> {target}")
        break  # one mesh dir is enough

    for urdf in urdfs:
        target = dest / urdf.name
        try:
            ensure_urdf_meshes_resolvable(
                target,
                search_dirs=[dest / "meshes", dest],
                output_path=target,
            )
            _console.print(f"[green]mesh paths repaired[/] {urdf.name}")
        except ValueError as err:
            _console.print(
                f"[red]mesh repair failed[/] {urdf.name}: {err}"
            )
            raise typer.Exit(code=1) from err

    # Scaffold yamls (one per URDF).
    for urdf in urdfs:
        dest_urdf = dest / urdf.name
        result = scaffold_yaml_file(dest_urdf, overwrite=force)
        if result.created:
            _console.print(
                f"[green]scaffolded[/] {result.yaml_path.name}  "
                f"[dim](ik_map: {len(result.preset.ik_map)}/17, "
                f"dof_order: {len(result.preset.dof_order)})[/]"
            )
        else:
            _console.print(
                f"[yellow]skipped[/] {result.yaml_path.name}  [dim]({result.actions[0]})[/]"
            )

    _console.print(
        f"\n[bold]Preset ready:[/] {dest}\n"
        f"  hhtools robot list                 # verify registration\n"
        f"  hhtools robot info {name!r}        # review ik_map + DOF\n"
        f"  hhtools robot schema {name!r} -o out.csv   # CSV header"
    )
    refresh()


@app.command("validate")
def validate_robot(
    name: str = typer.Argument(..., help="Preset name (see `hhtools robot list`)."),
    fix: bool = typer.Option(
        False, "--fix",
        help="Rewrite ik_map slots using topology inference (preserves other yaml keys).",
    ),
) -> None:
    """Check ``ik_map`` anatomy against the URDF kinematic tree."""
    refresh()
    preset = _get_preset(name)
    if preset.urdf_path is None or not preset.urdf_path.is_file():
        _console.print(f"[red]robot {name!r} has no URDF on disk[/]")
        raise typer.Exit(code=1)

    from hhtools.robot.kinematics import (
        CRITICAL_IK_SLOTS,
        infer_ik_map_from_kinematics,
        infer_smooth_joint_filter_masks,
        prepare_ik_map,
        validate_ik_map,
    )

    issues = validate_ik_map(preset.urdf_path, preset.ik_map)
    duplicate_trunk = [
        i for i in issues
        if "is shared with" in i.message
    ]
    critical = [
        i for i in issues
        if i in duplicate_trunk
        or i.slot in CRITICAL_IK_SLOTS
        or i.slot.endswith("_knee")
        or i.slot.endswith("_wrist")
        or i.slot == "head"
    ]
    optional = [i for i in issues if i not in critical]
    if not critical:
        if optional:
            _console.print(
                f"[yellow]~[/] {name}: optional ik_map slots need review "
                f"({len(optional)} warning(s))"
            )
            for issue in optional:
                _console.print(f"  [dim]• {issue.format()}[/]")
        else:
            _console.print(f"[green]✓[/] {name}: ik_map passes anatomical checks")
        raise typer.Exit(0)

    issues = critical

    kinematic = infer_ik_map_from_kinematics(preset.urdf_path)
    _console.print(
        Panel(
            "\n".join(f"• {issue.format()}" for issue in issues),
            title=f"{name}: ik_map issues",
            border_style="red",
            expand=False,
        )
    )
    hints = [
        f"  {slot}: {link!r}"
        for slot, link in kinematic.items()
        if any(issue.slot == slot for issue in issues)
    ]
    if hints:
        _console.print(
            "[dim]Topology-inferred suggestions:[/]\n" + "\n".join(hints)
        )

    if not fix:
        _console.print(
            "[yellow]Re-run with --fix to patch robot.yaml, then recalibrate "
            "in the viewer.[/]"
        )
        raise typer.Exit(code=1)

    repaired, changes = prepare_ik_map(preset.urdf_path, dict(preset.ik_map))
    remaining = validate_ik_map(preset.urdf_path, repaired)
    remaining_critical = [
        i for i in remaining
        if i.slot in CRITICAL_IK_SLOTS
        or i.slot.endswith("_knee")
        or i.slot.endswith("_wrist")
        or i.slot == "head"
    ]
    yaml_path = preset.meta.get("yaml_path")
    if not yaml_path:
        _console.print("[red]Cannot --fix: preset has no yaml_path in metadata[/]")
        raise typer.Exit(code=1)

    from hhtools.robot.yaml_io import (
        update_robot_yaml_ik_map,
        update_robot_yaml_smooth_joint_filter_masks,
    )

    update_robot_yaml_ik_map(yaml_path, repaired)
    smooth_masks = infer_smooth_joint_filter_masks(preset.urdf_path, repaired)
    if smooth_masks:
        update_robot_yaml_smooth_joint_filter_masks(yaml_path, smooth_masks)
        _console.print(
            f"[green]Updated smooth_joint_filter_masks[/] "
            f"({len(smooth_masks)} link(s))"
        )
    refresh()
    _console.print("[green]Updated ik_map[/] in " + str(yaml_path))
    for line in changes:
        _console.print(f"  · {line}")
    if remaining_critical:
        _console.print(
            "[yellow]Some critical issues remain after auto-repair — edit ik_map manually:[/]"
        )
        for issue in remaining_critical:
            _console.print(f"  • {issue.format()}")
        raise typer.Exit(code=1)
    _console.print(
        "[green]✓[/] ik_map repaired. Open the Robot tab and recalibrate before retargeting."
    )


@app.command("scaffold")
def scaffold(
    name: str = typer.Argument(..., help="Preset directory name under configs/robots/."),
    force: bool = typer.Option(
        False, "--force/--no-force",
        help="Re-generate yamls even if they already exist (loses user edits!).",
    ),
    configs_dir: Path = typer.Option(
        Path("configs/robots"), "--configs",
        help="Robots config root (relative paths resolve against the workspace).",
    ),
) -> None:
    """Regenerate ``robot.yaml`` / ``robot.<stem>.yaml`` from URDFs in a preset dir.

    Normal flow: users drop a URDF + mesh and the registry auto-scaffolds yaml
    the next time it scans.  This command is the explicit escape hatch for:

    * Forcing a re-scaffold after you edited the URDF (``--force``);
    * Recovering from accidentally deleting a yaml;
    * Previewing what the scaffold *would* write without touching the file
      (run without ``--force`` on a directory that already has yamls — the
      output lists ``skipped: yaml already exists`` for each).
    """
    target = (configs_dir / name).resolve()
    if not target.is_dir():
        _console.print(f"[red]{target} is not a directory[/]")
        raise typer.Exit(code=1)
    urdfs = sorted(target.glob("*.urdf"))
    if not urdfs:
        _console.print(f"[red]no .urdf files in {target}[/]")
        raise typer.Exit(code=1)
    for urdf in urdfs:
        result = scaffold_yaml_file(urdf, overwrite=force)
        if result.created:
            _console.print(
                f"[green]wrote[/] {result.yaml_path.name}  "
                f"[dim](preset: {result.preset.name}, "
                f"ik_map: {len(result.preset.ik_map)}/17, "
                f"dof_order: {len(result.preset.dof_order)})[/]"
            )
        else:
            _console.print(
                f"[yellow]skipped[/] {result.yaml_path.name}  [dim]({result.actions[0]})[/]"
            )
    refresh()


# --------------------------------------------------------------------------- internals


def _get_preset(name: str):
    from hhtools.robot.registry import get

    try:
        return get(name)
    except KeyError as err:
        _console.print(f"[red]{err}[/]")
        raise typer.Exit(code=1) from err


__all__ = ["app"]
