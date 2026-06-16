"""``hhtools convert`` — batch convert mocap files into the unified NPZ."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

from hhtools.core.resample import resample_motion
from hhtools.io import bvh, npz
from hhtools.io.base import load_motion

app = typer.Typer(no_args_is_help=True, help="Convert BVH / GLB to the unified NPZ.")
_console = Console()


@app.command("run")
def run_convert(
    inputs: list[Path] = typer.Argument(..., help="Files or directories to convert."),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory for NPZ files."),
    unit: str = typer.Option("cm", "--unit", help="Source unit for BVH files (cm, mm, m, ...)."),
    target_up_axis: str = typer.Option(
        "Z", "--up", case_sensitive=False, help="Internal up-axis (Z or Y)."
    ),
    target_fps: float | None = typer.Option(
        None, "--fps", help="Optional target framerate (resamples with SLERP)."
    ),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing NPZ files."),
) -> None:
    """Convert one or more motion files into the unified NPZ schema."""
    files = _collect_input_files(inputs, recursive=recursive)
    if not files:
        _console.print("[yellow]No supported motion files found.[/]")
        raise typer.Exit(code=1)

    out.mkdir(parents=True, exist_ok=True)
    _console.print(f"Converting [bold]{len(files)}[/] file(s) into [bold]{out}[/]")

    ok: list[Path] = []
    errors: list[tuple[Path, str]] = []

    with Progress(console=_console) as progress:
        task: TaskID = progress.add_task("convert", total=len(files))
        for src in files:
            try:
                motion = _load_one(src, unit=unit, target_up_axis=target_up_axis)
                if target_fps is not None and abs(target_fps - motion.framerate) > 1e-4:
                    motion = resample_motion(motion, target_fps)
                dst = out / (src.stem + ".npz")
                if dst.exists() and not overwrite:
                    errors.append((src, f"destination exists ({dst}); use --overwrite"))
                else:
                    npz.save_npz(motion, dst)
                    ok.append(dst)
            except Exception as exc:  # pragma: no cover - surfaced in CLI output
                errors.append((src, str(exc)))
            finally:
                progress.advance(task)

    table = Table(title="convert summary", show_lines=False)
    table.add_column("status", style="bold")
    table.add_column("path")
    for p in ok:
        table.add_row("[green]ok[/]", str(p))
    for src, msg in errors:
        table.add_row("[red]fail[/]", f"{src}  —  {msg}")
    _console.print(table)

    if errors and not ok:
        raise typer.Exit(code=2)


def _collect_input_files(inputs: list[Path], *, recursive: bool) -> list[Path]:
    extensions = {".bvh", ".glb", ".gltf", ".npz"}
    out: list[Path] = []
    for item in inputs:
        if item.is_file():
            if item.suffix.lower() in extensions:
                out.append(item)
        elif item.is_dir():
            iterator = item.rglob("*") if recursive else item.glob("*")
            for child in iterator:
                if child.is_file() and child.suffix.lower() in extensions:
                    out.append(child)
        else:
            _console.print(f"[yellow]skip: {item} does not exist[/]")
    return sorted(set(out))


def _load_one(src: Path, *, unit: str, target_up_axis: str):  # type: ignore[no-untyped-def]
    ext = src.suffix.lower()
    if ext == ".bvh":
        return bvh.load_bvh(src, unit=unit, target_up_axis=target_up_axis)
    return load_motion(src)


__all__ = ["app"]
