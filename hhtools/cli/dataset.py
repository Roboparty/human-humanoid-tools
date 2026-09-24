"""``hhtools import`` — import public datasets into the unified NPZ schema."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from hhtools.io.datasets import registered_datasets

app = typer.Typer(no_args_is_help=True, help="Import public SMPL-family motion datasets into NPZ.")
_console = Console()


@app.command("list")
def list_datasets() -> None:
    """List the available dataset adapters."""
    datasets = registered_datasets()
    table = Table(title="registered dataset adapters")
    table.add_column("name")
    table.add_column("requires")
    table.add_column("display name")
    if not datasets:
        _console.print(
            "[yellow]No dataset adapters have been implemented yet. The concrete adapters "
            "(AMASS, Motion-X, OMOMO, OmniContact, GRAB, PHUMA, KungFuAthlete, "
            "Humanoid-X, GVHMR, LAFAN, SOMA) "
            "will be added in milestone M6.[/]"
        )
        return
    for name, cls in datasets.items():
        table.add_row(name, cls.requires or "-", cls.display_name or "-")
    _console.print(table)


@app.command("run")
def run_import(
    dataset: str = typer.Option(..., "--dataset", help="Registered dataset name (see `list`)."),
    root: Path = typer.Option(..., "--root", help="Path to the dataset root on disk."),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory for unified NPZ files."),
    sequence: str | None = typer.Option(None, "--sequence", help="Process only one sequence."),
) -> None:
    """Import a registered dataset into the unified NPZ schema."""
    datasets = registered_datasets()
    if dataset not in datasets:
        _console.print(f"[red]Dataset '{dataset}' is not registered. Run `hhtools import list`.[/]")
        raise typer.Exit(code=2)

    adapter = datasets[dataset](root)
    out.mkdir(parents=True, exist_ok=True)
    seq_iter = [sequence] if sequence else list(adapter.list_sequences())
    _console.print(f"Importing {len(seq_iter)} sequence(s) from [bold]{dataset}[/]")

    from hhtools.io import npz

    ok_count = 0
    skip_count = 0
    fail_count = 0
    for sid in seq_iter:
        try:
            motion = adapter.load_motion(sid)
            dst = out / f"{motion.name}.npz"
            npz.save_npz(motion, dst)
            _console.print(f"  [green]ok[/] {dst}")
            ok_count += 1
        except NotImplementedError as exc:
            _console.print(f"  [yellow]skip[/] {sid}: {exc}")
            skip_count += 1
        except Exception as exc:
            _console.print(f"  [red]fail[/] {sid}: {type(exc).__name__}: {exc}")
            fail_count += 1

    _console.print(
        f"Import summary: [green]{ok_count} ok[/], "
        f"[yellow]{skip_count} skipped[/], [red]{fail_count} failed[/]"
    )
    if fail_count:
        raise typer.Exit(code=1)


__all__ = ["app"]
