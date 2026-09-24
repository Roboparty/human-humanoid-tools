"""``hhtools bodymodel`` — manage SMPL-family weights."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from hhtools.bodymodels.download import check_body_models, default_body_model_root, run_wizard

app = typer.Typer(no_args_is_help=True, help="Manage SMPL / SMPL-H / SMPL-X body model weights.")
_console = Console()


@app.command("check")
def bodymodel_check(
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Body-model root to inspect (defaults to the configured hhtools location).",
    ),
) -> None:
    """Check required SMPL-family weights as a shell readiness probe.

    Exits non-zero when any required model is missing.
    """
    target = Path(root) if root else default_body_model_root()
    status = check_body_models(target)
    _console.print(f"body model root: [bold]{target}[/]")
    for model, present in status.items():
        tag = "[green]OK[/]" if present else "[red]missing[/]"
        _console.print(f"  {tag}  {model}")
    if not status or not all(status.values()):
        raise typer.Exit(code=1)


@app.command("setup")
def bodymodel_setup(root: Path | None = typer.Option(None, "--root")) -> None:
    """Print the download instructions for SMPL / SMPL-H / SMPL-X weights."""
    run_wizard(Path(root) if root else None)


__all__ = ["app"]
