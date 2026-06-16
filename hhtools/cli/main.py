"""``hhtools`` Typer application.

Sub-commands are registered lazily from ``sys.argv`` so ``hhtools web`` does not
import Viser / Newton / robot CLI modules (and their heavy deps) at startup.
"""

from __future__ import annotations

import importlib
import sys

import typer

from hhtools._version import __version__

app = typer.Typer(
    help="hhtools - Human-to-Humanoid Tools.",
    no_args_is_help=False,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

# (cli name, module path, help text)
_SUBCOMMANDS: list[tuple[str, str, str]] = [
    ("convert", "hhtools.cli.convert", "Convert BVH / GLB to the unified NPZ."),
    (
        "import",
        "hhtools.cli.dataset",
        "Import public SMPL-family datasets into NPZ (dataset flag).",
    ),
    (
        "bodymodel",
        "hhtools.cli.bodymodel",
        "Manage SMPL / SMPL-H / SMPL-X body model weights.",
    ),
    ("robot", "hhtools.cli.robot", "List or add humanoid robot presets."),
    ("retarget", "hhtools.cli.retarget", "Retarget an NPZ motion to a humanoid robot."),
    ("ui", "hhtools.cli.ui", "Launch the Viser-based web viewer (legacy)."),
    ("web", "hhtools.cli.web", "Launch the HTML / three.js web UI (recommended)."),
]


def _attach(name: str, module_path: str, help_text: str) -> None:
    module = importlib.import_module(module_path)
    app.add_typer(getattr(module, "app"), name=name, help=help_text)


def _subcommands_for_argv() -> list[tuple[str, str, str]]:
    """Load only the invoked subcommand (or all for top-level help)."""
    if len(sys.argv) < 2:
        return _SUBCOMMANDS
    arg = sys.argv[1]
    if arg.startswith("-"):
        return _SUBCOMMANDS
    for name, path, help_text in _SUBCOMMANDS:
        if arg == name:
            return [(name, path, help_text)]
    return _SUBCOMMANDS


for _name, _path, _help in _subcommands_for_argv():
    _attach(_name, _path, _help)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", help="Print the hhtools version and exit."
    ),
) -> None:
    if version:
        typer.echo(f"hhtools {__version__}")
        raise typer.Exit(code=0)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


if __name__ == "__main__":
    app()
