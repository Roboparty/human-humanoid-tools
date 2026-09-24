"""``hhtools`` Typer application.

Sub-commands are registered lazily from ``sys.argv`` so ``hhtools web`` does not
import Newton / robot CLI modules (and their heavy deps) at startup.
"""

from __future__ import annotations

import importlib
import sys

import typer

from hhtools._version import __version__
from hhtools.cli._stdio import configure_utf8_stdio

# Configure streams before importing a selected subcommand. Rich consoles created by
# those modules then inherit UTF-8 instead of a locale-dependent Windows code page.
configure_utf8_stdio()

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
    ("doctor", "hhtools.cli.doctor", "Check local runtime readiness without running jobs."),
    ("robot", "hhtools.cli.robot", "List or add humanoid robot presets."),
    ("retarget", "hhtools.cli.retarget", "Retarget an NPZ motion to a humanoid robot."),
    ("web", "hhtools.cli.web", "Launch the HTML / three.js web UI (recommended)."),
]


def _attach(name: str, module_path: str, help_text: str) -> None:
    module = importlib.import_module(module_path)
    app.add_typer(module.app, name=name, help=help_text)


def _subcommands_for_argv() -> list[tuple[str, str, str]]:
    """Load only the invoked subcommand, or all for explicit top-level help."""
    if len(sys.argv) < 2:
        # The landing page is deliberately cheap and does not need to import
        # every command tree merely to advertise stable entry points.
        return []
    arg = sys.argv[1]
    if arg == "agent":
        # The strict Agent command is registered directly below and lazily
        # imports its transport adapter.  Do not import unrelated solver/UI
        # command trees for a lightweight JSON request.
        return []
    if arg in {"--version", "-V"}:
        # Version reporting must stay instant and must not initialize optional
        # viewer, solver, or GPU-related command modules.
        return []
    if arg.startswith("-"):
        return _SUBCOMMANDS
    for name, path, help_text in _SUBCOMMANDS:
        if arg == name:
            return [(name, path, help_text)]
    return _SUBCOMMANDS


for _name, _path, _help in _subcommands_for_argv():
    _attach(_name, _path, _help)


@app.command(
    "agent",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def _agent(ctx: typer.Context) -> None:
    """Call the resident Agent REST service with strict JSON input/output."""

    # One passthrough Click command is deliberate: argparse inside the JSON
    # adapter converts *all* malformed or unknown tails into ApiError stdout,
    # instead of allowing Click/Rich to emit a second, non-JSON document.
    from hhtools.cli.agent import run

    raise typer.Exit(code=run(ctx.args))


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
        from hhtools.cli.home import print_homepage

        print_homepage(version=__version__)


if __name__ == "__main__":
    app()
