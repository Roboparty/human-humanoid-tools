from __future__ import annotations

import io
import sys

from rich.console import Console
from typer.testing import CliRunner

from hhtools.cli import home, main


def _render(*, width: int) -> str:
    output = io.StringIO()
    console = Console(
        file=output,
        width=width,
        color_system=None,
        force_terminal=False,
        highlight=False,
    )
    home.render_homepage(console, version="1.2.3")
    return output.getvalue()


def test_homepage_has_the_planned_sections_and_only_real_destinations() -> None:
    output = _render(width=110)

    assert "HHTOOLS   v1.2.3" in output
    assert "by Jagger Shen, Nora Sun and hhtools contributors" in output
    assert "WORKFLOWS" in output
    assert "Human → Robot" in output
    assert "Robot → Robot" in output
    assert "Batch" in output
    assert "TOOLS" in output
    assert "Convert Motion" in output
    assert "Robots" in output
    assert "System Check" in output
    assert "OPEN" in output
    assert "WebUI" in output
    assert "Desktop GUI" in output
    assert "$ hhtools retarget run" in output
    assert "$ hhtools web" in output
    assert "$ hhtools-desktop" in output
    assert "\x1b[" not in output


def test_homepage_keeps_every_command_visible_on_a_narrow_terminal() -> None:
    output = _render(width=52)

    for command in (
        "hhtools retarget run",
        "hhtools convert run",
        "hhtools robot list",
        "hhtools doctor",
        "hhtools web",
        "hhtools-desktop",
        "hhtools --help",
    ):
        assert command in output


def test_homepage_does_not_stretch_across_an_ultrawide_terminal() -> None:
    output = _render(width=180)

    assert max(len(line) for line in output.splitlines()) <= 104
    assert "$ hhtools retarget run" in output
    assert "$ hhtools-desktop" in output


def test_identity_remains_readable_on_a_very_narrow_terminal() -> None:
    output = _render(width=40)

    assert "HHTOOLS   v1.2.3" in output
    assert "Human motion → humanoid robots" in output
    assert "Jagger Shen, Nora Sun" in output


def test_root_without_arguments_renders_homepage_without_prompting() -> None:
    result = CliRunner().invoke(main.app, [])

    assert result.exit_code == 0
    assert "HHTOOLS" in result.stdout
    assert "WORKFLOWS" in result.stdout
    assert "Full command reference" in result.stdout
    assert "Usage:" not in result.stdout
    assert "\x1b[" not in result.stdout
    assert result.stderr == ""


def test_root_help_and_version_keep_their_existing_behavior() -> None:
    runner = CliRunner()

    help_result = runner.invoke(main.app, ["--help"])
    version_result = runner.invoke(main.app, ["--version"])

    assert help_result.exit_code == 0
    assert "Usage:" in help_result.stdout
    assert "Commands" in help_result.stdout
    assert "retarget" in help_result.stdout
    assert version_result.exit_code == 0
    assert version_result.stdout.startswith("hhtools ")


def test_no_argument_startup_does_not_import_command_trees(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["hhtools"])

    assert main._subcommands_for_argv() == []
