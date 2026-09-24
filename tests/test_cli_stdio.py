from __future__ import annotations

import os
import subprocess
import sys

from hhtools.cli import main


def test_cli_entrypoint_overrides_legacy_windows_encoding() -> None:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "0"
    env["PYTHONIOENCODING"] = "cp1252"
    script = (
        "import sys; sys.argv = ['hhtools', 'web']; import hhtools.cli.main; print('✓ 中文文件名')"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert completed.stdout.decode("utf-8").strip() == "✓ 中文文件名"


def test_version_flag_does_not_load_optional_command_trees(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["hhtools", "--version"])

    assert main._subcommands_for_argv() == []


def test_legacy_ui_command_is_retired() -> None:
    commands = {name for name, _module, _help in main._SUBCOMMANDS}

    assert "ui" not in commands
    assert "web" in commands
