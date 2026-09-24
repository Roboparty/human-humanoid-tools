from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from hhtools.cli import desktop_sidecar, web
from hhtools.web import server
from hhtools.web.dependencies import MissingWebDependenciesError


def test_web_cli_defers_default_job_settings_to_persistent_backend_config(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(server, "run_web", lambda **kwargs: captured.update(kwargs))

    result = CliRunner().invoke(web.app, [])

    assert result.exit_code == 0, result.output
    assert captured["max_running_jobs"] is None
    assert captured["max_queued_jobs"] is None
    assert captured["max_batch_items"] is None
    assert captured["max_batch_total_frames"] is None


def test_web_cli_reads_env_and_explicit_options_override_it(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(server, "run_web", lambda **kwargs: captured.update(kwargs))

    result = CliRunner().invoke(
        web.app,
        ["--max-running-jobs", "3"],
        env={
            "HHTOOLS_MAX_RUNNING_JOBS": "7",
            "HHTOOLS_MAX_QUEUED_JOBS": "24",
            "HHTOOLS_MAX_BATCH_ITEMS": "5000",
            "HHTOOLS_MAX_BATCH_TOTAL_FRAMES": "0",
        },
    )

    assert result.exit_code == 0, result.output
    assert captured["max_running_jobs"] == 3
    assert captured["max_queued_jobs"] == 24
    assert captured["max_batch_items"] == 5000
    assert captured["max_batch_total_frames"] == 0


def test_web_cli_reads_packaged_runtime_paths_from_environment(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(server, "run_web", lambda **kwargs: captured.update(kwargs))

    result = CliRunner().invoke(
        web.app,
        ["--source", "/explicit/motions"],
        env={
            "HHTOOLS_SOURCE_ROOT": "/bundled/motions",
            "HHTOOLS_SAVE_DIR": "/home/user/.local/share/hhtools/save_npz",
            "HHTOOLS_CACHE_DIR": "/home/user/.cache/hhtools",
        },
    )

    assert result.exit_code == 0, result.output
    assert captured["source_root"] == Path("/explicit/motions")
    assert captured["save_dir"] == Path("/home/user/.local/share/hhtools/save_npz")
    assert captured["cache_dir"] == Path("/home/user/.cache/hhtools")


def test_web_cli_rejects_negative_job_limits() -> None:
    result = CliRunner().invoke(web.app, ["--max-running-jobs", "-1"])

    assert result.exit_code == 2


def test_web_cli_reports_missing_runtime_dependencies(monkeypatch) -> None:
    def fail_to_start(**_kwargs) -> None:
        raise MissingWebDependenciesError(("uvicorn", "python-multipart"))

    monkeypatch.setattr(server, "run_web", fail_to_start)

    result = CliRunner().invoke(web.app, [])

    assert result.exit_code == 1
    assert "uvicorn, python-multipart" in result.output
    assert "uv sync --locked --extra web" in result.output
    assert "ModuleNotFoundError" not in result.output


def _desktop_args(tmp_path: Path) -> list[str]:
    return [
        "--source",
        str(tmp_path / "source"),
        "--save-dir",
        str(tmp_path / "save"),
        "--cache",
        str(tmp_path / "cache"),
        "--port",
        "43123",
        "--session-secret",
        "unit-test-secret",
    ]


def test_desktop_sidecar_reads_job_limits_from_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict = {}
    monkeypatch.setenv("HHTOOLS_MAX_RUNNING_JOBS", "2")
    monkeypatch.setenv("HHTOOLS_MAX_QUEUED_JOBS", "32")
    monkeypatch.setenv("HHTOOLS_MAX_BATCH_ITEMS", "0")
    monkeypatch.setenv("HHTOOLS_MAX_BATCH_TOTAL_FRAMES", "2500000")
    monkeypatch.setattr(
        server,
        "run_desktop_sidecar",
        lambda **kwargs: captured.update(kwargs),
    )

    desktop_sidecar.main(_desktop_args(tmp_path))

    assert captured["max_running_jobs"] == 2
    assert captured["max_queued_jobs"] == 32
    assert captured["max_batch_items"] == 0
    assert captured["max_batch_total_frames"] == 2_500_000


def test_desktop_sidecar_explicit_limit_overrides_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict = {}
    monkeypatch.setenv("HHTOOLS_MAX_RUNNING_JOBS", "8")
    monkeypatch.setattr(
        server,
        "run_desktop_sidecar",
        lambda **kwargs: captured.update(kwargs),
    )

    desktop_sidecar.main([*_desktop_args(tmp_path), "--max-running-jobs", "1"])

    assert captured["max_running_jobs"] == 1


def test_desktop_sidecar_rejects_negative_job_limit(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        desktop_sidecar.main([*_desktop_args(tmp_path), "--max-queued-jobs", "-1"])

    assert error.value.code == 2


def test_desktop_sidecar_reports_missing_runtime_dependencies(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    def fail_to_start(**_kwargs) -> None:
        raise MissingWebDependenciesError(("uvicorn",))

    monkeypatch.setattr(server, "run_desktop_sidecar", fail_to_start)

    with pytest.raises(SystemExit) as error:
        desktop_sidecar.main(_desktop_args(tmp_path))

    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "uvicorn" in captured.err
    assert "uv sync --locked --extra web" in captured.err
    assert "ModuleNotFoundError" not in captured.err
