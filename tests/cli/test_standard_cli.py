from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from hhtools.cli import bodymodel, dataset, robot


class _ImportAdapter:
    outcomes: dict[str, object] = {}

    def __init__(self, _root) -> None:
        pass

    def list_sequences(self) -> list[str]:
        return list(self.outcomes)

    def load_motion(self, sequence: str):
        outcome = self.outcomes[sequence]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(name=outcome)


def _install_import_adapter(monkeypatch: pytest.MonkeyPatch, outcomes: dict[str, object]) -> None:
    adapter = type("ImportAdapter", (_ImportAdapter,), {"outcomes": outcomes})
    monkeypatch.setattr(dataset, "registered_datasets", lambda: {"test": adapter})
    monkeypatch.setattr("hhtools.io.npz.save_npz", lambda _motion, _path: None)


def test_import_run_summarizes_outcomes_and_fails_on_load_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_import_adapter(
        monkeypatch,
        {
            "good": "converted",
            "unsupported": NotImplementedError("not supported"),
            "broken": ValueError("invalid clip"),
        },
    )

    result = CliRunner().invoke(
        dataset.app,
        ["run", "--dataset", "test", "--root", str(tmp_path), "--out", str(tmp_path / "out")],
    )

    assert result.exit_code == 1
    assert "Import summary: 1 ok, 1 skipped, 1 failed" in result.stdout


def test_import_run_treats_skips_as_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _install_import_adapter(monkeypatch, {"unsupported": NotImplementedError("not supported")})

    result = CliRunner().invoke(
        dataset.app,
        ["run", "--dataset", "test", "--root", str(tmp_path), "--out", str(tmp_path / "out")],
    )

    assert result.exit_code == 0
    assert "Import summary: 0 ok, 1 skipped, 0 failed" in result.stdout


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [
        ({"smpl": True, "smplh": True, "smplx": True}, 0),
        ({"smpl": True, "smplh": False, "smplx": True}, 1),
        ({}, 1),
    ],
)
def test_bodymodel_check_is_a_strict_readiness_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    status: dict[str, bool],
    exit_code: int,
) -> None:
    monkeypatch.setattr(bodymodel, "check_body_models", lambda _root: status)

    result = CliRunner().invoke(bodymodel.app, ["check", "--root", str(tmp_path)])

    assert result.exit_code == exit_code


def test_bodymodel_check_help_documents_readiness_exit_status() -> None:
    result = CliRunner().invoke(bodymodel.app, ["check", "--help"])

    assert result.exit_code == 0
    assert "readiness probe" in result.stdout
    assert "Exits non-zero" in result.stdout


def test_robot_schema_reserves_stdout_for_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns = ["time", *(f"root_{index}" for index in range(7)), "joint_a", "joint_b"]
    monkeypatch.setattr(robot, "refresh", lambda: None)
    monkeypatch.setattr(robot, "_get_preset", lambda _name: object())
    monkeypatch.setattr(robot, "load_robot", lambda _preset, *, compile_mjcf: object())
    monkeypatch.setattr(robot, "header_columns", lambda _model: columns)

    result = CliRunner().invoke(robot.app, ["schema", "test"])

    assert result.exit_code == 0
    assert result.stdout == ",".join(columns) + "\n"
    assert result.stderr == "10 columns: time, 7 root (xyz+xyzw), 2 DOF\n"


def test_robot_info_reports_an_unknown_preset_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(robot, "refresh", lambda: None)

    def missing_preset(_name: str) -> None:
        raise KeyError("missing")

    monkeypatch.setattr("hhtools.robot.registry.get", missing_preset)

    result = CliRunner().invoke(robot.app, ["info", "missing", "--no-mjcf"])

    assert result.exit_code == 1
    assert result.stdout.count("missing") == 1
    assert "Failed to load" not in result.stdout
