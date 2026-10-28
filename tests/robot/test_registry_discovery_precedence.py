"""Packaged defaults must preserve user robot definitions and calibrations."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hhtools.robot import registry


def _write_robot(root: Path, name: str, calibration: str) -> bytes:
    folder = root / name
    folder.mkdir(parents=True)
    document = (
        f"name: {name}\n"
        f"display_name: {calibration}\n"
        f"retarget:\n  references:\n    smpl:\n      calibration: {calibration}.yaml\n"
    ).encode()
    (folder / "robot.yaml").write_bytes(document)
    (folder / f"{calibration}.yaml").write_text(
        f"marker: {calibration}\n", encoding="utf-8",
    )
    return document


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("explicit_override", [False, True])
def test_user_calibration_wins_over_bundle_unless_explicitly_overridden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    read_only: bool,
    explicit_override: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace_robots = workspace / "configs" / "robots"
    bundled_robots = tmp_path / "bundle" / "robots"
    user_robots = tmp_path / "user" / "robots"
    explicit_robots = tmp_path / "explicit" / "robots"
    _write_robot(workspace_robots, "rp1", "workspace")
    _write_robot(bundled_robots, "rp1", "bundled")
    _write_robot(bundled_robots, "g1_29dof", "bundled_g1")
    user_document = _write_robot(user_robots, "rp1", "user_calibration")
    _write_robot(explicit_robots, "rp1", "explicit_calibration")
    monkeypatch.setattr(registry, "__file__", str(workspace / "hhtools" / "robot" / "registry.py"))
    monkeypatch.setattr(registry, "user_robot_dir", lambda: user_robots)
    monkeypatch.setattr(registry, "_CACHE", None)
    monkeypatch.setenv(
        "HHTOOLS_BUNDLED_ROBOT_PATH",
        os.pathsep.join((str(tmp_path / "missing_bundle"), str(bundled_robots))),
    )
    monkeypatch.setenv("HHTOOLS_ROBOT_PATH", str(explicit_robots) if explicit_override else "")

    discover = registry.list_presets_readonly if read_only else registry.refresh
    robots = {robot.name: robot for robot in discover()}
    expected_root = explicit_robots if explicit_override else user_robots
    expected_calibration = "explicit_calibration" if explicit_override else "user_calibration"

    assert robots["rp1"].root_dir == expected_root / "rp1"
    assert robots["rp1"].meta["retarget"]["references"]["smpl"]["calibration"] == (
        f"{expected_calibration}.yaml"
    )
    assert robots["g1_29dof"].root_dir == bundled_robots / "g1_29dof"
    assert (user_robots / "rp1" / "robot.yaml").read_bytes() == user_document
    assert (user_robots / "rp1" / "user_calibration.yaml").read_text() == "marker: user_calibration\n"
