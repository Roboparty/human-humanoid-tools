from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from hhtools.cli import convert


@pytest.mark.parametrize("successful_count", [0, 1])
def test_convert_returns_failure_for_any_failed_input(tmp_path, monkeypatch, successful_count):
    broken = tmp_path / "broken.bvh"
    broken.write_text("invalid")
    if successful_count:
        (tmp_path / "good.bvh").write_text("valid")

    def load(source, **kwargs):
        if source == broken:
            raise ValueError("invalid clip")
        return SimpleNamespace(framerate=50.0)

    monkeypatch.setattr(convert, "_load_one", load)
    monkeypatch.setattr(convert.npz, "save_npz", lambda motion, path: path.touch())
    result = CliRunner().invoke(convert.app, ["run", str(tmp_path), "--out", str(tmp_path / "out")])
    assert result.exit_code == 2
    assert "invalid clip" in result.stdout
    if successful_count:
        assert (tmp_path / "out/good.npz").is_file()


def test_convert_existing_destination_is_an_explicit_failure(tmp_path, monkeypatch):
    source = tmp_path / "source.bvh"
    source.touch()
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "source.npz").write_bytes(b"preserve")
    monkeypatch.setattr(
        convert,
        "_load_one",
        lambda *args, **kwargs: SimpleNamespace(framerate=50.0),
    )
    result = CliRunner().invoke(convert.app, ["run", str(source), "--out", str(destination)])
    assert result.exit_code == 2
    assert (destination / "source.npz").read_bytes() == b"preserve"
