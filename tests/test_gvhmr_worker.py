from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


def _load_worker() -> ModuleType:
    worker_path = Path(__file__).parents[1] / "hhtools" / "integrations" / "gvhmr_worker.py"
    spec = importlib.util.spec_from_file_location("hhtools_gvhmr_worker", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hydra_alias_handles_common_non_override_safe_video_names(tmp_path: Path) -> None:
    worker = _load_worker()
    video = tmp_path / "视频 (take=1),#final.mp4"
    video.write_bytes(b"video")
    output_root = tmp_path / "output"
    output_root.mkdir()

    try:
        alias = worker._hydra_safe_video_alias(video, output_root)  # noqa: SLF001
    except OSError:
        pytest.skip("file symlinks are not available on this host")

    assert alias.parent == tmp_path / ".hhtools-gvhmr-input"
    assert alias.name.startswith("source_")
    assert alias.suffix == ".mp4"
    assert alias.read_bytes() == b"video"
    assert all(character.isascii() for character in alias.name)


def test_video_normalization_requests_a_30_fps_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _load_worker()
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    output_root = tmp_path / "output"
    output_root.mkdir()
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"normalized")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(worker.shutil, "which", lambda _name: "/runtime/bin/ffmpeg")
    monkeypatch.setattr(worker.subprocess, "run", fake_run)

    normalized = worker._normalize_video_framerate(video, output_root)  # noqa: SLF001

    assert normalized.read_bytes() == b"normalized"
    assert commands[0][0] == "/runtime/bin/ffmpeg"
    assert commands[0][commands[0].index("-vf") + 1] == "fps=30"
    assert commands[0][commands[0].index("-map") + 1] == "0:v:0"
    assert "-an" in commands[0]
