from __future__ import annotations

from pathlib import Path

import pytest
import typer

from hhtools.cli import retarget
from hhtools.io.datasets.omomo import OmomoAdapter
from hhtools.io.datasets.parc_ms import ParcMsAdapter


def test_interaction_pickle_routes_parc_to_meshmimic_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = tmp_path / "parc_ms" / "jump" / "jump.pkl"
    clip.parent.mkdir(parents=True)
    clip.touch()
    (clip.parent / "jump_terrain.obj").touch()
    expected = object()
    calls: list[tuple[Path, str]] = []

    def load(adapter: ParcMsAdapter, sequence: str):
        calls.append((adapter.root, sequence))
        return expected

    monkeypatch.setattr(ParcMsAdapter, "load_motion", load)

    assert retarget._load_motion_any(clip) is expected
    assert calls == [(clip.parent.parent, "jump/jump.pkl")]


def test_interaction_pickle_routes_omomo_to_intermimic_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = tmp_path / "OMOMO" / "carry" / "carry.pkl"
    clip.parent.mkdir(parents=True)
    clip.touch()
    (clip.parent / "box_cleaned_simplified.obj").touch()
    expected = object()
    calls: list[tuple[Path, str]] = []

    def load(adapter: OmomoAdapter, sequence: str):
        calls.append((adapter.root, sequence))
        return expected

    monkeypatch.setattr(OmomoAdapter, "load_motion", load)

    assert retarget._load_motion_any(clip) is expected
    assert calls == [(clip.parent.parent, "carry/carry.pkl")]


def test_interaction_pickle_requires_a_recognized_scene_sidecar(tmp_path: Path) -> None:
    clip = tmp_path / "unknown" / "clip.pkl"
    clip.parent.mkdir(parents=True)
    clip.touch()

    with pytest.raises(typer.BadParameter, match="OMOMO object mesh or a PARC terrain"):
        retarget._load_motion_any(clip)
