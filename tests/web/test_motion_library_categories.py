from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from hhtools.web.library.motion_library_categories import (
    _SIDECAR_DIRECTORY_CACHE_MAXSIZE,
    _cached_sidecar_category,
    infer_motion_category,
)


@pytest.fixture(autouse=True)
def _reset_sidecar_directory_cache():
    _cached_sidecar_category.cache_clear()
    yield
    _cached_sidecar_category.cache_clear()


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        ("omomo", "object"),
        ("OMOMO", "object"),
        ("parc_ms", "terrain"),
        ("meshmimic_holosoma", "terrain"),
        ("holosoma", "terrain"),
        ("amass", "motion"),
        ("motion_x", "motion"),
        ("phuma", "motion"),
        ("lafan", "motion"),
        ("mocap", "motion"),
        ("soma", "motion"),
        ("xsens_mocap", "motion"),
        ("gvhmr", "motion"),
        ("kungfu_athlete", "motion"),
        ("glb", "motion"),
        ("unified_npz", "motion"),
        ("unknown", "motion"),
    ],
)
def test_dataset_adapter_mapping(dataset: str, expected: str) -> None:
    assert infer_motion_category({"dataset": dataset}) == expected


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("mimic", "motion"),
        ("intermimic", "object"),
        ("meshmimic", "terrain"),
        ("auto", "motion"),
    ],
)
def test_upload_profile_mapping(profile: str, expected: str) -> None:
    assert infer_motion_category({"upload_profile": profile}) == expected


def test_scene_specific_dataset_wins_over_stale_upload_profile() -> None:
    entry = {"dataset": "omomo", "upload_profile": "mimic"}
    assert infer_motion_category(entry) == "object"


def test_omomo_object_sidecar_classifies_unknown_adapter(tmp_path: Path) -> None:
    clip = tmp_path / "take" / "take.pkl"
    clip.parent.mkdir()
    clip.write_bytes(b"")
    (clip.parent / "chair_cleaned_simplified.obj").write_text("", encoding="utf-8")

    assert infer_motion_category({"dataset": "unknown", "source_path": clip}) == "object"


@pytest.mark.parametrize("terrain_name", ["terrain.obj", "take_terrain.obj"])
def test_terrain_sidecar_classifies_unknown_adapter(
    tmp_path: Path,
    terrain_name: str,
) -> None:
    clip = tmp_path / "take" / "take.pkl"
    clip.parent.mkdir()
    clip.write_bytes(b"")
    (clip.parent / terrain_name).write_text("", encoding="utf-8")

    assert infer_motion_category({"dataset": "unknown", "source_path": clip}) == "terrain"


def test_sidecar_enumeration_is_shared_by_clips_in_the_same_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "take"
    parent.mkdir()
    first = parent / "first.pkl"
    second = parent / "second.pkl"
    first.write_bytes(b"")
    second.write_bytes(b"")
    (parent / "terrain.obj").write_text("", encoding="utf-8")
    original_iterdir = Path.iterdir
    enumerations = 0

    def count_iterdir(path: Path):
        nonlocal enumerations
        if path == parent:
            enumerations += 1
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", count_iterdir)

    assert infer_motion_category({"source_path": first}) == "terrain"
    assert infer_motion_category({"source_path": second}) == "terrain"
    assert enumerations == 1


def test_sidecar_cache_invalidates_when_directory_contents_change(tmp_path: Path) -> None:
    parent = tmp_path / "take"
    parent.mkdir()
    clip = parent / "take.pkl"
    clip.write_bytes(b"")

    assert infer_motion_category({"source_path": clip}) == "motion"

    (parent / "chair_cleaned_simplified.obj").write_text("", encoding="utf-8")
    directory_stat = parent.stat()
    # Make the metadata transition deterministic on filesystems with coarse
    # timestamp updates while preserving the production invalidation path.
    os.utime(
        parent,
        ns=(directory_stat.st_atime_ns, directory_stat.st_mtime_ns + 1_000_000_000),
    )

    assert infer_motion_category({"source_path": clip}) == "object"


def test_sidecar_directory_cache_is_bounded(tmp_path: Path) -> None:
    for index in range(_SIDECAR_DIRECTORY_CACHE_MAXSIZE + 8):
        parent = tmp_path / f"take-{index}"
        parent.mkdir()
        clip = parent / "take.pkl"
        clip.write_bytes(b"")
        assert infer_motion_category({"source_path": clip}) == "motion"

    cache_info = _cached_sidecar_category.cache_info()
    assert cache_info.maxsize == _SIDECAR_DIRECTORY_CACHE_MAXSIZE
    assert cache_info.currsize == _SIDECAR_DIRECTORY_CACHE_MAXSIZE


@pytest.mark.parametrize(
    ("source_path", "expected"),
    [
        ("/data/motions/mimic/AMASS/walk.npz", "motion"),
        ("/data/motions/intermimic/OMOMO/take/take.pkl", "object"),
        ("/data/motions/meshmimic/parc_ms/take/take.pkl", "terrain"),
        (r"C:\motions\intermimic\OMOMO\take\take.pkl", "object"),
        (r"C:\motions\meshmimic\holosoma\parkour_1\parkour_1.npy", "terrain"),
    ],
)
def test_grouping_path_is_cross_platform_fallback(source_path: str, expected: str) -> None:
    assert infer_motion_category({"source_path": source_path}) == expected


def test_sequence_or_folder_label_can_supply_grouping_hint() -> None:
    assert infer_motion_category({"sequence_id": "intermimic/clip.pkl"}) == "object"
    assert infer_motion_category({"folder_label": "meshmimic"}) == "terrain"


def test_existing_category_is_idempotent() -> None:
    entry = {
        "motion_category": "terrain",
        "dataset": "omomo",
        "upload_profile": "intermimic",
    }
    assert infer_motion_category(entry) == "terrain"


@dataclass
class _EntryObject:
    dataset: str
    source_path: Path


def test_library_entry_like_object_is_supported(tmp_path: Path) -> None:
    assert infer_motion_category(_EntryObject("parc_ms", tmp_path / "clip.pkl")) == "terrain"


def test_filesystem_errors_safely_fall_back_to_motion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_directory_read(_path: Path):
        raise PermissionError("library mount is unavailable")

    monkeypatch.setattr(Path, "iterdir", deny_directory_read)

    clip = tmp_path / "locked" / "clip.pkl"
    clip.parent.mkdir()
    clip.write_bytes(b"")

    assert infer_motion_category({"source_path": clip}) == "motion"
    assert infer_motion_category(None) == "motion"
