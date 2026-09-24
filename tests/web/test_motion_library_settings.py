from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhtools.utils import paths
from hhtools.web.library.motion_library_settings import (
    MOTION_LIBRARY_MARKER_FILENAME,
    MotionLibrarySettings,
    MotionLibrarySettingsStore,
    effective_motion_library_root,
    motion_library_marker_path,
    motion_library_marker_payload,
    updated_motion_library_settings,
    validate_motion_library_marker,
    validate_motion_library_settings,
)


def test_settings_store_round_trips_an_absolute_root_atomically(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "config" / "motion-library-settings.json"
    library_root = tmp_path / "motion-library"
    store = MotionLibrarySettingsStore(settings_path)

    store.save(MotionLibrarySettings(root=library_root))

    assert store.load() == MotionLibrarySettings(root=library_root.resolve())
    assert json.loads(settings_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "root": str(library_root.resolve()),
    }
    assert not list(settings_path.parent.glob("*.tmp"))


def test_settings_store_round_trips_null_as_follow_default(tmp_path: Path) -> None:
    settings_path = tmp_path / "motion-library-settings.json"
    store = MotionLibrarySettingsStore(settings_path)

    store.save(MotionLibrarySettings())

    assert store.load() == MotionLibrarySettings()
    assert json.loads(settings_path.read_text(encoding="utf-8"))["root"] is None


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"schema_version": 2, "root": None},
        {"schema_version": 1},
        {"schema_version": 1, "root": None, "unexpected": True},
        {"schema_version": 1, "root": "relative/library"},
        {"schema_version": 1, "root": ""},
        {"schema_version": 1, "root": 7},
    ],
)
def test_settings_store_uses_defaults_for_invalid_content(
    tmp_path: Path,
    payload: object,
) -> None:
    settings_path = tmp_path / "motion-library-settings.json"
    settings_path.write_text(json.dumps(payload), encoding="utf-8")

    assert MotionLibrarySettingsStore(settings_path).load() == MotionLibrarySettings()


@pytest.mark.parametrize(
    "patch",
    [
        {},
        {"unexpected": "value"},
        {"root": "relative/library"},
        {"root": True},
    ],
)
def test_settings_patch_is_strict(patch: object) -> None:
    with pytest.raises(ValueError):
        updated_motion_library_settings(MotionLibrarySettings(), patch)


def test_settings_patch_can_select_and_reset_a_root(tmp_path: Path) -> None:
    selected = updated_motion_library_settings(
        MotionLibrarySettings(),
        {"root": str(tmp_path / "selected")},
    )

    assert selected.root == (tmp_path / "selected").resolve()
    assert updated_motion_library_settings(selected, {"root": None}).root is None


def test_environment_root_overrides_persistent_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = validate_motion_library_settings(str(tmp_path / "selected"))
    override = tmp_path / "environment"
    monkeypatch.setenv(paths.HHTOOLS_MOTION_LIBRARY_ROOT_ENV, str(override))

    assert effective_motion_library_root(selected) == override.resolve()


def test_platform_default_preserves_an_existing_legacy_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = tmp_path / ".config" / "hhtools" / "motions"
    legacy.mkdir(parents=True)
    monkeypatch.delenv(paths.HHTOOLS_MOTION_LIBRARY_ROOT_ENV, raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(paths, "user_data_dir", lambda *_args: str(tmp_path / "data"))

    assert paths.user_motion_library_root() == legacy


def test_platform_default_uses_data_dir_for_a_new_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(paths.HHTOOLS_MOTION_LIBRARY_ROOT_ENV, raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(paths, "user_data_dir", lambda *_args: str(tmp_path / "data"))

    assert paths.user_motion_library_root() == tmp_path / "data" / "motions"


def test_xdg_and_settings_path_overrides_are_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xdg = tmp_path / "xdg"
    settings_path = tmp_path / "custom" / "library.json"
    monkeypatch.delenv(paths.HHTOOLS_MOTION_LIBRARY_ROOT_ENV, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv(
        paths.HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH_ENV,
        str(settings_path),
    )

    assert paths.user_motion_library_root() == xdg / "hhtools" / "motions"
    assert paths.user_motion_library_settings_path() == settings_path


def test_marker_path_is_reserved_but_not_created(tmp_path: Path) -> None:
    marker = motion_library_marker_path(tmp_path)

    assert marker == tmp_path.resolve() / MOTION_LIBRARY_MARKER_FILENAME
    assert not marker.exists()


def test_canonical_motion_library_marker_is_strictly_validated(tmp_path: Path) -> None:
    marker = motion_library_marker_path(tmp_path)
    marker.write_text(json.dumps(motion_library_marker_payload()), encoding="utf-8")

    assert validate_motion_library_marker(tmp_path) is True


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"schema_version": True, "managed_by": "hhtools"},
        {"schema_version": 2, "managed_by": "hhtools"},
        {"schema_version": 1, "managed_by": "another-tool"},
        {"schema_version": 1, "managed_by": "hhtools", "unexpected": True},
    ],
)
def test_invalid_motion_library_marker_contents_are_rejected(
    tmp_path: Path,
    payload: object,
) -> None:
    motion_library_marker_path(tmp_path).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ownership marker"):
        validate_motion_library_marker(tmp_path)
