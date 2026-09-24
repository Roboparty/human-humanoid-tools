from __future__ import annotations

import errno
import os
from pathlib import Path

from hhtools.web.library import motion_library_links as links


def _prepare_single_file_drop(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    source = tmp_path / "source" / "walk.npz"
    source.parent.mkdir()
    source.write_bytes(b"motion-data")

    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / source.name).write_bytes(source.read_bytes())

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(links, "auto_resolve_source_files", lambda _rels: [source])
    return source, upload


def test_single_file_uses_hardlink_when_symlink_is_denied(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source, upload = _prepare_single_file_drop(tmp_path, monkeypatch)

    def deny_symlink(*_args, **_kwargs) -> None:
        raise OSError(errno.EPERM, "symlink privilege denied")

    monkeypatch.setattr(Path, "symlink_to", deny_symlink)

    library_dir, label, mode = links.materialize_drop(
        [source.name],
        folder_label="windows-test",
        upload_drop=upload,
    )

    materialized = library_dir / source.name
    assert label == "windows-test"
    assert mode == "hardlink"
    assert materialized.read_bytes() == source.read_bytes()
    assert os.path.samefile(materialized, source)


def test_single_file_copies_when_symlink_and_hardlink_are_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source, upload = _prepare_single_file_drop(tmp_path, monkeypatch)

    def deny_symlink(*_args, **_kwargs) -> None:
        raise OSError(errno.EPERM, "symlink privilege denied")

    def deny_hardlink(*_args, **_kwargs) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(Path, "symlink_to", deny_symlink)
    monkeypatch.setattr(os, "link", deny_hardlink)

    library_dir, _label, mode = links.materialize_drop(
        [source.name],
        folder_label="copy-test",
        upload_drop=upload,
    )

    materialized = library_dir / source.name
    assert mode == "copy"
    assert materialized.read_bytes() == source.read_bytes()
    assert not os.path.samefile(materialized, source)


def test_folder_upload_copies_when_directory_symlink_is_denied(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "source" / "demo-folder"
    source_file = source_dir / "walk.npz"
    source_dir.mkdir(parents=True)
    source_file.write_bytes(b"folder-motion")

    upload = tmp_path / "upload"
    uploaded_file = upload / "demo-folder" / "walk.npz"
    uploaded_file.parent.mkdir(parents=True)
    uploaded_file.write_bytes(source_file.read_bytes())

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(links, "auto_resolve_source_dir", lambda _rels: source_dir)
    monkeypatch.setattr(links, "auto_resolve_source_files", lambda _rels: [source_file])

    def deny_symlink(*_args, **_kwargs) -> None:
        raise OSError(errno.EPERM, "directory symlink privilege denied")

    monkeypatch.setattr(Path, "symlink_to", deny_symlink)

    library_dir, label, mode = links.materialize_drop(
        ["demo-folder/walk.npz"],
        folder_label="folder-copy",
        upload_drop=upload,
    )

    assert label == "folder-copy"
    assert mode == "copy"
    assert (library_dir / "demo-folder" / "walk.npz").read_bytes() == b"folder-motion"
