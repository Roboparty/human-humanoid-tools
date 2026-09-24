#!/usr/bin/env python3
"""Install HHTools' six curated robots from pinned official archives.

Models are installed into the user Robot Library, never the Git repository.
Each archive is pinned to a commit; only the primary URDF, meshes referenced by
that URDF, and upstream license files are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

MANIFEST_NAME, MAX_ARCHIVE_BYTES, MAX_MEMBER_BYTES = "SOURCE.json", 2 * 1024**3, 256 * 1024**2


@dataclass(frozen=True, slots=True)
class RobotSource:
    name: str
    display_name: str
    repository: str
    commit: str
    main_urdf: str
    license_paths: tuple[str, ...]


SOURCES: tuple[RobotSource, ...] = (
    RobotSource(
        "g1_29dof", "Unitree G1",
        "https://github.com/unitreerobotics/unitree_ros.git",
        "7d6075f7f58588b189b940130e3edab3c839b2df",
        "robots/g1_description/g1_29dof.urdf", ("LICENSE",),
    ),
    RobotSource(
        "roboto_origin", "ROBOTO_ORIGIN (RPO)",
        "https://github.com/Roboparty/rpo_description.git",
        "37aac9ca665e92731444a1618320078e7ba21569",
        "urdf/rpo.urdf", ("LICENSE",),
    ),
    RobotSource(
        "agibot_x2_ultra", "AgiBot X2",
        "https://github.com/AgibotTech/agibot_x2_urdf.git",
        "77f43eb0904dae4c48ccd9154fee824f8ffd4d38",
        "X2_URDF-v1.4.0/X2-Ultra.urdf", ("LICENSE",),
    ),
    RobotSource(
        "asimov_1", "Asimov 1",
        "https://github.com/menloresearch/asimov-1.git",
        "b8420ffe99159065152aa1321a03147c0962f251",
        "sim-model/urdf/asimov_1.urdf", ("HARDWARE-LICENSE.txt",),
    ),
    RobotSource(
        "fourier_gr2", "Fourier GR-2",
        "https://github.com/FFTAI/Wiki-GRx-Models.git",
        "7d96c758f048fe1bf92b3258864d94771ae0c093",
        "GRX/GR2/gr2v3_8_7/basic_urdf/gr2v3_8_7.urdf", ("LICENSE",),
    ),
    RobotSource(
        "berkeley_humanoid_lite", "Berkeley Humanoid Lite",
        "https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Assets.git",
        "fc90fedd008b1e56a22e3c5221548d6b24f49707",
        (
            "data/robots/berkeley_humanoid/berkeley_humanoid_lite/"
            "urdf/berkeley_humanoid_lite.urdf"
        ), ("LICENCE",),
    ),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _repository_slug(repository: str) -> str:
    parsed = urllib.parse.urlparse(repository)
    slug = parsed.path.strip("/").removesuffix(".git")
    if parsed.scheme != "https" or parsed.hostname != "github.com" or slug.count("/") != 1:
        raise ValueError(f"unsupported source repository: {repository}")
    return slug


def _safe_relative(path: str) -> str:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ValueError(f"unsafe archive path: {path!r}")
    return candidate.as_posix()


def _download_archive(source: RobotSource, output: Path) -> int:
    slug = _repository_slug(source.repository)
    url = f"https://codeload.github.com/{slug}/tar.gz/{source.commit}"
    headers = {"User-Agent": "hhtools-builtin-robot-installer/1"}

    partial = output.with_suffix(".part")
    total = 0
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60.0) as response, partial.open(
            "wb"
        ) as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    raise RuntimeError(f"{source.name}: archive exceeds 2 GiB")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        partial.replace(output)
        return total
    finally:
        partial.unlink(missing_ok=True)


def _archive_index(
    archive: tarfile.TarFile,
    source: RobotSource,
) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    roots: set[str] = set()
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{source.name}: archive contains unsafe path {member.name!r}")
        if not path.parts:
            continue
        roots.add(path.parts[0])
        if not member.isfile() or len(path.parts) < 2:
            continue
        relative = _safe_relative(PurePosixPath(*path.parts[1:]).as_posix())
        if relative in members:
            raise ValueError(f"{source.name}: duplicate archive member {relative!r}")
        members[relative] = member
    if len(roots) != 1 or not next(iter(roots)).endswith(source.commit):
        raise ValueError(f"{source.name}: archive root does not match pinned commit")
    return members


def _member_bytes(
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    relative: str,
) -> bytes:
    relative = _safe_relative(relative)
    member = members.get(relative)
    if member is None:
        raise FileNotFoundError(f"archive is missing {relative}")
    if member.size > MAX_MEMBER_BYTES:
        raise ValueError(f"archive member is too large: {relative}")
    stream = archive.extractfile(member)
    if stream is None:
        raise FileNotFoundError(f"archive cannot read {relative}")
    data = stream.read(MAX_MEMBER_BYTES + 1)
    if len(data) != member.size:
        raise ValueError(f"archive member size mismatch: {relative}")
    return data


def _mesh_source_path(
    raw_uri: str,
    source: RobotSource,
    members: dict[str, tarfile.TarInfo],
) -> str:
    uri = raw_uri.strip().replace("\\", "/")
    package_uri = uri.startswith("package://")
    payload = uri.removeprefix("package://").lstrip("/")
    urdf_directory = posixpath.dirname(source.main_urdf)
    candidates = [
        posixpath.normpath(posixpath.join(urdf_directory, payload)),
        posixpath.normpath(payload),
    ]
    if package_uri and "/" in payload and not payload.startswith("../"):
        candidates.append(
            posixpath.normpath(posixpath.join(urdf_directory, payload.split("/", 1)[1]))
        )
    for candidate in candidates:
        try:
            safe = _safe_relative(candidate)
        except ValueError:
            continue
        if safe in members:
            return safe

    suffix = "/".join(part for part in payload.split("/") if part not in ("", ".", ".."))
    matches = [path for path in members if path == suffix or path.endswith(f"/{suffix}")]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{source.name}: cannot uniquely resolve mesh reference {raw_uri!r}"
        )
    return matches[0]


def _source_record(source_path: str, installed_path: str, data: bytes) -> dict[str, object]:
    return {
        "source_path": source_path,
        "installed_path": installed_path,
        "source_sha256": _sha256_bytes(data),
    }


def _materialize_model(
    source: RobotSource,
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    staged: Path,
) -> list[dict[str, object]]:
    urdf_data = _member_bytes(archive, members, source.main_urdf)
    root = ET.fromstring(urdf_data)
    meshes = staged / "meshes"
    meshes.mkdir(parents=True)
    copied: dict[str, str] = {}
    claimed: dict[str, str] = {}
    records: list[dict[str, object]] = []

    for element in root.iter("mesh"):
        uri = element.get("filename")
        if not uri:
            continue
        source_path = _mesh_source_path(uri, source, members)
        installed_name = copied.get(source_path)
        if installed_name is None:
            data = _member_bytes(archive, members, source_path)
            if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"{source.name}: unresolved Git LFS mesh {source_path}")
            digest = _sha256_bytes(data)
            installed_name = PurePosixPath(source_path).name
            if claimed.get(installed_name.lower(), digest) != digest:
                installed_name = f"{digest[:12]}-{installed_name}"
            claimed[installed_name.lower()] = digest
            (meshes / installed_name).write_bytes(data)
            copied[source_path] = installed_name
            records.append(_source_record(source_path, f"meshes/{installed_name}", data))
        element.set("filename", f"meshes/{installed_name}")

    installed_urdf = staged / PurePosixPath(source.main_urdf).name
    ET.ElementTree(root).write(installed_urdf, encoding="utf-8", xml_declaration=True)
    records.insert(0, _source_record(source.main_urdf, installed_urdf.name, urdf_data))
    return records


def _materialize_licenses(
    source: RobotSource,
    archive: tarfile.TarFile,
    members: dict[str, tarfile.TarInfo],
    staged: Path,
) -> list[dict[str, object]]:
    destination = staged / "licenses"
    destination.mkdir()
    records: list[dict[str, object]] = []
    for source_path in source.license_paths:
        data = _member_bytes(archive, members, source_path)
        output = destination / PurePosixPath(source_path).name
        output.write_bytes(data)
        records.append(_source_record(source_path, output.relative_to(staged).as_posix(), data))
    return records


def _set_display_name(yaml_path: Path, display_name: str) -> None:
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    with yaml_path.open("r", encoding="utf-8") as stream:
        payload = yaml.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid generated robot yaml: {yaml_path}")
    payload["display_name"] = display_name
    with yaml_path.open("w", encoding="utf-8") as stream:
        yaml.dump(payload, stream)


def _installed_files(staged: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(staged).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(staged.rglob("*"))
        if path.is_file() and path.name != MANIFEST_NAME
    ]


def _write_manifest(
    source: RobotSource,
    staged: Path,
    source_files: list[dict[str, object]],
) -> None:
    payload = {
        "schema_version": 1,
        "preset": source.name,
        "display_name": source.display_name,
        "source": {
            "repository": source.repository,
            "commit": source.commit,
            "main_urdf": source.main_urdf,
            "license_paths": list(source.license_paths),
            "materialized_with": "github-codeload-selective",
        },
        "source_files": source_files,
        "installed_files": _installed_files(staged),
    }
    (staged / MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _installed_file_matches(record: object, target: Path, seen: set[str]) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        relative = _safe_relative(str(record["path"]))
        size = int(record["bytes"])
        digest = str(record["sha256"])
    except (KeyError, TypeError, ValueError):
        return False
    path = target / relative
    if relative in seen or path.is_symlink() or not path.is_file():
        return False
    seen.add(relative)
    return path.stat().st_size == size and _sha256_file(path) == digest


def _is_up_to_date(source: RobotSource, target: Path) -> bool:
    try:
        manifest = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    upstream = manifest.get("source")
    source_matches = (
        manifest.get("preset") == source.name
        and manifest.get("display_name") == source.display_name
        and isinstance(upstream, dict)
        and upstream.get("repository") == source.repository
        and upstream.get("commit") == source.commit
        and upstream.get("main_urdf") == source.main_urdf
        and upstream.get("license_paths") == list(source.license_paths)
    )
    installed = manifest.get("installed_files")
    if not source_matches or not isinstance(installed, list) or not installed:
        return False
    seen: set[str] = set()
    return all(_installed_file_matches(record, target, seen) for record in installed) and (
        "robot.yaml" in seen
    )


def _publish(staged: Path, target: Path, *, replace: bool) -> None:
    if target.is_symlink():
        raise ValueError(f"refusing to replace symlinked preset: {target}")
    backup = target.parent / f".{target.name}.previous"
    if backup.exists():
        raise FileExistsError(f"refusing to overwrite stale backup: {backup}")
    if target.exists():
        if not replace:
            raise FileExistsError(f"preset already exists: {target}; pass --replace to replace it")
        target.rename(backup)
    try:
        staged.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    if backup.exists():
        try:
            shutil.rmtree(backup)
        except OSError as error:
            print(
                f"warning: installed model but could not remove {backup}: {error}",
                file=sys.stderr,
            )


def _install(source: RobotSource, library: Path, *, replace: bool) -> str:
    target = library / source.name
    if target.is_symlink():
        raise ValueError(f"refusing to use symlinked preset: {target}")
    if target.exists() and not replace and _is_up_to_date(source, target):
        return f"up to date: {source.name}"

    print(f"fetching {source.display_name} @ {source.commit[:12]}...", flush=True)
    with tempfile.TemporaryDirectory(prefix=f"hhtools-{source.name}-") as temp:
        archive_path = Path(temp) / "source.tar.gz"
        size = _download_archive(source, archive_path)
        print(f"  archive: {size / (1024 * 1024):.1f} MiB", flush=True)
        with tarfile.open(archive_path, "r:gz") as archive, tempfile.TemporaryDirectory(
            prefix=f".{source.name}-install-",
            dir=library,
        ) as install_temp:
            members = _archive_index(archive, source)
            staged = Path(install_temp) / source.name
            staged.mkdir()
            source_files = _materialize_model(source, archive, members, staged)
            source_files.extend(_materialize_licenses(source, archive, members, staged))

            from hhtools.robot.scaffold import scaffold_yaml_file
            from hhtools.robot.urdf_normalize import (
                detect_mesh_path_issues,
                ensure_urdf_meshes_resolvable,
            )

            urdf = staged / PurePosixPath(source.main_urdf).name
            ensure_urdf_meshes_resolvable(urdf, output_path=urdf)
            issues = detect_mesh_path_issues(urdf)
            if issues:
                raise ValueError(f"{source.name}: unresolved mesh: {'; '.join(issues[:5])}")
            scaffold = scaffold_yaml_file(urdf, root_dir=staged, overwrite=True)
            if scaffold.preset.name != source.name:
                raise ValueError(
                    f"{source.name}: scaffold generated {scaffold.preset.name!r}"
                )
            _set_display_name(scaffold.yaml_path, source.display_name)
            _write_manifest(source, staged, source_files)
            mesh_count = len(list((staged / "meshes").iterdir()))
            _publish(staged, target, replace=replace)
    return f"installed: {source.name} ({mesh_count} referenced meshes)"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        help="Robot Library root (default: the platform HHTools user library)",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=tuple(source.name for source in SOURCES),
        help="Install only this preset; repeat to select several",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace an existing preset from a different source",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.destination is None:
        from hhtools.utils.paths import user_robot_dir

        library = user_robot_dir().resolve()
    else:
        library = args.destination.expanduser().resolve()
    if library == Path(library.anchor):
        raise ValueError("the filesystem root cannot be a Robot Library")
    library.mkdir(parents=True, exist_ok=True)

    selected = set(args.only or (source.name for source in SOURCES))
    for source in SOURCES:
        if source.name in selected:
            print(_install(source, library, replace=args.replace), flush=True)
    print(f"Robot Library: {library}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
