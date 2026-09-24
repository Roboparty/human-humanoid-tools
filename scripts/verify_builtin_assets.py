#!/usr/bin/env python3
"""Validate the release allowlist for bundled motions and curated robots.

The manifest is deliberately path-based. Release staging must copy only paths
returned by this module; walking ``assets/motions`` or a user's Robot Library
would risk including caches, downloads, generated jobs, or private calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

DEFAULT_MANIFEST = REPOSITORY_ROOT / "configs" / "builtin-assets.json"
EXPECTED_MOTION_ENTRIES = 30
EXPECTED_MOTION_FILES = 58
EXPECTED_ROBOTS = 6


class BuiltinAssetManifestError(ValueError):
    """Raised when the release allowlist is incomplete or unsafe."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BuiltinAssetManifestError(f"cannot read JSON manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise BuiltinAssetManifestError(f"manifest root must be an object: {path}")
    return payload


def _safe_relative(raw: object, *, prefix: str | None = None) -> str:
    value = str(raw or "").replace("\\", "/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise BuiltinAssetManifestError(f"unsafe manifest path: {value!r}")
    if prefix is not None and not value.startswith(prefix.rstrip("/") + "/"):
        raise BuiltinAssetManifestError(f"manifest path is outside {prefix}: {value}")
    return value


def _git_paths(repository: Path, revision: str) -> set[str] | None:
    if not (repository / ".git").exists():
        return None
    available = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=repository,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if available.returncode != 0:
        # CI normally uses a depth-one checkout. The current tracked tree is
        # still checked below, while a full clone additionally verifies the
        # historical baseline object.
        return None
    try:
        output = subprocess.check_output(
            [
                "git",
                "ls-tree",
                "-r",
                "-z",
                "--name-only",
                revision,
                "--",
                "assets/motions",
            ],
            cwd=repository,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise BuiltinAssetManifestError(
            f"cannot inspect baseline commit {revision}: {error}"
        ) from error
    return {
        item.decode("utf-8")
        for item in output.split(b"\0")
        if item and b"/.hhtools_analysis/" not in item
    }


def _git_index_paths(repository: Path) -> set[str] | None:
    if not (repository / ".git").exists():
        return None
    try:
        output = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "assets/motions"],
            cwd=repository,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise BuiltinAssetManifestError(f"cannot inspect tracked motion files: {error}") from error
    return {
        item.decode("utf-8")
        for item in output.split(b"\0")
        if item and b"/.hhtools_analysis/" not in item
    }


def motion_packaging_paths(payload: dict[str, Any]) -> tuple[str, ...]:
    """Return the exact repository-relative files allowed in a motion bundle."""

    motions = payload.get("motions")
    shared = payload.get("motion_shared_files")
    if not isinstance(motions, list) or not isinstance(shared, list):
        raise BuiltinAssetManifestError("motions and motion_shared_files must be arrays")
    paths: list[str] = []
    for motion in motions:
        if not isinstance(motion, dict) or not isinstance(motion.get("files"), list):
            raise BuiltinAssetManifestError("every motion must declare a files array")
        paths.extend(_safe_relative(path, prefix="assets/motions") for path in motion["files"])
    paths.extend(_safe_relative(path, prefix="assets/motions") for path in shared)
    if len(paths) != len(set(paths)):
        raise BuiltinAssetManifestError("motion files overlap across allowlist entries")
    return tuple(paths)


def _validate_motion_entries(payload: dict[str, Any], repository: Path) -> tuple[str, ...]:
    motions = payload.get("motions")
    if not isinstance(motions, list) or len(motions) != EXPECTED_MOTION_ENTRIES:
        raise BuiltinAssetManifestError(
            f"expected {EXPECTED_MOTION_ENTRIES} motion entries, got "
            f"{len(motions) if isinstance(motions, list) else 'invalid'}"
        )
    ids: set[str] = set()
    primaries: set[str] = set()
    for motion in motions:
        if not isinstance(motion, dict):
            raise BuiltinAssetManifestError("motion entries must be objects")
        motion_id = str(motion.get("id") or "")
        if not motion_id or motion_id in ids:
            raise BuiltinAssetManifestError(f"duplicate or empty motion id: {motion_id!r}")
        ids.add(motion_id)
        if motion.get("profile") not in {"mimic", "intermimic", "meshmimic"}:
            raise BuiltinAssetManifestError(f"invalid profile for {motion_id}")
        primary = _safe_relative(motion.get("primary"), prefix="assets/motions")
        files = motion.get("files")
        if not isinstance(files, list) or primary not in files:
            raise BuiltinAssetManifestError(f"{motion_id} does not include its primary file")
        if primary in primaries:
            raise BuiltinAssetManifestError(f"duplicate motion primary: {primary}")
        primaries.add(primary)

    paths = motion_packaging_paths(payload)
    if len(paths) != EXPECTED_MOTION_FILES:
        raise BuiltinAssetManifestError(
            f"expected {EXPECTED_MOTION_FILES} motion files, got {len(paths)}"
        )
    for relative in paths:
        path = repository / relative
        if path.is_symlink() or not path.is_file():
            raise BuiltinAssetManifestError(
                f"declared motion file is missing or symlinked: {relative}"
            )

    baseline = payload.get("baseline")
    if not isinstance(baseline, dict):
        raise BuiltinAssetManifestError("baseline must be an object")
    revision = str(baseline.get("commit") or "")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise BuiltinAssetManifestError("baseline.commit must be a full lowercase Git hash")
    exclusions = payload.get("baseline_exclusions")
    if not isinstance(exclusions, list):
        raise BuiltinAssetManifestError("baseline_exclusions must be an array")
    excluded_paths: set[str] = set()
    for exclusion in exclusions:
        if not isinstance(exclusion, dict) or not str(exclusion.get("reason") or "").strip():
            raise BuiltinAssetManifestError("every baseline exclusion needs a path and reason")
        excluded_paths.add(_safe_relative(exclusion.get("path"), prefix="assets/motions"))
    if set(paths) & excluded_paths:
        raise BuiltinAssetManifestError("excluded motion files cannot also be packaged")

    baseline_paths = _git_paths(repository, revision)
    reviewed_paths = set(paths) | excluded_paths
    if baseline_paths is not None and reviewed_paths != baseline_paths:
        missing = sorted(baseline_paths - reviewed_paths)
        extra = sorted(reviewed_paths - baseline_paths)
        raise BuiltinAssetManifestError(
            f"reviewed motion paths differ from baseline; missing={missing}, extra={extra}"
        )
    tracked_paths = _git_index_paths(repository)
    if tracked_paths is not None and reviewed_paths != tracked_paths:
        missing = sorted(tracked_paths - reviewed_paths)
        extra = sorted(reviewed_paths - tracked_paths)
        raise BuiltinAssetManifestError(
            f"reviewed motion paths differ from the Git index; missing={missing}, extra={extra}"
        )
    return paths


def _validate_robot_entries(payload: dict[str, Any]) -> tuple[str, ...]:
    from hhtools.application.robots import _BUILTIN_ROBOT_PRESET_NAMES
    from scripts.install_builtin_robots import SOURCES

    robots = payload.get("robots")
    if not isinstance(robots, list) or len(robots) != EXPECTED_ROBOTS:
        raise BuiltinAssetManifestError(
            f"expected {EXPECTED_ROBOTS} robots, got "
            f"{len(robots) if isinstance(robots, list) else 'invalid'}"
        )
    declared: dict[str, dict[str, Any]] = {}
    for robot in robots:
        if not isinstance(robot, dict):
            raise BuiltinAssetManifestError("robot entries must be objects")
        preset = str(robot.get("preset") or "")
        if not preset or preset in declared:
            raise BuiltinAssetManifestError(f"duplicate or empty robot preset: {preset!r}")
        declared[preset] = robot

    sources = {source.name: source for source in SOURCES}
    expected = set(_BUILTIN_ROBOT_PRESET_NAMES)
    if set(declared) != expected or set(declared) != set(sources):
        raise BuiltinAssetManifestError(
            "robot allowlist differs from runtime/installer definitions: "
            f"manifest={sorted(declared)}, runtime={sorted(expected)}, "
            f"installer={sorted(sources)}"
        )
    for preset, robot in declared.items():
        source = sources[preset]
        expected_fields = {
            "display_name": source.display_name,
            "repository": source.repository,
            "commit": source.commit,
            "main_urdf": source.main_urdf,
            "license_paths": list(source.license_paths),
            "bundle_manifest": "SOURCE.json",
        }
        for field, expected_value in expected_fields.items():
            if robot.get(field) != expected_value:
                raise BuiltinAssetManifestError(
                    f"{preset}.{field} differs from the pinned installer source"
                )
    return tuple(robot["preset"] for robot in robots)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_robot_bundles(payload: dict[str, Any], robot_root: Path) -> dict[str, int]:
    """Validate and count exact robot files allowed from an installed Robot Library."""

    files = 0
    total_bytes = 0
    for robot in payload["robots"]:
        preset = str(robot["preset"])
        root = robot_root / preset
        source_manifest = root / str(robot["bundle_manifest"])
        source = _read_json(source_manifest)
        upstream = source.get("source")
        if (
            source.get("preset") != preset
            or source.get("display_name") != robot["display_name"]
            or not isinstance(upstream, dict)
            or upstream.get("repository") != robot["repository"]
            or upstream.get("commit") != robot["commit"]
            or upstream.get("main_urdf") != robot["main_urdf"]
            or upstream.get("license_paths") != robot["license_paths"]
        ):
            raise BuiltinAssetManifestError(f"installed robot source differs: {preset}")
        installed = source.get("installed_files")
        if not isinstance(installed, list) or not installed:
            raise BuiltinAssetManifestError(f"installed robot has no file allowlist: {preset}")
        seen: set[str] = set()
        for record in installed:
            if not isinstance(record, dict):
                raise BuiltinAssetManifestError(f"invalid installed file record: {preset}")
            relative = _safe_relative(record.get("path"))
            if relative in seen:
                raise BuiltinAssetManifestError(
                    f"duplicate installed robot file: {preset}/{relative}"
                )
            seen.add(relative)
            path = root / relative
            expected_size = int(record.get("bytes", -1))
            expected_hash = str(record.get("sha256") or "")
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != expected_size
                or _sha256(path) != expected_hash
            ):
                raise BuiltinAssetManifestError(
                    f"installed robot file differs: {preset}/{relative}"
                )
            files += 1
            total_bytes += expected_size
        if "robot.yaml" not in seen:
            raise BuiltinAssetManifestError(f"installed robot has no robot.yaml: {preset}")
    return {"files": files, "bytes": total_bytes}


def validate_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    repository: Path = REPOSITORY_ROOT,
    robot_root: Path | None = None,
) -> dict[str, int]:
    payload = _read_json(manifest_path)
    if payload.get("schema_version") != 1:
        raise BuiltinAssetManifestError("unsupported builtin asset manifest schema")
    policy = payload.get("policy")
    if not isinstance(policy, dict) or policy.get("include_only_declared") is not True:
        raise BuiltinAssetManifestError("release policy must include only declared assets")
    motion_paths = _validate_motion_entries(payload, repository)
    robots = _validate_robot_entries(payload)
    summary = {
        "motion_entries": EXPECTED_MOTION_ENTRIES,
        "motion_files": len(motion_paths),
        "robots": len(robots),
    }
    if robot_root is not None:
        robot_summary = validate_robot_bundles(payload, robot_root.expanduser().resolve())
        summary["robot_files"] = robot_summary["files"]
        summary["robot_bytes"] = robot_summary["bytes"]
    return summary


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--robot-root", type=Path)
    parser.add_argument(
        "--print-motion-files",
        action="store_true",
        help="Print the exact motion paths a release staging step may copy",
    )
    parser.add_argument("--json", action="store_true", help="Print the validation summary as JSON")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    payload = _read_json(args.manifest)
    summary = validate_manifest(args.manifest, robot_root=args.robot_root)
    if args.print_motion_files:
        print("\n".join(motion_packaging_paths(payload)))
    elif args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(
            "verified builtin assets: "
            f"{summary['motion_entries']} motions / {summary['motion_files']} files, "
            f"{summary['robots']} robots"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
