"""Verify release code, UI and numerical resources in the actual built wheel."""

from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parents[1]
NUMERICAL_RESOURCES = (
    "hhtools/data/reference_poses/soma_zero_frame0.bvh",
    "hhtools/data/reference_poses/lafan_zero_frame0.bvh",
    "hhtools/data/reference_poses/mocap_zero_frame0.bvh",
    "hhtools/data/reference_poses/xsens_mocap_zero_frame0.bvh",
    "hhtools/data/reference_poses/glb_zero_frame0.npz",
    "hhtools/io/data/parc_ms_skeleton_reference.npz",
)


def verify_wheel(wheel: Path, *, source_root: Path = REPO_ROOT) -> None:
    if not wheel.is_file():
        raise SystemExit(f"wheel not found: {wheel}")

    required = {
        "hhtools/web/static/index.html",
        "hhtools/viewer/anatomy.py",
        "hhtools/viewer/cache.py",
        "hhtools/viewer/library.py",
        "hhtools/io/isaaclab_trajectory.py",
        "hhtools/py.typed",
        *NUMERICAL_RESOURCES,
    }
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = sorted(required - names)
        if missing:
            raise SystemExit(f"wheel is missing required files: {', '.join(missing)}")
        # Existence alone would accept truncated or stale rest-pose data. These
        # bytes determine calibration and scale, so compare to the build input.
        for name in NUMERICAL_RESOURCES:
            source = source_root / name
            if not source.is_file():
                raise SystemExit(f"source numerical resource is missing: {name}")
            if archive.read(name) != source.read_bytes():
                raise SystemExit(f"wheel numerical resource differs from source: {name}")
    if not any(
        name.startswith("hhtools/web/static/assets/") and name.endswith(".js") for name in names
    ):
        raise SystemExit("wheel has no built Web renderer JavaScript")
    if not any(
        name.startswith("hhtools/web/static/assets/") and name.endswith(".css") for name in names
    ):
        raise SystemExit("wheel has no built Web renderer stylesheet")

    forbidden_files = {
        "hhtools/cli/ui.py",
        "hhtools/viewer/app.py",
        "hhtools/viewer/markdown_compat.py",
        "hhtools/viewer/theme.py",
    }
    forbidden_prefixes = (
        "hhtools/viewer/panels/",
        "hhtools/viewer/renderers/",
        "hhtools/web/frontend-old/",
    )
    retired = sorted(
        name for name in names if name in forbidden_files or name.startswith(forbidden_prefixes)
    )
    if retired:
        raise SystemExit(f"wheel contains retired sources: {', '.join(retired)}")

    print(f"verified {wheel.name}: {len(names)} files")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_wheel_contents.py DIST.whl")
    verify_wheel(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
