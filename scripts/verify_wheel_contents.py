"""Verify that a built wheel contains the supported UI and excludes retired sources."""

from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZipFile


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_wheel_contents.py DIST.whl")
    wheel = Path(sys.argv[1])
    if not wheel.is_file():
        raise SystemExit(f"wheel not found: {wheel}")

    with ZipFile(wheel) as archive:
        names = set(archive.namelist())

    required = {
        "hhtools/web/static/index.html",
        "hhtools/viewer/anatomy.py",
        "hhtools/viewer/cache.py",
        "hhtools/viewer/library.py",
    }
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"wheel is missing required files: {', '.join(missing)}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
