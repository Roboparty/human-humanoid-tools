"""Fault injection for release resources that source imports cannot validate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "verify_wheel_contents", REPO / "scripts" / "verify_wheel_contents.py"
)
assert spec is not None and spec.loader is not None
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def _wheel(path: Path, *, omit: str = "", corrupt: str = "") -> Path:
    files = {
        "hhtools/web/static/index.html": b"<html></html>",
        "hhtools/web/static/assets/app.js": b"export {}",
        "hhtools/web/static/assets/app.css": b"body {}",
        "hhtools/viewer/anatomy.py": b"",
        "hhtools/viewer/cache.py": b"",
        "hhtools/viewer/library.py": b"",
        "hhtools/io/isaaclab_trajectory.py": b"",
        "hhtools/py.typed": b"",
        **{name: (REPO / name).read_bytes() for name in verifier.NUMERICAL_RESOURCES},
    }
    with ZipFile(path, "w") as archive:
        for name, data in files.items():
            if name != omit:
                archive.writestr(name, b"stale or truncated data" if name == corrupt else data)
    return path


def test_accepts_matching_numerical_resources(tmp_path: Path) -> None:
    verifier.verify_wheel(_wheel(tmp_path / "complete.whl"))


@pytest.mark.parametrize("name", verifier.NUMERICAL_RESOURCES)
def test_rejects_missing_numerical_resource(tmp_path: Path, name: str) -> None:
    with pytest.raises(SystemExit, match="missing required files"):
        verifier.verify_wheel(_wheel(tmp_path / "missing.whl", omit=name))


@pytest.mark.parametrize("name", verifier.NUMERICAL_RESOURCES)
def test_rejects_changed_numerical_resource(tmp_path: Path, name: str) -> None:
    with pytest.raises(SystemExit, match="differs from source"):
        verifier.verify_wheel(_wheel(tmp_path / "corrupt.whl", corrupt=name))
