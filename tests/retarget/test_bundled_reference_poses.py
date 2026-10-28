"""Bundled zero-frame reference BVHs must ship inside the hhtools package.

Regression for issue 13: the desktop installer installs hhtools from a wheel,
and when these files lived outside the package the lookup silently returned
None, so LAFAN calibration fell back to a synthetic bind rest.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
import tomllib

import pytest

from hhtools.retarget.newton_basic.rest_pose import (
    _BUNDLED_REFERENCE_BVH,
    bundled_reference_bvh_path,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("reference", sorted(_BUNDLED_REFERENCE_BVH))
def test_bundled_reference_resolves_inside_package(reference: str) -> None:
    path = bundled_reference_bvh_path(reference)
    assert path is not None, reference
    pkg_root = Path(str(resources.files("hhtools"))).resolve()
    assert pkg_root in path.parents


def test_reference_poses_declared_as_package_data() -> None:
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())
    globs = cfg["tool"]["setuptools"]["package-data"]["hhtools"]
    pkg = REPO / "hhtools"
    shipped = {p for g in globs for p in pkg.glob(g)}
    for name in _BUNDLED_REFERENCE_BVH.values():
        assert pkg / "data" / "reference_poses" / name in shipped, name


def test_lafan_calibration_reference_uses_bundled_bvh() -> None:
    from hhtools.retarget.calibration import reference as ref

    ref._lafan17_from_zero_bvh.cache_clear()
    assert ref._lafan17_from_zero_bvh() is not None


@pytest.mark.parametrize(
    ("reference", "loader"),
    [
        ("soma_bvh", "_soma17_from_zero_bvh"),
        ("lafan_bvh", "_lafan17_from_zero_bvh"),
        ("mocap_bvh", "_mocap17_from_zero_bvh"),
        ("xsens_mocap", "_xsens17_from_zero_bvh"),
    ],
)
@pytest.mark.parametrize("damaged", [False, True], ids=["missing", "corrupt"])
def test_required_bvh_reference_never_silently_uses_synthetic_pose(
    tmp_path: Path, monkeypatch, reference: str, loader: str, damaged: bool,
) -> None:
    from hhtools.retarget.calibration import reference as ref
    from hhtools.retarget.newton_basic import rest_pose

    if damaged:
        (tmp_path / _BUNDLED_REFERENCE_BVH[reference]).write_text("truncated BVH\n")
    monkeypatch.setattr(rest_pose, "_reference_poses_dir", lambda: tmp_path)
    cached_loader = getattr(ref, loader)
    cached_loader.cache_clear()
    try:
        message = "expected 'HIERARCHY'" if damaged else "Bundled reference pose .* missing or invalid"
        with pytest.raises(ValueError, match=message):
            ref.load_reference_pose(reference)
    finally:
        cached_loader.cache_clear()
