"""Batch-retarget mimic-family clips (flat mocap → Newton IK).

Supported ``--dataset`` values (registered adapters):

* ``amass`` — ``*.npz`` (default reference ``smpl``)
* ``lafan`` / ``mocap`` / ``soma`` / ``xsens_mocap`` — ``*.bvh``
* ``glb`` — ``*.glb`` / ``*.gltf``
* ``unified_npz`` — generic folder of motion ``*.npz``

Output matches Web export (flat CSV under ``out_root``, nested relatives preserved)::

    <out_root>/<optional/rel/dirs>/<stem>.csv

Usage::

    python scripts/batch_mimic_retarget.py \\
        --robot rp1 \\
        --dataset amass \\
        --in ~/motions/AMASS \\
        --out ~/motions/AMASS_rp1 \\
        --limit 5

    python scripts/batch_mimic_retarget.py \\
        --robot unitree_g1__g1_29dof \\
        --dataset lafan \\
        --in ~/motions/LAFAN \\
        --out ~/motions/LAFAN_g1 \\
        --skip-existing

Requires the NVIDIA ``newton`` package for the IK backend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _batch_retarget_common import (  # noqa: E402
    DATASET_TO_REFERENCE,
    add_common_args,
    run_batch_main,
)

_MIMIC_DATASETS = (
    "amass",
    "lafan",
    "mocap",
    "soma",
    "xsens_mocap",
    "glb",
    "unified_npz",
    "phuma",
    "motion_x",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(
        p,
        default_reference=None,
        default_backend="newton",
        datasets=_MIMIC_DATASETS,
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    def _resolve_reference(ns: argparse.Namespace) -> str:
        if ns.reference:
            return ns.reference
        return DATASET_TO_REFERENCE.get(ns.dataset, "smpl")

    return run_batch_main(
        args,
        script_path=Path(__file__).resolve(),
        scene_default=False,
        resolve_reference=_resolve_reference,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
