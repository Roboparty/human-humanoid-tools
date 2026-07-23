"""Batch-retarget intermimic / human–object clips (interaction-mesh).

Supported ``--dataset`` values:

* ``omomo`` — ``<clip>/<clip>.pkl`` + object mesh sidecars (SMPL-X / ``smplx``)

Output matches Web export contents (uncompressed folders)::

    <out_root>/<clip>/
        <clip>.csv
        object_0_<name>.csv     # interaction object track (robot frame)
        <object_mesh>.obj       # centred mesh scaled to robot frame (when present)

Usage::

    python scripts/batch_intermimic_retarget.py \\
        --robot rp1 \\
        --dataset omomo \\
        --in ~/motions/OMOMO \\
        --out ~/motions/OMOMO_rp1 \\
        --limit 5

Each clip runs in a subprocess by default. Requires ``mujoco`` and ``osqp``.
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

_INTERMIMIC_DATASETS = ("omomo",)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(
        p,
        default_reference=None,
        default_backend="interaction_mesh",
        datasets=_INTERMIMIC_DATASETS,
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    def _resolve_reference(ns: argparse.Namespace) -> str:
        if ns.reference:
            return ns.reference
        return DATASET_TO_REFERENCE.get(ns.dataset, "smplx")

    return run_batch_main(
        args,
        script_path=Path(__file__).resolve(),
        scene_default=True,
        resolve_reference=_resolve_reference,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
