"""Batch-retarget meshmimic family clips (terrain + interaction-mesh).

Supported ``--dataset`` values:

* ``parc_ms`` — ``<clip>/<clip>.pkl`` + ``<clip>_terrain.obj`` (SMPL / ``smpl``)
* ``holosoma`` — ``<clip>/<clip>.npy`` + ``terrain.obj`` (SMPL-X / ``smplx``)

Output matches Web export contents (uncompressed folders for large batches)::

    <out_root>/<clip>/
        <clip>.csv              # robot trajectory (header + root_z bake)
        <clip>_terrain.obj      # terrain in robot frame (when present)

Usage::

    python scripts/batch_meshmimic_retarget.py \\
        --robot rp1 \\
        --dataset parc_ms \\
        --in ~/motions/parc_ms \\
        --out ~/motions/parc_ms_rp1 \\
        --limit 5

    python scripts/batch_meshmimic_retarget.py \\
        --robot rp1 \\
        --dataset holosoma \\
        --in ~/motions/holosoma \\
        --out ~/motions/holosoma_rp1 \\
        --skip-existing

Each clip runs in a subprocess by default so native crashes skip that clip.
Requires ``mujoco`` and ``osqp``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python scripts/...`` without installing the package on PYTHONPATH.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _batch_retarget_common import (  # noqa: E402
    DATASET_TO_REFERENCE,
    add_common_args,
    run_batch_main,
)

# CLI aliases → registered adapter names.
_DATASET_ALIASES: dict[str, str] = {
    "parc_ms": "parc_ms",
    "holosoma": "meshmimic_holosoma",
    "meshmimic_holosoma": "meshmimic_holosoma",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(
        p,
        default_reference=None,
        default_backend="interaction_mesh",
        datasets=tuple(_DATASET_ALIASES),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.dataset = _DATASET_ALIASES[args.dataset]

    def _resolve_reference(ns: argparse.Namespace) -> str:
        if ns.reference:
            return ns.reference
        return DATASET_TO_REFERENCE.get(ns.dataset, "smpl")

    return run_batch_main(
        args,
        script_path=Path(__file__).resolve(),
        scene_default=True,
        resolve_reference=_resolve_reference,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
