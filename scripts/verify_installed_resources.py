"""Load real numerical resources and compare source/installed reference results.

Run first in the source environment with --output, then with the installed
wheel's Python in isolated mode (-I), --installed and --compare.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sysconfig
from pathlib import Path

import numpy as np

import hhtools
from hhtools.io.npz import load_npz
from hhtools.retarget.calibration.reference import load_reference_pose
from hhtools.retarget.newton_basic.rest_pose import rest_pose_from_bundled_reference


def fingerprint(array: np.ndarray) -> dict:
    value = np.asarray(array)
    if not np.isfinite(value).all():
        raise ValueError("non-finite numerical resource")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    package = Path(hhtools.__file__).resolve().parent
    if args.installed and not package.is_relative_to(Path(sysconfig.get_path("purelib")).resolve()):
        raise SystemExit(f"expected installed wheel, imported source package: {package}")

    values = {}
    for name in ("soma_bvh", "lafan_bvh", "mocap_bvh", "xsens_mocap"):
        pose = load_reference_pose(name)
        if pose.fallback:
            raise SystemExit(f"required reference silently fell back: {name}")
        rest = rest_pose_from_bundled_reference(name)
        values[name] = {
            "names": list(pose.joint_names),
            "height": pose.height_m,
            "positions": fingerprint(pose.positions),
            "quaternions": fingerprint(pose.quaternions),
            "rest_positions": fingerprint(rest.positions),
            "rest_quaternions": fingerprint(rest.quaternions),
        }
    for relative in (
        "data/reference_poses/glb_zero_frame0.npz",
        "io/data/parc_ms_skeleton_reference.npz",
    ):
        motion = load_npz(package / relative)
        values[relative] = {
            "names": motion.bone_names,
            "fps": motion.framerate,
            "positions": fingerprint(motion.positions),
            "quaternions": fingerprint(motion.quaternions),
        }
    if args.compare and values != json.loads(args.compare.read_text(encoding="utf-8")):
        raise SystemExit("installed numerical resources differ from source results")
    if args.output:
        args.output.write_text(json.dumps(values, indent=2), encoding="utf-8")
    print(f"verified six numerical resources from {package}")


if __name__ == "__main__":
    main()
