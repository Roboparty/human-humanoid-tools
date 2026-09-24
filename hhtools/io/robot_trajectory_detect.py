"""Lightweight identification of robot trajectory export files."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROBOT_TRAJECTORY_EXTENSIONS = (".csv", ".pkl", ".npz")


def trajectory_robot_name(path: Path) -> str | None:
    """Return an embedded source-robot id when the trajectory declares one.

    Canonical hhtools CSV files carry ``# robot: <preset>`` before the column
    header.  Reading only that comment prefix keeps library discovery cheap and
    lets R2R hide trajectories known to belong to a different source robot.
    Foreign files without identity metadata remain selectable for compatibility.
    """

    path = Path(path)
    if path.suffix.casefold() != ".csv" or not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as stream:
            for _index, line in zip(range(64), stream, strict=False):
                stripped = line.strip()
                if not stripped:
                    continue
                if not stripped.startswith("#"):
                    break
                key, separator, value = stripped.lstrip("#").strip().partition(":")
                if separator and key.strip().casefold() in {"robot", "source_robot"}:
                    robot = value.strip()
                    return robot or None
    except (OSError, UnicodeDecodeError):
        return None
    return None


def sniff_robot_csv(path: Path) -> bool:  # noqa: PLR0911 - each format exit is explicit
    """Return whether a CSV looks like a robot trajectory rather than an object track."""

    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    lowered = stripped.lower()
                    if "frame:" in lowered and "retarget_robot" in lowered:
                        return False
                    continue
                raw_columns = [column.strip() for column in stripped.split(",")]
                columns = [column.lower() for column in raw_columns]
                if "pos_x" in columns and "root_x" not in columns:
                    return False
                if "root_x" in columns or any(column.startswith("dof_") for column in columns):
                    return True
                try:
                    numbers = [float(column) for column in raw_columns]
                except ValueError:
                    return False
                return len(numbers) >= 8
    except (OSError, UnicodeDecodeError):
        return False
    return False


def joint_q_width_from_pkl(path: Path) -> int:  # noqa: PLR0911 - defensive format sniffer
    """Return joint-q column count for an hhtools robot-export pickle, else zero."""

    if path.name.lower() == "terrain.pkl":
        return 0
    try:
        with path.open("rb") as stream:
            blob = pickle.load(stream)
    except Exception:  # noqa: BLE001 - format sniffer treats malformed inputs as non-matches
        return 0
    if not isinstance(blob, dict):
        return 0
    if "motion_data" in blob and not blob.get("hhtools_export"):
        robot = blob.get("robot")
        if not (isinstance(robot, dict) and "joint_q" in robot):
            return 0
    robot = blob.get("robot", blob)
    if not isinstance(robot, dict) or "joint_q" not in robot:
        return 0
    try:
        joint_q = np.asarray(robot["joint_q"])
    except Exception:  # noqa: BLE001 - format sniffer treats malformed inputs as non-matches
        return 0
    if joint_q.ndim != 2 or joint_q.shape[1] < 8:
        return 0
    return int(joint_q.shape[1])


def joint_q_width_from_npz(path: Path) -> int:
    """Return joint-q column count for an hhtools robot-export NPZ, else zero."""

    try:
        archive = np.load(path, allow_pickle=False)
    except Exception:  # noqa: BLE001 - format sniffer treats malformed inputs as non-matches
        return 0
    try:
        with archive:
            keys = set(archive.files)
            joint_q_key = next(
                (key for key in ("joint_q", "qpos", "q") if key in keys),
                None,
            )
            if joint_q_key is None:
                return 0
            joint_q = np.asarray(archive[joint_q_key])
            if joint_q.ndim != 2 or joint_q.shape[1] < 8:
                return 0
            return int(joint_q.shape[1])
    except Exception:  # noqa: BLE001 - format sniffer treats malformed inputs as non-matches
        return 0


def is_human_sidecar_csv(path: Path) -> bool:
    """Return whether a CSV name is reserved for a human/object sidecar."""

    name = path.name.lower()
    return name == "motion_actor.csv" or (
        name.endswith(".csv") and name.startswith(("prop_", "object_"))
    )


def is_robot_export_trajectory(path: Path) -> bool:
    """Return whether a file has the structure of a supported robot trajectory export."""

    if not path.is_file():
        return False
    extension = path.suffix.lower()
    if extension not in ROBOT_TRAJECTORY_EXTENSIONS:
        return False
    if extension == ".csv":
        return not is_human_sidecar_csv(path) and sniff_robot_csv(path)
    if extension == ".pkl":
        return joint_q_width_from_pkl(path) > 0
    if extension == ".npz":
        return joint_q_width_from_npz(path) > 0
    return False


# Private compatibility aliases used by the existing service implementation.
_ROBOT_TRAJ_EXTS = ROBOT_TRAJECTORY_EXTENSIONS
_is_robot_export_trajectory = is_robot_export_trajectory
_joint_q_width_from_npz = joint_q_width_from_npz
_joint_q_width_from_pkl = joint_q_width_from_pkl
_sniff_robot_csv = sniff_robot_csv


__all__ = [
    "ROBOT_TRAJECTORY_EXTENSIONS",
    "is_robot_export_trajectory",
    "joint_q_width_from_npz",
    "joint_q_width_from_pkl",
    "sniff_robot_csv",
    "trajectory_robot_name",
]
