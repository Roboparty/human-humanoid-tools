"""Arm geometry helpers for retarget (shoulder→wrist reach)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from hhtools.robot.loader import URDFRobotModel

_log = logging.getLogger(__name__)

__all__ = ["estimate_shoulder_to_wrist_reach"]

Side = Literal["left", "right"]


def _ik_shoulder_wrist_links(model: "URDFRobotModel", side: Side) -> tuple[str, str] | None:
    ik = dict(model.preset.ik_map or {})
    shoulder = ik.get(f"{side}_shoulder")
    wrist = ik.get(f"{side}_wrist")
    if not shoulder or not wrist:
        return None
    return str(shoulder), str(wrist)


def _joint_origin_translation(joint) -> np.ndarray:
    origin = getattr(joint, "origin", None)
    if origin is None:
        return np.zeros(3, dtype=np.float64)
    mat = np.asarray(origin, dtype=np.float64)
    if mat.shape == (4, 4):
        return mat[:3, 3].copy()
    if mat.size >= 3:
        return mat.reshape(-1)[:3].copy()
    return np.zeros(3, dtype=np.float64)


def _static_chain_length(
    model: "URDFRobotModel",
    shoulder_link: str,
    wrist_link: str,
) -> float | None:
    """Sum of URDF joint-origin segment lengths along shoulder → wrist."""
    joint_by_child = {j.child: j for j in model.urdf.robot.joints}
    cur = wrist_link
    total = 0.0
    visited = 0
    while cur != shoulder_link:
        joint = joint_by_child.get(cur)
        if joint is None:
            return None
        total += float(np.linalg.norm(_joint_origin_translation(joint)))
        cur = str(joint.parent)
        visited += 1
        if visited > 64:
            return None
    return total


def _fk_link_distance(
    model: "URDFRobotModel",
    shoulder_link: str,
    wrist_link: str,
    *,
    joint_q: dict[str, float] | None,
) -> float | None:
    q = dict(joint_q) if joint_q is not None else model.zero_configuration()
    saved = model.zero_configuration()
    try:
        model.apply_configuration(q)
        t_sh = np.asarray(model.urdf.get_transform(shoulder_link), dtype=np.float64)
        t_wr = np.asarray(model.urdf.get_transform(wrist_link), dtype=np.float64)
    except Exception:
        return None
    finally:
        model.apply_configuration(saved)
    return float(np.linalg.norm(t_sh[:3, 3] - t_wr[:3, 3]))


def estimate_shoulder_to_wrist_reach(
    model: "URDFRobotModel",
    *,
    side: Side,
    joint_q: dict[str, float] | None = None,
    margin_m: float = 0.0,
) -> float | None:
    """Maximum shoulder→wrist reach (metres) from URDF chain geometry.

    Uses the longer of:

    * static sum of joint-origin segment lengths along the kinematic chain
      (fully extended upper bound), and
    * FK distance at ``joint_q`` (defaults to zero / T-pose).

    Used for :class:`~hhtools.retarget.newton_basic.config.ArmChainConfig.max_reach`
    in hand-ground-contact pre-IK constraints.
    """
    links = _ik_shoulder_wrist_links(model, side)
    if links is None:
        return None
    shoulder_link, wrist_link = links

    static_len = _static_chain_length(model, shoulder_link, wrist_link)
    fk_len = _fk_link_distance(model, shoulder_link, wrist_link, joint_q=joint_q)

    candidates = [v for v in (static_len, fk_len) if v is not None and v > 1e-4]
    if not candidates:
        _log.debug(
            "shoulder→wrist reach failed for %r side=%s",
            model.preset.name,
            side,
        )
        return None
    return float(max(candidates) + float(margin_m))


def infer_side_from_shoulder_name(name: str) -> Side | None:
    """Map LAFAN / canonical shoulder effector names to ``left`` / ``right``."""
    low = str(name).lower().replace("_", "")
    if low.startswith("left") or low.startswith("larm") or low == "lshoulder":
        return "left"
    if low.startswith("right") or low.startswith("rarm") or low == "rshoulder":
        return "right"
    if "left" in low:
        return "left"
    if "right" in low:
        return "right"
    return None
