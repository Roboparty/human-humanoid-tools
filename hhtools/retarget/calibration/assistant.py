"""Deterministic calibration proposals, validation, and visual previews."""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from hhtools.robot.kinematics import CRITICAL_IK_SLOTS, KinematicModel

from .calibration import _reference_pose_for_calibration

if TYPE_CHECKING:
    from hhtools.core.motion import Motion
    from hhtools.retarget.calibration.reference import HumanReferencePose
    from hhtools.robot.loader import URDFRobotModel

CALIBRATION_ALGORITHM = "hhtools.calibration.kinematic.v1"
R2R_CALIBRATION_ALGORITHM = "hhtools.r2r-calibration.kinematic.v1"
CALIBRATION_PREVIEW_WIDTH = 1200
CALIBRATION_PREVIEW_HEIGHT = 700

_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_ALIGNMENT_ERROR_DEG = 30.0
_ALIGNMENT_WARNING_DEG = 20.0
_PROPOSAL_TARGET_DEG = 15.0

_EDGE_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("hips_spine", "hips", "spine", "trunk"),
    ("spine_chest", "spine", "chest", "trunk"),
    ("chest_head", "chest", "head", "head"),
    ("left_upper_arm", "left_shoulder", "left_elbow", "left_arm"),
    ("left_forearm", "left_elbow", "left_wrist", "left_arm"),
    ("right_upper_arm", "right_shoulder", "right_elbow", "right_arm"),
    ("right_forearm", "right_elbow", "right_wrist", "right_arm"),
    ("left_thigh", "left_hip", "left_knee", "left_leg"),
    ("left_shin", "left_knee", "left_ankle", "left_leg"),
    ("right_thigh", "right_hip", "right_knee", "right_leg"),
    ("right_shin", "right_knee", "right_ankle", "right_leg"),
)
_REQUIRED_GROUPS = frozenset({"left_arm", "right_arm", "left_leg", "right_leg"})
_GROUP_ENDPOINTS = {
    "left_arm": ("left_shoulder", "left_wrist"),
    "right_arm": ("right_shoulder", "right_wrist"),
    "left_leg": ("left_hip", "left_ankle"),
    "right_leg": ("right_hip", "right_ankle"),
}


@dataclass(frozen=True, slots=True)
class CalibrationAssessment:
    valid: bool
    score: float
    mapped_slots: int
    missing_slots: tuple[str, ...]
    non_distal_targets: tuple[str, ...]
    unknown_joints: tuple[str, ...]
    missing_joints: tuple[str, ...]
    limit_violations: tuple[str, ...]
    near_limit_joints: tuple[str, ...]
    changed_joint_count: int
    edge_errors_deg: dict[str, float]
    unavailable_edges: tuple[str, ...]
    alignment_errors: tuple[str, ...]
    alignment_warnings: tuple[str, ...]
    symmetry_errors_deg: dict[str, float]
    foot_height_delta_m: float | None


@dataclass(frozen=True, slots=True)
class _GeometrySnapshot:
    robot_points: dict[str, NDArray[np.float64]]
    reference_points: dict[str, NDArray[np.float64]]
    robot_vectors: dict[str, NDArray[np.float64]]
    reference_vectors: dict[str, NDArray[np.float64]]


def _mapped_links(model: URDFRobotModel) -> dict[str, str]:
    links: dict[str, str] = {}
    for canonical, raw in (model.preset.ik_map or {}).items():
        if isinstance(raw, Mapping):
            target = raw.get("t_body") or raw.get("link") or raw.get("body")
        else:
            target = raw
        if isinstance(target, str) and target:
            links[str(canonical)] = target
    return links


def _reference_points(reference: HumanReferencePose) -> dict[str, NDArray[np.float64]]:
    result: dict[str, NDArray[np.float64]] = {}
    for index, native in enumerate(reference.joint_names):
        canonical = reference.source_to_canonical.get(native, native)
        result.setdefault(canonical, np.asarray(reference.positions[index], dtype=np.float64))
    return result


def _forward_from_shoulders(points: Mapping[str, NDArray[np.float64]]) -> NDArray | None:
    left = points.get("left_shoulder")
    right = points.get("right_shoulder")
    if left is None or right is None:
        left = points.get("left_hip")
        right = points.get("right_hip")
    if left is None or right is None:
        return None
    forward = np.cross(left - right, _WORLD_UP)
    forward[2] = 0.0
    norm = float(np.linalg.norm(forward))
    return None if norm < 1e-8 else forward / norm


def _rotate_z(points: Mapping[str, NDArray], radians: float) -> dict[str, NDArray]:
    cosine = math.cos(radians)
    sine = math.sin(radians)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return {name: rotation @ value for name, value in points.items()}


def _robot_points_at_q(
    model: URDFRobotModel,
    joint_q: Mapping[str, float],
    links: Mapping[str, str],
) -> dict[str, NDArray[np.float64]]:
    model.apply_configuration(dict(joint_q))
    available = set(model.link_names())
    points: dict[str, NDArray[np.float64]] = {}
    for canonical, link in links.items():
        if link not in available:
            continue
        try:
            transform = np.asarray(model.urdf.get_transform(link), dtype=np.float64)
        except Exception:  # noqa: BLE001 - one unreachable optional link is reported later
            continue
        if transform.shape == (4, 4) and np.isfinite(transform).all():
            points[canonical] = transform[:3, 3].copy()
    return points


def _aligned_geometry(
    model: URDFRobotModel,
    reference: HumanReferencePose,
    joint_q: Mapping[str, float],
) -> _GeometrySnapshot:
    links = _mapped_links(model)
    robot = _robot_points_at_q(model, joint_q, links)
    human = _reference_points(reference)
    robot_forward = _forward_from_shoulders(robot)
    human_forward = _forward_from_shoulders(human)
    if robot_forward is not None and human_forward is not None:
        robot_yaw = math.atan2(float(robot_forward[1]), float(robot_forward[0]))
        human_yaw = math.atan2(float(human_forward[1]), float(human_forward[0]))
        human = _rotate_z(human, robot_yaw - human_yaw)

    robot_vectors: dict[str, NDArray[np.float64]] = {}
    human_vectors: dict[str, NDArray[np.float64]] = {}
    for key, parent, child, _group in _EDGE_SPECS:
        robot_parent = robot.get(parent)
        robot_child = robot.get(child)
        human_parent = human.get(parent)
        human_child = human.get(child)
        if robot_parent is not None and robot_child is not None:
            vector = robot_child - robot_parent
            if float(np.linalg.norm(vector)) > 1e-7:
                robot_vectors[key] = vector
        if human_parent is not None and human_child is not None:
            vector = human_child - human_parent
            if float(np.linalg.norm(vector)) > 1e-7:
                human_vectors[key] = vector
    return _GeometrySnapshot(robot, human, robot_vectors, human_vectors)


def _angle_degrees(first: NDArray, second: NDArray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-8 or second_norm <= 1e-8:
        return 180.0
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _joint_bounds(model: URDFRobotModel) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for joint in model.actuated_joints:
        lower = float(joint.limit_lower) if joint.limit_lower is not None else -math.pi
        upper = float(joint.limit_upper) if joint.limit_upper is not None else math.pi
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            lower, upper = -math.pi, math.pi
        bounds[joint.name] = (lower, upper)
    return bounds


def _non_distal_targets(
    model: URDFRobotModel,
    links: Mapping[str, str],
) -> tuple[str, ...]:
    urdf_path = model.preset.urdf_path
    if urdf_path is None:
        return ()
    kinematics = KinematicModel.from_urdf(urdf_path)
    actuated = {joint.name for joint in model.actuated_joints}
    non_distal: list[str] = []
    for canonical in ("head", "left_wrist", "right_wrist", "left_ankle", "right_ankle"):
        target = links.get(canonical)
        if target is None:
            continue
        pending = list(kinematics.children_of.get(target, ()))
        seen: set[str] = set()
        while pending:
            child = pending.pop()
            if child in seen:
                continue
            seen.add(child)
            if kinematics.joint_for_child.get(child) in actuated:
                non_distal.append(canonical)
                break
            pending.extend(kinematics.children_of.get(child, ()))
    return tuple(non_distal)


def normalized_joint_q(
    model: URDFRobotModel,
    values: Mapping[str, float] | None,
    *,
    clamp: bool,
) -> tuple[dict[str, float], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return a full ordered joint map plus unknown/missing/limit diagnostics."""

    bounds = _joint_bounds(model)
    supplied = dict(values or {})
    unknown = tuple(sorted(set(supplied).difference(bounds)))
    missing = tuple(sorted(set(bounds).difference(supplied)))
    violations: list[str] = []
    result: dict[str, float] = {}
    for name, (lower, upper) in bounds.items():
        raw = supplied.get(name, 0.0)
        number = float(raw)
        if not math.isfinite(number):
            violations.append(name)
            number = 0.0
        if number < lower or number > upper:
            violations.append(name)
            if clamp:
                number = min(upper, max(lower, number))
        result[name] = number
    return result, unknown, missing, tuple(sorted(set(violations)))


def assess_calibration_pose(
    model: URDFRobotModel,
    reference: str,
    joint_q: Mapping[str, float],
    *,
    reference_motion: Motion | None = None,
    reference_pose: HumanReferencePose | None = None,
) -> CalibrationAssessment:
    """Evaluate hard limits and geometry without trusting a visual verdict."""

    normalized, unknown, missing, violations = normalized_joint_q(
        model,
        joint_q,
        clamp=False,
    )
    ref = reference_pose or _reference_pose_for_calibration(
        reference,
        motion=reference_motion,
    )
    try:
        geometry = _aligned_geometry(model, ref, normalized)
    finally:
        model.apply_configuration(model.zero_configuration())

    links = _mapped_links(model)
    available_links = set(model.link_names())
    valid_mappings = {
        canonical
        for canonical, link in links.items()
        if link in available_links and canonical in geometry.reference_points
    }
    missing_slots = tuple(sorted(CRITICAL_IK_SLOTS.difference(valid_mappings)))
    non_distal_targets = _non_distal_targets(model, links)
    edge_errors: dict[str, float] = {}
    unavailable: list[str] = []
    alignment_errors: list[str] = []
    alignment_warnings: list[str] = []
    for key, _parent, _child, group in _EDGE_SPECS:
        robot_vector = geometry.robot_vectors.get(key)
        human_vector = geometry.reference_vectors.get(key)
        if robot_vector is None or human_vector is None:
            unavailable.append(key)
            if group in _REQUIRED_GROUPS:
                alignment_errors.append(key)
            continue
        angle = _angle_degrees(robot_vector, human_vector)
        edge_errors[key] = round(angle, 6)
        if group in _REQUIRED_GROUPS and angle > _ALIGNMENT_ERROR_DEG:
            alignment_errors.append(key)
        elif group in _REQUIRED_GROUPS and angle > _ALIGNMENT_WARNING_DEG:
            alignment_warnings.append(key)

    symmetry: dict[str, float] = {}
    for label, left_key, right_key in (
        ("upper_arm", "left_upper_arm", "right_upper_arm"),
        ("forearm", "left_forearm", "right_forearm"),
        ("thigh", "left_thigh", "right_thigh"),
        ("shin", "left_shin", "right_shin"),
    ):
        left = geometry.robot_vectors.get(left_key)
        right = geometry.robot_vectors.get(right_key)
        if left is None or right is None:
            continue
        mirrored = np.asarray(right, dtype=np.float64).copy()
        mirrored[1] *= -1.0
        symmetry[label] = round(_angle_degrees(left, mirrored), 6)

    foot_delta: float | None = None
    left_foot = geometry.robot_points.get("left_ankle")
    right_foot = geometry.robot_points.get("right_ankle")
    if left_foot is not None and right_foot is not None:
        foot_delta = abs(float(left_foot[2] - right_foot[2]))

    bounds = _joint_bounds(model)
    near_limits = []
    for name, value in normalized.items():
        lower, upper = bounds[name]
        span = upper - lower
        if span > 0 and min(value - lower, upper - value) < span * 0.03:
            near_limits.append(name)

    changed = sum(abs(value) > 1e-4 for value in normalized.values())
    hard_failure = bool(
        unknown
        or missing
        or violations
        or missing_slots
        or alignment_errors
        or (foot_delta is not None and foot_delta > 0.08)
    )
    required_angles = [
        angle
        for key, angle in edge_errors.items()
        if next(spec[3] for spec in _EDGE_SPECS if spec[0] == key) in _REQUIRED_GROUPS
    ]
    mean_error = float(np.mean(required_angles)) if required_angles else 180.0
    score = max(0.0, min(1.0, 1.0 - mean_error / 90.0))
    return CalibrationAssessment(
        valid=not hard_failure,
        score=score,
        mapped_slots=len(valid_mappings),
        missing_slots=missing_slots,
        non_distal_targets=non_distal_targets,
        unknown_joints=unknown,
        missing_joints=missing,
        limit_violations=violations,
        near_limit_joints=tuple(near_limits),
        changed_joint_count=changed,
        edge_errors_deg=edge_errors,
        unavailable_edges=tuple(unavailable),
        alignment_errors=tuple(sorted(set(alignment_errors))),
        alignment_warnings=tuple(sorted(set(alignment_warnings))),
        symmetry_errors_deg=symmetry,
        foot_height_delta_m=foot_delta,
    )


def _chain_joint_names(
    model: URDFRobotModel,
    proximal_link: str,
    distal_link: str,
) -> list[str]:
    urdf_path = model.preset.urdf_path
    if urdf_path is None:
        return []
    kinematics = KinematicModel.from_urdf(urdf_path)
    actuated = {joint.name for joint in model.actuated_joints}
    result: list[str] = []
    current: str | None = distal_link
    seen: set[str] = set()
    while current is not None and current not in seen:
        seen.add(current)
        joint = kinematics.joint_for_child.get(current)
        if joint in actuated:
            result.append(joint)
        if current == proximal_link:
            return list(reversed(result))
        current = kinematics.parent_of.get(current)
    return []


def _solve_group(
    model: URDFRobotModel,
    reference: HumanReferencePose,
    joint_q: dict[str, float],
    *,
    group: str,
    locked_joints: frozenset[str],
) -> dict[str, float]:
    from scipy.optimize import minimize

    links = _mapped_links(model)
    proximal_name, distal_name = _GROUP_ENDPOINTS[group]
    proximal_link = links.get(proximal_name)
    distal_link = links.get(distal_name)
    if proximal_link is None or distal_link is None:
        return joint_q
    variable_names = [
        name
        for name in _chain_joint_names(model, proximal_link, distal_link)
        if name not in locked_joints
    ]
    if not variable_names:
        return joint_q

    baseline_geometry = _aligned_geometry(model, reference, joint_q)
    edge_keys = [spec[0] for spec in _EDGE_SPECS if spec[3] == group]
    targets = {
        key: baseline_geometry.reference_vectors[key]
        for key in edge_keys
        if key in baseline_geometry.reference_vectors
    }
    if len(targets) != len(edge_keys):
        return joint_q

    bounds_by_name = _joint_bounds(model)
    bounds = [bounds_by_name[name] for name in variable_names]
    spans = np.asarray([max(upper - lower, 0.25) for lower, upper in bounds])
    baseline = np.asarray([joint_q[name] for name in variable_names], dtype=np.float64)
    threshold = math.cos(math.radians(_PROPOSAL_TARGET_DEG))

    def candidate(values: NDArray) -> dict[str, float]:
        result = dict(joint_q)
        result.update(
            {name: float(value) for name, value in zip(variable_names, values, strict=True)}
        )
        return result

    def dots(values: NDArray) -> tuple[float, ...]:
        geometry = _aligned_geometry(model, reference, candidate(values))
        output = []
        for key in edge_keys:
            robot_vector = geometry.robot_vectors.get(key)
            target = targets[key]
            if robot_vector is None:
                output.append(-1.0)
                continue
            denominator = float(np.linalg.norm(robot_vector) * np.linalg.norm(target))
            output.append(
                -1.0
                if denominator <= 1e-10
                else float(np.clip(np.dot(robot_vector, target) / denominator, -1.0, 1.0))
            )
        return tuple(output)

    constraints = [
        {
            "type": "ineq",
            "fun": lambda values, index=index: dots(values)[index] - threshold,
        }
        for index in range(len(edge_keys))
    ]

    starts = [baseline]
    preferred_sign = 1.0 if group.startswith("left") else -1.0
    for amplitude in (0.85, 1.2, -1.0):
        proposal = baseline.copy()
        for index, name in enumerate(variable_names):
            if "shoulder_roll" in name or "hip_roll" in name:
                lower, upper = bounds[index]
                proposal[index] = min(upper, max(lower, preferred_sign * amplitude))
        starts.append(proposal)
    for elbow_value in (-0.6, 0.6):
        proposal = starts[2].copy()
        for index, name in enumerate(variable_names):
            if "elbow" in name or "knee" in name:
                lower, upper = bounds[index]
                proposal[index] = min(upper, max(lower, elbow_value))
        starts.append(proposal)

    # Deterministic interior starts keep unusual joint naming/topology from
    # making the zero-pose Jacobian the only path the optimizer sees.
    random = np.random.default_rng(
        int.from_bytes(hashlib.sha256(f"{model.preset.name}:{group}".encode()).digest()[:8])
    )
    for _ in range(2):
        starts.append(
            np.asarray(
                [
                    random.uniform(lower + 0.15 * (upper - lower), upper - 0.15 * (upper - lower))
                    for lower, upper in bounds
                ],
                dtype=np.float64,
            )
        )

    best_values = baseline
    best_score = math.inf
    for start in starts:
        interior = np.asarray(
            [
                min(upper - 1e-7, max(lower + 1e-7, value))
                for value, (lower, upper) in zip(start, bounds, strict=True)
            ],
            dtype=np.float64,
        )
        try:
            result = minimize(
                lambda values: float(np.sum(((values - baseline) / spans) ** 2)),
                interior,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 160, "ftol": 1e-9},
            )
            values = np.asarray(result.x, dtype=np.float64)
            similarities = dots(values)
        except Exception:  # noqa: BLE001 - retain the best deterministic fallback
            continue
        violation = sum(max(0.0, threshold - value) ** 2 for value in similarities)
        movement = float(np.sum(((values - baseline) / spans) ** 2))
        score = violation * 1_000_000.0 + movement
        if score < best_score:
            best_score = score
            best_values = values

    baseline_dots = dots(baseline)
    best_dots = dots(best_values)
    baseline_violation = sum(max(0.0, threshold - value) ** 2 for value in baseline_dots)
    best_violation = sum(max(0.0, threshold - value) ** 2 for value in best_dots)
    if best_violation >= baseline_violation - 1e-10:
        return joint_q
    return candidate(best_values)


def propose_calibration_pose(
    model: URDFRobotModel,
    reference: str,
    seed_joint_q: Mapping[str, float] | None = None,
    *,
    locked_joints: frozenset[str] = frozenset(),
    reference_motion: Motion | None = None,
    reference_pose: HumanReferencePose | None = None,
) -> tuple[dict[str, float], CalibrationAssessment]:
    """Find a small, limit-constrained pose adjustment for canonical chains."""

    joint_q, unknown, _missing, _violations = normalized_joint_q(
        model,
        seed_joint_q,
        clamp=True,
    )
    if unknown:
        raise ValueError("seed calibration contains joints outside the robot DOF order")
    unknown_locks = locked_joints.difference(joint_q)
    if unknown_locks:
        raise ValueError("locked calibration joints are outside the robot DOF order")
    ref = reference_pose or _reference_pose_for_calibration(
        reference,
        motion=reference_motion,
    )
    try:
        for group in ("left_arm", "right_arm", "left_leg", "right_leg"):
            geometry = _aligned_geometry(model, ref, joint_q)
            errors = [
                _angle_degrees(geometry.robot_vectors[key], geometry.reference_vectors[key])
                for key, _parent, _child, edge_group in _EDGE_SPECS
                if edge_group == group
                and key in geometry.robot_vectors
                and key in geometry.reference_vectors
            ]
            if errors and max(errors) > _PROPOSAL_TARGET_DEG:
                joint_q = _solve_group(
                    model,
                    ref,
                    joint_q,
                    group=group,
                    locked_joints=locked_joints,
                )
    finally:
        model.apply_configuration(model.zero_configuration())
    assessment = assess_calibration_pose(
        model,
        reference,
        joint_q,
        reference_motion=reference_motion,
        reference_pose=ref,
    )
    return joint_q, assessment


def _dashed_line(
    draw: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: str,
) -> None:
    delta = np.asarray(end) - np.asarray(start)
    length = float(np.linalg.norm(delta))
    if length <= 1.0:
        return
    direction = delta / length
    cursor = 0.0
    while cursor < length:
        dash_end = min(length, cursor + 6.0)
        first = np.asarray(start) + direction * cursor
        second = np.asarray(start) + direction * dash_end
        draw.line((*first.tolist(), *second.tolist()), fill=fill, width=1)
        cursor += 11.0


def render_calibration_preview_png(
    model: URDFRobotModel,
    reference: str,
    joint_q: Mapping[str, float],
    assessment: CalibrationAssessment,
    *,
    reference_motion: Motion | None = None,
    reference_pose: HumanReferencePose | None = None,
) -> bytes:
    """Render deterministic front/side landmark overlays for a vision model."""

    from PIL import Image, ImageDraw, ImageFont

    ref = reference_pose or _reference_pose_for_calibration(
        reference,
        motion=reference_motion,
    )
    try:
        geometry = _aligned_geometry(model, ref, joint_q)
    finally:
        model.apply_configuration(model.zero_configuration())
    common = sorted(set(geometry.robot_points).intersection(geometry.reference_points))
    root = "hips" if "hips" in common else (common[0] if common else None)
    if root is None:
        raise ValueError("calibration preview has no common mapped landmarks")
    robot_root = geometry.robot_points[root]
    human_root = geometry.reference_points[root]
    robot = {name: point - robot_root for name, point in geometry.robot_points.items()}
    human = {name: point - human_root for name, point in geometry.reference_points.items()}
    ratios = [
        float(np.linalg.norm(robot[name]) / np.linalg.norm(human[name]))
        for name in common
        if float(np.linalg.norm(human[name])) > 1e-6
        and float(np.linalg.norm(robot[name])) > 1e-6
    ]
    scale = float(np.median(ratios)) if ratios else 1.0
    human = {name: point * scale for name, point in human.items()}

    forward = _forward_from_shoulders(robot)
    if forward is not None:
        yaw = math.atan2(float(forward[1]), float(forward[0]))
        robot = _rotate_z(robot, -yaw)
        human = _rotate_z(human, -yaw)

    image = Image.new("RGB", (CALIBRATION_PREVIEW_WIDTH, CALIBRATION_PREVIEW_HEIGHT), "#10151d")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (28, 20),
        f"{model.preset.display_name}  |  {reference}  |  score {assessment.score:.3f}  |  "
        f"{'VALID' if assessment.valid else 'NEEDS ADJUSTMENT'}",
        fill="#f4f7fb",
        font=font,
    )
    draw.text(
        (28, 42),
        "Blue = reference   Orange = robot   Dashed = semantic correspondence",
        fill="#9eabc0",
        font=font,
    )

    panel_width = (CALIBRATION_PREVIEW_WIDTH - 84) // 2
    panel_height = CALIBRATION_PREVIEW_HEIGHT - 110
    for panel_index, (title, horizontal_axis) in enumerate((("FRONT", 1), ("SIDE", 0))):
        left = 28 + panel_index * (panel_width + 28)
        top = 76
        right = left + panel_width
        bottom = top + panel_height
        draw.rounded_rectangle((left, top, right, bottom), radius=12, outline="#2d3748", width=2)
        draw.text((left + 14, top + 12), title, fill="#c8d2e2", font=font)
        values = [
            (point[horizontal_axis], point[2])
            for points in (robot, human)
            for point in points.values()
        ]
        minimum_x = min(value[0] for value in values)
        maximum_x = max(value[0] for value in values)
        minimum_z = min(value[1] for value in values)
        maximum_z = max(value[1] for value in values)
        span = max(maximum_x - minimum_x, maximum_z - minimum_z, 0.2)
        center_x = 0.5 * (minimum_x + maximum_x)
        center_z = 0.5 * (minimum_z + maximum_z)
        pixels_per_unit = 0.78 * min(panel_width, panel_height) / span

        def project(
            point: NDArray,
            panel_left: float = left,
            panel_top: float = top,
            axis: int = horizontal_axis,
            horizontal_center: float = center_x,
            vertical_center: float = center_z,
            scale_value: float = pixels_per_unit,
        ) -> tuple[float, float]:
            return (
                panel_left
                + panel_width / 2
                + (float(point[axis]) - horizontal_center) * scale_value,
                panel_top
                + panel_height / 2
                - (float(point[2]) - vertical_center) * scale_value,
            )

        for _key, parent, child, _group in _EDGE_SPECS:
            if parent in human and child in human:
                draw.line(
                    (*project(human[parent]), *project(human[child])),
                    fill="#4ba9ff",
                    width=4,
                )
            if parent in robot and child in robot:
                draw.line(
                    (*project(robot[parent]), *project(robot[child])),
                    fill="#f5a623",
                    width=5,
                )
        for name in common:
            reference_point = project(human[name])
            robot_point = project(robot[name])
            _dashed_line(draw, reference_point, robot_point, "#65748b")
            draw.ellipse(
                (
                    reference_point[0] - 4,
                    reference_point[1] - 4,
                    reference_point[0] + 4,
                    reference_point[1] + 4,
                ),
                fill="#4ba9ff",
            )
            draw.ellipse(
                (
                    robot_point[0] - 5,
                    robot_point[1] - 5,
                    robot_point[0] + 5,
                    robot_point[1] + 5,
                ),
                fill="#f5a623",
            )
        for key in assessment.alignment_errors:
            spec = next((item for item in _EDGE_SPECS if item[0] == key), None)
            if spec is None or spec[2] not in robot:
                continue
            point = project(robot[spec[2]])
            draw.text(
                (point[0] + 7, point[1] - 7),
                f"{key}: {assessment.edge_errors_deg.get(key, 180.0):.1f} deg",
                fill="#ff6b6b",
                font=font,
            )

    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


__all__ = [
    "CALIBRATION_ALGORITHM",
    "CALIBRATION_PREVIEW_HEIGHT",
    "CALIBRATION_PREVIEW_WIDTH",
    "R2R_CALIBRATION_ALGORITHM",
    "CalibrationAssessment",
    "assess_calibration_pose",
    "normalized_joint_q",
    "propose_calibration_pose",
    "render_calibration_preview_png",
]
