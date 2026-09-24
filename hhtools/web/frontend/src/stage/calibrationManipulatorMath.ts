import type { StageVec3 } from "./types";

const EPSILON = 1e-8;

export interface CalibrationDragLimit {
  readonly lower: number;
  readonly upper: number;
}

export function resolvedCalibrationDragLimit(
  limit: { readonly lower?: number; readonly upper?: number },
): CalibrationDragLimit {
  const lower = Number.isFinite(limit.lower) ? limit.lower! : -Math.PI;
  const upper = Number.isFinite(limit.upper) ? limit.upper! : Math.PI;
  return upper > lower ? { lower, upper } : { lower: -Math.PI, upper: Math.PI };
}

function dot(left: StageVec3, right: StageVec3): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function subtract(left: StageVec3, right: StageVec3): StageVec3 {
  return [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
}

function normalize(value: StageVec3): StageVec3 | null {
  const length = Math.hypot(value[0], value[1], value[2]);
  return length > EPSILON
    ? [value[0] / length, value[1] / length, value[2] / length]
    : null;
}

/** Intersect a pointer ray with the joint's rotation plane. */
export function jointPlanePoint(
  rayOrigin: StageVec3,
  rayDirection: StageVec3,
  pivot: StageVec3,
  axis: StageVec3,
): StageVec3 | null {
  const denominator = dot(rayDirection, axis);
  if (Math.abs(denominator) <= EPSILON) return null;
  const distance = dot(subtract(pivot, rayOrigin), axis) / denominator;
  if (!Number.isFinite(distance)) return null;
  return [
    rayOrigin[0] + rayDirection[0] * distance,
    rayOrigin[1] + rayDirection[1] * distance,
    rayOrigin[2] + rayDirection[2] * distance,
  ];
}

/** Return a normalized pivot-to-pointer vector on the rotation plane. */
export function jointDragVector(
  point: StageVec3,
  pivot: StageVec3,
): StageVec3 | null {
  return normalize(subtract(point, pivot));
}

/** Signed angular displacement from start to current around a world-space axis. */
export function signedJointDragAngle(
  start: StageVec3,
  current: StageVec3,
  axis: StageVec3,
): number {
  const cross: StageVec3 = [
    start[1] * current[2] - start[2] * current[1],
    start[2] * current[0] - start[0] * current[2],
    start[0] * current[1] - start[1] * current[0],
  ];
  return Math.atan2(dot(axis, cross), dot(start, current));
}

export function calibrationDragValue(
  startValue: number,
  angle: number,
  limit: CalibrationDragLimit,
): number {
  const value = Number.isFinite(startValue + angle) ? startValue + angle : 0;
  return Math.min(limit.upper, Math.max(limit.lower, value));
}

/** Closest signed distance along an axis to a pointer ray. */
export function jointAxisParameter(
  rayOrigin: StageVec3,
  rayDirection: StageVec3,
  pivot: StageVec3,
  axis: StageVec3,
): number | null {
  const offset = subtract(rayOrigin, pivot);
  const directionAxis = dot(rayDirection, axis);
  const denominator = 1 - directionAxis * directionAxis;
  if (Math.abs(denominator) <= EPSILON) return null;
  const value = (
    dot(axis, offset) - directionAxis * dot(rayDirection, offset)
  ) / denominator;
  return Number.isFinite(value) ? value : null;
}
