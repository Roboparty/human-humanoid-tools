import type { CalibrationAngleUnit } from "@/stage/calibrationInteraction";

export interface CalibrationJointLimit {
  readonly name: string;
  readonly lower?: number;
  readonly upper?: number;
  readonly type?: string;
}

export interface ResolvedCalibrationJointLimit {
  readonly name: string;
  readonly lower: number;
  readonly upper: number;
  readonly type?: string;
}

export type CalibrationJointRegion =
  | "torso"
  | "left-arm"
  | "right-arm"
  | "left-leg"
  | "right-leg"
  | "head"
  | "hands"
  | "other";

const LEFT_TOKEN = /(^|[_\-.])(left|l)(?=[_\-.]|$)/;
const RIGHT_TOKEN = /(^|[_\-.])(right|r)(?=[_\-.]|$)/;

function hasAny(value: string, tokens: readonly string[]): boolean {
  return tokens.some((token) => value.includes(token));
}

export function classifyCalibrationJoint(name: string): CalibrationJointRegion {
  const normalized = name.trim().toLowerCase().replace(/\s+/g, "_");
  const left = normalized.startsWith("left") || LEFT_TOKEN.test(normalized);
  const right = normalized.startsWith("right") || RIGHT_TOKEN.test(normalized);
  if (hasAny(normalized, ["finger", "thumb", "hand", "gripper"])) return "hands";
  if (hasAny(normalized, ["head", "neck", "antenna"])) return "head";
  const arm = hasAny(normalized, ["shoulder", "elbow", "wrist", "arm"]);
  if (arm && left) return "left-arm";
  if (arm && right) return "right-arm";
  const leg = hasAny(normalized, ["hip", "knee", "ankle", "leg", "foot", "toe"]);
  if (leg && left) return "left-leg";
  if (leg && right) return "right-leg";
  if (hasAny(normalized, ["pelvis", "waist", "torso", "spine", "chest", "trunk", "root"])) {
    return "torso";
  }
  return "other";
}

export function calibrationJointMatches(
  name: string,
  query: string,
  region: CalibrationJointRegion | "all",
): boolean {
  const queryMatches =
    !query.trim() || name.toLowerCase().includes(query.trim().toLowerCase());
  return queryMatches && (region === "all" || classifyCalibrationJoint(name) === region);
}

export function angleForDisplay(
  valueRad: number,
  unit: CalibrationAngleUnit,
): number {
  return unit === "deg" ? valueRad * 180 / Math.PI : valueRad;
}

export function angleFromDisplay(
  value: number,
  unit: CalibrationAngleUnit,
): number {
  return unit === "deg" ? value * Math.PI / 180 : value;
}

export function formatCalibrationAngle(
  valueRad: number,
  unit: CalibrationAngleUnit,
): string {
  return angleForDisplay(valueRad, unit).toFixed(unit === "deg" ? 2 : 3);
}

export function resolveCalibrationJointLimits(
  limits: readonly CalibrationJointLimit[],
  values: Readonly<Record<string, number>> = {},
): readonly ResolvedCalibrationJointLimit[] {
  const resolved: ResolvedCalibrationJointLimit[] = [];
  const seen = new Set<string>();
  for (const candidate of [
    ...limits,
    ...Object.keys(values).map<CalibrationJointLimit>((name) => ({ name })),
  ]) {
    const name = candidate.name.trim();
    if (!name || seen.has(name)) continue;
    seen.add(name);
    let lower = Number.isFinite(candidate.lower) ? candidate.lower! : -Math.PI;
    let upper = Number.isFinite(candidate.upper) ? candidate.upper! : Math.PI;
    if (upper <= lower) [lower, upper] = [-Math.PI, Math.PI];
    resolved.push({
      name,
      lower,
      upper,
      ...(candidate.type ? { type: candidate.type } : {}),
    });
  }
  return resolved;
}

export function clampCalibrationValue(
  value: number,
  limit: ResolvedCalibrationJointLimit,
): number {
  const finite = Number.isFinite(value) ? value : 0;
  return Math.min(limit.upper, Math.max(limit.lower, finite));
}

export function normalizeCalibrationValues(
  limits: readonly CalibrationJointLimit[],
  values: Readonly<Record<string, number>> = {},
): Record<string, number> {
  return Object.fromEntries(
    resolveCalibrationJointLimits(limits, values).map((limit) => [
      limit.name,
      clampCalibrationValue(values[limit.name] ?? 0, limit),
    ]),
  );
}

export function setCalibrationJointValue(
  limits: readonly CalibrationJointLimit[],
  values: Readonly<Record<string, number>>,
  name: string,
  value: number,
): Record<string, number> {
  const limit = resolveCalibrationJointLimits(limits, values).find(
    (candidate) => candidate.name === name,
  );
  if (!limit) return { ...values };
  return { ...values, [name]: clampCalibrationValue(value, limit) };
}

export function zeroCalibrationValues(
  limits: readonly CalibrationJointLimit[],
  values: Readonly<Record<string, number>>,
): Record<string, number> {
  return Object.fromEntries(
    resolveCalibrationJointLimits(limits, values).map((limit) => [
      limit.name,
      clampCalibrationValue(0, limit),
    ]),
  );
}

export function zeroCalibrationRegion(
  limits: readonly CalibrationJointLimit[],
  values: Readonly<Record<string, number>>,
  region: CalibrationJointRegion | "all",
): Record<string, number> {
  const next = normalizeCalibrationValues(limits, values);
  for (const limit of resolveCalibrationJointLimits(limits, values)) {
    if (calibrationJointMatches(limit.name, "", region)) {
      next[limit.name] = clampCalibrationValue(0, limit);
    }
  }
  return next;
}

export function isNearCalibrationLimit(
  value: number,
  limit: ResolvedCalibrationJointLimit,
): boolean {
  const span = limit.upper - limit.lower;
  return (
    span > 0 &&
    (value - limit.lower < span * 0.03 || limit.upper - value < span * 0.03)
  );
}

export type { CalibrationAngleUnit } from "@/stage/calibrationInteraction";
