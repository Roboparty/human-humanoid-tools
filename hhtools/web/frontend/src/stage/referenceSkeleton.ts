import type {
  StageMotionPayload,
  StageQuaternion,
  StageRobotPayload,
  StageVec3,
} from "./types";
import { REFERENCE_SKELETON_VISUAL } from "./visualStyle.ts";

export interface ReferenceSkeletonEdge {
  readonly child: number;
  readonly parent: number;
}

export interface ReferenceJointMapping {
  readonly semantic: string;
  readonly targetLink: string;
  readonly index: number;
  readonly quaternion: StageQuaternion | null;
}

export interface PreparedReferenceSkeleton {
  readonly frame: readonly StageVec3[];
  readonly edges: readonly ReferenceSkeletonEdge[];
  readonly excluded: ReadonlySet<number>;
  readonly mappings: readonly ReferenceJointMapping[];
  readonly mappingByJoint: ReadonlyMap<number, ReferenceJointMapping>;
  readonly color: number;
}

export function normalizeReferenceSemantic(value: unknown): string {
  return String(value ?? "").trim().toLowerCase().replace(/[^a-z0-9]/g, "");
}

export function referenceTargetLink(value: unknown): string | null {
  if (typeof value === "string") return value || null;
  if (!value || typeof value !== "object") return null;
  const candidate = value as Readonly<Record<string, unknown>>;
  for (const key of ["t_body", "link", "body", "target"] as const) {
    if (typeof candidate[key] === "string" && candidate[key]) {
      return candidate[key];
    }
  }
  return null;
}

function referenceQuaternion(
  reference: StageMotionPayload,
  index: number,
): StageQuaternion | null {
  const value = reference.quaternions?.[0]?.[index];
  if (!value || value.length < 4 || value.some((item) => !Number.isFinite(item))) {
    return null;
  }
  return [value[0], value[1], value[2], value[3]];
}

/** Resolve the backend reference and target IK map without creating Three objects. */
export function prepareReferenceSkeleton(
  reference: StageMotionPayload,
  robot: StageRobotPayload | null,
): PreparedReferenceSkeleton {
  const frame = reference.positions[0] ?? [];
  const excluded = new Set(
    (reference.exclude_joint_indices ?? []).filter(
      (index) => Number.isInteger(index) && index >= 0 && index < frame.length,
    ),
  );
  const edges = reference.parent_indices.flatMap((parent, child) =>
    parent >= 0 &&
    parent < frame.length &&
    child < frame.length &&
    !excluded.has(child) &&
    !excluded.has(parent)
      ? [{ child, parent }]
      : [],
  );

  const canonicalIndex = new Map<string, number>();
  reference.parent_indices.forEach((_, index) => {
    const boneName = reference.bone_names?.[index] ?? `joint_${index}`;
    const name = reference.canonical_names?.[index] ?? boneName;
    canonicalIndex.set(normalizeReferenceSemantic(name), index);
  });
  reference.parent_indices.forEach((_, index) => {
    const key = normalizeReferenceSemantic(
      reference.bone_names?.[index] ?? `joint_${index}`,
    );
    if (!canonicalIndex.has(key)) canonicalIndex.set(key, index);
  });

  const mappings: ReferenceJointMapping[] = [];
  for (const [semantic, rawTarget] of Object.entries(robot?.ik_map ?? {})) {
    const targetLink = referenceTargetLink(rawTarget);
    const index = canonicalIndex.get(normalizeReferenceSemantic(semantic));
    if (!targetLink || index == null || excluded.has(index)) continue;
    mappings.push({
      semantic,
      targetLink,
      index,
      quaternion: referenceQuaternion(reference, index),
    });
  }

  return {
    frame,
    edges,
    excluded,
    mappings,
    mappingByJoint: new Map(mappings.map((mapping) => [mapping.index, mapping])),
    color:
      typeof reference.color === "number" && Number.isFinite(reference.color)
        ? reference.color
        : REFERENCE_SKELETON_VISUAL.color,
  };
}
