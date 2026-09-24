import * as THREE from "three";

import type { StageMotionPayload, StageVec3 } from "./types";
import { CAPSULE_BODY_VISUAL } from "./visualStyle.ts";

interface PrimitiveGeometry {
  readonly vertices: readonly StageVec3[];
  readonly faces: readonly (readonly [number, number, number])[];
}

export interface CapsuleBodyResource {
  readonly mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;
  readonly edges: readonly (readonly [number, number])[];
  readonly visibleJoints: readonly number[];
  readonly positions: Float32Array;
}

const SEGMENTS = 6;
function unitCylinder(segments: number): PrimitiveGeometry {
  const vertices: StageVec3[] = [];
  for (let ring = 0; ring < 2; ring += 1) {
    for (let index = 0; index < segments; index += 1) {
      const angle = (index / segments) * Math.PI * 2;
      vertices.push([Math.cos(angle), Math.sin(angle), ring]);
    }
  }
  const faces: Array<readonly [number, number, number]> = [];
  for (let index = 0; index < segments; index += 1) {
    const next = (index + 1) % segments;
    faces.push(
      [index, next, index + segments],
      [next, next + segments, index + segments],
    );
  }
  return { vertices, faces };
}

function unitIcosphere(): PrimitiveGeometry {
  const ratio = (1 + Math.sqrt(5)) / 2;
  const vertices = [
    [-1, ratio, 0], [1, ratio, 0], [-1, -ratio, 0], [1, -ratio, 0],
    [0, -1, ratio], [0, 1, ratio], [0, -1, -ratio], [0, 1, -ratio],
    [ratio, 0, -1], [ratio, 0, 1], [-ratio, 0, -1], [-ratio, 0, 1],
  ].map(([x, y, z]): StageVec3 => {
    const length = Math.hypot(x, y, z);
    return [x / length, y / length, z / length];
  });
  const faces: Array<readonly [number, number, number]> = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
  ];
  return { vertices, faces };
}

const CYLINDER = unitCylinder(SEGMENTS);
const ICOSPHERE = unitIcosphere();

function framePair(
  motion: StageMotionPayload,
  frame: number,
): { first: number; second: number; blend: number } {
  const maximum = Math.max(0, motion.positions.length - 1);
  const clamped = THREE.MathUtils.clamp(frame, 0, maximum);
  const first = Math.floor(clamped);
  const blend = clamped - first;
  if (blend <= 1e-5 || first >= maximum) return { first, second: first, blend: 0 };
  const second = first + 1;
  const sourceGap = motion.frame_indices?.[second] != null
    ? motion.frame_indices[second] - motion.frame_indices[first]
    : 1;
  if (sourceGap > 1) {
    const nearest = blend >= 0.5 ? second : first;
    return { first: nearest, second: nearest, blend: 0 };
  }
  return { first, second, blend };
}

function coordinate(
  current: readonly StageVec3[],
  next: readonly StageVec3[] | undefined,
  joint: number,
  axis: 0 | 1 | 2,
  blend: number,
): number {
  const start = current[joint]?.[axis] ?? 0;
  const end = next?.[joint]?.[axis];
  return end == null ? start : start + (end - start) * blend;
}

/** Build the legacy tube-and-joint body used when no baked skin is available. */
export function createCapsuleBodyResource(
  motion: StageMotionPayload,
): CapsuleBodyResource | null {
  if (!motion.positions.length || !motion.parent_indices.length) return null;
  const excluded = new Set(motion.exclude_joint_indices ?? []);
  const edges = motion.parent_indices.flatMap((parent, child) =>
    parent >= 0 && !excluded.has(child) && !excluded.has(parent)
      ? [[parent, child] as const]
      : [],
  );
  const visibleJoints = motion.parent_indices.flatMap((_, joint) =>
    excluded.has(joint) ? [] : [joint],
  );
  const cylinderVertexCount = edges.length * CYLINDER.vertices.length;
  const positions = new Float32Array(
    (cylinderVertexCount + visibleJoints.length * ICOSPHERE.vertices.length) * 3,
  );
  const indices: number[] = [];
  edges.forEach((_, edgeIndex) => {
    const offset = edgeIndex * CYLINDER.vertices.length;
    CYLINDER.faces.forEach((face) => {
      indices.push(face[0] + offset, face[1] + offset, face[2] + offset);
    });
  });
  visibleJoints.forEach((_, jointIndex) => {
    const offset = cylinderVertexCount + jointIndex * ICOSPHERE.vertices.length;
    ICOSPHERE.faces.forEach((face) => {
      indices.push(face[0] + offset, face[1] + offset, face[2] + offset);
    });
  });

  const geometry = new THREE.BufferGeometry();
  const attribute = new THREE.BufferAttribute(positions, 3);
  attribute.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("position", attribute);
  geometry.setIndex(indices);
  const material = new THREE.MeshStandardMaterial({
    color: CAPSULE_BODY_VISUAL.color,
    roughness: CAPSULE_BODY_VISUAL.roughness,
    metalness: CAPSULE_BODY_VISUAL.metalness,
    side: THREE.DoubleSide,
    flatShading: true,
  });
  const resource = {
    mesh: new THREE.Mesh(geometry, material),
    edges,
    visibleJoints,
    positions,
  };
  resource.mesh.name = "source-capsule-body";
  resource.mesh.frustumCulled = false;
  setCapsuleBodyFrame(resource, motion, 0);
  return resource;
}

/** Rebuild the small dynamic buffer at one fractional motion frame. */
export function setCapsuleBodyFrame(
  resource: CapsuleBodyResource,
  motion: StageMotionPayload,
  frame: number,
): void {
  const pair = framePair(motion, frame);
  const current = motion.positions[pair.first];
  const next = pair.blend > 1e-5 ? motion.positions[pair.second] : undefined;
  if (!current) return;
  let offset = 0;

  for (const [parent, child] of resource.edges) {
    const start = [0, 1, 2].map((axis) =>
      coordinate(current, next, parent, axis as 0 | 1 | 2, pair.blend),
    );
    const end = [0, 1, 2].map((axis) =>
      coordinate(current, next, child, axis as 0 | 1 | 2, pair.blend),
    );
    let dx = end[0] - start[0];
    let dy = end[1] - start[1];
    let dz = end[2] - start[2];
    const length = Math.hypot(dx, dy, dz) || 1e-6;
    dx /= length;
    dy /= length;
    dz /= length;
    const reference = Math.abs(dx) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    let ax = dy * reference[2] - dz * reference[1];
    let ay = dz * reference[0] - dx * reference[2];
    let az = dx * reference[1] - dy * reference[0];
    const normalLength = Math.hypot(ax, ay, az) || 1;
    ax /= normalLength;
    ay /= normalLength;
    az /= normalLength;
    const ux = dy * az - dz * ay;
    const uy = dz * ax - dx * az;
    const uz = dx * ay - dy * ax;
    for (const vertex of CYLINDER.vertices) {
      resource.positions[offset++] = start[0]
        + ax * vertex[0] * CAPSULE_BODY_VISUAL.boneRadius
        + ux * vertex[1] * CAPSULE_BODY_VISUAL.boneRadius
        + dx * vertex[2] * length;
      resource.positions[offset++] = start[1]
        + ay * vertex[0] * CAPSULE_BODY_VISUAL.boneRadius
        + uy * vertex[1] * CAPSULE_BODY_VISUAL.boneRadius
        + dy * vertex[2] * length;
      resource.positions[offset++] = start[2]
        + az * vertex[0] * CAPSULE_BODY_VISUAL.boneRadius
        + uz * vertex[1] * CAPSULE_BODY_VISUAL.boneRadius
        + dz * vertex[2] * length;
    }
  }

  for (const joint of resource.visibleJoints) {
    const center = [0, 1, 2].map((axis) =>
      coordinate(current, next, joint, axis as 0 | 1 | 2, pair.blend),
    );
    for (const vertex of ICOSPHERE.vertices) {
      resource.positions[offset++] = center[0]
        + vertex[0] * CAPSULE_BODY_VISUAL.jointRadius;
      resource.positions[offset++] = center[1]
        + vertex[1] * CAPSULE_BODY_VISUAL.jointRadius;
      resource.positions[offset++] = center[2]
        + vertex[2] * CAPSULE_BODY_VISUAL.jointRadius;
    }
  }
  resource.mesh.geometry.getAttribute("position").needsUpdate = true;
}

export function disposeCapsuleBody(resource: CapsuleBodyResource): void {
  resource.mesh.geometry.dispose();
  resource.mesh.material.dispose();
}
