import * as THREE from "three";

import type { ReferenceJointMapping } from "./referenceSkeleton";

const REFERENCE_LABELS: Readonly<Record<string, string>> = {
  hips: "Hips",
  chest: "Chest",
  neck: "Neck",
  head: "Head",
  lefthip: "Left hip",
  righthip: "Right hip",
  leftknee: "Left knee",
  rightknee: "Right knee",
  leftankle: "Left ankle",
  rightankle: "Right ankle",
  leftshoulder: "Left shoulder",
  rightshoulder: "Right shoulder",
  leftelbow: "Left elbow",
  rightelbow: "Right elbow",
  leftwrist: "Left wrist",
  rightwrist: "Right wrist",
};

export interface CalibrationMappingProjection {
  readonly key: string;
  readonly referenceX: number;
  readonly referenceY: number;
  readonly targetX: number;
  readonly targetY: number;
}

export interface ProjectedEndpoints {
  readonly referenceX: number;
  readonly referenceY: number;
  readonly targetX: number;
  readonly targetY: number;
}

export interface CalibrationLabelBox {
  readonly key: string;
  readonly anchorX: number;
  readonly anchorY: number;
  readonly width: number;
  readonly height: number;
}

export interface CalibrationLabelPosition {
  readonly key: string;
  readonly left: number;
  readonly top: number;
}

/** Place labels outside the actor and pack each side without overlap. */
export function layoutCalibrationLabels(
  boxes: readonly CalibrationLabelBox[],
  viewportWidth: number,
  viewportHeight: number,
  padding = 4,
  gap = 3,
): readonly CalibrationLabelPosition[] {
  if (viewportWidth <= 0 || viewportHeight <= 0) return [];
  const center = viewportWidth / 2;
  const groups = [
    boxes.filter((box) => box.anchorX < center),
    boxes.filter((box) => box.anchorX >= center),
  ];
  return groups.flatMap((group, side) => {
    const ordered = [...group].sort(
      (left, right) => left.anchorY - right.anchorY || left.anchorX - right.anchorX,
    );
    let cursor = padding;
    const positions = ordered.map((box) => {
      const left = Math.min(
        viewportWidth - padding - box.width,
        Math.max(
          padding,
          side === 0 ? box.anchorX - box.width - 8 : box.anchorX + 8,
        ),
      );
      const preferredTop = Math.min(
        viewportHeight - padding - box.height,
        Math.max(padding, box.anchorY - box.height / 2),
      );
      const top = Math.max(cursor, preferredTop);
      cursor = top + box.height + gap;
      return { key: box.key, left, top };
    });
    const overflow = Math.max(0, cursor - gap + padding - viewportHeight);
    return positions.map((position) => ({
      ...position,
      top: Math.max(padding, position.top - overflow),
    }));
  });
}

function normalizedSemantic(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9]/g, "");
}

export function calibrationMappingKey(mapping: ReferenceJointMapping): string {
  return `${mapping.index}\u0000${mapping.semantic}\u0000${mapping.targetLink}`;
}

export function calibrationMappingLabel(mapping: ReferenceJointMapping): string {
  const semantic = REFERENCE_LABELS[normalizedSemantic(mapping.semantic)]
    ?? mapping.semantic.replaceAll("_", " ");
  return `${semantic} · ${mapping.targetLink}`;
}

/** Project two world-space points onto the CSS-pixel Stage overlay. */
export function projectCalibrationEndpoints(
  referenceWorld: THREE.Vector3,
  targetWorld: THREE.Vector3,
  camera: THREE.Camera,
  width: number,
  height: number,
): ProjectedEndpoints | null {
  referenceWorld.project(camera);
  targetWorld.project(camera);
  const inFrustum =
    referenceWorld.z >= -1 &&
    referenceWorld.z <= 1 &&
    targetWorld.z >= -1 &&
    targetWorld.z <= 1;
  if (!inFrustum || width <= 0 || height <= 0) return null;
  return {
    referenceX: (referenceWorld.x * 0.5 + 0.5) * width,
    referenceY: (-referenceWorld.y * 0.5 + 0.5) * height,
    targetX: (targetWorld.x * 0.5 + 0.5) * width,
    targetY: (-targetWorld.y * 0.5 + 0.5) * height,
  };
}
