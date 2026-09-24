import * as THREE from "three";

export const DEFAULT_CAMERA_TARGET = new THREE.Vector3(0, 0.9, 0);
export const DEFAULT_CAMERA_OFFSET = new THREE.Vector3(2.6, 1, 3.2);

/** Bounds only rendered geometry; hidden Stage layers must not affect Reset. */
export function visibleObjectBounds(root: THREE.Object3D | null): THREE.Box3 | null {
  if (!root) return null;
  root.updateWorldMatrix(true, true);
  const bounds = new THREE.Box3();
  const candidate = new THREE.Box3();
  let found = false;
  root.traverseVisible((object) => {
    const geometry = (object as THREE.Mesh).geometry;
    if (!geometry) return;
    // Animated body buffers mutate in place, so Reset must refresh their bounds.
    geometry.computeBoundingBox();
    if (!geometry.boundingBox || geometry.boundingBox.isEmpty()) return;
    candidate.copy(geometry.boundingBox).applyMatrix4(object.matrixWorld);
    if (!found) bounds.copy(candidate);
    else bounds.union(candidate);
    found = true;
  });
  return found ? bounds : null;
}

export function combinedVisibleBounds(
  roots: readonly (THREE.Object3D | null)[],
): THREE.Box3 | null {
  let combined: THREE.Box3 | null = null;
  for (const root of roots) {
    const bounds = visibleObjectBounds(root);
    if (!bounds) continue;
    if (combined) combined.union(bounds);
    else combined = bounds.clone();
  }
  return combined;
}

/** Reframe when an asynchronously loaded focus object is actually replaced. */
export function cameraFrameKey(
  revision: number,
  roots: readonly (THREE.Object3D | null)[],
): string {
  return `${revision}:${roots.map((root) => root?.uuid ?? "pending").join("|")}`;
}

export function cameraFrame(
  bounds: THREE.Box3 | null,
  fitBounds = true,
  viewport = {
    verticalFovRadians: THREE.MathUtils.degToRad(50),
    aspect: 1,
  },
): { target: THREE.Vector3; position: THREE.Vector3; span: number } {
  if (!bounds) {
    return {
      target: DEFAULT_CAMERA_TARGET.clone(),
      position: DEFAULT_CAMERA_TARGET.clone().add(DEFAULT_CAMERA_OFFSET),
      span: 1.6,
    };
  }
  const target = bounds.getCenter(new THREE.Vector3());
  const span = Math.max(0.55, bounds.getSize(new THREE.Vector3()).length());
  if (!fitBounds) {
    return {
      target,
      position: target.clone().add(DEFAULT_CAMERA_OFFSET),
      span,
    };
  }
  const verticalHalfFov = Math.max(0.05, viewport.verticalFovRadians / 2);
  const horizontalHalfFov = Math.atan(
    Math.tan(verticalHalfFov) * Math.max(0.1, viewport.aspect),
  );
  const limitingHalfFov = Math.min(verticalHalfFov, horizontalHalfFov);
  const distance = Math.max(
    1.35,
    (span * 0.5 * 1.12) / Math.sin(limitingHalfFov),
  );
  const offset = new THREE.Vector3(0.58, 0.44, 0.68)
    .normalize()
    .multiplyScalar(distance);
  return {
    target,
    position: target.clone().add(offset),
    span,
  };
}
