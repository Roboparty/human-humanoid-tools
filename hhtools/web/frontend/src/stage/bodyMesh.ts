import * as THREE from "three";

import type { StageBodyMeshPayload } from "./types";
import { BAKED_BODY_VISUAL } from "./visualStyle.ts";

export interface BodyMeshResource {
  readonly mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;
  readonly vertices: Float32Array;
  readonly numVerts: number;
  readonly numFrames: number;
}

export type CompleteBodyMeshPayload = StageBodyMeshPayload & {
  readonly vertices_gz_b64: string;
  readonly num_verts: number;
  readonly num_frames: number;
  readonly triangles: readonly (readonly [number, number, number])[];
};

/** Decode the backend's gzip/base64 vertex cache in browsers and Electron. */
export async function decodeBodyMeshVertices(encoded: string): Promise<Float32Array> {
  if (typeof globalThis.atob !== "function") {
    throw new Error("Body mesh decoding requires browser base64 support");
  }
  if (typeof DecompressionStream === "undefined") {
    throw new Error("This browser does not support gzip decompression");
  }
  const compressed = Uint8Array.from(globalThis.atob(encoded), (character) =>
    character.charCodeAt(0),
  );
  const stream = new Blob([compressed])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"));
  const buffer = await new Response(stream).arrayBuffer();
  if (buffer.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Body mesh vertex buffer is not float32 aligned");
  }
  return new Float32Array(buffer);
}

export function isCompleteBodyMesh(
  payload: StageBodyMeshPayload | undefined,
): payload is CompleteBodyMeshPayload {
  return Boolean(
    payload?.available &&
      typeof payload.vertices_gz_b64 === "string" &&
      Number.isInteger(payload.num_verts) &&
      (payload.num_verts ?? 0) > 0 &&
      Number.isInteger(payload.num_frames) &&
      (payload.num_frames ?? 0) > 0 &&
      Array.isArray(payload.triangles) &&
      payload.triangles.length > 0,
  );
}

function triangleIndices(
  triangles: readonly (readonly [number, number, number])[],
  numVerts: number,
): number[] {
  const indices: number[] = [];
  for (const triangle of triangles) {
    if (
      triangle.length !== 3 ||
      triangle.some(
        (index) => !Number.isInteger(index) || index < 0 || index >= numVerts,
      )
    ) {
      throw new Error("Body mesh triangle index is out of range");
    }
    indices.push(triangle[0], triangle[1], triangle[2]);
  }
  return indices;
}

export async function createBodyMeshResource(
  payload: CompleteBodyMeshPayload,
): Promise<BodyMeshResource> {
  const vertices = await decodeBodyMeshVertices(payload.vertices_gz_b64);
  const expectedLength = payload.num_frames * payload.num_verts * 3;
  if (vertices.length !== expectedLength) {
    throw new Error(
      `Body mesh vertex buffer has ${vertices.length} values; expected ${expectedLength}`,
    );
  }

  const geometry = new THREE.BufferGeometry();
  let material: THREE.MeshStandardMaterial | null = null;
  try {
    const firstFrame = vertices.slice(0, payload.num_verts * 3);
    const positions = new THREE.Float32BufferAttribute(firstFrame, 3);
    positions.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("position", positions);
    geometry.setIndex(triangleIndices(payload.triangles, payload.num_verts));
    geometry.computeVertexNormals();
    material = new THREE.MeshStandardMaterial({
      color: BAKED_BODY_VISUAL.color,
      roughness: BAKED_BODY_VISUAL.roughness,
      metalness: BAKED_BODY_VISUAL.metalness,
      side: THREE.DoubleSide,
      flatShading: true,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = "source-body-mesh";
    mesh.frustumCulled = false;
    return {
      mesh,
      vertices,
      numVerts: payload.num_verts,
      numFrames: payload.num_frames,
    };
  } catch (error) {
    geometry.dispose();
    material?.dispose();
    throw error;
  }
}

/** Copy one interpolated frame into the live dynamic position attribute. */
export function setBodyMeshFrame(resource: BodyMeshResource, frame: number): void {
  const maximum = resource.numFrames - 1;
  const safeFrame = Number.isFinite(frame)
    ? THREE.MathUtils.clamp(frame, 0, maximum)
    : 0;
  const first = Math.floor(safeFrame);
  const blend = safeFrame - first;
  const frameSize = resource.numVerts * 3;
  const firstOffset = first * frameSize;
  const positions = resource.mesh.geometry.getAttribute("position") as THREE.BufferAttribute;
  const target = positions.array as Float32Array;

  if (blend <= 1e-5 || first >= maximum) {
    target.set(resource.vertices.subarray(firstOffset, firstOffset + frameSize));
  } else {
    const secondOffset = firstOffset + frameSize;
    for (let index = 0; index < frameSize; index += 1) {
      const value = resource.vertices[firstOffset + index];
      target[index] = value + (resource.vertices[secondOffset + index] - value) * blend;
    }
  }
  positions.needsUpdate = true;
}

export function disposeBodyMesh(resource: BodyMeshResource): void {
  resource.mesh.geometry.dispose();
  resource.mesh.material.dispose();
}
