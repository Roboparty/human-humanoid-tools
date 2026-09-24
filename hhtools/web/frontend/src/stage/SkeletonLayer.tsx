import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

import { timelineFrameAtTime, type StagePlaybackRef } from "./playback";
import type { StageMotionPayload } from "./types";
import {
  SKELETON_VISUALS,
  type SkeletonVisualVariant,
} from "./visualStyle";

interface SkeletonLayerProps {
  motion: StageMotionPayload | null;
  visible: boolean;
  /** Shared StageScene cursor; all animated layers read the same frame. */
  playback?: StagePlaybackRef;
  variant?: SkeletonVisualVariant;
  name?: string;
}

interface SkeletonEdge {
  child: number;
  parent: number;
}

/**
 * Resolve a fractional preview frame.  Interpolation is only allowed when
 * adjacent preview samples are adjacent source frames; interpolating over a
 * sparse gap makes the skeleton visibly slide between unrelated poses.
 */
function resolvePlaybackFrame(
  frameIndices: readonly number[] | undefined,
  frame: number,
  maximumFrame: number,
): { first: number; second: number; blend: number } {
  const first = Math.min(maximumFrame, Math.max(0, Math.floor(frame)));
  const blend = frame - first;
  if (blend <= 1e-5 || first >= maximumFrame) {
    return { first, second: first, blend: 0 };
  }
  const second = first + 1;
  const sourceGap = frameIndices && frameIndices.length > second
    ? frameIndices[second] - frameIndices[first]
    : 1;
  if (sourceGap > 1) {
    const nearest = blend >= 0.5 ? second : first;
    return { first: nearest, second: nearest, blend: 0 };
  }
  return { first, second, blend };
}

/** Write one (possibly interpolated) pose directly into the R3F objects. */
function writeFrame(
  motion: StageMotionPayload,
  edges: readonly SkeletonEdge[],
  spheres: readonly (THREE.Mesh | undefined)[],
  lineGeometry: THREE.BufferGeometry | null,
  frame: number,
): void {
  const maxFrame = motion.positions.length - 1;
  if (maxFrame < 0) return;
  const { first, second, blend } = resolvePlaybackFrame(
    motion.frame_indices,
    frame,
    maxFrame,
  );
  const current = motion.positions[first];
  const next = blend > 1e-5 ? motion.positions[second] : undefined;
  if (!current) return;

  const coordinate = (index: number, axis: 0 | 1 | 2): number => {
    const a = current[index];
    if (!a) return 0;
    const b = next?.[index];
    return b ? a[axis] + (b[axis] - a[axis]) * blend : a[axis];
  };

  spheres.forEach((sphere, index) => {
    if (!sphere || !current[index]) return;
    sphere.position.set(
      coordinate(index, 0),
      coordinate(index, 1),
      coordinate(index, 2),
    );
  });

  const attribute = lineGeometry?.getAttribute("position");
  if (!attribute) return;
  const positions = attribute.array as ArrayLike<number> & { [index: number]: number };
  let offset = 0;
  for (const edge of edges) {
    if (!current[edge.child] || !current[edge.parent]) continue;
    positions[offset++] = coordinate(edge.child, 0);
    positions[offset++] = coordinate(edge.child, 1);
    positions[offset++] = coordinate(edge.child, 2);
    positions[offset++] = coordinate(edge.parent, 0);
    positions[offset++] = coordinate(edge.parent, 1);
    positions[offset++] = coordinate(edge.parent, 2);
  }
  attribute.needsUpdate = true;
}

/** Renders the source motion at the shared StageScene playback cursor. */
export function SkeletonLayer({
  motion,
  visible,
  playback,
  variant = "source",
  name = "source-skeleton",
}: SkeletonLayerProps) {
  const visual = SKELETON_VISUALS[variant];
  const firstFrame = motion?.positions[0] ?? [];
  const spheresRef = useRef<Array<THREE.Mesh | undefined>>([]);
  const lineGeometryRef = useRef<THREE.BufferGeometry | null>(null);
  const lastFrameRef = useRef<number | null>(null);
  const edges = useMemo<SkeletonEdge[]>(() => {
    if (!motion) return [];
    const excluded = new Set(motion.exclude_joint_indices ?? []);
    return motion.parent_indices.flatMap((parent, child) =>
      parent >= 0 && !excluded.has(child) && !excluded.has(parent)
        ? [{ child, parent }]
        : [],
    );
  }, [motion]);
  const linePositions = useMemo(
    () => {
      const positions = new Float32Array(edges.length * 2 * 3);
      let offset = 0;
      for (const edge of edges) {
        const child = firstFrame[edge.child];
        const parent = firstFrame[edge.parent];
        if (!child || !parent) continue;
        positions[offset++] = child[0];
        positions[offset++] = child[1];
        positions[offset++] = child[2];
        positions[offset++] = parent[0];
        positions[offset++] = parent[1];
        positions[offset++] = parent[2];
      }
      return positions;
    },
    [edges, firstFrame],
  );
  const excluded = new Set(motion?.exclude_joint_indices ?? []);

  // Reset the geometry whenever a new clip is selected.  The clock itself is
  // owned by StageScene and shared with the body/environment layers.
  useEffect(() => {
    spheresRef.current.length = firstFrame.length;
    lastFrameRef.current = null;
    if (motion) {
      const frame = timelineFrameAtTime(motion, playback?.current.elapsed ?? 0);
      writeFrame(
        motion,
        edges,
        spheresRef.current,
        lineGeometryRef.current,
        frame,
      );
      lastFrameRef.current = frame;
    }
  }, [motion, edges, firstFrame.length, playback]);

  useFrame(() => {
    // StageScene advances one shared cursor; this layer only applies it.
    if (!visible || !motion || motion.positions.length === 0) return;
    const frame = timelineFrameAtTime(motion, playback?.current.elapsed ?? 0);
    if (lastFrameRef.current === frame) return;
    writeFrame(
      motion,
      edges,
      spheresRef.current,
      lineGeometryRef.current,
      frame,
    );
    lastFrameRef.current = frame;
  });

  return (
    <group visible={visible && motion !== null} name={name}>
      {firstFrame.map((_, index) => (
        <mesh
          key={index}
          ref={(mesh) => {
            if (mesh) spheresRef.current[index] = mesh;
            else delete spheresRef.current[index];
          }}
          position={firstFrame[index]}
          visible={!excluded.has(index)}
        >
          <sphereGeometry
            args={[visual.jointRadius, visual.jointSegments, visual.jointSegments]}
          />
          <meshStandardMaterial
            color={visual.color}
            roughness={visual.roughness}
            metalness={visual.metalness}
            emissive={visual.emissive}
            transparent={visual.opacity < 1}
            opacity={visual.opacity}
          />
        </mesh>
      ))}
      <lineSegments>
        <bufferGeometry ref={lineGeometryRef}>
          <bufferAttribute
            attach="attributes-position"
            args={[linePositions, 3]}
            count={linePositions.length / 3}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color={visual.color}
          transparent
          opacity={visual.lineOpacity}
        />
      </lineSegments>
    </group>
  );
}
