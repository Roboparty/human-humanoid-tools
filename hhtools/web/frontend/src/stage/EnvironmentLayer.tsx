import { useEffect, useMemo, useRef, useState, type MutableRefObject } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

import type {
  StageMotionPayload,
  StageObjectPayload,
  StageTerrainPayload,
  StageTimelinePayload,
} from "./types";
import {
  timelineFrameAtTime,
  timelineFrameCount,
  type StagePlaybackRef,
} from "./playback";
import {
  ENVIRONMENT_VISUALS,
  type EnvironmentVisualVariant,
} from "./visualStyle";

interface EnvironmentLayerProps {
  motion: StageMotionPayload | null;
  visible: boolean;
  /** Shared cursor keeps props synchronized with skeleton and body playback. */
  playback?: StagePlaybackRef;
  timeline?: StageTimelinePayload | null;
  variant?: EnvironmentVisualVariant;
  name?: string;
}

interface FramePair {
  first: number;
  second: number;
  blend: number;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

/** Resolve a fractional index without interpolating across a sparse gap. */
function resolveFrame(
  frameIndices: readonly number[] | undefined,
  frame: number,
  maximum: number,
): FramePair {
  if (maximum <= 0) return { first: 0, second: 0, blend: 0 };
  const clampedFrame = clamp(frame, 0, maximum);
  const first = Math.min(maximum, Math.floor(clampedFrame));
  const blend = clampedFrame - first;
  if (blend <= 1e-5 || first >= maximum) {
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

function writeObjectTransform(
  object: StageObjectPayload,
  mesh: THREE.Object3D,
  cursor: number,
  quaternionScratch: THREE.Quaternion,
): void {
  const maximum = object.positions.length - 1;
  const pair = resolveFrame(undefined, cursor, maximum);
  const a = object.positions[pair.first];
  if (a) {
    const b = pair.blend > 1e-5 ? object.positions[pair.second] : undefined;
    mesh.position.set(
      b ? a[0] + (b[0] - a[0]) * pair.blend : a[0],
      b ? a[1] + (b[1] - a[1]) * pair.blend : a[1],
      b ? a[2] + (b[2] - a[2]) * pair.blend : a[2],
    );
  }

  const qa = object.quaternions?.[pair.first];
  const qb = pair.blend > 1e-5 ? object.quaternions?.[pair.second] : undefined;
  const target = mesh.quaternion;
  if (!qa || qa.some((value) => !Number.isFinite(value))) {
    target.identity();
    return;
  }
  target.set(qa[0], qa[1], qa[2], qa[3]);
  if (!qb || qb.some((value) => !Number.isFinite(value))) return;
  quaternionScratch.set(qb[0], qb[1], qb[2], qb[3]);
  target.slerp(quaternionScratch, pair.blend);
}

function rgbToHex(color: StageObjectPayload["color"], fallback: number): number {
  if (!color || color.length < 3 || color.some((value) => !Number.isFinite(value))) {
    return fallback;
  }
  // SceneObject's transport contract uses integer RGB channels in 0..255.
  const channels = color.map((value) => clamp(value, 0, 255));
  return (Math.round(channels[0]) << 16)
    | (Math.round(channels[1]) << 8)
    | Math.round(channels[2]);
}

function validTerrain(terrain: StageTerrainPayload | null | undefined): terrain is StageTerrainPayload {
  return Boolean(
    terrain
    && terrain.vertices.length > 0
    && terrain.faces.length > 0,
  );
}

/** Declarative triangulated terrain. Coordinates remain in hhtools Z-up space. */
function TerrainMesh({
  terrain,
  variant,
}: {
  terrain: StageTerrainPayload;
  variant: EnvironmentVisualVariant;
}) {
  const visual = ENVIRONMENT_VISUALS[variant];
  const geometryRef = useRef<THREE.BufferGeometry | null>(null);
  const positions = useMemo(() => {
    const values = new Float32Array(terrain.vertices.length * 3);
    terrain.vertices.forEach((vertex, index) => {
      values[index * 3] = vertex[0];
      values[index * 3 + 1] = vertex[1];
      values[index * 3 + 2] = vertex[2];
    });
    return values;
  }, [terrain]);
  const indices = useMemo(() => {
    const values = new Uint32Array(terrain.faces.length * 3);
    terrain.faces.forEach((face, index) => {
      values[index * 3] = face[0];
      values[index * 3 + 1] = face[1];
      values[index * 3 + 2] = face[2];
    });
    return values;
  }, [terrain]);

  // BufferGeometry does not derive normals when attributes are attached from
  // JSX. Recompute them whenever a new terrain payload is published so the
  // standard material receives the same shaded mesh as the legacy renderer.
  useEffect(() => {
    const geometry = geometryRef.current;
    if (!geometry) return;
    geometry.deleteAttribute("normal");
    geometry.computeVertexNormals();
  }, [positions, indices]);

  return (
    <mesh name="source-terrain">
      <bufferGeometry ref={geometryRef}>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
          count={positions.length / 3}
        />
        <bufferAttribute
          attach="index"
          args={[indices, 1]}
          count={indices.length}
        />
      </bufferGeometry>
      <meshStandardMaterial
        color={visual.terrainColor}
        roughness={visual.terrainRoughness}
        side={THREE.DoubleSide}
        flatShading
        transparent={visual.terrainOpacity < 1}
        opacity={visual.terrainOpacity}
      />
    </mesh>
  );
}

function disposeObject(root: THREE.Object3D): void {
  root.traverse((node) => {
    const mesh = node as THREE.Mesh;
    mesh.geometry?.dispose();
    const materials = Array.isArray(mesh.material)
      ? mesh.material
      : mesh.material
        ? [mesh.material]
        : [];
    for (const material of materials) {
      for (const value of Object.values(material)) {
        if (value instanceof THREE.Texture) value.dispose();
      }
      material.dispose();
    }
  });
}

function objectMeshUrl(
  motion: StageMotionPayload,
  object: StageObjectPayload,
  index: number,
): string | null {
  if (!object.has_mesh) return null;
  const source = motion.object_mesh_source ?? (
    motion.token ? { kind: "motion" as const, token: motion.token } : null
  );
  if (!source) return null;
  const query = new URLSearchParams({ token: source.token });
  if (typeof object.scale === "number" && object.scale > 0) {
    query.set("scale", String(object.scale));
  }
  if (source.kind === "r2r") {
    if (!object.mesh_file) return null;
    query.set("mesh", object.mesh_file);
    return `/api/r2r/scene_glb?${query.toString()}`;
  }
  if (source.kind === "dataset") {
    if (!object.mesh_file) return null;
    query.set("mesh", object.mesh_file);
    return `/api/dataset/scene_glb?${query.toString()}`;
  }
  query.set("index", String(object.source_index ?? index));
  return `/api/object_glb?${query.toString()}`;
}

/** Use the real prop mesh when available, retaining the box while it loads. */
function InteractionObject({
  object,
  index,
  meshUrl,
  objectRefs,
  variant,
}: {
  object: StageObjectPayload;
  index: number;
  meshUrl: string | null;
  objectRefs: MutableRefObject<Array<THREE.Object3D | undefined>>;
  variant: EnvironmentVisualVariant;
}) {
  const visual = ENVIRONMENT_VISUALS[variant];
  const [loaded, setLoaded] = useState<{
    object: StageObjectPayload;
    meshUrl: string;
    index: number;
    scene: THREE.Group;
  } | null>(null);
  const scene =
    loaded?.object === object &&
    loaded.meshUrl === meshUrl &&
    loaded.index === index
      ? loaded.scene
      : null;
  const initialPosition = object.positions[0] ?? [0, 0, 0];
  const initialQuaternion = object.quaternions[0] ?? [0, 0, 0, 1];
  const opacity = clamp(object.opacity ?? visual.objectOpacity, 0, 1);

  useEffect(() => {
    let active = true;
    let ownedScene: THREE.Group | null = null;
    setLoaded(null);
    if (!object.has_mesh || !meshUrl) return undefined;
    new GLTFLoader().load(
      meshUrl,
      (gltf) => {
        ownedScene = gltf.scene;
        if (!active) {
          disposeObject(gltf.scene);
          return;
        }
        setLoaded({ object, meshUrl, index, scene: gltf.scene });
      },
      undefined,
      () => {
        // A box with the serialized extents is the deliberate mesh fallback.
      },
    );

    return () => {
      active = false;
      if (ownedScene) disposeObject(ownedScene);
    };
  }, [index, meshUrl, object]);

  return (
    <group
      ref={(group) => {
        if (group) objectRefs.current[index] = group;
        else delete objectRefs.current[index];
      }}
      name={object.name || `interaction-object-${index + 1}`}
      position={initialPosition}
      quaternion={initialQuaternion}
    >
      {scene ? (
        <primitive object={scene} dispose={null} />
      ) : (
        <mesh>
          <boxGeometry args={[...object.extents]} />
          <meshStandardMaterial
            color={rgbToHex(object.color, visual.objectColor)}
            transparent={opacity < 1}
            opacity={opacity}
            roughness={visual.objectRoughness}
          />
        </mesh>
      )}
    </group>
  );
}

/**
 * Renders source terrain and interaction props independently from the body.
 * Props follow the motion timeline even when the skeleton/body layer is hidden.
 */
export function EnvironmentLayer({
  motion,
  visible,
  playback,
  timeline,
  variant = "source",
  name = "source-environment",
}: EnvironmentLayerProps) {
  const objectRefs = useRef<Array<THREE.Object3D | undefined>>([]);
  const lastFrameRef = useRef<number | null>(null);
  const quaternionScratch = useRef(new THREE.Quaternion());
  const objects = motion?.objects ?? [];
  useEffect(() => {
    objectRefs.current.length = objects.length;
    lastFrameRef.current = null;
  }, [motion, objects.length]);

  useFrame(() => {
    if (!visible || !motion || objects.length === 0) return;
    const activeTimeline = timeline ?? motion;
    const currentFrame = timelineFrameAtTime(
      activeTimeline,
      playback?.current.elapsed ?? 0,
    );
    if (lastFrameRef.current === currentFrame) return;

    const timelineLength = Math.max(
      timelineFrameCount(activeTimeline),
      ...objects.map((object) => object.positions.length),
      0,
    );

    const sourceMaximum = Math.max(
      0,
      (timelineFrameCount(activeTimeline) || timelineLength) - 1,
    );
    const sourceFrame = resolveFrame(
      activeTimeline.frame_indices,
      currentFrame,
      sourceMaximum,
    );
    objects.forEach((object, index) => {
      const mesh = objectRefs.current[index];
      if (!mesh || object.positions.length === 0) return;
      // Object tracks are normally sampled with the skeleton. Normalizing the
      // cursor also handles sidecars that contain a different sample count.
      const objectMaximum = object.positions.length - 1;
      const normalized = sourceMaximum > 0
        ? (sourceFrame.first + sourceFrame.blend) / sourceMaximum
        : 0;
      const objectCursor = normalized * objectMaximum;
      writeObjectTransform(
        object,
        mesh,
        objectCursor,
        quaternionScratch.current,
      );
    });
    lastFrameRef.current = currentFrame;
  });

  if (!motion) return null;
  return (
    <group name={name} visible={visible}>
      {validTerrain(motion.terrain) && (
        <TerrainMesh terrain={motion.terrain} variant={variant} />
      )}
      {objects.map((object, index) => (
        <InteractionObject
          key={`${object.name ?? "object"}-${index}`}
          object={object}
          index={index}
          meshUrl={objectMeshUrl(motion, object, index)}
          objectRefs={objectRefs}
          variant={variant}
        />
      ))}
    </group>
  );
}
