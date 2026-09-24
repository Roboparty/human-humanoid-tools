import { useFrame } from "@react-three/fiber";
import { useEffect, useRef, useState } from "react";

import {
  createBodyMeshResource,
  disposeBodyMesh,
  isCompleteBodyMesh,
  setBodyMeshFrame,
  type BodyMeshResource,
} from "./bodyMesh";
import { timelineFrameAtTime, type StagePlaybackRef } from "./playback";
import type { StageMotionPayload } from "./types";

interface BodyMeshLayerProps {
  motion: StageMotionPayload | null;
  visible: boolean;
  playback?: StagePlaybackRef;
  /** Stage keeps its skeleton fallback until asynchronous decoding succeeds. */
  onReadyChange?: (ready: boolean) => void;
}

function AnimatedBody({
  resource,
  playback,
  motion,
  sourceFrameCount,
  visible,
}: {
  resource: BodyMeshResource;
  playback?: StagePlaybackRef;
  motion: StageMotionPayload;
  sourceFrameCount: number;
  visible: boolean;
}) {
  const lastFrame = useRef<number | null>(null);

  useFrame(() => {
    if (!visible) return;
    const sourceLast = Math.max(0, sourceFrameCount - 1);
    const sourceFrame = timelineFrameAtTime(
      motion,
      playback?.current.elapsed ?? 0,
    );
    const fraction = sourceLast > 0
      ? Math.min(1, Math.max(0, sourceFrame / sourceLast))
      : 0;
    const frame = fraction * Math.max(0, resource.numFrames - 1);
    if (lastFrame.current === frame) return;
    setBodyMeshFrame(resource, frame);
    lastFrame.current = frame;
  });

  return <primitive object={resource.mesh} dispose={null} />;
}

/** Decodes and animates an optional baked body in the shared Z-up world. */
export function BodyMeshLayer({
  motion,
  visible,
  playback,
  onReadyChange,
}: BodyMeshLayerProps) {
  const bodyMesh = motion?.body_mesh;
  const [loaded, setLoaded] = useState<{
    owner: typeof bodyMesh;
    resource: BodyMeshResource;
  } | null>(null);
  const resourceRef = useRef<BodyMeshResource | null>(null);
  const resource = loaded && loaded.owner === bodyMesh ? loaded.resource : null;

  useEffect(() => {
    let cancelled = false;
    if (resourceRef.current) disposeBodyMesh(resourceRef.current);
    resourceRef.current = null;
    setLoaded(null);
    onReadyChange?.(false);
    if (!isCompleteBodyMesh(bodyMesh)) return undefined;

    void createBodyMeshResource(bodyMesh)
      .then((next) => {
        if (cancelled) {
          disposeBodyMesh(next);
          return;
        }
        resourceRef.current = next;
        setLoaded({ owner: bodyMesh, resource: next });
        onReadyChange?.(true);
      })
      .catch((error: unknown) => {
        if (!cancelled) console.warn("Unable to load source body mesh", error);
      });

    return () => {
      cancelled = true;
      if (resourceRef.current) disposeBodyMesh(resourceRef.current);
      resourceRef.current = null;
    };
  }, [bodyMesh, onReadyChange]);

  if (!resource || !motion) return null;
  return (
    <group name="source-body" visible={visible}>
      <AnimatedBody
        resource={resource}
        playback={playback}
        motion={motion}
        sourceFrameCount={motion.positions.length || resource.numFrames}
        visible={visible}
      />
    </group>
  );
}
