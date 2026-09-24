import { useFrame } from "@react-three/fiber";
import { useEffect, useRef, useState } from "react";

import {
  createCapsuleBodyResource,
  disposeCapsuleBody,
  setCapsuleBodyFrame,
  type CapsuleBodyResource,
} from "./capsuleBody";
import { timelineFrameAtTime, type StagePlaybackRef } from "./playback";
import type { StageMotionPayload } from "./types";

function AnimatedCapsule({
  motion,
  resource,
  playback,
  visible,
}: {
  motion: StageMotionPayload;
  resource: CapsuleBodyResource;
  playback?: StagePlaybackRef;
  visible: boolean;
}) {
  const lastFrame = useRef<number | null>(null);

  useFrame(() => {
    if (!visible) return;
    const frame = timelineFrameAtTime(motion, playback?.current.elapsed ?? 0);
    if (lastFrame.current === frame) return;
    setCapsuleBodyFrame(resource, motion, frame);
    lastFrame.current = frame;
  });

  return <primitive object={resource.mesh} dispose={null} />;
}

/** React owner for the legacy-compatible body available to every motion type. */
export function CapsuleBodyLayer({
  motion,
  playback,
  visible,
}: {
  motion: StageMotionPayload | null;
  playback?: StagePlaybackRef;
  visible: boolean;
}) {
  const [loaded, setLoaded] = useState<{
    owner: StageMotionPayload;
    resource: CapsuleBodyResource;
  } | null>(null);
  const resourceRef = useRef<CapsuleBodyResource | null>(null);
  const resource = loaded?.owner === motion ? loaded.resource : null;

  useEffect(() => {
    if (resourceRef.current) disposeCapsuleBody(resourceRef.current);
    resourceRef.current = null;
    setLoaded(null);
    if (!motion) return undefined;
    const next = createCapsuleBodyResource(motion);
    if (!next) return undefined;
    resourceRef.current = next;
    setLoaded({ owner: motion, resource: next });
    return () => {
      if (resourceRef.current === next) resourceRef.current = null;
      disposeCapsuleBody(next);
    };
  }, [motion]);

  if (!motion || !resource) return null;
  return (
    <group name="source-capsule" visible={visible}>
      <AnimatedCapsule
        motion={motion}
        resource={resource}
        playback={playback}
        visible={visible}
      />
    </group>
  );
}
