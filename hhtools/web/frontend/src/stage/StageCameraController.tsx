import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useRef, type RefObject } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import {
  cameraFrame,
  cameraFrameKey,
  combinedVisibleBounds,
  visibleObjectBounds,
} from "./camera";

interface StageCameraControllerProps {
  content: RefObject<THREE.Group | null>;
  focusTargets: readonly RefObject<THREE.Group | null>[];
  followTarget: RefObject<THREE.Group | null>;
  frameRevision: number;
  follow: boolean;
  calibration: boolean;
  interactionLocked?: boolean;
}

/** Owns orbit interaction, content framing, and H2R robot following. */
export function StageCameraController({
  content,
  focusTargets,
  followTarget,
  frameRevision,
  follow,
  calibration,
  interactionLocked = false,
}: StageCameraControllerProps) {
  const { camera, gl } = useThree();
  const controlsRef = useRef<OrbitControls | null>(null);
  const framedKey = useRef<string | null>(null);
  const manualUntil = useRef(0);
  const focus = useRef(new THREE.Vector3());

  useEffect(() => {
    const controls = new OrbitControls(camera, gl.domElement);
    controls.target.set(0, 0.9, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;
    controls.enableZoom = false;
    controls.enabled = !interactionLocked;
    controls.update();
    controlsRef.current = controls;

    const markManual = () => {
      manualUntil.current = performance.now() + 2800;
    };
    const smoothWheel = (event: WheelEvent): void => {
      if (!controls.enabled) return;
      let delta = event.deltaY;
      if (event.deltaMode === 1) delta *= 16;
      else if (event.deltaMode === 2) delta *= 400;
      const step = THREE.MathUtils.clamp(-delta / 120, -2.5, 2.5);
      const scale = Math.pow(0.968, step);
      const offset = camera.position.clone().sub(controls.target);
      const distance = offset.length();
      if (distance < 1e-6) return;
      offset.setLength(
        THREE.MathUtils.clamp(
          distance * scale,
          controls.minDistance,
          controls.maxDistance,
        ),
      );
      camera.position.copy(controls.target).add(offset);
      controls.update();
      markManual();
      event.preventDefault();
    };

    controls.addEventListener("start", markManual);
    controls.addEventListener("end", markManual);
    gl.domElement.addEventListener("wheel", smoothWheel, { passive: false });
    return () => {
      controls.removeEventListener("start", markManual);
      controls.removeEventListener("end", markManual);
      gl.domElement.removeEventListener("wheel", smoothWheel);
      controls.dispose();
      controlsRef.current = null;
    };
  }, [camera, gl]);

  useEffect(() => {
    if (controlsRef.current) controlsRef.current.enabled = !interactionLocked;
  }, [interactionLocked]);

  useFrame((_state, delta) => {
    const controls = controlsRef.current;
    if (!controls) return;

    const focusObjects = focusTargets.map((target) => target.current);
    const nextFrameKey = cameraFrameKey(frameRevision, focusObjects);
    if (framedKey.current !== nextFrameKey) {
      const targetBounds = calibration
        ? visibleObjectBounds(content.current)
        : combinedVisibleBounds(focusObjects);
      const bounds = targetBounds ?? visibleObjectBounds(content.current);
      if (bounds) {
        // The old workbench fits robots and calibration precisely, but keeps
        // ordinary motion clips at its more spacious default camera distance.
        const frame = cameraFrame(bounds, calibration || targetBounds !== null, {
          verticalFovRadians:
            camera instanceof THREE.PerspectiveCamera
              ? THREE.MathUtils.degToRad(camera.fov)
              : THREE.MathUtils.degToRad(50),
          aspect:
            camera instanceof THREE.PerspectiveCamera ? camera.aspect : 1,
        });
        controls.target.copy(frame.target);
        camera.position.copy(frame.position);
        controls.minDistance = calibration ? Math.max(0.28, frame.span * 0.12) : 0;
        controls.maxDistance = calibration ? Math.max(frame.span * 6, 18) : Infinity;
        controls.update();
        framedKey.current = nextFrameKey;
      }
    }

    if (follow && !calibration && followTarget.current) {
      followTarget.current.getWorldPosition(focus.current);
      const offset = focus.current.clone().sub(controls.target);
      const jump = offset.lengthSq() > 0.25;
      if (jump || performance.now() > manualUntil.current) {
        offset.multiplyScalar(jump ? 1 : Math.min(1, Math.max(delta, 0) * 12));
        controls.target.add(offset);
        camera.position.add(offset);
      }
    }
    controls.update();
  });

  return null;
}
