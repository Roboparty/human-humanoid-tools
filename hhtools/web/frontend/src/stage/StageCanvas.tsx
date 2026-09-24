import { Canvas, useFrame } from "@react-three/fiber";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type RefObject,
} from "react";
import * as THREE from "three";

import { BodyMeshLayer } from "./BodyMeshLayer";
import type { CalibrationMappingOverlayHandle } from "./CalibrationMappingOverlay";
import type { CalibrationDisplayOptions } from "./calibrationDisplay";
import type { CalibrationInteractionModel } from "./calibrationInteraction";
import { CapsuleBodyLayer } from "./CapsuleBodyLayer";
import { EnvironmentLayer } from "./EnvironmentLayer";
import { ManipulatorLayer } from "./ManipulatorLayer";
import { ReferenceSkeletonLayer } from "./ReferenceSkeletonLayer";
import { RobotLayer } from "./RobotLayer";
import { SkeletonLayer } from "./SkeletonLayer";
import { StageCameraController } from "./StageCameraController";
import { advancePlayback, type StagePlaybackRef } from "./playback";
import { projectR2rStageVisibility } from "./presentation";
import type { RobotLinkPoseReader } from "./robotPoseReader";
import type {
  StageLayerId,
  StageMotionPayload,
  StageR2rPresentationPayload,
  StageRobotPayload,
  StageRobotTrajectoryPayload,
  StageTimelinePayload,
} from "./types";

function GroundGrid() {
  const gridRef = useRef<THREE.GridHelper | null>(null);

  useEffect(() => {
    const material = gridRef.current?.material;
    if (!material || Array.isArray(material)) return;
    material.opacity = 0.35;
    material.transparent = true;
    material.needsUpdate = true;
  }, []);

  return (
    <gridHelper
      ref={gridRef}
      args={[20, 40, 0x99a0ab, 0xd2d6dd]}
    />
  );
}

/** Advances the cursor owned by Stage inside the one R3F render loop. */
function PlaybackClock({
  timeline,
  playback,
  onPlaybackChange,
}: {
  timeline: StageTimelinePayload | null;
  playback: StagePlaybackRef;
  onPlaybackChange?: () => void;
}) {
  const reportElapsed = useRef(0);
  useFrame((_state, delta) => {
    const cursor = playback.current;
    const result = advancePlayback(cursor, timeline, delta);
    if (result === "idle") return;
    if (result === "looped" || result === "ended") {
      reportElapsed.current = 0;
      onPlaybackChange?.();
      return;
    }
    reportElapsed.current += Math.min(0.1, Math.max(0, delta));
    if (reportElapsed.current >= 0.1) {
      reportElapsed.current %= 0.1;
      onPlaybackChange?.();
    }
  });
  return null;
}

function calibrationLink(
  interaction: CalibrationInteractionModel | null,
  jointName: string | null,
): string | null {
  if (!jointName) return null;
  return (
    interaction?.jointLimits.find((joint) => joint.name === jointName)
      ?.child_link ?? null
  );
}

function R2rLayers({
  presentation,
  playback,
  visibleLayers,
  onSourceRobotChange,
  onTargetRobotChange,
  targetPoseReader,
  mappingOverlay,
  onTargetPoseReaderChange,
  calibrationDisplay,
  selectedLink,
  hoveredLink,
}: {
  presentation: StageR2rPresentationPayload;
  playback: StagePlaybackRef;
  visibleLayers: readonly StageLayerId[];
  onSourceRobotChange: (object: THREE.Group | null) => void;
  onTargetRobotChange: (object: THREE.Group | null) => void;
  targetPoseReader: RefObject<RobotLinkPoseReader | null>;
  mappingOverlay?: RefObject<CalibrationMappingOverlayHandle | null>;
  onTargetPoseReaderChange: (reader: RobotLinkPoseReader | null) => void;
  calibrationDisplay: CalibrationDisplayOptions;
  selectedLink: string | null;
  hoveredLink: string | null;
}) {
  const visibility = projectR2rStageVisibility(presentation, visibleLayers);
  if (presentation.phase === "calibration") {
    return (
      <>
        <ReferenceSkeletonLayer
          reference={presentation.calibrationReference}
          robot={presentation.target.robot}
          visible={visibility.calibrationReference}
          name="r2r-calibration-reference"
          poseReader={targetPoseReader}
          mappingOverlay={mappingOverlay}
          mappedOnly={calibrationDisplay.mappedOnly}
          sourceOpacity={calibrationDisplay.referenceOpacity}
        />
        <RobotLayer
          robot={presentation.target.robot}
          trajectory={presentation.target.trajectory}
          playback={playback}
          visible={visibility.targetRobot}
          opacity={calibrationDisplay.robotOpacity}
          name="r2r-target-robot"
          onObjectChange={onTargetRobotChange}
          onPoseReaderChange={onTargetPoseReaderChange}
          selectedLink={selectedLink}
          hoveredLink={hoveredLink}
        />
      </>
    );
  }

  return (
    <>
      <RobotLayer
        robot={presentation.source.robot}
        trajectory={presentation.source.trajectory}
        playback={playback}
        visible={visibility.sourceRobot}
        name="r2r-source-robot"
        onObjectChange={onSourceRobotChange}
      />
      <SkeletonLayer
        motion={presentation.source.skeleton}
        visible={visibility.sourceSkeleton}
        playback={playback}
        variant="r2r-source"
        name="r2r-source-skeleton"
      />
      <EnvironmentLayer
        motion={presentation.source.environment}
        visible={visibility.sourceScene}
        playback={playback}
        timeline={presentation.source.trajectory ?? presentation.source.skeleton}
        variant="scaled"
        name="r2r-source-environment"
      />
      <RobotLayer
        robot={presentation.target.robot}
        trajectory={presentation.target.trajectory}
        playback={playback}
        visible={visibility.targetRobot}
        name="r2r-target-robot"
        onObjectChange={onTargetRobotChange}
      />
      <SkeletonLayer
        motion={presentation.target.skeleton}
        visible={visibility.targetSkeleton}
        playback={playback}
        variant="scaled"
        name="r2r-target-skeleton"
      />
      <EnvironmentLayer
        motion={presentation.target.environment}
        visible={visibility.targetScene}
        playback={playback}
        timeline={presentation.target.trajectory ?? presentation.target.skeleton}
        variant="scaled"
        name="r2r-target-environment"
      />
    </>
  );
}

/** Static scene content migrated from the legacy renderer bootstrap. */
function StageScene({
  motion,
  scaledMotion,
  robot,
  robotTrajectory,
  r2r,
  timeline,
  playback,
  onPlaybackChange,
  visibleLayers,
  robotOpacity,
  cameraRevision,
  followRobot,
  calibration,
  calibrationDisplay,
  calibrationInteraction,
  mappingOverlay,
}: {
  motion: StageMotionPayload | null;
  scaledMotion: StageMotionPayload | null;
  robot: StageRobotPayload | null;
  robotTrajectory: StageRobotTrajectoryPayload | null;
  r2r: StageR2rPresentationPayload | null;
  timeline: StageTimelinePayload | null;
  playback: StagePlaybackRef;
  onPlaybackChange?: () => void;
  visibleLayers: readonly StageLayerId[];
  robotOpacity: number;
  cameraRevision: number;
  followRobot: boolean;
  calibration: boolean;
  calibrationDisplay: CalibrationDisplayOptions;
  calibrationInteraction: CalibrationInteractionModel | null;
  mappingOverlay?: RefObject<CalibrationMappingOverlayHandle | null>;
}) {
  const content = useRef<THREE.Group | null>(null);
  const followedRobot = useRef<THREE.Group | null>(null);
  const sourceRobot = useRef<THREE.Group | null>(null);
  const targetRobot = useRef<THREE.Group | null>(null);
  const targetPoseReader = useRef<RobotLinkPoseReader | null>(null);
  const publishFollowedRobot = useCallback((object: THREE.Group | null) => {
    followedRobot.current = object;
  }, []);
  const publishSourceRobot = useCallback((object: THREE.Group | null) => {
    sourceRobot.current = object;
  }, []);
  const publishTargetRobot = useCallback((object: THREE.Group | null) => {
    targetRobot.current = object;
  }, []);
  const publishTargetPoseReader = useCallback(
    (reader: RobotLinkPoseReader | null) => {
      targetPoseReader.current = reader;
    },
    [],
  );
  const bodyMesh = motion?.body_mesh;
  const [bodyStatus, setBodyStatus] = useState<{
    owner: typeof bodyMesh;
    ready: boolean;
  }>({ owner: undefined, ready: false });
  const [manipulating, setManipulating] = useState(false);
  const [hoveredJoint, setHoveredJoint] = useState<string | null>(null);
  const selectedLink = calibrationLink(
    calibrationInteraction,
    calibrationInteraction?.selectedJoint ?? null,
  );
  const hoveredLink = calibrationLink(calibrationInteraction, hoveredJoint);
  const bodyReady = bodyStatus.owner === bodyMesh && bodyStatus.ready;
  const publishBodyReady = useCallback(
    (ready: boolean) => setBodyStatus({ owner: bodyMesh, ready }),
    [bodyMesh],
  );
  const hasBodyMesh = bodyMesh?.available === true;
  const hasEnvironment = Boolean(
    motion?.terrain || (motion?.objects && motion.objects.length > 0),
  );
  const hasScaledEnvironment = Boolean(
    scaledMotion?.terrain ||
      (scaledMotion?.objects && scaledMotion.objects.length > 0),
  );
  const skeletonVisible = motion !== null && visibleLayers.includes("skeleton");
  const bodyVisible =
    !calibration && hasBodyMesh && bodyReady && visibleLayers.includes("body");
  const capsuleVisible =
    !calibration && motion !== null && !bodyReady && visibleLayers.includes("body");
  const robotVisible = robot !== null && visibleLayers.includes("robot");
  const environmentVisible =
    !calibration && hasEnvironment && visibleLayers.includes("objects");
  const scaledSkeletonVisible =
    !calibration &&
    scaledMotion !== null &&
    visibleLayers.includes("scaled-skeleton");
  const scaledEnvironmentVisible =
    !calibration &&
    hasScaledEnvironment &&
    visibleLayers.includes("scaled-scene");

  return (
    <>
      <PlaybackClock
        timeline={timeline}
        playback={playback}
        onPlaybackChange={onPlaybackChange}
      />
      <StageCameraController
        content={content}
        focusTargets={r2r ? [sourceRobot, targetRobot] : [followedRobot]}
        followTarget={followedRobot}
        frameRevision={cameraRevision}
        follow={followRobot}
        calibration={calibration}
        interactionLocked={manipulating}
      />
      <ambientLight intensity={0.55} />
      <hemisphereLight args={[0xffffff, 0x8899aa, 1.35]} />
      <directionalLight position={[3, 6, 4]} intensity={1.5} />
      <directionalLight position={[-3, 4, -2]} intensity={0.85} />

      <GroundGrid />
      <group name="hhtools-world" rotation={[-Math.PI / 2, 0, 0]}>
        <axesHelper args={[1.2]} />
        <group ref={content} name="stage-content">
          {r2r ? (
            <R2rLayers
              presentation={r2r}
              playback={playback}
              visibleLayers={visibleLayers}
              onSourceRobotChange={publishSourceRobot}
              onTargetRobotChange={publishTargetRobot}
              targetPoseReader={targetPoseReader}
              mappingOverlay={mappingOverlay}
              onTargetPoseReaderChange={publishTargetPoseReader}
              calibrationDisplay={calibrationDisplay}
              selectedLink={selectedLink}
              hoveredLink={hoveredLink}
            />
          ) : (
            <>
              {calibration ? (
                <ReferenceSkeletonLayer
                  reference={motion}
                  robot={robot}
                  visible={skeletonVisible}
                  poseReader={targetPoseReader}
                  mappingOverlay={mappingOverlay}
                  mappedOnly={calibrationDisplay.mappedOnly}
                  sourceOpacity={calibrationDisplay.referenceOpacity}
                />
              ) : (
                <SkeletonLayer
                  motion={motion}
                  visible={skeletonVisible}
                  playback={playback}
                  variant="source"
                />
              )}
              <CapsuleBodyLayer
                motion={motion}
                visible={capsuleVisible}
                playback={playback}
              />
              <BodyMeshLayer
                motion={motion}
                visible={bodyVisible}
                playback={playback}
                onReadyChange={publishBodyReady}
              />
              <EnvironmentLayer
                motion={motion}
                visible={environmentVisible}
                playback={playback}
              />
              <SkeletonLayer
                motion={scaledMotion}
                visible={scaledSkeletonVisible}
                // Before IK there is no robot timeline to compare against, so
                // keep the optional scaled preview on its reference frame.
                playback={robotTrajectory ? playback : undefined}
                variant="scaled"
                name="scaled-skeleton"
              />
              <EnvironmentLayer
                motion={scaledMotion}
                visible={scaledEnvironmentVisible}
                playback={robotTrajectory ? playback : undefined}
                timeline={robotTrajectory ?? scaledMotion}
                variant="scaled"
                name="scaled-environment"
              />
              <RobotLayer
                robot={robot}
                trajectory={robotTrajectory}
                playback={playback}
                visible={robotVisible}
                opacity={robotOpacity}
                onObjectChange={publishFollowedRobot}
                onPoseReaderChange={publishTargetPoseReader}
                selectedLink={selectedLink}
                hoveredLink={hoveredLink}
              />
            </>
          )}
          {calibrationInteraction && (
            <ManipulatorLayer
              interaction={calibrationInteraction}
              onDraggingChange={setManipulating}
              onHoveredJointChange={setHoveredJoint}
            />
          )}
        </group>
      </group>
    </>
  );
}

export function StageCanvas({
  motion,
  scaledMotion = null,
  robot = null,
  robotTrajectory = null,
  r2r = null,
  timeline,
  playback,
  onPlaybackChange,
  visibleLayers,
  robotOpacity = 1,
  cameraRevision = 0,
  followRobot = false,
  calibration = false,
  calibrationDisplay,
  calibrationInteraction = null,
  mappingOverlay,
}: {
  motion: StageMotionPayload | null;
  scaledMotion?: StageMotionPayload | null;
  robot?: StageRobotPayload | null;
  robotTrajectory?: StageRobotTrajectoryPayload | null;
  r2r?: StageR2rPresentationPayload | null;
  timeline: StageTimelinePayload | null;
  playback: StagePlaybackRef;
  onPlaybackChange?: () => void;
  visibleLayers: readonly StageLayerId[];
  robotOpacity?: number;
  cameraRevision?: number;
  followRobot?: boolean;
  calibration?: boolean;
  calibrationDisplay: CalibrationDisplayOptions;
  calibrationInteraction?: CalibrationInteractionModel | null;
  mappingOverlay?: RefObject<CalibrationMappingOverlayHandle | null>;
}) {
  return (
    <Canvas
      id="three-canvas"
      data-stage-renderer="react-three-fiber"
      className="absolute inset-0 block h-full w-full"
      flat
      dpr={[1, 2]}
      camera={{
        fov: 50,
        near: 0.01,
        far: 200,
        position: [2.6, 1.9, 3.2],
      }}
      gl={{ antialias: true, alpha: true }}
      onCreated={({ gl }) => {
        // Keep the CSS stage background visible through the transparent canvas.
        gl.setClearColor(0x000000, 0);
      }}
    >
      <StageScene
        motion={motion}
        scaledMotion={scaledMotion}
        robot={robot}
        robotTrajectory={robotTrajectory}
        r2r={r2r}
        timeline={timeline}
        playback={playback}
        onPlaybackChange={onPlaybackChange}
        visibleLayers={visibleLayers}
        robotOpacity={robotOpacity}
        cameraRevision={cameraRevision}
        followRobot={followRobot}
        calibration={calibration}
        calibrationDisplay={calibrationDisplay}
        calibrationInteraction={calibrationInteraction}
        mappingOverlay={mappingOverlay}
      />
    </Canvas>
  );
}
