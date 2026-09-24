import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useLocaleText } from "@/LocaleProvider";

import {
  timelineDuration,
  type StagePlaybackRef,
  type StagePlaybackState,
} from "./playback";
import {
  CalibrationMappingOverlay,
  type CalibrationMappingOverlayHandle,
} from "./CalibrationMappingOverlay";
import {
  DEFAULT_CALIBRATION_DISPLAY,
  type CalibrationDisplayOptions,
} from "./calibrationDisplay";
import { CalibrationJointHud } from "./CalibrationJointHud";
import type { CalibrationInteractionModel } from "./calibrationInteraction";
import { prepareReferenceSkeleton } from "./referenceSkeleton";
import { StageViewMenu } from "./StageViewMenu";
import { StageCanvas } from "./StageCanvas";
import { StageEmpty } from "./StageEmpty";
import { StagePlaybackBar } from "./StagePlaybackBar";
import {
  defaultR2rStageLayers,
  defaultStageLayers,
  r2rLayerAvailability,
  r2rPlaybackTimeline,
  r2rVisibilityIdentity,
  standardStageLayerAvailability,
  type StandardStageLayerId,
  type StagePresentation,
} from "./presentation";
import type {
  R2rStageLayerId,
  StageMotionPayload,
  StageLayerId,
  StageR2rPresentationPayload,
  StageRobotPayload,
  StageRobotTrajectoryPayload,
  StageTimelinePayload,
} from "./types";

export function Stage({
  motion = null,
  scaledMotion = null,
  robot = null,
  robotTrajectory = null,
  presentation = "empty",
  r2r = null,
  calibrationDisplay = DEFAULT_CALIBRATION_DISPLAY,
  calibrationInteraction = null,
  layerPreset = null,
}: {
  motion?: StageMotionPayload | null;
  scaledMotion?: StageMotionPayload | null;
  robot?: StageRobotPayload | null;
  robotTrajectory?: StageRobotTrajectoryPayload | null;
  presentation?: StagePresentation;
  r2r?: StageR2rPresentationPayload | null;
  calibrationDisplay?: CalibrationDisplayOptions;
  calibrationInteraction?: CalibrationInteractionModel | null;
  layerPreset?: readonly StageLayerId[] | null;
}) {
  const text = useLocaleText();
  const [visibleLayers, setVisibleLayers] = useState<StageLayerId[]>([]);
  const [r2rVisibleLayers, setR2rVisibleLayers] = useState<StageLayerId[]>([]);
  const [cameraRevision, setCameraRevision] = useState(0);
  const r2rVisibilityKey = useRef<string | null>(null);
  const mappingOverlay = useRef<CalibrationMappingOverlayHandle | null>(null);
  const timeline: StageTimelinePayload | null = r2r
    ? r2rPlaybackTimeline(r2r)
    : robotTrajectory && robotTrajectory.frames.length > 0
      ? robotTrajectory
      : motion;
  const playback = useMemo<StagePlaybackRef>(() => {
    const duration = timelineDuration(timeline);
    return {
      current: {
        elapsed: 0,
        frame: 0,
        duration,
        playing: Boolean(timeline && duration > 0),
      },
    };
  }, [timeline]);
  const [playbackView, setPlaybackView] = useState<{
    owner: StagePlaybackRef;
    value: StagePlaybackState;
  }>(
    () => ({ owner: playback, value: { ...playback.current } }),
  );
  const playbackSnapshot = playbackView.owner === playback
    ? playbackView.value
    : playback.current;
  const availability = useMemo(
    () => standardStageLayerAvailability({ motion, scaledMotion, robot }),
    [motion, robot, scaledMotion],
  );
  const calibrating = r2r
    ? r2r.phase === "calibration"
    : presentation.endsWith("-calibration");
  const activeCalibrationInteraction = calibrating
    ? calibrationInteraction
    : null;
  const calibrationReference = r2r ? r2r.calibrationReference : motion;
  const calibrationRobot = r2r ? r2r.target.robot : robot;
  const calibrationMappings = useMemo(
    () =>
      calibrating && calibrationReference
        ? prepareReferenceSkeleton(calibrationReference, calibrationRobot).mappings
        : [],
    [calibrating, calibrationReference, calibrationRobot],
  );
  const r2rAvailability = useMemo(
    () => (r2r ? r2rLayerAvailability(r2r) : undefined),
    [r2r],
  );
  const availablePresetLayers = useMemo(
    () =>
      layerPreset?.filter((layer) =>
        layer.startsWith("r2r-")
          ? Boolean(r2rAvailability?.[layer as R2rStageLayerId])
          : availability[layer as StandardStageLayerId],
      ) ?? null,
    [availability, layerPreset, r2rAvailability],
  );
  const r2rCameraIdentity = r2r ? r2rVisibilityIdentity(r2r) : "";
  const activeLayers = r2r ? r2rVisibleLayers : visibleLayers;
  const setActiveLayers = r2r ? setR2rVisibleLayers : setVisibleLayers;
  const r2rHasContent = Boolean(
    r2r &&
      (r2r.calibrationReference ||
        r2r.source.robot ||
        r2r.source.skeleton ||
        r2r.source.environment ||
        r2r.target.robot ||
        r2r.target.skeleton ||
        r2r.target.environment),
  );
  const hasContent = r2r
    ? r2rHasContent
    : motion !== null || scaledMotion !== null || robot !== null;
  const publishPlayback = useCallback(() => {
    setPlaybackView({ owner: playback, value: { ...playback.current } });
  }, [playback]);

  // New inputs and view transitions receive deterministic legacy defaults.
  // Manual toggles remain local until one of those presentation facts changes.
  useEffect(() => {
    if (r2r) return;
    setVisibleLayers(defaultStageLayers({
      mode: presentation,
      motion,
      scaledMotion,
      robot,
      robotTrajectory,
    }));
  }, [motion, presentation, r2r, robot, robotTrajectory, scaledMotion]);

  useEffect(() => {
    if (!r2r) return;
    const identity = r2rVisibilityIdentity(r2r);
    if (r2rVisibilityKey.current === identity) return;
    r2rVisibilityKey.current = identity;
    setR2rVisibleLayers(defaultR2rStageLayers(r2r));
  }, [r2r]);

  useEffect(() => {
    if (!availablePresetLayers) return;
    if (r2r) setR2rVisibleLayers(availablePresetLayers);
    else setVisibleLayers(availablePresetLayers);
  }, [availablePresetLayers, r2r]);

  useEffect(() => {
    if (hasContent) setCameraRevision((revision) => revision + 1);
  }, [hasContent, motion, presentation, r2rCameraIdentity, robot, scaledMotion]);

  return (
    <main
      className="app-content @container relative col-start-2 row-start-2 min-h-0 min-w-0 overflow-hidden bg-stage-canvas"
      aria-label={text("Workspace content", "工作区内容")}
    >
      <StageCanvas
        motion={motion}
        scaledMotion={scaledMotion}
        robot={robot}
        robotTrajectory={robotTrajectory}
        r2r={r2r}
        timeline={timeline}
        playback={playback}
        onPlaybackChange={publishPlayback}
        visibleLayers={activeLayers}
        robotOpacity={calibrating ? calibrationDisplay.robotOpacity : 1}
        cameraRevision={cameraRevision}
        followRobot={
          presentation === "h2r-result" && activeLayers.includes("robot")
        }
        calibration={calibrating}
        calibrationDisplay={calibrationDisplay}
        calibrationInteraction={activeCalibrationInteraction}
        mappingOverlay={mappingOverlay}
      />
      <CalibrationMappingOverlay
        ref={mappingOverlay}
        mappings={calibrationMappings}
        visible={calibrating}
        labels={calibrationDisplay.labels}
        mappingLines={calibrationDisplay.mappingLines}
      />
      {activeCalibrationInteraction?.selectedJoint && (
        <CalibrationJointHud
          interaction={activeCalibrationInteraction}
          jointName={activeCalibrationInteraction.selectedJoint}
          onClose={() =>
            activeCalibrationInteraction.onSelectedJointChange(null)
          }
        />
      )}
      <StageEmpty
        visible={
          r2r
            ? !r2rHasContent
            : motion === null && scaledMotion === null && robot === null
        }
      />
      <StageViewMenu
        value={activeLayers}
        onValueChange={setActiveLayers}
        availability={availability}
        calibration={calibrating}
        r2rAvailability={r2rAvailability}
      />
      {hasContent && (
        <div className="pointer-events-none absolute right-3 bottom-3 left-3 z-20 flex min-w-0 items-center gap-2.5">
          <button
            type="button"
            className="pointer-events-auto grid size-11 shrink-0 cursor-pointer place-items-center rounded-full border-[1.5px] border-primary/30 bg-surface/[.88] text-primary shadow-[0_1px_3px_rgba(0,0,0,0.08)] backdrop-blur-[20px] transition hover:border-primary hover:bg-primary/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring active:scale-95 @max-[520px]:size-[38px]"
            aria-label={text("Reset view", "回到默认视角")}
            title={text("Reset view", "回到默认视角")}
            onClick={() => setCameraRevision((revision) => revision + 1)}
          >
            <span
              className="size-[22px] bg-current [mask:url(/icons/stage/reset-view.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/stage/reset-view.svg)_center/contain_no-repeat]"
              aria-hidden="true"
            />
          </button>
          {!calibrating && (
            <StagePlaybackBar
              timeline={timeline}
              playback={playback}
              snapshot={playbackSnapshot}
              onChange={publishPlayback}
            />
          )}
        </div>
      )}
    </main>
  );
}
