import type {
  R2rLayerAvailability,
  R2rStageLayerId,
  StageLayerId,
  StageMotionPayload,
  StageR2rPresentationPayload,
  StageRobotPayload,
  StageRobotTrajectoryPayload,
  StageTimelinePayload,
} from "./types";
import { timelineDuration } from "./playback.ts";

export type StagePresentation =
  | "motion"
  | "robot"
  | "h2r"
  | "h2r-result"
  | "h2r-calibration"
  | "r2r"
  | "r2r-result"
  | "r2r-calibration"
  | "analysis"
  | "video-to-motion"
  | "empty";

interface PresentationInput {
  readonly mode: StagePresentation;
  readonly motion: StageMotionPayload | null;
  readonly scaledMotion: StageMotionPayload | null;
  readonly robot: StageRobotPayload | null;
  readonly robotTrajectory: StageRobotTrajectoryPayload | null;
}

export type StandardStageLayerId = Exclude<StageLayerId, R2rStageLayerId>;
export type StandardStageLayerAvailability = Readonly<
  Record<StandardStageLayerId, boolean>
>;

function hasEnvironment(motion: StageMotionPayload | null): boolean {
  return Boolean(
    motion?.terrain || (motion?.objects && motion.objects.length > 0),
  );
}

/** Renderer capabilities that drive the disabled state of the H2R controls. */
export function standardStageLayerAvailability({
  motion,
  scaledMotion,
  robot,
}: Pick<
  PresentationInput,
  "motion" | "scaledMotion" | "robot"
>): StandardStageLayerAvailability {
  const hasSkeleton = Boolean(motion?.positions.length);
  const hasBody = Boolean(
    motion &&
      (motion.body_mesh?.available ||
        (motion.positions.length > 0 && motion.parent_indices.length > 0)),
  );
  return {
    skeleton: hasSkeleton,
    body: hasBody,
    objects: hasEnvironment(motion),
    "scaled-skeleton": Boolean(scaledMotion?.positions.length),
    "scaled-scene": hasEnvironment(scaledMotion),
    robot: Boolean(robot?.links.length),
  };
}

/** Reference skeleton visibility is independent from the six legacy controls. */
export function projectStageMenuValue(
  requested: readonly StageLayerId[],
  calibration: boolean,
  r2r: boolean,
): StageLayerId[] {
  return calibration && !r2r
    ? requested.filter((layer) => layer === "robot")
    : [...requested];
}

/** Preserve the hidden reference bit when calibration exposes only Robot. */
export function applyStageMenuValue(
  current: readonly StageLayerId[],
  next: readonly StageLayerId[],
  calibration: boolean,
  r2r: boolean,
): StageLayerId[] {
  if (!calibration || r2r) return [...next];
  const preserved = current.filter((layer) => layer !== "robot");
  return next.includes("robot") ? [...preserved, "robot"] : preserved;
}

export function r2rLayerAvailability(
  presentation: StageR2rPresentationPayload,
): R2rLayerAvailability {
  return {
    "r2r-source-robot": Boolean(presentation.source.robot?.links.length),
    "r2r-source-skeleton": Boolean(presentation.source.skeleton?.positions.length),
    "r2r-source-scene": hasEnvironment(presentation.source.environment),
    "r2r-target-robot": Boolean(
      presentation.target.robot?.links.length && presentation.phase !== "source",
    ),
    "r2r-target-skeleton": Boolean(presentation.target.skeleton?.positions.length),
    "r2r-target-scene": hasEnvironment(presentation.target.environment),
  };
}

/** Reproduce the old source and overlay presets for each new R2R payload. */
export function defaultR2rStageLayers(
  presentation: StageR2rPresentationPayload,
): StageLayerId[] {
  const available = r2rLayerAvailability(presentation);
  if (presentation.phase === "calibration") {
    return available["r2r-target-robot"] ? ["r2r-target-robot"] : [];
  }
  if (presentation.phase === "result") {
    const overlay: readonly R2rStageLayerId[] = [
      "r2r-source-robot",
      "r2r-target-robot",
      "r2r-target-skeleton",
      "r2r-target-scene",
    ];
    return overlay.filter((layer) => available[layer]);
  }
  const source: readonly R2rStageLayerId[] = [
    "r2r-source-robot",
    "r2r-source-scene",
  ];
  return source.filter((layer) => available[layer]);
}

export interface R2rStageVisibilityPlan {
  readonly sourceRobot: boolean;
  readonly sourceSkeleton: boolean;
  readonly sourceScene: boolean;
  readonly targetRobot: boolean;
  readonly targetSkeleton: boolean;
  readonly targetScene: boolean;
  readonly calibrationReference: boolean;
}

/** Calibration is a physical Stage mode; ordinary toggles cannot leak into it. */
export function projectR2rStageVisibility(
  presentation: StageR2rPresentationPayload,
  requested: readonly StageLayerId[],
): R2rStageVisibilityPlan {
  const available = r2rLayerAvailability(presentation);
  if (presentation.phase === "calibration") {
    return {
      sourceRobot: false,
      sourceSkeleton: false,
      sourceScene: false,
      targetRobot: available["r2r-target-robot"],
      targetSkeleton: false,
      targetScene: false,
      calibrationReference: presentation.calibrationReference !== null,
    };
  }
  return {
    sourceRobot:
      available["r2r-source-robot"] && requested.includes("r2r-source-robot"),
    sourceSkeleton:
      available["r2r-source-skeleton"] && requested.includes("r2r-source-skeleton"),
    sourceScene:
      available["r2r-source-scene"] && requested.includes("r2r-source-scene"),
    targetRobot:
      available["r2r-target-robot"] && requested.includes("r2r-target-robot"),
    targetSkeleton:
      available["r2r-target-skeleton"] && requested.includes("r2r-target-skeleton"),
    targetScene:
      available["r2r-target-scene"] && requested.includes("r2r-target-scene"),
    calibrationReference: false,
  };
}

/** Ignore navigation churn while resetting defaults for new workflow identities. */
export function r2rVisibilityIdentity(
  presentation: StageR2rPresentationPayload,
): string {
  return [
    presentation.phase,
    presentation.source.robot?.name ?? "",
    presentation.target.robot?.name ?? "",
    presentation.sourceToken ?? "",
    presentation.resultToken ?? "",
  ].join("|");
}

/** Pick one clock that spans every visible R2R actor on the shared canvas. */
export function r2rPlaybackTimeline(
  presentation: StageR2rPresentationPayload,
): StageTimelinePayload | null {
  if (presentation.phase === "calibration") return null;
  const timelines: readonly (StageTimelinePayload | null)[] = [
    presentation.target.trajectory,
    presentation.source.trajectory,
    presentation.target.skeleton,
    presentation.source.skeleton,
  ];
  let longest: StageTimelinePayload | null = null;
  for (const timeline of timelines) {
    if (timeline && timelineDuration(timeline) > timelineDuration(longest)) {
      longest = timeline;
    }
  }
  return longest;
}

function isParcMotion(motion: StageMotionPayload): boolean {
  return (
    motion.dataset === "parc_ms" ||
    motion.meta?.dataset === "parc_ms" ||
    motion.source_format === "parc_ms_pkl" ||
    motion.meta?.source_format === "parc_ms_pkl"
  );
}

/** Project legacy display defaults without coupling Three components to workflows. */
export function defaultStageLayers({
  mode,
  motion,
  scaledMotion,
  robot,
  robotTrajectory,
}: PresentationInput): StageLayerId[] {
  const available = standardStageLayerAvailability({ motion, scaledMotion, robot });
  if (mode === "empty") return [];
  if (mode === "robot") return available.robot ? ["robot"] : [];

  if (mode === "h2r-calibration" || mode === "r2r-calibration") {
    return [
      ...(available.skeleton ? ["skeleton" as const] : []),
      ...(available.robot ? ["robot" as const] : []),
    ];
  }

  if (mode === "h2r-result" || mode === "r2r-result") {
    return [
      ...(available.skeleton ? ["skeleton" as const] : []),
      ...(available["scaled-skeleton"] ? ["scaled-skeleton" as const] : []),
      ...(available["scaled-scene"] ? ["scaled-scene" as const] : []),
      ...(available.robot ? ["robot" as const] : []),
    ];
  }

  if (mode === "analysis" && available.robot) {
    return [
      ...(available["scaled-scene"] ? ["scaled-scene" as const] : []),
      "robot",
    ];
  }

  const layers: StageLayerId[] = [];
  if (motion) {
    const hasSkin = motion.body_mesh?.available === true;
    if (available.skeleton && (!hasSkin || isParcMotion(motion))) {
      layers.push("skeleton");
    }
    if (available.body && hasSkin) layers.push("body");
    if (available.objects) layers.push("objects");
  }

  const comparison = mode === "h2r" || mode === "r2r";
  // H2R preloads its calibrated overlay for manual inspection, but the legacy
  // presentation keeps it hidden until a result exists or the user enables it.
  if (mode === "r2r" && (scaledMotion || robotTrajectory)) {
    if (available.skeleton && !layers.includes("skeleton")) layers.push("skeleton");
    if (available["scaled-skeleton"]) layers.push("scaled-skeleton");
    if (available["scaled-scene"]) layers.push("scaled-scene");
  }
  if (comparison && available.robot) layers.push("robot");
  return layers;
}
