import type {
  StageMotionPayload,
  StageTimelinePayload,
} from "./types";

/** Mutable cursor shared by all animated R3F layers in one Stage. */
export interface StagePlaybackState {
  elapsed: number;
  frame: number;
  duration: number;
  playing: boolean;
  /** Optional until the user changes it; defaults are shared across new clips. */
  speed?: number;
  loop?: boolean;
}

export interface StagePlaybackRef {
  readonly current: StagePlaybackState;
}

export const MIN_PLAYBACK_SPEED = 0.1;
export const MAX_PLAYBACK_SPEED = 4;
export const DEFAULT_PLAYBACK_SPEED = 1;
export const DEFAULT_PLAYBACK_LOOP = true;

let sessionSpeed = DEFAULT_PLAYBACK_SPEED;
let sessionLoop = DEFAULT_PLAYBACK_LOOP;

export function normalizePlaybackSpeed(value: unknown): number {
  const speed = Number(value);
  return Number.isFinite(speed)
    ? Math.min(MAX_PLAYBACK_SPEED, Math.max(MIN_PLAYBACK_SPEED, speed))
    : DEFAULT_PLAYBACK_SPEED;
}

export function playbackSpeed(state: StagePlaybackState): number {
  return state.speed === undefined
    ? sessionSpeed
    : normalizePlaybackSpeed(state.speed);
}

export function playbackLoop(state: StagePlaybackState): boolean {
  return typeof state.loop === "boolean" ? state.loop : sessionLoop;
}

export function setPlaybackSpeed(
  state: StagePlaybackState,
  value: unknown,
): number {
  const speed = normalizePlaybackSpeed(value);
  state.speed = speed;
  sessionSpeed = speed;
  return speed;
}

export function setPlaybackLoop(
  state: StagePlaybackState,
  value: boolean,
): boolean {
  state.loop = value;
  sessionLoop = value;
  return value;
}

export function togglePlaybackLoop(state: StagePlaybackState): boolean {
  return setPlaybackLoop(state, !playbackLoop(state));
}

export type PlaybackAdvanceResult = "idle" | "advanced" | "looped" | "ended";

export function timelineFrameCount(timeline: StageTimelinePayload | null): number {
  if (!timeline) return 0;
  return "frames" in timeline ? timeline.frames.length : timeline.positions.length;
}

/** Return the full duration represented by a motion or robot trajectory. */
export function timelineDuration(timeline: StageTimelinePayload | null): number {
  if (!timeline) return 0;
  for (const declared of [timeline.playback_duration, timeline.duration]) {
    if (typeof declared === "number" && Number.isFinite(declared) && declared > 0) {
      return declared;
    }
  }
  const framerate = timeline.framerate ?? timeline.sample_rate;
  if (typeof framerate === "number" && Number.isFinite(framerate) && framerate > 0) {
    const frameCount =
      timeline.num_frames_total ??
      timeline.playback_frames ??
      timelineFrameCount(timeline);
    return Math.max(0, (frameCount - 1) / framerate);
  }
  const frameCount = timelineFrameCount(timeline);
  return frameCount > 1 ? (frameCount - 1) / 30 : 0;
}

/** Convert elapsed seconds into one payload's serialized-frame coordinate. */
export function timelineFrameAtTime(
  timeline: StageTimelinePayload | null,
  elapsedSeconds: number,
): number {
  const frameCount = timelineFrameCount(timeline);
  if (frameCount < 2) return 0;
  const duration = timelineDuration(timeline);
  if (duration <= 0) return 0;
  // The owner clock performs looping. Individual layers clamp so a shorter
  // secondary payload holds its last frame instead of wrapping out of sync.
  const normalized = Math.min(1, Math.max(0, elapsedSeconds / duration));
  return normalized * (frameCount - 1);
}

/** Advance the shared cursor with the original speed and loop semantics. */
export function advancePlayback(
  state: StagePlaybackState,
  timeline: StageTimelinePayload | null,
  deltaSeconds: number,
): PlaybackAdvanceResult {
  if (
    !timeline ||
    !state.playing ||
    !Number.isFinite(state.duration) ||
    state.duration <= 0
  ) {
    return "idle";
  }

  // A backgrounded tab must not carry a large clock jump into the next frame.
  const delta = Number.isFinite(deltaSeconds)
    ? Math.min(0.1, Math.max(0, deltaSeconds))
    : 0;
  const step = delta * playbackSpeed(state);
  if (step <= 0) return "idle";

  const elapsed = Number.isFinite(state.elapsed) ? Math.max(0, state.elapsed) : 0;
  const next = elapsed + step;
  if (next >= state.duration) {
    if (playbackLoop(state)) {
      // Restart on the exact first pose; modulo overshoot looks like a jump.
      state.elapsed = 0;
      state.frame = 0;
      return "looped";
    }
    state.elapsed = state.duration;
    state.frame = timelineFrameAtTime(timeline, state.duration);
    state.playing = false;
    return "ended";
  }

  state.elapsed = next;
  state.frame = timelineFrameAtTime(timeline, next);
  return "advanced";
}

/** Compatibility names retained for motion-only layer tests and callers. */
export function motionDuration(motion: StageMotionPayload | null): number {
  return timelineDuration(motion);
}

export function frameAtTime(
  motion: StageMotionPayload | null,
  elapsedSeconds: number,
): number {
  return timelineFrameAtTime(motion, elapsedSeconds);
}
