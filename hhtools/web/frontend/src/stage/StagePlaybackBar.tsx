import type { ChangeEvent, CSSProperties } from "react";

import {
  MAX_PLAYBACK_SPEED,
  MIN_PLAYBACK_SPEED,
  playbackLoop,
  playbackSpeed,
  setPlaybackSpeed,
  timelineFrameCount,
  togglePlaybackLoop,
  type StagePlaybackRef,
  type StagePlaybackState,
} from "./playback";
import type { StageTimelinePayload } from "./types";

interface StagePlaybackBarProps {
  timeline: StageTimelinePayload | null;
  playback: StagePlaybackRef;
  snapshot: StagePlaybackState;
  onChange(): void;
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0:00.00";
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds % 60).toFixed(2).padStart(5, "0")}`;
}

function PlaybackButton({
  label,
  icon,
  onClick,
  primary = false,
}: {
  label: string;
  icon: string;
  onClick(): void;
  primary?: boolean;
}) {
  return (
    <button
      type="button"
      className={
        primary
          ? "grid size-8 shrink-0 cursor-pointer place-items-center rounded-md border border-primary bg-primary text-primary-foreground hover:bg-accent-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
          : "grid size-8 shrink-0 cursor-pointer place-items-center rounded-md border border-transparent text-muted-foreground hover:bg-accent hover:text-accent-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
      }
      aria-label={label}
      title={label}
      onClick={onClick}
    >
      <span
        className="size-4 bg-current [mask:var(--playback-icon)_center/contain_no-repeat] [-webkit-mask:var(--playback-icon)_center/contain_no-repeat]"
        style={{ "--playback-icon": `url(${icon})` } as CSSProperties}
        aria-hidden="true"
      />
    </button>
  );
}

/** Compact controls for the renderer-owned shared motion cursor. */
export function StagePlaybackBar({
  timeline,
  playback,
  snapshot,
  onChange,
}: StagePlaybackBarProps) {
  const frameCount = timelineFrameCount(timeline);
  if (!timeline || frameCount < 2) return null;
  const maximumFrame = Math.max(0, frameCount - 1);
  const speed = playbackSpeed(snapshot);
  const loop = playbackLoop(snapshot);

  const seek = (frame: number, pause = false): void => {
    const cursor = playback.current;
    const nextFrame = Math.min(maximumFrame, Math.max(0, frame));
    cursor.frame = nextFrame;
    cursor.elapsed = maximumFrame > 0
      ? (nextFrame / maximumFrame) * cursor.duration
      : 0;
    if (pause) cursor.playing = false;
    onChange();
  };

  const step = (direction: -1 | 1): void => {
    seek(Math.round(playback.current.frame) + direction, true);
  };

  const scrub = (event: ChangeEvent<HTMLInputElement>): void => {
    seek(Number(event.currentTarget.value));
  };

  const toggle = (): void => {
    const cursor = playback.current;
    if (cursor.frame >= maximumFrame && !cursor.playing) seek(0);
    cursor.playing = !cursor.playing;
    onChange();
  };

  const changeSpeed = (value: unknown): void => {
    setPlaybackSpeed(playback.current, value);
    onChange();
  };

  const toggleLoop = (): void => {
    togglePlaybackLoop(playback.current);
    onChange();
  };

  return (
    <div
      className="pointer-events-auto flex min-w-0 flex-1 items-center gap-1.5 rounded-lg border border-border-subtle bg-surface/[.88] px-2.5 py-2 shadow-[0_1px_3px_rgba(0,0,0,0.08)] backdrop-blur-[20px]"
      role="group"
      aria-label="Motion playback"
      data-playback-frame={Math.round(snapshot.frame)}
    >
      <PlaybackButton
        label="Previous frame"
        icon="/icons/playback/step-back.svg"
        onClick={() => step(-1)}
      />
      <PlaybackButton
        label={snapshot.playing ? "Pause" : "Play"}
        icon={snapshot.playing ? "/icons/playback/pause.svg" : "/icons/playback/play.svg"}
        onClick={toggle}
        primary
      />
      <PlaybackButton
        label="Next frame"
        icon="/icons/playback/step-forward.svg"
        onClick={() => step(1)}
      />
      <input
        className="mx-1 h-4 min-w-12 flex-1 cursor-pointer accent-primary"
        type="range"
        min={0}
        max={maximumFrame}
        step={1}
        value={Math.min(maximumFrame, Math.max(0, Math.round(snapshot.frame)))}
        aria-label="Motion frame"
        aria-valuetext={`Frame ${Math.round(snapshot.frame) + 1} of ${frameCount}`}
        onChange={scrub}
      />
      <span className="min-w-[88px] text-right text-[11px] leading-tight text-muted-foreground tabular-nums @max-[440px]:hidden">
        {Math.round(snapshot.frame) + 1} / {frameCount}
      </span>
      <span className="min-w-[86px] text-right text-[11px] leading-tight text-muted-foreground tabular-nums @max-[520px]:hidden">
        {formatTime(snapshot.elapsed)} / {formatTime(snapshot.duration)}
      </span>
      <label
        className="flex shrink-0 items-center gap-1.5 text-[11px] text-muted-foreground select-none @max-[390px]:hidden"
        title="Playback speed; double-click to reset to 1x"
        onDoubleClick={() => changeSpeed(1)}
      >
        <span className="sr-only">Playback speed</span>
        <input
          className="h-4 w-[70px] cursor-pointer accent-primary @max-[520px]:hidden"
          type="range"
          min={MIN_PLAYBACK_SPEED}
          max={MAX_PLAYBACK_SPEED}
          step={0.1}
          value={speed}
          aria-label="Playback speed"
          onChange={(event) => changeSpeed(event.currentTarget.value)}
        />
        <span className="min-w-8 text-right tabular-nums">{speed.toFixed(1)}×</span>
      </label>
      <button
        type="button"
        className={
          loop
            ? "grid size-8 shrink-0 cursor-pointer place-items-center rounded-md border border-transparent bg-primary/10 text-primary transition-colors hover:bg-primary/15 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
            : "grid size-8 shrink-0 cursor-pointer place-items-center rounded-md border border-transparent text-muted-foreground opacity-40 transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
        }
        aria-label="Toggle loop playback"
        aria-pressed={loop}
        title={loop ? "Loop playback: on" : "Loop playback: off"}
        onClick={toggleLoop}
      >
        <span
          className="size-4 bg-current [mask:url(/icons/playback/loop.svg)_center/contain_no-repeat] [-webkit-mask:url(/icons/playback/loop.svg)_center/contain_no-repeat]"
          aria-hidden="true"
        />
      </button>
    </div>
  );
}
