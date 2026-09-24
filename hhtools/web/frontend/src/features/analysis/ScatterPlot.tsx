import { useEffect, useMemo, useRef, useState, type PointerEvent, type WheelEvent } from "react";

import { useLocaleText } from "@/LocaleProvider";

import type { DatasetClip } from "./api";
import { clipsWithScatter, scatterBounds } from "./model";

const WIDTH = 640;
const HEIGHT = 220;
const CENTER_X = WIDTH / 2;
const CENTER_Y = HEIGHT / 2;
const CLUSTER_COLORS = ["#0071e3", "#e05d44", "#2da44e", "#8250df", "#bf8700", "#0a7b83"];

interface ViewTransform {
  readonly scale: number;
  readonly panX: number;
  readonly panY: number;
}

interface DragState {
  readonly pointerId: number;
  readonly lastX: number;
  readonly lastY: number;
}

export interface ScatterPlotProps {
  readonly clips: readonly DatasetClip[];
  readonly visibleIds: ReadonlySet<string>;
  readonly selectedIds: ReadonlySet<string>;
  readonly subsetIds: ReadonlySet<string>;
  readonly onActivate: (clip: DatasetClip, additive: boolean) => void;
}

function label(clip: DatasetClip, fallback = "clip"): string {
  return clip.clip_id || clip.source_path.split(/[\\/]/).pop() || fallback;
}

function clusterColor(cluster: number | null): string {
  if (cluster === null || cluster < 0) return "#8c959f";
  return CLUSTER_COLORS[cluster % CLUSTER_COLORS.length];
}

function svgPoint(event: WheelEvent<SVGSVGElement>): { x: number; y: number } {
  const rect = event.currentTarget.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * WIDTH,
    y: ((event.clientY - rect.top) / rect.height) * HEIGHT,
  };
}

export function ScatterPlot({
  clips,
  visibleIds,
  selectedIds,
  subsetIds,
  onActivate,
}: ScatterPlotProps) {
  const text = useLocaleText();
  const clipName = (clip: DatasetClip) => label(clip, text("clip", "片段"));
  const points = useMemo(() => clipsWithScatter(clips), [clips]);
  const bounds = useMemo(() => scatterBounds(points), [points]);
  const [view, setView] = useState<ViewTransform>({ scale: 1, panX: 0, panY: 0 });
  const [hoverId, setHoverId] = useState<string | null>(null);
  const drag = useRef<DragState | null>(null);
  const dragDistance = useRef(0);

  useEffect(() => {
    setView({ scale: 1, panX: 0, panY: 0 });
    setHoverId(null);
  }, [clips]);

  if (!bounds) {
    return <p className="text-xs text-muted-foreground">{text("No embedding coordinates.", "没有嵌入坐标。")}</p>;
  }

  const baseX = (value: number) =>
    18 + ((value - bounds.minX) / Math.max(bounds.maxX - bounds.minX, 1e-6)) * 604;
  const baseY = (value: number) =>
    198 - ((value - bounds.minY) / Math.max(bounds.maxY - bounds.minY, 1e-6)) * 176;
  const project = (clip: (typeof points)[number]) => ({
    x: CENTER_X + view.panX + (baseX(clip.scatter[0]) - CENTER_X) * view.scale,
    y: CENTER_Y + view.panY + (baseY(clip.scatter[1]) - CENTER_Y) * view.scale,
  });
  const hoverClip = points.find((clip) => clip.clip_id === hoverId) ?? null;
  const hoverPoint = hoverClip ? project(hoverClip) : null;

  function reset(): void {
    setView({ scale: 1, panX: 0, panY: 0 });
  }

  function handleWheel(event: WheelEvent<SVGSVGElement>): void {
    event.preventDefault();
    const cursor = svgPoint(event);
    setView((current) => {
      const scale = Math.max(0.25, Math.min(10, current.scale * (event.deltaY > 0 ? 0.9 : 1.1)));
      const ratio = scale / current.scale;
      return {
        scale,
        panX: cursor.x - CENTER_X - (cursor.x - CENTER_X - current.panX) * ratio,
        panY: cursor.y - CENTER_Y - (cursor.y - CENTER_Y - current.panY) * ratio,
      };
    });
  }

  function handlePointerDown(event: PointerEvent<SVGSVGElement>): void {
    if (event.button !== 0) return;
    if (event.target instanceof Element && event.target.closest("[data-scatter-point]")) {
      dragDistance.current = 0;
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { pointerId: event.pointerId, lastX: event.clientX, lastY: event.clientY };
    dragDistance.current = 0;
  }

  function handlePointerMove(event: PointerEvent<SVGSVGElement>): void {
    const current = drag.current;
    if (!current || current.pointerId !== event.pointerId) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const dx = ((event.clientX - current.lastX) / rect.width) * WIDTH;
    const dy = ((event.clientY - current.lastY) / rect.height) * HEIGHT;
    dragDistance.current += Math.abs(event.clientX - current.lastX) + Math.abs(event.clientY - current.lastY);
    drag.current = { ...current, lastX: event.clientX, lastY: event.clientY };
    setView((value) => ({ ...value, panX: value.panX + dx, panY: value.panY + dy }));
  }

  function finishDrag(event: PointerEvent<SVGSVGElement>): void {
    if (drag.current?.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    drag.current = null;
  }

  return (
    <div className="grid gap-2">
      <div className="flex items-center justify-between gap-2 text-[11px] text-muted-foreground">
        <span>{text("Embedding", "嵌入图")} · {Math.round(view.scale * 100)}%</span>
        <button
          type="button"
          className="grid size-7 place-items-center rounded-md text-muted-foreground hover:bg-accent hover:text-accent-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
          aria-label={text("Reset embedding view", "重置嵌入图视图")}
          title={text("Reset view", "重置视图")}
          onClick={reset}
        >
          <img className="size-4" src="/icons/stage/reset-view.svg" alt="" />
        </button>
      </div>
      <div className="relative">
        <svg
          className="h-[220px] w-full touch-none overflow-hidden rounded-md border border-border-subtle bg-background"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="group"
          aria-label={text("Embedding scatter plot", "嵌入散点图")}
          onWheel={handleWheel}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={finishDrag}
          onPointerCancel={finishDrag}
          onPointerLeave={() => {
            if (!drag.current) setHoverId(null);
          }}
        >
          <line x1="18" y1="198" x2="622" y2="198" className="stroke-border" />
          <line x1="18" y1="22" x2="18" y2="198" className="stroke-border" />
          {points.map((clip) => {
            const point = project(clip);
            const selected = selectedIds.has(clip.clip_id);
            const recommended = subsetIds.has(clip.clip_id);
            const visible = visibleIds.has(clip.clip_id);
            return (
              <circle
                key={clip.clip_id}
                data-scatter-point
                cx={point.x}
                cy={point.y}
                r={selected || recommended ? 6 : 4}
                fill={clusterColor(clip.cluster_id)}
                opacity={visible ? 1 : 0.16}
                stroke={selected ? "var(--text)" : recommended ? "#ff9f0a" : "var(--surface)"}
                strokeWidth={selected || recommended ? 2 : 1}
                className="cursor-pointer outline-none focus:stroke-[3px]"
                role="button"
                tabIndex={0}
                aria-label={text(`Preview ${clipName(clip)}`, `预览 ${clipName(clip)}`)}
                aria-pressed={selected}
                onPointerEnter={() => setHoverId(clip.clip_id)}
                onFocus={() => setHoverId(clip.clip_id)}
                onBlur={() => setHoverId(null)}
                onClick={(event) => {
                  event.stopPropagation();
                  if (dragDistance.current <= 4) onActivate(clip, event.shiftKey);
                }}
                onKeyDown={(event) => {
                  if (event.key !== "Enter" && event.key !== " ") return;
                  event.preventDefault();
                  onActivate(clip, event.shiftKey);
                }}
              >
                <title>{clipName(clip)}</title>
              </circle>
            );
          })}
        </svg>
        {hoverClip && hoverPoint && (
          <div
            className="pointer-events-none absolute z-10 max-w-[240px] -translate-x-1/2 rounded-md border border-border-subtle bg-foreground px-2 py-1 text-[10px] text-background shadow-sm"
            style={{
              left: `${(Math.max(16, Math.min(WIDTH - 16, hoverPoint.x)) / WIDTH) * 100}%`,
              top: `${(Math.max(4, Math.min(HEIGHT - 34, hoverPoint.y + 12)) / HEIGHT) * 100}%`,
            }}
            role="tooltip"
          >
            <span className="block truncate font-medium">{clipName(hoverClip)}</span>
            <span className="opacity-80">
              {text("Cluster", "聚类")} {hoverClip.cluster_id ?? "-"}
              {!visibleIds.has(hoverClip.clip_id)
                ? text(" · outside filter", " · 不在筛选范围内")
                : ""}
            </span>
          </div>
        )}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-muted-foreground">
        <span><i className="mr-1 inline-block size-2 rounded-full bg-[#ff9f0a]" />{text("recommended", "推荐")}</span>
        <span><i className="mr-1 inline-block size-2 rounded-full bg-foreground" />{text("manual", "手动")}</span>
      </div>
    </div>
  );
}
