import type { DatasetClip } from "./api.ts";

export interface AnalysisFilters {
  readonly tag: string;
  readonly kind: string;
  readonly folder: string;
  readonly metric?: string;
  readonly metricRange?: readonly [number, number] | null;
}

export interface ScatterBounds {
  readonly minX: number;
  readonly maxX: number;
  readonly minY: number;
  readonly maxY: number;
}

export type ScatterClip = DatasetClip & {
  readonly scatter: readonly [number, number];
};

export function clipsWithScatter(clips: readonly DatasetClip[]): ScatterClip[] {
  return clips.filter(
    (clip): clip is ScatterClip =>
      Array.isArray(clip.scatter) &&
      clip.scatter.length === 2 &&
      Number.isFinite(clip.scatter[0]) &&
      Number.isFinite(clip.scatter[1]),
  );
}

/** Bounds always come from the complete valid result, never the filtered view. */
export function scatterBounds(clips: readonly ScatterClip[]): ScatterBounds | null {
  if (!clips.length) return null;
  const xs = clips.map((clip) => clip.scatter[0]);
  const ys = clips.map((clip) => clip.scatter[1]);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
}

export function clipMatchesFilters(
  clip: DatasetClip,
  filters: AnalysisFilters,
): boolean {
  if (filters.tag !== "all" && !clip.tags.includes(filters.tag)) return false;
  if (filters.kind !== "all" && clip.source_kind !== filters.kind) return false;
  if (filters.folder !== "all" && clip.folder_label !== filters.folder) return false;
  if (!filters.metric || !filters.metricRange) return true;
  const value = Number(clip.metrics[filters.metric]);
  return (
    Number.isFinite(value) &&
    value >= filters.metricRange[0] &&
    value <= filters.metricRange[1]
  );
}

/**
 * Legacy scatter semantics: a normal click selects only that point (or clears
 * the sole selection); Shift toggles one point without disturbing the rest.
 * Recommended subset points are previewable but not duplicated manually.
 */
export function selectScatterClip(
  current: ReadonlySet<string>,
  id: string,
  additive: boolean,
  recommended: ReadonlySet<string>,
): Set<string> {
  const next = new Set(current);
  if (recommended.has(id)) return next;
  if (additive) {
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  }
  if (next.has(id) && next.size === 1) return new Set();
  return new Set([id]);
}
