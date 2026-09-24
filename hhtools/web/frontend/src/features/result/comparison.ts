import type { StageLayerId } from "@/stage/types";

export type ComparisonPreset = "source" | "target" | "result" | "overlay";
export type ComparisonWorkflow = "h2r" | "r2r";

export const COMPARISON_PRESETS = [
  "source",
  "target",
  "result",
  "overlay",
] as const satisfies readonly ComparisonPreset[];

export const DEFAULT_COMPARISON_PRESET: ComparisonPreset = "overlay";
export const COMPARISON_PRESET_STORAGE_KEY = "hhtools.comparison-presets";

type ComparisonPresetState = Partial<Record<ComparisonWorkflow, ComparisonPreset>>;

interface ComparisonStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

const COMPARISON_LAYERS: Readonly<
  Record<ComparisonWorkflow, Readonly<Record<ComparisonPreset, readonly StageLayerId[]>>>
> = {
  h2r: {
    source: ["skeleton", "body", "objects"],
    target: ["scaled-skeleton", "scaled-scene"],
    result: ["scaled-scene", "robot"],
    overlay: ["skeleton", "scaled-skeleton", "scaled-scene", "robot"],
  },
  r2r: {
    source: ["r2r-source-robot", "r2r-source-scene"],
    target: ["r2r-target-skeleton", "r2r-target-scene"],
    result: ["r2r-target-robot", "r2r-target-scene"],
    overlay: [
      "r2r-source-robot",
      "r2r-target-robot",
      "r2r-target-skeleton",
      "r2r-target-scene",
    ],
  },
};

/** Project one comparison intent without coupling the result panel to Stage. */
export function comparisonLayers(
  workflow: ComparisonWorkflow,
  preset: ComparisonPreset,
): readonly StageLayerId[] {
  return COMPARISON_LAYERS[workflow][preset];
}

export function isComparisonPreset(value: unknown): value is ComparisonPreset {
  return COMPARISON_PRESETS.some((preset) => preset === value);
}

function readState(
  storage: Pick<ComparisonStorage, "getItem"> | undefined,
): ComparisonPresetState {
  try {
    const value = JSON.parse(
      storage?.getItem(COMPARISON_PRESET_STORAGE_KEY) ?? "{}",
    ) as unknown;
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    const record = value as Readonly<Record<string, unknown>>;
    return {
      ...(isComparisonPreset(record.h2r) ? { h2r: record.h2r } : {}),
      ...(isComparisonPreset(record.r2r) ? { r2r: record.r2r } : {}),
    };
  } catch {
    return {};
  }
}

/** Read one validated UI-only preset without coupling feature state to globals. */
export function storedComparisonPreset(
  storage: Pick<ComparisonStorage, "getItem"> | undefined,
  workflow: ComparisonWorkflow,
): ComparisonPreset {
  return readState(storage)[workflow] ?? DEFAULT_COMPARISON_PRESET;
}

/** Persist one workflow while preserving the other workflow's valid preset. */
export function storeComparisonPreset(
  storage: ComparisonStorage | undefined,
  workflow: ComparisonWorkflow,
  preset: ComparisonPreset,
): void {
  if (!storage) return;
  const next = { ...readState(storage), [workflow]: preset };
  try {
    storage.setItem(COMPARISON_PRESET_STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Restricted browser contexts may reject persistence; live React state remains valid.
  }
}
