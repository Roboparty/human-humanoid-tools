export type ExportFormat = "csv" | "pkl";

export interface TrackingDiagnosticPoint {
  readonly frame: number;
  readonly time_s: number;
  readonly mean_error_m: number;
  readonly max_error_m: number;
  readonly source_contacts: number;
  readonly target_contacts: number;
}

export interface EffectorDiagnostic {
  readonly canonical: string;
  readonly target_link: string;
  readonly sample_count: number;
  readonly mean_error_m: number;
  readonly p95_error_m: number;
  readonly max_error_m: number;
}

export interface FootContactDiagnostic {
  readonly side: "left" | "right";
  readonly canonical: string;
  readonly target_link: string;
  readonly agreement_ratio: number;
  readonly recall_ratio: number;
  readonly source_contact_ratio: number;
  readonly target_contact_ratio: number;
  readonly target_slide_mean_mps: number;
  readonly target_slide_p95_mps: number;
}

export interface ContactDiagnostics {
  readonly available: boolean;
  readonly reason?: string;
  readonly agreement_ratio?: number;
  readonly recall_ratio?: number;
  readonly target_slide_mean_mps?: number;
  readonly target_slide_p95_mps?: number;
  readonly feet: readonly FootContactDiagnostic[];
}

export type TrackingQualityBand = "stable" | "review" | "high_error";

/** Backend display band; limits scale with robot height when it is known. */
export interface TrackingQuality {
  readonly band: TrackingQualityBand;
  readonly stable_max_m: number;
  readonly review_max_m: number;
  readonly robot_height_m?: number | null;
  readonly p95_height_ratio?: number | null;
}

export interface ResultDiagnosticsPayload {
  readonly schema_version: number;
  readonly available: boolean;
  readonly reason?: string;
  readonly quality?: TrackingQuality;
  readonly frame_count?: number;
  readonly mapped_effectors?: number;
  readonly requested_effectors?: number;
  readonly tracking?: {
    readonly reference?: "ik_targets" | "scaled_preview";
    readonly mean_error_m: number;
    readonly p95_error_m: number;
    readonly max_error_m: number;
    readonly effectors: readonly EffectorDiagnostic[];
    readonly series: readonly TrackingDiagnosticPoint[];
  };
  readonly contact?: ContactDiagnostics;
}

export interface ExportOptions {
  readonly format: ExportFormat;
  readonly fps?: number;
  readonly csvHeader?: boolean;
  readonly start?: number;
  readonly end?: number;
}

export interface ExportFormValues {
  readonly format: ExportFormat;
  readonly fps: string;
  readonly start: string;
  readonly end: string;
  readonly csvHeader: boolean;
}

export type ExportValidation =
  | { readonly valid: true; readonly options: ExportOptions }
  | { readonly valid: false; readonly error: string };

export interface TrackingChart {
  readonly width: number;
  readonly height: number;
  readonly peak: number;
  readonly meanPoints: string;
  readonly maxPoints: string;
}

function optionalNumber(
  value: string,
  label: string,
  minimum: number,
): number | undefined | string {
  if (!value.trim()) return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < minimum) {
    return `${label} must be ${minimum === 0 ? "zero or greater" : "greater than zero"}.`;
  }
  return parsed;
}

/** Validate the editable export form before constructing a download URL. */
export function validateExportOptions(values: ExportFormValues): ExportValidation {
  const fps = optionalNumber(values.fps, "Export FPS", Number.MIN_VALUE);
  if (typeof fps === "string") return { valid: false, error: fps };

  const start = optionalNumber(values.start, "Start time", 0);
  if (typeof start === "string") return { valid: false, error: start };

  const end = optionalNumber(values.end, "End time", 0);
  if (typeof end === "string") return { valid: false, error: end };
  if (end !== undefined && end <= (start ?? 0)) {
    return {
      valid: false,
      error: "End time must be greater than start time.",
    };
  }

  return {
    valid: true,
    options: {
      format: values.format,
      fps,
      csvHeader: values.csvHeader,
      start,
      end,
    },
  };
}

/** Build scale-independent SVG paths from finite per-frame diagnostics. */
export function buildTrackingChart(
  series: readonly TrackingDiagnosticPoint[],
  width = 320,
  height = 72,
): TrackingChart | null {
  const points = series.filter(
    (point) =>
      Number.isFinite(point.mean_error_m) &&
      Number.isFinite(point.max_error_m) &&
      point.mean_error_m >= 0 &&
      point.max_error_m >= 0,
  );
  if (points.length < 2) return null;

  const peak = Math.max(...points.map((point) => point.max_error_m), 0.001);
  const path = (key: "mean_error_m" | "max_error_m") =>
    points
      .map((point, index) => {
        const x = (index * width) / Math.max(1, points.length - 1);
        const y = height - (point[key] / peak) * (height - 8) - 4;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

  return {
    width,
    height,
    peak,
    meanPoints: path("mean_error_m"),
    maxPoints: path("max_error_m"),
  };
}

/** Absolute limits for payloads that predate height-normalized bands. */
export const FALLBACK_STABLE_P95_M = 0.05;
export const FALLBACK_REVIEW_P95_M = 0.1;

export interface ResolvedTrackingQuality {
  readonly band: TrackingQualityBand | "unavailable";
  readonly stableMaxM?: number;
  readonly reviewMaxM?: number;
  readonly robotHeightM?: number;
}

function positiveFinite(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : undefined;
}

/** Resolve the tracking band, preferring the backend's height-normalized limits. */
export function resolveTrackingQuality(
  diagnostics: ResultDiagnosticsPayload,
): ResolvedTrackingQuality {
  const p95 = diagnostics.tracking?.p95_error_m;
  if (!diagnostics.available || p95 == null || !Number.isFinite(p95)) {
    return { band: "unavailable" };
  }
  const stable = positiveFinite(diagnostics.quality?.stable_max_m);
  const review = positiveFinite(diagnostics.quality?.review_max_m);
  const normalized = stable !== undefined && review !== undefined && stable <= review;
  const stableMaxM = normalized ? stable : FALLBACK_STABLE_P95_M;
  const reviewMaxM = normalized ? review : FALLBACK_REVIEW_P95_M;
  const band = p95 <= stableMaxM ? "stable" : p95 <= reviewMaxM ? "review" : "high_error";
  return {
    band,
    stableMaxM,
    reviewMaxM,
    robotHeightM: normalized
      ? positiveFinite(diagnostics.quality?.robot_height_m ?? undefined)
      : undefined,
  };
}

export function topEffectors(
  effectors: readonly EffectorDiagnostic[],
  limit = 5,
): readonly EffectorDiagnostic[] {
  return [...effectors]
    .sort((left, right) => right.p95_error_m - left.p95_error_m)
    .slice(0, limit);
}
