import { requestJson, type Fetcher } from "@/lib/api";
import {
  getGvhmrRuntimeStatus,
  type GvhmrRuntimeStatus,
} from "@/features/video-to-motion/api";

export {
  invalidateGvhmrRuntimeStatus,
  type GvhmrRuntimeStatus,
} from "@/features/video-to-motion/api";

export interface JobAdmissionSnapshot {
  readonly mode: "unlimited" | "queued" | string;
  readonly max_running_jobs: number;
  readonly max_queued_jobs: number;
  readonly max_batch_items: number;
  readonly max_batch_total_frames: number;
  readonly running_jobs: number;
  readonly queued_jobs: number;
  readonly reserved_jobs: number;
  readonly cancelling_jobs: number;
  readonly closed: boolean;
  readonly editable: boolean;
}

export interface JobAdmissionLimits {
  readonly max_running_jobs: number;
  readonly max_queued_jobs: number;
  readonly max_batch_items: number;
  readonly max_batch_total_frames: number;
}

export interface MotionLibrarySettingsSnapshot {
  readonly root: string;
  readonly default_root: string;
  readonly editable: boolean;
  readonly readonly_reason?: string | null;
}

export interface GvhmrOptionalComponentState {
  readonly requested: boolean;
  readonly configured: boolean;
  readonly root?: string;
  readonly python?: string;
  readonly bodyModels?: string;
  readonly runtime: "local" | "docker";
  readonly guideUrl: string;
  readonly estimatedAdditionalBytes: number;
}

export interface GvhmrSetupResult {
  readonly action: "cancelled" | "configured" | "guide-opened";
  readonly state: GvhmrOptionalComponentState;
}

export interface DesktopSettingsBridge {
  getOptionalComponents(): Promise<{
    readonly gvhmr: GvhmrOptionalComponentState;
  }>;
  setupGvhmr(): Promise<GvhmrSetupResult>;
  selectDirectory(): Promise<string | null>;
}

/** Return only the optional Electron capabilities used by Workspace Settings. */
export function desktopSettingsBridge(
  host: unknown = globalThis,
): DesktopSettingsBridge | null {
  const candidate = (
    host as { readonly hhtoolsDesktop?: Partial<DesktopSettingsBridge> }
  ).hhtoolsDesktop;
  return typeof candidate?.getOptionalComponents === "function" &&
    typeof candidate.setupGvhmr === "function" &&
    typeof candidate.selectDirectory === "function"
    ? (candidate as DesktopSettingsBridge)
    : null;
}

export function getJobAdmissionSettings(
  options: { readonly signal?: AbortSignal; readonly fetcher?: Fetcher } = {},
): Promise<JobAdmissionSnapshot> {
  return requestJson<JobAdmissionSnapshot>(
    "/api/settings/job-admission",
    { signal: options.signal },
    options.fetcher,
  );
}

export function updateJobAdmissionSettings(
  limits: JobAdmissionLimits,
  options: { readonly signal?: AbortSignal; readonly fetcher?: Fetcher } = {},
): Promise<JobAdmissionSnapshot> {
  return requestJson<JobAdmissionSnapshot>(
    "/api/settings/job-admission",
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(limits),
      signal: options.signal,
    },
    options.fetcher,
  );
}

export function getMotionLibrarySettings(
  options: { readonly signal?: AbortSignal; readonly fetcher?: Fetcher } = {},
): Promise<MotionLibrarySettingsSnapshot> {
  return requestJson<MotionLibrarySettingsSnapshot>(
    "/api/settings/motion-library",
    { signal: options.signal },
    options.fetcher,
  );
}

export function updateMotionLibrarySettings(
  root: string,
  options: { readonly signal?: AbortSignal; readonly fetcher?: Fetcher } = {},
): Promise<MotionLibrarySettingsSnapshot> {
  return requestJson<MotionLibrarySettingsSnapshot>(
    "/api/settings/motion-library",
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ root }),
      signal: options.signal,
    },
    options.fetcher,
  );
}

export async function getGvhmrRuntimeSettings(
  options: { readonly signal?: AbortSignal; readonly fetcher?: Fetcher } = {},
): Promise<GvhmrRuntimeStatus> {
  return getGvhmrRuntimeStatus(
    options.signal ?? new AbortController().signal,
    options.fetcher,
  );
}
