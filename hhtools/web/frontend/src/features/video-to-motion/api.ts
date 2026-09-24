import { createRequestCache } from "@/lib/requestCache";
import type { StageMotionPayload } from "@/stage/types";

export const SMPLX_DOWNLOAD_URL =
  "https://smpl-x.is.tue.mpg.de/download.php";

export const SUPPORTED_VIDEO_EXTENSIONS = [
  "mp4",
  "mov",
  "mkv",
  "avi",
  "webm",
  "m4v",
] as const;

export interface GvhmrRuntimeStatus {
  readonly ready: boolean;
  readonly checks?: Readonly<Record<string, boolean | undefined>> & {
    readonly smplx_neutral?: boolean;
  };
  readonly missing: readonly string[];
  readonly runtime?: "local" | "docker" | string;
  readonly root?: string | null;
  readonly body_models_root?: string | null;
  readonly python?: string | null;
  readonly uses_official_weights?: boolean;
}

/** Use the structured readiness contract instead of matching localized errors. */
export function isSmplxNeutralMissing(
  status: GvhmrRuntimeStatus | null | undefined,
): boolean {
  return status?.checks?.smplx_neutral === false;
}

/** The dedicated download link replaces the backend's path-heavy model hint. */
export function visibleGvhmrMissing(
  status: GvhmrRuntimeStatus | null | undefined,
): readonly string[] {
  const checks = status?.checks;
  if (!checks) return status?.missing ?? [];
  const failedChecks = Object.entries(checks)
    .filter(([, ready]) => ready === false)
    .map(([name]) => name);
  const messages: string[] = [];
  if (checks.official_repo === false) messages.push("GVHMR is not configured.");
  if (
    ["checkpoint_gvhmr", "checkpoint_hmr2", "checkpoint_vitpose", "checkpoint_yolov8"]
      .some((name) => checks[name] === false)
  ) {
    messages.push("GVHMR checkpoints are incomplete.");
  }
  if (checks.python_executable === false || checks.python_environment === false) {
    messages.push("GVHMR Python environment is unavailable.");
  }
  if (checks.ffmpeg === false) messages.push("FFmpeg is unavailable.");
  if (checks.cuda === false) messages.push("CUDA is unavailable.");
  if (checks.docker_cli === false || checks.docker_engine === false) {
    messages.push("Docker is unavailable.");
  } else if (checks.runtime_image === false) {
    messages.push("GVHMR runtime image is unavailable.");
  }
  if (messages.length) return messages;
  if (failedChecks.length === 1 && failedChecks[0] === "smplx_neutral") return [];
  if (failedChecks.length) return ["GVHMR runtime is unavailable."];
  return status?.missing ?? [];
}

export interface MotionResult extends Partial<StageMotionPayload> {
  readonly positions?: StageMotionPayload["positions"];
  readonly parent_indices?: readonly number[];
  readonly linked_folder?: string;
}

/** Convert a completed motion result into the renderer's data-only contract. */
export function toStageMotionPayload(
  result: MotionResult,
): StageMotionPayload | null {
  if (
    !Array.isArray(result.positions) ||
    !Array.isArray(result.parent_indices) ||
    result.positions.length === 0 ||
    result.parent_indices.length === 0
  ) {
    return null;
  }
  // The backend already returns the registered Motion payload. Preserve its
  // token, scene, body mesh, metadata, and Library identity for H2R handoff.
  return result as StageMotionPayload;
}

export interface MotionResultSummary {
  readonly name: string;
  readonly token: string | null;
  readonly frames: number | null;
  readonly duration: number | null;
  readonly framerate: number | null;
  readonly linkedFolder: string | null;
}

export interface VideoToMotionJob {
  readonly id: string;
  readonly kind: string;
  readonly status: string;
  readonly progress: number;
  readonly message?: string;
  readonly error?: string | null;
  readonly result?: MotionResult | null;
  readonly scope?: "current_session" | "persistent";
}

export interface DesktopGvhmrSetupResult {
  readonly action: "cancelled" | "configured" | "guide-opened";
}

interface DesktopGvhmrBridge {
  setupGvhmr(): Promise<DesktopGvhmrSetupResult>;
}

type Fetcher = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>;

type LocaleText = (english: string, chinese: string) => string;

/** Keep technical tracebacks in Tasks while presenting an actionable workflow error. */
export function visibleGvhmrFailure(
  error: unknown,
  text: LocaleText,
): string {
  const raw = error instanceof Error ? error.message : String(error);
  const missingModule = raw.match(/ModuleNotFoundError:\s+No module named ['"]([^'"]+)['"]/);
  if (missingModule) {
    return text(
      `The configured GVHMR Python environment is missing “${missingModule[1]}”. Reopen Set up and choose or repair the GVHMR environment. Technical details remain in Tasks.`,
      `所选 GVHMR Python 环境缺少“${missingModule[1]}”。请重新打开“配置”，选择或修复 GVHMR 环境；技术详情保留在任务列表中。`,
    );
  }
  if (/Traceback \(most recent call last\):|runtime exited with code/i.test(raw)) {
    return text(
      "The GVHMR runtime failed. Recheck its Python environment and optional component setup. Technical details remain in Tasks.",
      "GVHMR 运行失败。请重新检查其 Python 环境与可选组件配置；技术详情保留在任务列表中。",
    );
  }
  return raw;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function desktopBridge(host: unknown = globalThis): DesktopGvhmrBridge | null {
  const candidate = (host as { hhtoolsDesktop?: Partial<DesktopGvhmrBridge> })
    .hhtoolsDesktop;
  return typeof candidate?.setupGvhmr === "function"
    ? (candidate as DesktopGvhmrBridge)
    : null;
}

async function responseError(response: Response): Promise<Error> {
  let message = `${response.status} ${response.statusText}`.trim();
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string" && body.detail.trim()) {
      message = body.detail;
    }
  } catch {
    // Keep the HTTP status when the response has no JSON error body.
  }
  return new Error(message || "Request failed");
}

async function requestJson<T>(
  url: string,
  init: RequestInit,
  fetcher: Fetcher,
): Promise<T> {
  const response = await fetcher(url, init);
  if (!response.ok) throw await responseError(response);
  return (await response.json()) as T;
}

function delay(ms: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(signal.reason);
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      globalThis.clearTimeout(timer);
      reject(signal.reason);
    };
    const timer = globalThis.setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export function isSupportedVideoName(name: string): boolean {
  const separator = name.lastIndexOf(".");
  if (separator < 0) return false;
  const extension = name.slice(separator + 1).toLowerCase();
  return (SUPPORTED_VIDEO_EXTENSIONS as readonly string[]).includes(extension);
}

/** GVHMR's reusable result boundary is intentionally narrower than Motion import. */
export function isGvhmrResultName(name: string): boolean {
  const separator = name.lastIndexOf(".");
  return separator >= 0 && name.slice(separator + 1).toLowerCase() === "pt";
}

export function parseOptionalFocalLength(value: string): number | undefined {
  const normalized = value.trim();
  if (!normalized) return undefined;
  const parsed = Number(normalized);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error("Focal length must be a positive integer.");
  }
  return parsed;
}

export function boundedProgress(value: unknown): number {
  const parsed = finiteNumber(value);
  return parsed === null ? 0 : Math.max(0, Math.min(1, parsed));
}

export function formatFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const value = bytes / 1024 ** exponent;
  return `${value >= 10 || exponent === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[exponent]}`;
}

export function summarizeMotionResult(
  result: MotionResult,
  fallbackName: string,
): MotionResultSummary {
  return {
    name:
      typeof result.name === "string" && result.name.trim()
        ? result.name
        : fallbackName,
    token:
      typeof result.token === "string" && result.token ? result.token : null,
    frames:
      finiteNumber(result.num_frames_total) ??
      finiteNumber(result.playback_frames) ??
      (Array.isArray(result.positions) ? result.positions.length : null),
    duration:
      finiteNumber(result.playback_duration) ?? finiteNumber(result.duration),
    framerate:
      finiteNumber(result.framerate) ?? finiteNumber(result.sample_rate),
    linkedFolder:
      typeof result.linked_folder === "string" && result.linked_folder
        ? result.linked_folder
        : null,
  };
}

export function canSetupGvhmrInDesktop(host: unknown = globalThis): boolean {
  return desktopBridge(host) !== null;
}

export async function setupGvhmrInDesktop(
  host: unknown = globalThis,
): Promise<DesktopGvhmrSetupResult> {
  const bridge = desktopBridge(host);
  if (!bridge) throw new Error("GVHMR setup is available in the desktop app only.");
  const result = await bridge.setupGvhmr();
  if (result.action === "configured") gvhmrStatusCache.invalidate();
  return result;
}

const gvhmrStatusCache = createRequestCache<GvhmrRuntimeStatus>(5_000);

export function invalidateGvhmrRuntimeStatus(): void {
  gvhmrStatusCache.invalidate();
}

function normalizeGvhmrRuntimeStatus(status: GvhmrRuntimeStatus): GvhmrRuntimeStatus {
  const normalized = {
    ...status,
    ready: status.ready === true,
    missing: Array.isArray(status.missing)
      ? status.missing.filter((item): item is string => typeof item === "string")
      : [],
  };
  return { ...normalized, missing: visibleGvhmrMissing(normalized) };
}

export async function getGvhmrRuntimeStatus(
  signal: AbortSignal,
  fetcher: Fetcher = fetch,
): Promise<GvhmrRuntimeStatus> {
  const load = async (requestSignal?: AbortSignal, requestFetcher: Fetcher = fetch) =>
    normalizeGvhmrRuntimeStatus(
      await requestJson<GvhmrRuntimeStatus>(
        "/api/video-to-motion/status",
        { signal: requestSignal },
        requestFetcher,
      ),
    );
  if (fetcher !== fetch) return load(signal, fetcher);
  return gvhmrStatusCache.read(() => load(), signal);
}

export async function startVideoToMotion(
  input: {
    video: File;
    staticCamera: boolean;
    focalLength?: number;
  },
  signal: AbortSignal,
  fetcher: Fetcher = fetch,
): Promise<string> {
  const query = new URLSearchParams({
    static_cam: String(input.staticCamera),
  });
  if (input.focalLength !== undefined) {
    query.set("f_mm", String(input.focalLength));
  }

  const form = new FormData();
  form.append("files", input.video, input.video.name);
  const response = await requestJson<{ job_id?: unknown }>(
    `/api/video-to-motion/upload?${query.toString()}`,
    { method: "POST", body: form, signal },
    fetcher,
  );
  if (typeof response.job_id !== "string" || !response.job_id) {
    throw new Error("The server did not return a job ID.");
  }
  return response.job_id;
}

export async function waitForVideoToMotion(
  jobId: string,
  options: {
    signal: AbortSignal;
    onUpdate(job: VideoToMotionJob): void;
    pollIntervalMs?: number;
  },
  fetcher: Fetcher = fetch,
): Promise<MotionResult> {
  const interval = Math.max(0, options.pollIntervalMs ?? 500);
  for (;;) {
    const response = await requestJson<VideoToMotionJob>(
      `/api/job/${encodeURIComponent(jobId)}`,
      { signal: options.signal },
      fetcher,
    );
    const job = { ...response, progress: boundedProgress(response.progress) };
    if (job.kind !== "video_to_motion") {
      throw new Error(`Job ${job.id} has kind ${job.kind}; expected video_to_motion.`);
    }

    if (job.status === "done") {
      if (job.scope === "persistent") {
        throw new Error("This completed job no longer has a live motion result.");
      }
      if (!job.result) throw new Error("The job completed without a motion result.");
      return job.result;
    }
    if (["error", "failed", "cancelled"].includes(job.status)) {
      throw new Error(job.error || "Video-to-motion failed.");
    }
    if (job.status !== "pending" && job.status !== "running") {
      throw new Error(`Job ${job.id} returned an unknown status.`);
    }
    options.onUpdate(job);
    await delay(interval, options.signal);
  }
}
