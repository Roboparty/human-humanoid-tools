/**
 * Small same-origin HTTP boundary shared by the new React features.
 *
 * Feature modules describe their own payloads, while this module owns the
 * mechanics that every FastAPI endpoint needs: error decoding, multipart
 * uploads, and polling a background job. Keeping those mechanics here avoids
 * reintroducing one bespoke request helper per inspector.
 */

export type Fetcher = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>;

export interface JobSnapshot<TResult = unknown> {
  readonly id: string;
  readonly kind: string;
  readonly status: string;
  readonly progress?: number;
  readonly clip_progress?: number;
  readonly message?: string;
  readonly error?: string | null;
  readonly result?: TResult | null;
  readonly scope?: "current_session" | "persistent" | string;
}

export interface JobStartResponse {
  readonly job_id: string;
}

/** A browser File with the relative path preserved by a folder picker. */
export type UploadFile = File & {
  /** Mutable client-only alias used when a drop supplies a relative path. */
  _relpath?: string;
};

/** Render only a portable leaf name when an API identity contains a host path. */
export function displayFileName(path: string, fallback = "Untitled"): string {
  return path.split(/[\\/]/).filter(Boolean).pop() || fallback;
}

export interface UploadOptions {
  readonly query?: Record<string, string | number | boolean | undefined>;
  readonly signal?: AbortSignal;
  readonly fetcher?: Fetcher;
}

/** Turn FastAPI's string/list/object `detail` into one readable message. */
export function apiDetailMessage(detail: unknown): string | undefined {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const message = detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          return String(item.msg);
        }
        return JSON.stringify(item);
      })
      .filter(Boolean)
      .join("; ");
    return message || undefined;
  }
  if (detail && typeof detail === "object" && "msg" in detail) {
    const message = String(detail.msg);
    return message.trim() || undefined;
  }
  return detail == null ? undefined : JSON.stringify(detail);
}

/** Error carrying the HTTP status without leaking a Response into features. */
export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function responseError(response: Response): Promise<ApiError> {
  let message = `${response.status} ${response.statusText}`.trim();
  try {
    const body = (await response.json()) as { detail?: unknown };
    message = apiDetailMessage(body.detail) || message;
  } catch {
    // Some proxy/server failures are plain text or empty responses.
  }
  return new ApiError(message || "Request failed", response.status);
}

function withQuery(
  url: string,
  query: Record<string, string | number | boolean | undefined> = {},
): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== undefined) params.set(key, String(value));
  }
  const encoded = params.toString();
  if (!encoded) return url;
  return `${url}${url.includes("?") ? "&" : "?"}${encoded}`;
}

/** Perform one JSON request and decode the common FastAPI error shape. */
export async function requestJson<T>(
  url: string,
  init: RequestInit = {},
  fetcher: Fetcher = fetch,
): Promise<T> {
  const response = await fetcher(url, init);
  if (!response.ok) throw await responseError(response);
  return (await response.json()) as T;
}

/**
 * Upload one or more browser files while retaining folder-relative names.
 * FastAPI's upload boundary uses those names to rebuild mesh/sidecar trees.
 */
export async function uploadFiles<T>(
  url: string,
  files: Iterable<UploadFile | File>,
  options: UploadOptions = {},
): Promise<T> {
  const form = new FormData();
  for (const file of files) {
    const candidate = file as UploadFile;
    const relativePath =
      candidate._relpath || candidate.webkitRelativePath || candidate.name;
    form.append("files", file, relativePath);
  }
  return requestJson<T>(
    withQuery(url, options.query),
    { method: "POST", body: form, signal: options.signal },
    options.fetcher,
  );
}

function abortError(signal: AbortSignal): Error {
  if (signal.reason instanceof Error) return signal.reason;
  const error = new Error("The request was aborted.");
  error.name = "AbortError";
  return error;
}

export function boundedProgress(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, Math.min(1, value))
    : 0;
}

function delay(ms: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(abortError(signal));
  return new Promise((resolve, reject) => {
    let timer: ReturnType<typeof globalThis.setTimeout>;
    const onAbort = () => {
      globalThis.clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
      reject(abortError(signal));
    };
    timer = globalThis.setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
    // Abort may have happened between the initial check and listener setup.
    if (signal.aborted) onAbort();
  });
}

export interface WaitForJobOptions<TResult> {
  readonly signal?: AbortSignal;
  readonly expectedKind?: string;
  readonly pollIntervalMs?: number;
  readonly onUpdate?: (job: JobSnapshot<TResult>) => void;
  readonly fetcher?: Fetcher;
}

/** Poll a FastAPI job until its result is available or it fails. */
export async function waitForJob<TResult>(
  jobId: string,
  options: WaitForJobOptions<TResult> = {},
): Promise<TResult> {
  const signal = options.signal ?? new AbortController().signal;
  const interval = Math.max(0, options.pollIntervalMs ?? 350);
  for (;;) {
    if (signal.aborted) throw abortError(signal);
    const job = await requestJson<JobSnapshot<TResult>>(
      `/api/job/${encodeURIComponent(jobId)}`,
      { signal },
      options.fetcher,
    );
    if (options.expectedKind && job.kind !== options.expectedKind) {
      throw new Error(
        `Job ${job.id} has kind ${job.kind}; expected ${options.expectedKind}.`,
      );
    }
    const snapshot: JobSnapshot<TResult> = {
      ...job,
      progress: boundedProgress(job.progress),
      clip_progress: boundedProgress(job.clip_progress),
    };
    options.onUpdate?.(snapshot);
    if (job.status === "done") {
      if (job.scope === "persistent") {
        throw new Error("This completed job no longer has a live result.");
      }
      if (job.result == null) {
        throw new Error(job.error || "The job completed without a result.");
      }
      return job.result;
    }
    if (["error", "failed", "cancelled"].includes(job.status)) {
      throw new Error(job.error || job.message || "The job failed.");
    }
    if (job.status !== "pending" && job.status !== "running") {
      throw new Error(`Job ${job.id} returned an unknown status.`);
    }
    await delay(interval, signal);
  }
}
