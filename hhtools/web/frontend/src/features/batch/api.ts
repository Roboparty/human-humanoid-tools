import {
  requestJson,
  uploadFiles,
  waitForJob,
  type Fetcher,
  type JobSnapshot,
  type UploadFile,
} from "@/lib/api";
import type { MotionLibraryEntry } from "@/features/motion/api";

export type BatchBackend = "newton" | "interaction_mesh";
export type BatchFormat = "csv" | "pkl";

export interface BatchFailure {
  readonly stem?: string;
  readonly stage?: string;
  readonly reason?: string;
  readonly log_rel?: string;
  readonly stash_error?: string;
}

export interface BatchResult {
  readonly written: readonly string[];
  readonly errors: readonly string[];
  readonly failures: readonly BatchFailure[];
  readonly failure_log?: string | null;
  readonly format: BatchFormat | string;
  readonly download_name: string;
  readonly artifact_path: string;
  readonly clip_count?: number;
  readonly batch_size?: number;
  readonly requested_batch_size?: number;
  readonly solver_mode?: string;
}

export interface CompletedBatch {
  readonly jobId: string;
  readonly result: BatchResult;
}

interface JobOptions<TResult> {
  readonly signal?: AbortSignal;
  readonly fetcher?: Fetcher;
  readonly pollIntervalMs?: number;
  readonly onUpdate?: (job: JobSnapshot<TResult>) => void;
}

interface BatchInputResult {
  readonly entries: readonly MotionLibraryEntry[];
  readonly clip_count: number;
  readonly source?: string;
  readonly profile?: string;
}

async function postJob<TResult>(
  url: string,
  kind: string,
  body: unknown,
  options: JobOptions<TResult>,
): Promise<{ jobId: string; result: TResult }> {
  const started = await requestJson<{ job_id?: unknown }>(
    url,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: options.signal,
    },
    options.fetcher,
  );
  if (typeof started.job_id !== "string" || !started.job_id) {
    throw new Error("The server did not return a Batch job ID.");
  }
  const result = await waitForJob<TResult>(started.job_id, {
    ...options,
    expectedKind: kind,
  });
  return { jobId: started.job_id, result };
}

async function uploadBatchInputs(
  url: string,
  kind: string,
  files: Iterable<UploadFile | File>,
  profile: string,
  options: JobOptions<BatchInputResult>,
): Promise<BatchInputResult> {
  const started = await uploadFiles<{ job_id?: unknown }>(url, files, {
    query: { profile },
    signal: options.signal,
    fetcher: options.fetcher,
  });
  if (typeof started.job_id !== "string" || !started.job_id) {
    throw new Error("The server did not return a Batch import job ID.");
  }
  return waitForJob<BatchInputResult>(started.job_id, {
    ...options,
    expectedKind: kind,
  });
}

function scanBatchInputs(
  url: string,
  source: string,
  profile: string,
  options: Pick<JobOptions<BatchInputResult>, "signal" | "fetcher">,
): Promise<BatchInputResult> {
  return requestJson(
    url,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, profile }),
      signal: options.signal,
    },
    options.fetcher,
  );
}

export function uploadHumanBatchInputs(
  files: Iterable<UploadFile | File>,
  profile = "auto",
  options: JobOptions<BatchInputResult> = {},
): Promise<BatchInputResult> {
  return uploadBatchInputs("/api/basket/upload", "basket_upload", files, profile, options);
}

export function scanHumanBatchInputs(
  source: string,
  profile = "auto",
  options: Pick<JobOptions<BatchInputResult>, "signal" | "fetcher"> = {},
): Promise<BatchInputResult> {
  return scanBatchInputs("/api/basket/scan", source, profile, options);
}

export function uploadR2rBatchInputs(
  files: Iterable<UploadFile | File>,
  profile = "auto",
  options: JobOptions<BatchInputResult> = {},
): Promise<BatchInputResult> {
  return uploadBatchInputs(
    "/api/r2r/basket/upload",
    "r2r_basket_upload",
    files,
    profile,
    options,
  );
}

export function scanR2rBatchInputs(
  source: string,
  profile = "auto",
  options: Pick<JobOptions<BatchInputResult>, "signal" | "fetcher"> = {},
): Promise<BatchInputResult> {
  return scanBatchInputs("/api/r2r/basket/scan", source, profile, options);
}

export interface HumanBatchRequest {
  readonly robot: string;
  readonly entries: readonly MotionLibraryEntry[];
  readonly reference: string;
  readonly backend: BatchBackend;
  readonly out_dir: string;
  readonly format: BatchFormat;
  readonly csv_header: boolean;
  readonly foot_clamp_anti_penetration: boolean;
  readonly batch_size?: number;
  readonly retarget_fps?: number;
  readonly export_fps?: number;
  readonly t_start?: number;
  readonly t_end?: number;
}

export async function runHumanBatch(
  request: HumanBatchRequest,
  options: JobOptions<BatchResult> = {},
): Promise<CompletedBatch> {
  return postJob("/api/batch/retarget", "batch", request, options);
}

export interface R2rBatchRequest {
  readonly source: string;
  readonly target: string;
  readonly entries: readonly MotionLibraryEntry[];
  readonly backend: BatchBackend;
  readonly out_dir: string;
  readonly format: BatchFormat;
  readonly csv_header: boolean;
  readonly source_fps?: number;
  readonly retarget_fps?: number;
  readonly export_fps?: number;
  readonly t_start?: number;
  readonly t_end?: number;
}

export async function runR2rBatch(
  request: R2rBatchRequest,
  options: JobOptions<BatchResult> = {},
): Promise<CompletedBatch> {
  return postJob("/api/r2r/batch/retarget", "r2r_batch", request, options);
}

export function batchDownloadUrl(jobId: string): string {
  return `/api/job/${encodeURIComponent(jobId)}/download`;
}

