/** Typed boundary for the dataset analysis endpoints. */

import {
  apiDetailMessage,
  requestJson,
  uploadFiles,
  waitForJob,
  type Fetcher,
  type JobSnapshot,
  type UploadFile,
} from "@/lib/api";
import { createRequestCache } from "@/lib/requestCache";
import type {
  StageMotionPayload,
  StageRobotPayload,
  StageRobotTrajectoryPayload,
} from "@/stage/types";

export type AnalysisEmbedding = "handcrafted" | "pae";

export interface DatasetClip {
  readonly clip_id: string;
  readonly source_kind: "human" | "robot" | string;
  readonly source_path: string;
  readonly dataset: string;
  readonly folder_label: string;
  readonly metrics: Readonly<Record<string, number | string | boolean | null>>;
  readonly tags: readonly string[];
  readonly embedding: readonly number[] | null;
  readonly scatter: readonly [number, number] | null;
  readonly cluster_id: number | null;
  readonly error: string | null;
}

export interface Histogram {
  readonly counts: readonly number[];
  readonly edges: readonly number[];
  readonly min: number;
  readonly max: number;
  readonly mean: number;
  readonly median: number;
}

export interface DatasetSummary {
  readonly num_clips: number;
  readonly num_ok: number;
  readonly num_error: number;
  readonly numeric_keys: readonly string[];
  readonly histograms: Readonly<Record<string, Histogram>>;
  readonly tag_counts: Readonly<Record<string, number>>;
  readonly tag_order: readonly string[];
  readonly cluster_counts: Readonly<Record<string, number>>;
  readonly folder_counts: Readonly<Record<string, number>>;
}

export interface DatasetAnalysisResult {
  readonly meta: {
    readonly source_root: string;
    readonly embedding: AnalysisEmbedding | string;
    readonly fingerprint?: string;
    readonly metric_schema?: number;
    readonly generated_at?: number;
  };
  readonly clips: readonly DatasetClip[];
  readonly summary: DatasetSummary;
}

export interface DatasetRobotPreviewResult {
  readonly preview_token: string;
  readonly trajectory: StageRobotTrajectoryPayload;
  readonly robot: string;
  readonly inferred_robot: string;
  readonly num_frames: number;
  readonly framerate: number;
  readonly has_scene: boolean;
  readonly scaled_scene?: {
    readonly terrain?: StageMotionPayload["terrain"];
    readonly objects?: StageMotionPayload["objects"];
  } | null;
  readonly name: string;
}

/** Complete Analysis-owned projection for one robot trajectory preview. */
export interface AnalysisRobotPreview {
  readonly robot: StageRobotPayload;
  readonly trajectory: StageRobotTrajectoryPayload;
  readonly scene: DatasetRobotPreviewResult["scaled_scene"];
  readonly previewToken: string;
}

export interface DatasetUploadSummary {
  readonly source: string;
  readonly user_source_root?: string | null;
  readonly clip_count: number;
  readonly human_count: number;
  readonly robot_count: number;
  readonly folders: Readonly<Record<string, number>>;
  readonly clips: readonly { clip_id: string; folder_label: string }[];
  readonly entries_preview?: readonly Record<string, unknown>[];
}

export interface DatasetCatalog {
  readonly stages?: Readonly<Record<string, Record<string, unknown>>>;
  readonly data_sources?: Readonly<Record<string, string>>;
  readonly tags?: Readonly<Record<string, Record<string, unknown>>>;
  readonly metrics?: Readonly<Record<string, Record<string, unknown>>>;
  readonly categories?: Readonly<Record<string, Record<string, unknown>>>;
  readonly [key: string]: unknown;
}

const datasetCatalogCache = createRequestCache<DatasetCatalog>(1_000);

interface RequestOptions {
  readonly signal?: AbortSignal;
  readonly fetcher?: Fetcher;
}

interface JobOptions<TResult> extends RequestOptions {
  readonly onUpdate?: (job: JobSnapshot<TResult>) => void;
  readonly pollIntervalMs?: number;
}

function jsonPost<T>(
  url: string,
  body: unknown,
  options: RequestOptions = {},
): Promise<T> {
  return requestJson<T>(
    url,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: options.signal,
    },
    options.fetcher,
  );
}

export function getDatasetCatalog(
  options: RequestOptions = {},
): Promise<DatasetCatalog> {
  if (options.fetcher) {
    return requestJson<DatasetCatalog>(
      "/api/dataset/catalog",
      { signal: options.signal },
      options.fetcher,
    );
  }
  return datasetCatalogCache.read(
    () => requestJson<DatasetCatalog>("/api/dataset/catalog"),
    options.signal,
  );
}

export function scanDataset(
  source: string,
  options: RequestOptions = {},
): Promise<DatasetUploadSummary> {
  return jsonPost("/api/dataset/scan", { source }, options);
}

export function uploadDataset(
  files: Iterable<UploadFile | File>,
  options: RequestOptions & {
    readonly appendTo?: string;
    readonly userSourceRoot?: string;
  } = {},
): Promise<DatasetUploadSummary> {
  return uploadFiles<DatasetUploadSummary>("/api/dataset/upload", files, {
    query: {
      append_to: options.appendTo,
      user_source_root: options.userSourceRoot,
    },
    signal: options.signal,
    fetcher: options.fetcher,
  });
}

export function removeDatasetUploadFolder(
  source: string,
  folderLabel: string,
  options: RequestOptions = {},
): Promise<DatasetUploadSummary> {
  return jsonPost(
    "/api/dataset/upload/remove",
    { source, folder_label: folderLabel },
    options,
  );
}

export async function analyzeDataset(
  body: {
    readonly source?: string;
    readonly embedding: AnalysisEmbedding;
    readonly force: boolean;
  },
  options: JobOptions<DatasetAnalysisResult> = {},
): Promise<DatasetAnalysisResult> {
  const started = await jsonPost<{ job_id?: unknown }>(
    "/api/dataset/analyze",
    body,
    options,
  );
  if (typeof started.job_id !== "string" || !started.job_id) {
    throw new Error("The server did not return an analysis job ID.");
  }
  return waitForJob<DatasetAnalysisResult>(started.job_id, {
    signal: options.signal,
    fetcher: options.fetcher,
    expectedKind: "dataset_analyze",
    pollIntervalMs: options.pollIntervalMs,
    onUpdate: options.onUpdate,
  });
}

export async function previewDatasetRobot(
  body: {
    readonly source_path: string;
    readonly robot?: string;
  },
  options: JobOptions<DatasetRobotPreviewResult> = {},
): Promise<DatasetRobotPreviewResult> {
  const started = await jsonPost<{ job_id?: unknown }>(
    "/api/dataset/preview_robot",
    body,
    options,
  );
  if (typeof started.job_id !== "string" || !started.job_id) {
    throw new Error("The server did not return a robot preview job ID.");
  }
  return waitForJob<DatasetRobotPreviewResult>(started.job_id, {
    signal: options.signal,
    fetcher: options.fetcher,
    expectedKind: "dataset_robot_preview",
    pollIntervalMs: options.pollIntervalMs,
    onUpdate: options.onUpdate,
  });
}

export function getCachedDatasetResult(
  source: string | undefined,
  embedding: AnalysisEmbedding,
  options: RequestOptions = {},
): Promise<{ readonly available: boolean } & Partial<DatasetAnalysisResult>> {
  const query = new URLSearchParams({ embedding });
  if (source) query.set("source", source);
  return requestJson(
    `/api/dataset/result?${query.toString()}`,
    { signal: options.signal },
    options.fetcher,
  );
}

export function computeDatasetSubset(
  clips: readonly DatasetClip[],
  k: number,
  alpha: number,
  options: RequestOptions = {},
): Promise<{ readonly selected: readonly string[]; readonly count: number }> {
  return jsonPost("/api/dataset/subset", { clips, k, alpha }, options);
}

async function postBlob(
  url: string,
  body: unknown,
  options: RequestOptions = {},
): Promise<Blob> {
  const fetcher = options.fetcher ?? fetch;
  const response = await fetcher(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: options.signal,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`.trim();
    try {
      const payload = (await response.json()) as { detail?: unknown };
      message = apiDetailMessage(payload.detail) || message;
    } catch {
      // Keep the status when a proxy returns plain text.
    }
    throw new Error(message || "Export failed.");
  }
  return response.blob();
}

export function exportDatasetManifest(
  body: Readonly<Record<string, unknown>>,
  options: RequestOptions = {},
): Promise<Blob> {
  return postBlob("/api/dataset/export_manifest", body, options);
}

export function exportRobotSubset(
  body: Readonly<Record<string, unknown>>,
  options: RequestOptions = {},
): Promise<Blob> {
  return postBlob("/api/dataset/export_robot_zip", body, options);
}

export function motionEntryForAnalysisClip(clip: DatasetClip) {
  return {
    dataset: clip.dataset,
    folder_label: clip.folder_label,
    sequence_id: clip.source_path.split(/[\\/]/).pop() || clip.clip_id,
    source_path: clip.source_path,
    stem: clip.clip_id.split("/").pop() || clip.clip_id,
  };
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
}
