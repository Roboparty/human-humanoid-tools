import {
  requestJson,
  uploadFiles,
  waitForJob,
  type Fetcher,
  type JobSnapshot,
  type UploadFile,
} from "@/lib/api";
import { createRequestCache } from "@/lib/requestCache";
import type { StageMotionPayload } from "@/stage/types";

export type MotionProfile = "mimic" | "intermimic" | "meshmimic";
export type MotionCategory = "motion" | "object" | "terrain";
export type MotionAssetKind = "human_motion" | "robot_trajectory";

/** One row returned by the backend Motion Library scanner. */
export interface MotionLibraryEntry {
  readonly dataset?: string;
  readonly folder_label?: string;
  readonly sequence_id?: string;
  readonly source_path: string;
  readonly stem?: string;
  readonly label?: string;
  readonly name?: string;
  readonly display_name?: string;
  readonly origin?: string;
  readonly reference?: string;
  readonly upload_profile?: string;
  readonly upload_drop?: string;
  readonly export_subdir?: string;
  readonly token?: string;
  readonly suggested_backend?: string;
  readonly motion_category?: MotionCategory;
  readonly asset_kind?: MotionAssetKind;
  readonly source_robot?: string;
  readonly job_id?: string;
}

export interface MotionLibraryResponse {
  readonly source_root: string;
  readonly motions_library_root: string;
  readonly folders: readonly string[];
  readonly entries: readonly MotionLibraryEntry[];
}

export function humanMotionEntries(
  entries: readonly MotionLibraryEntry[],
): readonly MotionLibraryEntry[] {
  return entries.filter((entry) => entry.asset_kind !== "robot_trajectory");
}

export function robotTrajectoryEntries(
  entries: readonly MotionLibraryEntry[],
): readonly MotionLibraryEntry[] {
  return entries.filter((entry) => entry.asset_kind === "robot_trajectory");
}

const motionLibraryCache = createRequestCache<MotionLibraryResponse>(1_000);

export function invalidateMotionLibrary(): void {
  motionLibraryCache.invalidate();
}

/** Full result emitted by `/api/motion/load_library` or `/api/motion/upload`. */
export interface MotionPayload extends StageMotionPayload {
  readonly name: string;
  readonly token: string;
  readonly positions: readonly (readonly (readonly [number, number, number])[])[];
  readonly parent_indices: readonly number[];
  readonly source_format?: string;
  readonly up_axis?: string;
  readonly bone_names?: readonly string[];
  readonly dataset?: string;
  readonly origin?: string;
  readonly library_entry?: MotionLibraryEntry;
  readonly linked_folder?: string;
  readonly materialize_mode?: "symlink" | "hardlink" | "copy" | "pending" | string;
}

export interface MotionJob extends JobSnapshot<MotionPayload> {
  readonly kind: string;
}

export interface MotionUploadStart {
  readonly job_id: string;
  readonly linked?: boolean;
  readonly folder_label?: string;
  readonly materialize_mode?: string;
  readonly motions_library_root?: string;
}

export interface LoadMotionOptions {
  readonly signal?: AbortSignal;
  readonly fetcher?: Fetcher;
  readonly onUpdate?: (job: MotionJob) => void;
  readonly pollIntervalMs?: number;
  readonly usage?: "human_to_robot";
  /** Optional guard for callers that already know which route created a job. */
  readonly expectedKind?: "motion_load" | "motion_link";
}

export interface UploadMotionOptions extends LoadMotionOptions {
  readonly profile?: MotionProfile;
  readonly libraryFolderLabel?: string;
}

/** Read the merged source + managed Motion Library list. */
export function getMotionLibrary(
  options: { signal?: AbortSignal; fetcher?: Fetcher } = {},
): Promise<MotionLibraryResponse> {
  if (options.fetcher) {
    return requestJson<MotionLibraryResponse>(
      "/api/library",
      { signal: options.signal },
      options.fetcher,
    );
  }
  return motionLibraryCache.read(
    () => requestJson<MotionLibraryResponse>("/api/library"),
    options.signal,
  );
}

/** Human-motion view over the shared, coalesced Library catalog. */
export async function getHumanMotionLibrary(
  options: { signal?: AbortSignal; fetcher?: Fetcher } = {},
): Promise<MotionLibraryResponse> {
  const library = await getMotionLibrary(options);
  return { ...library, entries: humanMotionEntries(library.entries) };
}

/** Start loading one library row; the server performs parsing in a job. */
export async function startMotionLibraryLoad(
  entry: MotionLibraryEntry,
  options: { signal?: AbortSignal; fetcher?: Fetcher; usage?: "human_to_robot" } = {},
): Promise<string> {
  const response = await requestJson<{ job_id?: unknown }>(
    "/api/motion/load_library",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        options.usage ? { ...entry, usage: options.usage } : entry,
      ),
      signal: options.signal,
    },
    options.fetcher,
  );
  if (typeof response.job_id !== "string" || !response.job_id) {
    throw new Error("The server did not return a motion job ID.");
  }
  return response.job_id;
}

/** Start a multipart upload. Folder-relative names are kept by `uploadFiles`. */
export async function startMotionUpload(
  files: Iterable<UploadFile | File>,
  options: UploadMotionOptions = {},
): Promise<MotionUploadStart> {
  const response = await uploadFiles<MotionUploadStart>(
    "/api/motion/upload",
    files,
    {
      query: {
        profile: options.profile ?? "mimic",
        library_folder_label: options.libraryFolderLabel,
      },
      signal: options.signal,
      fetcher: options.fetcher,
    },
  );
  if (typeof response.job_id !== "string" || !response.job_id) {
    throw new Error("The server did not return a motion upload job ID.");
  }
  return response;
}

/** Complete a previously started Motion job and return its serialized payload. */
export function waitForMotionJob(
  jobId: string,
  options: LoadMotionOptions = {},
): Promise<MotionPayload> {
  return waitForJob<MotionPayload>(jobId, {
    signal: options.signal,
    expectedKind: options.expectedKind,
    pollIntervalMs: options.pollIntervalMs,
    onUpdate: options.onUpdate,
    fetcher: options.fetcher,
  });
}

/** Load a library row end-to-end. */
export async function loadMotionLibraryEntry(
  entry: MotionLibraryEntry,
  options: LoadMotionOptions = {},
): Promise<MotionPayload> {
  const jobId = await startMotionLibraryLoad(entry, options);
  return waitForMotionJob(jobId, { ...options, expectedKind: "motion_load" });
}

/** Upload one file/folder end-to-end and return the first loaded clip. */
export async function uploadMotion(
  files: Iterable<UploadFile | File>,
  options: UploadMotionOptions = {},
): Promise<MotionPayload> {
  const started = await startMotionUpload(files, options);
  return waitForMotionJob(started.job_id, {
    ...options,
    expectedKind: "motion_link",
  });
}

/**
 * Narrow the backend payload to the renderer contract without copying large
 * arrays. The server already validates these fields while serializing Motion.
 */
export function toStageMotionPayload(
  payload: MotionPayload,
): StageMotionPayload | null {
  if (
    !Array.isArray(payload.positions) ||
    !Array.isArray(payload.parent_indices) ||
    payload.positions.length === 0 ||
    payload.parent_indices.length === 0
  ) {
    return null;
  }
  return payload;
}
