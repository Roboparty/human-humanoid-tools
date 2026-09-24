/** Typed FastAPI boundary for the Robot -> Robot workflow. */

import {
  requestJson,
  uploadFiles,
  waitForJob,
  type Fetcher,
  type JobSnapshot,
} from "@/lib/api";
import {
  getMotionLibrary,
  robotTrajectoryEntries,
  type MotionLibraryEntry,
} from "@/features/motion/api";
import {
  getRobotLibrary,
  loadRobot,
  type RobotPayload,
  type RobotSummary,
} from "@/features/robot/api";
import type {
  ExportOptions,
  ResultDiagnosticsPayload,
} from "@/features/result/model";
import type {
  StageMatrix4,
  StageMotionPayload,
  StageObjectPayload,
  StageTerrainPayload,
  StageVec3,
} from "@/stage/types";
import type { CalibrationJointWorld } from "@/stage/calibrationInteraction";
import type { ExecutionProvenance } from "@/types/execution";

export { getRobotLibrary, loadRobot };
export type { MotionLibraryEntry, RobotPayload, RobotSummary };

export type R2rBackend = "newton" | "interaction_mesh";

export interface R2rRobotFrame {
  readonly root?: readonly [number, number, number, number, number, number, number];
  readonly mesh_z_lift?: number;
  readonly links: Readonly<Record<string, StageMatrix4>>;
}

export interface R2rRobotTrajectory {
  readonly frames: readonly R2rRobotFrame[];
  readonly frame_indices?: readonly number[];
  readonly duration?: number;
  readonly playback_duration?: number;
  readonly playback_frames?: number;
  readonly num_frames_total?: number;
  readonly framerate?: number;
  readonly sample_rate?: number;
}

export interface R2rScenePayload {
  readonly terrain?: StageTerrainPayload | null;
  readonly objects?: readonly StageObjectPayload[];
}

export interface R2rSourceResult {
  readonly token: string;
  readonly source_robot: string;
  readonly name?: string;
  readonly num_frames: number;
  readonly framerate: number;
  readonly dof_names?: readonly string[];
  readonly trajectory: R2rRobotTrajectory;
  readonly skeleton_preview?: StageMotionPayload;
  readonly scaled_scene?: R2rScenePayload;
  readonly has_scene?: boolean;
  readonly upload_profile?: string;
  readonly suggested_backend?: R2rBackend | string;
}

export interface R2rJointLimit {
  readonly name: string;
  readonly lower?: number;
  readonly upper?: number;
  readonly value?: number;
  readonly type?: string;
  readonly child_link?: string;
  readonly parent_link?: string;
  readonly axis?: StageVec3;
}

export interface R2rCalibrationReference {
  readonly positions: readonly (readonly StageVec3[])[];
  readonly parent_indices: readonly number[];
  readonly exclude_joint_indices?: readonly number[];
  readonly bone_names?: readonly string[];
}

export interface R2rCalibrationSession {
  readonly joint_q: Readonly<Record<string, number>>;
  readonly joint_limits: readonly R2rJointLimit[];
  readonly joint_world: Readonly<Record<string, CalibrationJointWorld>>;
  readonly reference: R2rCalibrationReference;
  readonly reference_name: string;
  readonly ground_offset_z: number;
  readonly has_saved_calibration?: boolean;
}

export interface R2rCalibrationPose {
  readonly links: readonly string[];
  readonly link_transforms: Readonly<Record<string, StageMatrix4>>;
  readonly joint_world: Readonly<Record<string, CalibrationJointWorld>>;
  readonly ground_offset_z: number;
}

export interface R2rRetargetResult {
  readonly trajectory: R2rRobotTrajectory;
  readonly export_token: string;
  readonly stem: string;
  readonly num_frames: number;
  readonly source_fps: number;
  readonly scaled_preview?: StageMotionPayload;
  readonly scaled_scene?: R2rScenePayload;
  readonly diagnostics?: ResultDiagnosticsPayload;
  readonly has_scene?: boolean;
  readonly execution_provenance: ExecutionProvenance;
}

interface RequestOptions {
  readonly signal?: AbortSignal;
  readonly fetcher?: Fetcher;
}

interface JobOptions<TResult> extends RequestOptions {
  readonly onUpdate?: (job: JobSnapshot<TResult>) => void;
  readonly pollIntervalMs?: number;
}

function postJson<T>(
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

function validFps(value: number | undefined): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : undefined;
}

/** Return only robot trajectories; human motions cannot cross into R2R. */
export async function getR2rLibrary(
  options: RequestOptions = {},
): Promise<readonly MotionLibraryEntry[]> {
  const library = await getMotionLibrary(options);
  return robotTrajectoryEntries(library.entries);
}

/** Keep known-incompatible trajectories out of a loaded source robot's picker. */
export function r2rEntriesForSourceRobot(
  entries: readonly MotionLibraryEntry[],
  sourceRobot?: string | null,
): readonly MotionLibraryEntry[] {
  const robot = sourceRobot?.trim();
  if (!robot) return entries;
  return entries.filter(
    (entry) => !entry.source_robot || entry.source_robot === robot,
  );
}

export async function loadR2rLibraryEntry(
  entry: MotionLibraryEntry,
  sourceRobot: string,
  sourceFps?: number,
  options: JobOptions<R2rSourceResult> = {},
): Promise<R2rSourceResult> {
  const started = await postJson<{ job_id: string }>(
    "/api/r2r/source/library",
    {
      ...entry,
      source_robot: sourceRobot,
      source_fps: validFps(sourceFps),
    },
    options,
  );
  return waitForJob<R2rSourceResult>(started.job_id, {
    ...options,
    expectedKind: "r2r_source_library",
  });
}

export async function uploadR2rTrajectory(
  files: Iterable<File>,
  sourceRobot: string,
  sourceFps?: number,
  options: JobOptions<R2rSourceResult> & { readonly profile?: string } = {},
): Promise<R2rSourceResult> {
  const started = await uploadFiles<{ job_id: string }>(
    "/api/r2r/source/upload",
    files,
    {
      query: {
        source_robot: sourceRobot,
        profile: options.profile ?? "auto",
        source_fps: validFps(sourceFps),
      },
      signal: options.signal,
      fetcher: options.fetcher,
    },
  );
  return waitForJob<R2rSourceResult>(started.job_id, {
    ...options,
    expectedKind: "r2r_source_upload",
  });
}

export function getR2rCalibrationStatus(
  target: string,
  source: string,
  options: RequestOptions = {},
): Promise<{ readonly calibrated: boolean }> {
  const query = new URLSearchParams({ target, source });
  return requestJson(
    `/api/r2r/calibration/status?${query.toString()}`,
    { signal: options.signal },
    options.fetcher,
  );
}

export function getR2rCalibrationSession(
  target: string,
  source: string,
  options: RequestOptions = {},
): Promise<R2rCalibrationSession> {
  return postJson("/api/r2r/calibration/session", { target, source }, options);
}

export function saveR2rCalibration(
  target: string,
  source: string,
  jointQ: Readonly<Record<string, number>>,
  options: RequestOptions = {},
): Promise<{ readonly ok: boolean; readonly path: string }> {
  return postJson(
    "/api/r2r/calibration/save",
    { target, source, joint_q: jointQ },
    options,
  );
}

export function previewR2rCalibrationPose(
  robot: string,
  jointQ: Readonly<Record<string, number>>,
  options: RequestOptions = {},
): Promise<R2rCalibrationPose> {
  return postJson("/api/robot/fk_preview", { robot, joint_q: jointQ }, options);
}

export async function runR2rRetarget(
  request: {
    readonly target: string;
    readonly source: string;
    readonly sourceToken: string;
    readonly backend: R2rBackend;
    readonly retargetFps?: number;
  },
  options: JobOptions<R2rRetargetResult> = {},
): Promise<R2rRetargetResult> {
  const started = await postJson<{ job_id: string }>(
    "/api/r2r/retarget",
    {
      target: request.target,
      source: request.source,
      source_token: request.sourceToken,
      backend: request.backend,
      retarget_fps: validFps(request.retargetFps),
    },
    options,
  );
  return waitForJob<R2rRetargetResult>(started.job_id, {
    ...options,
    expectedKind: "r2r_retarget",
  });
}

export function r2rExportUrl(
  token: string,
  options: Partial<ExportOptions> = {},
): string {
  const query = new URLSearchParams({
    fmt: options.format ?? "csv",
    csv_header: String(options.csvHeader ?? true),
  });
  const fps = validFps(options.fps);
  if (fps !== undefined) query.set("fps", String(fps));
  if (options.start !== undefined) query.set("t_start", String(options.start));
  if (options.end !== undefined) query.set("t_end", String(options.end));
  return `/api/export/${encodeURIComponent(token)}?${query.toString()}`;
}
