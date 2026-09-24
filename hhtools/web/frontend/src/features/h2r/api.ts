import {
  requestJson,
  waitForJob,
  type Fetcher,
  type JobSnapshot,
} from "@/lib/api";
import { createRequestCache } from "@/lib/requestCache";
import type {
  ExportOptions,
  ResultDiagnosticsPayload,
} from "@/features/result/model";
import type {
  StageMatrix4,
  StageMotionPayload,
  StageRobotTrajectoryPayload,
} from "@/stage/types";
import type { CalibrationJointWorld } from "@/stage/calibrationInteraction";
import type { ExecutionProvenance } from "@/types/execution";

export interface CalibrationStatus {
  readonly calibrated: boolean;
  readonly bundled?: boolean;
  readonly path?: string | null;
  readonly joint_q?: Readonly<Record<string, number>> | null;
}

export interface CalibrationJointLimit {
  readonly name: string;
  readonly child_link?: string;
  readonly parent_link?: string;
  readonly lower?: number;
  readonly upper?: number;
  readonly value?: number;
  readonly type?: string;
}

export interface CalibrationSession {
  readonly joint_q: Readonly<Record<string, number>>;
  readonly joint_limits: readonly CalibrationJointLimit[];
  readonly joint_world: Readonly<Record<string, CalibrationJointWorld>>;
  readonly reference: StageMotionPayload;
  readonly reference_name: string;
  readonly ground_offset_z: number;
  readonly has_saved_calibration: boolean;
}

export interface CalibrationPose {
  readonly links: readonly string[];
  readonly link_transforms: Readonly<Record<string, StageMatrix4>>;
  readonly joint_world: Readonly<Record<string, CalibrationJointWorld>>;
  readonly ground_offset_z: number;
}

export interface CalibrationProposal {
  readonly joint_q: Readonly<Record<string, number>>;
  readonly validation: {
    readonly valid: boolean;
    readonly score: number;
    readonly changed_joint_count: number;
    readonly edge_errors_deg: Readonly<Record<string, number>>;
    readonly near_limit_joints: readonly string[];
    readonly alignment_errors: readonly string[];
    readonly alignment_warnings: readonly string[];
    readonly foot_height_delta_m?: number | null;
  };
}

export interface H2rScenePayload {
  readonly terrain?: StageMotionPayload["terrain"];
  readonly objects?: StageMotionPayload["objects"];
}

/** Calibration-scaled human motion and scene, computed before IK runs. */
export interface ScaledPreviewResult {
  readonly preview: StageMotionPayload;
  readonly scaled_scene: H2rScenePayload | null;
}

/** Live result retained by FastAPI until its export token expires. */
export interface RetargetResult {
  readonly trajectory: StageRobotTrajectoryPayload;
  readonly scaled_preview?: StageMotionPayload | null;
  readonly scaled_scene?: H2rScenePayload | null;
  readonly diagnostics?: ResultDiagnosticsPayload;
  readonly export_token: string;
  readonly stem?: string;
  readonly motion_source_fps?: number;
  readonly retarget_fps?: number;
  readonly source_fps?: number;
  readonly has_scene?: boolean;
  readonly num_frames: number;
  readonly execution_provenance: ExecutionProvenance;
}

export interface RetargetRequest {
  readonly robot: string;
  readonly motion_token: string;
  readonly reference: string;
  readonly backend: "newton" | "interaction_mesh";
  readonly retarget_fps?: number;
}

interface RequestOptions {
  readonly signal?: AbortSignal;
  readonly fetcher?: Fetcher;
}

const calibrationReferencesCache = createRequestCache<readonly string[]>(1_000);

function jsonPost<T>(
  url: string,
  body: Readonly<Record<string, unknown>>,
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

export async function getCalibrationReferences(
  options: RequestOptions = {},
): Promise<readonly string[]> {
  const load = async (signal?: AbortSignal, fetcher?: Fetcher) => {
    const response = await requestJson<{ references?: unknown }>(
      "/api/calibration/references",
      { signal },
      fetcher,
    );
    return Array.isArray(response.references)
      ? response.references.filter(
          (reference): reference is string => typeof reference === "string",
        )
      : [];
  };
  if (options.fetcher) return load(options.signal, options.fetcher);
  return calibrationReferencesCache.read(() => load(), options.signal);
}

export function getCalibrationStatus(
  robot: string,
  reference: string,
  options: RequestOptions = {},
): Promise<CalibrationStatus> {
  const query = new URLSearchParams({ robot, reference });
  return requestJson<CalibrationStatus>(
    `/api/calibration/status?${query.toString()}`,
    { signal: options.signal },
    options.fetcher,
  );
}

export function startCalibrationSession(
  body: {
    readonly robot: string;
    readonly reference: string;
    readonly motion_token?: string;
  },
  options: RequestOptions = {},
): Promise<CalibrationSession> {
  return jsonPost<CalibrationSession>(
    "/api/calibration/session",
    body,
    options,
  );
}

export function saveCalibration(
  body: {
    readonly robot: string;
    readonly reference: string;
    readonly joint_q: Readonly<Record<string, number>>;
    readonly motion_token?: string;
  },
  options: RequestOptions = {},
): Promise<{ readonly ok: boolean; readonly path?: string }> {
  return jsonPost("/api/calibration/save", body, options);
}

export function proposeCalibration(
  body: {
    readonly robot: string;
    readonly reference: string;
    readonly joint_q: Readonly<Record<string, number>>;
    readonly motion_token?: string;
    readonly locked_joints?: readonly string[];
  },
  options: RequestOptions = {},
): Promise<CalibrationProposal> {
  return jsonPost("/api/calibration/propose", body, options);
}

export function previewCalibrationPose(
  robot: string,
  jointQ: Readonly<Record<string, number>>,
  options: RequestOptions = {},
): Promise<CalibrationPose> {
  return jsonPost("/api/robot/fk_preview", { robot, joint_q: jointQ }, options);
}

export function loadScaledPreview(
  body: {
    readonly robot: string;
    readonly motion_token: string;
    readonly reference: string;
  },
  options: RequestOptions = {},
): Promise<ScaledPreviewResult> {
  return jsonPost("/api/scaled_preview", body, options);
}

/** Start the existing Web H2R job and wait for its live result payload. */
export async function retarget(
  body: RetargetRequest,
  options: RequestOptions & {
    readonly onUpdate?: (job: JobSnapshot<RetargetResult>) => void;
  } = {},
): Promise<RetargetResult> {
  const started = await jsonPost<{ job_id?: unknown }>(
    "/api/retarget",
    {
      ...body,
      backend: body.backend,
      foot_clamp_anti_penetration: false,
    },
    options,
  );
  if (typeof started.job_id !== "string" || !started.job_id) {
    throw new Error("The server did not return a retarget job ID.");
  }
  return waitForJob<RetargetResult>(started.job_id, {
    signal: options.signal,
    fetcher: options.fetcher,
    expectedKind: "retarget",
    onUpdate: options.onUpdate,
  });
}

/** Build a same-origin download URL; export itself is synchronous, not a job. */
export function retargetExportUrl(
  token: string,
  options: ExportOptions,
): string {
  const query = new URLSearchParams({ fmt: options.format });
  if (options.fps !== undefined) query.set("fps", String(options.fps));
  if (options.csvHeader === false) query.set("csv_header", "0");
  if (options.start !== undefined) query.set("t_start", String(options.start));
  if (options.end !== undefined) query.set("t_end", String(options.end));
  return `/api/export/${encodeURIComponent(token)}?${query.toString()}`;
}
