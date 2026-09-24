/** Robot-specific types and request functions for the inspector. */

import {
  requestJson,
  uploadFiles,
  type Fetcher,
  type UploadFile,
} from "@/lib/api";
import { createRequestCache } from "@/lib/requestCache";
import type { StageRobotPayload } from "@/stage/types";

export interface RobotSummary {
  readonly name: string;
  readonly display_name: string;
  readonly has_urdf: boolean;
  readonly num_dof: number;
  readonly builtin?: boolean;
  readonly deletable: boolean;
}

export interface RobotsResponse {
  readonly robots: readonly RobotSummary[];
  readonly library_dir: string;
}

const robotLibraryCache = createRequestCache<RobotsResponse>(1_000);

/** Complete zero-pose payload returned by POST /api/robot/select. */
export type RobotPayload = StageRobotPayload;

export interface RobotRequestOptions {
  readonly signal?: AbortSignal;
  readonly fetcher?: Fetcher;
}

/** Read the current catalog; the backend remains authoritative for availability. */
export async function getRobotLibrary(
  options: RobotRequestOptions = {},
): Promise<RobotsResponse> {
  const load = async (signal?: AbortSignal, fetcher?: Fetcher) => {
    const response = await requestJson<RobotsResponse>(
      "/api/robots",
      { signal },
      fetcher,
    );
    return {
      library_dir:
        typeof response.library_dir === "string" ? response.library_dir : "",
      robots: Array.isArray(response.robots)
        ? response.robots.filter(
            (robot): robot is RobotSummary =>
              Boolean(robot) &&
              typeof robot.name === "string" &&
              typeof robot.display_name === "string",
          )
        : [],
    };
  };
  if (options.fetcher) return load(options.signal, options.fetcher);
  return robotLibraryCache.read(() => load(), options.signal);
}

/** Load one robot and return its serialized zero-pose model for the Stage. */
export async function loadRobot(
  name: string,
  options: RobotRequestOptions = {},
): Promise<RobotPayload> {
  const normalized = name.trim();
  if (!normalized) throw new Error("Select a robot first.");
  return requestJson<RobotPayload>(
    "/api/robot/select",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: normalized }),
      signal: options.signal,
    },
    options.fetcher,
  );
}

/** Persist a complete URDF bundle and return its loaded zero-pose payload. */
export function uploadRobot(
  files: Iterable<UploadFile | File>,
  name: string,
  options: RobotRequestOptions = {},
): Promise<RobotPayload> {
  return uploadFiles<RobotPayload>("/api/robot/upload", files, {
    query: { name },
    signal: options.signal,
    fetcher: options.fetcher,
  });
}

/** Remove one backend-confirmed user robot from the persistent library. */
export function deleteRobot(
  name: string,
  options: RobotRequestOptions = {},
): Promise<{ readonly ok: boolean; readonly deleted: string }> {
  return requestJson(
    `/api/robot/${encodeURIComponent(name)}`,
    { method: "DELETE", signal: options.signal },
    options.fetcher,
  );
}
