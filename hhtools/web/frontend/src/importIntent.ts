export type ApplicationImportTarget =
  | "motion-file"
  | "motion-folder"
  | "video-file"
  | "robot-urdf"
  | "robot-mesh-folder";

/** Monotonic identity lets a mounted feature handle each menu request once. */
export interface ApplicationImportRequest {
  readonly id: number;
  readonly target: ApplicationImportTarget;
}
