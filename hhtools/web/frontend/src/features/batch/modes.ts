import { isVideoToMotionEnabled } from "../../platformCapabilities.ts";

export type BatchMode = "v2m" | "h2r" | "r2r";

const MODES = [
  { id: "v2m", label: "V2M" },
  { id: "h2r", label: "H2R" },
  { id: "r2r", label: "R2R" },
] as const;

export function getBatchModes(videoToMotionEnabled = isVideoToMotionEnabled()) {
  return MODES.filter((mode) => mode.id !== "v2m" || videoToMotionEnabled);
}
