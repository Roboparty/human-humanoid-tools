import type { MotionLibraryEntry } from "@/features/motion/api";
import type { UploadFile } from "@/lib/api";

import type { BatchBackend } from "./api";

const DATASET_REFERENCES: Readonly<Record<string, string>> = {
  amass: "smpl",
  motion_x: "smplx",
  phuma: "smpl",
  lafan: "lafan_bvh",
  mocap: "mocap_bvh",
  soma: "soma_bvh",
  xsens_mocap: "xsens_mocap",
  gvhmr: "gvhmr",
  omomo: "smplx",
  meshmimic_holosoma: "smplx",
  glb: "glb",
  unified_npz: "smpl",
  parc_ms: "smpl",
};

export function entryKey(entry: MotionLibraryEntry): string {
  return (
    entry.source_path ||
    entry.token ||
    [entry.folder_label, entry.stem].filter(Boolean).join("/")
  );
}

export function entryTitle(entry: MotionLibraryEntry): string {
  return (
    entry.stem ||
    entry.sequence_id ||
    entry.display_name ||
    entry.label ||
    entry.name ||
    entry.source_path.split("/").pop() ||
    "Untitled motion"
  );
}

export function entryReference(
  entry: MotionLibraryEntry,
  fallback = "smpl",
): string {
  const explicit = entry.reference?.trim();
  if (explicit) return explicit;
  return (entry.dataset && DATASET_REFERENCES[entry.dataset]) || fallback;
}

export function appendUniqueEntries(
  current: readonly MotionLibraryEntry[],
  incoming: readonly MotionLibraryEntry[],
): readonly MotionLibraryEntry[] {
  const keys = new Set(current.map(entryKey));
  const next = [...current];
  for (const entry of incoming) {
    const key = entryKey(entry);
    if (!key || keys.has(key)) continue;
    keys.add(key);
    next.push(entry);
  }
  return next;
}

/** Keep the backend's complete Library row when a generated motion is published. */
export function publishedMotionEntry(value: unknown): MotionLibraryEntry | null {
  if (!value || typeof value !== "object") return null;
  const sourcePath = (value as { source_path?: unknown }).source_path;
  return typeof sourcePath === "string" && sourcePath.trim()
    ? (value as MotionLibraryEntry)
    : null;
}

export function suggestedBackend(
  entries: readonly MotionLibraryEntry[],
): BatchBackend | undefined {
  const suggestion = [...entries]
    .reverse()
    .find((entry) => entry.suggested_backend)?.suggested_backend;
  if (suggestion === "newton" || suggestion === "interaction_mesh") {
    return suggestion;
  }
  return undefined;
}

export function optionalPositiveNumber(value: string): number | undefined {
  const normalized = value.trim();
  if (!normalized) return undefined;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

export function optionalNonNegativeNumber(value: string): number | undefined {
  const normalized = value.trim();
  if (!normalized) return undefined;
  const parsed = Number(normalized);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : undefined;
}

export function timeRangeError(start: string, end: string): string | null {
  const parsedStart = optionalNonNegativeNumber(start);
  const parsedEnd = optionalNonNegativeNumber(end);
  if ((start.trim() && parsedStart === undefined) || (end.trim() && parsedEnd === undefined)) {
    return "Enter a valid non-negative time range.";
  }
  if (parsedStart !== undefined && parsedEnd !== undefined && parsedStart > parsedEnd) {
    return "Start time cannot be later than end time.";
  }
  return null;
}

export function uploadFileKey(file: UploadFile | File): string {
  const upload = file as UploadFile;
  const path = upload._relpath || upload.webkitRelativePath || upload.name;
  return `${path}:${upload.size}:${upload.lastModified}`;
}
