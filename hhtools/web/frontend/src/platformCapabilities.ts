import type { ApplicationImportTarget } from "./importIntent";
import type { ViewId } from "./navigation";

interface DesktopCapabilityHost {
  readonly hhtoolsDesktop?: {
    readonly capabilities?: {
      readonly videoToMotion?: boolean;
    };
  };
}

/** Ordinary Web and older desktop bridges retain their existing workflows. */
export function isVideoToMotionEnabled(host: unknown = globalThis): boolean {
  return (host as DesktopCapabilityHost | null | undefined)
    ?.hhtoolsDesktop?.capabilities?.videoToMotion !== false;
}

export function isViewAvailable(
  view: ViewId,
  videoToMotionEnabled = isVideoToMotionEnabled(),
): boolean {
  return view !== "video-to-motion" || videoToMotionEnabled;
}

export function isImportAvailable(
  target: ApplicationImportTarget,
  videoToMotionEnabled = isVideoToMotionEnabled(),
): boolean {
  return target !== "video-file" || videoToMotionEnabled;
}
