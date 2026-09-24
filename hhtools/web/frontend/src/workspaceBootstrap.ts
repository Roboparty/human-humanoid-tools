import { getDatasetCatalog } from "@/features/analysis/api";
import { getCalibrationReferences } from "@/features/h2r/api";
import { getMotionLibrary } from "@/features/motion/api";
import { getRobotLibrary } from "@/features/robot/api";
import { listTasks } from "@/features/tasks/api";

type BootstrapLoader = (signal: AbortSignal) => Promise<unknown>;

export interface WorkspaceBootstrapLoaders {
  readonly motionLibrary: BootstrapLoader;
  readonly robotCatalog: BootstrapLoader;
  readonly datasetCatalog: BootstrapLoader;
  readonly calibrationReferences: BootstrapLoader;
  readonly taskHistory: BootstrapLoader;
}

const defaultLoaders: WorkspaceBootstrapLoaders = {
  motionLibrary: (signal) => getMotionLibrary({ signal }),
  robotCatalog: (signal) => getRobotLibrary({ signal }),
  datasetCatalog: (signal) => getDatasetCatalog({ signal }),
  calibrationReferences: (signal) => getCalibrationReferences({ signal }),
  taskHistory: (signal) => listTasks({ signal }),
};

/** Wait for every lightweight core catalog to succeed or fail before reveal. */
export async function preloadCoreWorkspace(
  signal: AbortSignal,
  loaders: WorkspaceBootstrapLoaders = defaultLoaders,
): Promise<void> {
  await Promise.allSettled(
    Object.values(loaders).map((load) => load(signal)),
  );
}
