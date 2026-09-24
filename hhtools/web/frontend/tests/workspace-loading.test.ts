import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const appSource = await readFile(new URL("../src/App.tsx", import.meta.url), "utf8");
const batchSource = await readFile(
  new URL("../src/features/batch/BatchView.tsx", import.meta.url),
  "utf8",
);
const bootstrapSource = await readFile(
  new URL("../src/workspaceBootstrap.ts", import.meta.url),
  "utf8",
);

test("core workspaces stay eager while Video to Motion loads on first use", () => {
  assert.doesNotMatch(appSource, /mountedViews/);

  for (const view of [
    "AnalysisView",
    "BatchView",
    "HumanToRobotView",
    "RobotToRobotView",
    "RobotView",
  ]) {
    assert.match(appSource, new RegExp(`import \\{ ${view} \\}`));
  }
  assert.match(
    appSource,
    /lazy\(async \(\) => \(\{[\s\S]*features\/video-to-motion\/VideoToMotionView/,
  );
  assert.match(appSource, /fallback=\{<VideoWorkspaceLoading \/>\}/);
  assert.match(batchSource, /lazy\(async \(\) => \(\{[\s\S]*\.\/VideoBatchView/);
  assert.match(batchSource, /Loading Video Batch/);
  assert.match(appSource, /preloadCoreWorkspace/);
  assert.match(appSource, /coreWorkspaceReady \? "" : "invisible"/);
  assert.doesNotMatch(bootstrapSource, /video-to-motion|Gvhmr|SMPL/);
});
