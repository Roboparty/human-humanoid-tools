import assert from "node:assert/strict";
import { registerHooks } from "node:module";
import test from "node:test";

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (!specifier.startsWith("@/")) return nextResolve(specifier, context);
    const url = new URL(`../src/${specifier.slice(2)}.ts`, import.meta.url);
    return nextResolve(url.href, context);
  },
});

const { canExportTaskResult, isWorkflowResultTask } = await import(
  "../src/features/tasks/api.ts"
);

test("only completed workflow tasks expose result export", () => {
  for (const kind of [
    "video_to_motion",
    "retarget",
    "r2r_retarget",
    "batch",
    "r2r_batch",
  ]) {
    assert.equal(isWorkflowResultTask({ kind }), true);
    assert.equal(canExportTaskResult({ kind, can_download: true }), true);
    assert.equal(canExportTaskResult({ kind, can_download: false }), false);
  }

  for (const kind of ["motion_load", "motion_link", "dataset_analyze"]) {
    assert.equal(isWorkflowResultTask({ kind }), false);
    assert.equal(canExportTaskResult({ kind, can_download: true }), false);
  }
});
