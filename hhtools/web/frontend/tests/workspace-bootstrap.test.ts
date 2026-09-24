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

const { preloadCoreWorkspace } = await import("../src/workspaceBootstrap.ts");

test("core workspace preload starts every lightweight catalog and tolerates failures", async () => {
  const calls: string[] = [];
  const loader = (name: string, reject = false) => async () => {
    calls.push(name);
    if (reject) throw new Error(name);
  };
  const loaders = {
    motionLibrary: loader("motion"),
    robotCatalog: loader("robots"),
    datasetCatalog: loader("dataset", true),
    calibrationReferences: loader("calibration"),
    taskHistory: loader("tasks"),
  };

  await preloadCoreWorkspace(new AbortController().signal, loaders);

  assert.deepEqual(calls.sort(), [
    "calibration",
    "dataset",
    "motion",
    "robots",
    "tasks",
  ]);
});
