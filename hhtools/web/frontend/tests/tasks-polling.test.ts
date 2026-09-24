import assert from "node:assert/strict";
import test from "node:test";

import { taskPollingDelay } from "../src/features/tasks/polling.ts";

test("closed idle and hidden task drawers do not poll", () => {
  assert.equal(
    taskPollingDelay({
      visible: true,
      drawerOpen: false,
      hasActiveTasks: false,
      consecutiveFailures: 0,
    }),
    null,
  );
  assert.equal(
    taskPollingDelay({
      visible: false,
      drawerOpen: true,
      hasActiveTasks: true,
      consecutiveFailures: 0,
    }),
    null,
  );
});

test("active jobs poll quickly while an open idle drawer backs off", () => {
  assert.equal(
    taskPollingDelay({
      visible: true,
      drawerOpen: false,
      hasActiveTasks: true,
      consecutiveFailures: 0,
    }),
    2_500,
  );
  assert.equal(
    taskPollingDelay({
      visible: true,
      drawerOpen: true,
      hasActiveTasks: false,
      consecutiveFailures: 0,
    }),
    15_000,
  );
  assert.equal(
    taskPollingDelay({
      visible: true,
      drawerOpen: true,
      hasActiveTasks: true,
      consecutiveFailures: 5,
    }),
    60_000,
  );
});
