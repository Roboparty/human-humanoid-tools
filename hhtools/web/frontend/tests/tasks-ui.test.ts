import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL("../src/features/tasks/TaskDrawer.tsx", import.meta.url),
  "utf8",
);

test("workflow task rows keep a stable Library-style Export action", () => {
  assert.match(source, /isWorkflowResultTask\(task\)/);
  assert.match(source, /className=\{TASK_ACTION_CLASS\}/);
  assert.match(source, /text\("Export", "导出"\)/);
  assert.match(source, /text-primary/);
  assert.match(source, /chevron-down\.svg/);
  assert.match(source, /-rotate-90/);
  assert.doesNotMatch(source, /Export Result|导出结果/);
});
