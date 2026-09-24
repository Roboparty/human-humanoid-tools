import assert from "node:assert/strict";
import test from "node:test";

import {
  workflowPipelineState,
  workflowStatusToneClass,
  type WorkflowStatusTone,
} from "../src/components/workflowPipeline.ts";

test("marks completed, active, and pending pipeline steps", () => {
  assert.equal(workflowPipelineState(0, 2, 1), "complete");
  assert.equal(workflowPipelineState(1, 2, 1), "complete");
  assert.equal(workflowPipelineState(2, 2, 1), "active");
  assert.equal(workflowPipelineState(3, 2, 1), "pending");
});

test("allows the final successful step to turn green", () => {
  assert.equal(workflowPipelineState(3, 3, 3), "complete");
});

test("maps workflow status tones to semantic Tailwind colors", () => {
  const expected: Readonly<Record<WorkflowStatusTone, string>> = {
    neutral: "text-muted-foreground",
    info: "text-primary",
    success: "text-success",
    warning: "text-warning",
    danger: "text-danger",
  };

  for (const [tone, className] of Object.entries(expected)) {
    assert.equal(workflowStatusToneClass(tone as WorkflowStatusTone), className);
  }
});
