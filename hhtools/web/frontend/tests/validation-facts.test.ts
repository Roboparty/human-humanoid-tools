import assert from "node:assert/strict";
import test from "node:test";

import {
  calibrationValidationFacts,
  motionValidationFacts,
  robotValidationFacts,
} from "../src/components/validationFacts.ts";
import type {
  StageMotionPayload,
  StageRobotPayload,
} from "../src/stage/types.ts";

const motion: StageMotionPayload = {
  positions: [[
    [0, 0, 0],
    [0, 0, 1],
  ]],
  parent_indices: [-1, 0],
  framerate: 30,
  body_mesh: { available: false, reason: "weights unavailable" },
  objects: [{ extents: [1, 1, 1], positions: [], quaternions: [] }],
};

const robot: StageRobotPayload = {
  name: "robot",
  display_name: "Robot",
  links: ["pelvis"],
  actuated_joints: ["hip"],
  num_dof: 1,
  ik_map: { hips: "pelvis", head: "missing" },
  link_transforms_zero: {},
};

test("derives compact Motion validation without another state model", () => {
  const facts = motionValidationFacts(motion);
  assert.deepEqual(facts.map((fact) => fact.tone), [
    "ok",
    "ok",
    "ok",
    "neutral",
    "ok",
  ]);
  assert.match(facts[4].label, /1 interaction object/);
});

test("reports robot mapping and renderability problems", () => {
  const facts = robotValidationFacts(robot);
  assert.equal(facts[0].tone, "ok");
  assert.equal(facts[2].tone, "error");
  assert.equal(facts[3].tone, "warning");
});

test("localizes Motion and Robot validation labels on request", () => {
  const chinese = (_english: string, translated: string) => translated;
  assert.match(motionValidationFacts(motion, chinese)[0].label, /可播放轨迹/);
  assert.match(robotValidationFacts(robot, chinese)[0].label, /可控自由度/);
});

test("reports changed and near-limit calibration joints", () => {
  const facts = calibrationValidationFacts(
    robot,
    [{ name: "hip", lower: -1, upper: 1 }],
    { hip: 0.98 },
  );
  assert.match(facts[2].label, /1\/1 joints/);
  assert.equal(facts[3].tone, "warning");
});
