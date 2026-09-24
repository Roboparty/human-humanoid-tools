import assert from "node:assert/strict";
import test from "node:test";

import {
  calibrationDragValue,
  jointAxisParameter,
  jointDragVector,
  jointPlanePoint,
  signedJointDragAngle,
} from "../src/stage/calibrationManipulatorMath.ts";

test("projects a pointer ray onto the plane normal to the joint axis", () => {
  assert.deepEqual(
    jointPlanePoint([1, 2, 5], [0, 0, -1], [0, 0, 1], [0, 0, 1]),
    [1, 2, 1],
  );
  assert.equal(
    jointPlanePoint([0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]),
    null,
  );
});

test("measures a signed rotation around the supplied world axis", () => {
  const start = jointDragVector([1, 0, 0], [0, 0, 0]);
  const quarterTurn = jointDragVector([0, 1, 0], [0, 0, 0]);
  assert.ok(start && quarterTurn);
  assert.ok(
    Math.abs(signedJointDragAngle(start, quarterTurn, [0, 0, 1]) - Math.PI / 2) < 1e-9,
  );
  assert.ok(
    Math.abs(signedJointDragAngle(start, quarterTurn, [0, 0, -1]) + Math.PI / 2) < 1e-9,
  );
});

test("clamps a dragged joint value to its limits", () => {
  assert.equal(calibrationDragValue(0.25, 0.5, { lower: -1, upper: 1 }), 0.75);
  assert.equal(calibrationDragValue(0.75, 0.5, { lower: -1, upper: 1 }), 1);
  assert.equal(calibrationDragValue(-0.75, -0.5, { lower: -1, upper: 1 }), -1);
});

test("measures pointer movement along a prismatic joint axis", () => {
  assert.equal(
    jointAxisParameter([0.6, 2, 0], [0, -1, 0], [0, 0, 0], [1, 0, 0]),
    0.6,
  );
  assert.equal(
    jointAxisParameter([0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]),
    null,
  );
});
