import assert from "node:assert/strict";
import test from "node:test";

import {
  angleForDisplay,
  angleFromDisplay,
  calibrationJointMatches,
  classifyCalibrationJoint,
  isNearCalibrationLimit,
  normalizeCalibrationValues,
  resolveCalibrationJointLimits,
  setCalibrationJointValue,
  zeroCalibrationRegion,
} from "../src/components/calibrationEditorState.ts";
import {
  DEFAULT_CALIBRATION_DISPLAY,
  updateCalibrationDisplay,
} from "../src/stage/calibrationDisplay.ts";

const limits = [
  { name: "left_shoulder_pitch", lower: -1, upper: 2 },
  { name: "right_knee", lower: -2, upper: 0 },
  { name: "waist_yaw", lower: 1, upper: 1 },
] as const;

test("normalizes limits and clamps every canonical joint value", () => {
  const resolved = resolveCalibrationJointLimits(limits, { extra_joint: 9 });
  assert.deepEqual(resolved[2], {
    name: "waist_yaw",
    lower: -Math.PI,
    upper: Math.PI,
  });
  assert.deepEqual(resolved[3], {
    name: "extra_joint",
    lower: -Math.PI,
    upper: Math.PI,
  });
  assert.deepEqual(
    normalizeCalibrationValues(limits, {
      left_shoulder_pitch: 5,
      right_knee: -5,
      waist_yaw: Number.NaN,
      extra_joint: 9,
    }),
    {
      left_shoulder_pitch: 2,
      right_knee: -2,
      waist_yaw: 0,
      extra_joint: Math.PI,
    },
  );
  assert.equal(
    setCalibrationJointValue(limits, {}, "left_shoulder_pitch", -4)
      .left_shoulder_pitch,
    -1,
  );
});

test("uses the legacy three-percent near-limit threshold", () => {
  const limit = { name: "joint", lower: -1, upper: 1 };
  assert.equal(isNearCalibrationLimit(-0.95, limit), true);
  assert.equal(isNearCalibrationLimit(0, limit), false);
  assert.equal(isNearCalibrationLimit(0.95, limit), true);
});

test("keeps radians canonical while converting degree inputs", () => {
  assert.ok(Math.abs(angleForDisplay(Math.PI / 2, "deg") - 90) < 1e-8);
  assert.ok(Math.abs(angleFromDisplay(180, "deg") - Math.PI) < 1e-8);
  assert.equal(angleFromDisplay(0.5, "rad"), 0.5);
});

test("retains prismatic type metadata for metre-based editors", () => {
  assert.deepEqual(
    resolveCalibrationJointLimits([
      { name: "linear_joint", type: "prismatic", lower: -0.2, upper: 0.3 },
    ])[0],
    {
      name: "linear_joint",
      type: "prismatic",
      lower: -0.2,
      upper: 0.3,
    },
  );
});

test("filters named regions and zeros only the selected region", () => {
  assert.equal(classifyCalibrationJoint("left_shoulder_pitch"), "left-arm");
  assert.equal(classifyCalibrationJoint("right_knee"), "right-leg");
  assert.equal(classifyCalibrationJoint("waist_yaw"), "torso");
  assert.equal(
    calibrationJointMatches("left_shoulder_pitch", "pitch", "left-arm"),
    true,
  );
  assert.equal(calibrationJointMatches("right_knee", "pitch", "right-leg"), false);
  assert.deepEqual(
    zeroCalibrationRegion(
      limits,
      { left_shoulder_pitch: 1, right_knee: -1, waist_yaw: 0.5 },
      "left-arm",
    ),
    { left_shoulder_pitch: 0, right_knee: -1, waist_yaw: 0.5 },
  );
});

test("retains legacy display defaults and clamps opacity controls", () => {
  assert.deepEqual(DEFAULT_CALIBRATION_DISPLAY, {
    mappedOnly: true,
    labels: true,
    mappingLines: true,
    referenceOpacity: 0.82,
    robotOpacity: 0.72,
  });
  assert.deepEqual(
    updateCalibrationDisplay(DEFAULT_CALIBRATION_DISPLAY, {
      labels: false,
      referenceOpacity: 2,
      robotOpacity: 0,
    }),
    {
      ...DEFAULT_CALIBRATION_DISPLAY,
      labels: false,
      referenceOpacity: 1,
      robotOpacity: 0.2,
    },
  );
});
