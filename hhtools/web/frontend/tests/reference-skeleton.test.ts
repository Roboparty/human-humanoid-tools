import assert from "node:assert/strict";
import test from "node:test";
import * as THREE from "three";

import {
  calibrationMappingKey,
  calibrationMappingLabel,
  layoutCalibrationLabels,
  projectCalibrationEndpoints,
} from "../src/stage/calibrationOverlay.ts";
import {
  normalizeReferenceSemantic,
  prepareReferenceSkeleton,
  referenceTargetLink,
} from "../src/stage/referenceSkeleton.ts";
import type {
  StageMotionPayload,
  StageRobotPayload,
} from "../src/stage/types.ts";
import { REFERENCE_SKELETON_VISUAL } from "../src/stage/visualStyle.ts";

const reference: StageMotionPayload = {
  positions: [[
    [0, 0, 1],
    [1, 0, 1],
    [2, 0, 1],
    [3, 0, 1],
  ]],
  parent_indices: [-1, 0, 1, 2],
  bone_names: ["Pelvis", "LeftHand", "Spare Bone", "Hidden"],
  canonical_names: ["hips", "left_wrist", "other", "head"],
  quaternions: [[
    [0, 0, 0, 1],
    [0.1, 0.2, 0.3, 0.9],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
  ]],
  exclude_joint_indices: [3],
  color: 0x123456,
};

const robot: StageRobotPayload = {
  name: "target",
  display_name: "Target",
  links: ["pelvis_link", "wrist_link", "spare_link", "head_link"],
  link_transforms_zero: {},
  ik_map: {
    hips: "pelvis_link",
    left_wrist: { t_body: "wrist_link" },
    spare_bone: { link: "spare_link" },
    head: { target: "head_link" },
  },
};

test("maps calibration landmarks through canonical and bone names", () => {
  const prepared = prepareReferenceSkeleton(reference, robot);
  assert.equal(prepared.color, 0x123456);
  assert.deepEqual(prepared.edges, [
    { child: 1, parent: 0 },
    { child: 2, parent: 1 },
  ]);
  assert.deepEqual(
    prepared.mappings.map(({ semantic, targetLink, index }) => ({
      semantic,
      targetLink,
      index,
    })),
    [
      { semantic: "hips", targetLink: "pelvis_link", index: 0 },
      { semantic: "left_wrist", targetLink: "wrist_link", index: 1 },
      { semantic: "spare_bone", targetLink: "spare_link", index: 2 },
    ],
  );
  assert.deepEqual(
    prepared.mappingByJoint.get(1)?.quaternion,
    [0.1, 0.2, 0.3, 0.9],
  );
  assert.equal(prepared.mappingByJoint.has(3), false);
});

test("normalizes IK map shapes used by robot presets", () => {
  assert.equal(normalizeReferenceSemantic("Left_Wrist-01"), "leftwrist01");
  assert.equal(referenceTargetLink("pelvis"), "pelvis");
  assert.equal(referenceTargetLink({ t_body: "wrist" }), "wrist");
  assert.equal(referenceTargetLink({ link: "ankle" }), "ankle");
  assert.equal(referenceTargetLink({ body: "head" }), "head");
  assert.equal(referenceTargetLink({ target: "foot" }), "foot");
  assert.equal(referenceTargetLink({}), null);
});

test("keeps the original mapped and context reference appearance", () => {
  assert.deepEqual(REFERENCE_SKELETON_VISUAL, {
    color: 0x5eb3ff,
    jointRadius: 0.022,
    jointSegments: 12,
    sourceOpacity: 0.82,
    mappedScale: 1.12,
    contextScale: 0.62,
    lineOpacityFactor: 0.38,
    mapped: {
      roughness: 0.34,
      metalness: 0.03,
      emissive: 0x0a4d92,
      emissiveIntensity: 0.62,
    },
    context: {
      roughness: 0.48,
      metalness: 0.02,
      emissive: 0x1a3a66,
      emissiveIntensity: 0.18,
      opacityFactor: 0.32,
    },
  });
});

test("formats the original semantic label and a stable mapping identity", () => {
  const mapping = prepareReferenceSkeleton(reference, robot).mappings[1];
  assert.equal(calibrationMappingLabel(mapping), "Left wrist · wrist_link");
  assert.equal(
    calibrationMappingKey(mapping),
    "1\u0000left_wrist\u0000wrist_link",
  );
});

test("projects reference-to-robot endpoints and hides points outside depth", () => {
  const camera = new THREE.PerspectiveCamera(50, 2, 0.1, 100);
  camera.position.set(0, 0, 5);
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);

  const projected = projectCalibrationEndpoints(
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(1, 0, 0),
    camera,
    200,
    100,
  );
  assert.ok(projected);
  assert.equal(projected.referenceX, 100);
  assert.equal(projected.referenceY, 50);
  assert.ok(projected.targetX > projected.referenceX);
  assert.equal(projected.targetY, 50);

  assert.equal(
    projectCalibrationEndpoints(
      new THREE.Vector3(0, 0, 10),
      new THREE.Vector3(1, 0, 0),
      camera,
      200,
      100,
    ),
    null,
  );
});

test("packs projected calibration labels inside two non-overlapping lanes", () => {
  const layout = layoutCalibrationLabels(
    [
      { key: "left-a", anchorX: 80, anchorY: 2, width: 70, height: 18 },
      { key: "left-b", anchorX: 90, anchorY: 5, width: 70, height: 18 },
      { key: "right-a", anchorX: 160, anchorY: 95, width: 70, height: 18 },
      { key: "right-b", anchorX: 170, anchorY: 98, width: 70, height: 18 },
    ],
    240,
    120,
  );
  const byKey = Object.fromEntries(layout.map((item) => [item.key, item]));
  assert.ok(byKey["left-a"].top >= 4);
  assert.ok(byKey["right-b"].top + 18 <= 116);
  assert.ok(byKey["left-b"].top >= byKey["left-a"].top + 21);
  assert.ok(byKey["right-b"].top >= byKey["right-a"].top + 21);
});
