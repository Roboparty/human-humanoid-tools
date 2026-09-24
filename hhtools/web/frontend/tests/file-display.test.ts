import assert from "node:assert/strict";
import test from "node:test";

import { displayFileName } from "../src/lib/api.ts";

test("renders only a portable leaf name from internal path identities", () => {
  assert.equal(displayFileName("/srv/private/motions/walk.bvh"), "walk.bvh");
  assert.equal(displayFileName("C:\\Users\\name\\walk.bvh"), "walk.bvh");
  assert.equal(displayFileName("", "Motion"), "Motion");
});
