import assert from "node:assert/strict";
import test from "node:test";

import {
  comparisonLayers,
  COMPARISON_PRESET_STORAGE_KEY,
  isComparisonPreset,
  storedComparisonPreset,
  storeComparisonPreset,
} from "../src/features/result/comparison.ts";

test("comparison presets project the legacy H2R and R2R layer groups", () => {
  assert.deepEqual(comparisonLayers("h2r", "source"), [
    "skeleton",
    "body",
    "objects",
  ]);
  assert.deepEqual(comparisonLayers("h2r", "result"), [
    "scaled-scene",
    "robot",
  ]);
  assert.deepEqual(comparisonLayers("r2r", "target"), [
    "r2r-target-skeleton",
    "r2r-target-scene",
  ]);
  assert.deepEqual(comparisonLayers("r2r", "overlay"), [
    "r2r-source-robot",
    "r2r-target-robot",
    "r2r-target-skeleton",
    "r2r-target-scene",
  ]);
});

function memoryStorage(initial: string | null = null) {
  let value = initial;
  return {
    getItem(key: string): string | null {
      assert.equal(key, COMPARISON_PRESET_STORAGE_KEY);
      return value;
    },
    setItem(key: string, next: string): void {
      assert.equal(key, COMPARISON_PRESET_STORAGE_KEY);
      value = next;
    },
    value: () => value,
  };
}

test("validates the four comparison presets", () => {
  for (const preset of ["source", "target", "result", "overlay"]) {
    assert.equal(isComparisonPreset(preset), true);
  }
  assert.equal(isComparisonPreset("all"), false);
  assert.equal(isComparisonPreset(null), false);
});

test("reads validated workflow presets and falls back to overlay", () => {
  assert.equal(storedComparisonPreset(undefined, "h2r"), "overlay");
  assert.equal(storedComparisonPreset(memoryStorage("not-json"), "r2r"), "overlay");
  assert.equal(
    storedComparisonPreset(memoryStorage('{"h2r":"source","r2r":"invalid"}'), "h2r"),
    "source",
  );
  assert.equal(
    storedComparisonPreset(memoryStorage('{"h2r":"source","r2r":"invalid"}'), "r2r"),
    "overlay",
  );
});

test("stores one workflow without discarding the other valid preset", () => {
  const storage = memoryStorage('{"h2r":"source","r2r":"target","extra":true}');
  storeComparisonPreset(storage, "h2r", "result");
  assert.deepEqual(JSON.parse(storage.value() ?? "{}"), {
    h2r: "result",
    r2r: "target",
  });
});

test("storage failures do not escape the preference boundary", () => {
  const storage = {
    getItem(): string | null {
      throw new Error("blocked");
    },
    setItem(): void {
      throw new Error("blocked");
    },
  };
  assert.equal(storedComparisonPreset(storage, "h2r"), "overlay");
  assert.doesNotThrow(() => storeComparisonPreset(storage, "r2r", "source"));
});
