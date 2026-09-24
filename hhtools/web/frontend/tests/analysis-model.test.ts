import assert from "node:assert/strict";
import test from "node:test";

import {
  clipMatchesFilters,
  clipsWithScatter,
  scatterBounds,
  selectScatterClip,
} from "../src/features/analysis/model.ts";
import type { DatasetClip } from "../src/features/analysis/api.ts";

function clip(
  id: string,
  scatter: readonly [number, number],
  overrides: Partial<DatasetClip> = {},
): DatasetClip {
  return {
    clip_id: id,
    source_kind: "human",
    source_path: `/data/${id}.bvh`,
    dataset: "test",
    folder_label: "walks",
    metrics: { complexity: 1 },
    tags: ["quality_ok"],
    embedding: null,
    scatter,
    cluster_id: 0,
    error: null,
    ...overrides,
  };
}

test("scatter bounds stay anchored to all valid clips while filters dim points", () => {
  const all = clipsWithScatter([
    clip("left", [-10, -2]),
    clip("middle", [0, 1], { tags: ["quality_bad"] }),
    clip("right", [20, 5]),
  ]);
  const visible = all.filter((item) =>
    clipMatchesFilters(item, {
      tag: "quality_ok",
      kind: "all",
      folder: "all",
    }),
  );

  assert.deepEqual(scatterBounds(all), {
    minX: -10,
    maxX: 20,
    minY: -2,
    maxY: 5,
  });
  assert.deepEqual(visible.map((item) => item.clip_id), ["left", "right"]);
});

test("scatter selection keeps legacy single-click and Shift-toggle behavior", () => {
  const recommended = new Set(["recommended"]);

  assert.deepEqual([...selectScatterClip(new Set(["a", "b"]), "c", false, recommended)], ["c"]);
  assert.deepEqual([...selectScatterClip(new Set(["a"]), "a", false, recommended)], []);
  assert.deepEqual([...selectScatterClip(new Set(["a"]), "b", true, recommended)], ["a", "b"]);
  assert.deepEqual([...selectScatterClip(new Set(["a", "b"]), "a", true, recommended)], ["b"]);
  assert.deepEqual(
    [...selectScatterClip(new Set(["a"]), "recommended", false, recommended)],
    ["a"],
  );
});

test("metric range participates in the visible-set filter", () => {
  const filters = {
    tag: "all",
    kind: "all",
    folder: "all",
    metric: "complexity",
    metricRange: [1, 2] as const,
  };
  assert.equal(clipMatchesFilters(clip("inside", [0, 0]), filters), true);
  assert.equal(
    clipMatchesFilters(clip("outside", [0, 0], { metrics: { complexity: 3 } }), filters),
    false,
  );
});
