import assert from "node:assert/strict";
import test from "node:test";

const { buildTrackingChart, resolveTrackingQuality, topEffectors, validateExportOptions } =
  await import("../src/features/result/model.ts");

test("export options preserve valid FPS, time window, format, and CSV header", () => {
  assert.deepEqual(
    validateExportOptions({
      format: "csv",
      fps: "60",
      start: "1.25",
      end: "2.5",
      csvHeader: false,
    }),
    {
      valid: true,
      options: {
        format: "csv",
        fps: 60,
        start: 1.25,
        end: 2.5,
        csvHeader: false,
      },
    },
  );
});

test("export options reject invalid numeric bounds and empty windows", () => {
  assert.deepEqual(
    validateExportOptions({
      format: "pkl",
      fps: "0",
      start: "",
      end: "",
      csvHeader: true,
    }),
    { valid: false, error: "Export FPS must be greater than zero." },
  );
  assert.deepEqual(
    validateExportOptions({
      format: "csv",
      fps: "",
      start: "4",
      end: "4",
      csvHeader: true,
    }),
    { valid: false, error: "End time must be greater than start time." },
  );
  assert.deepEqual(
    validateExportOptions({
      format: "csv",
      fps: "",
      start: "",
      end: "0",
      csvHeader: true,
    }),
    { valid: false, error: "End time must be greater than start time." },
  );
  assert.deepEqual(
    validateExportOptions({
      format: "csv",
      fps: "",
      start: "-0.1",
      end: "",
      csvHeader: true,
    }),
    { valid: false, error: "Start time must be zero or greater." },
  );
});

test("tracking chart uses finite frame errors and reports its peak", () => {
  const chart = buildTrackingChart([
    {
      frame: 0,
      time_s: 0,
      mean_error_m: 0.01,
      max_error_m: 0.05,
      source_contacts: 0,
      target_contacts: 0,
    },
    {
      frame: 1,
      time_s: 1 / 30,
      mean_error_m: 0.04,
      max_error_m: 0.1,
      source_contacts: 1,
      target_contacts: 1,
    },
  ]);

  assert.equal(chart?.peak, 0.1);
  assert.match(chart?.meanPoints ?? "", /^0\.0,/);
  assert.match(chart?.meanPoints ?? "", /320\.0,/);
  assert.doesNotMatch(chart?.maxPoints ?? "", /NaN|Infinity/);
});

test("top effectors are ranked by P95 error without mutating the payload", () => {
  const effectors = [
    { canonical: "left_wrist", target_link: "hand", sample_count: 2, mean_error_m: 0.01, p95_error_m: 0.02, max_error_m: 0.03 },
    { canonical: "left_ankle", target_link: "foot", sample_count: 2, mean_error_m: 0.03, p95_error_m: 0.08, max_error_m: 0.1 },
    { canonical: "hips", target_link: "pelvis", sample_count: 2, mean_error_m: 0.02, p95_error_m: 0.04, max_error_m: 0.05 },
  ] as const;

  assert.deepEqual(
    topEffectors(effectors, 2).map((effector) => effector.canonical),
    ["left_ankle", "hips"],
  );
  assert.deepEqual(effectors.map((effector) => effector.canonical), [
    "left_wrist",
    "left_ankle",
    "hips",
  ]);
});

function diagnostics(p95: number, quality?: Record<string, unknown>) {
  return {
    schema_version: 1,
    available: true,
    quality,
    tracking: { mean_error_m: p95, p95_error_m: p95, max_error_m: p95, effectors: [], series: [] },
  };
}

test("tracking quality uses robot-height-normalized limits from the backend", () => {
  // 1.8 m robot: 4 % / 8 % -> 7.2 cm / 14.4 cm, so 9 cm is only "review".
  const quality = { band: "review", stable_max_m: 0.072, review_max_m: 0.144, robot_height_m: 1.8 };
  assert.deepEqual(resolveTrackingQuality(diagnostics(0.09, quality)), {
    band: "review",
    stableMaxM: 0.072,
    reviewMaxM: 0.144,
    robotHeightM: 1.8,
  });
  assert.equal(resolveTrackingQuality(diagnostics(0.15, quality)).band, "high_error");
  assert.equal(resolveTrackingQuality(diagnostics(0.07, quality)).band, "stable");
});

test("tracking quality falls back to absolute limits and flags large deviations", () => {
  assert.equal(resolveTrackingQuality(diagnostics(0.05)).band, "stable");
  assert.equal(resolveTrackingQuality(diagnostics(0.08)).band, "review");
  assert.equal(resolveTrackingQuality(diagnostics(0.12)).band, "high_error");
  const broken = { band: "stable", stable_max_m: 0.2, review_max_m: 0.1 };
  assert.equal(resolveTrackingQuality(diagnostics(0.12, broken)).band, "high_error");
  assert.equal(
    resolveTrackingQuality({ schema_version: 1, available: false }).band,
    "unavailable",
  );
});
