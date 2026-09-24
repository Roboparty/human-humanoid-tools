import assert from "node:assert/strict";
import test from "node:test";

const { buildTrackingChart, topEffectors, validateExportOptions } = await import(
  "../src/features/result/model.ts"
);

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
