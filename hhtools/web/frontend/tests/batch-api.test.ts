import assert from "node:assert/strict";
import { registerHooks } from "node:module";
import test from "node:test";

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (!specifier.startsWith("@/")) return nextResolve(specifier, context);
    const extension = specifier.endsWith(".ts") ? "" : ".ts";
    const url = new URL(`../src/${specifier.slice(2)}${extension}`, import.meta.url);
    return nextResolve(url.href, context);
  },
});

const {
  batchDownloadUrl,
  runHumanBatch,
  runR2rBatch,
  scanHumanBatchInputs,
  uploadR2rBatchInputs,
} = await import("../src/features/batch/api.ts");
const {
  appendUniqueEntries,
  entryReference,
  optionalNonNegativeNumber,
  optionalPositiveNumber,
  publishedMotionEntry,
  suggestedBackend,
  timeRangeError,
  uploadFileKey,
} = await import("../src/features/batch/model.ts");

test("Batch draft helpers preserve intrinsic references and deduplicate inputs", () => {
  const existing = [{ source_path: "/motions/walk.npz", dataset: "amass" }];
  const incoming = [
    { source_path: "/motions/walk.npz", reference: "smplx" },
    {
      source_path: "/motions/reach.npy",
      dataset: "motion_x",
      suggested_backend: "interaction_mesh",
    },
  ];
  const combined = appendUniqueEntries(existing, incoming);
  assert.equal(combined.length, 2);
  assert.equal(entryReference(existing[0]), "smpl");
  assert.equal(entryReference(incoming[1]), "smplx");
  assert.equal(suggestedBackend(incoming), "interaction_mesh");
});

test("published V2M motions retain their complete Library row for H2R", () => {
  const entry = {
    source_path: "/motions/generated.pt",
    dataset: "gvhmr",
    reference: "gvhmr",
    token: "generated-token",
    suggested_backend: "newton",
  };
  assert.equal(publishedMotionEntry(entry), entry);
  assert.equal(publishedMotionEntry({ dataset: "gvhmr" }), null);
  assert.equal(publishedMotionEntry({ source_path: "  " }), null);
  assert.deepEqual(appendUniqueEntries([entry], [entry]), [entry]);
});

test("Batch settings helpers validate optional FPS and time ranges", () => {
  assert.equal(optionalPositiveNumber(" 60 "), 60);
  assert.equal(optionalPositiveNumber("0"), undefined);
  assert.equal(optionalNonNegativeNumber("0"), 0);
  assert.equal(optionalNonNegativeNumber("-1"), undefined);
  assert.equal(timeRangeError("1", "2"), null);
  assert.match(timeRangeError("2", "1") ?? "", /cannot be later/);
  assert.match(timeRangeError("bad", "") ?? "", /non-negative/);
});

test("Video queue identity includes relative path and file metadata", () => {
  const file = new File(["video"], "turn.mp4", { lastModified: 42 });
  Object.assign(file, { _relpath: "session/turn.mp4" });
  assert.equal(uploadFileKey(file), "session/turn.mp4:5:42");
});

test("H2R Batch sends entries and returns its download identity", async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const completed = await runHumanBatch(
    {
      robot: "g1_29dof",
      entries: [{ source_path: "/motions/walk.bvh", reference: "smpl" }],
      reference: "smpl",
      backend: "newton",
      out_dir: "batch_export",
      format: "csv",
      csv_header: true,
      foot_clamp_anti_penetration: false,
      batch_size: 8,
    },
    {
      pollIntervalMs: 0,
      fetcher: async (input, init) => {
        calls.push({ url: String(input), init });
        if (calls.length === 1) return Response.json({ job_id: "batch-job" });
        return Response.json({
          id: "batch-job",
          kind: "batch",
          status: "done",
          result: {
            written: ["walk.csv"],
            errors: [],
            failures: [],
            format: "csv",
            download_name: "batch_export.zip",
            artifact_path: "/tmp/batch_export.zip",
          },
        });
      },
    },
  );
  assert.equal(completed.jobId, "batch-job");
  assert.equal(completed.result.written.length, 1);
  assert.equal(batchDownloadUrl(completed.jobId), "/api/job/batch-job/download");
  assert.deepEqual(calls.map((call) => call.url), [
    "/api/batch/retarget",
    "/api/job/batch-job",
  ]);
  assert.equal(JSON.parse(String(calls[0].init?.body)).batch_size, 8);
});

test("R2R Batch enforces its dedicated job kind", async () => {
  const completed = await runR2rBatch(
    {
      source: "roboto_origin",
      target: "g1_29dof",
      entries: [{ source_path: "/robot/walk.csv" }],
      backend: "interaction_mesh",
      out_dir: "r2r_batch_export",
      format: "pkl",
      csv_header: false,
    },
    {
      pollIntervalMs: 0,
      fetcher: async (input) =>
        String(input) === "/api/r2r/batch/retarget"
          ? Response.json({ job_id: "r2r-batch-job" })
          : Response.json({
              id: "r2r-batch-job",
              kind: "r2r_batch",
              status: "done",
              result: {
                written: [],
                errors: ["walk failed"],
                failures: [{ stem: "walk", stage: "retarget", reason: "failed" }],
                format: "pkl",
                download_name: "r2r_batch_export.zip",
                artifact_path: "/tmp/r2r_batch_export.zip",
              },
            }),
    },
  );
  assert.equal(completed.result.failures[0]?.stem, "walk");
});

test("Batch scan and upload keep their separate input contracts", async () => {
  let scanBody: unknown;
  const scanned = await scanHumanBatchInputs("/data/motions", "auto", {
    fetcher: async (_input, init) => {
      scanBody = JSON.parse(String(init?.body));
      return Response.json({ entries: [], clip_count: 0, source: "/data/motions" });
    },
  });
  assert.deepEqual(scanBody, { source: "/data/motions", profile: "auto" });
  assert.equal(scanned.source, "/data/motions");

  const uploaded = await uploadR2rBatchInputs(
    [new File(["q"], "walk.csv")],
    "mimic",
    {
      pollIntervalMs: 0,
      fetcher: async (input) =>
        String(input).startsWith("/api/r2r/basket/upload")
          ? Response.json({ job_id: "upload-job" })
          : Response.json({
              id: "upload-job",
              kind: "r2r_basket_upload",
              status: "done",
              result: { entries: [{ source_path: "/tmp/walk.csv" }], clip_count: 1 },
            }),
    },
  );
  assert.equal(uploaded.entries[0]?.source_path, "/tmp/walk.csv");
});
