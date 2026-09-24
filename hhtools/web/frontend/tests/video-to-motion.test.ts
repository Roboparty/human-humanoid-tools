import assert from "node:assert/strict";
import { registerHooks } from "node:module";
import test from "node:test";

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (!specifier.startsWith("@/")) return nextResolve(specifier, context);
    const url = new URL(`../src/${specifier.slice(2)}.ts`, import.meta.url);
    return nextResolve(url.href, context);
  },
});

const {
  boundedProgress,
  canSetupGvhmrInDesktop,
  getGvhmrRuntimeStatus,
  invalidateGvhmrRuntimeStatus,
  isSmplxNeutralMissing,
  isGvhmrResultName,
  isSupportedVideoName,
  parseOptionalFocalLength,
  SMPLX_DOWNLOAD_URL,
  setupGvhmrInDesktop,
  startVideoToMotion,
  summarizeMotionResult,
  toStageMotionPayload,
  visibleGvhmrFailure,
  visibleGvhmrMissing,
  waitForVideoToMotion,
} = await import("../src/features/video-to-motion/api.ts");

test("uses the desktop setup bridge only when Electron exposes it", async () => {
  const webHost = {};
  const desktopHost = {
    hhtoolsDesktop: {
      setupGvhmr: async () => ({ action: "configured" as const }),
    },
  };
  assert.equal(canSetupGvhmrInDesktop(webHost), false);
  assert.equal(canSetupGvhmrInDesktop(desktopHost), true);
  assert.deepEqual(await setupGvhmrInDesktop(desktopHost), {
    action: "configured",
  });
});

test("validates video names and optional focal length", () => {
  assert.equal(isSupportedVideoName("walk.MP4"), true);
  assert.equal(isSupportedVideoName("walk.txt"), false);
  assert.equal(isGvhmrResultName("walk.PT"), true);
  assert.equal(isGvhmrResultName("walk.pt.json"), false);
  assert.equal(parseOptionalFocalLength(""), undefined);
  assert.equal(parseOptionalFocalLength(" 35 "), 35);
  assert.throws(() => parseOptionalFocalLength("1.5"), /positive integer/);
  assert.throws(() => parseOptionalFocalLength("0"), /positive integer/);
});

test("normalizes runtime status responses", async () => {
  const status = await getGvhmrRuntimeStatus(
    new AbortController().signal,
    async () =>
      Response.json({
        ready: false,
        checks: { smplx_neutral: false },
        missing: ["valid", 42],
        body_models_root: "/models",
      }),
  );
  assert.deepEqual(status.missing, []);
  assert.equal(status.ready, false);
  assert.equal(status.checks?.smplx_neutral, false);
  assert.equal(status.body_models_root, "/models");
});

test("coalesces and briefly caches the default GVHMR status probe", async () => {
  const originalFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return Response.json({ ready: true, missing: [] });
  };
  invalidateGvhmrRuntimeStatus();
  try {
    const first = getGvhmrRuntimeStatus(new AbortController().signal);
    const second = getGvhmrRuntimeStatus(new AbortController().signal);
    await Promise.all([first, second]);
    await getGvhmrRuntimeStatus(new AbortController().signal);
    assert.equal(calls, 1);

    invalidateGvhmrRuntimeStatus();
    await getGvhmrRuntimeStatus(new AbortController().signal);
    assert.equal(calls, 2);
  } finally {
    invalidateGvhmrRuntimeStatus();
    globalThis.fetch = originalFetch;
  }
});

test("links the SMPL-X download only to structured model absence", () => {
  assert.equal(SMPLX_DOWNLOAD_URL, "https://smpl-x.is.tue.mpg.de/download.php");
  assert.equal(
    isSmplxNeutralMissing({
      ready: false,
      checks: { smplx_neutral: false },
      missing: ["localized or server-defined message"],
      body_models_root: "/models",
    }),
    true,
  );
  assert.equal(
    isSmplxNeutralMissing({
      ready: false,
      checks: { smplx_neutral: true },
      missing: ["licensed SMPL-X neutral model"],
    }),
    false,
  );
  assert.equal(
    isSmplxNeutralMissing({
      ready: false,
      missing: ["licensed SMPL-X neutral model"],
    }),
    false,
  );
});

test("replaces only the path-heavy SMPL-X missing detail", () => {
  assert.deepEqual(
    visibleGvhmrMissing({
      ready: false,
      checks: { smplx_neutral: false, cuda: false },
      missing: [
        "licensed SMPL-X neutral model: /private/models/SMPLX_NEUTRAL.npz",
        "CUDA is not available",
      ],
    }),
    ["CUDA is unavailable."],
  );
  assert.deepEqual(
    visibleGvhmrMissing({
      ready: false,
      checks: { smplx_neutral: true },
      missing: ["another runtime problem"],
    }),
    ["another runtime problem"],
  );
});

test("keeps GVHMR tracebacks in Tasks and returns a concise workflow error", () => {
  const translate = (english: string) => english;
  const visible = visibleGvhmrFailure(
    new Error(
      "GVHMR local runtime exited with code 1.\nTraceback (most recent call last):\n" +
        "ModuleNotFoundError: No module named 'hydra'",
    ),
    translate,
  );

  assert.match(visible, /missing “hydra”/);
  assert.match(visible, /Technical details remain in Tasks/);
  assert.doesNotMatch(visible, /Traceback|ModuleNotFoundError/);
  assert.equal(
    visibleGvhmrFailure(new Error("Upload rejected."), translate),
    "Upload rejected.",
  );
});

test("starts the official-weight upload contract", async () => {
  let requestedUrl = "";
  let requestedBody: FormData | null = null;
  const video = new File(["video"], "turn.mov", { type: "video/quicktime" });
  const jobId = await startVideoToMotion(
    { video, staticCamera: false, focalLength: 50 },
    new AbortController().signal,
    async (input, init) => {
      requestedUrl = String(input);
      requestedBody = init?.body as FormData;
      return Response.json({ job_id: "job-123" });
    },
  );
  assert.equal(jobId, "job-123");
  assert.equal(requestedUrl, "/api/video-to-motion/upload?static_cam=false&f_mm=50");
  assert.deepEqual([...requestedBody!.keys()], ["files"]);
  assert.equal(requestedBody!.has("checkpoint"), false);
});

test("polls progress and returns the completed motion", async () => {
  const updates: number[] = [];
  const responses = [
    { id: "job-1", kind: "video_to_motion", status: "running", progress: 1.4 },
    {
      id: "job-1",
      kind: "video_to_motion",
      status: "done",
      progress: 1,
      result: { name: "motion", token: "motion-token" },
    },
  ];
  const result = await waitForVideoToMotion(
    "job-1",
    {
      signal: new AbortController().signal,
      pollIntervalMs: 0,
      onUpdate: (job) => updates.push(job.progress),
    },
    async () => Response.json(responses.shift()),
  );
  assert.deepEqual(updates, [1]);
  assert.equal(result.token, "motion-token");
});

test("summarizes total frames without retaining frame arrays", () => {
  assert.equal(boundedProgress(-1), 0);
  assert.deepEqual(
    summarizeMotionResult(
      {
        positions: [[], [], []],
        playback_frames: 3,
        num_frames_total: 3_000,
        duration: 100,
        sample_rate: 30,
        linked_folder: "gvhmr-turn",
      },
      "turn.mov",
    ),
    {
      name: "turn.mov",
      token: null,
      frames: 3_000,
      duration: 100,
      framerate: 30,
      linkedFolder: "gvhmr-turn",
    },
  );
});

test("projects a completed motion payload for the Stage", () => {
  const result = {
    positions: [[[0, 0, 0]]],
    parent_indices: [-1],
    token: "motion-token",
    name: "generated-motion",
    terrain: { vertices: [[0, 0, 0]], faces: [[0, 0, 0]] },
    body_mesh: { available: false, reason: "weights unavailable" },
    library_entry: { source_path: "/library/generated.pt" },
  } satisfies MotionResult;
  const payload = toStageMotionPayload(result);
  assert.strictEqual(payload, result);
  assert.equal(payload?.token, "motion-token");
  assert.equal(payload?.library_entry?.source_path, "/library/generated.pt");
  assert.ok(payload?.terrain);
  assert.equal(toStageMotionPayload({ name: "incomplete" }), null);
});
