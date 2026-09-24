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

const { loadScaledPreview, retarget: runH2rRetarget, retargetExportUrl } =
  await import("../src/features/h2r/api.ts");
const {
  getR2rLibrary,
  r2rEntriesForSourceRobot,
  r2rExportUrl,
  runR2rRetarget,
} =
  await import("../src/features/r2r/api.ts");
const { deleteRobot, uploadRobot } = await import("../src/features/robot/api.ts");
const {
  analyzeDataset,
  computeDatasetSubset,
  getCachedDatasetResult,
  previewDatasetRobot,
  removeDatasetUploadFolder,
  uploadDataset,
} = await import("../src/features/analysis/api.ts");
const { getHumanMotionLibrary, uploadMotion } =
  await import("../src/features/motion/api.ts");

test("Motion import uploads a GVHMR result with the mimic profile", async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const file = new File(["result"], "walk.pt", {
    type: "application/octet-stream",
  });
  const result = await uploadMotion([file], {
    profile: "mimic",
    pollIntervalMs: 0,
    fetcher: async (input, init) => {
      calls.push({ url: String(input), init });
      if (calls.length === 1) return Response.json({ job_id: "motion-import" });
      return Response.json({
        id: "motion-import",
        kind: "motion_link",
        status: "done",
        progress: 1,
        result: {
          name: "walk",
          token: "motion-token",
          positions: [[[0, 0, 0]]],
          parent_indices: [-1],
          body_mesh: { available: true },
          library_entry: { source_path: "/library/walk.pt" },
        },
      });
    },
  });

  assert.equal(calls[0].url, "/api/motion/upload?profile=mimic");
  assert.equal((calls[0].init?.body as FormData).get("files") instanceof File, true);
  assert.equal(calls[1].url, "/api/job/motion-import");
  assert.equal(result.token, "motion-token");
  assert.equal(result.body_mesh?.available, true);
  assert.equal(result.library_entry?.source_path, "/library/walk.pt");
});

test("H2R preserves the selected backend and polls its retarget job", async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const result = await runH2rRetarget(
    {
      robot: "g1_29dof",
      motion_token: "motion-token",
      reference: "smpl",
      backend: "newton",
      retarget_fps: 60,
    },
    {
      fetcher: async (input, init) => {
        calls.push({ url: String(input), init });
        if (calls.length === 1) return Response.json({ job_id: "h2r-job" });
        return Response.json({
          id: "h2r-job",
          kind: "retarget",
          status: "done",
          progress: 1,
          result: {
            trajectory: { frames: [] },
            export_token: "h2r-export",
            num_frames: 12,
          },
        });
      },
    },
  );

  assert.deepEqual(
    calls.map((call) => call.url),
    ["/api/retarget", "/api/job/h2r-job"],
  );
  assert.deepEqual(JSON.parse(String(calls[0].init?.body)), {
    robot: "g1_29dof",
    motion_token: "motion-token",
    reference: "smpl",
    backend: "newton",
    retarget_fps: 60,
    foot_clamp_anti_penetration: false,
  });
  assert.equal(result.export_token, "h2r-export");
});

test("H2R requests its calibrated pre-retarget preview through the typed route", async () => {
  let requestUrl = "";
  let requestInit: RequestInit | undefined;
  const result = await loadScaledPreview(
    {
      robot: "g1_29dof",
      motion_token: "motion-token",
      reference: "smpl",
    },
    {
      fetcher: async (input, init) => {
        requestUrl = String(input);
        requestInit = init;
        return Response.json({
          preview: { positions: [[[0, 0, 0]]], parent_indices: [-1] },
          scaled_scene: null,
        });
      },
    },
  );

  assert.equal(requestUrl, "/api/scaled_preview");
  assert.equal(requestInit?.method, "POST");
  assert.deepEqual(JSON.parse(String(requestInit?.body)), {
    robot: "g1_29dof",
    motion_token: "motion-token",
    reference: "smpl",
  });
  assert.equal(result.preview.positions.length, 1);
  assert.equal(result.scaled_scene, null);
});

test("R2R library accepts only entries explicitly marked robot_trajectory", async () => {
  const entries = await getR2rLibrary({
    fetcher: async () =>
      Response.json({
        source_root: "/source",
        motions_library_root: "/library",
        folders: [],
        entries: [
          { source_path: "/robot.csv", asset_kind: "robot_trajectory" },
          { source_path: "/human.bvh", asset_kind: "human_motion" },
          { source_path: "/unknown.npz" },
        ],
      }),
  });

  assert.deepEqual(
    entries.map((entry) => entry.source_path),
    ["/robot.csv"],
  );
});

test("human and robot workflows project separate logical libraries", async () => {
  const fetcher = async () =>
    Response.json({
      source_root: "/source",
      motions_library_root: "/library",
      folders: [],
      entries: [
        { source_path: "/human.bvh", asset_kind: "human_motion" },
        {
          source_path: "/g1.csv",
          asset_kind: "robot_trajectory",
          source_robot: "g1_29dof",
        },
        {
          source_path: "/x2.csv",
          asset_kind: "robot_trajectory",
          source_robot: "agibot_x2_ultra",
        },
        { source_path: "/foreign.csv", asset_kind: "robot_trajectory" },
      ],
    });

  const human = await getHumanMotionLibrary({ fetcher });
  const robot = await getR2rLibrary({ fetcher });

  assert.deepEqual(
    human.entries.map((entry) => entry.source_path),
    ["/human.bvh"],
  );
  assert.deepEqual(
    robot.map((entry) => entry.source_path),
    ["/g1.csv", "/x2.csv", "/foreign.csv"],
  );
  assert.deepEqual(
    r2rEntriesForSourceRobot(robot, "g1_29dof").map(
      (entry) => entry.source_path,
    ),
    ["/g1.csv", "/foreign.csv"],
  );
});

test("R2R sends the retarget contract and enforces the r2r job kind", async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const fetcher = async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    if (calls.length === 1) return Response.json({ job_id: "r2r-job" });
    return Response.json({
      id: "r2r-job",
      kind: "r2r_retarget",
      status: "done",
      result: {
        trajectory: { frames: [] },
        export_token: "r2r-export",
        stem: "walk",
        num_frames: 8,
        source_fps: 24,
      },
    });
  };

  await runR2rRetarget(
    {
      target: "g1_29dof",
      source: "roboto_origin",
      sourceToken: "source-token",
      backend: "interaction_mesh",
      retargetFps: 24,
    },
    { fetcher, pollIntervalMs: 0 },
  );
  assert.deepEqual(
    calls.map((call) => call.url),
    ["/api/r2r/retarget", "/api/job/r2r-job"],
  );
  assert.deepEqual(JSON.parse(String(calls[0].init?.body)), {
    target: "g1_29dof",
    source: "roboto_origin",
    source_token: "source-token",
    backend: "interaction_mesh",
    retarget_fps: 24,
  });

  await assert.rejects(
    runR2rRetarget(
      {
        target: "g1_29dof",
        source: "roboto_origin",
        sourceToken: "source-token",
        backend: "newton",
      },
      {
        pollIntervalMs: 0,
        fetcher: async (input) =>
          String(input) === "/api/r2r/retarget"
            ? Response.json({ job_id: "wrong-kind" })
            : Response.json({
                id: "wrong-kind",
                kind: "retarget",
                status: "done",
                result: {},
              }),
      },
    ),
    /expected r2r_retarget/,
  );
});

test("workflow export URLs encode tokens and requested options", () => {
  const h2r = new URL(
    retargetExportUrl("h2r token/1", {
      format: "pkl",
      fps: 60,
      csvHeader: false,
      start: 1.25,
      end: 2.5,
    }),
    "http://localhost",
  );
  assert.equal(h2r.pathname, "/api/export/h2r%20token%2F1");
  assert.deepEqual(Object.fromEntries(h2r.searchParams), {
    fmt: "pkl",
    fps: "60",
    csv_header: "0",
    t_start: "1.25",
    t_end: "2.5",
  });

  const r2r = new URL(
    r2rExportUrl("r2r token", {
      format: "csv",
      fps: 30,
      csvHeader: false,
      start: 0.5,
      end: 4,
    }),
    "http://localhost",
  );
  assert.equal(r2r.pathname, "/api/export/r2r%20token");
  assert.deepEqual(Object.fromEntries(r2r.searchParams), {
    fmt: "csv",
    fps: "30",
    csv_header: "false",
    t_start: "0.5",
    t_end: "4",
  });
});

test("analysis starts the dataset job and enforces its job kind", async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const result = await analyzeDataset(
    { source: "/data/motions", embedding: "handcrafted", force: true },
    {
      pollIntervalMs: 0,
      fetcher: async (input, init) => {
        calls.push({ url: String(input), init });
        if (calls.length === 1) return Response.json({ job_id: "analysis-job" });
        return Response.json({
          id: "analysis-job",
          kind: "dataset_analyze",
          status: "done",
          result: {
            meta: { source_root: "/data/motions", embedding: "handcrafted" },
            clips: [],
            summary: { num_clips: 0, num_ok: 0, num_error: 0 },
          },
        });
      },
    },
  );
  assert.deepEqual(calls.map((call) => call.url), [
    "/api/dataset/analyze",
    "/api/job/analysis-job",
  ]);
  assert.deepEqual(JSON.parse(String(calls[0].init?.body)), {
    source: "/data/motions",
    embedding: "handcrafted",
    force: true,
  });
  assert.equal(result.meta.source_root, "/data/motions");
});

test("analysis existing result reads cache without starting work", async () => {
  let requestedUrl = "";
  const result = await getCachedDatasetResult(
    "/data/my motions",
    "handcrafted",
    {
      fetcher: async (input) => {
        requestedUrl = String(input);
        return Response.json({ available: false });
      },
    },
  );
  assert.equal(
    requestedUrl,
    "/api/dataset/result?embedding=handcrafted&source=%2Fdata%2Fmy+motions",
  );
  assert.equal(result.available, false);
});

test("analysis robot preview uses its dedicated job contract", async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const result = await previewDatasetRobot(
    { source_path: "/data/robot/walk.csv", robot: "g1_29dof" },
    {
      pollIntervalMs: 0,
      fetcher: async (input, init) => {
        calls.push({ url: String(input), init });
        if (calls.length === 1) return Response.json({ job_id: "preview-job" });
        return Response.json({
          id: "preview-job",
          kind: "dataset_robot_preview",
          status: "done",
          result: {
            preview_token: "scene-token",
            trajectory: { frames: [{ links: {} }] },
            robot: "g1_29dof",
            inferred_robot: "g1_29dof",
            num_frames: 1,
            framerate: 30,
            has_scene: false,
            scaled_scene: null,
            name: "walk",
          },
        });
      },
    },
  );

  assert.deepEqual(calls.map((call) => call.url), [
    "/api/dataset/preview_robot",
    "/api/job/preview-job",
  ]);
  assert.deepEqual(JSON.parse(String(calls[0].init?.body)), {
    source_path: "/data/robot/walk.csv",
    robot: "g1_29dof",
  });
  assert.equal(result.preview_token, "scene-token");
});

test("analysis upload preserves append query and folder-relative filenames", async () => {
  let requestUrl = "";
  let form: FormData | null = null;
  const file = new File(["motion"], "walk.bvh", { type: "application/octet-stream" }) as File & {
    _relpath?: string;
  };
  file._relpath = "LAFAN/walk.bvh";
  await uploadDataset([file], {
    appendTo: "/tmp/hhtools/dataset/abc",
    userSourceRoot: "/home/nora/data",
    fetcher: async (input, init) => {
      requestUrl = String(input);
      form = init?.body as FormData;
      return Response.json({
        source: "/tmp/hhtools/dataset/abc",
        clip_count: 1,
        human_count: 1,
        robot_count: 0,
        folders: { LAFAN: 1 },
        clips: [],
      });
    },
  });
  assert.match(requestUrl, /append_to=%2Ftmp%2Fhhtools%2Fdataset%2Fabc/);
  assert.match(requestUrl, /user_source_root=%2Fhome%2Fnora%2Fdata/);
  assert.equal(form?.get("files") instanceof File, true);
  assert.equal((form?.get("files") as File).name, "LAFAN/walk.bvh");
});

test("analysis upload basket removes one folder through the typed route", async () => {
  let request: { url: string; init?: RequestInit } | undefined;
  const result = await removeDatasetUploadFolder(
    "/tmp/hhtools/dataset/abc",
    "LAFAN/walks",
    {
      fetcher: async (input, init) => {
        request = { url: String(input), init };
        return Response.json({
          source: "/tmp/hhtools/dataset/abc",
          clip_count: 2,
          human_count: 2,
          robot_count: 0,
          folders: { CMU: 2 },
          clips: [],
        });
      },
    },
  );

  assert.equal(request?.url, "/api/dataset/upload/remove");
  assert.equal(request?.init?.method, "POST");
  assert.deepEqual(JSON.parse(String(request?.init?.body)), {
    source: "/tmp/hhtools/dataset/abc",
    folder_label: "LAFAN/walks",
  });
  assert.deepEqual(result.folders, { CMU: 2 });
});

test("analysis subset sends the analyzed clips and selection parameters", async () => {
  const clip = {
    clip_id: "walk",
    source_kind: "human",
    source_path: "/data/walk.bvh",
    dataset: "lafan",
    folder_label: "LAFAN",
    metrics: { complexity: 1 },
    tags: [],
    embedding: [0.1],
    scatter: null,
    cluster_id: null,
    error: null,
  };
  let body: unknown;
  const value = await computeDatasetSubset([clip], 1, 0.75, {
    fetcher: async (_input, init) => {
      body = JSON.parse(String(init?.body));
      return Response.json({ selected: ["walk"], count: 1 });
    },
  });
  assert.deepEqual(body, { clips: [clip], k: 1, alpha: 0.75 });
  assert.deepEqual(value, { selected: ["walk"], count: 1 });
});

test("robot import preserves bundle paths and deletion encodes its name", async () => {
  const file = new File(["solid robot"], "arm.stl") as File & { _relpath?: string };
  file._relpath = "meshes/arm.stl";
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const fetcher = async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    if (init?.method === "DELETE") return Response.json({ ok: true, deleted: "my robot" });
    return Response.json({ name: "my_robot", display_name: "My Robot", links: [], link_transforms_zero: {} });
  };
  await uploadRobot([file], "my_robot", { fetcher });
  await deleteRobot("my robot", { fetcher });
  assert.equal(calls[0].url, "/api/robot/upload?name=my_robot");
  assert.equal(((calls[0].init?.body as FormData).get("files") as File).name, "meshes/arm.stl");
  assert.equal(calls[1].url, "/api/robot/my%20robot");
  assert.equal(calls[1].init?.method, "DELETE");
});
