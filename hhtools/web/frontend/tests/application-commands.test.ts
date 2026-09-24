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
  PROJECT_README_URL,
  createApplicationMenus,
  storeTheme,
  storedTheme,
  storedThemeOverride,
  viewForImport,
} = await import("../src/appCommands.ts");
const { getJobAdmissionSettings, updateJobAdmissionSettings } = await import(
  "../src/features/settings/api.ts"
);
const { proposeCalibration } = await import("../src/features/h2r/api.ts");

test("application menu descriptors retain the five-menu contract", () => {
  const navigation: string[] = [];
  const imports: string[] = [];
  let toggledTheme = false;
  const menus = createApplicationMenus({
    locale: "en",
    theme: "light",
    canExportResult: false,
    canExitApplication: false,
    onNavigate: (view) => navigation.push(view),
    onImport: (target) => imports.push(target),
    onExportResult: () => undefined,
    onOpenSettings: () => undefined,
    onToggleTheme: () => {
      toggledTheme = true;
    },
    onOpenTutorial: () => undefined,
    onOpenAbout: () => undefined,
    onExitApplication: () => undefined,
  });

  assert.deepEqual(
    menus.map((menu) => menu.label),
    ["File", "Workflows", "Analysis", "Settings", "Help"],
  );
  const commands = menus.flatMap((menu) => menu.commands);
  commands.find((command) => command.id === "navigate-r2r")?.run();
  commands.find((command) => command.id === "import-motion-file")?.run();
  commands.find((command) => command.id === "toggle-theme")?.run();
  assert.deepEqual(navigation, ["r2r"]);
  assert.deepEqual(imports, ["motion-file"]);
  assert.equal(toggledTheme, true);
  assert.equal(
    commands.find((command) => command.id === "export-current-result")?.enabled,
    false,
  );
  assert.equal(commands.some((command) => command.id === "exit-application"), false);
  assert.equal(
    commands.find((command) => command.id === "toggle-theme")?.label,
    "Dark Mode",
  );
  assert.equal(commands.some((command) => "shortcut" in command), false);
  assert.deepEqual(
    menus.find((menu) => menu.id === "settings")?.commands.map(({ id }) => id),
    ["open-settings", "toggle-theme"],
  );
});

test("desktop application menu retains an executable Exit command", () => {
  let exitCount = 0;
  const menus = createApplicationMenus({
    locale: "en",
    theme: "light",
    canExportResult: false,
    canExitApplication: true,
    onNavigate: () => undefined,
    onImport: () => undefined,
    onExportResult: () => undefined,
    onOpenSettings: () => undefined,
    onToggleTheme: () => undefined,
    onOpenTutorial: () => undefined,
    onOpenAbout: () => undefined,
    onExitApplication: () => {
      exitCount += 1;
    },
  });

  const exitCommand = menus
    .find((menu) => menu.id === "file")
    ?.commands.find((command) => command.id === "exit-application");
  assert.equal(exitCommand?.dividerBefore, true);
  assert.notEqual(exitCommand?.enabled, false);
  exitCommand?.run();
  assert.equal(exitCount, 1);
});

test("application menus localize without changing command identity", () => {
  const menus = createApplicationMenus({
    locale: "zh-CN",
    theme: "dark",
    canExportResult: false,
    canExitApplication: false,
    onNavigate: () => undefined,
    onImport: () => undefined,
    onExportResult: () => undefined,
    onOpenSettings: () => undefined,
    onToggleTheme: () => undefined,
    onOpenTutorial: () => undefined,
    onOpenAbout: () => undefined,
    onExitApplication: () => undefined,
  });
  assert.deepEqual(
    menus.map((menu) => menu.label),
    ["文件", "工作流", "分析", "设置", "帮助"],
  );
  assert.deepEqual(
    menus.find((menu) => menu.id === "settings")?.commands.map((command) => [
      command.id,
      command.label,
    ]),
    [
      ["open-settings", "设置"],
      ["toggle-theme", "浅色模式"],
    ],
  );
});

test("import intents select the owning persistent workspace", () => {
  assert.equal(viewForImport("motion-folder"), "motion");
  assert.equal(viewForImport("robot-mesh-folder"), "robot-assets");
  assert.equal(viewForImport("video-file"), "video-to-motion");
});

test("theme follows the system until a valid manual preference is stored", () => {
  assert.equal(storedThemeOverride({ getItem: () => "dark" }), "dark");
  assert.equal(storedThemeOverride({ getItem: () => "unknown" }), null);
  assert.equal(storedTheme({ getItem: () => "dark" }, "light"), "dark");
  assert.equal(storedTheme({ getItem: () => "light" }, "dark"), "light");
  assert.equal(storedTheme({ getItem: () => null }, "dark"), "dark");
  assert.equal(storedTheme({ getItem: () => "unknown" }, "dark"), "dark");
  assert.equal(
    storedTheme(
      {
        getItem: () => {
          throw new Error("storage unavailable");
        },
      },
      "dark",
    ),
    "dark",
  );

  let stored = "";
  storeTheme(
    {
      setItem: (_key, value) => {
        stored = value;
      },
    },
    "light",
  );
  assert.equal(stored, "light");
  assert.equal(
    PROJECT_README_URL,
    "https://github.com/Eleanor1018/human-humanoid-tools#readme",
  );
});

test("job-admission settings use the typed GET and PATCH contracts", async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const fetcher = async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    return Response.json({
      mode: "queued",
      max_running_jobs: 2,
      max_queued_jobs: 16,
      max_batch_items: 0,
      max_batch_total_frames: 0,
      running_jobs: 1,
      queued_jobs: 3,
      reserved_jobs: 0,
      cancelling_jobs: 0,
      closed: false,
      editable: true,
    });
  };

  const before = await getJobAdmissionSettings({ fetcher });
  const after = await updateJobAdmissionSettings(
    {
      max_running_jobs: 2,
      max_queued_jobs: 16,
      max_batch_items: 0,
      max_batch_total_frames: 0,
    },
    { fetcher },
  );

  assert.equal(before.running_jobs, 1);
  assert.equal(after.max_queued_jobs, 16);
  assert.equal(calls[0].url, "/api/settings/job-admission");
  assert.equal(calls[0].init?.method, undefined);
  assert.equal(calls[1].url, "/api/settings/job-admission");
  assert.equal(calls[1].init?.method, "PATCH");
  assert.deepEqual(JSON.parse(String(calls[1].init?.body)), {
    max_running_jobs: 2,
    max_queued_jobs: 16,
    max_batch_items: 0,
    max_batch_total_frames: 0,
  });
});

test("calibration proposal keeps the current editable pose as its seed", async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const fetcher = async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    return Response.json({
      joint_q: {
        left_shoulder_roll_joint: 1.2,
        right_shoulder_roll_joint: -1.2,
      },
      validation: {
        valid: true,
        score: 0.94,
        changed_joint_count: 2,
        edge_errors_deg: { left_upper_arm: 5.0 },
        near_limit_joints: [],
        alignment_errors: [],
        alignment_warnings: [],
        foot_height_delta_m: 0,
      },
    });
  };

  const result = await proposeCalibration(
    {
      robot: "agibot_x2_ultra",
      reference: "smplx",
      joint_q: { left_shoulder_roll_joint: 0.3 },
      motion_token: "motion-token",
    },
    { fetcher },
  );

  assert.equal(result.validation.valid, true);
  assert.equal(result.joint_q.left_shoulder_roll_joint, 1.2);
  assert.equal(calls[0].url, "/api/calibration/propose");
  assert.deepEqual(JSON.parse(String(calls[0].init?.body)), {
    robot: "agibot_x2_ultra",
    reference: "smplx",
    joint_q: { left_shoulder_roll_joint: 0.3 },
    motion_token: "motion-token",
  });
});
