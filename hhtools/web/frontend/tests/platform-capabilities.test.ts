import assert from "node:assert/strict";
import { registerHooks } from "node:module";
import test from "node:test";

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (!specifier.startsWith("@/")) return nextResolve(specifier, context);
    return nextResolve(new URL(`../src/${specifier.slice(2)}.ts`, import.meta.url).href, context);
  },
});

const { isImportAvailable, isVideoToMotionEnabled, isViewAvailable } =
  await import("../src/platformCapabilities.ts");
const { createApplicationMenus } = await import("../src/appCommands.ts");
const { getNavigationGroups } = await import("../src/navigation.ts");
const { getBatchModes } = await import("../src/features/batch/modes.ts");
const { getTutorialSteps } = await import("../src/features/tutorial/model.ts");
const { getVideoToMotionSettings } = await import("../src/features/settings/api.ts");

function menus(videoToMotionEnabled?: boolean) {
  return createApplicationMenus({
    locale: "en",
    theme: "light",
    canExportResult: true,
    canExitApplication: true,
    videoToMotionEnabled,
    onNavigate: () => undefined,
    onImport: () => undefined,
    onExportResult: () => undefined,
    onOpenSettings: () => undefined,
    onToggleTheme: () => undefined,
    onOpenTutorial: () => undefined,
    onOpenAbout: () => undefined,
    onExitApplication: () => undefined,
  });
}

test("video capability defaults to existing behavior for Web and older desktop bridges", () => {
  for (const host of [undefined, null, {}, { hhtoolsDesktop: {} }, {
    hhtoolsDesktop: { capabilities: {} },
  }, { hhtoolsDesktop: { capabilities: { videoToMotion: true } } }]) {
    assert.equal(isVideoToMotionEnabled(host), true);
  }
  assert.equal(isVideoToMotionEnabled({
    hhtoolsDesktop: { capabilities: { videoToMotion: false } },
  }), false);
});

test("disabled video capability removes entry points while retaining motion, robot and retarget workflows", () => {
  const available = isVideoToMotionEnabled({
    hhtoolsDesktop: { capabilities: { videoToMotion: false } },
  });
  const navigation = getNavigationGroups(available).flatMap(group => group.items.map(item => item.id));
  assert.deepEqual(navigation, ["motion", "robot-assets", "h2r", "r2r", "batch", "dataset-viz"]);
  assert.deepEqual(getBatchModes(available).map(mode => mode.id), ["h2r", "r2r"]);
  const commands = menus(available).flatMap(menu => menu.commands.map(command => command.id));
  assert.equal(commands.includes("import-video"), false);
  assert.equal(commands.includes("navigate-video-to-motion"), false);
  for (const command of [
    "import-motion-file", "import-motion-folder", "import-robot-urdf",
    "import-robot-mesh-folder", "navigate-h2r", "navigate-r2r", "navigate-batch",
    "export-current-result", "exit-application",
  ]) {
    assert.ok(commands.includes(command), `${command} remains available`);
  }
  assert.equal(isViewAvailable("video-to-motion", available), false);
  assert.equal(isImportAvailable("video-file", available), false);
  assert.equal(isViewAvailable("h2r", available), true);
  assert.equal(isViewAvailable("r2r", available), true);
  assert.equal(isImportAvailable("motion-file", available), true);
  assert.equal(isImportAvailable("motion-folder", available), true);
  const tutorial = getTutorialSteps(available);
  assert.equal(tutorial.length, 9);
  assert.ok(tutorial.every(step => isViewAvailable(step.view, available)));
  assert.doesNotMatch(tutorial.at(-1)!.body.en, /Video to Motion/);
  assert.doesNotMatch(tutorial.at(-1)!.body.zh, /视频到动作/);
});

test("enabled and default capabilities retain video navigation, imports, batches and tutorial copy", () => {
  for (const available of [undefined, true]) {
    const commands = menus(available).flatMap(menu => menu.commands.map(command => command.id));
    assert.ok(commands.includes("import-video"));
    assert.ok(commands.includes("navigate-video-to-motion"));
    assert.ok(getNavigationGroups(available).some(group => group.items.some(item => item.id === "video-to-motion")));
    assert.deepEqual(getBatchModes(available).map(mode => mode.id), ["v2m", "h2r", "r2r"]);
    assert.equal(isViewAvailable("video-to-motion", available), true);
    assert.equal(isImportAvailable("video-file", available), true);
    assert.match(getTutorialSteps(available).at(-1)!.body.en, /Video to Motion/);
  }
});

test("default UI capability lookups read the live desktop bridge without a platform-name check", () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, "hhtoolsDesktop");
  Object.defineProperty(globalThis, "hhtoolsDesktop", {
    configurable: true,
    value: { capabilities: { videoToMotion: false } },
  });
  try {
    assert.equal(isVideoToMotionEnabled(), false);
    assert.equal(isViewAvailable("video-to-motion"), false);
    assert.equal(isImportAvailable("video-file"), false);
    assert.ok(getNavigationGroups().every(group => group.items.every(item => item.id !== "video-to-motion")));
    assert.ok(menus().every(menu => menu.commands.every(command => command.id !== "import-video" && command.id !== "navigate-video-to-motion")));
    assert.deepEqual(getBatchModes().map(mode => mode.id), ["h2r", "r2r"]);
    assert.doesNotMatch(getTutorialSteps().at(-1)!.body.en, /Video to Motion/);
  } finally {
    if (previous) Object.defineProperty(globalThis, "hhtoolsDesktop", previous);
    else Reflect.deleteProperty(globalThis, "hhtoolsDesktop");
  }
});

test("disabled video settings issue neither runtime HTTP probes nor optional-component IPC", async () => {
  let requests = 0;
  let componentReads = 0;
  const settings = await getVideoToMotionSettings({
    host: {
      hhtoolsDesktop: {
        capabilities: { videoToMotion: false },
        selectDirectory: async () => "/motions",
        getOptionalComponents: async () => { componentReads += 1; return {}; },
        setupGvhmr: async () => { throw new Error("Must not set up GVHMR"); },
      },
    },
    fetcher: async () => { requests += 1; throw new Error("Must not probe GVHMR"); },
  });
  assert.deepEqual(settings, { runtime: null, component: null });
  assert.equal(requests, 0);
  assert.equal(componentReads, 0);
});

test("Web and enabled desktop settings still load the video runtime", async () => {
  for (const host of [{}, { hhtoolsDesktop: { capabilities: { videoToMotion: true } } }]) {
    const requests: string[] = [];
    const settings = await getVideoToMotionSettings({
      host,
      fetcher: async input => {
        requests.push(String(input));
        return Response.json({ ready: true, missing: [] });
      },
    });
    assert.deepEqual(requests, ["/api/video-to-motion/status"]);
    assert.equal(settings.runtime?.ready, true);
  }
});
