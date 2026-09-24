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
  desktopSettingsBridge,
  getGvhmrRuntimeSettings,
  getMotionLibrarySettings,
  updateMotionLibrarySettings,
} = await import("../src/features/settings/api.ts");

test("motion-library settings use the typed GET and PATCH routes", async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const fetcher = async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    return Response.json({
      root: "/srv/motions",
      default_root: "/home/user/Motions",
      editable: true,
    });
  };

  const before = await getMotionLibrarySettings({ fetcher });
  const after = await updateMotionLibrarySettings("/srv/new motions", {
    fetcher,
  });

  assert.equal(before.root, "/srv/motions");
  assert.equal(after.editable, true);
  assert.equal(calls[0].url, "/api/settings/motion-library");
  assert.equal(calls[0].init?.method, undefined);
  assert.equal(calls[1].url, "/api/settings/motion-library");
  assert.equal(calls[1].init?.method, "PATCH");
  assert.deepEqual(JSON.parse(String(calls[1].init?.body)), {
    root: "/srv/new motions",
  });
});

test("GVHMR settings normalize the backend readiness response", async () => {
  const status = await getGvhmrRuntimeSettings({
    fetcher: async () =>
      Response.json({
        ready: 1,
        missing: ["licensed SMPL-X model", 42, null],
        runtime: "local",
      }),
  });

  assert.equal(status.ready, false);
  assert.deepEqual(status.missing, ["licensed SMPL-X model"]);
  assert.equal(status.runtime, "local");
});

test("Workspace Settings accepts only the complete desktop capability bridge", async () => {
  assert.equal(desktopSettingsBridge({}), null);
  assert.equal(
    desktopSettingsBridge({
      hhtoolsDesktop: { selectDirectory: async () => "/tmp" },
    }),
    null,
  );

  const bridge = desktopSettingsBridge({
    hhtoolsDesktop: {
      getOptionalComponents: async () => ({
        gvhmr: {
          requested: false,
          configured: true,
          runtime: "local" as const,
          guideUrl: "https://example.com/gvhmr",
          estimatedAdditionalBytes: 1,
        },
      }),
      setupGvhmr: async () => ({
        action: "configured" as const,
        state: {
          requested: false,
          configured: true,
          runtime: "local" as const,
          guideUrl: "https://example.com/gvhmr",
          estimatedAdditionalBytes: 1,
        },
      }),
      selectDirectory: async () => "/srv/motions",
    },
  });

  assert.ok(bridge);
  assert.equal(await bridge.selectDirectory(), "/srv/motions");
  assert.equal((await bridge.getOptionalComponents()).gvhmr.configured, true);
  assert.equal((await bridge.setupGvhmr()).action, "configured");
});
