import assert from "node:assert/strict";
import test from "node:test";

import {
  WORKSPACE_LOCALE_STORAGE_KEY,
  storedLocale,
  storedLocaleOverride,
  storeLocale,
  systemLocale,
} from "../src/localization.ts";
import {
  DEFAULT_WORKSPACE_LAYOUT,
  WORKSPACE_LAYOUT_STORAGE_KEY,
  storedWorkspaceLayout,
  storeWorkspaceLayout,
} from "../src/workspaceLayout.ts";

function storage(initial: Readonly<Record<string, string>> = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem(key: string) {
      return values.get(key) ?? null;
    },
    setItem(key: string, value: string) {
      values.set(key, value);
    },
  };
}

test("locale restores legacy preferences and otherwise follows Chinese browsers", () => {
  assert.equal(
    storedLocale(
      storage({ [WORKSPACE_LOCALE_STORAGE_KEY]: '{"locale":"zh-CN"}' }),
      ["en-US"],
    ),
    "zh-CN",
  );
  assert.equal(storedLocale(undefined, ["zh-Hans-SG", "en-US"]), "zh-CN");
  assert.equal(storedLocale(undefined, ["fr-FR"]), "en");
  assert.equal(systemLocale(["zh-Hans-SG", "en-US"]), "zh-CN");
  assert.equal(systemLocale(["en-US", "zh-CN"]), "en");
  assert.equal(storedLocaleOverride(undefined), null);
  assert.equal(
    storedLocaleOverride(
      storage({ [WORKSPACE_LOCALE_STORAGE_KEY]: '{"locale":"en"}' }),
    ),
    "en",
  );
});

test("locale writes preserve other legacy workspace preferences", () => {
  const target = storage({
    [WORKSPACE_LOCALE_STORAGE_KEY]: '{"theme":"dark","locale":"en"}',
  });
  storeLocale(target, "zh-CN");
  assert.deepEqual(
    JSON.parse(target.getItem(WORKSPACE_LOCALE_STORAGE_KEY) ?? "{}"),
    { theme: "dark", locale: "zh-CN" },
  );
});

test("panel visibility is validated, persisted, and resettable", () => {
  const target = storage({
    [WORKSPACE_LAYOUT_STORAGE_KEY]:
      '{"sidebarHidden":true,"inspectorHidden":"yes"}',
  });
  assert.deepEqual(storedWorkspaceLayout(target), {
    sidebarHidden: true,
    inspectorHidden: false,
  });
  storeWorkspaceLayout(target, DEFAULT_WORKSPACE_LAYOUT);
  assert.deepEqual(
    JSON.parse(target.getItem(WORKSPACE_LAYOUT_STORAGE_KEY) ?? "{}"),
    DEFAULT_WORKSPACE_LAYOUT,
  );
});
