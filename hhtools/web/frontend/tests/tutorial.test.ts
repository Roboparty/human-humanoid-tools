import assert from "node:assert/strict";
import test from "node:test";

import {
  LEGACY_TUTORIAL_STORAGE_KEY,
  TUTORIAL_STORAGE_KEY,
  TUTORIAL_STEPS,
  hasSeenFirstRunTutorial,
  markFirstRunTutorialSeen,
  rememberTutorialSeen,
  shouldAutoOpenTutorial,
} from "../src/features/tutorial/model.ts";

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

test("first-run tutorial records the current flag and honors the legacy flag", () => {
  const fresh = storage();
  assert.equal(hasSeenFirstRunTutorial(fresh), false);
  markFirstRunTutorialSeen(fresh);
  assert.equal(fresh.getItem(TUTORIAL_STORAGE_KEY), "1");
  assert.equal(hasSeenFirstRunTutorial(fresh), true);

  assert.equal(
    hasSeenFirstRunTutorial(storage({ [LEGACY_TUTORIAL_STORAGE_KEY]: "1" })),
    true,
  );
  assert.equal(
    hasSeenFirstRunTutorial(storage({ [TUTORIAL_STORAGE_KEY]: "0" })),
    false,
  );
});

test("first-run tutorial tolerates unavailable browser storage", () => {
  assert.equal(hasSeenFirstRunTutorial(undefined), false);
  assert.equal(
    hasSeenFirstRunTutorial({
      getItem() {
        throw new Error("unavailable");
      },
    }),
    false,
  );
  assert.doesNotThrow(() =>
    markFirstRunTutorialSeen({
      setItem() {
        throw new Error("unavailable");
      },
    }),
  );
});

test("desktop persistence survives changing localhost origins", async () => {
  let desktopSeen = false;
  const bridge = {
    async hasSeenTutorial() {
      return desktopSeen;
    },
    async markTutorialSeen() {
      desktopSeen = true;
    },
  };

  assert.equal(await shouldAutoOpenTutorial(storage(), bridge), true);
  await rememberTutorialSeen(storage(), bridge);
  assert.equal(await shouldAutoOpenTutorial(storage(), bridge), false);
});

test("tutorial retains the nine-step product journey with plain localized copy", () => {
  assert.deepEqual(
    TUTORIAL_STEPS.map(({ id }) => id),
    [
      "welcome",
      "motion-import",
      "motion-library",
      "robot-import",
      "calibration",
      "retarget",
      "stage-layers",
      "export",
      "done",
    ],
  );
  assert.deepEqual(
    TUTORIAL_STEPS.map(({ view }) => view),
    [
      "motion",
      "motion",
      "motion",
      "robot-assets",
      "h2r",
      "h2r",
      "motion",
      "h2r",
      "motion",
    ],
  );
  assert.equal(
    TUTORIAL_STEPS[0].anchor,
    '[data-tutorial="workspace-navigation"]',
  );
  assert.equal(TUTORIAL_STEPS[0].placement, "right");
  assert.equal(TUTORIAL_STEPS[6].anchor, ".stage-view-menu");
  assert.equal(TUTORIAL_STEPS[7].anchor, '[data-tutorial="h2r-result"]');
  for (const step of TUTORIAL_STEPS) {
    assert.doesNotMatch(step.title.en, /<[^>]+>/);
    assert.doesNotMatch(step.title.zh, /<[^>]+>/);
    assert.doesNotMatch(step.body.en, /<[^>]+>/);
    assert.doesNotMatch(step.body.zh, /<[^>]+>/);
  }
});
