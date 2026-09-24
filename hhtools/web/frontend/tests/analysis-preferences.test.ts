import assert from "node:assert/strict";
import test from "node:test";

import {
  FORCE_REANALYSIS_STORAGE_KEY,
  storedForceReanalysis,
  storeForceReanalysis,
} from "../src/features/analysis/preferences.ts";

test("force re-analysis defaults off and restores only an explicit opt-in", () => {
  assert.equal(storedForceReanalysis(undefined), false);
  assert.equal(storedForceReanalysis({ getItem: () => null }), false);
  assert.equal(storedForceReanalysis({ getItem: () => "false" }), false);
  assert.equal(storedForceReanalysis({ getItem: () => "true" }), true);
});

test("force re-analysis persistence is storage-error safe", () => {
  assert.equal(
    storedForceReanalysis({
      getItem: () => {
        throw new Error("blocked");
      },
    }),
    false,
  );

  let saved: readonly [string, string] | null = null;
  storeForceReanalysis(
    {
      setItem: (key, value) => {
        saved = [key, value];
      },
    },
    true,
  );
  assert.deepEqual(saved, [FORCE_REANALYSIS_STORAGE_KEY, "true"]);
  storeForceReanalysis(
    {
      setItem: (key, value) => {
        saved = [key, value];
      },
    },
    false,
  );
  assert.deepEqual(saved, [FORCE_REANALYSIS_STORAGE_KEY, "false"]);
  assert.doesNotThrow(() =>
    storeForceReanalysis(
      {
        setItem: () => {
          throw new Error("blocked");
        },
      },
      false,
    ),
  );
});
