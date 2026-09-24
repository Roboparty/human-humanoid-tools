import assert from "node:assert/strict";
import test from "node:test";

import { createRequestCache } from "../src/lib/requestCache.ts";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((accept, decline) => {
    resolve = accept;
    reject = decline;
  });
  return { promise, resolve, reject };
}

test("coalesces concurrent reads and keeps consumer cancellation isolated", async () => {
  const cache = createRequestCache<number>(1_000);
  const pending = deferred<number>();
  const cancelled = new AbortController();
  let calls = 0;
  const loader = () => {
    calls += 1;
    return pending.promise;
  };

  const first = cache.read(loader, cancelled.signal);
  const second = cache.read(loader);
  cancelled.abort();
  pending.resolve(42);

  await assert.rejects(first, { name: "AbortError" });
  assert.equal(await second, 42);
  assert.equal(calls, 1);
});

test("reuses a short-lived value and reloads after expiry or invalidation", async () => {
  let now = 100;
  let calls = 0;
  const cache = createRequestCache(50, () => now);
  const loader = async () => ++calls;

  assert.equal(await cache.read(loader), 1);
  assert.equal(await cache.read(loader), 1);
  now = 151;
  assert.equal(await cache.read(loader), 2);
  cache.invalidate();
  assert.equal(await cache.read(loader), 3);
});

test("does not cache rejected requests", async () => {
  const cache = createRequestCache<number>(1_000);
  let calls = 0;
  const loader = async () => {
    calls += 1;
    if (calls === 1) throw new Error("temporary");
    return 7;
  };

  await assert.rejects(cache.read(loader), /temporary/);
  assert.equal(await cache.read(loader), 7);
  assert.equal(calls, 2);
});
