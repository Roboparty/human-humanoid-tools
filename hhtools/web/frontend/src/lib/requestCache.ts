export interface RequestCache<T> {
  read(loader: () => Promise<T>, signal?: AbortSignal): Promise<T>;
  invalidate(): void;
}

function abortError(signal: AbortSignal): Error {
  if (signal.reason instanceof Error) return signal.reason;
  const error = new Error("The request was aborted.");
  error.name = "AbortError";
  return error;
}

function waitForConsumer<T>(request: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return request;
  if (signal.aborted) return Promise.reject(abortError(signal));
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      signal.removeEventListener("abort", onAbort);
      reject(abortError(signal));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    request.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (reason: unknown) => {
        signal.removeEventListener("abort", onAbort);
        reject(reason);
      },
    );
  });
}

/** Coalesce concurrent reads while keeping cancellation local to each consumer. */
export function createRequestCache<T>(
  ttlMs: number,
  clock: () => number = Date.now,
): RequestCache<T> {
  let cached: T | undefined;
  let hasCached = false;
  let expiresAt = 0;
  let inFlight: Promise<T> | null = null;
  let generation = 0;

  return {
    read(loader, signal) {
      const now = clock();
      if (hasCached && now < expiresAt) {
        return waitForConsumer(Promise.resolve(cached as T), signal);
      }
      if (!inFlight) {
        const requestGeneration = generation;
        const request = Promise.resolve()
          .then(loader)
          .then((value) => {
            if (requestGeneration === generation && ttlMs > 0) {
              cached = value;
              hasCached = true;
              expiresAt = clock() + ttlMs;
            }
            return value;
          });
        let tracked: Promise<T>;
        tracked = request.finally(() => {
          if (inFlight === tracked) inFlight = null;
        });
        inFlight = tracked;
      }
      return waitForConsumer(inFlight, signal);
    },
    invalidate() {
      generation += 1;
      cached = undefined;
      hasCached = false;
      expiresAt = 0;
      inFlight = null;
    },
  };
}
