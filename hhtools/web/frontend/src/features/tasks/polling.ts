export interface TaskPollingState {
  readonly visible: boolean;
  readonly drawerOpen: boolean;
  readonly hasActiveTasks: boolean;
  readonly consecutiveFailures: number;
}

const ACTIVE_POLL_MS = 2_500;
const OPEN_IDLE_POLL_MS = 15_000;
const MAX_BACKOFF_MS = 60_000;

export function taskPollingDelay(state: TaskPollingState): number | null {
  if (!state.visible || (!state.drawerOpen && !state.hasActiveTasks)) {
    return null;
  }
  const base = state.hasActiveTasks ? ACTIVE_POLL_MS : OPEN_IDLE_POLL_MS;
  const failures = Math.max(0, Math.min(6, Math.floor(state.consecutiveFailures)));
  return Math.min(MAX_BACKOFF_MS, base * 2 ** failures);
}
