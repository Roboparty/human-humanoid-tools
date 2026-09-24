export interface ActivityBlocker {
  start(type: 'prevent-app-suspension'): number
  stop(id: number): void
}

/** Keep finite startup/install work running when macOS moves the app to the background. */
export async function withAppActivity<T>(
  blocker: ActivityBlocker | undefined,
  operation: () => Promise<T>
): Promise<T> {
  const id = blocker?.start('prevent-app-suspension')
  try {
    return await operation()
  } finally {
    if (id !== undefined) blocker?.stop(id)
  }
}
