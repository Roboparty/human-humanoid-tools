import type { AppPhase } from '../shared/runtime-state'

type ShutdownJoiner = () => void | Promise<void>

const PHASE_ORDER: readonly AppPhase[] = [
  'starting',
  'backend-starting',
  'ready',
  'after-window-open',
  'shutting-down'
]

export interface ShutdownResult {
  timedOut: boolean
  failures: Array<{ name: string; reason: unknown }>
}

export class AppLifecycle {
  private currentPhase: AppPhase = 'starting'
  private readonly shutdownJoiners = new Map<string, ShutdownJoiner>()

  get phase(): AppPhase {
    return this.currentPhase
  }

  transition(next: AppPhase): void {
    if (next === this.currentPhase) return

    const currentIndex = PHASE_ORDER.indexOf(this.currentPhase)
    const nextIndex = PHASE_ORDER.indexOf(next)

    // Startup phases never move backward; backend recovery is a SidecarSupervisor concern.
    if (nextIndex < currentIndex) {
      throw new Error(`Invalid lifecycle transition: ${this.currentPhase} -> ${next}`)
    }
    this.currentPhase = next
  }

  registerShutdownJoiner(name: string, joiner: ShutdownJoiner): () => void {
    if (this.shutdownJoiners.has(name)) {
      throw new Error(`Shutdown joiner already registered: ${name}`)
    }
    this.shutdownJoiners.set(name, joiner)
    return () => this.shutdownJoiners.delete(name)
  }

  async runShutdownJoiners(timeoutMs = 5_000): Promise<ShutdownResult> {
    this.transition('shutting-down')

    const failures: ShutdownResult['failures'] = []

    // Joiners are independent, so one failure must not prevent the other resources from closing.
    const work = Promise.all(
      [...this.shutdownJoiners.entries()].map(async ([name, joiner]) => {
        try {
          await joiner()
        } catch (reason) {
          failures.push({ name, reason })
        }
      })
    )

    let timer: NodeJS.Timeout | undefined

    // Bound total shutdown time so an unresponsive child cannot trap Electron indefinitely.
    const timeout = new Promise<'timeout'>((resolve) => {
      timer = setTimeout(() => resolve('timeout'), timeoutMs)
    })
    const result = await Promise.race([work.then(() => 'complete' as const), timeout])
    if (timer !== undefined) clearTimeout(timer)

    return { timedOut: result === 'timeout', failures }
  }
}
