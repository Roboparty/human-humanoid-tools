import { spawn, type ChildProcess, type SpawnOptions } from 'node:child_process'
import { EventEmitter } from 'node:events'

import type { SidecarState } from '../shared/runtime-state'
import type { LoggerLike } from './desktop-logger'

export interface SidecarSnapshot {
  state: SidecarState
  origin: string
  pid?: number
  error?: string
}

export interface SidecarConfig {
  command: string
  args: string[]
  cwd: string
  env: NodeJS.ProcessEnv
  origin: string
  sessionSecret: string
  startTimeoutMs?: number
  stopTimeoutMs?: number
}

type SpawnProcess = (command: string, args: readonly string[], options: SpawnOptions) => ChildProcess
type RequestProcessStop = (child: ChildProcess) => Promise<void>
type KillProcessTree = (child: ChildProcess) => Promise<void>

export interface SidecarDependencies {
  spawnProcess?: SpawnProcess
  requestProcessStop?: RequestProcessStop
  killProcessTree?: KillProcessTree
  fetchHealth?: typeof fetch
}

function messageFrom(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason)
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds))
}

async function runTaskkill(child: ChildProcess, force: boolean): Promise<void> {
  if (child.pid === undefined) return
  const args = ['/pid', String(child.pid), '/T']
  if (force) args.push('/F')
  await new Promise<void>((resolve) => {
    const killer = spawn('taskkill', args, { windowsHide: true, stdio: 'ignore' })
    killer.once('error', () => resolve())
    killer.once('exit', () => resolve())
  })
}

async function defaultRequestProcessStop(child: ChildProcess): Promise<void> {
  if (process.platform === 'win32') {
    // uv/venv launchers can create another Python process, so stop the whole Windows tree.
    await runTaskkill(child, false)
  } else {
    child.kill('SIGTERM')
  }
}

async function defaultKillProcessTree(child: ChildProcess): Promise<void> {
  if (child.pid === undefined) return
  if (process.platform !== 'win32') {
    child.kill('SIGKILL')
    return
  }
  await runTaskkill(child, true)
}

export class SidecarSupervisor {
  private readonly emitter = new EventEmitter()
  private readonly spawnProcess: SpawnProcess
  private readonly requestProcessStop: RequestProcessStop
  private readonly killProcessTree: KillProcessTree
  private readonly fetchHealth: typeof fetch
  private child?: ChildProcess

  // A generation token prevents late events from an old process corrupting a restarted process.
  private generation = 0
  private startPromise?: Promise<SidecarSnapshot>

  // Expected exits are shutdowns; every other exit after spawn is reported as a crash.
  private expectedExitGeneration?: number
  private currentState: SidecarState = 'stopped'
  private lastError?: string

  constructor(
    private readonly config: SidecarConfig,
    private readonly logger: LoggerLike,
    dependencies: SidecarDependencies = {}
  ) {
    this.spawnProcess = dependencies.spawnProcess ?? spawn
    this.requestProcessStop = dependencies.requestProcessStop ?? defaultRequestProcessStop
    this.killProcessTree = dependencies.killProcessTree ?? defaultKillProcessTree
    this.fetchHealth = dependencies.fetchHealth ?? fetch
  }

  get snapshot(): SidecarSnapshot {
    return {
      state: this.currentState,
      origin: this.config.origin,
      pid: this.child?.pid,
      error: this.lastError
    }
  }

  onStateChange(listener: (snapshot: SidecarSnapshot) => void): () => void {
    this.emitter.on('state', listener)
    return () => this.emitter.off('state', listener)
  }

  updateEnvironment(environment: NodeJS.ProcessEnv): void {
    if (this.currentState === 'starting' || this.currentState === 'stopping') {
      throw new Error('Cannot replace the sidecar environment during a state transition')
    }
    // Optional components are configured after the window opens. Replacing the
    // spawn environment lets restart() pick up those paths without relaunching Electron.
    this.config.env = environment
  }

  start(): Promise<SidecarSnapshot> {
    if (this.currentState === 'ready') return Promise.resolve(this.snapshot)

    // Coalesce concurrent startup requests into one child process and one readiness result.
    if (this.startPromise !== undefined) return this.startPromise

    this.startPromise = this.startInternal().finally(() => {
      this.startPromise = undefined
    })
    return this.startPromise
  }

  async restart(): Promise<SidecarSnapshot> {
    await this.stop()
    return this.start()
  }

  async stop(): Promise<void> {
    const child = this.child
    if (child === undefined) {
      this.setState('stopped')
      return
    }

    const generation = this.generation
    this.expectedExitGeneration = generation
    this.setState('stopping')
    await this.requestProcessStop(child)

    const exited = await this.waitForExit(child, this.config.stopTimeoutMs ?? 5_000)
    if (!exited) {
      // Graceful termination gets a deadline; force-kill is only the final cleanup fallback.
      this.logger.warn('Sidecar did not stop before timeout', { pid: child.pid })
      await this.killProcessTree(child)
      await this.waitForExit(child, 1_000)
    }

    if (generation === this.generation) {
      this.child = undefined
      this.setState('stopped')
    }
  }

  private async startInternal(): Promise<SidecarSnapshot> {
    const generation = ++this.generation
    this.lastError = undefined
    this.expectedExitGeneration = undefined
    this.setState('starting')
    this.logger.info('Starting Python sidecar', { origin: this.config.origin })

    const child = this.spawnProcess(this.config.command, this.config.args, {
      cwd: this.config.cwd,
      env: this.config.env,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe']
    })
    this.child = child

    child.stdout?.on('data', (chunk: Buffer | string) =>
      this.logger.processOutput('stdout', chunk.toString('utf8'))
    )
    child.stderr?.on('data', (chunk: Buffer | string) =>
      this.logger.processOutput('stderr', chunk.toString('utf8'))
    )
    child.on('error', (error) => this.logger.error('Sidecar process error', { error: error.message }))
    child.once('exit', (code, signal) => this.handleExit(generation, code, signal))

    const spawnFailure = new Promise<never>((_resolve, reject) => {
      child.once('error', reject)
    })

    try {
      // A spawned PID is not sufficient: the WebUI is usable only after FastAPI is listening.
      await Promise.race([this.waitForHealth(generation), spawnFailure])
      if (generation !== this.generation) throw new Error('Sidecar start was superseded')
      this.setState('ready')
      this.logger.info('Python sidecar is ready', { pid: child.pid, origin: this.config.origin })
      return this.snapshot
    } catch (reason) {
      this.lastError = messageFrom(reason)
      this.logger.error('Python sidecar failed to start', { error: this.lastError })
      this.expectedExitGeneration = generation
      if (child.exitCode === null) await this.requestProcessStop(child)
      const exited = await this.waitForExit(child, 1_000)
      if (!exited) await this.killProcessTree(child)
      if (generation === this.generation) {
        this.child = undefined
        this.setState('crashed')
      }
      throw reason
    }
  }

  private async waitForHealth(generation: number): Promise<void> {
    const deadline = Date.now() + (this.config.startTimeoutMs ?? 60_000)
    let attemptDelay = 100

    while (Date.now() < deadline) {
      if (generation !== this.generation) throw new Error('Sidecar health check was superseded')
      if (this.child?.exitCode !== null) throw new Error('Sidecar exited before becoming ready')

      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 1_500)
      try {
        // Health checks pass through the same authentication guard as renderer requests.
        const response = await this.fetchHealth(`${this.config.origin}/api/health`, {
          headers: { 'X-HHTools-Session': this.config.sessionSecret },
          signal: controller.signal
        })
        if (response.ok) return
      } catch {
        // Startup polling is expected to fail until uvicorn begins listening.
      } finally {
        clearTimeout(timer)
      }

      await delay(attemptDelay)

      // Poll quickly at first, then cap retries to avoid needless CPU use during heavy imports.
      attemptDelay = Math.min(1_000, Math.round(attemptDelay * 1.5))
    }
    throw new Error('Timed out waiting for the Python sidecar health check')
  }

  private handleExit(generation: number, code: number | null, signal: NodeJS.Signals | null): void {
    // Ignore an exit event from a process that restart() has already superseded.
    if (generation !== this.generation) return
    this.child = undefined

    if (this.expectedExitGeneration === generation || this.currentState === 'stopping') {
      this.setState('stopped')
      return
    }

    this.lastError = `Sidecar exited unexpectedly (code=${String(code)}, signal=${String(signal)})`
    this.logger.error(this.lastError)
    this.setState('crashed')
  }

  private waitForExit(child: ChildProcess, timeoutMs: number): Promise<boolean> {
    if (child.exitCode !== null) return Promise.resolve(true)
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        child.removeListener('exit', onExit)
        resolve(false)
      }, timeoutMs)
      const onExit = (): void => {
        clearTimeout(timer)
        resolve(true)
      }
      child.once('exit', onExit)
    })
  }

  private setState(next: SidecarState): void {
    if (next === this.currentState) return
    this.currentState = next
    this.emitter.emit('state', this.snapshot)
  }
}
