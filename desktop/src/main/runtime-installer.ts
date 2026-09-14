import { spawn, type ChildProcess, type SpawnOptions } from 'node:child_process'
import { existsSync, lstatSync } from 'node:fs'
import { dirname, resolve } from 'node:path'

import type {
  RuntimeInstallMode,
  RuntimeInstallProgress,
  RuntimeInstallResult
} from '../shared/installer-api'

export interface RuntimeInstallCommand {
  command: string
  args: string[]
}

export interface RuntimeInstallerOptions {
  scriptPath: string
  platform?: NodeJS.Platform
  env?: NodeJS.ProcessEnv
  spawnProcess?: (
    command: string,
    args: readonly string[],
    options: SpawnOptions
  ) => ChildProcess
  onProgress?(progress: RuntimeInstallProgress): void
}

const INSTALL_ENVIRONMENT = new Set([
  'DBUS_SESSION_BUS_ADDRESS',
  'DISPLAY',
  'HOME',
  'HHTOOLS_INSTALL_ROOT',
  'HTTPS_PROXY',
  'HTTP_PROXY',
  'LANG',
  'LC_ALL',
  'NO_PROXY',
  'PATH',
  'SSL_CERT_DIR',
  'SSL_CERT_FILE',
  'TMPDIR',
  'WAYLAND_DISPLAY',
  'XAUTHORITY',
  'XDG_CACHE_HOME',
  'XDG_DATA_HOME',
  'XDG_RUNTIME_DIR'
])
const POLKIT_EXECUTABLE = '/usr/bin/pkexec'

function installerEnvironment(source: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  return Object.fromEntries(
    Object.entries(source).filter(
      ([key, value]) => value !== undefined && INSTALL_ENVIRONMENT.has(key.toUpperCase())
    )
  )
}

export function runtimeInstallCommand(
  mode: RuntimeInstallMode,
  scriptPath: string,
  platform: NodeJS.Platform = process.platform
): RuntimeInstallCommand {
  if (platform !== 'linux' && platform !== 'darwin') {
    throw new Error('The runtime installer page is available only on Linux and macOS')
  }
  if (platform === 'darwin' && mode !== 'user') {
    throw new Error('macOS supports current-user runtime installation only')
  }
  const script = resolve(scriptPath)
  if (!existsSync(script)) throw new Error(`Bundled runtime installer is missing: ${script}`)
  const metadata = lstatSync(script)
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    throw new Error(`Bundled runtime installer is not a regular file: ${script}`)
  }
  return mode === 'system'
    ? { command: POLKIT_EXECUTABLE, args: ['/bin/sh', script, '--system'] }
    : { command: '/bin/sh', args: [script] }
}

function cleanOutput(value: string): string {
  return value
    .replaceAll(/\u001B\[[0-?]*[ -/]*[@-~]/g, '')
    .replaceAll(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, '')
    .trim()
    .slice(-2_000)
}

function progressPhase(line: string): RuntimeInstallProgress['phase'] {
  const normalized = line.toLowerCase()
  if (normalized.includes('doctor') || normalized.includes('verif')) return 'verifying'
  return 'installing'
}

export class RuntimeInstaller {
  private readonly options: RuntimeInstallerOptions
  private progressPublisher: ((progress: RuntimeInstallProgress) => void) | undefined
  private child: ChildProcess | undefined
  private cancelled = false

  constructor(options: RuntimeInstallerOptions) {
    this.options = options
  }

  get running(): boolean {
    return this.child !== undefined
  }

  setProgressPublisher(
    publisher: ((progress: RuntimeInstallProgress) => void) | undefined
  ): void {
    this.progressPublisher = publisher
  }

  async start(mode: RuntimeInstallMode): Promise<RuntimeInstallResult> {
    if (this.child !== undefined) throw new Error('A runtime installation is already running')
    const command = runtimeInstallCommand(
      mode,
      this.options.scriptPath,
      this.options.platform ?? process.platform
    )
    this.cancelled = false
    this.publish({
      phase: 'preparing',
      message: mode === 'system'
        ? 'Waiting for system administrator authentication…'
        : 'Preparing the private HHTools runtime…'
    })

    return new Promise((resolveResult) => {
      let settled = false
      const finish = (result: RuntimeInstallResult): void => {
        if (settled) return
        settled = true
        this.child = undefined
        resolveResult(result)
      }
      const child = (this.options.spawnProcess ?? spawn)(command.command, command.args, {
        cwd: dirname(resolve(this.options.scriptPath)),
        env: installerEnvironment(this.options.env ?? process.env),
        detached: true,
        stdio: ['ignore', 'pipe', 'pipe']
      })
      this.child = child

      const forward = (channel: 'stdout' | 'stderr', chunk: Buffer | string): void => {
        const detail = cleanOutput(String(chunk))
        if (!detail) return
        const lines = detail.split(/\r?\n/).filter(Boolean)
        const message = lines.at(-1) ?? detail
        this.publish({
          phase: progressPhase(message),
          message,
          detail: `${channel}: ${detail}`
        })
      }
      child.stdout?.on('data', (chunk) => forward('stdout', chunk))
      child.stderr?.on('data', (chunk) => forward('stderr', chunk))
      child.once('error', (error) => {
        const message = command.command === POLKIT_EXECUTABLE && (error as NodeJS.ErrnoException).code === 'ENOENT'
          ? 'System authentication is unavailable because pkexec is not installed.'
          : error.message
        this.publish({ phase: 'failed', message })
        finish({ status: 'failed', message })
      })
      child.once('close', (code, signal) => {
        if (settled || this.child !== child) return
        if (this.cancelled) {
          const message = 'Runtime installation was cancelled.'
          this.publish({ phase: 'cancelled', message })
          finish({ status: 'cancelled', message })
          return
        }
        if (command.command === POLKIT_EXECUTABLE && code === 126) {
          const message = 'System authentication was cancelled.'
          this.publish({ phase: 'cancelled', message })
          finish({ status: 'cancelled', message })
          return
        }
        if (code === 0) {
          const message = 'HHTools runtime installed successfully. Restarting…'
          this.publish({ phase: 'completed', message })
          finish({ status: 'completed', message })
          return
        }
        const message = signal
          ? `Runtime installer stopped by ${signal}.`
          : `Runtime installer exited with status ${code ?? 'unknown'}.`
        this.publish({ phase: 'failed', message })
        finish({ status: 'failed', message })
      })
    })
  }

  cancel(): void {
    const child = this.child
    if (child?.pid === undefined) return
    this.cancelled = true
    try {
      process.kill(-child.pid, 'SIGTERM')
    } catch {
      try {
        child.kill('SIGTERM')
      } catch {
        // A system install may already be running as root. The OS owns that
        // process, so closing the unprivileged UI must never crash Electron.
      }
    }
  }

  private publish(progress: RuntimeInstallProgress): void {
    this.options.onProgress?.(progress)
    this.progressPublisher?.(progress)
  }
}
