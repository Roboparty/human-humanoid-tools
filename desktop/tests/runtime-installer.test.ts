import { EventEmitter } from 'node:events'
import { mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Script } from 'node:vm'

import { describe, expect, it, vi } from 'vitest'

import { installerDataUrl } from '../src/main/installer-page'
import {
  RuntimeInstaller,
  runtimeInstallCommand
} from '../src/main/runtime-installer'
import type { RuntimeInstallProgress } from '../src/shared/installer-api'

function installerFixture(): { root: string; script: string } {
  const root = mkdtempSync(join(tmpdir(), 'hhtools-runtime-installer-test-'))
  const script = join(root, 'install.sh')
  writeFileSync(script, '#!/bin/sh\nexit 0\n', 'utf8')
  return { root, script }
}

describe('runtime installer', () => {
  it('installs on macOS without elevation and rejects system installation', () => {
    const { script } = installerFixture()
    expect(runtimeInstallCommand('user', script, 'darwin')).toEqual({
      command: '/bin/sh', args: [script]
    })
    expect(() => runtimeInstallCommand('system', script, 'darwin')).toThrow('current-user')

    const url = installerDataUrl({
      version: '0.1.0', reason: 'Runtime required', allowSystemInstall: false
    })
    const html = decodeURIComponent(url.slice(url.indexOf(',') + 1))
    expect(html).toContain('value="user" checked')
    expect(html).not.toContain('value="system"')
  })

  it('uses a normal shell for user installs and an OS-owned prompt for system installs', () => {
    const { script } = installerFixture()

    expect(runtimeInstallCommand('user', script, 'linux')).toEqual({
      command: '/bin/sh',
      args: [script]
    })
    expect(runtimeInstallCommand('system', script, 'linux')).toEqual({
      command: '/usr/bin/pkexec',
      args: ['/bin/sh', script, '--system']
    })
    expect(runtimeInstallCommand('system', script, 'linux').args).not.toContain('-S')
  })

  it('rejects symlinked bootstrap scripts', () => {
    const { root, script } = installerFixture()
    const link = join(root, 'linked-install.sh')
    symlinkSync(script, link)

    expect(() => runtimeInstallCommand('user', link, 'linux')).toThrow('not a regular file')
  })

  it('streams bounded progress and reports a successful installer process', async () => {
    const { script } = installerFixture()
    const progress: RuntimeInstallProgress[] = []
    const child = new EventEmitter() as EventEmitter & {
      pid: number
      stdout: EventEmitter
      stderr: EventEmitter
      kill: ReturnType<typeof vi.fn>
    }
    child.pid = 12345
    child.stdout = new EventEmitter()
    child.stderr = new EventEmitter()
    child.kill = vi.fn()
    const spawnProcess = vi.fn((
      _command: string,
      _args: readonly string[],
      _options: import('node:child_process').SpawnOptions
    ) => child as never)
    const installer = new RuntimeInstaller({
      scriptPath: script,
      platform: 'linux',
      env: { HOME: '/home/test', HHTOOLS_INSTALL_ROOT: '/custom/runtime', SECRET_TOKEN: 'never-forward' },
      spawnProcess,
      onProgress: (update) => progress.push(update)
    })

    const resultPromise = installer.start('user')
    child.stdout.emit('data', 'Installing HHTools 0.1.0\n')
    child.stdout.emit('data', 'HHTools 0.1.0 is ready.\n')
    child.emit('close', 0, null)

    await expect(resultPromise).resolves.toEqual({
      status: 'completed',
      message: 'HHTools runtime installed successfully. Restarting…'
    })
    expect(progress.at(-1)?.phase).toBe('completed')
    expect(spawnProcess.mock.calls[0]?.[2].env).toEqual({
      HOME: '/home/test', HHTOOLS_INSTALL_ROOT: '/custom/runtime'
    })
  })

  it('treats a dismissed system authentication dialog as cancellation', async () => {
    const { script } = installerFixture()
    const child = new EventEmitter() as EventEmitter & {
      pid: number
      stdout: EventEmitter
      stderr: EventEmitter
      kill: ReturnType<typeof vi.fn>
    }
    child.pid = 12346
    child.stdout = new EventEmitter()
    child.stderr = new EventEmitter()
    child.kill = vi.fn()
    const installer = new RuntimeInstaller({
      scriptPath: script,
      platform: 'linux',
      spawnProcess: vi.fn(() => child as never)
    })

    const resultPromise = installer.start('system')
    child.emit('close', 126, null)

    await expect(resultPromise).resolves.toEqual({
      status: 'cancelled',
      message: 'System authentication was cancelled.'
    })
  })

  it('renders an animated, localized setup page without a password field', () => {
    const url = installerDataUrl({
      version: '0.1.0',
      reason: 'Runtime required'
    })
    const html = decodeURIComponent(url.slice(url.indexOf(',') + 1))

    expect(html).toContain('30 motions and 6 robots')
    expect(html).toContain('HHTools never receives your password')
    expect(html).toContain('设置本地运行环境')
    expect(html).toContain('prefers-reduced-motion')
    expect(html).not.toMatch(/type=["']password/i)
    expect(html).not.toContain('sudo -S')
    const inlineScript = html.match(/<script>([\s\S]*)<\/script>/)?.[1]
    if (inlineScript === undefined) throw new Error('installer page has no inline script')
    expect(() => new Script(inlineScript)).not.toThrow()
  })
})
