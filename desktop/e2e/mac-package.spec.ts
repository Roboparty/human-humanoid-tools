import { _electron as electron, expect, test } from '@playwright/test'
import { mkdir, mkdtemp, readFile, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

import type { HHToolsDesktopApi } from '../src/shared/desktop-api'
import type { RuntimeInstallProgress } from '../src/shared/installer-api'

test('installs and opens the macOS package without a checkout or system Python', async ({}, testInfo) => {
  test.skip(process.platform !== 'darwin' || !process.env.HHTOOLS_E2E_EXECUTABLE)
  // Includes downloads and macOS's first validation of scientific native libraries.
  test.setTimeout(1_800_000)
  const root = await mkdtemp(join(tmpdir(), 'hhtools-mac-package-'))
  const home = join(root, 'home')
  const runtimeRoot = join(home, 'Library', 'Application Support', 'hhtools')
  await mkdir(home)
  const env = Object.fromEntries(
    Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined)
  )
  for (const key of Object.keys(env)) {
    if (key.startsWith('HHTOOLS_') || key.startsWith('PYTHON') || key.startsWith('UV_')) {
      delete env[key]
    }
  }
  Object.assign(env, {
    HOME: home,
    // Finder launches do not inherit a developer's toolchain on PATH.
    PATH: '/usr/bin:/bin:/usr/sbin:/sbin',
    HHTOOLS_WEB_SETTINGS_PATH: join(root, 'settings.json'),
    HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH: join(root, 'motion-library.json')
  })
  const launch = () => electron.launch({
    executablePath: resolve(process.env.HHTOOLS_E2E_EXECUTABLE!),
    args: [`--user-data-dir=${join(root, 'user-data')}`, '--lang=en-US'],
    cwd: root,
    env
  })

  const setup = await launch()
  let setupClosed = false
  setup.on('close', () => { setupClosed = true })
  try {
    await expect.poll(() => setup.windows().find((page) => page.url().startsWith('data:'))?.url(), {
      timeout: 30_000
    }).toContain('Set%20up')
    const page = setup.windows().find((page) => page.url().startsWith('data:'))!
    await expect(page.locator('input[value="system"]')).toHaveCount(0)
    await expect(page.locator('#install')).toBeDisabled()
    await page.screenshot({ path: testInfo.outputPath('mac-first-run.png') })
    let reportFailure: (message: string) => void = () => {}
    const failed = new Promise<string>((resolveFailure) => { reportFailure = resolveFailure })
    await page.exposeFunction('reportInstallerProgress', (update: RuntimeInstallProgress) => {
      console.log(`[mac-package] ${update.phase}: ${update.message}`)
      if (update.phase === 'failed' || update.phase === 'cancelled') {
        reportFailure(update.detail ?? update.message)
      }
    })
    await page.evaluate(() => {
      window.hhtoolsInstaller.onProgress((update) => {
        const reporter = window as unknown as {
          reportInstallerProgress: (value: RuntimeInstallProgress) => Promise<void>
        }
        void reporter.reportInstallerProgress(update)
      })
    })
    // Playwright owns the next launch so it can verify the installed package after app.exit(0).
    await setup.evaluate(({ app }) => { app.relaunch = () => {} })
    const result = Promise.race([
      setup.waitForEvent('close', { timeout: 1_500_000 }).then(() => undefined), failed
    ])
    await page.locator('#license').check()
    await page.locator('#install').click()
    const failure = await result
    if (failure) throw new Error(failure)
    expect((await readFile(join(runtimeRoot, 'runtime-version'), 'utf8')).trim())
      .toMatch(/^0\.1\.0\+sha256\.[a-f0-9]{20}$/)
  } finally {
    if (!setupClosed) await setup.close()
  }

  const app = await launch()
  let backendPid: number | undefined
  try {
    await expect.poll(() => app.windows().find((page) => page.url().startsWith('http:'))?.url(), {
      timeout: 90_000
    }).toContain('http://127.0.0.1:')
    const page = app.windows().find((page) => page.url().startsWith('http:'))!
    await expect(page.locator('#app[data-hhtools-ready="true"]')).toBeVisible()
    const state = await page.evaluate(() =>
      (window.hhtoolsDesktop as HHToolsDesktopApi).getRuntimeState()
    )
    expect(state).toMatchObject({ appPhase: 'after-window-open', backendState: 'ready' })
    backendPid = state.backendPid
    expect(backendPid).toEqual(expect.any(Number))
    const logDirectory = join(root, 'user-data', 'logs')
    const logName = (await readdir(logDirectory)).find((name) => name.startsWith('desktop-'))!
    const log = await readFile(join(logDirectory, logName), 'utf8')
    expect(log).toContain('"runtimeKind":"managed"')
    await page.screenshot({ path: testInfo.outputPath('mac-packaged-app.png') })
  } finally {
    await app.close()
    if (backendPid !== undefined) {
      await expect.poll(() => {
        try { process.kill(backendPid!, 0); return true } catch { return false }
      }, { timeout: 10_000 }).toBe(false)
    }
  }
  await rm(root, { recursive: true, force: true })
})
