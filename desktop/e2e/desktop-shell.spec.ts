import { _electron as electron, expect, test as base, type Locator, type Page } from '@playwright/test'
import { mkdir, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import type { HHToolsDesktopApi } from '../src/shared/desktop-api'

const desktopRoot = fileURLToPath(new URL('..', import.meta.url))
const repositoryRoot = resolve(desktopRoot, '..')

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch {
    return false
  }
}

async function expectBorderless(button: Locator): Promise<void> {
  await expect(button).toBeVisible()
  expect(
    await button.evaluate((element) => getComputedStyle(element).borderTopColor)
  ).toBe('rgba(0, 0, 0, 0)')
}

type DesktopFixture = { page: Page }

const test = base.extend<DesktopFixture>({
  page: async ({}, use, testInfo) => {
    const packagedExecutable = process.env.HHTOOLS_E2E_EXECUTABLE
    const electronApp = await electron.launch({
      ...(packagedExecutable === undefined
        ? {}
        : { executablePath: packagedExecutable }),
      args: [
        `--user-data-dir=${testInfo.outputPath('user-data')}`,
        '--lang=en-US',
        ...(process.env.HHTOOLS_E2E_SOFTWARE_RENDERING === '1'
          ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] : []),
        ...(packagedExecutable === undefined
          ? [join(desktopRoot, 'out', 'main', 'index.js')]
          : [])
      ],
      cwd: desktopRoot,
      env: {
        ...process.env,
        HHTOOLS_REPO_ROOT: repositoryRoot,
        HHTOOLS_WEB_SETTINGS_PATH: testInfo.outputPath('web-settings.json'),
        HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH: testInfo.outputPath(
          'motion-library-settings.json'
        ),
        XDG_CONFIG_HOME: testInfo.outputPath('config'),
        ELECTRON_DISABLE_SECURITY_WARNINGS: 'true'
      }
    })

    let backendPid: number | undefined
    try {
      await expect
        .poll(
          () =>
            electronApp
              .windows()
              .some((window) => !window.url().startsWith('data:')),
          { timeout: 90_000 }
        )
        .toBe(true)
      const page = electronApp
        .windows()
        .find((window) => !window.url().startsWith('data:'))
      if (!page) throw new Error('Desktop main window did not open')
      const pageErrors: string[] = []
      page.on('pageerror', (error) => pageErrors.push(error.message))

      await expect(page.locator("#app[data-hhtools-ready='true']")).toBeVisible()
    const state = await page.evaluate(() =>
      (window.hhtoolsDesktop as HHToolsDesktopApi).getRuntimeState()
    )
    expect(state).toMatchObject({ appPhase: 'after-window-open', backendState: 'ready' })
      backendPid = state.backendPid
      expect(typeof backendPid).toBe('number')
      await use(page)
      expect(pageErrors).toEqual([])
    } finally {
      await electronApp.close()
      if (backendPid !== undefined) {
        await expect.poll(() => processIsAlive(backendPid as number), { timeout: 10_000 })
          .toBe(false)
      }
    }
  }
})

async function skipFirstRun(page: Page): Promise<void> {
  const overlay = page.locator('[data-tutorial-overlay]')
  await expect(overlay).toHaveAttribute('data-tutorial-step', 'welcome')
  await overlay.getByRole('button', { name: 'Skip tutorial' }).click()
  await expect(overlay).toBeHidden()
}

test('persists onboarding and restores tutorial focus', async ({ page }) => {
  await expect(page).toHaveTitle('Human-Humanoid Tools')
  await expect(page.locator("#app[data-hhtools-ready='true']")).toBeVisible()
  await expect(page.getByLabel('HHTOOLS', { exact: true })).toBeVisible()
  const firstRunTutorial = page.locator('[data-tutorial-overlay]').getByRole('dialog')
  await expect(firstRunTutorial).toBeVisible()
  await expect(firstRunTutorial).toHaveAccessibleName('1. Welcome to Human-Humanoid Tools')
  await expect(page.locator('[data-tutorial-overlay]')).toHaveAttribute(
    'data-tutorial-step',
    'welcome'
  )
  await firstRunTutorial.getByRole('button', { name: 'Skip tutorial' }).click()
  await expect(firstRunTutorial).toBeHidden()
  await page.reload()
  await expect(page.locator("#app[data-hhtools-ready='true']")).toBeVisible()
  await expect(firstRunTutorial).toBeHidden()
  await expect.poll(() => page.evaluate(() => window.hhtoolsDesktop.hasSeenTutorial())).toBe(true)

  const menu = page.getByRole('menubar', { name: 'Application menu' })
  const helpTrigger = menu.getByRole('menuitem', { name: 'Help', exact: true })
  const helpMenu = page.getByRole('menu', { name: 'Help' })
  await helpTrigger.hover()
  await expect(helpMenu.getByRole('menuitem')).toHaveText(['Tutorial', 'About hhtools'])
  await helpMenu.getByRole('menuitem', { name: 'Tutorial' }).click()
  const reopenedTutorial = page.locator('[data-tutorial-overlay]').getByRole('dialog')
  await expect(reopenedTutorial).toBeVisible()
  const tutorialOverlay = page.locator('[data-tutorial-overlay]')
  await expect(page.locator('#app')).toHaveAttribute('inert', '')
  await page.keyboard.press('Shift+Tab')
  await expect(tutorialOverlay.getByRole('button', { name: 'Next' })).toBeFocused()
  await page.keyboard.press('Tab')
  const skipTutorial = tutorialOverlay.getByRole('button', { name: 'Skip tutorial' })
  await expect(skipTutorial).toBeFocused()
  const nextTutorial = tutorialOverlay.getByRole('button', { name: 'Next' })
  for (let step = 1; step < 5; step += 1) {
    await nextTutorial.click()
  }
  const calibrationStep = page.locator('[data-tutorial="h2r-calibration"]')
  await expect(tutorialOverlay).toHaveAttribute(
    'data-tutorial-step',
    'calibration'
  )
  await expect(calibrationStep).toHaveAttribute('open', '')
  await skipTutorial.click()
  await expect(page.locator('#app')).not.toHaveAttribute('inert', '')
  await expect(calibrationStep).not.toHaveAttribute('open', '')
  await expect(helpTrigger).toBeFocused()

})

test('menus, settings, language and application information', async ({ page }) => {
  await skipFirstRun(page)
  const menu = page.getByRole('menubar', { name: 'Application menu' })
  await expect(menu.locator('[role="menuitem"][aria-haspopup="menu"]')).toHaveText([
    'File',
    'Workflows',
    'Analysis',
    'Settings',
    'Help'
  ])

  const fileTrigger = menu.getByRole('menuitem', { name: 'File', exact: true })
  const fileMenu = page.getByRole('menu', { name: 'File' })
  await fileTrigger.click()
  await expect(fileMenu).toBeVisible()
  await expect(fileMenu.getByRole('menuitem')).toHaveCount(7)
  for (const command of [
    'Import Motion File',
    'Import Motion Folder',
    'Import Video',
    'Import Robot URDF',
    'Import Robot Mesh Folder'
  ]) {
    await expect(fileMenu.getByRole('menuitem', { name: command, exact: true })).toBeEnabled()
  }
  await expect(fileMenu.getByRole('menuitem', { name: /^Current Result/ })).toBeDisabled()
  await expect(fileMenu.getByRole('menuitem', { name: 'Exit', exact: true })).toBeEnabled()

  const settingsTrigger = menu.getByRole('menuitem', { name: 'Settings', exact: true })
  const settingsMenu = page.getByRole('menu', { name: 'Settings' })
  await settingsTrigger.hover()
  await expect(fileMenu).toBeHidden()
  await expect(settingsMenu.getByRole('menuitem')).toHaveText(['Settings', 'Dark Mode'])
  await settingsMenu.getByRole('menuitem', { name: 'Settings', exact: true }).click()
  let settingsDialog = page.getByRole('dialog', { name: 'Workspace Settings' })
  await expect(settingsDialog).toBeVisible()
  const language = settingsDialog.getByLabel('Workspace language')
  await expect(language).toHaveValue('en')
  await language.selectOption('zh-CN')
  await expect(menu.locator('[role="menuitem"][aria-haspopup="menu"]')).toHaveText([
    '文件',
    '工作流',
    '分析',
    '设置',
    '帮助'
  ])
  const chineseInspector = page.getByRole('complementary', { name: '检查器' })
  await expect(chineseInspector.getByRole('heading', { name: '动作', exact: true })).toBeVisible()
  await expect(chineseInspector.getByText('拖入动作文件或文件夹')).toBeVisible()
  await expect(chineseInspector.getByRole('heading', { name: '资源库' })).toBeVisible()
  settingsDialog = page.getByRole('dialog', { name: '工作区设置' })
  for (const removedCopy of [
    '语言、布局、资源库与后台任务',
    '设置菜单和导航语言',
    '修改立即生效，原目录内容不会移动。',
    '· 运行中: 0 · 等待中: 0'
  ]) {
    await expect(settingsDialog).not.toContainText(removedCopy)
  }
  await settingsDialog.getByLabel('工作区语言').selectOption('en')
  settingsDialog = page.getByRole('dialog', { name: 'Workspace Settings' })
  const leftNavigation = settingsDialog.getByLabel('Show left navigation')
  const rightInspector = settingsDialog.getByLabel('Show right inspector')
  await leftNavigation.uncheck()
  await expect(page.locator('#sidebar')).toBeHidden()
  await rightInspector.uncheck()
  await expect(page.getByRole('complementary', { name: 'Inspector' })).toBeHidden()
  await leftNavigation.check()
  await rightInspector.check()
  await expect(page.locator('#sidebar')).toBeVisible()
  await expect(page.getByRole('complementary', { name: 'Inspector' })).toBeVisible()
  await expect(settingsDialog.getByRole('button', { name: 'Choose directory' })).toBeEnabled()
  await expect(settingsDialog.getByRole('button', { name: 'Refresh settings' })).toBeEnabled()
  await expect(settingsDialog.getByLabel('Maximum running jobs')).toBeEnabled()
  await expect(settingsDialog.getByLabel('Maximum queued jobs')).toBeEnabled()
  const forceReanalysis = settingsDialog.getByLabel('Force re-analysis')
  await expect(forceReanalysis).not.toBeChecked()
  await forceReanalysis.check()
  await expect(forceReanalysis).toBeChecked()
  await expect.poll(() =>
    page.evaluate(() => localStorage.getItem('hhtools.analysis.force-reanalysis'))
  ).toBe('true')
  await expect(settingsDialog.getByRole('button', { name: 'Save' })).toBeEnabled()
  await expect(settingsDialog).not.toContainText(
    'Language, layout, libraries, and background jobs'
  )
  await expect(settingsDialog).not.toContainText('Menus and navigation language')
  await expect(settingsDialog).not.toContainText(
    'Changes apply immediately; existing files are not moved.'
  )
  await expect(settingsDialog).not.toContainText(/Running:\s*0.*Queued:\s*0/)
  await settingsDialog.getByRole('button', { name: 'Close' }).click()
  await expect(settingsDialog).toBeHidden()
  await settingsTrigger.hover()
  await settingsMenu.getByRole('menuitem', { name: 'Settings', exact: true }).click()
  settingsDialog = page.getByRole('dialog', { name: 'Workspace Settings' })
  await expect(settingsDialog.getByLabel('Force re-analysis')).toBeChecked()
  await settingsDialog.getByLabel('Force re-analysis').uncheck()
  await expect.poll(() =>
    page.evaluate(() => localStorage.getItem('hhtools.analysis.force-reanalysis'))
  ).toBe('false')
  await settingsDialog.getByRole('button', { name: 'Close' }).click()

  await settingsTrigger.click()
  await settingsMenu.getByRole('menuitem', { name: 'Dark Mode', exact: true }).click()
  await expect(page.locator('#app')).toHaveAttribute('data-theme', 'dark')
  await settingsTrigger.click()
  await settingsMenu.getByRole('menuitem', { name: 'Light Mode', exact: true }).click()
  await expect(page.locator('#app')).toHaveAttribute('data-theme', 'light')

  const helpTrigger = menu.getByRole('menuitem', { name: 'Help', exact: true })
  const helpMenu = page.getByRole('menu', { name: 'Help' })
  await helpTrigger.hover()
  await helpMenu.getByRole('menuitem', { name: 'About hhtools' }).click()
  const aboutDialog = page.getByRole('dialog', { name: 'Human-Humanoid Tools' })
  await expect(aboutDialog).toBeVisible()
  await expect(aboutDialog).toContainText('Humanoid motion retargeting and dataset analysis')
  await expect(aboutDialog).toContainText('Jagger Shen, Nora Sun and hhtools contributors')
  await expect(aboutDialog).toContainText('2026')
  await expect(aboutDialog).toContainText('Apache-2.0')
  await expect(aboutDialog.getByRole('link', { name: 'github.com/Roboparty/human-humanoid-tools' }))
    .toHaveAttribute(
      'href',
      'https://github.com/Roboparty/human-humanoid-tools'
    )
  await expect(aboutDialog.getByRole('link', { name: 'shenyaojie@roboparty.com' }))
    .toHaveAttribute('href', 'mailto:shenyaojie@roboparty.com')
  await expect(aboutDialog.getByRole('link', { name: 'sunlancheng@roboparty.com' }))
    .toHaveAttribute('href', 'mailto:sunlancheng@roboparty.com')
  await aboutDialog.getByRole('button', { name: 'Close' }).click()

  const workflowsMenu = page.getByRole('menu', { name: 'Workflows' })
  const workflowsTrigger = menu.getByRole('menuitem', { name: 'Workflows', exact: true })
  await workflowsTrigger.hover()
  await expect(workflowsMenu).toBeVisible()
  await expect(workflowsMenu.getByRole('menuitem')).toHaveText([
    'Video to Motion',
    'Human to Robot',
    'Robot to Robot',
    'Batch'
  ])
  const workflowsBounds = await workflowsMenu.boundingBox()
  if (!workflowsBounds) throw new Error('Workflows menu has no layout box')
  await page.mouse.move(workflowsBounds.x + 12, workflowsBounds.y - 1.5)
  await expect(workflowsMenu).toBeVisible()
  await workflowsMenu.getByRole('menuitem', { name: 'Video to Motion' }).hover()
  await expect(workflowsMenu).toBeVisible()
  await menu.getByRole('menuitem', { name: 'Analysis', exact: true }).hover()
  await expect(workflowsMenu).toBeHidden()
  await expect(page.getByRole('menu', { name: 'Analysis' })).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('menu', { name: 'Analysis' })).toBeHidden()
  const activeViewBeforeShortcut = await page.locator('#app').getAttribute('data-active-view')
  await page.keyboard.press('Alt+3')
  await expect(page.locator('#app')).toHaveAttribute(
    'data-active-view',
    activeViewBeforeShortcut ?? ''
  )


})

test('motion, robot and video import controls', async ({ page }, testInfo) => {
  await skipFirstRun(page)
  const sidebar = page.getByRole('complementary', { name: 'Workspace navigation' })
  await expect(sidebar.getByRole('button')).toHaveText([
    'Motion',
    'Robot',
    'Video → Motion',
    'Human → Robot',
    'Robot → Robot',
    'Batch',
    'Data Analysis'
  ])
  await expect(sidebar.locator('.sidebar-icon')).toHaveCount(7)
  expect(
    await sidebar.locator('.sidebar-icon').evaluateAll((icons) =>
      icons.every((icon) => getComputedStyle(icon).maskImage.includes('/icons/sidebar/'))
    )
  ).toBe(true)
  const motionButton = sidebar.getByRole('button', { name: 'Motion', exact: true })
  await motionButton.click()
  await expect(motionButton).toHaveAttribute(
    'aria-current',
    'page'
  )

  const inspector = page.getByRole('complementary', { name: 'Inspector' })
  await expect(inspector.getByRole('heading', { name: 'Motion', exact: true })).toBeVisible()
  await expect(inspector.getByRole('heading', { name: 'Library' })).toBeVisible()
  const profilePicker = inspector.getByRole('radiogroup', { name: 'Motion import type' })
  await expect(profilePicker.getByRole('radio')).toHaveText([
    'mimic',
    'intermimic',
    'meshmimic'
  ])
  await expect(
    profilePicker.getByRole('radio', { name: 'mimic', exact: true })
  ).toBeChecked()
  await expect(inspector.getByText('Drop a motion file or folder')).toBeVisible()
  expect((await inspector.boundingBox())?.width).toBeCloseTo(360, 0)
  const motionRefresh = inspector.getByRole('button', { name: 'Refresh Motion Library' })
  await expectBorderless(motionRefresh)
  await expect(inspector.getByRole('searchbox', { name: 'Search the Motion Library' }))
    .toBeVisible()
  await expect(inspector.getByLabel('Motion library category')).toBeVisible()
  const setDirectory = inspector.getByRole('button', { name: 'Set directory' })
  await expect(setDirectory).toBeEnabled()
  await expect(inspector.getByRole('button', { name: 'Choose library directory' }))
    .toHaveCount(0)
  await expect(inspector.getByRole('button', { name: 'Remove folder' })).toHaveCount(0)
  await expect(inspector.getByLabel('Managed Motion Library folder')).toHaveCount(0)
  await expect(inspector.locator('button[aria-label$="to H2R Batch"]')).toHaveCount(0)
  await setDirectory.click()
  const settingsDialog = page.getByRole('dialog', { name: 'Workspace Settings' })
  await expect(settingsDialog).toBeVisible()
  await settingsDialog.getByRole('button', { name: 'Close' }).click()
  await profilePicker.getByRole('radio', { name: 'intermimic', exact: true }).click()
  await expect(
    inspector.getByText('Drop an object-interaction motion folder')
  ).toBeVisible()
  await expect(inspector.getByRole('button', { name: 'Choose file' })).toHaveCount(0)

  await sidebar.getByRole('button', { name: 'Robot', exact: true }).click()
  await expect(inspector.getByRole('heading', { name: 'Robot', exact: true })).toBeVisible()
  await expect(inspector.getByRole('group', { name: 'URDF import area' })).toBeVisible()
  await expect(inspector.getByRole('group', { name: 'Robot mesh import area' })).toBeVisible()
  const chooseMeshFolder = inspector.getByRole('button', { name: 'Choose mesh folder' })
  await expect(chooseMeshFolder).toBeEnabled()
  await expect(inspector.getByText('No URDF selected.')).toBeVisible()
  const meshFolderInput = inspector
    .getByRole('group', { name: 'Robot mesh import area' })
    .locator('input[type="file"][webkitdirectory]')
  await expect(meshFolderInput).toHaveCount(1)
  await expect(meshFolderInput).toHaveAttribute('multiple', '')
  const meshFolder = testInfo.outputPath('robot-meshes')
  await mkdir(meshFolder, { recursive: true })
  await writeFile(join(meshFolder, 'body.stl'), 'solid body\nendsolid body\n')
  await meshFolderInput.setInputFiles(meshFolder)
  await expect(inspector.getByText('1 mesh asset · choose the .urdf file')).toBeVisible()
  await expectBorderless(inspector.getByRole('button', { name: 'Refresh Robot Library' }))
  await expect(inspector.getByRole('heading', { name: 'Robot Library' })).toBeVisible()
  const robotRows = inspector
    .getByRole('list', { name: 'Robot models' })
    .locator('button[aria-label^="Load robot "]')
  expect(
    await robotRows.evaluateAll((rows) =>
      rows.every((row) => !/\bLoad(?:ed)?\b/.test(row.textContent ?? ''))
    )
  ).toBe(true)

  await sidebar.getByRole('button', { name: 'Video → Motion' }).click()
  await expect(inspector.getByRole('heading', { name: 'Video → Motion' })).toBeVisible()
  await expect(page.getByRole('list', { name: 'Video to Motion pipeline' })).toBeVisible()
  await expect(
    inspector.locator('section[aria-label="Video → Motion"] > div > details')
  ).toHaveCount(4)
  const v2mPage = inspector.locator('section[aria-label="Video → Motion"]')
  const v2mSteps = v2mPage.locator(':scope > div > details')
  await expect(v2mSteps.nth(0).locator('[data-status-tone]')).toHaveAttribute(
    'data-status-tone',
    'neutral'
  )
  const invalidVideo = testInfo.outputPath('not-a-video.txt')
  await writeFile(invalidVideo, 'not a video')
  await v2mPage
    .getByRole('group', { name: 'Video import area' })
    .locator('input[type="file"]')
    .setInputFiles(invalidVideo)
  const invalidVideoStatus = v2mSteps.nth(0).locator('[data-status-tone]')
  await expect(invalidVideoStatus).toHaveAttribute('data-status-tone', 'danger')
  await expect(invalidVideoStatus).toHaveClass(/text-danger/)
  await expect(invalidVideoStatus).toHaveText('Invalid video')
  await expect(inspector.getByRole('group', { name: 'Video import area' })).toBeVisible()
  await expectBorderless(inspector.getByRole('button', { name: 'Refresh GVHMR status' }))


})

test('retarget, batch and analysis workflow navigation', async ({ page }) => {
  await skipFirstRun(page)
  const sidebar = page.getByRole('complementary', { name: 'Workspace navigation' })
  const inspector = page.getByRole('complementary', { name: 'Inspector' })
  await sidebar.getByRole('button', { name: 'Human → Robot' }).click()
  await expect(page.locator('#app')).toHaveAttribute('data-active-view', 'h2r')
  await expect(sidebar.getByRole('button', { name: 'Human → Robot' })).toHaveAttribute(
    'aria-current',
    'page'
  )
  await expect(inspector.getByRole('heading', { name: 'Human → Robot' })).toBeVisible()
  await expect(page.getByRole('list', { name: 'Human to Robot pipeline' })).toBeVisible()
  const h2rPage = inspector.locator('section[aria-label="Human → Robot"]')
  await expect(h2rPage.locator('details')).toHaveCount(4)
  expect(
    await h2rPage.locator('[data-status-tone]').evaluateAll((statuses) =>
      statuses.map((status) => status.getAttribute('data-status-tone'))
    )
  ).toEqual(['neutral', 'neutral', 'neutral', 'neutral'])
  await expect(inspector.getByLabel('Select human motion')).toBeVisible()
  await expect(inspector.getByRole('button', { name: 'Load motion' })).toBeDisabled()
  await expect(h2rPage.getByRole('button', { name: 'Import motion' })).toBeVisible()
  await h2rPage.getByRole('button', { name: 'Import motion' }).click()
  await expect(page.locator('#app')).toHaveAttribute('data-active-view', 'motion')
  await sidebar.getByRole('button', { name: 'Human → Robot' }).click()
  await h2rPage.locator('details').nth(1).locator('summary').click()
  await expect(h2rPage.getByRole('button', { name: 'Import robot' })).toBeVisible()
  await h2rPage.getByRole('button', { name: 'Import robot' }).click()
  await expect(page.locator('#app')).toHaveAttribute('data-active-view', 'robot-assets')
  await sidebar.getByRole('button', { name: 'Human → Robot' }).click()
  await expect(inspector.getByRole('heading', { name: 'Motion', exact: true })).toHaveCount(0)

  await sidebar.getByRole('button', { name: 'Robot → Robot' }).click()
  await expect(inspector.getByRole('heading', { name: 'Robot → Robot' })).toBeVisible()
  await expect(page.getByRole('list', { name: 'Robot to Robot pipeline' })).toBeVisible()
  const r2rPage = inspector.locator('section[aria-label="Robot → Robot"]')
  await expect(r2rPage.locator('details')).toHaveCount(5)
  expect(
    await r2rPage.locator('[data-status-tone]').evaluateAll((statuses) =>
      statuses.map((status) => status.getAttribute('data-status-tone'))
    )
  ).toEqual(['neutral', 'neutral', 'neutral', 'neutral', 'neutral'])
  const r2rSourceStep = r2rPage.locator('details').first()
  await expect(r2rSourceStep.getByLabel('Select source robot')).toHaveValue('')
  await expect(r2rSourceStep.getByRole('button', { name: 'Load' })).toBeDisabled()
  await expect(r2rPage.locator('button[aria-label="Import robot"]')).toHaveCount(2)
  await r2rPage.getByRole('button', { name: 'Import robot' }).first().click()
  await expect(page.locator('#app')).toHaveAttribute('data-active-view', 'robot-assets')
  await sidebar.getByRole('button', { name: 'Robot → Robot' }).click()

  await sidebar.getByRole('button', { name: 'Batch', exact: true }).click()
  await expect(inspector.getByRole('heading', { name: 'Batch' })).toBeVisible()
  await expectBorderless(inspector.getByRole('button', { name: 'Refresh Batch catalogs' }))
  const batchMode = inspector.getByRole('radiogroup', { name: 'Batch workflow' })
  await expect(batchMode.getByRole('radio', { name: 'H2R' })).toBeChecked()
  await batchMode.getByRole('radio', { name: 'R2R' }).click()
  await expect(inspector.getByText('1. Source trajectories')).toBeVisible()

  await sidebar.getByRole('button', { name: 'Data Analysis' }).click()
  await expect(inspector.getByRole('heading', { name: 'Data Analysis' })).toBeVisible()
  const analysisPipeline = page.getByRole('list', { name: 'Data Analysis pipeline' })
  await expect(analysisPipeline).toBeVisible()
  const analysisPage = inspector.locator('section[aria-label="Data Analysis"]')
  await expect(analysisPage.locator('details')).toHaveCount(4)
  await expect(analysisPage.getByLabel('Dataset source path')).toHaveCount(0)
  await expect(analysisPage.getByText('Original source path')).toHaveCount(0)
  await expect(analysisPage.getByRole('button', { name: 'Choose folder' })).toBeEnabled()
  await expect(analysisPage.getByRole('button', { name: 'Built-in library' })).toBeEnabled()
  await expect(analysisPage.getByLabel('Ignore cache')).toHaveCount(0)
  await analysisPage.locator('details').nth(1).locator('summary').click()
  const loadExistingResult = analysisPage.getByRole('button', { name: 'Load existing result' })
  await expect(loadExistingResult).toBeEnabled()
  await expect(analysisPage.getByRole('button', { name: 'Load cached' })).toHaveCount(0)
  await expect(analysisPipeline.locator('li').nth(0)).toHaveAttribute('data-state', 'active')
  await analysisPage.getByRole('button', { name: 'Built-in library' }).click()
  await expect(analysisPipeline.locator('li').nth(0)).toHaveAttribute('data-state', 'complete')
  await expect(analysisPipeline.locator('li').nth(1)).toHaveAttribute('data-state', 'active')
  await expect(analysisPipeline.locator('li').nth(0).locator('span').last()).toHaveClass(
    /text-success/
  )
  await expect(analysisPipeline.locator('li').nth(1).locator('span').last()).toHaveClass(
    /text-primary/
  )
  await expect(page.locator('.workspace-drawer-handle, .col-resizer')).toHaveCount(0)


})

test('renders the stage and visibility controls', async ({ page }) => {
  await skipFirstRun(page)
  const stage = page.getByRole('main', { name: 'Workspace content' })
  await expect(stage.locator('[data-stage-renderer="react-three-fiber"]')).toHaveCount(1)
  const stageCanvas = stage.locator('canvas')
  await expect(stageCanvas).toHaveCount(1)
  await expect(stage.getByText('Drop a motion here to preview')).toBeVisible()
  const canvasInfo = await stageCanvas.evaluate((element) => {
    const canvas = element as HTMLCanvasElement
    const gl = canvas.getContext('webgl2') ?? canvas.getContext('webgl')
    return {
      width: canvas.width,
      height: canvas.height,
      clientWidth: canvas.clientWidth,
      clientHeight: canvas.clientHeight,
      hasWebGL: gl !== null,
    }
  })
  expect(canvasInfo.hasWebGL).toBe(true)
  expect(canvasInfo.width).toBeGreaterThan(0)
  expect(canvasInfo.height).toBeGreaterThan(0)
  expect(canvasInfo.width / canvasInfo.height).toBeCloseTo(
    canvasInfo.clientWidth / canvasInfo.clientHeight,
    1,
  )
  const stageMenu = stage.locator('[data-slot="toggle-group"][aria-label="Stage visibility"]')
  const stageToggles = stageMenu.locator('[data-slot="toggle-group-item"]')
  await expect(stageMenu).toBeVisible()
  await expect(stageToggles).toHaveText([
    'Skeleton',
    'Body',
    'Objects/Terrain',
    'Scaled',
    'Scaled',
    'Robot'
  ])
  expect(
    await stageToggles.evaluateAll((toggles) =>
      toggles.map((toggle) => toggle.getAttribute('data-family'))
    )
  ).toEqual([
    'skeleton',
    'body',
    'scene',
    'scaled-skeleton',
    'scaled-scene',
    'robot'
  ])
  const stageColors = await page.locator(':root').evaluate((root) => {
    const style = getComputedStyle(root)
    return [
      '--stage-skeleton-accent',
      '--stage-body-accent',
      '--stage-scene-accent',
      '--stage-scaled-skeleton-accent',
      '--stage-scaled-scene-accent',
      '--stage-robot-accent'
    ].map((name) => style.getPropertyValue(name).trim())
  })
  expect(stageColors.every(Boolean)).toBe(true)
  expect(new Set(stageColors).size).toBe(stageColors.length)
  expect(
    await stageToggles.evaluateAll((toggles) =>
      toggles.every((toggle) => (toggle as HTMLButtonElement).disabled)
    )
  ).toBe(true)
  await expect(stageMenu.getByRole('button', { name: 'Body', exact: true })).toHaveAttribute(
    'aria-pressed',
    'false'
  )
  await expect(stageMenu.getByRole('button', { name: 'Robot', exact: true })).toHaveAttribute(
    'aria-pressed',
    'false'
  )
  const [stageBounds, menuBounds, menuStyle, bodyStyle, eyeMask] = await Promise.all([
    stage.boundingBox(),
    stageMenu.boundingBox(),
    stageMenu.evaluate((element) => {
      const style = getComputedStyle(element)
      return {
        borderRadius: style.borderRadius,
        gap: style.gap,
        padding: style.padding
      }
    }),
    stageMenu.getByRole('button', { name: 'Body', exact: true }).evaluate((element) => {
      const style = getComputedStyle(element)
      return {
        backgroundColor: style.backgroundColor,
        fontSize: style.fontSize,
        padding: style.padding
      }
    }),
    stageMenu.locator('.stage-layer-eye').first().evaluate(
      (element) => getComputedStyle(element).maskImage
    )
  ])
  expect(menuBounds?.x).toBeCloseTo((stageBounds?.x ?? 0) + 12, 0)
  expect(menuBounds?.y).toBeCloseTo((stageBounds?.y ?? 0) + 12, 0)
  expect(menuBounds?.width).toBeGreaterThan(300)
  expect(menuBounds?.height).toBeGreaterThan(64)
  expect(menuStyle).toEqual({ borderRadius: '8px', gap: '4px', padding: '6px 8px' })
  expect(bodyStyle).toEqual({
    backgroundColor: 'rgba(0, 0, 0, 0)',
    fontSize: '12px',
    padding: '6px 12px'
  })
  expect(eyeMask).toContain('/icons/stage/eye.svg')

})
