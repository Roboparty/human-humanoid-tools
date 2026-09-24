/**
 * Electron main-process bootstrap.
 *
 * Startup is intentionally linear: resolve the external Python runtime, create a
 * secured localhost session, start and health-check the sidecar, load the existing
 * WebUI, then reveal the window. Shutdown follows the reverse ownership chain.
 */
import { randomBytes } from 'node:crypto'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { app, BrowserWindow, dialog, powerSaveBlocker, session } from 'electron'

import { DESKTOP_CHANNELS } from '../shared/desktop-api'
import type { RuntimeState } from '../shared/runtime-state'
import { AppLifecycle } from './app-lifecycle'
import { withAppActivity } from './app-activity'
import { DesktopTutorialState } from './desktop-tutorial-state'
import { DesktopLogger } from './desktop-logger'
import { diagnosticsDataUrl } from './diagnostics-page'
import { installerDataUrl } from './installer-page'
import {
  configureGraphicsCommandLine,
  decideGraphicsStartup,
  softwareRenderingRelaunchArgs
} from './graphics-mode'
import { registerDesktopHandlers } from './ipc/register-desktop-handlers'
import { registerInstallerHandlers } from './ipc/register-installer-handlers'
import { createMainWindow } from './main-window'
import { findAvailablePort } from './network'
import { OptionalComponentStore, runGvhmrSetup } from './optional-components'
import { RuntimeInstaller } from './runtime-installer'
import {
  buildSidecarEnvironment,
  resolveRuntime,
  RuntimeNotFoundError,
  type RuntimeConfig
} from './runtime-resolver'
import { configureDesktopSession } from './security/configure-session'
import { SidecarSupervisor, type SidecarSnapshot } from './sidecar-supervisor'
import { WindowStateStore } from './window-state-store'

const lifecycle = new AppLifecycle()
// Chromium graphics switches are immutable after `ready`, so configure the
// second (software-rendered) launch as soon as Electron is imported.
const softwareRendering = configureGraphicsCommandLine(app.commandLine)

interface GraphicsProbeResult {
  webgl2: boolean
  renderer?: string
  error?: string
}

let graphicsProbe: GraphicsProbeResult | undefined

/** Probe WebGL2 before Python or the real Three.js WebUI starts. */
async function probeWebGL2(): Promise<GraphicsProbeResult> {
  const probeWindow = new BrowserWindow({
    width: 16,
    height: 16,
    show: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true
    }
  })

  try {
    await probeWindow.loadURL('data:text/html;charset=utf-8,<title>hhtools graphics probe</title>')
    const result: unknown = await probeWindow.webContents.executeJavaScript(`(() => {
      const canvas = document.createElement('canvas')
      const context = canvas.getContext('webgl2', { failIfMajorPerformanceCaveat: false })
      if (context === null) return { webgl2: false }

      const debugInfo = context.getExtension('WEBGL_debug_renderer_info')
      const renderer = debugInfo === null
        ? context.getParameter(context.RENDERER)
        : context.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
      return { webgl2: true, renderer: String(renderer) }
    })()`, true)

    if (
      typeof result === 'object' &&
      result !== null &&
      'webgl2' in result &&
      typeof result.webgl2 === 'boolean'
    ) {
      return {
        webgl2: result.webgl2,
        renderer: 'renderer' in result && typeof result.renderer === 'string'
          ? result.renderer
          : undefined
      }
    }
    return { webgl2: false, error: 'The graphics probe returned an invalid result.' }
  } catch (reason) {
    return {
      webgl2: false,
      error: reason instanceof Error ? reason.message : String(reason)
    }
  } finally {
    if (!probeWindow.isDestroyed()) probeWindow.destroy()
  }
}

async function prepareDesktopGraphics(): Promise<boolean> {
  graphicsProbe = await probeWebGL2()
  const action = decideGraphicsStartup(graphicsProbe.webgl2, softwareRendering)
  if (action === 'start') return true

  if (action === 'relaunch') {
    console.warn('WebGL2 is unavailable; relaunching hhtools with software rendering.')
    app.relaunch({ args: softwareRenderingRelaunchArgs() })
    app.exit(0)
    return false
  }

  const detail = graphicsProbe.error === undefined ? '' : ` ${graphicsProbe.error}`
  throw new Error(
    `WebGL2 is unavailable${softwareRendering ? ' with the bundled software renderer' : ''}.${detail}`
  )
}

function desktopIconPath(): string {
  return join(app.getAppPath(), 'resources', 'icon.png')
}

// These references are process-wide because requestSingleInstanceLock() guarantees one owner.
let mainWindow: BrowserWindow | undefined
let supervisor: SidecarSupervisor | undefined
let logger: DesktopLogger | undefined
let runtime: RuntimeConfig | undefined
let backendOrigin: string | undefined
let allowQuit = false
let shutdownPromise: Promise<void> | undefined
let crashDialogOpen = false
let optionalComponents: OptionalComponentStore | undefined
let installerCleanup: (() => void) | undefined

function desktopPreloadPath(): string {
  return join(dirname(fileURLToPath(import.meta.url)), '../preload/index.cjs')
}

function installerPreloadPath(): string {
  return join(dirname(fileURLToPath(import.meta.url)), '../preload/installer.cjs')
}

function runtimeState(snapshot: SidecarSnapshot | undefined = supervisor?.snapshot): RuntimeState {
  return {
    appPhase: lifecycle.phase,
    backendState: snapshot?.state ?? 'stopped',
    backendOrigin,
    backendPid: snapshot?.pid,
    error: snapshot?.error
  }
}

function broadcastRuntimeState(snapshot?: SidecarSnapshot): void {
  if (mainWindow !== undefined && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(DESKTOP_CHANNELS.runtimeStateChanged, runtimeState(snapshot))
  }
}

async function offerCrashRecovery(snapshot: SidecarSnapshot): Promise<void> {
  if (
    crashDialogOpen ||
    lifecycle.phase === 'shutting-down' ||
    mainWindow === undefined ||
    mainWindow.isDestroyed()
  ) {
    return
  }
  crashDialogOpen = true
  try {
    const result = await dialog.showMessageBox(mainWindow, {
      type: 'error',
      title: 'Human-Humanoid Tools backend stopped',
      message: 'The local Python backend stopped unexpectedly.',
      detail: snapshot.error ?? 'See the desktop log for details.',
      buttons: ['Restart backend', 'Close'],
      defaultId: 0,
      cancelId: 1
    })
    if (result.response === 0 && supervisor !== undefined) await supervisor.restart()
  } finally {
    crashDialogOpen = false
  }
}

async function startDesktop(): Promise<void> {
  app.setAppUserModelId('com.roboparty.hhtools.desktop.alpha')
  const userData = app.getPath('userData')

  optionalComponents = new OptionalComponentStore({
    userData,
    localAppData: process.env.LOCALAPPDATA,
    env: process.env,
  })
  runtime = resolveRuntime({
    appPath: app.getAppPath(),
    cwd: process.cwd(),
    userData,
    isPackaged: app.isPackaged,
    resourcesPath: process.resourcesPath,
    appVersion: app.getVersion(),
  })
  logger = new DesktopLogger(runtime.logDirectory)
  logger.info('Desktop startup began', {
    runtimeKind: runtime.kind,
    repoRoot: runtime.repoRoot,
    packaged: app.isPackaged
  })
  logger.info('Graphics preflight passed', {
    renderer: graphicsProbe?.renderer,
    softwareRendering
  })

  const sidecarEnvironment = (): NodeJS.ProcessEnv => ({
    ...buildSidecarEnvironment(
      runtime!.repoRoot,
      process.env,
      runtime!.bodyModelsRoot,
      runtime!.bundledRobotRoot,
      runtime!.kind === 'checkout'
    ),
    // A user-selected optional-component path must win over checkout defaults.
    ...optionalComponents?.sidecarEnvironment(),
  })

  const port = await findAvailablePort()
  const secret = randomBytes(32).toString('hex')
  backendOrigin = `http://127.0.0.1:${port}`

  // Electron injects this per-launch secret below the renderer boundary. WebUI code never sees it.
  configureDesktopSession(session.defaultSession, backendOrigin, secret)

  const preloadPath = desktopPreloadPath()
  const stateStore = new WindowStateStore(join(userData, 'window-state.json'))
  const tutorialState = new DesktopTutorialState(userData)
  const windowResult = createMainWindow({
    iconPath: desktopIconPath(),
    preloadPath,
    trustedOrigin: backendOrigin,
    stateStore,
    logger
  })
  mainWindow = windowResult.window

  supervisor = new SidecarSupervisor(
    {
      command: runtime.pythonExecutable,
      args: [
        '-m',
        'hhtools.cli.desktop_sidecar',
        '--source',
        runtime.sourceRoot,
        '--save-dir',
        runtime.saveDirectory,
        '--cache',
        runtime.cacheDirectory,
        '--host',
        '127.0.0.1',
        '--port',
        String(port)
      ],
      cwd: runtime.workingDirectory,
      env: {
        ...sidecarEnvironment(),
        // Use the environment rather than argv so the secret is absent from process listings.
        HHTOOLS_DESKTOP_SESSION_SECRET: secret
      },
      origin: backendOrigin,
      sessionSecret: secret
    },
    logger
  )

  const removeStateListener = supervisor.onStateChange((snapshot) => {
    broadcastRuntimeState(snapshot)
    if (snapshot.state === 'crashed' && lifecycle.phase === 'after-window-open') {
      void offerCrashRecovery(snapshot)
    }
  })

  // Every resource owned by Main registers one cleanup hook in the same shutdown coordinator.
  lifecycle.registerShutdownJoiner('runtime-state-listener', removeStateListener)
  lifecycle.registerShutdownJoiner('python-sidecar', () => supervisor?.stop())

  const removeDesktopHandlers = registerDesktopHandlers({
    mainWindow,
    trustedOrigin: backendOrigin,
    getRuntimeState: () => runtimeState(),
    getOptionalComponents: () => optionalComponents!.getState(),
    hasSeenTutorial: () => tutorialState.hasSeenTutorial(),
    markTutorialSeen: () => tutorialState.markTutorialSeen(),
    restartBackend: async () => {
      if (lifecycle.phase === 'shutting-down' || supervisor === undefined) {
        throw new Error('The application is shutting down')
      }
      const snapshot = await supervisor.restart()
      return runtimeState(snapshot)
    },
    setupGvhmr: () => runGvhmrSetup({
      mainWindow: mainWindow!,
      store: optionalComponents!,
      onConfigured: async () => {
        if (supervisor === undefined) return
        supervisor.updateEnvironment({
          ...sidecarEnvironment(),
          HHTOOLS_DESKTOP_SESSION_SECRET: secret,
        })
        await supervisor.restart()
      },
    }),
  })
  lifecycle.registerShutdownJoiner('desktop-ipc', removeDesktopHandlers)

  lifecycle.transition('backend-starting')
  broadcastRuntimeState()

  // start() resolves only after the authenticated /api/health endpoint answers successfully.
  await supervisor.start()
  lifecycle.transition('ready')
  broadcastRuntimeState()

  await mainWindow.loadURL(backendOrigin)
  await windowResult.readyToShow

  // Keeping the window hidden until backend and renderer are ready avoids a blank or error flash.
  if (!mainWindow.isDestroyed()) mainWindow.show()
  lifecycle.transition('after-window-open')
  broadcastRuntimeState()
  logger.info('Desktop window opened', { origin: backendOrigin })
}

async function showStartupFailure(reason: unknown): Promise<void> {
  if (
    reason instanceof RuntimeNotFoundError &&
    app.isPackaged &&
    (process.platform === 'linux' || process.platform === 'darwin')
  ) {
    await showRuntimeInstaller(reason)
    return
  }
  const message = reason instanceof Error ? reason.message : String(reason)
  logger?.error('Desktop startup failed', { error: message })

  if (mainWindow === undefined || mainWindow.isDestroyed()) {
    mainWindow = new BrowserWindow({
      width: 760,
      height: 520,
      show: false,
      autoHideMenuBar: true,
      icon: desktopIconPath(),
      webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true }
    })
  }
  await mainWindow.loadURL(
    diagnosticsDataUrl({
      title: 'Human-Humanoid Tools could not start',
      message,
      stage: lifecycle.phase,
      logPath: logger?.filePath,
      pythonPath: runtime?.pythonExecutable
    })
  )
  if (!mainWindow.isDestroyed()) mainWindow.show()
}

async function showRuntimeInstaller(reason: RuntimeNotFoundError): Promise<void> {
  const window = new BrowserWindow({
    width: 860,
    height: 720,
    minWidth: 700,
    minHeight: 620,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#191d24',
    icon: desktopIconPath(),
    title: 'Set up Human-Humanoid Tools',
    webPreferences: {
      preload: installerPreloadPath(),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true
    }
  })
  mainWindow = window
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
  window.webContents.on('will-navigate', (event) => event.preventDefault())

  const installer = new RuntimeInstaller({
    scriptPath: join(process.resourcesPath, 'bootstrap', 'install.sh')
  })
  installerCleanup = registerInstallerHandlers({
    window,
    installer,
    activityBlocker: process.platform === 'darwin' ? powerSaveBlocker : undefined,
    onInstalled: () => {
      installerCleanup?.()
      installerCleanup = undefined
      allowQuit = true
      app.relaunch()
      app.exit(0)
    }
  })
  lifecycle.registerShutdownJoiner('runtime-installer', () => installerCleanup?.())
  window.once('closed', () => {
    installerCleanup?.()
    installerCleanup = undefined
  })

  await window.loadURL(
    installerDataUrl({
      version: app.getVersion(),
      reason: reason.message,
      allowSystemInstall: process.platform === 'linux'
    })
  )
  if (!window.isDestroyed()) window.show()
}

async function shutdown(): Promise<void> {
  // Electron can emit before-quit more than once; all callers share one shutdown operation.
  if (shutdownPromise !== undefined) return shutdownPromise
  shutdownPromise = (async () => {
    const result = await lifecycle.runShutdownJoiners(7_000)
    if (result.timedOut) logger?.warn('Desktop shutdown timed out')
    for (const failure of result.failures) {
      logger?.error('Shutdown joiner failed', {
        name: failure.name,
        error: failure.reason instanceof Error ? failure.reason.message : String(failure.reason)
      })
    }
    logger?.info('Desktop shutdown complete')
    await logger?.close()
    allowQuit = true
    app.quit()
  })()
  return shutdownPromise
}

const hasSingleInstanceLock = app.requestSingleInstanceLock()
if (!hasSingleInstanceLock) {
  app.quit()
} else {
  app.on('second-instance', () => {
    if (mainWindow === undefined || mainWindow.isDestroyed()) return
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.show()
    mainWindow.focus()
  })

  app.whenReady()
    .then(() => withAppActivity(process.platform === 'darwin' ? powerSaveBlocker : undefined, async () => {
      if (!(await prepareDesktopGraphics())) return
      await startDesktop()
    }))
    .catch((reason: unknown) => void showStartupFailure(reason))

  app.on('window-all-closed', () => {
    // Destroying the temporary WebGL probe leaves no windows as well. Do not
    // turn that expected preflight event into an application shutdown.
    if (mainWindow !== undefined) app.quit()
  })
  app.on('before-quit', (event) => {
    if (allowQuit) return

    // Delay the actual quit until the sidecar, IPC handlers, and log stream are closed.
    event.preventDefault()
    void shutdown()
  })
}
