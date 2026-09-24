import { BrowserWindow, screen, shell } from 'electron'

import type { LoggerLike } from './desktop-logger'
import { waitForWindowReadiness } from './window-readiness'
import { WindowStateStore } from './window-state-store'

export interface MainWindowResult {
  window: BrowserWindow
  readyToShow: Promise<void>
}

function isExternalHttpUrl(url: string): boolean {
  try {
    const parsed = new URL(url)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:'
  } catch {
    return false
  }
}

export function createMainWindow(options: {
  iconPath: string
  preloadPath: string
  trustedOrigin: string
  stateStore: WindowStateStore
  logger: LoggerLike
}): MainWindowResult {
  const displays = screen.getAllDisplays()
  const primary = screen.getPrimaryDisplay()
  const state = options.stateStore.load(displays, primary)

  const window = new BrowserWindow({
    x: state.x,
    y: state.y,
    width: state.width,
    height: state.height,
    minWidth: 1024,
    minHeight: 700,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#ffffff',
    icon: options.iconPath,
    title: 'Human-Humanoid Tools',
    webPreferences: {
      // The WebUI is treated as untrusted web content and reaches desktop APIs only via preload.
      preload: options.preloadPath,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true
    }
  })

  if (state.maximized) window.maximize()

  const saveState = (): void => {
    if (window.isDestroyed() || window.isMinimized()) return

    // Save normal bounds while maximized so the next restored window is not screen-sized.
    const bounds = window.isMaximized() ? window.getNormalBounds() : window.getBounds()
    options.stateStore.save({ ...bounds, maximized: window.isMaximized() })
  }
  window.on('close', saveState)

  window.webContents.setWindowOpenHandler(({ url }) => {
    // Never create arbitrary Electron child windows; normal web links belong in the OS browser.
    if (isExternalHttpUrl(url)) void shell.openExternal(url)
    return { action: 'deny' }
  })
  window.webContents.on('will-navigate', (event, url) => {
    try {
      if (new URL(url).origin === options.trustedOrigin) return
    } catch {
      // Invalid or non-network navigation is denied below.
    }
    event.preventDefault()
    if (isExternalHttpUrl(url)) void shell.openExternal(url)
  })
  window.webContents.on('render-process-gone', (_event, details) => {
    options.logger.error('Renderer process exited', {
      reason: details.reason,
      exitCode: details.exitCode
    })
  })
  window.webContents.on('unresponsive', () => options.logger.warn('Renderer became unresponsive'))
  window.webContents.on('did-fail-load', (_event, code, description, validatedUrl) => {
    options.logger.error('Renderer failed to load', { code, description, url: validatedUrl })
  })

  // `did-finish-load` is a necessary fallback on Wayland, where an initially
  // hidden BrowserWindow can finish rendering without emitting ready-to-show.
  const readyToShow = waitForWindowReadiness(window, window.webContents)
  return { window, readyToShow }
}
