import { ipcMain, type BrowserWindow, type IpcMainInvokeEvent } from 'electron'

import {
  INSTALLER_CHANNELS,
  type RuntimeInstallMode,
  type RuntimeInstallProgress,
  type RuntimeInstallResult
} from '../../shared/installer-api'
import type { RuntimeInstaller } from '../runtime-installer'
import { withAppActivity, type ActivityBlocker } from '../app-activity'

function assertInstallerSender(event: IpcMainInvokeEvent, window: BrowserWindow): void {
  if (window.isDestroyed() || event.sender !== window.webContents) {
    throw new Error('Rejected installer IPC from an unknown WebContents')
  }
}

export function registerInstallerHandlers(options: {
  window: BrowserWindow
  installer: RuntimeInstaller
  activityBlocker?: ActivityBlocker
  onInstalled(): void
}): () => void {
  let restartTimer: NodeJS.Timeout | undefined
  const publish = (progress: RuntimeInstallProgress): void => {
    if (!options.window.isDestroyed() && !options.window.webContents.isDestroyed()) {
      options.window.webContents.send(INSTALLER_CHANNELS.progress, progress)
    }
  }

  const start = async (
    event: IpcMainInvokeEvent,
    mode: RuntimeInstallMode
  ): Promise<RuntimeInstallResult> => {
    assertInstallerSender(event, options.window)
    if (mode !== 'user' && mode !== 'system') throw new Error('Invalid installation mode')
    if (restartTimer !== undefined) throw new Error('Runtime installation is already complete')
    const result = await withAppActivity(options.activityBlocker, () => options.installer.start(mode))
    if (result.status === 'completed') restartTimer = setTimeout(options.onInstalled, 500)
    return result
  }

  const cancel = (event: IpcMainInvokeEvent): void => {
    assertInstallerSender(event, options.window)
    options.installer.cancel()
  }

  ipcMain.handle(INSTALLER_CHANNELS.start, start)
  ipcMain.handle(INSTALLER_CHANNELS.cancel, cancel)
  options.installer.setProgressPublisher(publish)

  return () => {
    if (restartTimer !== undefined) {
      clearTimeout(restartTimer)
      restartTimer = undefined
    }
    options.installer.setProgressPublisher(undefined)
    options.installer.cancel()
    ipcMain.removeHandler(INSTALLER_CHANNELS.start)
    ipcMain.removeHandler(INSTALLER_CHANNELS.cancel)
  }
}
