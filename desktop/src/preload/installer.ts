import { contextBridge, ipcRenderer } from 'electron'

import {
  INSTALLER_CHANNELS,
  type HHToolsInstallerApi,
  type RuntimeInstallProgress
} from '../shared/installer-api'

const installerApi: HHToolsInstallerApi = {
  start: (mode) => ipcRenderer.invoke(INSTALLER_CHANNELS.start, mode),
  cancel: async () => {
    await ipcRenderer.invoke(INSTALLER_CHANNELS.cancel)
  },
  onProgress: (listener: (progress: RuntimeInstallProgress) => void) => {
    const wrapped = (
      _event: Electron.IpcRendererEvent,
      progress: RuntimeInstallProgress
    ): void => listener(progress)
    ipcRenderer.on(INSTALLER_CHANNELS.progress, wrapped)
    return () => ipcRenderer.removeListener(INSTALLER_CHANNELS.progress, wrapped)
  }
}

contextBridge.exposeInMainWorld('hhtoolsInstaller', installerApi)
