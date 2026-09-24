import { contextBridge, ipcRenderer } from 'electron'

import { DESKTOP_CHANNELS, type HHToolsDesktopApi } from '../shared/desktop-api'
import type { RuntimeState } from '../shared/runtime-state'

// Expose named operations only. The renderer never receives raw ipcRenderer or Node primitives.
const desktopApi: HHToolsDesktopApi = {
  getRuntimeState: () => ipcRenderer.invoke(DESKTOP_CHANNELS.getRuntimeState),
  getOptionalComponents: () => ipcRenderer.invoke(DESKTOP_CHANNELS.getOptionalComponents),
  restartBackend: () => ipcRenderer.invoke(DESKTOP_CHANNELS.restartBackend),
  setupGvhmr: () => ipcRenderer.invoke(DESKTOP_CHANNELS.setupGvhmr),
  selectDirectory: () => ipcRenderer.invoke(DESKTOP_CHANNELS.selectDirectory),
  openExternal: (url: string) => ipcRenderer.invoke(DESKTOP_CHANNELS.openExternal, url),
  exitApplication: async () => {
    await ipcRenderer.invoke(DESKTOP_CHANNELS.exitApplication)
  },
  hasSeenTutorial: () => ipcRenderer.invoke(DESKTOP_CHANNELS.hasSeenTutorial),
  markTutorialSeen: async () => {
    await ipcRenderer.invoke(DESKTOP_CHANNELS.markTutorialSeen)
  },
  onRuntimeStateChanged: (listener: (state: RuntimeState) => void) => {
    const wrapped = (_event: Electron.IpcRendererEvent, state: RuntimeState): void => listener(state)
    ipcRenderer.on(DESKTOP_CHANNELS.runtimeStateChanged, wrapped)
    return () => ipcRenderer.removeListener(DESKTOP_CHANNELS.runtimeStateChanged, wrapped)
  }
}

contextBridge.exposeInMainWorld('hhtoolsDesktop', desktopApi)
