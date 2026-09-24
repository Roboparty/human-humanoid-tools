import type { EventEmitter } from 'node:events'

/**
 * Resolve when Chromium says the window can paint or when the document has
 * loaded. Some Wayland compositors do not emit Electron's `ready-to-show` for
 * an initially hidden window, even though `did-finish-load` has fired.
 */
export function waitForWindowReadiness(
  windowEvents: EventEmitter,
  webContentsEvents: EventEmitter
): Promise<void> {
  return new Promise((resolve) => {
    let settled = false

    const finish = (): void => {
      if (settled) return
      settled = true
      windowEvents.removeListener('ready-to-show', finish)
      webContentsEvents.removeListener('did-finish-load', finish)
      resolve()
    }

    windowEvents.once('ready-to-show', finish)
    webContentsEvents.once('did-finish-load', finish)
  })
}
