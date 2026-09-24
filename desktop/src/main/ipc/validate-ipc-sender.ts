import type { BrowserWindow, IpcMainInvokeEvent } from 'electron'

export function assertTrustedIpcSender(
  event: IpcMainInvokeEvent,
  mainWindow: BrowserWindow,
  trustedOrigin: string
): void {
  // IPC channel names are not an authorization boundary: validate the owning WebContents too.
  if (mainWindow.isDestroyed() || event.sender !== mainWindow.webContents) {
    throw new Error('Rejected IPC from an unknown WebContents')
  }
  if (event.senderFrame === null || event.senderFrame !== event.sender.mainFrame) {
    // A compromised iframe must not inherit the main frame's desktop privileges.
    throw new Error('Rejected IPC from a child frame')
  }

  let senderOrigin: string
  try {
    senderOrigin = new URL(event.senderFrame.url).origin
  } catch {
    throw new Error('Rejected IPC with an invalid sender URL')
  }
  if (senderOrigin !== trustedOrigin) {
    throw new Error('Rejected IPC from an untrusted origin')
  }
}
