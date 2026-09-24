import type { BrowserWindow, IpcMainInvokeEvent } from 'electron'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const electronMocks = vi.hoisted(() => ({
  handlers: new Map<string, (event: IpcMainInvokeEvent, ...args: unknown[]) => unknown>(),
  handle: vi.fn(),
  removeHandler: vi.fn(),
  showOpenDialog: vi.fn(),
  openExternal: vi.fn()
}))

const securityMocks = vi.hoisted(() => ({
  assertTrustedIpcSender: vi.fn()
}))

vi.mock('electron', () => ({
  dialog: { showOpenDialog: electronMocks.showOpenDialog },
  ipcMain: {
    handle: electronMocks.handle,
    removeHandler: electronMocks.removeHandler
  },
  shell: { openExternal: electronMocks.openExternal }
}))

vi.mock('../src/main/ipc/validate-ipc-sender', () => ({
  assertTrustedIpcSender: securityMocks.assertTrustedIpcSender
}))

import { registerDesktopHandlers } from '../src/main/ipc/register-desktop-handlers'
import { DESKTOP_CHANNELS } from '../src/shared/desktop-api'

describe('registerDesktopHandlers', () => {
  beforeEach(() => {
    electronMocks.handlers.clear()
    vi.clearAllMocks()
    electronMocks.handle.mockImplementation((channel, handler) => {
      electronMocks.handlers.set(channel, handler)
    })
  })

  function register(): {
    event: IpcMainInvokeEvent
    mainWindow: BrowserWindow
    closeWindow: ReturnType<typeof vi.fn>
    hasSeenTutorial: ReturnType<typeof vi.fn>
    markTutorialSeen: ReturnType<typeof vi.fn>
  } {
    const event = {} as IpcMainInvokeEvent
    const closeWindow = vi.fn()
    const hasSeenTutorial = vi.fn(() => false)
    const markTutorialSeen = vi.fn()
    const mainWindow = { close: closeWindow } as unknown as BrowserWindow
    registerDesktopHandlers({
      mainWindow,
      trustedOrigin: 'http://127.0.0.1:43100',
      getRuntimeState: () => ({ appPhase: 'ready', backendState: 'ready' }),
      getOptionalComponents: () => ({
        gvhmr: {
          requested: false,
          configured: false,
          runtime: 'local',
          guideUrl: 'https://example.com/gvhmr',
          estimatedAdditionalBytes: 22,
        },
      }),
      setupGvhmr: async () => ({
        action: 'cancelled',
        state: {
          requested: false,
          configured: false,
          runtime: 'local',
          guideUrl: 'https://example.com/gvhmr',
          estimatedAdditionalBytes: 22,
        },
      }),
      restartBackend: async () => ({ appPhase: 'ready', backendState: 'ready' }),
      hasSeenTutorial,
      markTutorialSeen
    })
    return { event, mainWindow, closeWindow, hasSeenTutorial, markTutorialSeen }
  }

  it('opens a trusted native directory picker and returns the selected path', async () => {
    const { event, mainWindow } = register()
    electronMocks.showOpenDialog.mockResolvedValue({
      canceled: false,
      filePaths: ['C:\\motions']
    })

    const handler = electronMocks.handlers.get(DESKTOP_CHANNELS.selectDirectory)
    await expect(handler?.(event)).resolves.toBe('C:\\motions')

    expect(securityMocks.assertTrustedIpcSender).toHaveBeenCalledWith(
      event,
      mainWindow,
      'http://127.0.0.1:43100'
    )
    expect(electronMocks.showOpenDialog).toHaveBeenCalledWith(mainWindow, {
      properties: ['openDirectory', 'createDirectory']
    })
    expect(securityMocks.assertTrustedIpcSender.mock.invocationCallOrder[0]).toBeLessThan(
      electronMocks.showOpenDialog.mock.invocationCallOrder[0] ?? Number.POSITIVE_INFINITY
    )
  })

  it('returns null when the native directory picker is cancelled', async () => {
    const { event } = register()
    electronMocks.showOpenDialog.mockResolvedValue({ canceled: true, filePaths: [] })

    const handler = electronMocks.handlers.get(DESKTOP_CHANNELS.selectDirectory)

    await expect(handler?.(event)).resolves.toBeNull()
  })

  it('closes the trusted main window when application exit is requested', async () => {
    const { event, mainWindow, closeWindow } = register()

    const handler = electronMocks.handlers.get(DESKTOP_CHANNELS.exitApplication)
    await expect(handler?.(event)).resolves.toBeUndefined()

    expect(securityMocks.assertTrustedIpcSender).toHaveBeenCalledWith(
      event,
      mainWindow,
      'http://127.0.0.1:43100'
    )
    expect(closeWindow).toHaveBeenCalledOnce()
    expect(securityMocks.assertTrustedIpcSender.mock.invocationCallOrder[0]).toBeLessThan(
      closeWindow.mock.invocationCallOrder[0] ?? Number.POSITIVE_INFINITY
    )
  })

  it('does not close the window when application exit is requested by an untrusted sender', async () => {
    const { event, closeWindow } = register()
    securityMocks.assertTrustedIpcSender.mockImplementationOnce(() => {
      throw new Error('Rejected IPC from an unknown WebContents')
    })

    const handler = electronMocks.handlers.get(DESKTOP_CHANNELS.exitApplication)

    await expect(handler?.(event)).rejects.toThrow('Rejected IPC from an unknown WebContents')
    expect(closeWindow).not.toHaveBeenCalled()
  })

  it('returns persisted tutorial state to a trusted sender', () => {
    const { event, mainWindow, hasSeenTutorial } = register()
    hasSeenTutorial.mockReturnValue(true)

    const handler = electronMocks.handlers.get(DESKTOP_CHANNELS.hasSeenTutorial)
    expect(handler?.(event)).toBe(true)

    expect(securityMocks.assertTrustedIpcSender).toHaveBeenCalledWith(
      event,
      mainWindow,
      'http://127.0.0.1:43100'
    )
    expect(hasSeenTutorial).toHaveBeenCalledOnce()
    expect(securityMocks.assertTrustedIpcSender.mock.invocationCallOrder[0]).toBeLessThan(
      hasSeenTutorial.mock.invocationCallOrder[0] ?? Number.POSITIVE_INFINITY
    )
  })

  it('marks the tutorial seen only after validating the sender', () => {
    const { event, mainWindow, markTutorialSeen } = register()

    const handler = electronMocks.handlers.get(DESKTOP_CHANNELS.markTutorialSeen)
    expect(handler?.(event)).toBeUndefined()

    expect(securityMocks.assertTrustedIpcSender).toHaveBeenCalledWith(
      event,
      mainWindow,
      'http://127.0.0.1:43100'
    )
    expect(markTutorialSeen).toHaveBeenCalledOnce()
    expect(securityMocks.assertTrustedIpcSender.mock.invocationCallOrder[0]).toBeLessThan(
      markTutorialSeen.mock.invocationCallOrder[0] ?? Number.POSITIVE_INFINITY
    )
  })

  it('does not persist tutorial state for an untrusted sender', () => {
    const { event, markTutorialSeen } = register()
    securityMocks.assertTrustedIpcSender.mockImplementationOnce(() => {
      throw new Error('Rejected IPC from an unknown WebContents')
    })

    const handler = electronMocks.handlers.get(DESKTOP_CHANNELS.markTutorialSeen)

    expect(() => handler?.(event)).toThrow('Rejected IPC from an unknown WebContents')
    expect(markTutorialSeen).not.toHaveBeenCalled()
  })

  it('removes the directory picker handler during cleanup', () => {
    const mainWindow = {} as BrowserWindow
    const unregister = registerDesktopHandlers({
      mainWindow,
      trustedOrigin: 'http://127.0.0.1:43100',
      getRuntimeState: () => ({ appPhase: 'ready', backendState: 'ready' }),
      getOptionalComponents: () => ({
        gvhmr: {
          requested: false,
          configured: false,
          runtime: 'local',
          guideUrl: 'https://example.com/gvhmr',
          estimatedAdditionalBytes: 22,
        },
      }),
      setupGvhmr: async () => ({
        action: 'cancelled',
        state: {
          requested: false,
          configured: false,
          runtime: 'local',
          guideUrl: 'https://example.com/gvhmr',
          estimatedAdditionalBytes: 22,
        },
      }),
      restartBackend: async () => ({ appPhase: 'ready', backendState: 'ready' }),
      hasSeenTutorial: () => false,
      markTutorialSeen: () => undefined
    })

    unregister()

    expect(electronMocks.removeHandler).toHaveBeenCalledWith(DESKTOP_CHANNELS.selectDirectory)
    expect(electronMocks.removeHandler).toHaveBeenCalledWith(DESKTOP_CHANNELS.exitApplication)
    expect(electronMocks.removeHandler).toHaveBeenCalledWith(DESKTOP_CHANNELS.hasSeenTutorial)
    expect(electronMocks.removeHandler).toHaveBeenCalledWith(DESKTOP_CHANNELS.markTutorialSeen)
  })
})
