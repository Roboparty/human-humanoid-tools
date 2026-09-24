import type { BrowserWindow, IpcMainInvokeEvent } from 'electron'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { RuntimeInstallProgress } from '../src/shared/installer-api'

const electronMocks = vi.hoisted(() => ({
  handlers: new Map<string, (...args: unknown[]) => unknown>(),
  handle: vi.fn(),
  removeHandler: vi.fn()
}))

vi.mock('electron', () => ({
  ipcMain: {
    handle: electronMocks.handle,
    removeHandler: electronMocks.removeHandler
  }
}))

import { registerInstallerHandlers } from '../src/main/ipc/register-installer-handlers'
import type { RuntimeInstaller } from '../src/main/runtime-installer'
import { INSTALLER_CHANNELS } from '../src/shared/installer-api'

describe('registerInstallerHandlers', () => {
  beforeEach(() => {
    electronMocks.handlers.clear()
    vi.clearAllMocks()
    electronMocks.handle.mockImplementation((channel, handler) => {
      electronMocks.handlers.set(channel, handler)
    })
  })

  function register(result: 'completed' | 'failed' | 'cancelled' = 'failed') {
    const send = vi.fn()
    const webContents = { send, isDestroyed: vi.fn(() => false) }
    const window = {
      isDestroyed: vi.fn(() => false),
      webContents
    } as unknown as BrowserWindow
    const start = vi.fn(async () => ({ status: result, message: result }))
    const cancel = vi.fn()
    let progressPublisher: ((progress: RuntimeInstallProgress) => void) | undefined
    const setProgressPublisher = vi.fn(
      (publisher: ((progress: RuntimeInstallProgress) => void) | undefined) => {
        progressPublisher = publisher
      }
    )
    const installer = { start, cancel, setProgressPublisher } as unknown as RuntimeInstaller
    const onInstalled = vi.fn()
    const cleanup = registerInstallerHandlers({ window, installer, onInstalled })
    const event = { sender: webContents } as unknown as IpcMainInvokeEvent
    return {
      cancel,
      cleanup,
      event,
      onInstalled,
      progress: (update: RuntimeInstallProgress) => progressPublisher?.(update),
      send,
      setProgressPublisher,
      start,
      window
    }
  }

  it('starts only a validated installation mode from the setup window', async () => {
    const { event, start } = register()
    const handler = electronMocks.handlers.get(INSTALLER_CHANNELS.start)

    await expect(Promise.resolve(handler?.(event, 'user'))).resolves.toEqual({
      status: 'failed',
      message: 'failed'
    })
    expect(start).toHaveBeenCalledWith('user')
    await expect(Promise.resolve(handler?.(event, 'other'))).rejects.toThrow(
      'Invalid installation mode'
    )
    expect(start).toHaveBeenCalledOnce()
  })

  it('rejects an installer request from any other WebContents', async () => {
    const { start } = register()
    const handler = electronMocks.handlers.get(INSTALLER_CHANNELS.start)
    const event = { sender: {} } as IpcMainInvokeEvent

    await expect(Promise.resolve(handler?.(event, 'system'))).rejects.toThrow(
      'Rejected installer IPC from an unknown WebContents'
    )
    expect(start).not.toHaveBeenCalled()
  })

  it('sends progress only to the live setup window', () => {
    const { progress, send, window } = register()
    const update: RuntimeInstallProgress = {
      phase: 'installing',
      message: 'Installing dependencies'
    }

    progress(update)
    expect(send).toHaveBeenCalledWith(INSTALLER_CHANNELS.progress, update)
    vi.mocked(window.isDestroyed).mockReturnValue(true)
    progress(update)
    expect(send).toHaveBeenCalledOnce()
  })

  it('restarts only after a completed installation', async () => {
    vi.useFakeTimers()
    try {
      const { event, onInstalled } = register('completed')
      const handler = electronMocks.handlers.get(INSTALLER_CHANNELS.start)

      await Promise.resolve(handler?.(event, 'user'))
      expect(onInstalled).not.toHaveBeenCalled()
      await vi.advanceTimersByTimeAsync(500)
      expect(onInstalled).toHaveBeenCalledOnce()
    } finally {
      vi.useRealTimers()
    }
  })

  it('does not relaunch after the setup window is closed', async () => {
    vi.useFakeTimers()
    try {
      const { cleanup, event, onInstalled } = register('completed')
      const handler = electronMocks.handlers.get(INSTALLER_CHANNELS.start)

      await Promise.resolve(handler?.(event, 'user'))
      cleanup()
      await vi.advanceTimersByTimeAsync(500)
      expect(onInstalled).not.toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
    }
  })

  it('cancels work and removes both IPC handlers during cleanup', () => {
    const { cancel, cleanup, setProgressPublisher } = register()

    cleanup()

    expect(setProgressPublisher).toHaveBeenLastCalledWith(undefined)
    expect(cancel).toHaveBeenCalledOnce()
    expect(electronMocks.removeHandler).toHaveBeenCalledWith(INSTALLER_CHANNELS.start)
    expect(electronMocks.removeHandler).toHaveBeenCalledWith(INSTALLER_CHANNELS.cancel)
  })
})
