import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HHToolsDesktopApi } from '../src/shared/desktop-api'

const electronMocks = vi.hoisted(() => ({
  exposeInMainWorld: vi.fn(),
  invoke: vi.fn(),
}))

vi.mock('electron', () => ({
  contextBridge: { exposeInMainWorld: electronMocks.exposeInMainWorld },
  ipcRenderer: { invoke: electronMocks.invoke },
}))

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
  vi.resetModules()
})

describe('desktop preload capabilities', () => {
  it.each([
    ['darwin', false],
    ['win32', true],
    ['linux', true],
  ] as const)('exposes synchronous read-only Video2Motion support on %s', async (platform, enabled) => {
    vi.stubGlobal('process', { ...process, platform })

    await import('../src/preload/index')

    expect(electronMocks.exposeInMainWorld).toHaveBeenCalledOnce()
    const [name, api] = electronMocks.exposeInMainWorld.mock.calls[0] as [string, HHToolsDesktopApi]
    expect(name).toBe('hhtoolsDesktop')
    expect(api.capabilities.videoToMotion).toBe(enabled)
    expect(Object.isFrozen(api.capabilities)).toBe(true)
    expect(Reflect.set(api.capabilities, 'videoToMotion', !enabled)).toBe(false)
    expect(api.capabilities.videoToMotion).toBe(enabled)
    expect(electronMocks.invoke).not.toHaveBeenCalled()
  })
})
