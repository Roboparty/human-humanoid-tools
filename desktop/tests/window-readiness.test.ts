import { EventEmitter } from 'node:events'

import { describe, expect, it, vi } from 'vitest'

import { waitForWindowReadiness } from '../src/main/window-readiness'

describe('waitForWindowReadiness', () => {
  it('resolves on Electron ready-to-show and removes the load fallback', async () => {
    const windowEvents = new EventEmitter()
    const webContentsEvents = new EventEmitter()
    const resolved = vi.fn()
    void waitForWindowReadiness(windowEvents, webContentsEvents).then(resolved)

    windowEvents.emit('ready-to-show')
    await Promise.resolve()

    expect(resolved).toHaveBeenCalledOnce()
    expect(webContentsEvents.listenerCount('did-finish-load')).toBe(0)
  })

  it('resolves on did-finish-load when Wayland omits ready-to-show', async () => {
    const windowEvents = new EventEmitter()
    const webContentsEvents = new EventEmitter()
    const resolved = vi.fn()
    void waitForWindowReadiness(windowEvents, webContentsEvents).then(resolved)

    webContentsEvents.emit('did-finish-load')
    await Promise.resolve()

    expect(resolved).toHaveBeenCalledOnce()
    expect(windowEvents.listenerCount('ready-to-show')).toBe(0)
  })
})
