import { describe, expect, it, vi } from 'vitest'

import { AppLifecycle } from '../src/main/app-lifecycle'

describe('AppLifecycle', () => {
  it('allows forward-only phase transitions', () => {
    const lifecycle = new AppLifecycle()
    lifecycle.transition('backend-starting')
    lifecycle.transition('ready')

    expect(lifecycle.phase).toBe('ready')
    expect(() => lifecycle.transition('starting')).toThrow(/Invalid lifecycle transition/)
  })

  it('runs named shutdown joiners', async () => {
    const lifecycle = new AppLifecycle()
    const first = vi.fn()
    const second = vi.fn(async () => undefined)
    lifecycle.registerShutdownJoiner('first', first)
    lifecycle.registerShutdownJoiner('second', second)

    const result = await lifecycle.runShutdownJoiners()

    expect(first).toHaveBeenCalledOnce()
    expect(second).toHaveBeenCalledOnce()
    expect(result).toEqual({ timedOut: false, failures: [] })
    expect(lifecycle.phase).toBe('shutting-down')
  })

  it('records a failed joiner without blocking the others', async () => {
    const lifecycle = new AppLifecycle()
    const completed = vi.fn()
    lifecycle.registerShutdownJoiner('broken', () => {
      throw new Error('failed')
    })
    lifecycle.registerShutdownJoiner('completed', completed)

    const result = await lifecycle.runShutdownJoiners()

    expect(completed).toHaveBeenCalledOnce()
    expect(result.failures).toHaveLength(1)
    expect(result.failures[0]?.name).toBe('broken')
  })
})
