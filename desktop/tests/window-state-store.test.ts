import { describe, expect, it } from 'vitest'

import { normalizeWindowState } from '../src/main/window-state-store'

const primary = { workArea: { x: 0, y: 0, width: 1920, height: 1080 } }

describe('normalizeWindowState', () => {
  it('keeps valid bounds that intersect a display', () => {
    const state = normalizeWindowState(
      { x: 100, y: 80, width: 1200, height: 800, maximized: true },
      [primary],
      primary
    )

    expect(state).toEqual({ x: 100, y: 80, width: 1200, height: 800, maximized: true })
  })

  it('centers bounds that belonged to a removed display', () => {
    const state = normalizeWindowState(
      { x: 4000, y: 100, width: 1200, height: 800, maximized: false },
      [primary],
      primary
    )

    expect(state.x).toBe(240)
    expect(state.y).toBe(90)
    expect(state.maximized).toBe(false)
  })

  it('rejects undersized persisted windows', () => {
    const state = normalizeWindowState(
      { x: 0, y: 0, width: 400, height: 300, maximized: false },
      [primary],
      primary
    )

    expect(state.width).toBe(1440)
    expect(state.height).toBe(900)
  })
})
