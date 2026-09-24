import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { DesktopTutorialState } from '../src/main/desktop-tutorial-state'

const temporaryRoots: string[] = []

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) {
    rmSync(root, { recursive: true, force: true })
  }
})

describe('DesktopTutorialState', () => {
  it('persists the seen marker across desktop launches', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-tutorial-state-'))
    temporaryRoots.push(root)
    const userData = join(root, 'user-data')
    const state = new DesktopTutorialState(userData)

    expect(state.hasSeenTutorial()).toBe(false)
    expect(existsSync(userData)).toBe(false)

    state.markTutorialSeen()
    state.markTutorialSeen()

    expect(state.hasSeenTutorial()).toBe(true)
    expect(new DesktopTutorialState(userData).hasSeenTutorial()).toBe(true)
    expect(readFileSync(join(userData, 'tutorial.seen'), 'utf8')).toBe('1\n')
  })
})
