import { execFileSync, spawnSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import {
  assertPathInside,
  listApplicationSourceFiles
} from '../scripts/runtime-staging-policy.mjs'

describe('Windows runtime staging policy', () => {
  it.runIf(process.platform !== 'win32')('rejects staging on a non-Windows host', () => {
    const result = spawnSync(
      process.execPath,
      [join(import.meta.dirname, '..', 'scripts', 'prepare-runtime.mjs')],
      { encoding: 'utf8' }
    )

    expect(result.status).not.toBe(0)
    expect(result.stderr).toContain('must be staged on Windows')
  })

  it('selects tracked files without copying ignored or untracked content', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-staging-git-'))
    mkdirSync(join(root, 'hhtools'), { recursive: true })
    writeFileSync(join(root, 'hhtools', 'tracked.py'), 'tracked\n', 'utf8')
    writeFileSync(join(root, 'hhtools', 'untracked.secret'), 'private\n', 'utf8')
    writeFileSync(join(root, '.gitignore'), '*.secret\n', 'utf8')
    execFileSync('git', ['init', '--quiet'], { cwd: root })
    execFileSync(
      'git',
      ['-c', 'core.autocrlf=false', 'add', '.gitignore', 'hhtools/tracked.py'],
      { cwd: root }
    )

    expect(listApplicationSourceFiles(root, ['hhtools'], {})).toEqual({
      provenance: 'git-tracked-worktree',
      files: ['hhtools/tracked.py']
    })
  })

  it('requires explicit trust for a verified source archive without Git metadata', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-staging-archive-'))
    mkdirSync(join(root, 'hhtools'), { recursive: true })
    writeFileSync(join(root, 'hhtools', 'app.py'), '', 'utf8')

    expect(() => listApplicationSourceFiles(root, ['hhtools'], {})).toThrow(
      'HHTOOLS_TRUST_SOURCE_ARCHIVE=1'
    )
    expect(
      listApplicationSourceFiles(root, ['hhtools'], { HHTOOLS_TRUST_SOURCE_ARCHIVE: '1' })
    ).toEqual({ provenance: 'trusted-archive', files: ['hhtools/app.py'] })
  })

  it('rejects runtime destinations outside the staging root', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-staging-root-'))

    expect(() =>
      assertPathInside(root, join(root, 'python', 'Lib'), 'outside runtime', {
        allowRoot: false
      })
    ).not.toThrow()
    expect(() =>
      assertPathInside(root, join(root, '..', 'system-python'), 'outside runtime', {
        allowRoot: false
      })
    ).toThrow('outside runtime')
  })
})
