import { spawnSync } from 'node:child_process'
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

interface DesktopPackage {
  desktopName: string
  license: string
  scripts: Record<string, string>
  build: {
    productName: string
    extraResources: Array<{ from: string; to: string }>
    linux: {
      executableName: string
      extraResources: Array<{ from: string; to: string; filter: string[] }>
    }
    win: { extraResources: Array<{ from: string; to: string; filter: string[] }> }
    deb: {
      depends: string[]
      fpm: string[]
      afterInstall?: string
      afterRemove?: string
    }
    nsis: { include?: string }
  }
}

const desktopRoot = resolve(import.meta.dirname, '..')
const packageMetadata = JSON.parse(
  readFileSync(join(desktopRoot, 'package.json'), 'utf8')
) as DesktopPackage

function bootstrapFixture(): string {
  const root = mkdtempSync(join(tmpdir(), 'hhtools-bootstrap-fixture-'))
  const assets = join(root, 'assets')
  const bin = join(root, 'bin')
  mkdirSync(assets, { recursive: true })
  mkdirSync(bin, { recursive: true })
  writeFileSync(join(assets, 'hhtools-0.1.0-py3-none-any.whl'), 'wheel\n', 'utf8')
  writeFileSync(join(assets, 'requirements-all.txt'), 'dependency==1.0\n', 'utf8')
  writeFileSync(join(assets, 'installer-uv.toml'), '', 'utf8')
  const uv = join(bin, 'uv')
  writeFileSync(uv, "#!/bin/sh\nprintf '%s\\n' 'uv 0.12.9'\n", 'utf8')
  chmodSync(uv, 0o755)
  return root
}

describe('Linux package entry points', () => {
  it('keeps the desktop identity while separating GUI and CLI commands', () => {
    expect(packageMetadata.license).toBe('Apache-2.0')
    expect(packageMetadata.desktopName).toBe('hhtools')
    expect(packageMetadata.build.productName).toBe('Human-Humanoid Tools')
    expect(packageMetadata.build.linux.executableName).toBe('hhtools-desktop')

    // The thin GUI package does not install or shadow the Python CLI.
    expect(packageMetadata.build.deb.fpm).not.toContain(
      expect.stringContaining('/usr/bin/hhtools')
    )
    expect(packageMetadata.build.deb.afterInstall).toBeUndefined()
    expect(packageMetadata.build.deb.afterRemove).toBeUndefined()
  })

  it('packages the installer bootstrap without embedding a Linux Python runtime', () => {
    expect(packageMetadata.scripts.postinstall).toBe('npm run prepare:electron')
    expect(packageMetadata.scripts['dist:linux']).toContain('npm run prepare:electron')
    expect(packageMetadata.scripts['dist:linux']).not.toContain('npm run prepare:models')
    expect(packageMetadata.scripts['dist:linux']).toContain('npm run prepare:builtin')
    expect(packageMetadata.scripts['dist:linux']).toContain('npm run prepare:bootstrap')
    expect(packageMetadata.scripts['dist:linux']).not.toContain('prepare:runtime')
    expect(packageMetadata.build.extraResources).toEqual([
      { from: '.builtin', to: 'builtin', filter: ['**/*'] }
    ])
    expect(packageMetadata.build.linux.extraResources).toEqual([
      { from: '.bootstrap', to: 'bootstrap', filter: ['**/*'] }
    ])
    expect(packageMetadata.build.nsis.include).toBeUndefined()
  })

  it('restores a bundled runtime only for the standalone Windows installer', () => {
    expect(packageMetadata.scripts['dist:win']).toContain('npm run prepare:runtime')
    expect(packageMetadata.scripts['dist:win']).not.toContain('npm run prepare:models')
    expect(packageMetadata.scripts['dist:win']).not.toContain('prepare:bootstrap')
    expect(packageMetadata.build.win.extraResources).toEqual([
      { from: '.runtime', to: 'runtime', filter: ['**/*'] }
    ])
  })

  it('stages local runtime inputs without an unpublished release download', () => {
    const fixture = bootstrapFixture()
    const output = join(fixture, 'output')
    const result = spawnSync(
      process.execPath,
      [join(desktopRoot, 'scripts', 'prepare-bootstrap.mjs')],
      {
        cwd: desktopRoot,
        encoding: 'utf8',
        env: {
          ...process.env,
          HHTOOLS_DESKTOP_BOOTSTRAP_ASSET_DIR: fixture,
          HHTOOLS_DESKTOP_BOOTSTRAP_OUTPUT_DIR: output
        }
      }
    )

    expect(result.status, result.stderr).toBe(0)
    const installer = readFileSync(join(output, 'install.sh'), 'utf8')
    expect(installer).toContain("embedded_version='0.1.0'")
    expect(installer).toContain("embedded_wheel='hhtools-0.1.0-py3-none-any.whl'")
    const runtimeId = readFileSync(join(output, 'RUNTIME_ID'), 'utf8').trim()
    expect(runtimeId).toMatch(/^0\.1\.0\+sha256\.[a-f0-9]{20}$/)
    expect(installer).toContain(`embedded_runtime_id='${runtimeId}'`)
    expect(installer).not.toMatch(/(^|[;&|]\s*)curl(?:\s|$)/m)
    expect(
      spawnSync(process.platform === 'darwin' ? 'shasum' : 'sha256sum',
        [...(process.platform === 'darwin' ? ['-a', '256'] : []), '-c', 'SHA256SUMS'], {
        cwd: output,
        encoding: 'utf8'
      }).status
    ).toBe(0)
  })

  it('migrates only the exact legacy GUI alternative and explains dpkg recovery', () => {
    const beforeInstall = readFileSync(
      join(desktopRoot, 'scripts', 'linux-before-install.sh'),
      'utf8'
    )

    expect(packageMetadata.build.deb.fpm).toContain(
      '--before-install=scripts/linux-before-install.sh'
    )
    expect(beforeInstall).toContain("legacy_gui='/opt/Human-Humanoid Tools/hhtools'")
    expect(beforeInstall).toContain('update-alternatives --remove hhtools "$legacy_gui"')
    expect(beforeInstall).toContain('sudo apt-get -f install')
  })

  it('declares Electron libraries across supported Ubuntu releases', () => {
    expect(packageMetadata.build.deb.depends).toEqual(
      expect.arrayContaining([
        'libgbm1',
        'libasound2',
        'ca-certificates',
        'pkexec'
      ])
    )
    expect(packageMetadata.build.deb.depends).not.toContain('policykit-1')
    expect(packageMetadata.build.deb.depends).not.toContain('curl')
  })
})
