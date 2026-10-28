import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import {
  copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync,
  rmSync, writeFileSync
} from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import manifest from '../../configs/builtin-assets.json'

const temporaryRoots: string[] = []

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function fixture() {
  const root = mkdtempSync(join(tmpdir(), 'hhtools-builtin-assets-'))
  temporaryRoots.push(root)
  const desktop = join(root, 'desktop')
  const robotRoot = join(root, 'release-robots')
  const script = join(desktop, 'scripts', 'prepare-builtin-assets.mjs')
  mkdirSync(dirname(script), { recursive: true })
  copyFileSync(join(import.meta.dirname, '..', 'scripts', 'prepare-builtin-assets.mjs'), script)
  mkdirSync(join(root, 'configs'), { recursive: true })
  const manifestText = `${JSON.stringify(manifest, null, 2)}\n`
  writeFileSync(join(root, 'configs', 'builtin-assets.json'), manifestText)
  const motionPaths = [
    ...manifest.motions.flatMap((motion) => motion.files), ...manifest.motion_shared_files
  ]
  for (const relative of motionPaths) {
    const path = join(root, ...relative.split('/'))
    mkdirSync(dirname(path), { recursive: true })
    writeFileSync(path, `fixture for ${relative}\n`)
  }
  for (const robot of manifest.robots) {
    const directory = join(robotRoot, robot.preset)
    mkdirSync(directory, { recursive: true })
    const files = {
      'robot.yaml': `name: ${robot.preset}\nurdf: model.urdf\n`,
      'model.urdf': `<robot name="${robot.preset}"><link name="base"/></robot>\n`
    }
    const installedFiles = Object.entries(files).map(([path, content]) => {
      writeFileSync(join(directory, path), content)
      return { path, bytes: Buffer.byteLength(content), sha256: createHash('sha256').update(content).digest('hex') }
    })
    writeFileSync(join(directory, robot.bundle_manifest), JSON.stringify({
      preset: robot.preset,
      display_name: robot.display_name,
      source: {
        repository: robot.repository, commit: robot.commit,
        main_urdf: robot.main_urdf, license_paths: robot.license_paths
      },
      installed_files: installedFiles
    }))
  }
  return {
    desktop, robotRoot, manifestText,
    stage: () => spawnSync(process.execPath, [script], {
      encoding: 'utf8',
      env: { ...process.env, HHTOOLS_BUNDLED_ROBOT_DIR: robotRoot }
    })
  }
}

describe('standard built-in asset staging', () => {
  it('includes RP1 and the original six through the standard manifest, excluding user files', () => {
    const data = fixture()
    const privateCalibration = join(data.robotRoot, 'rp1', 'retarget_calibration_lafan_bvh.yaml')
    writeFileSync(privateCalibration, 'private calibration\n')
    mkdirSync(join(data.robotRoot, 'my_custom_robot'))
    writeFileSync(join(data.robotRoot, 'my_custom_robot', 'robot.yaml'), 'private robot\n')

    const result = data.stage()
    expect(result.status, result.stderr).toBe(0)
    const staged = join(data.desktop, '.builtin')
    expect(readdirSync(join(staged, 'robots')).sort()).toEqual([
      'agibot_x2_ultra', 'asimov_1', 'berkeley_humanoid_lite', 'fourier_gr2',
      'g1_29dof', 'roboto_origin', 'rp1'
    ])
    expect(readFileSync(join(staged, 'MANIFEST.json'), 'utf8')).toBe(data.manifestText)
    expect(readFileSync(join(staged, 'robots', 'rp1', 'SOURCE.json'), 'utf8')).toBe(
      readFileSync(join(data.robotRoot, 'rp1', 'SOURCE.json'), 'utf8')
    )
    expect(existsSync(join(staged, 'robots', 'rp1', 'retarget_calibration_lafan_bvh.yaml'))).toBe(false)
    expect(readFileSync(privateCalibration, 'utf8')).toBe('private calibration\n')
  })

  it('rejects a modified RP1 file while preserving the previous staged bundle', () => {
    const data = fixture()
    const staged = join(data.desktop, '.builtin')
    mkdirSync(staged)
    writeFileSync(join(staged, 'previous-build'), 'retain on failure\n')
    writeFileSync(join(data.robotRoot, 'rp1', 'model.urdf'), 'modified geometry\n')

    const result = data.stage()
    expect(result.status).not.toBe(0)
    expect(result.stderr).toContain('installed robot file differs: rp1/model.urdf')
    expect(readFileSync(join(staged, 'previous-build'), 'utf8')).toBe('retain on failure\n')
  })

  it('rejects an RP1 bundle from a different upstream revision', () => {
    const data = fixture()
    const sourcePath = join(data.robotRoot, 'rp1', 'SOURCE.json')
    const source = JSON.parse(readFileSync(sourcePath, 'utf8'))
    source.source.commit = '0'.repeat(40)
    writeFileSync(sourcePath, JSON.stringify(source))

    const result = data.stage()
    expect(result.status).not.toBe(0)
    expect(result.stderr).toContain('installed robot source differs from the allowlist: rp1')
    expect(existsSync(join(data.desktop, '.builtin'))).toBe(false)
  })
})
