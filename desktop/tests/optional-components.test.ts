import { existsSync, mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describe, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({
  dialog: {},
  shell: {},
}))

import { OptionalComponentStore } from '../src/main/optional-components'

function createGvhmrCheckout(root: string): string {
  const checkout = join(root, 'GVHMR')
  mkdirSync(join(checkout, 'tools', 'demo'), { recursive: true })
  writeFileSync(join(checkout, 'tools', 'demo', 'demo.py'), '', 'utf8')
  return checkout
}

describe('OptionalComponentStore', () => {
  it('consumes the NSIS selection marker exactly once', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-components-marker-'))
    const userData = join(root, 'user-data')
    const localAppData = join(root, 'local-app-data')
    const marker = join(localAppData, 'hhtools', 'installer', 'gvhmr.requested')
    mkdirSync(join(localAppData, 'hhtools', 'installer'), { recursive: true })
    writeFileSync(marker, '1\n', 'utf8')

    const store = new OptionalComponentStore({ userData, localAppData, env: {} })

    expect(store.getState({}).gvhmr.requested).toBe(true)
    expect(existsSync(marker)).toBe(false)

    const restored = new OptionalComponentStore({ userData, localAppData, env: {} })
    expect(restored.getState({}).gvhmr.requested).toBe(true)
  })

  it('persists a validated checkout and exposes it to the sidecar', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-components-config-'))
    const userData = join(root, 'user-data')
    const checkout = createGvhmrCheckout(root)
    const python = join(root, 'gvhmr-env', 'bin', 'python')
    const bodyModels = join(root, 'licensed-models')
    mkdirSync(join(root, 'gvhmr-env', 'bin'), { recursive: true })
    mkdirSync(bodyModels, { recursive: true })
    writeFileSync(python, '', 'utf8')
    const store = new OptionalComponentStore({ userData, env: {}, platform: 'linux' })

    expect(store.configureGvhmr(checkout, python, bodyModels)).toMatchObject({
      configured: true,
      requested: false,
      root: checkout,
      python,
      bodyModels,
      runtime: 'local',
    })
    expect(store.sidecarEnvironment({})).toEqual({
      HHTOOLS_GVHMR_ROOT: checkout,
      HHTOOLS_GVHMR_PYTHON: python,
      HHTOOLS_GVHMR_BODY_MODELS: bodyModels,
    })

    const restored = new OptionalComponentStore({ userData, env: {}, platform: 'linux' })
    expect(restored.getState({}).gvhmr.root).toBe(checkout)
    expect(restored.getState({}).gvhmr.python).toBe(python)
    expect(restored.getState({}).gvhmr.bodyModels).toBe(bodyModels)
  })

  it('keeps the Windows Docker setup independent from a Python path', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-components-windows-'))
    const checkout = createGvhmrCheckout(root)
    const store = new OptionalComponentStore({
      userData: join(root, 'user-data'),
      env: {},
      platform: 'win32',
    })

    expect(store.configureGvhmr(checkout)).toMatchObject({
      configured: true,
      root: checkout,
      runtime: 'docker',
    })
    expect(store.sidecarEnvironment({})).toEqual({
      HHTOOLS_GVHMR_ROOT: checkout,
      HHTOOLS_GVHMR_BODY_MODELS: join(checkout, 'inputs', 'checkpoints', 'body_models'),
    })
  })

  it('lets an explicit environment override persisted GVHMR paths', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-components-override-'))
    const userData = join(root, 'user-data')
    const checkout = createGvhmrCheckout(root)
    const configuredPython = join(root, 'configured-env', 'bin', 'python')
    const overridePython = join(root, 'override-env', 'bin', 'python')
    mkdirSync(join(root, 'configured-env', 'bin'), { recursive: true })
    mkdirSync(join(root, 'override-env', 'bin'), { recursive: true })
    writeFileSync(configuredPython, '', 'utf8')
    writeFileSync(overridePython, '', 'utf8')
    const store = new OptionalComponentStore({ userData, env: {}, platform: 'linux' })
    store.configureGvhmr(checkout, configuredPython, join(root, 'configured-models'))

    expect(store.getState({
      HHTOOLS_GVHMR_ROOT: checkout,
      HHTOOLS_GVHMR_PYTHON: overridePython,
      HHTOOLS_GVHMR_BODY_MODELS: join(root, 'override-models'),
    }).gvhmr).toMatchObject({
      root: checkout,
      python: overridePython,
      bodyModels: join(root, 'override-models'),
    })
  })

  it('rejects a folder that is not an official GVHMR checkout', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-components-invalid-'))
    const store = new OptionalComponentStore({ userData: join(root, 'user-data'), env: {} })

    expect(() => store.configureGvhmr(root)).toThrow('not an official GVHMR checkout')
  })
})
