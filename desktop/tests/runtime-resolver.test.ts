import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import {
  buildSidecarEnvironment,
  resolveRuntime,
  RuntimeNotFoundError
} from '../src/main/runtime-resolver'

describe('resolveRuntime', () => {
  it('finds a repository above the desktop working directory', () => {
    const root = fileURLToPath(new URL('../..', import.meta.url))
    const runtime = resolveRuntime({
      appPath: join(root, 'desktop'),
      cwd: join(root, 'desktop'),
      userData: join(root, '.test-user-data')
    })

    expect(runtime.repoRoot).toBe(resolve(root))
    expect(runtime.kind).toBe('checkout')
    expect(runtime.workingDirectory).toBe(resolve(root))
    expect(runtime.sourceRoot).toBe(join(root, 'assets', 'motions'))
    expect(runtime.cacheDirectory).toBe(join(root, '.test-user-data', 'hhtools-cache'))
  })

  it('keeps a packaged shell on the external checkout while exposing its model resource', () => {
    const resourcesPath = mkdtempSync(join(tmpdir(), 'hhtools-packaged-runtime-test-'))
    const repoRoot = mkdtempSync(join(tmpdir(), 'hhtools-packaged-checkout-test-'))
    const pythonExecutable = join(repoRoot, '.venv', 'bin', 'python')
    const bodyModels = join(resourcesPath, 'body_models')
    mkdirSync(join(repoRoot, 'hhtools'), { recursive: true })
    mkdirSync(dirname(pythonExecutable), { recursive: true })
    mkdirSync(join(bodyModels, 'smplx'), { recursive: true })
    writeFileSync(join(repoRoot, 'pyproject.toml'), '', 'utf8')
    writeFileSync(pythonExecutable, '', 'utf8')
    writeFileSync(join(bodyModels, 'smplx', 'SMPLX_NEUTRAL.npz'), 'model', 'utf8')

    const runtime = resolveRuntime({
      appPath: '/opt/hhtools',
      cwd: '/opt/hhtools',
      userData: join(resourcesPath, 'user-data'),
      isPackaged: true,
      resourcesPath,
      env: { HHTOOLS_REPO_ROOT: repoRoot },
      platform: 'linux'
    })

    expect(runtime.repoRoot).toBe(repoRoot)
    expect(runtime.pythonExecutable).toBe(pythonExecutable)
    expect(runtime.bodyModelsRoot).toBe(bodyModels)
  })

  it.each(['linux', 'darwin'] as const)('uses a completed user-managed %s runtime with packaged built-in assets', (platform) => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-managed-runtime-test-'))
    const home = join(root, 'home')
    const installRoot = platform === 'darwin'
      ? join(home, 'Library', 'Application Support', 'hhtools')
      : join(home, '.local', 'share', 'hhtools')
    const pythonExecutable = join(installRoot, 'tools', 'hhtools', 'bin', 'python')
    const resourcesPath = join(root, 'resources')
    const motions = join(resourcesPath, 'builtin', 'motions')
    const robots = join(resourcesPath, 'builtin', 'robots')
    const runtimeId = '0.1.0+sha256.0123456789abcdefabcd'
    mkdirSync(dirname(pythonExecutable), { recursive: true })
    mkdirSync(motions, { recursive: true })
    mkdirSync(robots, { recursive: true })
    mkdirSync(join(resourcesPath, 'bootstrap'), { recursive: true })
    writeFileSync(pythonExecutable, '', 'utf8')
    writeFileSync(join(installRoot, 'runtime-version'), `${runtimeId}\n`, 'utf8')
    writeFileSync(join(resourcesPath, 'bootstrap', 'RUNTIME_ID'), `${runtimeId}\n`, 'utf8')

    const runtime = resolveRuntime({
      appPath: '/opt/Human-Humanoid Tools/resources/app.asar',
      cwd: '/opt/Human-Humanoid Tools',
      userData: join(home, '.config', 'hhtools'),
      isPackaged: true,
      resourcesPath,
      appVersion: '0.1.0',
      env: { HOME: home },
      platform
    })

    expect(runtime.kind).toBe('managed')
    expect(runtime.repoRoot).toBeUndefined()
    expect(runtime.workingDirectory).toBe(installRoot)
    expect(runtime.pythonExecutable).toBe(pythonExecutable)
    expect(runtime.sourceRoot).toBe(motions)
    expect(runtime.bundledRobotRoot).toBe(robots)
  })

  it('does not select Linux system runtimes on macOS', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-mac-runtime-test-'))
    const python = join(root, 'system', 'tools', 'hhtools', 'bin', 'python')
    mkdirSync(dirname(python), { recursive: true })
    writeFileSync(python, '')
    writeFileSync(join(root, 'system', 'runtime-version'), '0.1.0\n')

    expect(() => resolveRuntime({
      appPath: join(root, 'app'),
      cwd: root,
      userData: join(root, 'data'),
      isPackaged: true,
      appVersion: '0.1.0',
      systemInstallRoot: join(root, 'system'),
      env: { HOME: join(root, 'home'), XDG_DATA_HOME: join(root, 'system') },
      platform: 'darwin'
    })).toThrow(RuntimeNotFoundError)
  })

  it('does not silently use the build checkout when a packaged app needs first-run setup', () => {
    const root = fileURLToPath(new URL('../..', import.meta.url))
    const home = mkdtempSync(join(tmpdir(), 'hhtools-mac-clean-home-'))
    expect(() => resolveRuntime({
      appPath: join(root, 'desktop', 'release', 'mac-arm64', 'HHTools.app'),
      cwd: root,
      userData: join(home, 'data'),
      isPackaged: true,
      env: { HOME: home },
      platform: 'darwin'
    })).toThrow(RuntimeNotFoundError)
  })

  it('prefers a complete bundled Windows runtime when no checkout override exists', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-bundled-runtime-test-'))
    const resourcesPath = join(root, 'resources')
    const repoRoot = join(resourcesPath, 'runtime', 'app')
    const pythonExecutable = join(resourcesPath, 'runtime', 'python', 'python.exe')
    mkdirSync(join(repoRoot, 'hhtools'), { recursive: true })
    mkdirSync(dirname(pythonExecutable), { recursive: true })
    writeFileSync(join(repoRoot, 'pyproject.toml'), '', 'utf8')
    writeFileSync(pythonExecutable, '', 'utf8')

    const runtime = resolveRuntime({
      appPath: 'C:\\Program Files\\HHTools',
      cwd: 'C:\\Program Files\\HHTools',
      userData: join(root, 'user-data'),
      isPackaged: true,
      resourcesPath,
      env: {},
      platform: 'win32'
    })

    expect(runtime.kind).toBe('bundled')
    expect(runtime.repoRoot).toBe(repoRoot)
    expect(runtime.pythonExecutable).toBe(pythonExecutable)
  })

  it('rejects a partial managed runtime without its atomic completion marker', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-partial-runtime-test-'))
    const home = join(root, 'home')
    const pythonExecutable = join(
      home,
      '.local',
      'share',
      'hhtools',
      'tools',
      'hhtools',
      'bin',
      'python'
    )
    mkdirSync(dirname(pythonExecutable), { recursive: true })
    writeFileSync(pythonExecutable, '', 'utf8')

    expect(() => resolveRuntime({
      appPath: '/opt/HHTools',
      cwd: '/opt/HHTools',
      userData: join(home, '.config', 'hhtools'),
      isPackaged: true,
      resourcesPath: join(root, 'resources'),
      systemInstallRoot: null,
      env: { HOME: home },
      platform: 'linux'
    })).toThrow(RuntimeNotFoundError)
  })

  it('rejects a managed runtime from another desktop release', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-stale-runtime-test-'))
    const installRoot = join(root, 'home', '.local', 'share', 'hhtools')
    const pythonExecutable = join(installRoot, 'tools', 'hhtools', 'bin', 'python')
    mkdirSync(dirname(pythonExecutable), { recursive: true })
    writeFileSync(pythonExecutable, '', 'utf8')
    writeFileSync(join(installRoot, 'runtime-version'), '0.0.9\n', 'utf8')

    expect(() => resolveRuntime({
      appPath: '/opt/HHTools',
      cwd: '/opt/HHTools',
      userData: join(root, 'user-data'),
      isPackaged: true,
      resourcesPath: join(root, 'resources'),
      appVersion: '0.1.0',
      systemInstallRoot: null,
      env: { HOME: join(root, 'home') },
      platform: 'linux'
    })).toThrow(RuntimeNotFoundError)
  })

  it('refreshes a same-version runtime when the packaged inputs change', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-runtime-identity-test-'))
    const home = join(root, 'home')
    const installRoot = join(home, '.local', 'share', 'hhtools')
    const pythonExecutable = join(installRoot, 'tools', 'hhtools', 'bin', 'python')
    const resourcesPath = join(root, 'resources')
    mkdirSync(dirname(pythonExecutable), { recursive: true })
    mkdirSync(join(resourcesPath, 'bootstrap'), { recursive: true })
    writeFileSync(pythonExecutable, '', 'utf8')
    writeFileSync(join(installRoot, 'runtime-version'), '0.1.0\n', 'utf8')
    writeFileSync(
      join(resourcesPath, 'bootstrap', 'RUNTIME_ID'),
      '0.1.0+sha256.0123456789abcdefabcd\n',
      'utf8'
    )

    expect(() => resolveRuntime({
      appPath: '/opt/HHTools',
      cwd: '/opt/HHTools',
      userData: join(root, 'user-data'),
      isPackaged: true,
      resourcesPath,
      appVersion: '0.1.0',
      systemInstallRoot: null,
      env: { HOME: home },
      platform: 'linux'
    })).toThrow(RuntimeNotFoundError)
  })

  it('finds a checkout-local Linux virtual environment', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-linux-runtime-test-'))
    const pythonExecutable = join(root, '.venv', 'bin', 'python')
    mkdirSync(join(root, 'hhtools'), { recursive: true })
    mkdirSync(dirname(pythonExecutable), { recursive: true })
    writeFileSync(join(root, 'pyproject.toml'), '', 'utf8')
    writeFileSync(pythonExecutable, '', 'utf8')

    const runtime = resolveRuntime({
      appPath: join(root, 'desktop'),
      cwd: root,
      userData: join(root, 'data'),
      env: {},
      platform: 'linux'
    })

    expect(runtime.pythonExecutable).toBe(pythonExecutable)
  })

  it('keeps development on the checkout even when a managed runtime exists', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-development-runtime-test-'))
    const repoRoot = join(root, 'repo')
    const checkoutPython = join(repoRoot, '.venv', 'bin', 'python')
    const managedRoot = join(root, 'home', '.local', 'share', 'hhtools')
    const managedPython = join(managedRoot, 'tools', 'hhtools', 'bin', 'python')
    mkdirSync(join(repoRoot, 'hhtools'), { recursive: true })
    mkdirSync(dirname(checkoutPython), { recursive: true })
    mkdirSync(dirname(managedPython), { recursive: true })
    writeFileSync(join(repoRoot, 'pyproject.toml'), '', 'utf8')
    writeFileSync(checkoutPython, '', 'utf8')
    writeFileSync(managedPython, '', 'utf8')
    writeFileSync(join(managedRoot, 'runtime-version'), '0.1.0\n', 'utf8')

    const runtime = resolveRuntime({
      appPath: join(repoRoot, 'desktop'),
      cwd: repoRoot,
      userData: join(root, 'user-data'),
      isPackaged: false,
      appVersion: '0.1.0',
      env: { HOME: join(root, 'home') },
      platform: 'linux'
    })

    expect(runtime.kind).toBe('checkout')
    expect(runtime.repoRoot).toBe(repoRoot)
    expect(runtime.pythonExecutable).toBe(checkoutPython)
  })

  it('honors an explicit checkout and Python runtime', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-runtime-test-'))
    mkdirSync(join(root, 'hhtools'), { recursive: true })
    writeFileSync(join(root, 'pyproject.toml'), '', 'utf8')
    const runtime = resolveRuntime({
      appPath: root,
      cwd: root,
      userData: join(root, 'data'),
      env: { HHTOOLS_REPO_ROOT: root, HHTOOLS_PYTHON: 'python-test' }
    })

    expect(runtime.pythonExecutable).toBe('python-test')
  })

  it('does not copy unrelated parent secrets into the sidecar environment', () => {
    const environment = buildSidecarEnvironment('C:\\repo', {
      PATH: 'C:\\bin',
      AWS_SECRET_ACCESS_KEY: 'do-not-copy',
      CUDA_PATH: 'C:\\cuda',
      HHTOOLS_MAX_RUNNING_JOBS: '2',
      HHTOOLS_MAX_QUEUED_JOBS: '32',
      HHTOOLS_WEB_SETTINGS_PATH: 'C:\\config\\web-settings.json',
      HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH: 'C:\\config\\motion-library-settings.json',
      HHTOOLS_ROBOT_DIR: 'C:\\config\\robots',
      LD_LIBRARY_PATH: '/opt/cuda/lib64',
      XDG_RUNTIME_DIR: '/run/user/1000',
      DISPLAY: ':1',
      WAYLAND_DISPLAY: 'wayland-0',
      XAUTHORITY: '/home/nora/.Xauthority',
      DBUS_SESSION_BUS_ADDRESS: 'unix:path=/run/user/1000/bus',
      MUJOCO_GL: 'egl',
      PYOPENGL_PLATFORM: 'egl',
      XDG_CONFIG_HOME: 'C:\\config',
      XDG_DATA_HOME: 'C:\\data',
      HHTOOLS_ARBITRARY_SECRET: 'do-not-copy-either'
    })

    expect(environment.PATH).toBe('C:\\bin')
    expect(environment.CUDA_PATH).toBe('C:\\cuda')
    expect(environment.HHTOOLS_MAX_RUNNING_JOBS).toBe('2')
    expect(environment.HHTOOLS_MAX_QUEUED_JOBS).toBe('32')
    expect(environment.HHTOOLS_WEB_SETTINGS_PATH).toBe('C:\\config\\web-settings.json')
    expect(environment.HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH).toBe(
      'C:\\config\\motion-library-settings.json'
    )
    expect(environment.HHTOOLS_ROBOT_DIR).toBe('C:\\config\\robots')
    expect(environment.LD_LIBRARY_PATH).toBe('/opt/cuda/lib64')
    expect(environment.XDG_RUNTIME_DIR).toBe('/run/user/1000')
    expect(environment.DISPLAY).toBe(':1')
    expect(environment.WAYLAND_DISPLAY).toBe('wayland-0')
    expect(environment.XAUTHORITY).toBe('/home/nora/.Xauthority')
    expect(environment.DBUS_SESSION_BUS_ADDRESS).toBe('unix:path=/run/user/1000/bus')
    expect(environment.MUJOCO_GL).toBe('egl')
    expect(environment.PYOPENGL_PLATFORM).toBe('egl')
    expect(environment.XDG_CONFIG_HOME).toBe('C:\\config')
    expect(environment.XDG_DATA_HOME).toBe('C:\\data')
    expect(environment.HHTOOLS_ARBITRARY_SECRET).toBeUndefined()
    expect(environment.AWS_SECRET_ACCESS_KEY).toBeUndefined()
    expect(environment.PYTHONDONTWRITEBYTECODE).toBe('1')
    expect(environment.PYTHONNOUSERSITE).toBe('1')
    expect(environment.PYTHONUTF8).toBe('1')
  })

  it('points both body-model consumers at a bundled neutral SMPL-X model', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-runtime-body-models-'))
    const bodyModels = join(root, 'configs', 'body_models')
    const neutral = join(bodyModels, 'smplx', 'SMPLX_NEUTRAL.npz')
    mkdirSync(dirname(neutral), { recursive: true })
    writeFileSync(neutral, 'model', 'utf8')

    const environment = buildSidecarEnvironment(root, {}, bodyModels)

    expect(environment.HHTOOLS_BODY_MODELS).toBe(bodyModels)
    expect(environment.HHTOOLS_GVHMR_BODY_MODELS).toBe(bodyModels)
  })

  it('preserves explicit body-model environment overrides', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-runtime-body-model-overrides-'))
    const neutral = join(root, 'configs', 'body_models', 'smplx', 'SMPLX_NEUTRAL.npz')
    mkdirSync(dirname(neutral), { recursive: true })
    writeFileSync(neutral, 'model', 'utf8')

    const environment = buildSidecarEnvironment(root, {
      HHTOOLS_BODY_MODELS: '/custom/hhtools-models',
      HHTOOLS_GVHMR_BODY_MODELS: '/custom/gvhmr-models'
    })

    expect(environment.HHTOOLS_BODY_MODELS).toBe('/custom/hhtools-models')
    expect(environment.HHTOOLS_GVHMR_BODY_MODELS).toBe('/custom/gvhmr-models')
  })

  it('keeps managed runtimes isolated while exposing packaged robots', () => {
    const root = mkdtempSync(join(tmpdir(), 'hhtools-runtime-robot-assets-'))
    const robots = join(root, 'builtin', 'robots')
    mkdirSync(robots, { recursive: true })

    const environment = buildSidecarEnvironment(undefined, { PATH: '/usr/bin' }, undefined, robots)

    expect(environment.PYTHONPATH).toBeUndefined()
    expect(environment.HHTOOLS_ROBOT_PATH).toBe(robots)
    expect(environment.PATH).toBe('/usr/bin')
  })

  it('does not inject a host PYTHONPATH into a bundled runtime', () => {
    const environment = buildSidecarEnvironment(
      'C:\\Program Files\\HHTools\\resources\\runtime\\app',
      { PYTHONPATH: 'C:\\untrusted-host-package', PATH: 'C:\\Windows\\System32' },
      undefined,
      undefined,
      false
    )

    expect(environment.PYTHONPATH).toBe('C:\\Program Files\\HHTools\\resources\\runtime\\app')
  })
})
