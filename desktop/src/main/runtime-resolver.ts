/** Resolve a checkout, bundled Windows runtime, or managed Linux/macOS runtime. */
import { existsSync, readFileSync } from 'node:fs'
import { delimiter, dirname, isAbsolute, join, resolve } from 'node:path'

export type RuntimeKind = 'checkout' | 'bundled' | 'managed'

export interface RuntimeConfig {
  kind: RuntimeKind
  repoRoot?: string
  workingDirectory: string
  pythonExecutable: string
  sourceRoot: string
  saveDirectory: string
  cacheDirectory: string
  logDirectory: string
  bodyModelsRoot?: string
  bundledRobotRoot?: string
}

export interface ResolveRuntimeOptions {
  appPath: string
  cwd: string
  userData: string
  isPackaged?: boolean
  resourcesPath?: string
  appVersion?: string
  /** Override the conventional system runtime root in deterministic tests. */
  systemInstallRoot?: string | null
  env?: NodeJS.ProcessEnv
  /** Override the host platform in deterministic resolver tests. */
  platform?: NodeJS.Platform
}

export class RuntimeNotFoundError extends Error {
  constructor(message = 'No usable HHTools Python runtime is installed.') {
    super(message)
    this.name = 'RuntimeNotFoundError'
  }
}

function isRepositoryRoot(candidate: string): boolean {
  return existsSync(join(candidate, 'pyproject.toml')) && existsSync(join(candidate, 'hhtools'))
}

function walkForRepository(start: string): string | undefined {
  let current = resolve(start)
  while (true) {
    if (isRepositoryRoot(current)) return current
    const parent = dirname(current)
    if (parent === current) return undefined
    current = parent
  }
}

function configuredRepositoryRoot(env: NodeJS.ProcessEnv): string | undefined {
  const configured = env.HHTOOLS_REPO_ROOT
  if (configured === undefined) return undefined
  // An explicit path is authoritative; fail early instead of silently using another runtime.
  const resolved = resolve(configured)
  if (!isRepositoryRoot(resolved)) {
    throw new Error(`HHTOOLS_REPO_ROOT is not a hhtools checkout: ${resolved}`)
  }
  return resolved
}

function discoveredRepositoryRoot(options: ResolveRuntimeOptions): string | undefined {
  // Dev and unpacked builds normally live below the repository, so walking upward is enough.
  for (const candidate of [options.cwd, options.appPath, dirname(options.appPath)]) {
    const found = walkForRepository(candidate)
    if (found !== undefined) return found
  }
  return undefined
}

function resolvePython(
  repoRoot: string,
  env: NodeJS.ProcessEnv,
  platform: NodeJS.Platform
): string {
  if (env.HHTOOLS_PYTHON !== undefined) {
    if (isAbsolute(env.HHTOOLS_PYTHON) && !existsSync(env.HHTOOLS_PYTHON)) {
      throw new Error(`HHTOOLS_PYTHON does not exist: ${env.HHTOOLS_PYTHON}`)
    }
    return env.HHTOOLS_PYTHON
  }

  const candidates =
    platform === 'win32'
      ? [join(repoRoot, '.venv', 'Scripts', 'python.exe')]
      : [join(repoRoot, '.venv', 'bin', 'python')]
  const localPython = candidates.find((candidate) => existsSync(candidate))
  return localPython ?? (platform === 'win32' ? 'python' : 'python3')
}

function bundledRuntime(
  options: ResolveRuntimeOptions,
  platform: NodeJS.Platform
): { repoRoot: string; pythonExecutable: string; runtimeRoot: string } | undefined {
  if (!options.isPackaged || options.resourcesPath === undefined) return undefined
  const runtimeRoot = join(options.resourcesPath, 'runtime')
  const repoRoot = join(runtimeRoot, 'app')
  const pythonExecutable = platform === 'win32'
    ? join(runtimeRoot, 'python', 'python.exe')
    : join(runtimeRoot, 'python', 'bin', 'python3')
  if (!existsSync(repoRoot) && !existsSync(pythonExecutable)) return undefined
  if (!isRepositoryRoot(repoRoot) || !existsSync(pythonExecutable)) {
    throw new Error(`The bundled HHTools runtime is incomplete: ${runtimeRoot}`)
  }
  return { repoRoot, pythonExecutable, runtimeRoot }
}

function managedRuntimeRoots(
  env: NodeJS.ProcessEnv,
  platform: NodeJS.Platform,
  systemInstallRoot: string | null = '/opt/hhtools'
): string[] {
  if (platform !== 'linux' && platform !== 'darwin') return []
  const roots = [env.HHTOOLS_INSTALL_ROOT]
  if (platform === 'darwin') {
    if (env.HOME) roots.push(join(env.HOME, 'Library', 'Application Support', 'hhtools'))
  } else {
    if (env.XDG_DATA_HOME) roots.push(join(env.XDG_DATA_HOME, 'hhtools'))
    else if (env.HOME) roots.push(join(env.HOME, '.local', 'share', 'hhtools'))
    if (systemInstallRoot) roots.push(systemInstallRoot)
  }
  return [
    ...new Set(
      roots.filter((root): root is string => Boolean(root)).map((root) => resolve(root))
    )
  ]
}

function managedRuntime(
  env: NodeJS.ProcessEnv,
  platform: NodeJS.Platform,
  expectedVersion?: string,
  systemInstallRoot?: string | null
): { root: string; pythonExecutable: string } | undefined {
  for (const root of managedRuntimeRoots(env, platform, systemInstallRoot)) {
    const marker = join(root, 'runtime-version')
    const pythonExecutable = join(root, 'tools', 'hhtools', 'bin', 'python')
    if (!existsSync(marker) || !existsSync(pythonExecutable)) continue
    try {
      const installedVersion = readFileSync(marker, 'utf8').trim()
      if (
        installedVersion
        && (expectedVersion === undefined || installedVersion === expectedVersion)
      ) {
        return { root, pythonExecutable }
      }
    } catch {
      // A partial or unreadable installation is not a runtime candidate.
    }
  }
  return undefined
}

function packagedResource(options: ResolveRuntimeOptions, ...parts: string[]): string | undefined {
  if (!options.isPackaged || options.resourcesPath === undefined) return undefined
  const candidate = join(options.resourcesPath, ...parts)
  return existsSync(candidate) ? candidate : undefined
}

function packagedRuntimeIdentity(options: ResolveRuntimeOptions): string | undefined {
  const identityPath = packagedResource(options, 'bootstrap', 'RUNTIME_ID')
  if (!identityPath) return undefined
  let identity: string
  try {
    identity = readFileSync(identityPath, 'utf8').trim()
  } catch (error) {
    throw new Error(`Cannot read the packaged runtime identity: ${String(error)}`)
  }
  if (!/^[0-9A-Za-z._+-]{1,128}$/.test(identity)) {
    throw new Error(`The packaged runtime identity is invalid: ${identityPath}`)
  }
  return identity
}

export function resolveRuntime(options: ResolveRuntimeOptions): RuntimeConfig {
  const env = options.env ?? process.env
  const platform = options.platform ?? process.platform
  const explicitRepoRoot = configuredRepositoryRoot(env)
  const packaged = explicitRepoRoot === undefined ? bundledRuntime(options, platform) : undefined
  const expectedManagedIdentity = explicitRepoRoot === undefined && packaged === undefined
    ? packagedRuntimeIdentity(options) ?? options.appVersion
    : options.appVersion
  const managed = explicitRepoRoot === undefined && packaged === undefined && options.isPackaged
    ? managedRuntime(env, platform, expectedManagedIdentity, options.systemInstallRoot)
    : undefined
  const repoRoot = explicitRepoRoot
    ?? packaged?.repoRoot
    ?? (managed === undefined && !options.isPackaged ? discoveredRepositoryRoot(options) : undefined)
  if (repoRoot === undefined && managed === undefined) {
    throw new RuntimeNotFoundError(
      'HHTools needs a local Python runtime. Install it here or set HHTOOLS_REPO_ROOT.'
    )
  }
  const pythonExecutable = managed?.pythonExecutable
    ?? packaged?.pythonExecutable
    ?? resolvePython(repoRoot!, env, platform)
  const packagedBodyModels =
    options.isPackaged && options.resourcesPath
      ? join(options.resourcesPath, 'body_models')
      : undefined
  const bodyModelsRoot = packagedBodyModels && existsSync(
    join(packagedBodyModels, 'smplx', 'SMPLX_NEUTRAL.npz')
  )
    ? packagedBodyModels
    : undefined
  const bundledMotions = packagedResource(options, 'builtin', 'motions')
  const bundledRobotRoot = packagedResource(options, 'builtin', 'robots')
  const sourceRoot = resolve(
    env.HHTOOLS_SOURCE_ROOT
      ?? bundledMotions
      ?? (repoRoot ? join(repoRoot, 'assets', 'motions') : join(options.userData, 'motions'))
  )

  return {
    kind: managed ? 'managed' : packaged ? 'bundled' : 'checkout',
    repoRoot,
    workingDirectory: managed?.root ?? packaged?.runtimeRoot ?? repoRoot!,
    pythonExecutable,
    sourceRoot,
    saveDirectory: resolve(env.HHTOOLS_SAVE_DIR ?? join(options.userData, 'save_npz')),
    // Keep Python's generated assets separate from Electron/Chromium's Cache directory.
    cacheDirectory: resolve(env.HHTOOLS_CACHE_DIR ?? join(options.userData, 'hhtools-cache')),
    logDirectory: resolve(env.HHTOOLS_LOG_DIR ?? join(options.userData, 'logs')),
    bodyModelsRoot,
    bundledRobotRoot
  }
}

const ENV_ALLOWLIST = new Set([
  'APPDATA',
  'COMSPEC',
  'DBUS_SESSION_BUS_ADDRESS',
  'DISPLAY',
  'HOME',
  'HHTOOLS_MAX_QUEUED_JOBS',
  'HHTOOLS_MAX_RUNNING_JOBS',
  'HHTOOLS_MOTION_LIBRARY_ROOT',
  'HHTOOLS_MOTION_LIBRARY_SETTINGS_PATH',
  'HHTOOLS_BODY_MODELS',
  'HHTOOLS_GVHMR_BODY_MODELS',
  'HHTOOLS_GVHMR_IMAGE',
  'HHTOOLS_GVHMR_PYTHON',
  'HHTOOLS_GVHMR_ROOT',
  'HHTOOLS_GVHMR_TIMEOUT_SECONDS',
  'HHTOOLS_ROBOT_DIR',
  'HHTOOLS_ROBOT_PATH',
  'HHTOOLS_WEB_SETTINGS_PATH',
  'LOCALAPPDATA',
  'LD_LIBRARY_PATH',
  'MUJOCO_GL',
  'NUMBER_OF_PROCESSORS',
  'PATH',
  'PATHEXT',
  'PROCESSOR_ARCHITECTURE',
  'PROGRAMDATA',
  'PYOPENGL_PLATFORM',
  'SYSTEMDRIVE',
  'SYSTEMROOT',
  'TEMP',
  'TMP',
  'USERPROFILE',
  'VIRTUAL_ENV',
  'WINDIR',
  'XDG_CONFIG_HOME',
  'XDG_DATA_HOME',
  'XDG_RUNTIME_DIR',
  'WAYLAND_DISPLAY',
  'XAUTHORITY'
])

export function buildSidecarEnvironment(
  repoRoot: string | undefined,
  source: NodeJS.ProcessEnv = process.env,
  packagedBodyModels?: string,
  bundledRobotRoot?: string,
  includeParentPythonPath = true
): NodeJS.ProcessEnv {
  const result: NodeJS.ProcessEnv = {}

  // Do not forward the entire Electron environment. Keep OS/runtime variables plus GPU toolchains.
  for (const [key, value] of Object.entries(source)) {
    if (
      value !== undefined &&
      (ENV_ALLOWLIST.has(key.toUpperCase()) ||
        key.toUpperCase().startsWith('CUDA_') ||
        key.toUpperCase().startsWith('NVIDIA_') ||
        key.toUpperCase().startsWith('CONDA_'))
    ) {
      result[key] = value
    }
  }

  // Import the working checkout and make Python logs deterministic and immediately visible.
  if (repoRoot !== undefined) {
    result.PYTHONPATH = [repoRoot, includeParentPythonPath ? source.PYTHONPATH : undefined]
      .filter(Boolean)
      .join(delimiter)
  }
  result.PYTHONDONTWRITEBYTECODE = '1'
  result.PYTHONNOUSERSITE = '1'
  result.PYTHONUTF8 = '1'
  result.PYTHONUNBUFFERED = '1'

  const bundledBodyModels = packagedBodyModels
    ?? (repoRoot ? join(repoRoot, 'configs', 'body_models') : undefined)
  const bundledSmplxNeutral = bundledBodyModels
    ? join(bundledBodyModels, 'smplx', 'SMPLX_NEUTRAL.npz')
    : undefined
  if (bundledBodyModels && bundledSmplxNeutral && existsSync(bundledSmplxNeutral)) {
    if (source.HHTOOLS_BODY_MODELS === undefined) {
      result.HHTOOLS_BODY_MODELS = bundledBodyModels
    }
    if (source.HHTOOLS_GVHMR_BODY_MODELS === undefined) {
      result.HHTOOLS_GVHMR_BODY_MODELS = bundledBodyModels
    }
  }
  if (bundledRobotRoot && source.HHTOOLS_ROBOT_PATH === undefined) {
    result.HHTOOLS_ROBOT_PATH = bundledRobotRoot
  }
  return result
}
