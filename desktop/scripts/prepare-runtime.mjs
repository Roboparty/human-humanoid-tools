import { spawnSync } from 'node:child_process'
import {
  cpSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  realpathSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs'
import { basename, dirname, join, parse, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

import { assertPathInside, listApplicationSourceFiles } from './runtime-staging-policy.mjs'

const desktopRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const repositoryRoot = resolve(desktopRoot, '..')
const outputRoot = resolve(desktopRoot, '.runtime')
const stagingRoot = resolve(desktopRoot, `.runtime.staging-${process.pid}`)
const runtimePython = join(stagingRoot, 'python')
const runtimeApplication = join(stagingRoot, 'app')

function fail(message) {
  throw new Error(`[prepare-runtime] ${message}`)
}

function assertSafeTargets() {
  for (const candidate of [outputRoot, stagingRoot]) {
    if (dirname(candidate) !== desktopRoot || basename(candidate).startsWith('.runtime') === false) {
      fail(`refusing to replace unexpected staging path: ${candidate}`)
    }
  }
}

function readVirtualEnvironmentConfiguration() {
  const path = join(repositoryRoot, '.venv', 'pyvenv.cfg')
  if (!existsSync(path)) fail(`missing ${path}; run uv sync --extra all --no-dev before packaging`)
  const values = {}
  for (const line of readFileSync(path, 'utf8').split(/\r?\n/)) {
    const match = line.match(/^\s*([^#=]+?)\s*=\s*(.*?)\s*$/)
    if (match) values[match[1].toLowerCase()] = match[2]
  }
  return { path, values }
}

function existingRealPath(candidate, description) {
  const resolved = resolve(candidate)
  if (!existsSync(resolved)) fail(`${description} is missing: ${resolved}`)
  return realpathSync(resolved)
}

function resolvePythonHome(configuration) {
  const configured = process.env.HHTOOLS_RUNTIME_PYTHON_HOME ?? configuration.values.home
  if (!configured) fail(`unable to read the base Python path from ${configuration.path}`)
  const pythonHome = existingRealPath(configured, 'Python home')
  if (pythonHome === parse(pythonHome).root) {
    fail(`refusing to stage a filesystem root as Python home: ${pythonHome}`)
  }
  return pythonHome
}

function virtualEnvironmentPython() {
  return join(repositoryRoot, '.venv', 'Scripts', 'python.exe')
}

function resolveSitePackages() {
  const override = process.env.HHTOOLS_RUNTIME_SITE_PACKAGES
  if (override) return existingRealPath(override, 'site-packages directory')

  const interpreter = virtualEnvironmentPython()
  if (!existsSync(interpreter)) fail(`Windows virtual-environment Python is missing: ${interpreter}`)
  const discovery = spawnSync(
    interpreter,
    ['-I', '-c', "import sysconfig; print(sysconfig.get_path('purelib'))"],
    { cwd: repositoryRoot, encoding: 'utf8' },
  )
  const discovered = discovery.stdout?.trim().split(/\r?\n/).at(-1)
  if (discovery.status !== 0 || !discovered) {
    fail(`unable to discover site-packages:\n${discovery.stdout}\n${discovery.stderr}`)
  }
  return existingRealPath(discovered, 'site-packages directory')
}

function shouldCopyPythonBase(sourceRoot, sourcePath) {
  const path = relative(sourceRoot, sourcePath).replaceAll('\\', '/')
  if (!path) return true
  if (/^Lib\/site-packages(?:\/|$)/i.test(path)) return false
  return !path.split('/').includes('__pycache__') && !path.endsWith('.pyc')
}

const developmentPackagePrefixes = [
  '__editable__.hhtools-',
  '__editable___hhtools_',
  '_pytest',
  'a1_coverage.pth',
  'coverage',
  'mypy',
  'mypyc',
  'pytest',
  'pytest_cov',
  'ruff',
]

function shouldCopySitePackage(sourceRoot, sourcePath) {
  const path = relative(sourceRoot, sourcePath).replaceAll('\\', '/')
  if (!path) return true
  const [topLevel] = path.split('/')
  if (developmentPackagePrefixes.some((prefix) => topLevel.startsWith(prefix))) return false
  if (topLevel === '_virtualenv.pth' || topLevel === '_virtualenv.py') return false
  return !path.split('/').includes('__pycache__') && !path.endsWith('.pyc')
}

function copyDirectory(source, destination, filter) {
  if (!existsSync(source)) fail(`required directory is missing: ${source}`)
  cpSync(source, destination, {
    recursive: true,
    force: true,
    filter: (candidate) => filter(source, candidate),
  })
}

function shouldCopyApplication(path) {
  const normalized = path.replaceAll('\\', '/')
  const segments = normalized.split('/')
  if (segments.includes('__pycache__') || segments.includes('node_modules')) return false
  if (normalized.endsWith('.pyc') || normalized.endsWith('.pyo') || normalized.endsWith('.map')) {
    return false
  }
  return !/\/(SMPL|SMPLH|SMPLX)_[^/]+\.(npz|pkl)$/i.test(`/${normalized}`)
}

const applicationInputs = [
  'hhtools',
  'configs',
  'assets/reference_poses',
  'docker/gvhmr',
  'LICENSE',
  'README.md',
  'pyproject.toml',
]

function copyApplicationFiles(selection) {
  let copied = 0
  for (const file of selection.files) {
    if (!shouldCopyApplication(file)) continue
    const source = join(repositoryRoot, file)
    const metadata = lstatSync(source)
    if (!metadata.isFile() || metadata.isSymbolicLink()) {
      fail(`application input must be a regular file: ${source}`)
    }
    const destination = join(runtimeApplication, file)
    assertPathInside(runtimeApplication, destination, 'application file escapes runtime root', {
      allowRoot: false,
    })
    mkdirSync(dirname(destination), { recursive: true })
    cpSync(source, destination, { force: true })
    copied += 1
  }
  return copied
}

function treeSummary(root) {
  let files = 0
  let bytes = 0
  const pending = [root]
  while (pending.length > 0) {
    const current = pending.pop()
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const path = join(current, entry.name)
      if (entry.isDirectory()) pending.push(path)
      else if (entry.isFile()) {
        files += 1
        bytes += statSync(path).size
      }
    }
  }
  return { files, bytes }
}

function assertNoSymlinks(root) {
  const pending = [root]
  while (pending.length > 0) {
    const current = pending.pop()
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const path = join(current, entry.name)
      if (entry.isSymbolicLink()) fail(`packaged runtime contains a symbolic link: ${path}`)
      if (entry.isDirectory()) pending.push(path)
    }
  }
}

function verifyRuntime() {
  const pythonExecutable = join(runtimePython, 'python.exe')
  if (!existsSync(pythonExecutable)) fail(`packaged Python executable is missing: ${pythonExecutable}`)
  const verification = spawnSync(
    pythonExecutable,
    [
      '-c',
      [
        'import fastapi, hhtools, mujoco, newton, torch, warp',
        'from hhtools.web.server import create_app',
        "print('runtime-ok', torch.__version__, mujoco.__version__)",
      ].join('; '),
    ],
    {
      cwd: runtimeApplication,
      encoding: 'utf8',
      env: {
        ...process.env,
        PYTHONNOUSERSITE: '1',
        PYTHONDONTWRITEBYTECODE: '1',
        PYTHONPATH: runtimeApplication,
        PYTHONUTF8: '1',
      },
    },
  )
  if (verification.status !== 0) {
    fail(`bundled Python import check failed:\n${verification.stdout}\n${verification.stderr}`)
  }
  return verification.stdout.trim()
}

if (process.platform !== 'win32') {
  fail('the bundled runtime is a Windows build input and must be staged on Windows')
}
assertSafeTargets()
rmSync(stagingRoot, { recursive: true, force: true })
try {
  mkdirSync(runtimeApplication, { recursive: true })
  const configuration = readVirtualEnvironmentConfiguration()
  const pythonHome = resolvePythonHome(configuration)
  const sitePackages = resolveSitePackages()
  console.log(`[prepare-runtime] Python: ${pythonHome}`)
  copyDirectory(pythonHome, runtimePython, shouldCopyPythonBase)
  copyDirectory(sitePackages, join(runtimePython, 'Lib', 'site-packages'), shouldCopySitePackage)
  const application = listApplicationSourceFiles(repositoryRoot, applicationInputs, process.env)
  const applicationFileCount = copyApplicationFiles(application)
  assertNoSymlinks(stagingRoot)
  const verification = verifyRuntime()
  const summary = treeSummary(stagingRoot)
  writeFileSync(
    join(stagingRoot, 'runtime-manifest.json'),
    `${JSON.stringify({
      schemaVersion: 1,
      platform: process.platform,
      applicationSourceProvenance: application.provenance,
      applicationFileCount,
      files: summary.files,
      bytes: summary.bytes,
    }, null, 2)}\n`,
    'utf8',
  )
  rmSync(outputRoot, { recursive: true, force: true })
  renameSync(stagingRoot, outputRoot)
  console.log(verification)
  console.log(
    `[prepare-runtime] Ready: ${summary.files.toLocaleString()} files, `
      + `${(summary.bytes / 1024 / 1024).toFixed(1)} MiB.`,
  )
} catch (error) {
  rmSync(stagingRoot, { recursive: true, force: true })
  throw error
}
