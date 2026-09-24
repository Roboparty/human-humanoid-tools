import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import {
  chmodSync,
  copyFileSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  realpathSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, dirname, join, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const desktopRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const repositoryRoot = resolve(desktopRoot, '..')
const defaultOutputRoot = resolve(desktopRoot, '.bootstrap')
const outputRoot = process.env.HHTOOLS_DESKTOP_BOOTSTRAP_OUTPUT_DIR
  ? resolve(process.env.HHTOOLS_DESKTOP_BOOTSTRAP_OUTPUT_DIR)
  : defaultOutputRoot
const stagingRoot = `${outputRoot}.staging-${process.pid}`
const templatePath = join(desktopRoot, 'scripts', 'install-packaged-runtime.sh')
const uvConfiguration = join(repositoryRoot, 'packaging', 'installer', 'uv.toml')
const uvVersion = '0.12.9'

// The embedded uv and locked native wheels must match the machine building the package.
if (!['linux:x64', 'darwin:arm64'].includes(`${process.platform}:${process.arch}`)) {
  throw new Error('[prepare-bootstrap] build on Linux x86_64 or macOS Apple Silicon')
}

function fail(message) {
  throw new Error(`[prepare-bootstrap] ${message}`)
}

function assertInside(root, path) {
  const candidate = relative(root, path)
  if (!candidate || candidate === '..' || candidate.startsWith(`..${sep}`)) {
    fail(`path escapes its expected root: ${path}`)
  }
}

function regularFile(path, description) {
  if (!existsSync(path)) fail(`${description} is missing: ${path}`)
  const metadata = lstatSync(path)
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    fail(`${description} must be a regular file: ${path}`)
  }
  return realpathSync(path)
}

function copyRegular(source, destination, description) {
  const canonical = regularFile(source, description)
  assertInside(stagingRoot, destination)
  mkdirSync(dirname(destination), { recursive: true })
  copyFileSync(canonical, destination)
  chmodSync(destination, 0o644)
}

function run(command, args, cwd = repositoryRoot) {
  const result = spawnSync(command, args, {
    cwd,
    encoding: 'utf8',
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  if (result.status !== 0) {
    fail(
      `command failed (${basename(command)} ${args.join(' ')}):\n`
        + `${result.stdout ?? ''}${result.stderr ?? ''}`,
    )
  }
  return String(result.stdout ?? '').trim()
}

function resolveUv(fixtureRoot) {
  if (fixtureRoot) return regularFile(join(fixtureRoot, 'bin', 'uv'), 'fixture uv')
  const configured = process.env.HHTOOLS_PACKAGING_UV_BIN
  // Package managers such as Homebrew expose uv through a symlink. Copy its real binary.
  if (configured) return regularFile(realpathSync(resolve(configured)), 'packaging uv')
  const discovered = spawnSync('which', ['uv'], { encoding: 'utf8' })
  const path = discovered.status === 0 ? discovered.stdout.trim() : ''
  if (!path) fail('uv is not on PATH; set HHTOOLS_PACKAGING_UV_BIN')
  return regularFile(realpathSync(path), 'packaging uv')
}

function sha256(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex')
}

function stageFixture(fixtureRoot, assetsRoot) {
  const fixtureAssets = join(fixtureRoot, 'assets')
  if (!existsSync(fixtureAssets)) fail(`fixture assets are missing: ${fixtureAssets}`)
  for (const name of ['requirements-all.txt', 'installer-uv.toml']) {
    copyRegular(join(fixtureAssets, name), join(assetsRoot, name), `fixture ${name}`)
  }
  const wheels = readdirSync(fixtureAssets).filter(
    (name) => /^hhtools-.+-py3-none-any\.whl$/.test(name)
  )
  if (wheels.length !== 1) fail(`fixture must contain exactly one HHTools wheel, got ${wheels.length}`)
  copyRegular(join(fixtureAssets, wheels[0]), join(assetsRoot, wheels[0]), 'fixture wheel')
}

function buildAssets(uv, assetsRoot) {
  run(uv, [
    'build',
    '--wheel',
    '--clear',
    '--force-pep517',
    '--out-dir',
    assetsRoot,
  ])
  run(uv, [
    'export',
    '--frozen',
    '--extra',
    'all',
    '--no-dev',
    '--no-emit-project',
    '--format',
    'requirements.txt',
    '--no-header',
    '--output-file',
    join(assetsRoot, 'requirements-all.txt'),
  ])
  copyRegular(
    uvConfiguration,
    join(assetsRoot, 'installer-uv.toml'),
    'uv installer configuration',
  )
  rmSync(join(assetsRoot, '.gitignore'), { force: true })
}

const packageJson = JSON.parse(readFileSync(join(desktopRoot, 'package.json'), 'utf8'))
const version = String(packageJson.version)
if (!/^[0-9A-Za-z._+-]+$/.test(version)) fail(`invalid desktop version: ${version}`)
const fixtureRoot = process.env.HHTOOLS_DESKTOP_BOOTSTRAP_ASSET_DIR
  ? resolve(process.env.HHTOOLS_DESKTOP_BOOTSTRAP_ASSET_DIR)
  : undefined
const uv = resolveUv(fixtureRoot)
const reportedUvVersion = run(uv, ['--version'], dirname(uv))
if (!reportedUvVersion.startsWith(`uv ${uvVersion}`)) {
  fail(`expected uv ${uvVersion}, got ${reportedUvVersion || 'no version output'}`)
}

if (outputRoot === defaultOutputRoot) {
  assertInside(desktopRoot, outputRoot)
  assertInside(desktopRoot, stagingRoot)
} else {
  const temporaryRoot = realpathSync(tmpdir())
  // macOS exposes the same temporary directory through /var and /private/var.
  const outputParent = realpathSync(dirname(outputRoot))
  assertInside(temporaryRoot, join(outputParent, basename(outputRoot)))
  assertInside(temporaryRoot, join(outputParent, basename(stagingRoot)))
}
regularFile(templatePath, 'packaged runtime installer')
rmSync(stagingRoot, { recursive: true, force: true })
try {
  const assetsRoot = join(stagingRoot, 'assets')
  const uvDestination = join(stagingRoot, 'bin', 'uv')
  mkdirSync(assetsRoot, { recursive: true })
  if (fixtureRoot) stageFixture(fixtureRoot, assetsRoot)
  else buildAssets(uv, assetsRoot)
  copyRegular(uv, uvDestination, 'uv executable')
  chmodSync(uvDestination, 0o755)

  const wheels = readdirSync(assetsRoot).filter(
    (name) => /^hhtools-.+-py3-none-any\.whl$/.test(name)
  )
  if (wheels.length !== 1) fail(`expected exactly one HHTools wheel, got ${wheels.length}`)
  const wheelName = wheels[0]
  const expectedPrefix = `hhtools-${version.replaceAll('-', '_')}-`
  if (!wheelName.startsWith(expectedPrefix)) {
    fail(`wheel ${wheelName} does not match desktop version ${version}`)
  }

  const template = readFileSync(templatePath, 'utf8')
  const payloadPaths = [
    `assets/${wheelName}`,
    'assets/requirements-all.txt',
    'assets/installer-uv.toml',
    'bin/uv',
  ]
  const payloadHashes = new Map(
    payloadPaths.map((path) => [path, sha256(join(stagingRoot, path))])
  )
  const runtimeDigest = createHash('sha256')
  for (const path of payloadPaths) runtimeDigest.update(`${path}\0${payloadHashes.get(path)}\n`)
  runtimeDigest.update(`installer-template\0${sha256(templatePath)}\n`)
  const runtimeId = `${version}+sha256.${runtimeDigest.digest('hex').slice(0, 20)}`
  writeFileSync(join(stagingRoot, 'RUNTIME_ID'), `${runtimeId}\n`, 'utf8')

  for (const field of [
    'embedded_version', 'embedded_wheel', 'embedded_runtime_id', 'embedded_platform', 'embedded_arch',
  ]) {
    if (!new RegExp(`^${field}=.*$`, 'm').test(template)) {
      fail(`packaged runtime installer has no ${field} field`)
    }
  }
  const installer = template
    .replace(/^embedded_version=.*$/m, `embedded_version='${version}'`)
    .replace(/^embedded_wheel=.*$/m, `embedded_wheel='${wheelName}'`)
    .replace(/^embedded_runtime_id=.*$/m, `embedded_runtime_id='${runtimeId}'`)
    .replace(/^embedded_platform=.*$/m, `embedded_platform='${process.platform === 'darwin' ? 'Darwin' : 'Linux'}'`)
    .replace(/^embedded_arch=.*$/m, `embedded_arch='${process.arch === 'arm64' ? 'arm64' : 'x86_64'}'`)
  if (/(^|[;&|]\s*)curl(?:\s|$)/m.test(installer)) {
    fail('packaged runtime installer must not invoke curl')
  }
  const installerPath = join(stagingRoot, 'install.sh')
  writeFileSync(installerPath, installer, 'utf8')
  chmodSync(installerPath, 0o755)

  const checksumPaths = [...payloadPaths, 'RUNTIME_ID']
  writeFileSync(
    join(stagingRoot, 'SHA256SUMS'),
    `${checksumPaths.map((path) => `${payloadHashes.get(path) ?? sha256(join(stagingRoot, path))}  ${path}`).join('\n')}\n`,
    'utf8',
  )

  rmSync(outputRoot, { recursive: true, force: true })
  renameSync(stagingRoot, outputRoot)
  const wheelBytes = lstatSync(join(outputRoot, 'assets', wheelName)).size
  const uvBytes = lstatSync(join(outputRoot, 'bin', 'uv')).size
  console.log(
    `[prepare-bootstrap] Ready: ${runtimeId}, ${wheelName} `
      + `(${(wheelBytes / 1024 / 1024).toFixed(1)} MiB), uv ${uvVersion} `
      + `(${(uvBytes / 1024 / 1024).toFixed(1)} MiB)`,
  )
} catch (error) {
  rmSync(stagingRoot, { recursive: true, force: true })
  throw error
}
