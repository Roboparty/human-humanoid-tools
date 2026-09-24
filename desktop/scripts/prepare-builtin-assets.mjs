import { createHash } from 'node:crypto'
import {
  chmodSync,
  closeSync,
  copyFileSync,
  existsSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  readSync,
  realpathSync,
  renameSync,
  rmSync,
} from 'node:fs'
import { dirname, join, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const desktopRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const repositoryRoot = resolve(desktopRoot, '..')
const manifestPath = join(repositoryRoot, 'configs', 'builtin-assets.json')
const outputRoot = resolve(desktopRoot, '.builtin')
const stagingRoot = resolve(desktopRoot, `.builtin.staging-${process.pid}`)
const expectedMotionEntries = 30
const expectedMotionFiles = 58
const expectedRobots = 6

function fail(message) {
  throw new Error(`[prepare-builtin-assets] ${message}`)
}

function assertInside(root, path) {
  const candidate = relative(root, path)
  if (!candidate || candidate === '..' || candidate.startsWith(`..${sep}`)) {
    fail(`path escapes its expected root: ${path}`)
  }
}

function safeRelative(value, prefix) {
  if (
    typeof value !== 'string'
    || value.length === 0
    || value.includes('\\')
    || value.startsWith('/')
  ) {
    fail(`unsafe manifest path: ${String(value)}`)
  }
  const parts = value.split('/')
  if (parts.some((part) => part.length === 0 || part === '.' || part === '..')) {
    fail(`unsafe manifest path: ${value}`)
  }
  if (prefix !== undefined && !value.startsWith(`${prefix}/`)) {
    fail(`manifest path is outside ${prefix}: ${value}`)
  }
  return value
}

function readJson(path) {
  try {
    return JSON.parse(readFileSync(path, 'utf8'))
  } catch (error) {
    fail(`cannot read JSON ${path}: ${error instanceof Error ? error.message : String(error)}`)
  }
}

function requireRegularFile(path, root) {
  if (!existsSync(path)) fail(`missing file: ${path}`)
  const metadata = lstatSync(path)
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    fail(`expected a regular file: ${path}`)
  }
  assertInside(realpathSync(root), realpathSync(path))
  return metadata
}

function sha256(path) {
  const digest = createHash('sha256')
  const descriptor = openSync(path, 'r')
  const buffer = Buffer.allocUnsafe(1024 * 1024)
  try {
    while (true) {
      const bytesRead = readSync(descriptor, buffer, 0, buffer.length, null)
      if (bytesRead === 0) break
      digest.update(buffer.subarray(0, bytesRead))
    }
  } finally {
    closeSync(descriptor)
  }
  return digest.digest('hex')
}

function copyRegularFile(source, destination, sourceRoot) {
  requireRegularFile(source, sourceRoot)
  assertInside(stagingRoot, destination)
  mkdirSync(dirname(destination), { recursive: true })
  copyFileSync(source, destination)
  chmodSync(destination, 0o644)
}

function defaultRobotRoot() {
  const configured = process.env.HHTOOLS_BUNDLED_ROBOT_DIR
    ?? process.env.HHTOOLS_ROBOT_DIR
  if (configured) return resolve(configured)
  if (process.env.XDG_CONFIG_HOME) {
    return resolve(process.env.XDG_CONFIG_HOME, 'hhtools', 'robots')
  }
  if (process.platform === 'win32' && process.env.APPDATA) {
    return resolve(process.env.APPDATA, 'hhtools', 'robots')
  }
  if (process.env.HOME) return resolve(process.env.HOME, '.config', 'hhtools', 'robots')
  fail('cannot locate the curated Robot Library; set HHTOOLS_BUNDLED_ROBOT_DIR')
}

function validateRobotSource(robot, source) {
  const upstream = source?.source
  if (
    source?.preset !== robot.preset
    || source?.display_name !== robot.display_name
    || upstream?.repository !== robot.repository
    || upstream?.commit !== robot.commit
    || upstream?.main_urdf !== robot.main_urdf
    || JSON.stringify(upstream?.license_paths) !== JSON.stringify(robot.license_paths)
  ) {
    fail(`installed robot source differs from the allowlist: ${robot.preset}`)
  }
  if (!Array.isArray(source.installed_files) || source.installed_files.length === 0) {
    fail(`installed robot has no file allowlist: ${robot.preset}`)
  }
}

function stageMotions(manifest) {
  if (!Array.isArray(manifest.motions) || manifest.motions.length !== expectedMotionEntries) {
    fail(`expected ${expectedMotionEntries} motion entries`)
  }
  if (!Array.isArray(manifest.motion_shared_files)) {
    fail('motion_shared_files must be an array')
  }
  const paths = [
    ...manifest.motions.flatMap((motion) => {
      if (!Array.isArray(motion?.files) || !motion.files.includes(motion.primary)) {
        fail(`invalid motion entry: ${String(motion?.id)}`)
      }
      return motion.files
    }),
    ...manifest.motion_shared_files,
  ].map((path) => safeRelative(path, 'assets/motions'))
  if (paths.length !== expectedMotionFiles || new Set(paths).size !== paths.length) {
    fail(`expected ${expectedMotionFiles} unique motion files`)
  }
  if (paths.some((path) => path.includes('hmr4d_results'))) {
    fail('the broken GVHMR hmr4d_results artifact must not be bundled')
  }

  const motionRoot = join(repositoryRoot, 'assets', 'motions')
  let bytes = 0
  for (const path of paths) {
    const source = join(repositoryRoot, ...path.split('/'))
    const destination = join(stagingRoot, 'motions', ...path.split('/').slice(2))
    bytes += requireRegularFile(source, motionRoot).size
    copyRegularFile(source, destination, motionRoot)
  }
  return { entries: manifest.motions.length, files: paths.length, bytes }
}

function stageRobots(manifest) {
  if (!Array.isArray(manifest.robots) || manifest.robots.length !== expectedRobots) {
    fail(`expected ${expectedRobots} robots`)
  }
  const robotRoot = defaultRobotRoot()
  if (!existsSync(robotRoot)) {
    fail(`curated Robot Library not found at ${robotRoot}; run scripts/install_builtin_robots.py first`)
  }
  const presets = new Set()
  let files = 0
  let bytes = 0
  for (const robot of manifest.robots) {
    if (typeof robot?.preset !== 'string' || presets.has(robot.preset)) {
      fail(`duplicate or invalid robot preset: ${String(robot?.preset)}`)
    }
    presets.add(robot.preset)
    const sourceRoot = join(robotRoot, robot.preset)
    const sourceManifestPath = join(sourceRoot, safeRelative(robot.bundle_manifest))
    requireRegularFile(sourceManifestPath, sourceRoot)
    const sourceManifest = readJson(sourceManifestPath)
    validateRobotSource(robot, sourceManifest)

    const seen = new Set()
    for (const record of sourceManifest.installed_files) {
      const path = safeRelative(record?.path)
      if (seen.has(path)) fail(`duplicate installed robot file: ${robot.preset}/${path}`)
      seen.add(path)
      const source = join(sourceRoot, ...path.split('/'))
      const metadata = requireRegularFile(source, sourceRoot)
      if (
        !Number.isSafeInteger(record?.bytes)
        || metadata.size !== record.bytes
        || typeof record?.sha256 !== 'string'
        || !/^[a-f0-9]{64}$/.test(record.sha256)
        || sha256(source) !== record.sha256
      ) {
        fail(`installed robot file differs: ${robot.preset}/${path}`)
      }
      copyRegularFile(
        source,
        join(stagingRoot, 'robots', robot.preset, ...path.split('/')),
        sourceRoot,
      )
      files += 1
      bytes += metadata.size
    }
    if (!seen.has('robot.yaml')) fail(`installed robot has no robot.yaml: ${robot.preset}`)
    copyRegularFile(
      sourceManifestPath,
      join(stagingRoot, 'robots', robot.preset, 'SOURCE.json'),
      sourceRoot,
    )
  }
  return { entries: manifest.robots.length, files, bytes }
}

assertInside(desktopRoot, outputRoot)
assertInside(desktopRoot, stagingRoot)
requireRegularFile(manifestPath, repositoryRoot)
const manifest = readJson(manifestPath)
if (manifest?.schema_version !== 1 || manifest?.policy?.include_only_declared !== true) {
  fail('unsupported or unsafe built-in asset manifest')
}

rmSync(stagingRoot, { recursive: true, force: true })
try {
  mkdirSync(stagingRoot, { recursive: true })
  const motions = stageMotions(manifest)
  const robots = stageRobots(manifest)
  copyRegularFile(manifestPath, join(stagingRoot, 'MANIFEST.json'), repositoryRoot)
  rmSync(outputRoot, { recursive: true, force: true })
  renameSync(stagingRoot, outputRoot)
  const mib = (motions.bytes + robots.bytes) / 1024 / 1024
  console.log(
    `[prepare-builtin-assets] Ready: ${motions.entries} motions / ${motions.files} files, `
    + `${robots.entries} robots / ${robots.files} files (${mib.toFixed(1)} MiB)`,
  )
} catch (error) {
  rmSync(stagingRoot, { recursive: true, force: true })
  throw error
}
