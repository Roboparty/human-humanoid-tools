import {
  chmodSync,
  copyFileSync,
  existsSync,
  lstatSync,
  mkdirSync,
  realpathSync,
  rmSync,
} from 'node:fs'
import { dirname, join, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const desktopRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const repositoryRoot = resolve(desktopRoot, '..')
const stagingRoot = resolve(desktopRoot, '.models')
const source = join(repositoryRoot, 'configs', 'body_models', 'smplx', 'SMPLX_NEUTRAL.npz')
const destination = join(stagingRoot, 'smplx', 'SMPLX_NEUTRAL.npz')

function fail(message) {
  throw new Error(`[prepare-models] ${message}`)
}

function assertInside(root, path) {
  const candidate = relative(root, path)
  if (!candidate || candidate === '..' || candidate.startsWith(`..${sep}`)) {
    fail(`path escapes its expected root: ${path}`)
  }
}

function requireRegularFile(path) {
  if (!existsSync(path)) fail(`missing ${path}`)
  const metadata = lstatSync(path)
  if (metadata.isSymbolicLink() || !metadata.isFile()) {
    fail(`model must be a regular file: ${path}`)
  }
}

assertInside(desktopRoot, stagingRoot)
requireRegularFile(source)
assertInside(repositoryRoot, realpathSync(source))

rmSync(stagingRoot, { recursive: true, force: true })
mkdirSync(dirname(destination), { recursive: true })
copyFileSync(source, destination)
// Debian installs resources as root, so the desktop user needs explicit read access.
chmodSync(destination, 0o644)

const bytes = lstatSync(destination).size
console.log(`[prepare-models] Ready: SMPLX_NEUTRAL.npz (${(bytes / 1024 / 1024).toFixed(1)} MiB)`)
