import { execFileSync, spawnSync } from 'node:child_process'
import { existsSync, lstatSync, readdirSync, realpathSync } from 'node:fs'
import { isAbsolute, join, relative, sep } from 'node:path'

function fail(message) {
  throw new Error(`[prepare-runtime] ${message}`)
}

function collectArchiveFiles(repositoryRoot, relativePath, output) {
  const source = join(repositoryRoot, relativePath)
  let metadata
  try {
    metadata = lstatSync(source)
  } catch {
    fail(`required archive input is missing: ${source}`)
  }
  if (!metadata.isDirectory()) {
    output.push(relativePath.replaceAll('\\', '/'))
    return
  }
  for (const entry of readdirSync(source, { withFileTypes: true })) {
    collectArchiveFiles(repositoryRoot, join(relativePath, entry.name), output)
  }
}

/** Select application inputs from this checkout's own Git index. */
export function listApplicationSourceFiles(repositoryRoot, inputs, env = process.env) {
  const repositoryCheck = spawnSync(
    'git',
    ['rev-parse', '--show-toplevel'],
    { cwd: repositoryRoot, encoding: 'utf8' },
  )
  const reportedRoot = repositoryCheck.status === 0
    ? repositoryCheck.stdout.trim()
    : undefined
  const ownsGitIndex = reportedRoot
    && existsSync(reportedRoot)
    && realpathSync(reportedRoot) === realpathSync(repositoryRoot)

  if (ownsGitIndex) {
    const output = execFileSync(
      'git',
      ['ls-files', '-z', '--', ...inputs],
      { cwd: repositoryRoot, encoding: 'buffer' },
    )
    return {
      provenance: 'git-tracked-worktree',
      files: output.toString('utf8').split('\0').filter(Boolean),
    }
  }

  if (env.HHTOOLS_TRUST_SOURCE_ARCHIVE !== '1') {
    fail(
      'source tree has no matching Git index; package a checkout, or set '
        + 'HHTOOLS_TRUST_SOURCE_ARCHIVE=1 only for a verified `git archive` extraction',
    )
  }
  const files = []
  for (const input of inputs) collectArchiveFiles(repositoryRoot, input, files)
  return { provenance: 'trusted-archive', files }
}

export function assertPathInside(root, candidate, description, { allowRoot = true } = {}) {
  const escaped = relative(root, candidate)
  if (
    (!allowRoot && !escaped)
    || escaped === '..'
    || escaped.startsWith(`..${sep}`)
    || isAbsolute(escaped)
  ) {
    fail(`${description}: ${candidate}`)
  }
}
