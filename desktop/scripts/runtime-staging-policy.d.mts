export interface ApplicationSourceSelection {
  provenance: 'git-tracked-worktree' | 'trusted-archive'
  files: string[]
}

export function listApplicationSourceFiles(
  repositoryRoot: string,
  inputs: string[],
  env?: NodeJS.ProcessEnv
): ApplicationSourceSelection

export function assertPathInside(
  root: string,
  candidate: string,
  description: string,
  options?: { allowRoot?: boolean }
): void
