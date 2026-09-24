export type RuntimeInstallMode = 'user' | 'system'

export type RuntimeInstallPhase =
  | 'idle'
  | 'preparing'
  | 'installing'
  | 'verifying'
  | 'completed'
  | 'failed'
  | 'cancelled'

export interface RuntimeInstallProgress {
  phase: RuntimeInstallPhase
  message: string
  detail?: string
}

export interface RuntimeInstallResult {
  status: 'completed' | 'failed' | 'cancelled'
  message: string
}

export interface HHToolsInstallerApi {
  start(mode: RuntimeInstallMode): Promise<RuntimeInstallResult>
  cancel(): Promise<void>
  onProgress(listener: (progress: RuntimeInstallProgress) => void): () => void
}

export const INSTALLER_CHANNELS = {
  start: 'hhtools:installer:start',
  cancel: 'hhtools:installer:cancel',
  progress: 'hhtools:installer:progress'
} as const

declare global {
  interface Window {
    hhtoolsInstaller: HHToolsInstallerApi
  }
}
