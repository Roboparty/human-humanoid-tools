import type { RuntimeState } from './runtime-state'

export const DESKTOP_CHANNELS = {
  getRuntimeState: 'hhtools:get-runtime-state',
  getOptionalComponents: 'hhtools:get-optional-components',
  restartBackend: 'hhtools:restart-backend',
  setupGvhmr: 'hhtools:setup-gvhmr',
  selectDirectory: 'hhtools:select-directory',
  openExternal: 'hhtools:open-external',
  exitApplication: 'hhtools:exit-application',
  hasSeenTutorial: 'hhtools:has-seen-tutorial',
  markTutorialSeen: 'hhtools:mark-tutorial-seen',
  runtimeStateChanged: 'hhtools:runtime-state-changed'
} as const

export interface GvhmrOptionalComponentState {
  requested: boolean
  configured: boolean
  root?: string
  python?: string
  bodyModels?: string
  runtime: 'local' | 'docker'
  guideUrl: string
  estimatedAdditionalBytes: number
}

export interface OptionalComponentsState {
  gvhmr: GvhmrOptionalComponentState
}

export interface GvhmrSetupResult {
  action: 'cancelled' | 'configured' | 'guide-opened'
  state: GvhmrOptionalComponentState
}

export interface HHToolsDesktopApi {
  getRuntimeState(): Promise<RuntimeState>
  getOptionalComponents(): Promise<OptionalComponentsState>
  restartBackend(): Promise<RuntimeState>
  setupGvhmr(): Promise<GvhmrSetupResult>
  selectDirectory(): Promise<string | null>
  openExternal(url: string): Promise<void>
  exitApplication(): Promise<void>
  hasSeenTutorial(): Promise<boolean>
  markTutorialSeen(): Promise<void>
  onRuntimeStateChanged(listener: (state: RuntimeState) => void): () => void
}

declare global {
  interface Window {
    hhtoolsDesktop: HHToolsDesktopApi
  }
}
