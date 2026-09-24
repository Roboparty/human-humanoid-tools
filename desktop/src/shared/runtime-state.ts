export type AppPhase =
  | 'starting'
  | 'backend-starting'
  | 'ready'
  | 'after-window-open'
  | 'shutting-down'

export type SidecarState = 'stopped' | 'starting' | 'ready' | 'stopping' | 'crashed'

export interface RuntimeState {
  appPhase: AppPhase
  backendState: SidecarState
  backendOrigin?: string
  backendPid?: number
  error?: string
}
