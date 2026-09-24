import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

export interface RectangleLike {
  x: number
  y: number
  width: number
  height: number
}

export interface DisplayLike {
  workArea: RectangleLike
}

export interface PersistedWindowState extends RectangleLike {
  maximized: boolean
}

const DEFAULT_WIDTH = 1440
const DEFAULT_HEIGHT = 900
const MIN_WIDTH = 1024
const MIN_HEIGHT = 700

function isFiniteRectangle(value: Partial<RectangleLike>): value is RectangleLike {
  return [value.x, value.y, value.width, value.height].every(
    (item) => typeof item === 'number' && Number.isFinite(item)
  )
}

function intersects(a: RectangleLike, b: RectangleLike): boolean {
  const width = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x)
  const height = Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y)
  return width >= 80 && height >= 80
}

function centered(primary: DisplayLike): PersistedWindowState {
  const width = Math.min(DEFAULT_WIDTH, primary.workArea.width)
  const height = Math.min(DEFAULT_HEIGHT, primary.workArea.height)
  return {
    x: Math.round(primary.workArea.x + (primary.workArea.width - width) / 2),
    y: Math.round(primary.workArea.y + (primary.workArea.height - height) / 2),
    width,
    height,
    maximized: false
  }
}

export function normalizeWindowState(
  candidate: Partial<PersistedWindowState> | undefined,
  displays: DisplayLike[],
  primary: DisplayLike
): PersistedWindowState {
  if (
    candidate === undefined ||
    !isFiniteRectangle(candidate) ||
    candidate.width < MIN_WIDTH ||
    candidate.height < MIN_HEIGHT
  ) {
    return centered(primary)
  }

  const state: PersistedWindowState = {
    ...candidate,
    maximized: (candidate as Partial<PersistedWindowState>).maximized === true
  }

  // Reset windows saved on a disconnected monitor instead of reopening them off-screen.
  return displays.some((display) => intersects(state, display.workArea)) ? state : centered(primary)
}

export class WindowStateStore {
  constructor(private readonly filePath: string) {}

  load(displays: DisplayLike[], primary: DisplayLike): PersistedWindowState {
    let candidate: Partial<PersistedWindowState> | undefined
    if (existsSync(this.filePath)) {
      try {
        candidate = JSON.parse(readFileSync(this.filePath, 'utf8')) as Partial<PersistedWindowState>
      } catch {
        candidate = undefined
      }
    }
    return normalizeWindowState(candidate, displays, primary)
  }

  save(state: PersistedWindowState): void {
    mkdirSync(dirname(this.filePath), { recursive: true })
    writeFileSync(this.filePath, `${JSON.stringify(state, null, 2)}\n`, 'utf8')
  }
}
