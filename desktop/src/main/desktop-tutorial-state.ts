import { existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

const TUTORIAL_SEEN_MARKER = 'tutorial.seen'

export class DesktopTutorialState {
  private readonly markerPath: string

  constructor(userData: string) {
    this.markerPath = join(userData, TUTORIAL_SEEN_MARKER)
  }

  hasSeenTutorial(): boolean {
    return existsSync(this.markerPath)
  }

  markTutorialSeen(): void {
    mkdirSync(dirname(this.markerPath), { recursive: true })
    writeFileSync(this.markerPath, '1\n', 'utf8')
  }
}
