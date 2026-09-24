import { createWriteStream, mkdirSync, type WriteStream } from 'node:fs'
import { join } from 'node:path'

export interface LoggerLike {
  info(message: string, details?: Record<string, unknown>): void
  warn(message: string, details?: Record<string, unknown>): void
  error(message: string, details?: Record<string, unknown>): void
  processOutput(stream: 'stdout' | 'stderr', text: string): void
}

export class DesktopLogger implements LoggerLike {
  readonly filePath: string
  private readonly stream: WriteStream

  constructor(logDirectory: string) {
    mkdirSync(logDirectory, { recursive: true })
    const stamp = new Date().toISOString().replaceAll(':', '-').replaceAll('.', '-')
    this.filePath = join(logDirectory, `desktop-${stamp}.log`)
    this.stream = createWriteStream(this.filePath, { encoding: 'utf8', flags: 'a' })
  }

  info(message: string, details?: Record<string, unknown>): void {
    this.write('info', message, details)
  }

  warn(message: string, details?: Record<string, unknown>): void {
    this.write('warn', message, details)
  }

  error(message: string, details?: Record<string, unknown>): void {
    this.write('error', message, details)
  }

  processOutput(stream: 'stdout' | 'stderr', text: string): void {
    for (const line of text.split(/\r?\n/)) {
      if (line.length > 0) this.write(`sidecar:${stream}`, line)
    }
  }

  close(): Promise<void> {
    return new Promise((resolve) => this.stream.end(resolve))
  }

  private write(level: string, message: string, details?: Record<string, unknown>): void {
    const suffix = details === undefined ? '' : ` ${JSON.stringify(details)}`
    this.stream.write(`${new Date().toISOString()} [${level}] ${message}${suffix}\n`)
  }
}
