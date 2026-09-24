import { afterEach, describe, expect, it } from 'vitest'

import type { LoggerLike } from '../src/main/desktop-logger'
import { findAvailablePort } from '../src/main/network'
import { SidecarSupervisor } from '../src/main/sidecar-supervisor'

const silentLogger: LoggerLike = {
  info: () => undefined,
  warn: () => undefined,
  error: () => undefined,
  processOutput: () => undefined
}

const supervisors: SidecarSupervisor[] = []

afterEach(async () => {
  await Promise.all(supervisors.splice(0).map((supervisor) => supervisor.stop()))
})

describe('SidecarSupervisor', () => {
  it('waits for HTTP readiness and stops the child process', async () => {
    const port = await findAvailablePort()
    const script = `
      const http = require('node:http');
      const server = http.createServer((request, response) => {
        if (request.url === '/api/health' && request.headers['x-hhtools-session'] === 'test') {
          response.writeHead(200, {'content-type': 'application/json'}); response.end('{"ok":true}');
        } else { response.writeHead(401); response.end(); }
      });
      setTimeout(() => server.listen(${port}, '127.0.0.1'), 150);
      process.on('SIGTERM', () => server.close(() => process.exit(0)));
    `
    const supervisor = new SidecarSupervisor(
      {
        command: process.execPath,
        args: ['-e', script],
        cwd: process.cwd(),
        env: process.env,
        origin: `http://127.0.0.1:${port}`,
        sessionSecret: 'test',
        startTimeoutMs: 5_000,
        stopTimeoutMs: 2_000
      },
      silentLogger
    )
    supervisors.push(supervisor)

    const ready = await supervisor.start()
    expect(ready.state).toBe('ready')
    expect(ready.pid).toBeTypeOf('number')

    await supervisor.stop()
    expect(supervisor.snapshot.state).toBe('stopped')
  })

  it('classifies a startup timeout as crashed', async () => {
    const port = await findAvailablePort()
    const supervisor = new SidecarSupervisor(
      {
        command: process.execPath,
        args: ['-e', 'setInterval(() => {}, 1000)'],
        cwd: process.cwd(),
        env: process.env,
        origin: `http://127.0.0.1:${port}`,
        sessionSecret: 'test',
        startTimeoutMs: 250,
        stopTimeoutMs: 500
      },
      silentLogger
    )
    supervisors.push(supervisor)

    await expect(supervisor.start()).rejects.toThrow(/Timed out/)
    expect(supervisor.snapshot.state).toBe('crashed')
  })
})
