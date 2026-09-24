import { createServer } from 'node:net'

export function findAvailablePort(host = '127.0.0.1'): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer()
    server.unref()
    server.once('error', reject)
    server.listen(0, host, () => {
      const address = server.address()
      if (address === null || typeof address === 'string') {
        server.close()
        reject(new Error('Unable to allocate a localhost port'))
        return
      }
      const { port } = address
      server.close((error) => (error === undefined ? resolve(port) : reject(error)))
    })
  })
}
