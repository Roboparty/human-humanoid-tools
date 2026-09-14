// electronDist and the bundled uv are native build inputs, not cross-compiled payloads.
if (process.platform !== 'darwin' || process.arch !== 'arm64') {
  throw new Error('dist:mac requires an Apple Silicon Mac and an arm64 Node.js installation')
}
