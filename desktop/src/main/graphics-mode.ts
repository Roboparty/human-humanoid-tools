/**
 * Internal Linux graphics-mode selection.
 *
 * hhtools normally lets Chromium use the host GPU. If a real WebGL2 probe
 * fails, the app relaunches once with the private argument below and uses the
 * SwiftShader libraries bundled with Electron. Keeping the argument private
 * avoids forcing software rendering on machines with a working GPU.
 */
export const SOFTWARE_RENDERING_ARGUMENT = '--hhtools-software-rendering'

export interface CommandLineSwitches {
  appendSwitch(name: string, value?: string): void
}

export type GraphicsStartupAction = 'start' | 'relaunch' | 'fail'

export function softwareRenderingRequested(
  argv: readonly string[] = process.argv,
  platform: NodeJS.Platform = process.platform
): boolean {
  return platform === 'linux' && argv.includes(SOFTWARE_RENDERING_ARGUMENT)
}

/** Must run before Electron's ready event so Chromium receives the GL flags. */
export function configureGraphicsCommandLine(
  commandLine: CommandLineSwitches,
  argv: readonly string[] = process.argv,
  platform: NodeJS.Platform = process.platform
): boolean {
  const softwareRendering = softwareRenderingRequested(argv, platform)
  if (softwareRendering) {
    // SwANGLE is Chromium's supported software GLES path. `--disable-gpu` is
    // intentionally not used because the WebUI itself requires WebGL2. Modern
    // Chromium also requires an explicit opt-in before trusted content may use
    // its lower-security software WebGL implementation. The switch is confined
    // to this fallback launch; the BrowserWindow still uses the sandbox and is
    // restricted to hhtools' authenticated localhost origin.
    commandLine.appendSwitch('use-gl', 'angle')
    commandLine.appendSwitch('use-angle', 'swiftshader')
    commandLine.appendSwitch('enable-unsafe-swiftshader')
  }
  return softwareRendering
}

export function decideGraphicsStartup(
  webgl2Available: boolean,
  softwareRendering: boolean,
  platform: NodeJS.Platform = process.platform
): GraphicsStartupAction {
  if (webgl2Available) return 'start'
  if (platform === 'linux' && !softwareRendering) return 'relaunch'
  return 'fail'
}

/** Electron's relaunch API expects argv without the executable at index zero. */
export function softwareRenderingRelaunchArgs(argv: readonly string[] = process.argv): string[] {
  return [
    ...argv.slice(1).filter((argument) => argument !== SOFTWARE_RENDERING_ARGUMENT),
    SOFTWARE_RENDERING_ARGUMENT
  ]
}
