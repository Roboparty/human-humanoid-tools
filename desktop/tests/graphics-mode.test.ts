import { describe, expect, it, vi } from 'vitest'

import {
  configureGraphicsCommandLine,
  decideGraphicsStartup,
  SOFTWARE_RENDERING_ARGUMENT,
  softwareRenderingRelaunchArgs,
  softwareRenderingRequested
} from '../src/main/graphics-mode'

describe('Linux graphics mode', () => {
  it('keeps the normal hardware path free of forced GL switches', () => {
    const appendSwitch = vi.fn()

    expect(configureGraphicsCommandLine({ appendSwitch }, ['hhtools'], 'linux')).toBe(false)
    expect(appendSwitch).not.toHaveBeenCalled()
  })

  it('selects SwANGLE only for the private Linux software-rendering launch', () => {
    const appendSwitch = vi.fn()
    const argv = ['hhtools', SOFTWARE_RENDERING_ARGUMENT]

    expect(softwareRenderingRequested(argv, 'linux')).toBe(true)
    expect(configureGraphicsCommandLine({ appendSwitch }, argv, 'linux')).toBe(true)
    expect(appendSwitch.mock.calls).toEqual([
      ['use-gl', 'angle'],
      ['use-angle', 'swiftshader'],
      ['enable-unsafe-swiftshader']
    ])
  })

  it('does not inject Linux graphics switches on other platforms', () => {
    const appendSwitch = vi.fn()

    expect(
      configureGraphicsCommandLine(
        { appendSwitch },
        ['hhtools.exe', SOFTWARE_RENDERING_ARGUMENT],
        'win32'
      )
    ).toBe(false)
    expect(appendSwitch).not.toHaveBeenCalled()
  })

  it('relaunches once after hardware WebGL2 fails and never loops in software mode', () => {
    expect(decideGraphicsStartup(false, false, 'linux')).toBe('relaunch')
    expect(decideGraphicsStartup(false, true, 'linux')).toBe('fail')
    expect(decideGraphicsStartup(true, false, 'linux')).toBe('start')
    expect(decideGraphicsStartup(true, true, 'linux')).toBe('start')
  })

  it('builds deduplicated Electron relaunch arguments', () => {
    expect(
      softwareRenderingRelaunchArgs([
        '/opt/Human-Humanoid Tools/hhtools',
        '--example',
        SOFTWARE_RENDERING_ARGUMENT
      ])
    ).toEqual(['--example', SOFTWARE_RENDERING_ARGUMENT])
  })
})
