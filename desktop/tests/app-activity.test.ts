import { describe, expect, it, vi } from 'vitest'
import { withAppActivity } from '../src/main/app-activity'

describe('finite macOS activity', () => {
  it('prevents suspension only while work is active, including blocker id zero', async () => {
    const blocker = { start: vi.fn(() => 0), stop: vi.fn() }
    let finish!: (value: string) => void
    const result = withAppActivity(blocker, () => new Promise<string>((resolve) => { finish = resolve }))
    expect(blocker.start).toHaveBeenCalledWith('prevent-app-suspension')
    expect(blocker.stop).not.toHaveBeenCalled()
    finish('ready')
    await expect(result).resolves.toBe('ready')
    expect(blocker.stop).toHaveBeenCalledExactlyOnceWith(0)
  })

  it('releases the activity after a failed operation', async () => {
    const blocker = { start: vi.fn(() => 7), stop: vi.fn() }
    await expect(withAppActivity(blocker, async () => { throw new Error('installation failed') }))
      .rejects.toThrow('installation failed')
    expect(blocker.stop).toHaveBeenCalledExactlyOnceWith(7)
  })

  it('does not require a blocker on other platforms', async () => {
    await expect(withAppActivity(undefined, async () => 'ready')).resolves.toBe('ready')
  })
})
