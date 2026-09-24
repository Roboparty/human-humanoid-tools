import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  timeout: 120_000,
  expect: { timeout: 15_000 },
  workers: 1,
  reporter: 'line',
  use: {
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure'
  }
})
