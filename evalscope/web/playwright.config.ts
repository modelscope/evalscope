import { defineConfig, devices } from '@playwright/test'

/**
 * Playwright configuration for the EvalScope Web Console E2E suite.
 *
 * Determinism contract (Requirements 14.2, 14.6):
 * - Tests run against a LOCAL static preview build only (no external network).
 * - A fixed timezone, locale and color scheme remove environment-driven variance.
 * - A fixed random seed (PW_SEED) is exported to the test environment so any
 *   randomized test data can be reproduced across runs.
 * - External (non-localhost) network access is blocked / mocked in tests via
 *   route interception. See `e2e/fixtures.ts` for the shared `test` fixture that
 *   aborts any request whose host is not localhost, forcing tests to rely on
 *   desensitized fixtures instead of the live backend.
 */

/** Fixed seed for reproducible randomized test data (Requirement 14.6). */
export const PW_SEED = 20260701

/** Port used by `vite preview` (Vite's default preview port). */
const PREVIEW_PORT = 4173
const BASE_URL = `http://localhost:${PREVIEW_PORT}`

// Expose the deterministic seed and timezone to the test process so both the
// Playwright fixtures and any app-side seeding can read a single source.
process.env.PW_SEED = process.env.PW_SEED ?? String(PW_SEED)
process.env.TZ = 'UTC'

/** Shared, deterministic browser context options applied to every project. */
const deterministicContext = {
  baseURL: BASE_URL,
  timezoneId: 'UTC',
  locale: 'en-US',
  colorScheme: 'light' as const,
  // Local runs reuse the installed stable Chrome; CI installs Playwright's
  // pinned Chromium explicitly for hermetic execution.
  ...(process.env.CI ? {} : { channel: 'chrome' as const }),
  // Do not follow external redirects to third-party origins during tests.
  bypassCSP: false,
}

export default defineConfig({
  testDir: './e2e',
  // Fully deterministic ordering; no test-level parallelism flakiness.
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? [['github'], ['list']] : [['list']],
  snapshotPathTemplate: '{testDir}/visual/__screenshots__/{arg}-{projectName}-{platform}{ext}',

  expect: {
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.001,
      animations: 'disabled',
    },
  },

  use: {
    ...deterministicContext,
    actionTimeout: 10_000,
    navigationTimeout: 15_000,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },

  // Two fixed viewport presets covering the responsive contract breakpoints
  // exercised by the suite (mobile 390px and desktop 1024px, Requirement 14.2).
  projects: [
    {
      name: 'mobile-390',
      use: {
        ...devices['Desktop Chrome'],
        ...deterministicContext,
        viewport: { width: 390, height: 844 },
        isMobile: false,
        hasTouch: true,
      },
    },
    {
      name: 'desktop-1024',
      use: {
        ...devices['Desktop Chrome'],
        ...deterministicContext,
        viewport: { width: 1024, height: 768 },
      },
    },
  ],

  // Serve the local static preview build. Tests never reach the live backend;
  // API traffic is mocked via route interception in `e2e/fixtures.ts`.
  webServer: {
    command: `npm run preview -- --port ${PREVIEW_PORT} --strictPort`,
    url: BASE_URL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
