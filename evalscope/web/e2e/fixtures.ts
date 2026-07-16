import { test as base, expect } from '@playwright/test'

/**
 * Shared E2E test fixture that enforces the "no external network" contract
 * (Requirements 14.2, 14.6).
 *
 * Every request whose host is not localhost / 127.0.0.1 is aborted, so tests
 * cannot accidentally depend on the live backend or any third-party origin.
 * API responses must be provided via desensitized fixtures and `page.route`
 * mocks defined inside each spec.
 */

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]'])

function isLocalUrl(url: string): boolean {
  try {
    const { hostname } = new URL(url)
    return LOCAL_HOSTS.has(hostname)
  } catch {
    // Non-http(s) schemes (data:, blob:, about:) are considered local/safe.
    return true
  }
}

export const test = base.extend({
  page: async ({ page }, provide) => {
    await page.clock.setFixedTime(new Date('2026-07-01T00:00:00.000Z'))
    // Block any non-localhost request to guarantee deterministic, offline runs.
    await page.route('**/*', (route) => {
      const url = route.request().url()
      if (isLocalUrl(url)) {
        return route.continue()
      }
      return route.abort('blockedbyclient')
    })
    await provide(page)
  },
})

export { expect }
