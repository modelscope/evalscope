import { expect, type Page } from '@playwright/test'

/**
 * Light/dark dual-theme visual baseline capture helpers.
 *
 * Purpose (Requirements 14.6, 15.6): provide a reusable, deterministic way to
 * capture visual-regression baselines for a page or component in BOTH the light
 * and dark themes. These baselines act as the reference for asserting that the
 * decoupling/refactor work keeps DOM structure and visible text intact.
 *
 * This module is infrastructure only. The concrete stories/flows to snapshot are
 * supplied by tasks 19.1 (Storybook) and 19.2 (E2E flows); the diff threshold and
 * CI run strategy (>0.1% pixel diff fails, path-filtered execution, centralized
 * snapshot storage) are wired in task 19.3 via `playwright.config.ts`.
 *
 * Determinism contract:
 * - Runs against the local static preview only; external network is blocked by
 *   the shared fixture in `e2e/fixtures.ts` (Requirement 14.6).
 * - Viewport is fixed by the Playwright `mobile-390` / `desktop-1024` projects.
 * - Theme is forced through the application's own theme mechanism (the
 *   `evalscope-theme` localStorage key and the `data-theme` attribute on the
 *   document element), mirroring `.storybook/preview.tsx`, so captures reflect
 *   real rendered themes rather than an ad-hoc override.
 */

export type VisualTheme = 'light' | 'dark'

/** The two themes captured for every visual baseline. */
export const VISUAL_THEMES: readonly VisualTheme[] = ['light', 'dark'] as const

/**
 * localStorage key read by `ThemeProvider` (`src/contexts/ThemeContext.tsx`) on
 * initialisation. Kept in sync with `.storybook/preview.tsx`.
 */
export const THEME_STORAGE_KEY = 'evalscope-theme'

/** Screenshot options shared by page- and component-level captures. */
interface BaseCaptureOptions {
  /** Themes to capture; defaults to both light and dark. */
  themes?: readonly VisualTheme[]
  /** Disable animations for stable pixels. Defaults to true. */
  disableAnimations?: boolean
}

/** Options for capturing a full-page baseline. */
export interface PageBaselineOptions extends BaseCaptureOptions {
  /** Capture the full scrollable page rather than the viewport. Defaults to true. */
  fullPage?: boolean
}

/**
 * Switches an already-loaded page to `theme` using the app's theme mechanism.
 *
 * Writes the `evalscope-theme` localStorage key, then reloads so `ThemeProvider`
 * re-initialises from storage and applies `data-theme` itself. Resolves only once
 * `data-theme` matches the requested theme, giving a deterministic wait with no
 * reliance on timers.
 */
export async function applyTheme(page: Page, theme: VisualTheme): Promise<void> {
  await page.evaluate(
    ({ key, value }) => {
      window.localStorage.setItem(key, value)
      document.documentElement.setAttribute('data-theme', value)
    },
    { key: THEME_STORAGE_KEY, value: theme },
  )
  await page.reload()
  await page.waitForFunction(
    (value) => document.documentElement.getAttribute('data-theme') === value,
    theme,
  )
  // `data-theme` is set before route chunks and mocked API data finish loading.
  // Waiting only for the attribute captured the global Suspense spinner as the
  // baseline, which made the visual check pass without exercising the screen.
  await expect(page.getByRole('banner')).toBeVisible()
  await expect(page.locator('main .animate-spin')).toHaveCount(0)
  await page.waitForLoadState('networkidle')
}

/**
 * Captures a full-page visual baseline for the current route in both themes.
 *
 * Produces one snapshot per theme named `${name}-${theme}.png`. Playwright further
 * suffixes each name with the active project (`mobile-390` / `desktop-1024`) and
 * platform, so a single call yields the full light/dark x viewport baseline matrix.
 *
 * The page must already be navigated to the target route; the theme is toggled in
 * place via {@link applyTheme}, preserving the current URL.
 */
export async function captureThemeBaselines(
  page: Page,
  name: string,
  options: PageBaselineOptions = {},
): Promise<void> {
  const { themes = VISUAL_THEMES, fullPage = true, disableAnimations = true } = options
  for (const theme of themes) {
    await applyTheme(page, theme)
    await expect(page).toHaveScreenshot(`${name}-${theme}.png`, {
      fullPage,
      animations: disableAnimations ? 'disabled' : 'allow',
    })
  }
}
