/**
 * Pure logic for chart theme injection (Chart_Renderer).
 *
 * These helpers have no DOM, network, timer or randomness dependencies so they
 * can be exercised directly by property-based tests. They back the theme-aware
 * URL that `ChartFrame` passes to the underlying Plotly iframe.
 */

/** Theme identifier carried on every chart request path (Req 2.1). */
export type ChartTheme = 'light' | 'dark'

/** Query parameter name used to convey the active theme to the chart backend. */
export const THEME_PARAM = 'theme'

/**
 * Inject the active `theme` into a chart URL.
 *
 * The result always carries exactly one `theme` query parameter matching the
 * requested theme. If `baseSrc` already contains a `theme` parameter it is
 * replaced rather than appended, which makes the function idempotent: applying
 * `withTheme` to its own output never produces duplicate theme parameters
 * (Property 4). Existing query parameters and a trailing hash fragment are
 * preserved.
 *
 * @param baseSrc - Base chart URL. May be absolute or relative and may already
 *   contain a query string and/or hash fragment. Callers must not pre-append a
 *   theme parameter themselves.
 * @param theme - Active theme to encode into the URL.
 * @returns The URL with a single, canonical `theme` parameter.
 */
export function withTheme(baseSrc: string, theme: ChartTheme): string {
  // Split off the hash fragment first so query manipulation never touches it.
  const hashIndex = baseSrc.indexOf('#')
  const hash = hashIndex >= 0 ? baseSrc.slice(hashIndex) : ''
  const withoutHash = hashIndex >= 0 ? baseSrc.slice(0, hashIndex) : baseSrc

  // Separate the path portion from any existing query string.
  const queryIndex = withoutHash.indexOf('?')
  const path = queryIndex >= 0 ? withoutHash.slice(0, queryIndex) : withoutHash
  const queryString = queryIndex >= 0 ? withoutHash.slice(queryIndex + 1) : ''

  // `set` removes every existing `theme` entry and writes exactly one, giving
  // both replacement semantics and idempotency.
  const params = new URLSearchParams(queryString)
  params.set(THEME_PARAM, theme)

  const nextQuery = params.toString()
  return `${path}${nextQuery ? `?${nextQuery}` : ''}${hash}`
}
