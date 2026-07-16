// Feature: frontend-refactor-2026-07, Property 4: 图表 URL theme 注入幂等
//
// For any chart baseSrc (with or without an existing query string, with or
// without a hash fragment, with or without a pre-existing theme parameter) and
// any theme, withTheme's output carries exactly one theme parameter equal to
// the requested theme, and applying withTheme again to that output is a no-op
// (idempotent).

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { THEME_PARAM, withTheme, type ChartTheme } from './chartTheme'

/** Extract the query-string portion of a URL, excluding any hash fragment. */
function queryOf(url: string): string {
  const hashIndex = url.indexOf('#')
  const withoutHash = hashIndex >= 0 ? url.slice(0, hashIndex) : url
  const queryIndex = withoutHash.indexOf('?')
  return queryIndex >= 0 ? withoutHash.slice(queryIndex + 1) : ''
}

/** All values recorded for the theme parameter in a URL's query string. */
function themeValues(url: string): string[] {
  return new URLSearchParams(queryOf(url)).getAll(THEME_PARAM)
}

const themeArb: fc.Arbitrary<ChartTheme> = fc.constantFrom('light', 'dark')

// A path segment that does not itself contain query/hash delimiters.
const pathArb = fc
  .array(fc.stringMatching(/^[a-zA-Z0-9_-]+$/), { minLength: 0, maxLength: 4 })
  .map((segments) => `/${segments.join('/')}`)

// Non-theme query parameters. Keys avoid the theme parameter name and both keys
// and values avoid characters that would break query-string structure.
const queryParamArb = fc.tuple(
  fc.stringMatching(/^[a-zA-Z][a-zA-Z0-9_]*$/).filter((k) => k !== THEME_PARAM),
  fc.stringMatching(/^[a-zA-Z0-9_-]*$/),
)

// Optionally include a pre-existing theme parameter so the "replace not append"
// behaviour is exercised.
const existingThemeArb = fc.option(fc.constantFrom('light', 'dark', 'sepia', ''), { nil: undefined })

const hashArb = fc.option(fc.stringMatching(/^[a-zA-Z0-9_-]*$/), { nil: undefined })

const baseSrcArb: fc.Arbitrary<string> = fc
  .record({
    path: pathArb,
    params: fc.array(queryParamArb, { minLength: 0, maxLength: 4 }),
    existingTheme: existingThemeArb,
    // Position of the pre-existing theme param among the other params.
    themeFirst: fc.boolean(),
    hash: hashArb,
  })
  .map(({ path, params, existingTheme, themeFirst, hash }) => {
    const search = new URLSearchParams()
    if (existingTheme !== undefined && themeFirst) {
      search.append(THEME_PARAM, existingTheme)
    }
    for (const [key, value] of params) {
      search.append(key, value)
    }
    if (existingTheme !== undefined && !themeFirst) {
      search.append(THEME_PARAM, existingTheme)
    }
    const query = search.toString()
    const hashPart = hash !== undefined ? `#${hash}` : ''
    return `${path}${query ? `?${query}` : ''}${hashPart}`
  })

describe('withTheme (Property 4: 图表 URL theme 注入幂等)', () => {
  it('produces exactly one theme parameter equal to the requested theme', () => {
    fc.assert(
      fc.property(baseSrcArb, themeArb, (baseSrc, theme) => {
        const result = withTheme(baseSrc, theme)
        const values = themeValues(result)
        expect(values).toHaveLength(1)
        expect(values[0]).toBe(theme)
      }),
      { numRuns: 100 },
    )
  })

  it('is idempotent: applying withTheme to its own output is a no-op', () => {
    fc.assert(
      fc.property(baseSrcArb, themeArb, (baseSrc, theme) => {
        const once = withTheme(baseSrc, theme)
        const twice = withTheme(once, theme)
        expect(twice).toBe(once)
        // Re-applying still yields exactly one theme parameter.
        expect(themeValues(twice)).toEqual([theme])
      }),
      { numRuns: 100 },
    )
  })
})
