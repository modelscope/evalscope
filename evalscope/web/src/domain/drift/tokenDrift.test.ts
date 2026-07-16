import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { checkTokenDrift, MISSING_VALUE } from '../../../scripts/drift/tokenDrift.ts'
import type { TokenCategory, TokenDefinition } from '../../../design/tokens.source.ts'

// Feature: frontend-refactor-2026-07, Property 29: Token drift 逐项检测
//
// For any token single source of truth (SSOT) plus its Markdown-displayed values
// and CSS-defined values, the set of inconsistent entries returned by
// `checkTokenDrift` must be EXACTLY the set of tokens whose three values are not
// all identical — no false negatives (every drifted token is reported) and no
// false positives (no consistent token is reported).
//
// Strategy: programmatically synthesize a small TokenSource, generate byte-for-byte
// consistent CSS (via `--var: value;` declarations) and Markdown frontmatter (via
// `key: "value"` lines) from it, then randomly perturb the CSS and/or Markdown
// value of each comparable (token, theme) expectation. The expected drift set is
// precisely the perturbed expectations; the actual set from `checkTokenDrift` must
// equal it.
//
// Validates: Requirements 17.3, 17.4

// ── Generators ───────────────────────────────────────────────────────────────

// Token value characters that survive both parsers unchanged:
//   - CSS declaration values may not contain `;`, `{` or `}`.
//   - We quote Markdown values, so any char is safe there; a colon is still
//     excluded so it can never be mistaken for a CSS `name: value` boundary.
// Whitespace is collapsed to single spaces and trimmed so a byte-for-byte match
// is stable across CSS `trim()` and YAML unquoting.
const SAFE_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ().,-%'

const valueArb: fc.Arbitrary<string> = fc
  .array(fc.constantFrom(...SAFE_CHARS.split('')), { minLength: 1, maxLength: 14 })
  .map((chars) => chars.join('').replace(/\s+/g, ' ').trim())
  .filter((s) => s.length > 0)

interface Spec {
  readonly kind: 'shared' | 'themed'
  readonly darkValue: string
  readonly lightValue: string
}

const specArb: fc.Arbitrary<Spec> = fc.record({
  kind: fc.constantFrom<'shared' | 'themed'>('shared', 'themed'),
  darkValue: valueArb,
  lightValue: valueArb,
})

interface Pert {
  readonly css: boolean
  readonly md: boolean
}

const pertArb: fc.Arbitrary<Pert> = fc.record({ css: fc.boolean(), md: fc.boolean() })

// A shared token contributes one comparable expectation (dark only); a themed
// token contributes two (dark + light). Perturbations are generated to match.
function expectationCount(specs: readonly Spec[]): number {
  return specs.reduce((n, s) => n + (s.kind === 'themed' ? 2 : 1), 0)
}

const modelArb = fc.array(specArb, { minLength: 1, maxLength: 10 }).chain((specs) => {
  const count = expectationCount(specs)
  return fc.record({
    specs: fc.constant(specs),
    perts: fc.array(pertArb, { minLength: count, maxLength: count }),
  })
})

// ── Property 29 ──────────────────────────────────────────────────────────────

describe('checkTokenDrift (Property 29: Token drift 逐项检测)', () => {
  // One flattened, comparable expectation with its reporting id and drift flag.
  interface Exp {
    id: string
    drifted: boolean
  }

  it('reports exactly the tokens whose three values are not all identical', () => {
    fc.assert(
      fc.property(modelArb, ({ specs, perts }) => {
        const source: TokenDefinition[] = []
        const exps: Exp[] = []
        const rootDecls: string[] = []
        const darkDecls: string[] = []
        const lightDecls: string[] = []
        const mdLines: string[] = []

        // A perturbed surface value always differs from the canonical value, so
        // any perturbation guarantees the token drifts.
        const drift = (value: string, surface: string) => `${value}-DRIFT-${surface}`

        let p = 0
        specs.forEach((spec, i) => {
          const cssVar = `--tok-${i}`
          const key = `k-${i}`
          const category: TokenCategory = 'color'

          if (spec.kind === 'shared') {
            const canonical = spec.darkValue
            const pert = perts[p++]
            const cssValue = pert.css ? drift(canonical, 'css') : canonical
            const mdValue = pert.md ? drift(canonical, 'md') : canonical
            source.push({
              name: `tok-${i}`,
              category,
              value: { kind: 'shared', value: canonical },
              css: { var: cssVar },
              markdown: { key },
            })
            rootDecls.push(`  ${cssVar}: ${cssValue};`)
            mdLines.push(`${key}: "${mdValue}"`)
            // A shared token appears once → reported id is the bare name.
            exps.push({ id: `tok-${i}`, drifted: pert.css || pert.md })
            return
          }

          // Themed token: dark + light expectations under the same CSS var but
          // distinct Markdown keys. Appearing twice → ids are theme-qualified.
          const lightKey = `k-${i}-light`
          const darkCanonical = spec.darkValue
          const lightCanonical = spec.lightValue
          source.push({
            name: `tok-${i}`,
            category,
            value: { kind: 'themed', dark: darkCanonical, light: lightCanonical },
            css: { var: cssVar },
            markdown: { key, lightKey },
          })

          const darkPert = perts[p++]
          const darkCssValue = darkPert.css ? drift(darkCanonical, 'css') : darkCanonical
          const darkMdValue = darkPert.md ? drift(darkCanonical, 'md') : darkCanonical
          darkDecls.push(`  ${cssVar}: ${darkCssValue};`)
          mdLines.push(`${key}: "${darkMdValue}"`)
          exps.push({ id: `tok-${i} (dark)`, drifted: darkPert.css || darkPert.md })

          const lightPert = perts[p++]
          const lightCssValue = lightPert.css ? drift(lightCanonical, 'css') : lightCanonical
          const lightMdValue = lightPert.md ? drift(lightCanonical, 'md') : lightCanonical
          lightDecls.push(`  ${cssVar}: ${lightCssValue};`)
          mdLines.push(`${lightKey}: "${lightMdValue}"`)
          exps.push({ id: `tok-${i} (light)`, drifted: lightPert.css || lightPert.md })
        })

        const css = [
          ':root {',
          ...rootDecls,
          '}',
          '[data-theme="dark"] {',
          ...darkDecls,
          '}',
          '[data-theme="light"] {',
          ...lightDecls,
          '}',
        ].join('\n')

        const md = ['---', ...mdLines, '---', '', '# Design'].join('\n')

        const entries = checkTokenDrift(source, css, md)
        const actualIds = entries.map((entry) => entry.token)
        const expectedIds = exps.filter((exp) => exp.drifted).map((exp) => exp.id)

        // Exactness: no token is reported more than once.
        expect(new Set(actualIds).size).toBe(actualIds.length)
        // No false negatives and no false positives.
        expect(new Set(actualIds)).toEqual(new Set(expectedIds))
      }),
    )
  })
})

// ── Concrete examples (complement the property with grounded edge cases) ───────

describe('checkTokenDrift — concrete examples', () => {
  const sharedToken = (name: string, value: string, cssVar: string, key: string): TokenDefinition => ({
    name,
    category: 'radius' as TokenCategory,
    value: { kind: 'shared', value },
    css: { var: cssVar },
    markdown: { key },
  })

  it('returns no entries when SSOT, CSS and Markdown all agree', () => {
    const source = [sharedToken('radius-xs', '4px', '--radius-xs', 'rounded.xs')]
    const css = ':root {\n  --radius-xs: 4px;\n}'
    const md = '---\nrounded:\n  xs: "4px"\n---\n'
    expect(checkTokenDrift(source, css, md)).toEqual([])
  })

  it('reports a token when only the CSS value drifts', () => {
    const source = [sharedToken('radius-xs', '4px', '--radius-xs', 'rounded.xs')]
    const css = ':root {\n  --radius-xs: 5px;\n}'
    const md = '---\nrounded:\n  xs: "4px"\n---\n'
    expect(checkTokenDrift(source, css, md)).toEqual([
      { token: 'radius-xs', markdownValue: '4px', cssValue: '5px' },
    ])
  })

  it('reports a token when only the Markdown value drifts', () => {
    const source = [sharedToken('radius-xs', '4px', '--radius-xs', 'rounded.xs')]
    const css = ':root {\n  --radius-xs: 4px;\n}'
    const md = '---\nrounded:\n  xs: "6px"\n---\n'
    expect(checkTokenDrift(source, css, md)).toEqual([
      { token: 'radius-xs', markdownValue: '6px', cssValue: '4px' },
    ])
  })

  it('surfaces MISSING_VALUE when a surface omits the token', () => {
    const source = [sharedToken('radius-xs', '4px', '--radius-xs', 'rounded.xs')]
    const css = ':root {\n}'
    const md = '---\n---\n'
    expect(checkTokenDrift(source, css, md)).toEqual([
      { token: 'radius-xs', markdownValue: MISSING_VALUE, cssValue: MISSING_VALUE },
    ])
  })

  it('distinguishes dark/light drift for a themed token via qualified ids', () => {
    const themed: TokenDefinition = {
      name: 'accent',
      category: 'color' as TokenCategory,
      value: { kind: 'themed', dark: '#111', light: '#eee' },
      css: { var: '--accent' },
      markdown: { key: 'accent', lightKey: 'accent-light' },
    }
    const css =
      ':root {\n}\n[data-theme="dark"] {\n  --accent: #111;\n}\n[data-theme="light"] {\n  --accent: #000;\n}'
    const md = '---\naccent: "#111"\naccent-light: "#eee"\n---\n'
    // Only the light CSS value drifts (#000 vs #eee).
    expect(checkTokenDrift(source_of(themed), css, md)).toEqual([
      { token: 'accent (light)', markdownValue: '#eee', cssValue: '#000' },
    ])
  })
})

// Helper: wrap a single definition into a source array (keeps examples terse).
function source_of(token: TokenDefinition): TokenDefinition[] {
  return [token]
}
