// Feature: frontend-refactor-2026-07, Property 28: 折叠态不渲染重内容
//
// For any collapse state and content kind, shouldRenderHeavy(collapsed,
// contentType) must withhold heavy content (math blocks, code blocks and large
// tables spanning more than the line threshold) while it is collapsed, and
// render it otherwise. Light content ('text') is never gated and always
// renders. This encodes Requirement 16.3.

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import {
  HEAVY_CONTENT_KINDS,
  isHeavyContent,
  shouldRenderHeavy,
  type ContentKind,
} from './heavyContent'

// All four content kinds the classifier can produce.
const contentKindArb: fc.Arbitrary<ContentKind> = fc.constantFrom(
  'text',
  'math',
  'code',
  'large-table',
)

const collapsedArb: fc.Arbitrary<boolean> = fc.boolean()

/**
 * Independent reference for the render decision, kept separate from the
 * production expression so the property compares two implementations.
 */
function expectedRender(collapsed: boolean, contentType: ContentKind): boolean {
  const heavy = HEAVY_CONTENT_KINDS.includes(contentType)
  return !heavy || !collapsed
}

describe('shouldRenderHeavy (Property 28: 折叠态不渲染重内容)', () => {
  it('agrees with the independent reference for any collapse state and content kind', () => {
    fc.assert(
      fc.property(collapsedArb, contentKindArb, (collapsed, contentType) => {
        expect(shouldRenderHeavy(collapsed, contentType)).toBe(
          expectedRender(collapsed, contentType),
        )
      }),
      { numRuns: 100 },
    )
  })

  it('never renders heavy content while collapsed', () => {
    fc.assert(
      fc.property(contentKindArb, (contentType) => {
        if (isHeavyContent(contentType)) {
          expect(shouldRenderHeavy(true, contentType)).toBe(false)
        }
      }),
      { numRuns: 100 },
    )
  })

  it('always renders heavy content once expanded', () => {
    fc.assert(
      fc.property(contentKindArb, (contentType) => {
        if (isHeavyContent(contentType)) {
          expect(shouldRenderHeavy(false, contentType)).toBe(true)
        }
      }),
      { numRuns: 100 },
    )
  })

  it('always renders light text regardless of collapse state', () => {
    fc.assert(
      fc.property(collapsedArb, (collapsed) => {
        expect(shouldRenderHeavy(collapsed, 'text')).toBe(true)
      }),
      { numRuns: 100 },
    )
  })

  // Explicit coverage of the full 4 kinds x 2 collapse-state matrix.
  it('covers the 4x2 kind/collapse matrix explicitly', () => {
    // Heavy kinds: withheld when collapsed, rendered when expanded.
    expect(shouldRenderHeavy(true, 'math')).toBe(false)
    expect(shouldRenderHeavy(false, 'math')).toBe(true)
    expect(shouldRenderHeavy(true, 'code')).toBe(false)
    expect(shouldRenderHeavy(false, 'code')).toBe(true)
    expect(shouldRenderHeavy(true, 'large-table')).toBe(false)
    expect(shouldRenderHeavy(false, 'large-table')).toBe(true)
    // Light kind: always rendered.
    expect(shouldRenderHeavy(true, 'text')).toBe(true)
    expect(shouldRenderHeavy(false, 'text')).toBe(true)
  })
})
