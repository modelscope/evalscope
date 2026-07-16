// Feature: frontend-refactor-2026-07, Property 6: 自适应可视化维度映射
//
// For any non-negative integer dimension count n, selectVisualization(n)
// selects the visualization form purely from the count: 0 → 'empty',
// 1 → 'single-value', 2 → 'grouped-bar', n >= 3 → 'radar'. The mapping is total
// and deterministic, so an independent reference implementation must agree with
// the code under test on every generated input.

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { selectVisualization, type VizKind } from './adaptiveVisualization'

/**
 * Independent reference implementation of the dimension → visualization mapping.
 * Kept intentionally separate from the production code so the property compares
 * two implementations rather than restating the same expression.
 */
function expectedVisualization(dimensionCount: number): VizKind {
  if (dimensionCount === 0) return 'empty'
  if (dimensionCount === 1) return 'single-value'
  if (dimensionCount === 2) return 'grouped-bar'
  return 'radar'
}

// Non-negative integer dimension counts, including small boundary values and a
// wide range of larger counts that must all map to 'radar'.
const dimensionCountArb: fc.Arbitrary<number> = fc.nat({ max: 1000 })

describe('selectVisualization (Property 6: 自适应可视化维度映射)', () => {
  it('agrees with the independent reference for any non-negative integer n', () => {
    fc.assert(
      fc.property(dimensionCountArb, (n) => {
        expect(selectVisualization(n)).toBe(expectedVisualization(n))
      }),
      { numRuns: 100 },
    )
  })

  it('always returns a valid visualization kind', () => {
    const valid: readonly VizKind[] = ['empty', 'single-value', 'grouped-bar', 'radar']
    fc.assert(
      fc.property(dimensionCountArb, (n) => {
        expect(valid).toContain(selectVisualization(n))
      }),
      { numRuns: 100 },
    )
  })

  it('maps 3 or more dimensions to radar', () => {
    fc.assert(
      fc.property(fc.integer({ min: 3, max: 10000 }), (n) => {
        expect(selectVisualization(n)).toBe('radar')
      }),
      { numRuns: 100 },
    )
  })

  // Explicit boundary assertions for the discrete low-dimension cases.
  it('maps the discrete boundaries 0/1/2/3 explicitly', () => {
    expect(selectVisualization(0)).toBe('empty')
    expect(selectVisualization(1)).toBe('single-value')
    expect(selectVisualization(2)).toBe('grouped-bar')
    expect(selectVisualization(3)).toBe('radar')
  })

  it('maps a large dimension count to radar', () => {
    expect(selectVisualization(9999)).toBe('radar')
  })
})
