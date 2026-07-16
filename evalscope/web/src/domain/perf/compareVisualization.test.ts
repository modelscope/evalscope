// Feature: frontend-refactor-2026-07, Property 19: 对比可视化稀疏/趋势选择
//
// For any number of data points n, selectCompareVisualization(n) must pick the
// comparison visualization form dictated by a single fixed threshold:
//   - n <= 2 → 'sparse' (non-trending discrete point markers);
//   - n > 2  → 'trend'  (distribution / trend form).
// Invalid inputs — n <= 0 as well as non-finite values (NaN, Infinity,
// -Infinity) — collapse to the conservative 'sparse' form so the view never
// overstates a trend it cannot support. The choice depends only on n, so an
// independent reference function fully specifies it.
//
// Validates: Requirements 9.11, 9.12

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import type { CompareVisualization } from './compareVisualization'
import { selectCompareVisualization } from './compareVisualization'

/** The single sparse/trend threshold, mirrored here independently. */
const SPARSE_MAX_POINTS = 2

/**
 * Independent reference selector. Written separately from the implementation
 * so the property cross-checks behaviour rather than restating the code.
 */
function expectedVisualization(n: number): CompareVisualization {
  if (!Number.isFinite(n)) return 'sparse'
  return n > SPARSE_MAX_POINTS ? 'trend' : 'sparse'
}

describe('selectCompareVisualization — sparse/trend selection (Property 19: 对比可视化稀疏/趋势选择)', () => {
  it('matches the reference selector for any integer count (incl. <=0 and large)', () => {
    fc.assert(
      fc.property(fc.integer({ min: -1000, max: 1_000_000 }), (n) => {
        const result = selectCompareVisualization(n)
        expect(result).toBe(expectedVisualization(n))
        if (n > SPARSE_MAX_POINTS) {
          expect(result).toBe('trend')
        } else {
          expect(result).toBe('sparse')
        }
      }),
    )
  })

  it('selects the explicit boundary values 0/1/2/3', () => {
    expect(selectCompareVisualization(0)).toBe('sparse')
    expect(selectCompareVisualization(1)).toBe('sparse')
    expect(selectCompareVisualization(2)).toBe('sparse')
    expect(selectCompareVisualization(3)).toBe('trend')
  })

  it('collapses non-finite inputs (NaN/Infinity/-Infinity) to sparse', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY),
        (n) => {
          expect(selectCompareVisualization(n)).toBe('sparse')
        },
      ),
    )

    expect(selectCompareVisualization(Number.NaN)).toBe('sparse')
    expect(selectCompareVisualization(Number.POSITIVE_INFINITY)).toBe('sparse')
    expect(selectCompareVisualization(Number.NEGATIVE_INFINITY)).toBe('sparse')
  })
})
