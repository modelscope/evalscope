// Feature: frontend-refactor-2026-07, Property 18: low-sample tier boundaries
//
// For any non-negative sample count n, classifySampleSize(n) must return the
// low-sample tier dictated by the two fixed thresholds:
//   - n < 30           → 'critical' (strong warning, de-emphasize P90/P95/P99);
//   - 30 <= n < 100    → 'warn'     (warn/de-emphasize P95/P99);
//   - n >= 100         → 'ok'       (show normally).
// The boundaries are explicit and inclusive-lower: 29 → 'critical', 30 → 'warn',
// 99 → 'warn', 100 → 'ok'. The classification depends only on n, never on any
// other run state, so an independent reference function fully specifies it.
//
// Validates: Requirements 9.6, 9.7, 9.8

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import type { SampleTier } from './compareModel'
import { classifySampleSize } from './compareModel'

/** The two tier thresholds, mirrored here independently of the implementation. */
const CRITICAL_THRESHOLD = 30
const WARN_THRESHOLD = 100

/**
 * Independent reference classifier for a non-negative sample count. Written
 * separately from the implementation so the property cross-checks behaviour
 * rather than restating the same code.
 */
function expectedTier(n: number): SampleTier {
  if (n < CRITICAL_THRESHOLD) return 'critical'
  if (n < WARN_THRESHOLD) return 'warn'
  return 'ok'
}

describe('classifySampleSize — low-sample tier boundaries (Property 18: low-sample tier boundaries)', () => {
  it('matches the reference classifier for any non-negative sample count', () => {
    fc.assert(
      fc.property(fc.nat({ max: 1_000_000 }), (n) => {
        expect(classifySampleSize(n)).toBe(expectedTier(n))
      }),
    )
  })

  it('classifies the explicit boundary values 0/29/30/99/100/101', () => {
    expect(classifySampleSize(0)).toBe('critical')
    expect(classifySampleSize(29)).toBe('critical')
    expect(classifySampleSize(30)).toBe('warn')
    expect(classifySampleSize(99)).toBe('warn')
    expect(classifySampleSize(100)).toBe('ok')
    expect(classifySampleSize(101)).toBe('ok')
  })
})
