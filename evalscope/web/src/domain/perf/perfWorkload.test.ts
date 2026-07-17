// Feature: frontend-refactor-2026-07, Property 14: Workload normalization never renders INF and shows missing placeholders
//
// For any request rate and workload parameters, `normalizeWorkload`'s output
// must satisfy:
//   - when the rate is truly unlimited it is a semantic loop label
//     (`closed-loop` / `open-loop`), and when the rate is finite and positive it
//     contains that finite value — in both cases `rateLabel` never contains the
//     literal `INF` (case-insensitive);
//   - a missing/invalid concurrency or number-of-requests parameter renders as
//     the `N/A` placeholder, while a valid one renders as its string value.
//
// Validates: Requirements 8.5, 8.6, 8.7

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import type { PerfRunItem } from '../../api/types'
import {
  CLOSED_LOOP_LABEL,
  MISSING_WORKLOAD,
  normalizeWorkload,
  OPEN_LOOP_LABEL,
} from './perfWorkload'

/**
 * A count field (`parallel` / `number`) as it may arrive from an upstream
 * payload: a valid non-negative integer, a missing value (`undefined`/`null`),
 * or an invalid negative number. Returned as `unknown` because the raw payload
 * is not guaranteed to match the declared `PerfRunItem` types.
 */
const countArb: fc.Arbitrary<unknown> = fc.oneof(
  fc.nat({ max: 100_000 }), // valid non-negative count
  fc.constantFrom(undefined, null), // missing
  fc.integer({ min: -100_000, max: -1 }), // invalid negative
)

/**
 * A request-rate field as it may arrive from an upstream payload: a finite
 * positive number (limited), an explicit "no rate limit" form (`null`/
 * `undefined`/`Infinity`), a `<= 0` sentinel (EvalScope's `--rate -1` default),
 * or a textual `INF` marker.
 */
const rateArb: fc.Arbitrary<unknown> = fc.oneof(
  fc.double({ min: 1e-6, max: 1e6, noNaN: true, noDefaultInfinity: true }), // finite positive
  fc.constantFrom(null, undefined, Infinity), // unlimited
  fc.integer({ min: -100_000, max: 0 }), // <= 0 sentinel
  fc.constantFrom('INF', 'inf', 'Inf'), // textual marker
)

/** Build a raw run from the generated fields, cast to the declared type. */
function makeRun(parallel: unknown, number: unknown, rate: unknown): PerfRunItem {
  return { parallel, number, rate } as unknown as PerfRunItem
}

/** Mirror the module's notion of a usable finite count for the oracle. */
function isValidCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
}

/** Mirror the module's notion of an unlimited rate for the oracle. */
function isUnlimitedRate(rate: unknown): boolean {
  if (rate == null) return true
  if (typeof rate === 'string') return true
  if (typeof rate !== 'number') return true
  if (!Number.isFinite(rate)) return true
  return rate <= 0
}

describe('normalizeWorkload (Property 14: normalization never renders INF and shows missing placeholders)', () => {
  it('never emits an INF marker and uses semantic labels for unlimited rates', () => {
    fc.assert(
      fc.property(countArb, countArb, rateArb, (parallel, number, rate) => {
        const { rateLabel } = normalizeWorkload(makeRun(parallel, number, rate))

        // Invariant: the rate label must never surface the literal INF marker,
        // in any casing.
        expect(rateLabel.toLowerCase()).not.toContain('inf')

        if (isUnlimitedRate(rate)) {
          // Unlimited rate → a semantic loop label, never a numeric/INF value.
          expect([CLOSED_LOOP_LABEL, OPEN_LOOP_LABEL]).toContain(rateLabel)
          // closed-loop iff there is a positive concurrency limit.
          const hasConcurrencyLimit = isValidCount(parallel) && parallel > 0
          expect(rateLabel).toBe(hasConcurrencyLimit ? CLOSED_LOOP_LABEL : OPEN_LOOP_LABEL)
        } else {
          // Limited rate → the finite value is present in the label.
          expect(rateLabel).toContain(String(rate))
        }
      }),
    )
  })

  it('renders concurrency and number-of-requests, using N/A for missing/invalid values', () => {
    fc.assert(
      fc.property(countArb, countArb, rateArb, (parallel, number, rate) => {
        const { concurrency, numberOfRequests } = normalizeWorkload(makeRun(parallel, number, rate))

        if (isValidCount(parallel)) {
          expect(concurrency).toBe(String(parallel))
        } else {
          expect(concurrency).toBe(MISSING_WORKLOAD)
        }

        if (isValidCount(number)) {
          expect(numberOfRequests).toBe(String(number))
        } else {
          expect(numberOfRequests).toBe(MISSING_WORKLOAD)
        }
      }),
    )
  })
})
