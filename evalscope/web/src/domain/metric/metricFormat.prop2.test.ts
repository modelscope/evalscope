// Feature: frontend-refactor-2026-07, Property 2: missing placeholder is distinguishable from a legitimate 0
//
// For any metric spec (bounded or unbounded) and any value:
//   - When the value is `null`, `undefined` or `NaN` (a "missing" value),
//     formatMetric returns `primary === MISSING_PLACEHOLDER`,
//     `raw === MISSING_PLACEHOLDER` and `isMissing === true`.
//   - When the value is a legitimate finite number (including `0`, negatives and
//     positives), formatMetric returns `isMissing === false` and never renders
//     the missing placeholder. In particular a legitimate `0` is never mistaken
//     for a missing value, so a real zero stays distinguishable from an absent
//     one.
//
// Validates: Requirements 1.8

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { formatMetric, MISSING_PLACEHOLDER } from './metricFormat'
import type { MetricDisplaySpec } from './MetricDisplaySpec'

// Identity translate: unit labels resolve to their key. Never injects the
// missing placeholder, keeping the "legitimate value has no placeholder"
// assertion unambiguous.
const identityTranslate = (key: string): string => key

const precisionArb: fc.Arbitrary<number> = fc.integer({ min: 0, max: 6 })

// Bounded-ratio spec: displays as a percentage, no unit.
const boundedSpecArb: fc.Arbitrary<MetricDisplaySpec> = fc
  .record({
    key: fc.stringMatching(/^[a-z_]{1,12}$/),
    labelKey: fc.stringMatching(/^[a-z_.]{1,16}$/),
    direction: fc.constantFrom('higher-is-better' as const, 'lower-is-better' as const),
    rawPrecision: precisionArb,
    percentPrecision: precisionArb,
    storedAsHundred: fc.boolean(),
  })
  .map((partial) => ({ ...partial, boundedness: 'bounded' as const, unit: null }))

// Unbounded / benchmark-native spec: keeps a unit (or none), no percentage.
// `%` is excluded so the missing-placeholder check is never confused with a
// legitimate unit label.
const unboundedSpecArb: fc.Arbitrary<MetricDisplaySpec> = fc
  .record({
    key: fc.stringMatching(/^[a-zA-Z][a-zA-Z0-9_]*$/),
    labelKey: fc.stringMatching(/^[a-zA-Z][a-zA-Z0-9_.]*$/),
    direction: fc.constantFrom('higher-is-better' as const, 'lower-is-better' as const),
    unit: fc.option(fc.constantFrom('ms', 'tokens', 'requests', 'seconds', 'MB', 'count'), {
      nil: null,
    }),
    rawPrecision: precisionArb,
    percentPrecision: fc.integer({ min: 0, max: 4 }),
  })
  .map((partial) => ({ ...partial, boundedness: 'unbounded' as const }))

// Either kind of spec — the missing/legitimate distinction is spec-independent.
const specArb: fc.Arbitrary<MetricDisplaySpec> = fc.oneof(boundedSpecArb, unboundedSpecArb)

// Missing values: null, undefined and NaN all signal an absent metric.
const missingValueArb: fc.Arbitrary<number | null | undefined> = fc.constantFrom(
  null,
  undefined,
  Number.NaN,
)

// Legitimate finite values spanning 0, negatives and positives.
const finiteValueArb: fc.Arbitrary<number> = fc.oneof(
  fc.constant(0),
  fc.double({ min: -1e9, max: 1e9, noNaN: true, noDefaultInfinity: true }),
  fc.double({ min: -1, max: 1, noNaN: true, noDefaultInfinity: true }),
  fc.integer({ min: -1_000_000, max: 1_000_000 }),
)

describe('formatMetric (Property 2: missing placeholder is distinguishable from a legitimate 0)', () => {
  it('renders the missing placeholder with isMissing=true for null/undefined/NaN', () => {
    fc.assert(
      fc.property(specArb, missingValueArb, (spec, value) => {
        const result = formatMetric(value, spec, identityTranslate)

        expect(result.isMissing).toBe(true)
        expect(result.primary).toBe(MISSING_PLACEHOLDER)
        expect(result.raw).toBe(MISSING_PLACEHOLDER)
      }),
    )
  })

  it('never marks a legitimate finite value as missing nor renders the placeholder', () => {
    fc.assert(
      fc.property(specArb, finiteValueArb, (spec, value) => {
        const result = formatMetric(value, spec, identityTranslate)

        expect(result.isMissing).toBe(false)
        expect(result.primary).not.toBe(MISSING_PLACEHOLDER)
        expect(result.raw).not.toBe(MISSING_PLACEHOLDER)
      }),
    )
  })

  it('treats a legitimate 0 as a real value, distinct from a missing one', () => {
    fc.assert(
      fc.property(specArb, (spec) => {
        const zero = formatMetric(0, spec, identityTranslate)
        const missing = formatMetric(null, spec, identityTranslate)

        // A real zero is a present value...
        expect(zero.isMissing).toBe(false)
        expect(zero.primary).not.toBe(MISSING_PLACEHOLDER)
        expect(zero.raw).not.toBe(MISSING_PLACEHOLDER)

        // ...and is never conflated with the missing rendering.
        expect(zero.isMissing).not.toBe(missing.isMissing)
        expect(zero.primary).not.toBe(missing.primary)
      }),
    )
  })
})
