// Feature: frontend-refactor-2026-07, Property 3: 非 bounded 与 spec 缺失不触发百分比转换
//
// For any unbounded / benchmark-native metric spec (including the shared
// DEFAULT_METRIC_SPEC used when a spec is missing) and any value — whether it is
// greater than 1, less than 1 or negative — formatMetric preserves the unit and
// renders the raw value at the spec's rawPrecision without performing any
// percentage conversion. The primary text therefore never contains a `%` that
// results from normalization (unbounded specs are generated with a non-`%`
// unit or no unit to avoid ambiguity with `%` as a legitimate unit label). When
// the spec is missing, isSpecUndefined is true and the value is shown with 4
// decimal places.
//
// Validates: Requirements 1.1, 1.4, 1.5, 1.13

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import { formatMetric } from './metricFormat'
import {
  DEFAULT_METRIC_SPEC,
  DEFAULT_RAW_PRECISION,
  type MetricDisplaySpec,
} from './MetricDisplaySpec'

/** Identity translate: unit labels resolve to their key so no `%` is injected. */
const identityTranslate = (key: string): string => key

/** Number of decimal places present in a fixed-point numeric string. */
function decimalPlaces(numberStr: string): number {
  const dotIndex = numberStr.indexOf('.')
  return dotIndex < 0 ? 0 : numberStr.length - dotIndex - 1
}

// Finite values spanning > 1, < 1 and negatives. Bounds stay well below the
// exponential-notation threshold so toFixed produces plain decimal strings.
const valueArb: fc.Arbitrary<number> = fc.oneof(
  fc.double({ min: -1e9, max: 1e9, noNaN: true, noDefaultInfinity: true }),
  fc.double({ min: -1, max: 1, noNaN: true, noDefaultInfinity: true }),
  fc.integer({ min: -1_000_000, max: 1_000_000 }).map((n) => n),
)

// Unit labels for unbounded metrics: a plain unit or none. `%` is deliberately
// excluded — for unbounded metrics `%` would be a legitimate unit rather than a
// normalization artifact, so excluding it keeps the "no percentage conversion"
// assertion unambiguous.
const unitArb: fc.Arbitrary<string | null> = fc.option(
  fc.constantFrom('ms', 'tokens', 'requests', 'seconds', 'MB', 'tokens/s', 'count'),
  { nil: null },
)

const rawPrecisionArb: fc.Arbitrary<number> = fc.integer({ min: 0, max: 6 })

// Random unbounded (benchmark-native) spec. It is never reference-equal to
// DEFAULT_METRIC_SPEC, so isSpecUndefined must be false for these.
const unboundedSpecArb: fc.Arbitrary<MetricDisplaySpec> = fc
  .record({
    key: fc.stringMatching(/^[a-zA-Z][a-zA-Z0-9_]*$/),
    labelKey: fc.stringMatching(/^[a-zA-Z][a-zA-Z0-9_.]*$/),
    direction: fc.constantFrom('higher-is-better' as const, 'lower-is-better' as const),
    unit: unitArb,
    rawPrecision: rawPrecisionArb,
    percentPrecision: fc.integer({ min: 0, max: 4 }),
  })
  .map((partial) => ({ ...partial, boundedness: 'unbounded' as const }))

describe('formatMetric (Property 3: 非 bounded 与 spec 缺失不触发百分比转换)', () => {
  it('never converts an unbounded metric to a percentage, regardless of magnitude', () => {
    fc.assert(
      fc.property(unboundedSpecArb, valueArb, (spec, value) => {
        const result = formatMetric(value, spec, identityTranslate)

        // No normalization-driven percentage: primary carries no `%` because
        // the unit is never `%` for these specs.
        expect(result.primary.includes('%')).toBe(false)
        expect(result.raw.includes('%')).toBe(false)

        // Raw value is rendered at the spec's rawPrecision (Req 1.4, 1.7).
        expect(decimalPlaces(result.raw)).toBe(spec.rawPrecision)

        // Unit is preserved as-is (Req 1.4); a registered spec is not "undefined".
        expect(result.unitLabel).toBe(spec.unit ?? '')
        expect(result.isMissing).toBe(false)
        expect(result.isSpecUndefined).toBe(false)

        // Primary is the raw string joined with the unit — never a percentage.
        if (spec.unit === null) {
          expect(result.primary).toBe(result.raw)
        } else {
          expect(result.primary.startsWith(result.raw)).toBe(true)
          expect(result.primary.endsWith(spec.unit)).toBe(true)
        }
      }),
    )
  })

  it('marks a missing spec as undefined and shows the raw value at 4 decimals', () => {
    fc.assert(
      fc.property(valueArb, (value) => {
        const result = formatMetric(value, DEFAULT_METRIC_SPEC, identityTranslate)

        // Missing spec is detected by reference equality (Req 1.13).
        expect(result.isSpecUndefined).toBe(true)
        expect(result.isMissing).toBe(false)

        // Default 4-decimal raw value, no unit, no percentage inference.
        expect(decimalPlaces(result.raw)).toBe(DEFAULT_RAW_PRECISION)
        expect(result.unitLabel).toBe('')
        expect(result.primary).toBe(result.raw)
        expect(result.primary.includes('%')).toBe(false)
      }),
    )
  })
})
