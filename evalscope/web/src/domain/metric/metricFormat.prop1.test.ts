// Feature: frontend-refactor-2026-07, Property 1: Bounded ratio percentage display and rounding invariants
//
// For any bounded-ratio metric value and its MetricDisplaySpec
// (boundedness === 'bounded'), formatMetric must produce:
//   - `primary`: a percentage string with `percentPrecision` decimals, rounded
//     half up (ties toward +infinity);
//   - `raw`: the 0-1 ratio with `rawPrecision` decimals (default 4), rounded
//     half up;
//   - `unitLabel === ''`, `isMissing === false`, `isSpecUndefined === false`.
// When `spec.storedAsHundred` is set, the stored 0-100 value is normalized to
// 0-1 before both computations. The function is deterministic: repeated calls
// with the same (value, spec) return field-for-field identical output.
//
// Validates: Requirements 1.2, 1.3, 1.6, 1.7, 8.10

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import type { MetricDisplaySpec } from './MetricDisplaySpec'
import { formatMetric } from './metricFormat'

/**
 * Independent round-half-up oracle.
 *
 * Deliberately implemented with a different technique than the production code:
 * it manipulates the decimal digits of the float's shortest round-trip string
 * representation (`value.toString()`) instead of shifting via exponential
 * `Number()` parsing. Both must agree on every finite double. Ties (an exact
 * trailing `.5`) round toward positive infinity.
 */
function refRoundHalfUp(value: number, precision: number): number {
  if (!Number.isFinite(value)) {
    return value
  }
  const s = value.toString()
  // Exponential notation (e.g. "1e-8"): mirror the production arithmetic fallback.
  if (/[eE]/.test(s)) {
    const factor = 10 ** precision
    return Math.round(value * factor) / factor
  }

  const negative = s[0] === '-'
  const unsigned = negative ? s.slice(1) : s
  const dotIndex = unsigned.indexOf('.')
  const intStr = dotIndex === -1 ? unsigned : unsigned.slice(0, dotIndex)
  const fracStr = dotIndex === -1 ? '' : unsigned.slice(dotIndex + 1)

  // Already at or below the requested precision: no rounding needed.
  if (fracStr.length <= precision) {
    return value
  }

  // Magnitude split at the rounding position.
  const keptFrac = fracStr.slice(0, precision)
  const nextDigit = fracStr.charCodeAt(precision) - 48 // '0' === 48
  const restNonZero = /[1-9]/.test(fracStr.slice(precision + 1))
  const magnitude = BigInt(intStr + keptFrac)

  // Round-half-up toward +infinity.
  let rounded: bigint
  if (negative) {
    // Ties (exactly .5) keep magnitude (round toward +infinity == toward zero).
    const roundUpMagnitude = nextDigit > 5 || (nextDigit === 5 && restNonZero)
    rounded = -(roundUpMagnitude ? magnitude + 1n : magnitude)
  } else {
    // Ties (exactly .5) increase magnitude (round toward +infinity).
    const roundUpMagnitude = nextDigit >= 5
    rounded = roundUpMagnitude ? magnitude + 1n : magnitude
  }

  return Number(`${rounded}e${-precision}`)
}

/** Format a value exactly the way `formatMetric` does: round half up, then pad. */
function expectedFixed(value: number, precision: number): string {
  const safePrecision = precision >= 0 ? precision : 0
  return refRoundHalfUp(value, safePrecision).toFixed(safePrecision)
}

// A no-op translate function: bounded ratios have no unit label so `t` is never
// consulted, but formatMetric requires a translate argument.
const identityTranslate = (key: string): string => key

const precisionArb = fc.integer({ min: 0, max: 6 })

/**
 * Bounded-ratio spec generator. `boundedness` is fixed to 'bounded'; precisions
 * and the `storedAsHundred` flag vary. `unit` is null (bounded ratios display as
 * percentages).
 */
const boundedSpecArb: fc.Arbitrary<MetricDisplaySpec> = fc.record({
  key: fc.stringMatching(/^[a-z_]{1,12}$/),
  labelKey: fc.stringMatching(/^[a-z_.]{1,16}$/),
  direction: fc.constantFrom('higher-is-better', 'lower-is-better') as fc.Arbitrary<
    MetricDisplaySpec['direction']
  >,
  rawPrecision: precisionArb,
  percentPrecision: precisionArb,
  storedAsHundred: fc.boolean(),
}).map((partial) => ({
  ...partial,
  boundedness: 'bounded' as const,
  unit: null,
}))

/**
 * Generate a raw stored value paired with a spec. When `storedAsHundred` is set
 * the stored value lives in [0, 100]; otherwise it is a 0-1 ratio. A mix of
 * arbitrary doubles and "clean" fixed-decimal fractions is used so that exact
 * `.5` rounding boundaries are exercised alongside general floats.
 */
function storedValueArb(spec: MetricDisplaySpec): fc.Arbitrary<number> {
  const upper = spec.storedAsHundred ? 100 : 1
  const doubleArb = fc.double({ min: 0, max: upper, noNaN: true, noDefaultInfinity: true })
  // Clean fractions like n / 10^d that can land exactly on a rounding tie.
  const cleanArb = fc
    .tuple(fc.integer({ min: 0, max: 6 }), fc.nat({ max: 10 ** 6 }))
    .map(([d, n]) => Math.min((n / 10 ** d) % (upper + 1), upper))
  return fc.oneof(doubleArb, cleanArb)
}

const caseArb = boundedSpecArb.chain((spec) =>
  storedValueArb(spec).map((value) => ({ spec, value })),
)

describe('formatMetric — bounded ratio (Property 1: Bounded ratio percentage display and rounding invariants)', () => {
  it('renders primary as a round-half-up percentage and raw as the round-half-up ratio', () => {
    fc.assert(
      fc.property(caseArb, ({ spec, value }) => {
        const ratio = spec.storedAsHundred ? value / 100 : value

        const result = formatMetric(value, spec, identityTranslate)

        const expectedPrimary = `${expectedFixed(ratio * 100, spec.percentPrecision)}%`
        const expectedRaw = expectedFixed(ratio, spec.rawPrecision)

        expect(result.primary).toBe(expectedPrimary)
        expect(result.raw).toBe(expectedRaw)
        expect(result.unitLabel).toBe('')
        expect(result.isMissing).toBe(false)
        expect(result.isSpecUndefined).toBe(false)
      }),
      { numRuns: 100 },
    )
  })

  it('uses the default raw precision of 4 decimals when the spec keeps it', () => {
    const specWithDefaultRaw = boundedSpecArb.map((spec) => ({ ...spec, rawPrecision: 4 }))
    fc.assert(
      fc.property(
        specWithDefaultRaw.chain((spec) => storedValueArb(spec).map((value) => ({ spec, value }))),
        ({ spec, value }) => {
          const ratio = spec.storedAsHundred ? value / 100 : value
          const result = formatMetric(value, spec, identityTranslate)

          expect(result.raw).toBe(expectedFixed(ratio, 4))
          // A finite ratio always yields exactly 4 fractional digits at precision 4.
          expect(result.raw.split('.')[1]).toHaveLength(4)
        },
      ),
      { numRuns: 100 },
    )
  })

  it('is deterministic: repeated calls return field-for-field identical output', () => {
    fc.assert(
      fc.property(caseArb, ({ spec, value }) => {
        const first = formatMetric(value, spec, identityTranslate)
        const second = formatMetric(value, spec, identityTranslate)
        expect(second).toEqual(first)
      }),
      { numRuns: 100 },
    )
  })
})
