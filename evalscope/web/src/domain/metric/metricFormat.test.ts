// Feature: frontend-refactor-2026-07, metricFormat unit + property tests.
//
// Merged coverage for `formatMetric` / `roundHalfUp`:
//   - example-based boundary edges (round-half-up ties, missing values, bounded
//     ratio + storedAsHundred rendering);
//   - Property 1: bounded ratio percentage display and rounding invariants;
//   - Property 2: missing placeholder is distinguishable from a legitimate 0;
//   - Property 3: non-bounded / missing specs do not trigger percentage
//     conversion.
//
// Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.13, 8.10

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import {
  DEFAULT_METRIC_SPEC,
  DEFAULT_RAW_PRECISION,
  type MetricDisplaySpec,
} from './MetricDisplaySpec'
import { formatMetric, MISSING_PLACEHOLDER, roundHalfUp } from './metricFormat'

// Bounded ratios never consult the translate function; unbounded ones echo the
// key. Never injects the missing placeholder, keeping the "legitimate value has
// no placeholder" assertions unambiguous.
const identityTranslate = (key: string): string => key

const precisionArb: fc.Arbitrary<number> = fc.integer({ min: 0, max: 6 })

/**
 * Bounded-ratio spec generator shared by the percentage-rendering properties.
 * `boundedness` is fixed to 'bounded'; precisions and the `storedAsHundred`
 * flag vary. `unit` is null (bounded ratios display as percentages).
 */
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

/* ─── Boundary: example-based numeric edges ───────────────────── */

/** Build a bounded-ratio spec with sensible defaults, overridable per test. */
function boundedSpec(overrides: Partial<MetricDisplaySpec> = {}): MetricDisplaySpec {
  return {
    key: 'accuracy',
    labelKey: 'metric.accuracy',
    boundedness: 'bounded',
    direction: 'higher-is-better',
    unit: null,
    rawPrecision: 4,
    percentPrecision: 1,
    ...overrides,
  }
}

describe('roundHalfUp — round-half-up boundaries', () => {
  it('rounds an exact .5 tie toward +infinity at precision 0', () => {
    expect(roundHalfUp(0.5, 0)).toBe(1)
    expect(roundHalfUp(2.5, 0)).toBe(3)
    expect(roundHalfUp(1.5, 0)).toBe(2)
  })

  it('rounds floating-point trap ties correctly (avoids toFixed drift)', () => {
    // 92.05 * 10 drifts below 920.5 in binary; round-half-up must still yield 92.1.
    expect(roundHalfUp(92.05, 1)).toBe(92.1)
    // 1.005 * 100 drifts below 100.5 in binary; round-half-up must still yield 1.01.
    expect(roundHalfUp(1.005, 2)).toBe(1.01)
  })

  it('rounds negative ties toward +infinity (magnitude toward zero)', () => {
    expect(roundHalfUp(-0.5, 0)).toBe(0)
    expect(roundHalfUp(-2.5, 0)).toBe(-2)
    expect(roundHalfUp(-1.5, 0)).toBe(-1)
  })

  it('leaves values already at or below the requested precision unchanged', () => {
    expect(roundHalfUp(1.2, 4)).toBe(1.2)
    expect(roundHalfUp(0, 2)).toBe(0)
  })

  it('returns non-finite inputs unchanged', () => {
    expect(Number.isNaN(roundHalfUp(NaN, 2))).toBe(true)
    expect(roundHalfUp(Infinity, 2)).toBe(Infinity)
    expect(roundHalfUp(-Infinity, 2)).toBe(-Infinity)
  })
})

describe('formatMetric — missing values', () => {
  it('renders null as the missing placeholder with isMissing = true', () => {
    const result = formatMetric(null, boundedSpec(), identityTranslate)
    expect(result.primary).toBe(MISSING_PLACEHOLDER)
    expect(result.raw).toBe(MISSING_PLACEHOLDER)
    expect(result.unitLabel).toBe('')
    expect(result.isMissing).toBe(true)
  })

  it('renders undefined as the missing placeholder with isMissing = true', () => {
    const result = formatMetric(undefined, boundedSpec(), identityTranslate)
    expect(result.primary).toBe(MISSING_PLACEHOLDER)
    expect(result.raw).toBe(MISSING_PLACEHOLDER)
    expect(result.isMissing).toBe(true)
  })

  it('renders NaN as the missing placeholder with isMissing = true', () => {
    const result = formatMetric(NaN, boundedSpec(), identityTranslate)
    expect(result.primary).toBe(MISSING_PLACEHOLDER)
    expect(result.raw).toBe(MISSING_PLACEHOLDER)
    expect(result.isMissing).toBe(true)
  })

  it('treats a legitimate 0 as a real value, never as missing or blank', () => {
    const result = formatMetric(0, boundedSpec(), identityTranslate)
    expect(result.isMissing).toBe(false)
    expect(result.primary).not.toBe(MISSING_PLACEHOLDER)
    expect(result.primary).toBe('0.0%')
    expect(result.raw).toBe('0.0000')
  })
})

describe('formatMetric — bounded ratio rendering', () => {
  it('renders a 0-1 ratio as a percentage primary and 0-1 raw', () => {
    const result = formatMetric(0.92, boundedSpec(), identityTranslate)
    expect(result.primary).toBe('92.0%')
    expect(result.raw).toBe('0.9200')
    expect(result.unitLabel).toBe('')
    expect(result.isMissing).toBe(false)
    expect(result.isSpecUndefined).toBe(false)
  })
})

describe('formatMetric — storedAsHundred normalization', () => {
  it('normalizes a stored 0-100 value to a 0-1 ratio before formatting', () => {
    const result = formatMetric(92, boundedSpec({ storedAsHundred: true }), identityTranslate)
    expect(result.primary).toBe('92.0%')
    expect(result.raw).toBe('0.9200')
    expect(result.isMissing).toBe(false)
  })
})

/* ─── Property 1: bounded ratio percentage display + rounding ─── */

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

/* ─── Property 2: missing placeholder vs legitimate 0 ─────────── */

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

/* ─── Property 3: no percentage conversion for unbounded/missing ─ */

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

// Random unbounded (benchmark-native) spec. It is never reference-equal to
// DEFAULT_METRIC_SPEC, so isSpecUndefined must be false for these.
const nativeSpecArb: fc.Arbitrary<MetricDisplaySpec> = fc
  .record({
    key: fc.stringMatching(/^[a-zA-Z][a-zA-Z0-9_]*$/),
    labelKey: fc.stringMatching(/^[a-zA-Z][a-zA-Z0-9_.]*$/),
    direction: fc.constantFrom('higher-is-better' as const, 'lower-is-better' as const),
    unit: unitArb,
    rawPrecision: precisionArb,
    percentPrecision: fc.integer({ min: 0, max: 4 }),
  })
  .map((partial) => ({ ...partial, boundedness: 'unbounded' as const }))

describe('formatMetric (Property 3: non-bounded and missing specs do not trigger percentage conversion)', () => {
  it('never converts an unbounded metric to a percentage, regardless of magnitude', () => {
    fc.assert(
      fc.property(nativeSpecArb, valueArb, (spec, value) => {
        const result = formatMetric(value, spec, identityTranslate)

        // No normalization-driven percentage: primary carries no `%` because
        // the unit is never `%` for these specs.
        expect(result.primary.includes('%')).toBe(false)
        expect(result.raw.includes('%')).toBe(false)

        // Raw value is rendered at the spec's rawPrecision.
        expect(decimalPlaces(result.raw)).toBe(spec.rawPrecision)

        // Unit is preserved as-is; a registered spec is not "undefined".
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

        // Missing spec is detected by reference equality.
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
