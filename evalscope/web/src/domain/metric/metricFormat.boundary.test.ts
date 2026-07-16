// Feature: frontend-refactor-2026-07, Task 3.6: metricFormat boundary unit tests.
//
// Example-based (non-property) coverage of the numeric edges that the property
// tests assert only in aggregate:
//   - roundHalfUp tie behaviour (`.5` rounds toward +infinity, including the
//     floating-point trap values 92.05 and 1.005, plus negative ties);
//   - formatMetric missing-value handling (null / undefined / NaN → placeholder,
//     while a legitimate 0 is a real value);
//   - bounded-ratio percentage + raw rendering;
//   - storedAsHundred normalization from a 0-100 stored value to a 0-1 ratio.
//
// Validates: Requirements 1.2, 1.3, 1.8

import { describe, expect, it } from 'vitest'

import type { MetricDisplaySpec } from './MetricDisplaySpec'
import { formatMetric, MISSING_PLACEHOLDER, roundHalfUp } from './metricFormat'

// Bounded ratios never consult the translate function; unbounded ones echo the key.
const identityTranslate = (key: string): string => key

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

describe('formatMetric — missing values (Req 1.8)', () => {
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

describe('formatMetric — bounded ratio rendering (Req 1.2, 1.3)', () => {
  it('renders a 0-1 ratio as a percentage primary and 0-1 raw', () => {
    const result = formatMetric(0.92, boundedSpec(), identityTranslate)
    expect(result.primary).toBe('92.0%')
    expect(result.raw).toBe('0.9200')
    expect(result.unitLabel).toBe('')
    expect(result.isMissing).toBe(false)
    expect(result.isSpecUndefined).toBe(false)
  })
})

describe('formatMetric — storedAsHundred normalization (Req 1.2, 1.3)', () => {
  it('normalizes a stored 0-100 value to a 0-1 ratio before formatting', () => {
    const result = formatMetric(92, boundedSpec({ storedAsHundred: true }), identityTranslate)
    expect(result.primary).toBe('92.0%')
    expect(result.raw).toBe('0.9200')
    expect(result.isMissing).toBe(false)
  })
})
