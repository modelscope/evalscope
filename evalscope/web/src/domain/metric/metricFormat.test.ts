/**
 * Tests for the metric formatting primitives.
 *
 * Feature: metric-semantics-governance
 * - Property 27: formatMetric depends only on the display fields.
 * - Property 28: the frontend and the backend format the golden samples identically.
 * - Property 32: a colour scale is available exactly when the metric is a bounded quality metric.
 * - Property 38: a comparison verdict follows the metric's direction.
 */

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'
import goldenSamples from '../../../../metrics/semantics/golden_samples.json'
import {
  formatMetric,
  getBoundedQualityRatio,
  getComparisonVerdict,
  MISSING_PLACEHOLDER,
  roundHalfUp,
} from './metricFormat'
import type { MetricSemantics } from './MetricSemantics'
import { arbSemantics, ratioSemantics, secondsSemantics } from './__arbitraries__'

const NUM_RUNS = 100

describe('roundHalfUp', () => {
  it('rounds a tie toward positive infinity', () => {
    expect(roundHalfUp(0.5, 0)).toBe(1)
    expect(roundHalfUp(-0.5, 0)).toBe(0)
    expect(roundHalfUp(2.5, 0)).toBe(3)
  })

  it('avoids binary floating point drift', () => {
    expect(roundHalfUp(1.005, 2)).toBe(1.01)
  })

  it('returns non-finite input unchanged', () => {
    expect(Number.isNaN(roundHalfUp(NaN, 2))).toBe(true)
  })
})

describe('formatMetric', () => {
  it('renders a missing value as the placeholder', () => {
    for (const value of [null, undefined, NaN, Infinity]) {
      const formatted = formatMetric(value as number | null | undefined, ratioSemantics())
      expect(formatted.primary).toBe(MISSING_PLACEHOLDER)
      expect(formatted.isMissing).toBe(true)
    }
  })

  it('scales and suffixes a percent metric without a separator', () => {
    expect(formatMetric(0.8567, ratioSemantics()).primary).toBe('85.7%')
  })

  it('separates a number metric from its unit with a space', () => {
    expect(formatMetric(1.23456, secondsSemantics()).primary).toBe('1.235 s')
  })

  it('falls back to a plain number without semantics', () => {
    const formatted = formatMetric(0.5, null)
    expect(formatted.primary).toBe('0.5')
    expect(formatted.unitLabel).toBe('')
    expect(formatted.isDiagnosticFallback).toBe(true)
  })

  it('marks a diagnostic metric as a fallback so the UI drops colour scales', () => {
    const diagnostic: MetricSemantics = { ...ratioSemantics(), role: 'diagnostic', direction: 'none' }
    expect(formatMetric(0.5, diagnostic).isDiagnosticFallback).toBe(true)
  })

  it('Property 27: output ignores identity fields (raw_unit drives the raw text)', () => {
    fc.assert(
      fc.property(fc.double({ min: -1e6, max: 1e6, noNaN: true }), arbSemantics(), (value, semantics) => {
        const renamed: MetricSemantics = {
          ...semantics,
          semantic_id: 'another.semantic.id',
          metric_name: 'Another Name',
          comparison_group: 'another.group',
        }
        expect(formatMetric(value, renamed)).toEqual(formatMetric(value, semantics))
      }),
      { numRuns: NUM_RUNS },
    )
  })

  it('Property 27: the same input always produces the same output', () => {
    fc.assert(
      fc.property(fc.double({ min: -1e6, max: 1e6, noNaN: true }), arbSemantics(), (value, semantics) => {
        expect(formatMetric(value, semantics)).toEqual(formatMetric(value, semantics))
      }),
      { numRuns: NUM_RUNS },
    )
  })
})

describe('golden samples', () => {
  it('Property 28: matches the backend formatter character for character', () => {
    for (const sample of goldenSamples as Array<{ semantics: MetricSemantics | null; value: number | null; expected_primary: string }>) {
      expect(formatMetric(sample.value, sample.semantics).primary).toBe(sample.expected_primary)
    }
  })

  it('covers percent, number and missing paths', () => {
    const kinds = new Set(
      (goldenSamples as Array<{ semantics: MetricSemantics | null }>).map((s) => s.semantics?.display_kind ?? 'none'),
    )
    expect(kinds.has('percent')).toBe(true)
    expect(kinds.has('number')).toBe(true)
  })
})

describe('getBoundedQualityRatio', () => {
  it('normalizes a higher-is-better ratio as-is', () => {
    expect(getBoundedQualityRatio(0.25, ratioSemantics())).toBeCloseTo(0.25)
  })

  it('inverts a lower-is-better metric so fuller is better', () => {
    const wer: MetricSemantics = { ...ratioSemantics(), direction: 'lower_is_better' }
    expect(getBoundedQualityRatio(0.25, wer)).toBeCloseTo(0.75)
  })

  it('returns null for a diagnostic metric', () => {
    const diagnostic: MetricSemantics = { ...ratioSemantics(), role: 'diagnostic', direction: 'none' }
    expect(getBoundedQualityRatio(0.5, diagnostic)).toBeNull()
  })

  it('returns null without a value range', () => {
    expect(getBoundedQualityRatio(1.5, secondsSemantics())).toBeNull()
  })

  it('returns null without semantics or value', () => {
    expect(getBoundedQualityRatio(0.5, null)).toBeNull()
    expect(getBoundedQualityRatio(null, ratioSemantics())).toBeNull()
  })

  it('Property 32: a returned ratio is always within [0, 1]', () => {
    fc.assert(
      fc.property(fc.double({ min: -1e3, max: 1e3, noNaN: true }), arbSemantics(), (value, semantics) => {
        const ratio = getBoundedQualityRatio(value, semantics)
        if (ratio !== null) {
          expect(ratio).toBeGreaterThanOrEqual(0)
          expect(ratio).toBeLessThanOrEqual(1)
        }
      }),
      { numRuns: NUM_RUNS },
    )
  })

  it('Property 32: a scale exists only for a non-diagnostic bounded directed metric', () => {
    fc.assert(
      fc.property(fc.double({ min: 0, max: 1, noNaN: true }), arbSemantics(), (value, semantics) => {
        const eligible = semantics.role !== 'diagnostic'
          && semantics.direction !== 'none'
          && semantics.value_range != null
        expect(getBoundedQualityRatio(value, semantics) !== null).toBe(eligible)
      }),
      { numRuns: NUM_RUNS },
    )
  })
})

describe('getComparisonVerdict', () => {
  it('follows higher_is_better', () => {
    expect(getComparisonVerdict(0.1, ratioSemantics())).toBe('better')
    expect(getComparisonVerdict(-0.1, ratioSemantics())).toBe('worse')
  })

  it('follows lower_is_better', () => {
    const latency = secondsSemantics()
    expect(getComparisonVerdict(-0.1, latency)).toBe('better')
    expect(getComparisonVerdict(0.1, latency)).toBe('worse')
  })

  it('calls a zero delta equal', () => {
    expect(getComparisonVerdict(0, ratioSemantics())).toBe('equal')
  })

  it('never judges a diagnostic metric or a directionless one', () => {
    const diagnostic: MetricSemantics = { ...ratioSemantics(), role: 'diagnostic', direction: 'none' }
    expect(getComparisonVerdict(5, diagnostic)).toBe('incomparable')
    expect(getComparisonVerdict(5, null)).toBe('incomparable')
  })

  it('Property 38: the verdict is decided by the direction alone', () => {
    fc.assert(
      fc.property(fc.double({ min: -100, max: 100, noNaN: true }), arbSemantics(), (delta, semantics) => {
        const verdict = getComparisonVerdict(delta, semantics)
        if (semantics.role === 'diagnostic' || semantics.direction === 'none') {
          expect(verdict).toBe('incomparable')
        } else if (delta === 0) {
          expect(verdict).toBe('equal')
        } else {
          const improved = semantics.direction === 'higher_is_better' ? delta > 0 : delta < 0
          expect(verdict).toBe(improved ? 'better' : 'worse')
        }
      }),
      { numRuns: NUM_RUNS },
    )
  })
})
