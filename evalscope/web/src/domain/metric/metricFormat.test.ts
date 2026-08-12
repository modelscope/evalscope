/**
 * Tests for the metric formatting primitives.
 *
 * - formatMetric depends only on the display fields.
 * - the frontend and the backend format the golden samples identically.
 * - a colour scale is available exactly when the metric is a bounded quality metric.
 * - a comparison verdict follows the metric's direction.
 */

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'
import goldenSamples from '../../../../../tests/report/semantics/golden_samples.json'
import {
  MISSING_PLACEHOLDER,
  formatDifference,
  formatMetric,
  formatMetricIdentityLabel,
  formatMetricLabel,
  formatMetricLabels,
  getBoundedQualityRatio,
  getComparisonVerdict,
  metricIdentityKey,
  roundHalfUp,
} from './metricFormat'
import { metricSemanticsSchema } from './MetricSemantics'
import type { MetricSemantics } from './MetricSemantics'
import { arbSemantics, diagnosticSemantics, ratioSemantics, secondsSemantics } from './__arbitraries__'

const NUM_RUNS = 100

describe('formatMetricIdentityLabel', () => {
  it('matches the backend title-cased dimension labels', () => {
    expect(
      formatMetricIdentityLabel(
        {
          name: 'accuracy',
          aggregation: 'mean',
          dimensions: { level: 'overall', target: 'answer', strict: true },
        },
        ratioSemantics(),
      ),
    ).toBe('Accuracy ↑ · Answer · Overall · Yes')
  })

  it('places an overlap variant before its statistic', () => {
    expect(
      formatMetricIdentityLabel(
        {
          name: 'rouge',
          aggregation: 'mean',
          dimensions: { statistic: 'recall', variant: 'l' },
        },
        ratioSemantics(),
      ),
    ).toBe('Accuracy ↑ · L · Recall')
  })
})

describe('metricIdentityKey', () => {
  it('preserves dimension types and delimiter boundaries', () => {
    const numeric = metricIdentityKey({ name: 'accuracy', aggregation: 'mean', dimensions: { k: 1 } })
    const text = metricIdentityKey({ name: 'accuracy', aggregation: 'mean', dimensions: { k: '1' } })
    const embedded = metricIdentityKey({ name: 'accuracy', aggregation: 'mean', dimensions: { a: 'x,b=y' } })
    const separate = metricIdentityKey({ name: 'accuracy', aggregation: 'mean', dimensions: { a: 'x', b: 'y' } })

    expect(numeric).not.toBe(text)
    expect(embedded).not.toBe(separate)
  })

  it('normalizes negative zero like backend JSON scalar keys', () => {
    expect(metricIdentityKey({ name: 'accuracy', aggregation: 'mean', dimensions: { value: -0 } })).toBe(
      'accuracy:mean[value=0]',
    )
  })
})

describe('roundHalfUp', () => {
  it('rounds a tie toward positive infinity', () => {
    expect(roundHalfUp(0.5, 0)).toBe(1)
    expect(roundHalfUp(-0.5, 0)).toBe(0)
    expect(roundHalfUp(2.5, 0)).toBe(3)
    expect(roundHalfUp(-2.5, 0)).toBe(-2)
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

  it('output ignores identity fields (raw_unit drives the raw text)', () => {
    fc.assert(
      fc.property(fc.double({ min: -1e6, max: 1e6, noNaN: true }), arbSemantics(), (value, semantics) => {
        const renamed: MetricSemantics = {
          ...semantics,
          semantic_id: 'another.semantic.id',
          metric_name: 'Another Name',
        }
        expect(formatMetric(value, renamed)).toEqual(formatMetric(value, semantics))
      }),
      { numRuns: NUM_RUNS },
    )
  })

  it('the same input always produces the same output', () => {
    fc.assert(
      fc.property(fc.double({ min: -1e6, max: 1e6, noNaN: true }), arbSemantics(), (value, semantics) => {
        expect(formatMetric(value, semantics)).toEqual(formatMetric(value, semantics))
      }),
      { numRuns: NUM_RUNS },
    )
  })
})

describe('metric labels', () => {
  it('uses the semantic name and direction for scored metrics', () => {
    expect(formatMetricLabel('mean_acc', ratioSemantics())).toBe('Accuracy ↑')
    expect(formatMetricLabel('latency', secondsSemantics())).toBe('Latency ↓')
  })

  it('keeps the final name for diagnostic and unresolved metrics', () => {
    expect(formatMetricLabel('failed_requests', diagnosticSemantics())).toBe('failed_requests')
    expect(formatMetricLabel('unknown_metric', null)).toBe('unknown_metric')
  })

  it('disambiguates repeated labels within one report', () => {
    const labels = formatMetricLabels([
      { metricName: 'mean_acc', semantics: ratioSemantics() },
      { metricName: 'mean_fact_acc', semantics: { ...ratioSemantics(), role: 'auxiliary' } },
    ])

    expect(labels).toEqual({
      mean_acc: 'Accuracy ↑ (mean_acc)',
      mean_fact_acc: 'Accuracy ↑ (mean_fact_acc)',
    })
  })
})

describe('golden samples', () => {
  it('matches the backend formatter character for character', () => {
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

  it('validates every backend contract sample through the canonical Zod schema', () => {
    for (const sample of goldenSamples) {
      if (sample.semantics !== null) metricSemanticsSchema.parse(sample.semantics)
    }
  })

  it('keeps the Zod field set aligned with the backend Pydantic wire contract', () => {
    const backendContract = goldenSamples.find((sample) => sample.semantics !== null)?.semantics
    expect(backendContract).toBeTruthy()
    expect(Object.keys(metricSemanticsSchema.shape).sort()).toEqual(Object.keys(backendContract!).sort())
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

  it('a returned ratio is always within [0, 1]', () => {
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

  it('a scale exists only for a non-diagnostic bounded directed metric', () => {
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

describe('formatDifference', () => {
  const ACCURACY_RATIO: MetricSemantics = {
    semantic_id: 'quality.accuracy.ratio',
    metric_name: 'Accuracy',
    role: 'primary',
    direction: 'higher_is_better',
    value_range: { min: 0, max: 1 },
    display_kind: 'percent',
    display_multiplier: 100,
    display_unit: '%',
    display_precision: 1,
    contract_version: 1,
  }

  it('scales a native-ratio difference into percentage points', () => {
    // A swing from 0 to 1 on a ratio metric is 100 points. Formatting it through semantics whose
    // multiplier is ignored -- which happens outside the `percent` branch -- yields "1 pp", which
    // understates a full-range swing by a factor of a hundred.
    expect(formatDifference(1, ACCURACY_RATIO).primary).toBe('100 pp')
    expect(formatDifference(0.5, ACCURACY_RATIO).primary).toBe('50 pp')
    expect(formatDifference(0.408, ACCURACY_RATIO).primary).toBe('40.8 pp')
  })

  it('leaves a value already in display scale alone', () => {
    // Perf summary cells arrive pre-scaled, and their semantics say so with a multiplier of 1.
    const preScaled: MetricSemantics = { ...ACCURACY_RATIO, display_multiplier: 1 }

    expect(formatDifference(7.5, preScaled).primary).toBe('7.5 pp')
  })

  it('keeps the unit of a non-percent metric', () => {
    const seconds: MetricSemantics = {
      semantic_id: 'perf.latency.seconds',
      metric_name: 'Latency',
      role: 'primary',
      direction: 'lower_is_better',
      display_kind: 'number',
      display_unit: 's',
      display_precision: 3,
      contract_version: 1,
    }

    // A difference of seconds is seconds, so nothing is converted.
    expect(formatDifference(0.25, seconds).primary).toBe('0.25 s')
  })

  it('carries no quality, so callers cannot colour or rank it', () => {
    // A spread has a size but not a direction: 100 pp is neither good nor bad on its own.
    expect(formatDifference(0.5, ACCURACY_RATIO).isDiagnosticFallback).toBe(true)
  })

  it('reports a missing difference as missing', () => {
    expect(formatDifference(null, ACCURACY_RATIO).isMissing).toBe(true)
    expect(formatDifference(Number.NaN, ACCURACY_RATIO).isMissing).toBe(true)
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

  it('the verdict is decided by the direction alone', () => {
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
