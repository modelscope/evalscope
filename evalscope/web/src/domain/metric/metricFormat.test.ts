/**
 * Tests for the metric formatting primitives.
 *
 * - formatMetric depends only on the display fields.
 * - the frontend and the backend format the golden samples identically.
 * - a colour scale is available exactly when the metric is a bounded quality metric.
 * - a comparison verdict follows the metric's direction.
 */

import { describe, expect, it } from 'vitest'
import goldenFixtureJson from '../../../../../tests/report/semantics/golden_samples.json'
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
import type { MetricIdentity } from './metricFormat'
import { metricSemanticsSchema } from './MetricSemantics'
import type { MetricSemantics } from './MetricSemantics'
import { diagnosticSemantics, ratioSemantics, secondsSemantics } from './__arbitraries__'

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
    const diagnostic: MetricSemantics = { ...ratioSemantics(), kind: 'diagnostic', direction: 'none' }
    expect(formatMetric(0.5, diagnostic).isDiagnosticFallback).toBe(true)
  })

  it('ignores semantic identity fields when formatting a value', () => {
    const semantics = secondsSemantics()
    const renamed = { ...semantics, semantic_id: 'another.semantic.id', metric_name: 'Another Name' }

    expect(formatMetric(1.25, renamed)).toEqual(formatMetric(1.25, semantics))
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
      { metricName: 'mean_fact_acc', semantics: { ...ratioSemantics(), kind: 'quality' } },
    ])

    expect(labels).toEqual({
      mean_acc: 'Accuracy ↑ (mean_acc)',
      mean_fact_acc: 'Accuracy ↑ (mean_fact_acc)',
    })
  })
})

describe('golden samples', () => {
  // Which branches the shared fixture must exercise is asserted once, on the backend side, in
  // `tests/report/semantics/test_golden_samples.py::TestGoldenSampleCoverage`. This suite only
  // checks that the TypeScript implementation agrees with the pinned expectations.
  interface GoldenSampleSpec {
    id: string
    semantics_ref: string | null
    value: number | null
    expected_primary: string
    expected_raw: string
    /** Present only on samples that pin the label path. */
    identity?: MetricIdentity
    expected_label?: string
    legacy_name?: string
  }

  interface GoldenFixture {
    semantics: Record<string, MetricSemantics>
    samples: GoldenSampleSpec[]
  }

  const goldenFixture = goldenFixtureJson as unknown as GoldenFixture
  const goldenSamples = goldenFixture.samples.map((sample) => ({
    ...sample,
    semantics: sample.semantics_ref === null ? null : goldenFixture.semantics[sample.semantics_ref],
  }))

  it('matches the backend formatter character for character', () => {
    for (const sample of goldenSamples) {
      expect(formatMetric(sample.value, sample.semantics).primary).toBe(sample.expected_primary)
    }
  })

  it('renders the same label as the backend', () => {
    // `formatMetricIdentityLabel` and the backend `format_metric_label` are a second pair of
    // parallel implementations. Without this the two can drift -- they already had, over the
    // casing of an acronym in a dimension value.
    const labelled = goldenSamples.filter(
      (sample) => sample.identity != null && sample.expected_label != null,
    )
    expect(labelled.length).toBeGreaterThan(0)
    for (const sample of labelled) {
      expect(formatMetricIdentityLabel(sample.identity!, sample.semantics, sample.legacy_name)).toBe(
        sample.expected_label,
      )
    }
  })

  it('matches the backend raw text, which no backend surface renders', () => {
    for (const sample of goldenSamples) {
      expect(formatMetric(sample.value, sample.semantics).raw).toBe(sample.expected_raw)
    }
  })

  it('validates every backend contract sample through the canonical Zod schema', () => {
    for (const semantics of Object.values(goldenFixture.semantics)) {
      metricSemanticsSchema.parse(semantics)
    }
  })

  it('keeps the Zod field set aligned with the backend Pydantic wire contract', () => {
    const backendContract = Object.values(goldenFixture.semantics)[0]
    expect(backendContract).toBeTruthy()
    expect(Object.keys(metricSemanticsSchema.shape).sort()).toEqual(Object.keys(backendContract).sort())
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
    const diagnostic: MetricSemantics = { ...ratioSemantics(), kind: 'diagnostic', direction: 'none' }
    expect(getBoundedQualityRatio(0.5, diagnostic)).toBeNull()
  })

  it('returns null without a value range', () => {
    expect(getBoundedQualityRatio(1.5, secondsSemantics())).toBeNull()
  })

  it('returns null without semantics or value', () => {
    expect(getBoundedQualityRatio(0.5, null)).toBeNull()
    expect(getBoundedQualityRatio(null, ratioSemantics())).toBeNull()
  })

  it('clamps bounded ratios to [0, 1]', () => {
    expect(getBoundedQualityRatio(-2, ratioSemantics())).toBe(0)
    expect(getBoundedQualityRatio(3, ratioSemantics())).toBe(1)
  })
})

describe('formatDifference', () => {
  const ACCURACY_RATIO: MetricSemantics = {
    semantic_id: 'quality.accuracy.ratio',
    metric_name: 'Accuracy',
    kind: 'quality',
    direction: 'higher_is_better',
    value_range: { min: 0, max: 1 },
    display_kind: 'percent',
    display_multiplier: 100,
    display_unit: '%',
    display_precision: 1,
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
      kind: 'quality',
      direction: 'lower_is_better',
      display_kind: 'number',
      display_unit: 's',
      display_precision: 3,
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
    const diagnostic: MetricSemantics = { ...ratioSemantics(), kind: 'diagnostic', direction: 'none' }
    expect(getComparisonVerdict(5, diagnostic)).toBe('incomparable')
    expect(getComparisonVerdict(5, null)).toBe('incomparable')
  })

})
