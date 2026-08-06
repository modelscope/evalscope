/**
 * A metric must be rendered on its own scale.
 *
 * Feature: metric-semantics-governance. Regression test for a bug where report scores were
 * formatted through a fixed "0-1 ratio" contract. A benchmark reporting an official 0-100 scale
 * (arena_hard's WeightedScorePercent) then rendered as `8725%`, and a lower-is-better metric was
 * coloured and ranked as if higher were better. The guard is that a value is only ever formatted
 * with the semantics the backend attached to it.
 */

import { describe, expect, it } from 'vitest'

import { formatMetric, getBoundedQualityRatio } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'
import { RATIO_PERCENT_SEMANTICS } from './primaryMetrics'

/** An official 0-100 scale: already a percentage, so the multiplier is 1. */
const POINTS_100: MetricSemantics = {
  semantic_id: 'quality.score.points_100',
  metric_name: 'Score',
  role: 'primary',
  direction: 'higher_is_better',
  value_range: { min: 0, max: 100 },
  display_kind: 'percent',
  display_multiplier: 1,
  display_unit: '%',
  display_precision: 1,
  contract_version: 1,
}

/** WER: a bounded ratio where lower is better. */
const WER: MetricSemantics = {
  ...RATIO_PERCENT_SEMANTICS,
  semantic_id: 'quality.wer.ratio',
  metric_name: 'WER',
  direction: 'lower_is_better',
}

describe('native scales are preserved', () => {
  it('renders an official 0-100 score without re-scaling it', () => {
    expect(formatMetric(87.25, POINTS_100).primary).toBe('87.3%')
  })

  it('shows what the bug looked like, so the two contracts stay distinguishable', () => {
    // Formatting the same value through a 0-1 ratio contract is what produced `8725%`.
    expect(formatMetric(87.25, RATIO_PERCENT_SEMANTICS).primary).toBe('8725%')
    expect(formatMetric(87.25, POINTS_100).primary).not.toBe(
      formatMetric(87.25, RATIO_PERCENT_SEMANTICS).primary,
    )
  })

  it('normalizes a 0-100 score against its own range', () => {
    // 87.25 of 100 is a high score; treating it as a ratio would clamp to 1 regardless.
    expect(getBoundedQualityRatio(87.25, POINTS_100)).toBeCloseTo(0.8725)
  })

  it('inverts a lower-is-better metric so a low WER reads as good', () => {
    expect(getBoundedQualityRatio(0.05, WER)).toBeCloseTo(0.95)
    expect(getBoundedQualityRatio(0.9, WER)).toBeCloseTo(0.1)
  })

  it('gives an unbounded judge score no colour scale at all', () => {
    const judge: MetricSemantics = {
      semantic_id: 'quality.judge_score.unbounded',
      metric_name: 'Judge Score',
      role: 'primary',
      direction: 'higher_is_better',
      value_range: null,
      display_kind: 'number',
      display_precision: 2,
      contract_version: 1,
    }

    expect(getBoundedQualityRatio(7.5, judge)).toBeNull()
    expect(formatMetric(7.5, judge).primary).toBe('7.5')
  })
})
