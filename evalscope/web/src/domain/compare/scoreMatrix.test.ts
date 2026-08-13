// Direct tests for the score-matrix helpers.
//
// These reach 100% line coverage through ComparePage's render tests, which is not
// the same as covering their decisions: the "no verdict" cases (a metric with no
// direction, an equal value, a row with no spread) all return the same plain
// surface colour and are indistinguishable from each other in a rendered table.
// Asserting them here pins each decision separately.

import { describe, expect, it } from 'vitest'

import { comparisonDeltaBackground, computeDeltaRanges, signedDifference } from './scoreMatrix'
import type { MetricSemantics } from '@/domain/metric'

const HIGHER_IS_BETTER: MetricSemantics = {
  semantic_id: 'quality.accuracy',
  metric_name: 'Accuracy',
  kind: 'quality',
  direction: 'higher_is_better',
  display_kind: 'percent',
  display_unit: '%',
  display_precision: 2,
}

const LOWER_IS_BETTER: MetricSemantics = { ...HIGHER_IS_BETTER, direction: 'lower_is_better' }
const NO_DIRECTION: MetricSemantics = { ...HIGHER_IS_BETTER, kind: 'diagnostic', direction: 'none' }

const PLAIN = 'var(--bg-deep)'

describe('comparisonDeltaBackground', () => {
  it('tints an improvement with the success colour', () => {
    expect(comparisonDeltaBackground(0.1, 0.1, HIGHER_IS_BETTER)).toContain('var(--success)')
  })

  it('tints a regression with the danger colour', () => {
    expect(comparisonDeltaBackground(-0.1, 0.1, HIGHER_IS_BETTER)).toContain('var(--danger)')
  })

  it('reads the direction of the metric, not the sign of the delta', () => {
    // A decrease is the improvement when lower is better.
    expect(comparisonDeltaBackground(-0.1, 0.1, LOWER_IS_BETTER)).toContain('var(--success)')
    expect(comparisonDeltaBackground(0.1, 0.1, LOWER_IS_BETTER)).toContain('var(--danger)')
  })

  it('leaves a metric with no direction untinted', () => {
    // A diagnostic carries no better/worse, so a colour would assert a verdict it does not have.
    expect(comparisonDeltaBackground(0.5, 0.5, NO_DIRECTION)).toBe(PLAIN)
  })

  it('leaves an equal value untinted', () => {
    expect(comparisonDeltaBackground(0, 0.1, HIGHER_IS_BETTER)).toBe(PLAIN)
  })

  it('leaves a row with no spread untinted', () => {
    // Every report scored the same: there is no scale to place this delta on.
    expect(comparisonDeltaBackground(0.1, 0, HIGHER_IS_BETTER)).toBe(PLAIN)
  })

  it('leaves a metric with absent semantics untinted', () => {
    expect(comparisonDeltaBackground(0.1, 0.1, undefined)).toBe(PLAIN)
  })

  it('scales the tint with the delta relative to the row maximum', () => {
    const weight = (css: string) => Number(/(\d+)%/.exec(css)![1])
    const strongest = weight(comparisonDeltaBackground(0.2, 0.2, HIGHER_IS_BETTER))
    const weaker = weight(comparisonDeltaBackground(0.05, 0.2, HIGHER_IS_BETTER))

    expect(weaker).toBeLessThan(strongest)
    // Bounded at both ends so no cell is invisible and none is fully saturated.
    expect(weaker).toBeGreaterThanOrEqual(6)
    expect(strongest).toBeLessThanOrEqual(30)
  })

  it('clamps a delta larger than the row maximum to the strongest tint', () => {
    const at = comparisonDeltaBackground(0.2, 0.2, HIGHER_IS_BETTER)
    const beyond = comparisonDeltaBackground(0.9, 0.2, HIGHER_IS_BETTER)
    expect(beyond).toBe(at)
  })
})

describe('signedDifference', () => {
  it('prefixes a positive delta with an explicit plus', () => {
    expect(signedDifference(0.1, HIGHER_IS_BETTER).startsWith('+')).toBe(true)
  })

  it('leaves a negative delta to carry its own sign', () => {
    const formatted = signedDifference(-0.1, HIGHER_IS_BETTER)
    expect(formatted.startsWith('-')).toBe(true)
    expect(formatted.startsWith('+')).toBe(false)
  })

  it('does not sign a zero delta', () => {
    expect(signedDifference(0, HIGHER_IS_BETTER).startsWith('+')).toBe(false)
  })
})

describe('computeDeltaRanges', () => {
  const rows = [
    { dataset_id: 'gsm8k', a: 0.9, b: 0.5, c: 0.7 },
    { dataset_id: 'arc', a: 0.4, b: 0.4, c: 0.4 },
  ]

  it('keys the largest absolute delta by dataset id', () => {
    expect(computeDeltaRanges(rows, ['a', 'b', 'c'], 'a')).toEqual({ gsm8k: 0.4, arc: 0 })
  })

  it('scales each row independently of the others', () => {
    // `arc` has no spread even though `gsm8k` does, so its own range stays 0.
    const ranges = computeDeltaRanges(rows, ['a', 'b', 'c'], 'a')
    expect(ranges.arc).toBe(0)
  })

  it('follows the chosen baseline', () => {
    expect(computeDeltaRanges(rows, ['a', 'b', 'c'], 'b').gsm8k).toBeCloseTo(0.4)
  })

  it('never returns a negative range', () => {
    expect(computeDeltaRanges([{ dataset_id: 'x', a: 0.5 }], ['a'], 'a')).toEqual({ x: 0 })
  })

  it('returns an empty map for no rows', () => {
    expect(computeDeltaRanges([], ['a'], 'a')).toEqual({})
  })

  it('ignores missing scores and leaves a row without the selected baseline unscaled', () => {
    const sparseRows = [
      { dataset_id: 'accuracy', a: 0.8, b: 0.9 },
      { dataset_id: 'wer', b: 0.2, c: 0.1 },
    ]

    const ranges = computeDeltaRanges(sparseRows, ['a', 'b', 'c'], 'a')
    expect(ranges.accuracy).toBeCloseTo(0.1)
    expect(ranges.wer).toBe(0)
  })
})
