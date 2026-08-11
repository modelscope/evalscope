/**
 * Aggregating runs by what they measure, rather than listing them by when they happened.
 *
 * Feature: dashboard redesign. These tests pin the properties that make the aggregation safe to
 * present: a cell's spread is computed only among points sharing one metric, cross-cell ordering
 * uses a normalized spread and refuses to synthesize one for an unbounded metric, and a single run
 * is never reported as stable.
 */

import { describe, expect, it } from 'vitest'

import {
  aggregateRuns,
  cellKey,
  compareByInstability,
  computeCellStats,
  trendBounds,
  trendPosition,
} from './runAggregation'
import type { AggregatedRow, CellPoint } from './runAggregation'
import type { MetricSemantics } from '@/domain/metric'
import type { PerfRunSummary, ReportSummary } from '@/api/types'

const ACCURACY: MetricSemantics = {
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

/** Throughput has no ceiling, so it carries no `value_range`. */
const RPS: MetricSemantics = {
  semantic_id: 'perf.throughput.requests_per_second',
  metric_name: 'Best RPS',
  role: 'primary',
  direction: 'higher_is_better',
  display_kind: 'number',
  display_unit: 'req/s',
  display_precision: 4,
  contract_version: 1,
}

function point(timestamp: string, score: number): CellPoint {
  return { timestamp, score, runId: `run-${timestamp}` }
}

function report(over: Partial<ReportSummary> = {}): ReportSummary {
  return {
    name: 'run@qwen-plus',
    model_name: 'qwen-plus',
    dataset_name: 'iquiz',
    num_samples: 3,
    timestamp: '2026-08-07T10:00:00',
    primary_metrics: [
      { dataset_name: 'iquiz', identity: { name: 'accuracy', aggregation: 'mean', dimensions: {} }, score: 0.5, semantics: ACCURACY },
    ],
    quality_ratio: 0.5,
    ...over,
  } as ReportSummary
}

function perfRun(over: Partial<PerfRunSummary> = {}): PerfRunSummary {
  return {
    path: 'perf/run',
    model: 'qwen-plus',
    api_type: 'openai',
    dataset: 'openqa',
    num_runs: 1,
    total_requests: 10,
    success_rate: 1,
    best_rps: 0.12,
    best_latency: 1.2,
    is_embedding: false,
    has_html: false,
    timestamp: '2026-08-07T09:00:00',
    ...over,
  } as PerfRunSummary
}

describe('computeCellStats', () => {
  it('describes a history in the metric native scale', () => {
    const stats = computeCellStats([point('t1', 0.5), point('t2', 1), point('t3', 0.5)], ACCURACY)

    expect(stats).not.toBeNull()
    expect(stats!.latest).toBe(0.5)
    expect(stats!.min).toBe(0.5)
    expect(stats!.max).toBe(1)
    expect(stats!.spread).toBeCloseTo(0.5)
    expect(stats!.runs).toBe(3)
    // Mean of 0.5, 1, 0.5 — reported as-is, not rescaled to a percentage.
    expect(stats!.mean).toBeCloseTo(2 / 3)
  })

  it('normalizes spread against the declared range', () => {
    // A full-range swing on a [0, 1] metric is a relative spread of 1.
    const full = computeCellStats([point('t1', 0), point('t2', 1)], ACCURACY)
    expect(full!.relativeSpread).toBeCloseTo(1)

    const half = computeCellStats([point('t1', 0.5), point('t2', 1)], ACCURACY)
    expect(half!.relativeSpread).toBeCloseTo(0.5)
  })

  it('reports no relative spread for an unbounded metric', () => {
    // Dividing by the mean would make a near-zero throughput look infinitely unstable, and there
    // is no declared ceiling to divide by, so the honest answer is "not on that scale".
    const stats = computeCellStats([point('t1', 0.12), point('t2', 0.23)], RPS)

    expect(stats!.spread).toBeCloseTo(0.11)
    expect(stats!.relativeSpread).toBeNull()
  })

  it('gives a single run zero spread but does not call it stable', () => {
    const stats = computeCellStats([point('t1', 0.6)], ACCURACY)

    expect(stats!.runs).toBe(1)
    expect(stats!.spread).toBe(0)
    expect(stats!.stddev).toBe(0)
  })

  it('returns null for an empty history', () => {
    expect(computeCellStats([], ACCURACY)).toBeNull()
  })

  it('flags a series holding values from outside the declared range', () => {
    // Real values from omni_doc_bench_v1_6's `overall`, which is declared on [0, 1]: two runs are on
    // a 0-100 scale and one is on 0-1, because the benchmark changed what it emits. The spread is
    // then 95.81, which as percentage points would read "9581 pp" -- an instability no metric bounded
    // by [0, 1] can have, so it would look like a rendering fault instead of a data problem.
    const stats = computeCellStats(
      [point('t1', 96.7749), point('t2', 96.5051), point('t3', 0.9651)],
      ACCURACY,
    )

    expect(stats!.outOfRange).toBe(true)
    // Faithful arithmetic is kept; only its interpretation is withheld.
    expect(stats!.spread).toBeCloseTo(95.8098, 3)
    // Capped, so the cell still sorts among the widest instead of reporting an impossible ratio.
    expect(stats!.relativeSpread).toBe(1)
  })

  it('does not flag a series that stays within its range', () => {
    const stats = computeCellStats([point('t1', 0), point('t2', 1)], ACCURACY)

    expect(stats!.outOfRange).toBe(false)
    expect(stats!.relativeSpread).toBe(1)
  })

  it('cannot flag a metric that declares no range', () => {
    // Without a declared range there is nothing to be outside of.
    const stats = computeCellStats([point('t1', 0.12), point('t2', 990)], RPS)

    expect(stats!.outOfRange).toBe(false)
    expect(stats!.relativeSpread).toBeNull()
  })
})

describe('aggregateRuns', () => {
  it('collapses repeated runs of one benchmark into a single cell', () => {
    const rows = aggregateRuns(
      [
        report({ name: 'a', timestamp: '2026-08-07T08:00:00' }),
        report({ name: 'b', timestamp: '2026-08-07T09:00:00' }),
        report({ name: 'c', timestamp: '2026-08-07T10:00:00' }),
      ],
      [],
    )

    expect(rows).toHaveLength(1)
    expect(rows[0].stats.runs).toBe(3)
    // Oldest first, so a sparkline reads left to right as time.
    expect(rows[0].cell.history.map((p) => p.runId)).toEqual(['a', 'b', 'c'])
  })

  it('keeps semantics from the newest run when input is out of order', () => {
    const newestSemantics = { ...ACCURACY, display_precision: 3 }
    const rows = aggregateRuns(
      [
        report({ name: 'old', timestamp: '2026-08-07T08:00:00' }),
        report({
          name: 'new',
          timestamp: '2026-08-07T10:00:00',
          primary_metrics: [{
            dataset_name: 'iquiz',
            identity: { name: 'accuracy', aggregation: 'mean', dimensions: {} },
            score: 0.7,
            semantics: newestSemantics,
          }],
        }),
        report({ name: 'middle', timestamp: '2026-08-07T09:00:00' }),
      ],
      [],
    )

    expect(rows[0].cell.semantics?.display_precision).toBe(3)
  })

  it('splits a multi-dataset run into one cell per dataset', () => {
    const rows = aggregateRuns(
      [
        report({
          primary_metrics: [
            { dataset_name: 'general_mcq', identity: { name: 'accuracy', aggregation: 'mean', dimensions: {} }, score: 1, semantics: ACCURACY },
            { dataset_name: 'iquiz', identity: { name: 'accuracy', aggregation: 'mean', dimensions: {} }, score: 0.5, semantics: ACCURACY },
          ],
        }),
      ],
      [],
    )

    expect(rows.map((r) => r.cell.benchmark).sort()).toEqual(['general_mcq', 'iquiz'])
    expect(rows.every((r) => r.stats.runs === 1)).toBe(true)
  })

  it('keeps eval and perf cells separate and labels perf from its semantics', () => {
    const rows = aggregateRuns([report()], [perfRun(), perfRun({ best_rps: 0.23, timestamp: '2026-08-07T09:30:00' })], {
      best_rps: RPS,
    })

    const perf = rows.find((r) => r.cell.kind === 'perf')
    expect(perf).toBeDefined()
    expect(perf!.cell.metricName).toBe('Best RPS')
    expect(perf!.stats.runs).toBe(2)
    expect(perf!.stats.latest).toBeCloseTo(0.23)
    expect(rows.filter((r) => r.cell.kind === 'eval')).toHaveLength(1)
  })

  it('skips runs without a primary metric rather than plotting a gap', () => {
    const rows = aggregateRuns(
      [report({ primary_metrics: [] })],
      [],
    )

    expect(rows).toHaveLength(0)
  })

  it('does not merge two metrics of one benchmark into one cell', () => {
    // Precision and recall share a benchmark but not a scale; merging them would make the spread
    // meaningless.
    const rows = aggregateRuns(
      [
        report({
          primary_metrics: [
            { dataset_name: 'conll2003', identity: { name: 'precision', aggregation: 'mean', dimensions: {} }, score: 0.88, semantics: ACCURACY },
            { dataset_name: 'conll2003', identity: { name: 'recall', aggregation: 'mean', dimensions: {} }, score: 0.93, semantics: ACCURACY },
          ],
        }),
      ],
      [],
    )

    expect(rows).toHaveLength(2)
    expect(new Set(rows.map((r) => cellKey(r.cell))).size).toBe(2)
  })
})

describe('compareByInstability', () => {
  function row(relativeSpread: number | null, runs = 2, model = 'm', benchmark = 'b'): AggregatedRow {
    return {
      cell: { kind: 'eval', model, benchmark, metricName: 'mean_acc', semantics: ACCURACY, history: [] },
      stats: { latest: 0, mean: 0, stddev: 0, min: 0, max: 0, spread: 0, outOfRange: false, relativeSpread, runs },
    }
  }

  it('puts the widest spread first', () => {
    const sorted = [row(0.2), row(1), row(0.5)].sort(compareByInstability)
    expect(sorted.map((r) => r.stats.relativeSpread)).toEqual([1, 0.5, 0.2])
  })

  it('places metrics with no comparable scale after those that have one', () => {
    // A spread in req/s is not a smaller or larger version of a spread in accuracy points, so an
    // unbounded metric is never interleaved with bounded ones.
    const sorted = [row(null), row(0.1), row(null, 5)].sort(compareByInstability)
    expect(sorted[0].stats.relativeSpread).toBe(0.1)
    // Among unrankable cells, the more-measured one comes first.
    expect(sorted[1].stats.runs).toBe(5)
  })

  it('is deterministic for equal spreads', () => {
    const sorted = [row(0.5, 2, 'z', 'a'), row(0.5, 2, 'a', 'b')].sort(compareByInstability)
    expect(sorted.map((r) => r.cell.model)).toEqual(['a', 'z'])
  })
})

describe('trendBounds', () => {
  it('prefers the declared range so a small wobble draws small', () => {
    // Without this, 50%-60% would be stretched to fill the whole box and read as a crisis.
    const bounds = trendBounds([0.5, 0.6], ACCURACY)

    expect(bounds).toEqual({ low: 0, high: 1, declared: true })
    expect(trendPosition(0.5, bounds)).toBeCloseTo(0.5)
    expect(trendPosition(0.6, bounds)).toBeCloseTo(0.6)
  })

  it('falls back to zero-to-max for an unbounded metric and says so', () => {
    const bounds = trendBounds([0.12, 0.23], RPS)

    expect(bounds.low).toBe(0)
    expect(bounds.high).toBeCloseTo(0.23)
    // `declared: false` is what lets the UI state that heights are relative to this series only.
    expect(bounds.declared).toBe(false)
    expect(trendPosition(0.23, bounds)).toBeCloseTo(1)
  })

  it('never produces a zero-width extent', () => {
    // A flat series would otherwise divide by zero and every bar would be NaN tall.
    const flat = trendBounds([0, 0], RPS)
    expect(flat.high).toBeGreaterThan(flat.low)
    expect(Number.isFinite(trendPosition(0, flat))).toBe(true)

    const empty = trendBounds([], RPS)
    expect(empty.high).toBeGreaterThan(empty.low)
  })

  it('positions by magnitude, not by quality', () => {
    // A lower-is-better metric is not inverted here: a low error rate sits low, and the colour
    // scale is what reports that being low is good.
    const wer: MetricSemantics = { ...ACCURACY, metric_name: 'WER', direction: 'lower_is_better' }
    const bounds = trendBounds([0.043], wer)

    expect(trendPosition(0.043, bounds)).toBeCloseTo(0.043)
  })

  it('abandons the declared range when the series does not fit inside it', () => {
    // Real omni_doc_bench_v1_6 values: two runs on a 0-100 scale, one on 0-1. Clamping all three to
    // [0, 1] would draw three identical full-height bars and hide the whole point.
    const bounds = trendBounds([96.7749, 96.5051, 0.9651], ACCURACY)

    expect(bounds.declared).toBe(false)
    expect(bounds.high).toBeCloseTo(96.7749)
    // The odd run out is now visibly near the floor rather than pinned to the ceiling.
    expect(trendPosition(0.9651, bounds)).toBeLessThan(0.02)
    expect(trendPosition(96.7749, bounds)).toBeCloseTo(1)
  })

  it('clamps a position to the extent it was given', () => {
    const bounds = trendBounds([0.5], ACCURACY)

    expect(trendPosition(1.5, bounds)).toBe(1)
    expect(trendPosition(-0.2, bounds)).toBe(0)
  })
})
