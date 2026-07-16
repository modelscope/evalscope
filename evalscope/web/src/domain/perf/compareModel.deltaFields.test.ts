// Feature: frontend-refactor-2026-07, Property 15: Delta 汇总字段完整
//
// For any baseline and candidate metric set, buildCompareModel must produce a
// MetricDelta for every metric in the union of both runs' summary rows, and
// each MetricDelta must simultaneously carry all four summary fields — a
// baseline value, a candidate value, an absolute delta and a percent delta —
// each a fully-formed FormattedMetric (with `primary` and `raw` strings), plus
// a direction-aware `verdict`. This holds regardless of the metric names, the
// numeric values, the timestamps or the run paths.
//
// Validates: Requirements 9.1

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import type { PerfDetailResponse } from '../../api/types'
import type { FormattedMetric } from '../metric/metricFormat'
import type { DeltaVerdict, MetricDelta } from './compareModel'
import { buildCompareModel } from './compareModel'

/** The four verdicts buildCompareModel may emit. */
const VERDICTS: DeltaVerdict[] = ['improvement', 'regression', 'neutral', 'incomputable']

/**
 * Assert a value is a fully-formed FormattedMetric: an object exposing string
 * `primary` and `raw` fields (Property 15 requires these on every delta cell).
 */
function assertFormattedMetric(field: unknown): asserts field is FormattedMetric {
  if (typeof field !== 'object' || field === null) {
    throw new Error('Expected a formatted metric object')
  }
  const metric = field as Record<string, unknown>
  if (typeof metric.primary !== 'string' || typeof metric.raw !== 'string') {
    throw new Error('Expected formatted metric primary/raw strings')
  }
}

/** Assert a single MetricDelta carries all four summary fields plus a verdict. */
function assertDeltaComplete(delta: MetricDelta): void {
  if (typeof delta.metricKey !== 'string') throw new Error('Expected a metric key')
  assertFormattedMetric(delta.baseline)
  assertFormattedMetric(delta.candidate)
  assertFormattedMetric(delta.absoluteDelta)
  assertFormattedMetric(delta.percentDelta)
  if (!VERDICTS.includes(delta.verdict)) throw new Error(`Unexpected verdict: ${delta.verdict}`)
}

// A pool of representative perf metric names spanning lower-is-better latency
// metrics, higher-is-better throughput metrics and a bounded success rate, so
// generated runs mix directions and boundedness.
const METRIC_NAME_POOL = [
  'Average latency (s)',
  'Average TTFT (s)',
  'Average TPOT (s)',
  'Output throughput (tokens/s)',
  'Request throughput (req/s)',
  'Success rate',
  'Number of requests',
  'Concurrency',
  'P99 latency (s)',
  'Total tokens',
]

/** A single summary row: `[metricName, value]`. */
const summaryRowArb: fc.Arbitrary<(string | number)[]> = fc.tuple(
  fc.constantFrom(...METRIC_NAME_POOL),
  fc.double({ min: 0, max: 100000, noNaN: true, noDefaultInfinity: true }),
)

/**
 * Generate a set of summary rows with at least one metric. Duplicate metric
 * names are collapsed by buildCompareModel (first occurrence wins), so the
 * generator does not need to dedupe.
 */
const summaryRowsArb: fc.Arbitrary<(string | number)[][]> = fc.array(summaryRowArb, {
  minLength: 1,
  maxLength: 8,
})

/** ISO-8601 timestamp generator spanning a range of distinct instants. */
const timestampArb: fc.Arbitrary<string> = fc
  .integer({ min: 0, max: 3_000_000_000 })
  .map((secs) => new Date(secs * 1000).toISOString())

/** Generate a full PerfDetailResponse with a random metric set and identity. */
function perfRunArb(index: number): fc.Arbitrary<PerfDetailResponse> {
  return fc.record({
    path: fc
      .array(fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz0123456789_-'), { minLength: 1, maxLength: 12 })
      .map((parts) => `perf/${parts.join('')}-${index}`),
    model: fc.constantFrom('model-a', 'model-b', 'model-c'),
    api_type: fc.constant('openai_api'),
    dataset: fc.constantFrom('openqa', 'longalpaca', 'sharegpt'),
    generated_at: timestampArb,
    basic_info: fc.constant<Record<string, string>>({ 'Total requests': '100' }),
    summary_columns: fc.constant(['Metric', 'Value']),
    summary_rows: summaryRowsArb,
    best_config: fc.dictionary(
      fc.constantFrom('Concurrency', 'Rate', 'Requests', 'Max tokens'),
      fc.oneof(fc.integer({ min: 0, max: 1000 }), fc.double({ min: 0, max: 1000, noNaN: true })).map(String),
      { maxKeys: 4 },
    ),
    recommendations: fc.constant<string[]>([]),
    num_runs: fc.constant(1),
    is_embedding: fc.constant(false),
    has_html: fc.constant(true),
  })
}

/** Generate 2+ runs, each with an independent path so ids stay distinct. */
const runsArb: fc.Arbitrary<PerfDetailResponse[]> = fc
  .integer({ min: 2, max: 5 })
  .chain((count) => fc.tuple(...Array.from({ length: count }, (_, i) => perfRunArb(i))))

describe('buildCompareModel — delta field completeness (Property 15: Delta 汇总字段完整)', () => {
  it('extracts metrics and sample counts from the real archive wide-table shape', () => {
    const columns = [
      'Conc.', 'Rate', 'RPS', 'Avg Lat.(s)', 'P99 Lat.(s)', 'Avg TTFT(ms)',
      'P99 TTFT(ms)', 'Avg TPOT(ms)', 'P99 TPOT(ms)', 'Gen. tok/s', 'Success Rate',
    ]
    const makeWideRun = (
      path: string,
      generatedAt: string,
      requests: number,
      row: (string | number)[],
    ): PerfDetailResponse => ({
      path,
      model: 'qwen-plus',
      api_type: 'openai',
      dataset: 'openqa',
      generated_at: generatedAt,
      basic_info: { 'Total Requests': String(requests) },
      summary_columns: columns,
      summary_rows: [row],
      best_config: {},
      recommendations: [],
      num_runs: 1,
      is_embedding: false,
      has_html: true,
    })
    const baseline = makeWideRun(
      'run-p1', '2026-07-15T15:37:19', 4,
      ['1', 'INF', '1.0326', '0.968', '1.080', '467.79', '597.30', '16.13', '17.10', '33.04', '100.0%'],
    )
    const candidate = makeWideRun(
      'run-p2', '2026-07-15T15:39:04', 6,
      ['2', 'INF', '1.9256', '1.027', '1.210', '493.17', '656.15', '17.21', '17.77', '61.62', '100.0%'],
    )

    const model = buildCompareModel([candidate, baseline], '')
    const rps = model.deltas.find((delta) => delta.metricKey === 'rps')
    const latency = model.deltas.find((delta) => delta.metricKey === 'latency')
    const success = model.deltas.find((delta) => delta.metricKey === 'success_rate')

    expect(model.sampleCounts).toEqual({ 'run-p1': 4, 'run-p2': 6 })
    expect(rps?.baseline.primary).toBe('1.03 req/s')
    expect(rps?.candidate.primary).toBe('1.93 req/s')
    expect(rps?.verdict).toBe('improvement')
    expect(latency?.verdict).toBe('regression')
    expect(success?.absoluteDelta.primary).toBe('0.0 pp')
    expect(model.configDiff).toEqual(expect.arrayContaining([
      { key: 'Concurrency', baseline: '1', candidate: '2' },
      { key: 'Number of requests', baseline: '4', candidate: '6' },
    ]))
    expect(model.workloadMismatch).toBe(true)
  })

  it('keeps every delta field complete for default and explicit baselines', () => {
    fc.assert(
      fc.property(runsArb, fc.nat(), (runs, baselineIndex) => {
        const baselineId = runs[baselineIndex % runs.length].path
        const models = [buildCompareModel(runs, ''), buildCompareModel(runs, baselineId)]

        for (const model of models) {
          if (model.deltas.length === 0) throw new Error('Expected at least one metric delta')
          for (const delta of model.deltas) {
            assertDeltaComplete(delta)
          }
        }
        expect(models[1].baselineId).toBe(baselineId)
      }),
      { numRuns: 40 },
    )
  })
})
