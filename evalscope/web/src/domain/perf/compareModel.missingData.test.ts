// Feature: frontend-refactor-2026-07, Property 20: 缺数据度量去强调且保留可算度量
//
// For any pair of baseline/candidate metric sets, buildCompareModel must mark a
// metric's MetricDelta verdict as 'incomputable' whenever the metric is missing
// on either side — either because the row is absent from that run's summary_rows
// or because its cell is non-numeric (toNumeric → null). At the same time, every
// metric that carries a valid numeric value on BOTH sides must still produce a
// computable delta: its verdict is never 'incomputable' and its absoluteDelta is
// a fully-formed (non-missing) FormattedMetric. This "de-emphasize the missing,
// keep the computable" behaviour holds regardless of metric names, values, run
// paths or timestamps.
//
// Validates: Requirements 9.14

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import type { PerfDetailResponse } from '../../api/types'
import { buildCompareModel } from './compareModel'

/** Non-numeric cell tokens: toNumeric() returns null for each (missing value). */
const NON_NUMERIC_TOKENS = ['N/A', '-', 'error', 'null', '']

/**
 * Per-metric scenario describing how a metric appears across the two runs.
 * Every scenario except `bothValid` leaves the metric missing on one side, so
 * its delta must be incomputable.
 */
type Scenario =
  | 'bothValid' // valid numeric on both sides → computable
  | 'missingBaseline' // row absent from baseline run → incomputable
  | 'missingCandidate' // row absent from candidate run → incomputable
  | 'nonNumericBaseline' // baseline cell non-numeric → incomputable
  | 'nonNumericCandidate' // candidate cell non-numeric → incomputable

const MISSING_SCENARIOS: Scenario[] = [
  'missingBaseline',
  'missingCandidate',
  'nonNumericBaseline',
  'nonNumericCandidate',
]

/** A single generated metric case, plus its expected computability. */
interface MetricCase {
  name: string
  scenario: Scenario
  baselineValue: number
  candidateValue: number
  nonNumericToken: string
}

// Fixed, distinct run identities. The baseline run is strictly older so the
// default baseline selection resolves to `baselineRun` deterministically.
const BASELINE_TIMESTAMP = '2020-01-01T00:00:00.000Z'
const CANDIDATE_TIMESTAMP = '2021-01-01T00:00:00.000Z'
const BASELINE_PATH = 'perf/baseline'
const CANDIDATE_PATH = 'perf/candidate'

// Realistic perf metric names spanning both directions and boundedness so spec
// resolution is exercised while the property is checked.
const METRIC_NAME_POOL = [
  'Average latency (s)',
  'Average TTFT (s)',
  'Average TPOT (s)',
  'Output throughput (tokens/s)',
  'Request throughput (req/s)',
  'Success rate',
  'P99 latency (s)',
  'Total tokens',
]

/** Positive, non-zero values so a computable percent delta is always defined. */
const metricValueArb: fc.Arbitrary<number> = fc.double({
  min: 1,
  max: 100000,
  noNaN: true,
  noDefaultInfinity: true,
})

/** Build a metric case for a given (unique) name with a randomly chosen scenario. */
function metricCaseArb(name: string, scenario: fc.Arbitrary<Scenario>): fc.Arbitrary<MetricCase> {
  return fc.record({
    name: fc.constant(name),
    scenario,
    baselineValue: metricValueArb,
    candidateValue: metricValueArb,
    nonNumericToken: fc.constantFrom(...NON_NUMERIC_TOKENS),
  })
}

/**
 * Generate a set of metric cases that always includes at least one `bothValid`
 * metric (a computable delta to retain) and at least one missing-data metric (an
 * incomputable delta to de-emphasize), plus a random mix in between. Names stay
 * unique so each metric yields exactly one delta.
 */
const metricCasesArb: fc.Arbitrary<MetricCase[]> = fc
  .uniqueArray(fc.constantFrom(...METRIC_NAME_POOL), {
    minLength: 2,
    maxLength: METRIC_NAME_POOL.length,
  })
  .chain((names) =>
    fc.tuple(
      // First name: guaranteed computable metric.
      metricCaseArb(names[0], fc.constant<Scenario>('bothValid')),
      // Second name: guaranteed missing-data metric.
      metricCaseArb(names[1], fc.constantFrom(...MISSING_SCENARIOS)),
      // Remaining names: any scenario.
      fc.tuple(
        ...names.slice(2).map((name) =>
          metricCaseArb(
            name,
            fc.constantFrom<Scenario>('bothValid', ...MISSING_SCENARIOS),
          ),
        ),
      ),
    ),
  )
  .map(([both, missing, rest]) => [both, missing, ...rest])

/** The baseline-side cell for a metric case, or undefined when the row is absent. */
function baselineCell(mc: MetricCase): string | number | undefined {
  if (mc.scenario === 'missingBaseline') return undefined
  if (mc.scenario === 'nonNumericBaseline') return mc.nonNumericToken
  return mc.baselineValue
}

/** The candidate-side cell for a metric case, or undefined when the row is absent. */
function candidateCell(mc: MetricCase): string | number | undefined {
  if (mc.scenario === 'missingCandidate') return undefined
  if (mc.scenario === 'nonNumericCandidate') return mc.nonNumericToken
  return mc.candidateValue
}

/** Assemble summary rows for one side from the metric cases, dropping absent rows. */
function toSummaryRows(
  cases: MetricCase[],
  pick: (mc: MetricCase) => string | number | undefined,
): (string | number)[][] {
  const rows: (string | number)[][] = []
  for (const mc of cases) {
    const cell = pick(mc)
    if (cell === undefined) continue
    rows.push([mc.name, cell])
  }
  return rows
}

/** Build a PerfDetailResponse with a fixed identity and the given summary rows. */
function makeRun(
  path: string,
  generatedAt: string,
  summaryRows: (string | number)[][],
): PerfDetailResponse {
  return {
    path,
    model: 'model-a',
    api_type: 'openai_api',
    dataset: 'openqa',
    generated_at: generatedAt,
    basic_info: { 'Total requests': '100' },
    summary_columns: ['Metric', 'Value'],
    summary_rows: summaryRows,
    best_config: {},
    recommendations: [],
    num_runs: 1,
    is_embedding: false,
    has_html: true,
  }
}

describe('buildCompareModel — missing-data de-emphasis (Property 20: 缺数据度量去强调且保留可算度量)', () => {
  it('marks metrics missing on either side incomputable while keeping both-sided metrics computable', () => {
    fc.assert(
      fc.property(metricCasesArb, (cases) => {
        const baselineRun = makeRun(
          BASELINE_PATH,
          BASELINE_TIMESTAMP,
          toSummaryRows(cases, baselineCell),
        )
        const candidateRun = makeRun(
          CANDIDATE_PATH,
          CANDIDATE_TIMESTAMP,
          toSummaryRows(cases, candidateCell),
        )

        const model = buildCompareModel([baselineRun, candidateRun], '')

        // The older run is the baseline; the newer run is the candidate.
        expect(model.baselineId).toBe(BASELINE_PATH)
        expect(model.candidateId).toBe(CANDIDATE_PATH)

        const deltaByKey = new Map(model.deltas.map((delta) => [delta.metricKey, delta]))

        for (const mc of cases) {
          const delta = deltaByKey.get(mc.name)
          expect(delta).toBeDefined()
          if (!delta) continue

          if (mc.scenario === 'bothValid') {
            // Computable metric retained: verdict is a real direction and the
            // absolute delta is a fully-formed (non-missing) value.
            expect(delta.verdict).not.toBe('incomputable')
            expect(delta.absoluteDelta.isMissing).toBe(false)
            expect(delta.percentDelta.isMissing).toBe(false)
            expect(delta.baseline.isMissing).toBe(false)
            expect(delta.candidate.isMissing).toBe(false)
          } else {
            // Missing on one side → de-emphasized incomputable delta.
            expect(delta.verdict).toBe('incomputable')
            expect(delta.absoluteDelta.isMissing).toBe(true)
            expect(delta.percentDelta.isMissing).toBe(true)
          }
        }
      }),
    )
  })
})
