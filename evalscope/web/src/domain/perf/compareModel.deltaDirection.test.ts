// Feature: frontend-refactor-2026-07, Property 16: Delta 方向与度量 direction 一致
//
// For any baseline / candidate numeric pair and a metric whose `direction` is
// inferred from its label, the `MetricDelta.verdict` produced by
// `buildCompareModel` must follow the direction rule:
//   - higher-is-better: candidate > baseline → improvement,
//                       candidate < baseline → regression;
//   - lower-is-better:  candidate > baseline → regression,
//                       candidate < baseline → improvement;
//   - candidate === baseline → neutral (either direction).
// The verdict is an informational direction annotation and never a hard
// pass/fail gate.
//
// Validates: Requirements 9.4, 9.5

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import type { PerfDetailResponse } from '../../api/types'
import { buildCompareModel, type DeltaVerdict } from './compareModel'

/**
 * Metric label whose `resolvePerfMetricSpec` direction is `higher-is-better`
 * (no latency/ttft/tpot/time/delay/duration keyword).
 */
const HIGHER_IS_BETTER_METRIC = 'Output throughput (tokens/s)'
/** Metric label whose inferred direction is `lower-is-better` (contains "latency"). */
const LOWER_IS_BETTER_METRIC = 'Average latency (s)'

const BASELINE_PATH = 'runs/baseline'
const CANDIDATE_PATH = 'runs/candidate'

/** Build a minimal `PerfDetailResponse` carrying just the fields the model reads. */
function makeRun(
  path: string,
  generatedAt: string,
  metricValues: Record<string, number>,
): PerfDetailResponse {
  return {
    path,
    model: 'test-model',
    api_type: 'openai_api',
    dataset: 'shared-workload',
    generated_at: generatedAt,
    basic_info: {},
    summary_columns: ['Metric', 'Value'],
    summary_rows: Object.entries(metricValues).map(([key, value]) => [key, value]),
    best_config: {},
    recommendations: [],
    num_runs: 1,
    is_embedding: false,
    has_html: false,
  }
}

/** The direction rule under test (Req 9.4, 9.5), expressed independently of the SUT. */
function expectedVerdict(baseline: number, candidate: number, higherIsBetter: boolean): DeltaVerdict {
  if (candidate === baseline) return 'neutral'
  const candidateIsHigher = candidate > baseline
  return candidateIsHigher === higherIsBetter ? 'improvement' : 'regression'
}

/**
 * A baseline value plus a candidate that is deliberately equal to, greater
 * than, or less than the baseline. Constructing the candidate from the baseline
 * (rather than drawing two independent values) guarantees all three relations
 * are exercised, and building `>` / `<` cases with a strictly positive gap
 * keeps them from collapsing back into equality under floating point.
 */
const valuePairArb = fc
  .record({
    baseline: fc.double({ min: -1e6, max: 1e6, noNaN: true, noDefaultInfinity: true }),
    relation: fc.constantFrom<'equal' | 'greater' | 'less'>('equal', 'greater', 'less'),
    gap: fc.double({ min: 1e-3, max: 1e6, noNaN: true, noDefaultInfinity: true }),
  })
  .map(({ baseline, relation, gap }) => {
    if (relation === 'equal') return { baseline, candidate: baseline }
    if (relation === 'greater') return { baseline, candidate: baseline + gap }
    return { baseline, candidate: baseline - gap }
  })

/** Locate the delta for a metric key in the built model. */
function verdictFor(runs: PerfDetailResponse[], metricKey: string): DeltaVerdict {
  const model = buildCompareModel(runs, BASELINE_PATH)
  const delta = model.deltas.find((d) => d.metricKey === metricKey)
  if (!delta) throw new Error(`missing delta for ${metricKey}`)
  return delta.verdict
}

describe('buildCompareModel verdict direction (Property 16: Delta 方向与度量 direction 一致)', () => {
  it('follows higher-is-better for a throughput metric', () => {
    fc.assert(
      fc.property(valuePairArb, ({ baseline, candidate }) => {
        const runs = [
          makeRun(BASELINE_PATH, '2026-06-01T00:00:00.000Z', { [HIGHER_IS_BETTER_METRIC]: baseline }),
          makeRun(CANDIDATE_PATH, '2026-07-01T00:00:00.000Z', { [HIGHER_IS_BETTER_METRIC]: candidate }),
        ]
        expect(verdictFor(runs, HIGHER_IS_BETTER_METRIC)).toBe(
          expectedVerdict(baseline, candidate, true),
        )
      }),
    )
  })

  it('follows lower-is-better for a latency metric', () => {
    fc.assert(
      fc.property(valuePairArb, ({ baseline, candidate }) => {
        const runs = [
          makeRun(BASELINE_PATH, '2026-06-01T00:00:00.000Z', { [LOWER_IS_BETTER_METRIC]: baseline }),
          makeRun(CANDIDATE_PATH, '2026-07-01T00:00:00.000Z', { [LOWER_IS_BETTER_METRIC]: candidate }),
        ]
        expect(verdictFor(runs, LOWER_IS_BETTER_METRIC)).toBe(
          expectedVerdict(baseline, candidate, false),
        )
      }),
    )
  })

  it('yields neutral on equal values and opposite verdicts across directions on unequal values', () => {
    fc.assert(
      fc.property(valuePairArb, ({ baseline, candidate }) => {
        const runs = [
          makeRun(BASELINE_PATH, '2026-06-01T00:00:00.000Z', {
            [HIGHER_IS_BETTER_METRIC]: baseline,
            [LOWER_IS_BETTER_METRIC]: baseline,
          }),
          makeRun(CANDIDATE_PATH, '2026-07-01T00:00:00.000Z', {
            [HIGHER_IS_BETTER_METRIC]: candidate,
            [LOWER_IS_BETTER_METRIC]: candidate,
          }),
        ]
        const model = buildCompareModel(runs, BASELINE_PATH)
        const higher = model.deltas.find((d) => d.metricKey === HIGHER_IS_BETTER_METRIC)?.verdict
        const lower = model.deltas.find((d) => d.metricKey === LOWER_IS_BETTER_METRIC)?.verdict

        if (candidate === baseline) {
          // Equal values are neutral regardless of direction.
          expect(higher).toBe('neutral')
          expect(lower).toBe('neutral')
        } else {
          // Same values, opposite directions ⇒ opposite verdicts.
          expect(higher).not.toBe('neutral')
          expect(lower).not.toBe('neutral')
          expect(higher).not.toBe(lower)
        }
      }),
    )
  })
})
