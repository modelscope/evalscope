// Feature: frontend-refactor-2026-07, Property 17: default baseline is the oldest run
//
// For any set of runs participating in a comparison, when no baseline is
// explicitly selected (empty `baselineId`), the default baseline chosen by
// `buildCompareModel` must be the oldest run: its `generated_at` timestamp is
// not later than the timestamp of any other run in the set.
//
// Validates: Requirements 9.2

import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import type { PerfDetailResponse } from '../../api/types'
import { buildCompareModel } from './compareModel'

/** Build a minimal `PerfDetailResponse` carrying just the fields the model reads. */
function makeRun(
  path: string,
  generatedAt: string,
  summaryRows: (string | number)[][],
): PerfDetailResponse {
  return {
    path,
    model: 'test-model',
    api_type: 'openai_api',
    dataset: 'shared-workload',
    generated_at: generatedAt,
    basic_info: {},
    summary_columns: ['Metric', 'Value'],
    summary_rows: summaryRows,
    best_config: {},
    recommendations: [],
    num_runs: 1,
    is_embedding: false,
    has_html: false,
  }
}

/** A handful of random summary rows so runs differ in their metric payloads. */
const summaryRowsArb = fc.array(
  fc.tuple(
    fc.string({ minLength: 1, maxLength: 12 }),
    fc.double({ min: -1e6, max: 1e6, noNaN: true, noDefaultInfinity: true }),
  ),
  { minLength: 0, maxLength: 5 },
)

/**
 * Epoch-millis range that stays inside the safe `Date` window, so
 * `new Date(ms).toISOString()` always yields a parseable ISO-8601 timestamp.
 */
const epochMillisArb = fc.integer({ min: 0, max: 4102444800000 }) // 1970-01-01 .. 2100-01-01

/**
 * A comparison set of 2+ runs, each with a distinct `path` and a distinct
 * `generated_at` ISO timestamp (drawn from unique epoch-millis values) plus
 * random summary rows.
 */
const runsArb = fc
  .uniqueArray(epochMillisArb, { minLength: 2, maxLength: 8 })
  .chain((millis) =>
    fc.tuple(...millis.map(() => summaryRowsArb)).map((rowsPerRun) =>
      millis.map((ms, index) =>
        makeRun(`runs/run-${index}`, new Date(ms).toISOString(), rowsPerRun[index]),
      ),
    ),
  )

describe('buildCompareModel default baseline (Property 17: default baseline is the oldest run)', () => {
  it('selects the oldest run as the default baseline when none is specified', () => {
    fc.assert(
      fc.property(runsArb, (runs) => {
        // Empty baselineId ⇒ the model must fall back to the default (oldest) run.
        const model = buildCompareModel(runs, '')

        const chosen = runs.find((run) => run.path === model.baselineId)
        expect(chosen).toBeDefined()

        const chosenTime = Date.parse((chosen as PerfDetailResponse).generated_at)
        // The chosen baseline's timestamp is not later than any other run's.
        for (const run of runs) {
          expect(chosenTime).toBeLessThanOrEqual(Date.parse(run.generated_at))
        }
      }),
    )
  })
})
