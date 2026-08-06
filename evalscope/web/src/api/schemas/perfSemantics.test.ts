/**
 * The perf schemas must declare `metric_semantics`.
 *
 * Feature: metric-semantics-governance. This is a regression test for a silent failure mode: zod's
 * `z.object` strips keys a schema does not mention, so an undeclared `metric_semantics` is dropped
 * during parsing. Nothing throws — the perf UI simply loses every direction and unit and falls
 * back to raw numbers, which looks plausible and is wrong.
 */

import { describe, expect, it } from 'vitest'

import { listPerfRunsResponseSchema, perfDetailResponseSchema } from './perf.schema'
import type { MetricSemantics } from '@/domain/metric'

const LATENCY: MetricSemantics = {
  semantic_id: 'perf.latency.seconds',
  metric_name: 'Avg Latency',
  role: 'primary',
  direction: 'lower_is_better',
  display_kind: 'number',
  display_unit: 's',
  display_precision: 3,
  contract_version: 1,
}

describe('perf schemas keep the semantics map', () => {
  it('preserves metric_semantics on the run list', () => {
    const parsed = listPerfRunsResponseSchema.parse({
      runs: [],
      total: 0,
      metric_semantics: { best_latency: LATENCY },
    })

    expect(parsed.metric_semantics?.best_latency).toEqual(LATENCY)
  })

  it('preserves metric_semantics on the run detail, keyed by the summary label', () => {
    const parsed = perfDetailResponseSchema.parse({
      path: 'runs/a',
      model: 'm',
      api_type: 'openai_api',
      dataset: 'openqa',
      generated_at: '2026-06-01T00:00:00Z',
      basic_info: {},
      summary_columns: ['Conc.', 'Avg Lat.(s)'],
      summary_rows: [['8', 1.2]],
      best_config: {},
      recommendations: [],
      num_runs: 1,
      is_embedding: false,
      has_html: false,
      // The backend keys this by the label it returned, not by a canonical name.
      metric_semantics: { 'Avg Lat.(s)': LATENCY },
    })

    expect(parsed.metric_semantics?.['Avg Lat.(s)']).toEqual(LATENCY)
  })

  it('still parses a response from a backend that sends no semantics', () => {
    const parsed = listPerfRunsResponseSchema.parse({ runs: [], total: 0 })

    expect(parsed.metric_semantics).toBeUndefined()
  })
})
