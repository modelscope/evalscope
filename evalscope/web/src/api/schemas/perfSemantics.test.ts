/** Ensure Zod preserves semantics at both wire locations: the run-list map and detail columns. */

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

describe('perf schemas keep metric semantics next to stable fields', () => {
  it('preserves metric_semantics on the run list', () => {
    const parsed = listPerfRunsResponseSchema.parse({
      runs: [],
      total: 0,
      metric_semantics: { best_latency: LATENCY },
    })

    expect(parsed.metric_semantics?.best_latency).toEqual(LATENCY)
  })

  it('preserves structured columns on the run detail', () => {
    const parsed = perfDetailResponseSchema.parse({
      path: 'runs/a',
      model: 'm',
      api_type: 'openai_api',
      dataset: 'openqa',
      generated_at: '2026-06-01T00:00:00Z',
      basic_info: {},
      summary_columns: [
        { key: 'concurrency', label: 'Conc.', semantics: null },
        { key: 'avg_latency', label: 'Avg Lat.(s)', semantics: LATENCY },
      ],
      summary_rows: [['8', 1.2]],
      summary_sample_counts: [1],
      total_requests: 1,
      best_config: {},
      recommendations: [],
      num_runs: 1,
      is_embedding: false,
      has_html: false,
    })

    expect(parsed.summary_columns[1]).toEqual({
      key: 'avg_latency',
      label: 'Avg Lat.(s)',
      semantics: LATENCY,
    })
  })

  it('still parses a response from a backend that sends no semantics', () => {
    const parsed = listPerfRunsResponseSchema.parse({ runs: [], total: 0 })

    expect(parsed.metric_semantics).toBeUndefined()
  })
})
