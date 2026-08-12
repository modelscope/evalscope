import { describe, expect, it } from 'vitest'

import type { PerfDetailResponse } from '../../api/types'
import type { MetricSemantics } from '../metric'
import { buildCompareModel, classifySampleSize } from './deltaModel'

const LATENCY: MetricSemantics = {
  semantic_id: 'perf.latency.seconds',
  metric_name: 'Latency',
  role: 'auxiliary',
  direction: 'lower_is_better',
  display_kind: 'number',
  display_unit: 's',
  display_precision: 2,
  contract_version: 1,
}

const THROUGHPUT: MetricSemantics = {
  semantic_id: 'perf.throughput.requests_per_second',
  metric_name: 'Request Throughput',
  role: 'auxiliary',
  direction: 'higher_is_better',
  display_kind: 'number',
  display_unit: 'req/s',
  display_precision: 2,
  contract_version: 1,
}

const SUCCESS: MetricSemantics = {
  semantic_id: 'quality.score.points_100',
  metric_name: 'Success Rate',
  role: 'auxiliary',
  direction: 'higher_is_better',
  value_range: { min: 0, max: 100 },
  display_kind: 'percent',
  display_multiplier: 1,
  display_unit: '%',
  display_precision: 1,
  contract_version: 1,
}

type SummaryColumn = PerfDetailResponse['summary_columns'][number]
type SummaryRow = PerfDetailResponse['summary_rows'][number]

const configColumn = (key: string, label: string): SummaryColumn => ({ key, label, semantics: null })
const metricColumn = (key: string, label: string, semantics: MetricSemantics): SummaryColumn => ({
  key,
  label,
  semantics,
})

const DEFAULT_COLUMNS = [
  configColumn('concurrency', 'Conc.'),
  configColumn('request_rate', 'Rate'),
  metricColumn('request_throughput', 'RPS', THROUGHPUT),
  metricColumn('avg_latency', 'Avg Lat.(s)', LATENCY),
  metricColumn('p99_latency', 'P99 Lat.(s)', LATENCY),
  metricColumn('success_rate', 'Success Rate', SUCCESS),
]

function makeRow(values: (string | number)[], sampleCount: number, columns: SummaryColumn[] = DEFAULT_COLUMNS): SummaryRow {
  const numericValues = Object.fromEntries(columns.map((column, index) => [column.key, values[index] === 'INF' ? -1 : values[index]]))
  const metricKeys = [...columns.filter((column) => column.semantics).map((column) => column.key), 'success_rate']
  const sampleCounts = Object.fromEntries(metricKeys.map((key) => [key, sampleCount]))
  return { values: numericValues as Record<string, number>, sample_counts: sampleCounts }
}

function makeRun(
  path: string,
  generatedAt: string,
  row: (string | number)[],
  overrides: Partial<PerfDetailResponse> = {},
): PerfDetailResponse {
  const columns = overrides.summary_columns ?? DEFAULT_COLUMNS
  return {
    path,
    model: 'model',
    api_type: 'openai_api',
    dataset: 'openqa',
    generated_at: generatedAt,
    basic_info: {},
    summary_columns: columns,
    summary_rows: [makeRow(row, 100, columns)],
    total_requests: 100,
    best_config: {},
    recommendations: [],
    num_runs: 1,
    is_embedding: false,
    has_html: false,
    ...overrides,
  }
}

describe('buildCompareModel', () => {
  it('selects the oldest run as the default baseline', () => {
    const newest = makeRun('new', '2026-06-02T00:00:00Z', ['8', 'INF', 11, 0.9, 1.1, 100])
    const oldest = makeRun('old', '2026-06-01T00:00:00Z', ['8', 'INF', 10, 1, 1.2, 100])

    expect(buildCompareModel([newest, oldest], '').baselineId).toBe('old')
  })

  it('uses metric semantics for direction and formatting', () => {
    const baseline = makeRun('old', '2026-06-01T00:00:00Z', ['8', 'INF', 10, 1, 1.2, 100])
    const candidate = makeRun('new', '2026-06-02T00:00:00Z', ['8', 'INF', 12, 0.8, 1.1, 90])

    const model = buildCompareModel([baseline, candidate], 'old')

    expect(model.deltas.find((delta) => delta.metricKey === 'request_throughput')?.verdict).toBe('improvement')
    expect(model.deltas.find((delta) => delta.metricKey === 'avg_latency')?.verdict).toBe('improvement')
    expect(model.deltas.find((delta) => delta.metricKey === 'success_rate')?.verdict).toBe('regression')
    expect(model.deltas.find((delta) => delta.metricKey === 'request_throughput')?.baseline.primary).toBe(
      '10 req/s',
    )
  })

  it('matches metrics by stable key when display labels change', () => {
    const baseline = makeRun('old', '2026-06-01T00:00:00Z', ['8', 'INF', 10, 1, 1.2, 100])
    const renamed = DEFAULT_COLUMNS.map((column) =>
      column.key === 'avg_latency' ? { ...column, label: 'Mean response time' } : column,
    )
    const candidate = makeRun('new', '2026-06-02T00:00:00Z', ['8', 'INF', 10, 0.8, 1.2, 100], {
      summary_columns: renamed,
    })

    const latency = buildCompareModel([baseline, candidate], 'old').deltas.find(
      (delta) => delta.metricKey === 'avg_latency',
    )

    expect(latency?.verdict).toBe('improvement')
    expect(latency?.metricLabel).toBe('Mean response time')
  })

  it('compares only rows with the same stable workload configuration', () => {
    const baseline = makeRun('old', '2026-06-01T00:00:00Z', ['8', 'INF', 10, 1, 1.2, 100], {
      summary_rows: [
        makeRow(['4', 'INF', 5, 0.5, 0.7, 100], 100),
        makeRow(['8', 'INF', 10, 1, 1.2, 100], 100),
      ],
    })
    const candidate = makeRun('new', '2026-06-02T00:00:00Z', ['8', 'INF', 11, 0.9, 1.1, 100])

    const model = buildCompareModel([baseline, candidate], 'old')

    expect(model.workloadMismatch).toBe(false)
    expect(model.deltas.find((delta) => delta.metricKey === 'request_throughput')?.baseline.raw).toBe('10')

    const mismatch = makeRun('other', '2026-06-03T00:00:00Z', ['16', 'INF', 20, 2, 2.2, 100])
    expect(buildCompareModel([baseline, mismatch], 'old').workloadMismatch).toBe(true)
  })

  it('keeps missing metrics as incomputable and records explicit sample counts', () => {
    const baseline = makeRun('old', '2026-06-01T00:00:00Z', ['8', 'INF', 10, 1], {
      summary_columns: DEFAULT_COLUMNS.slice(0, 4),
      summary_rows: [makeRow(['8', 'INF', 10, 1], 20, DEFAULT_COLUMNS.slice(0, 4))],
      total_requests: 20,
    })
    const candidate = makeRun('new', '2026-06-02T00:00:00Z', ['8', 'INF', 12], {
      summary_columns: DEFAULT_COLUMNS.slice(0, 3),
      summary_rows: [makeRow(['8', 'INF', 12], 30, DEFAULT_COLUMNS.slice(0, 3))],
      total_requests: 30,
    })

    const model = buildCompareModel([baseline, candidate], 'old')

    expect(model.sampleCounts).toEqual({ old: 20, new: 30 })
    expect(model.deltas.find((delta) => delta.metricKey === 'avg_latency')?.verdict).toBe('incomputable')
    expect(model.deltas.find((delta) => delta.metricKey === 'request_throughput')?.verdict).not.toBe(
      'incomputable',
    )
  })

  it('reports configuration differences using labels but compares stable keys', () => {
    const baseline = makeRun('old', '2026-06-01T00:00:00Z', ['8', 'INF', 10, 1, 1.2, 100])
    const candidate = makeRun('new', '2026-06-02T00:00:00Z', ['16', 'INF', 12, 0.8, 1.1, 100], {
      total_requests: 120,
    })

    const model = buildCompareModel([baseline, candidate], 'old')

    expect(model.configDiff).toEqual([
      { key: 'Conc.', baseline: '8', candidate: '16' },
      { key: 'Number of requests', baseline: '100', candidate: '120' },
    ])
  })

  it('uses request counts from the matched summary rows', () => {
    const baseline = makeRun('old', '2026-06-01T00:00:00Z', ['8', 'INF', 10, 1, 1.2, 100], {
      summary_rows: [
        makeRow(['4', 'INF', 5, 0.5, 0.7, 100], 200),
        makeRow(['8', 'INF', 10, 1, 1.2, 100], 20),
      ],
      total_requests: 220,
    })
    const candidate = makeRun('new', '2026-06-02T00:00:00Z', ['8', 'INF', 12, 0.8, 1.1, 100], {
      summary_rows: [makeRow(['8', 'INF', 12, 0.8, 1.1, 100], 25)],
      total_requests: 225,
    })

    const model = buildCompareModel([baseline, candidate], 'old')

    expect(model.sampleCounts).toEqual({ old: 20, new: 25 })
    expect(model.configDiff).toContainEqual({ key: 'Number of requests', baseline: '20', candidate: '25' })
  })
})

describe('classifySampleSize', () => {
  it.each([
    [0, 'critical'],
    [29, 'critical'],
    [30, 'warn'],
    [99, 'warn'],
    [100, 'ok'],
  ] as const)('classifies %s as %s', (count, expected) => {
    expect(classifySampleSize(count)).toBe(expected)
  })
})
