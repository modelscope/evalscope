import { describe, expect, it } from 'vitest'

import type { PerfDetailResponse } from '../../api/types'
import type { MetricSemantics } from '../metric'
import {
  buildCompareModel,
  buildMetricComparison,
  classifySampleSize,
  getFixedInputLength,
  getInputLengthEligibility,
} from './deltaModel'

const LATENCY: MetricSemantics = {
  semantic_id: 'perf.latency.seconds',
  metric_name: 'Latency',
  kind: 'quality',
  direction: 'lower_is_better',
  display_kind: 'number',
  display_unit: 's',
  display_precision: 2,
}

const THROUGHPUT: MetricSemantics = {
  semantic_id: 'perf.throughput.requests_per_second',
  metric_name: 'Request Throughput',
  kind: 'quality',
  direction: 'higher_is_better',
  display_kind: 'number',
  display_unit: 'req/s',
  display_precision: 2,
}

const SUCCESS: MetricSemantics = {
  semantic_id: 'quality.score.points_100',
  metric_name: 'Success Rate',
  kind: 'quality',
  direction: 'higher_is_better',
  value_range: { min: 0, max: 100 },
  display_kind: 'percent',
  display_multiplier: 1,
  display_unit: '%',
  display_precision: 1,
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

function makeRow(values: (string | number)[], count: number, columns: SummaryColumn[] = DEFAULT_COLUMNS): SummaryRow {
  const numericValues = Object.fromEntries(columns.map((column, index) => [
    column.key,
    values[index] === 'INF' ? -1 : values[index],
  ]))
  return {
    values: numericValues as Record<string, number>,
    sample_counts: Object.fromEntries(columns.filter((column) => column.semantics).map((column) => [column.key, count])),
  }
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

function randomRun(path: string, inputLength: number, concurrency = 1): PerfDetailResponse {
  return makeRun(path, '2026-06-01T00:00:00Z', [String(concurrency), 'INF', 10, 1, 1.2, 100], {
    dataset: 'random',
    configuration: {
      dataset: 'random',
      min_prompt_length: String(inputLength),
      max_prompt_length: String(inputLength),
      rate: '-1',
    },
  })
}

describe('buildCompareModel', () => {
  it('selects the oldest run by default and honours any explicit baseline', () => {
    const newest = makeRun('new', '2026-06-02T00:00:00Z', ['8', 'INF', 11, 0.9, 1.1, 100])
    const oldest = makeRun('old', '2026-06-01T00:00:00Z', ['8', 'INF', 10, 1, 1.2, 100])
    const middle = makeRun('middle', '2026-06-01T12:00:00Z', ['8', 'INF', 12, 0.8, 1, 100])

    expect(buildCompareModel([newest, oldest, middle], '').baselineId).toBe('old')
    expect(buildCompareModel([newest, oldest, middle], 'middle').baselineId).toBe('middle')
    expect(buildCompareModel([], '').baselineId).toBe('')
  })

  it('keeps all runs and aligns each workload without using input length as a group key', () => {
    const first = randomRun('a', 1024, 4)
    const second = randomRun('b', 2048, 4)
    const third = randomRun('c', 4096, 8)

    const model = buildCompareModel([first, second, third], 'a')

    expect(model.groups).toHaveLength(2)
    expect(model.groups[0].rows).toHaveLength(4)
    const latency = model.groups[0].rows.find((row) => row.metricKey === 'avg_latency')!
    expect(latency.values.a.formatted.primary).toBe('1 s')
    expect(latency.values.b.value).toBe(1)
    expect(latency.values.c.formatted.isMissing).toBe(true)
    expect(model.groups[1].rows[0].values.c.value).not.toBeNull()
  })

  it('keeps missing metrics and empty runs aligned with explicit sample counts', () => {
    const full = makeRun('a', '2026-06-01T00:00:00Z', ['8', 'INF', 10, 1, 1.2, 100])
    const throughputOnly = makeRun('b', '2026-06-02T00:00:00Z', ['8', 'INF', 12], {
      summary_columns: DEFAULT_COLUMNS.slice(0, 3),
      summary_rows: [makeRow(['8', 'INF', 12], 30, DEFAULT_COLUMNS.slice(0, 3))],
    })
    const empty = makeRun('c', '2026-06-03T00:00:00Z', ['8', 'INF', 13, 1.1, 1.3, 100], {
      summary_rows: [],
    })

    const group = buildCompareModel([full, throughputOnly, empty], 'a').groups[0]
    const latency = group.rows.find((row) => row.metricKey === 'avg_latency')!
    const throughput = group.rows.find((row) => row.metricKey === 'request_throughput')!

    expect(latency.values.a.sampleCount).toBe(100)
    expect(latency.values.b.formatted.isMissing).toBe(true)
    expect(latency.values.c.formatted.isMissing).toBe(true)
    expect(throughput.values.b.value).toBe(12)
    expect(throughput.values.b.sampleCount).toBe(30)
  })

  it('marks duplicate workloads as ambiguous instead of overwriting their values', () => {
    const run = makeRun('a', '2026-06-01T00:00:00Z', ['8', 'INF', 10, 1, 1.2, 100], {
      summary_rows: [
        makeRow(['8', 'INF', 10, 1, 1.2, 100], 100),
        makeRow(['8', 'INF', 20, 2, 2.2, 100], 100),
      ],
    })
    const other = makeRun('b', '2026-06-02T00:00:00Z', ['8', 'INF', 11, 0.9, 1.1, 100])

    const model = buildCompareModel([run, other], 'a')
    const latency = model.groups[0].rows.find((row) => row.metricKey === 'avg_latency')!

    expect(model.duplicateWorkloads).toBe(true)
    expect(latency.values.a.ambiguous).toBe(true)
    expect(latency.values.a.value).toBeNull()
    expect(latency.values.b.value).toBe(0.9)
  })

  it('reports configuration differences across every run', () => {
    const first = makeRun('a', '2026-06-01T00:00:00Z', ['8', 'INF', 10, 1, 1.2, 100], {
      configuration: { dataset: 'openqa', parallel: '8', number: '50' },
    })
    const second = makeRun('b', '2026-06-02T00:00:00Z', ['8', 'INF', 11, 0.9, 1.1, 100], {
      configuration: { dataset: 'openqa', parallel: '8', number: '60' },
      total_requests: 120,
    })
    const third = makeRun('c', '2026-06-03T00:00:00Z', ['8', 'INF', 12, 0.8, 1, 100], {
      configuration: { dataset: 'random', parallel: '8', number: '70' },
    })

    const diff = buildCompareModel([first, second, third], 'a').configDiff

    expect(diff).toEqual([
      { key: 'dataset', label: 'Dataset', values: { a: 'openqa', b: 'openqa', c: 'random' } },
      { key: 'number', label: 'Number of requests', values: { a: '50', b: '60', c: '70' } },
    ])
  })
})

describe('buildMetricComparison', () => {
  it('uses metric direction and keeps absolute values separate from deltas', () => {
    expect(buildMetricComparison(12, 10, THROUGHPUT).verdict).toBe('improvement')
    expect(buildMetricComparison(0.8, 1, LATENCY).verdict).toBe('improvement')
    expect(buildMetricComparison(90, 100, SUCCESS).verdict).toBe('regression')
    expect(buildMetricComparison(90, 100, SUCCESS).absoluteDelta.primary).toBe('-10 pp')
  })

  it('handles zero baselines and missing values without treating them as zero', () => {
    const zero = buildMetricComparison(2, 0, THROUGHPUT)
    expect(zero.absoluteDelta.primary).toBe('2 req/s')
    expect(zero.percentDelta.isMissing).toBe(true)
    expect(buildMetricComparison(null, 0, THROUGHPUT).verdict).toBe('incomputable')
  })
})

describe('input length eligibility', () => {
  it('accepts only fixed random lengths and returns actual closed-loop points', () => {
    const runs = [randomRun('a', 1024), randomRun('b', 2048), randomRun('c', 4096, 4)]
    const result = getInputLengthEligibility(runs)

    expect(result.eligible).toBe(true)
    expect(result.points).toEqual([
      { runPath: 'a', inputLength: 1024, concurrency: 1, requests: 100 },
      { runPath: 'b', inputLength: 2048, concurrency: 1, requests: 100 },
      { runPath: 'c', inputLength: 4096, concurrency: 4, requests: 100 },
    ])
    expect(getFixedInputLength(runs[0])).toBe(1024)
  })

  it('accepts a finite request rate when the report has a fixed concurrency limit', () => {
    const run = randomRun('rated-closed-loop', 1024, 8)
    const ratedRun = {
      ...run,
      configuration: { ...run.configuration, parallel: '8', rate: '10' },
      summary_rows: [makeRow(['8', 10, 10, 1, 1.2, 100], 100)],
    }

    const result = getInputLengthEligibility([ratedRun])

    expect(result.eligible).toBe(true)
    expect(result.points).toEqual([
      { runPath: 'rated-closed-loop', inputLength: 1024, concurrency: 8, requests: 100 },
    ])
  })

  it.each([
    ['missingConfiguration', { configuration: undefined }],
    ['notRandom', { configuration: { dataset: 'openqa', min_prompt_length: '1024', max_prompt_length: '1024', rate: '-1' } }],
    ['notRandom', { configuration: { dataset: 'random, openqa', min_prompt_length: '1024', max_prompt_length: '1024', rate: '-1' } }],
    ['variableLength', { configuration: { dataset: 'random', min_prompt_length: '1024', max_prompt_length: '2048', rate: '-1' } }],
    ['variableLength', { configuration: { dataset: 'random', min_prompt_length: '1024, 2048', max_prompt_length: '1024, 2048', rate: '-1' } }],
    ['openLoop', {
      configuration: { dataset: 'random', min_prompt_length: '1024', max_prompt_length: '1024', rate: '10' },
      summary_rows: [makeRow(['0', 10, 10, 1, 1.2, 100], 100)] as SummaryRow[],
    }],
    ['missingWorkload', { summary_rows: [] as SummaryRow[] }],
  ] as const)('rejects %s', (code, overrides) => {
    const run = randomRun('a', 1024)
    const result = getInputLengthEligibility([{ ...run, ...overrides }, randomRun('b', 2048)])
    expect(result.eligible).toBe(false)
    expect(result.issue?.code).toBe(code)
  })

  it('rejects duplicate input-length and concurrency combinations', () => {
    const result = getInputLengthEligibility([randomRun('a', 1024), randomRun('b', 1024)])
    expect(result.issue?.code).toBe('duplicatePoint')
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
