import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'

import PerfComparePage from './PerfComparePage'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { ReportsProvider } from '@/contexts/ReportsContext'
import type { PerfDetailResponse } from '@/api/types'
import DataTableFallback from '@/components/common/DataTableFallback'
import type { DataTableModel } from '@/components/common/DataTableFallback'

vi.mock('@/api/perf', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/perf')>()
  return { ...actual, getPerfDetail: vi.fn() }
})

vi.mock('@/components/charts/PlotlyChart', () => ({
  default: ({ src, fallbackTable, title }: { src: string; fallbackTable: DataTableModel; title?: string }) => (
    <div data-testid={`chart-${title}`} data-src={src}>
      {title && <h4>{title}</h4>}
      <DataTableFallback model={fallbackTable} />
    </div>
  ),
}))

import { getPerfDetail } from '@/api/perf'

type PerfDetailOverrides = Partial<PerfDetailResponse> & { __sampleCount?: number }

function makePerfDetail(overrides: PerfDetailOverrides = {}): PerfDetailResponse {
  const sampleCount = overrides.__sampleCount ?? 200
  const latencySemantics = {
    semantic_id: 'perf.latency.seconds',
    metric_name: 'Latency',
    kind: 'quality' as const,
    direction: 'lower_is_better' as const,
    display_kind: 'number' as const,
    display_unit: 's',
    display_precision: 2,
  }
  const throughputSemantics = {
    semantic_id: 'perf.throughput.tokens_per_second',
    metric_name: 'Output Throughput',
    kind: 'quality' as const,
    direction: 'higher_is_better' as const,
    display_kind: 'number' as const,
    display_unit: 'tok/s',
    display_precision: 2,
  }
  const columns: PerfDetailResponse['summary_columns'] = [
    { key: 'concurrency', label: 'Concurrency', semantics: null },
    { key: 'request_rate', label: 'Rate', semantics: null },
    { key: 'avg_latency', label: 'Average latency (s)', semantics: latencySemantics },
    { key: 'p90_latency', label: 'P90 latency (s)', semantics: latencySemantics },
    { key: 'p95_latency', label: 'P95 latency (s)', semantics: latencySemantics },
    { key: 'p99_latency', label: 'P99 latency (s)', semantics: latencySemantics },
    { key: 'output_token_throughput', label: 'Output throughput (tokens/s)', semantics: throughputSemantics },
  ]
  const base: PerfDetailResponse = {
    path: 'run-a',
    model: 'model-a',
    api_type: 'openai_api',
    dataset: 'openqa',
    generated_at: '2026-06-01T00:00:00Z',
    basic_info: { 'Total requests': String(sampleCount) },
    configuration: {
      dataset: 'openqa',
      parallel: '1',
      number: String(sampleCount),
      rate: '-1',
      min_prompt_length: '1024',
      max_prompt_length: '1024',
    },
    summary_columns: columns,
    summary_rows: [{
      values: {
        concurrency: 1,
        request_rate: -1,
        avg_latency: 1.2,
        p90_latency: 2.0,
        p95_latency: 2.5,
        p99_latency: 3.0,
        output_token_throughput: 500,
      },
      sample_counts: Object.fromEntries(columns.filter((column) => column.semantics).map((column) => [column.key, sampleCount])),
    }],
    total_requests: sampleCount,
    best_config: { concurrency: '10' },
    recommendations: [],
    num_runs: 1,
    is_embedding: false,
    has_html: false,
  }
  const merged = { ...base, ...overrides }
  delete merged.__sampleCount
  return merged as PerfDetailResponse
}

function makeRandomDetail(path: string, inputLength: number): PerfDetailResponse {
  return makePerfDetail({
    path,
    model: `model-${path}`,
    dataset: 'random',
    configuration: {
      dataset: 'random',
      parallel: '1',
      number: '200',
      rate: '-1',
      min_prompt_length: String(inputLength),
      max_prompt_length: String(inputLength),
    },
  })
}

function mockDetails(map: Record<string, PerfDetailResponse>, failed: string[] = []) {
  vi.mocked(getPerfDetail).mockImplementation(async (_root: string, path: string) => {
    if (failed.includes(path)) throw new Error('missing run')
    const detail = map[path]
    if (!detail) throw new Error('unknown run')
    return detail
  })
}

function LocationDisplay() {
  const location = useLocation()
  return <div data-testid="location">{`${location.pathname}${location.search}`}</div>
}

function renderPage(entries: string[]) {
  return render(
    <ThemeProvider>
      <LocaleProvider>
        <ReportsProvider>
          <MemoryRouter initialEntries={entries}>
            <Routes>
              <Route path="*" element={<><PerfComparePage /><LocationDisplay /></>} />
            </Routes>
          </MemoryRouter>
        </ReportsProvider>
      </LocaleProvider>
    </ThemeProvider>,
  )
}

async function renderLoaded(entries: string[]) {
  const utils = renderPage(entries)
  await act(async () => {
    await vi.runAllTimersAsync()
  })
  return utils
}

function compareUrl(paths: string[], extra: Record<string, string> = {}): string {
  const params = new URLSearchParams({ root_path: './outputs', paths: paths.join(';'), ...extra })
  return `/perf-compare?${params.toString()}`
}

function currentLocation(): string {
  return screen.getByTestId('location').textContent ?? ''
}

beforeEach(() => {
  vi.mocked(getPerfDetail).mockReset()
})

afterEach(() => {
  cleanup()
})

describe('PerfComparePage', () => {
  it('omits generation timing charts when a compared run is a decision API', async () => {
    mockDetails({
      'run-a': makePerfDetail({ path: 'run-a', api_type: 'systemone' }),
      'run-b': makePerfDetail({ path: 'run-b', api_type: 'autotrust' }),
    })
    await renderLoaded([compareUrl(['run-a', 'run-b'])])
    expect(screen.queryByText('TTFT (ms)')).not.toBeInTheDocument()
    expect(screen.queryByText('TPOT (ms)')).not.toBeInTheDocument()
  })

  it('uses one absolute-value summary for two or more runs and preserves missing columns', async () => {
    mockDetails({
      'run-a': makePerfDetail({ path: 'run-a' }),
      'run-b': makePerfDetail({
        path: 'run-b',
        model: 'model-b',
        generated_at: '2026-06-02T00:00:00Z',
        summary_rows: [{
          values: { concurrency: 1, request_rate: -1, avg_latency: 2.4 },
          sample_counts: { avg_latency: 200 },
        }],
      }),
      'run-c': makePerfDetail({
        path: 'run-c',
        model: 'model-c',
        generated_at: '2026-06-03T00:00:00Z',
        summary_rows: [],
      }),
    })

    await renderLoaded([compareUrl(['run-a', 'run-b', 'run-c'])])

    const table = screen.getByTestId('performance-summary-table')
    expect(within(table).getByTestId('run-heading-1')).toHaveTextContent('R1')
    expect(within(table).getByTestId('run-heading-2')).toHaveTextContent('R2')
    expect(within(table).getByTestId('run-heading-3')).toHaveTextContent('R3')
    expect(screen.queryByTestId('delta-table')).not.toBeInTheDocument()
    expect(screen.queryByTestId('all-runs-table')).not.toBeInTheDocument()
    expect(screen.queryByTestId('swap-baseline')).not.toBeInTheDocument()

    const cells = within(screen.getByTestId('summary-row-0-avg_latency')).getAllByRole('cell')
    expect(cells).toHaveLength(4)
    expect(cells[1]).toHaveTextContent('1.2 s')
    expect(cells[2]).toHaveTextContent('2.4 s')
    expect(cells[3]).toHaveTextContent('—')
    expect(cells[1]).not.toHaveTextContent('Baseline')
    expect(within(table).getAllByTestId('workload-group-0')).toHaveLength(1)
  })

  it('compares every run against an arbitrary baseline and persists the selection', async () => {
    mockDetails({
      'run-a': makePerfDetail({ path: 'run-a' }),
      'run-b': makePerfDetail({ path: 'run-b', model: 'model-b', generated_at: '2026-06-02T00:00:00Z' }),
      'run-c': makePerfDetail({
        path: 'run-c',
        model: 'model-c',
        generated_at: '2026-06-03T00:00:00Z',
        summary_rows: [{
          values: { concurrency: 1, request_rate: -1, avg_latency: 0.6 },
          sample_counts: { avg_latency: 200 },
        }],
      }),
    })
    await renderLoaded([compareUrl(['run-a', 'run-b', 'run-c'])])

    expect(screen.getByLabelText('Baseline run')).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Vs Baseline' }))
    fireEvent.change(screen.getByLabelText('Baseline run'), { target: { value: 'run-c' } })

    expect(currentLocation()).toContain('baseline=run-c')
    expect(screen.getByTestId('run-heading-3')).toHaveTextContent('Baseline')
    const first = screen.getByTestId('summary-cell-0-avg_latency-1')
    expect(first).toHaveTextContent('1.2 s')
    expect(first).toHaveTextContent('+0.6 s')
    expect(first).toHaveTextContent('+100 %')
    expect(screen.getByTestId('summary-cell-0-avg_latency-3')).toHaveTextContent('Baseline')
  })

  it('honours a baseline supplied in the initial query', async () => {
    mockDetails({
      'run-a': makePerfDetail({ path: 'run-a' }),
      'run-b': makePerfDetail({ path: 'run-b', model: 'model-b', generated_at: '2026-06-02T00:00:00Z' }),
    })
    await renderLoaded([compareUrl(['run-a', 'run-b'], { baseline: 'run-b' })])
    fireEvent.click(screen.getByRole('button', { name: 'Vs Baseline' }))
    expect(screen.getByLabelText('Baseline run')).toHaveValue('run-b')
    expect(screen.getByTestId('run-heading-2')).toHaveTextContent('Baseline')
  })

  it('de-emphasizes low-sample percentile cells without hiding other metrics', async () => {
    mockDetails({
      'run-a': makePerfDetail({ path: 'run-a', __sampleCount: 10 }),
      'run-b': makePerfDetail({ path: 'run-b', generated_at: '2026-06-02T00:00:00Z', __sampleCount: 12 }),
    })
    await renderLoaded([compareUrl(['run-a', 'run-b'])])

    expect(screen.getByTestId('low-sample-critical')).toBeInTheDocument()
    for (const key of ['p90_latency', 'p95_latency', 'p99_latency']) {
      expect(screen.getByTestId(`summary-cell-0-${key}-1`)).toHaveAttribute('data-deemphasized', 'true')
    }
    expect(screen.getByTestId('summary-cell-0-avg_latency-1')).toHaveAttribute('data-deemphasized', 'false')
  })

  it('shows workload and missing-data warnings without blocking the summary', async () => {
    mockDetails({
      'run-a': makePerfDetail({ path: 'run-a' }),
      'run-b': makePerfDetail({
        path: 'run-b',
        dataset: 'longalpaca',
        generated_at: '2026-06-02T00:00:00Z',
        summary_rows: [],
      }),
    })
    await renderLoaded([compareUrl(['run-a', 'run-b'])])

    expect(screen.getByTestId('workload-mismatch')).toBeInTheDocument()
    expect(screen.getByTestId('missing-perf-data')).toBeInTheDocument()
    expect(screen.getByTestId('performance-summary-table')).toBeInTheDocument()
    expect(screen.getByTestId('summary-cell-0-avg_latency-2')).toHaveAttribute('data-deemphasized', 'true')
  })

  it('labels open-loop workloads without rendering a negative concurrency', async () => {
    const openLoop = (path: string, rate: number) => {
      const base = makePerfDetail({
        path,
        generated_at: path === 'run-a' ? '2026-06-01T00:00:00Z' : '2026-06-02T00:00:00Z',
      })
      return {
        ...base,
        configuration: { ...base.configuration, parallel: '-1', rate: String(rate) },
        summary_rows: [{
          ...base.summary_rows[0],
          values: { ...base.summary_rows[0].values, concurrency: -1, request_rate: rate },
        }],
      }
    }
    mockDetails({ 'run-a': openLoop('run-a', -1), 'run-b': openLoop('run-b', 10) })
    await renderLoaded([compareUrl(['run-a', 'run-b'])])

    const summary = within(screen.getByTestId('performance-summary-table'))
    expect(summary.getByText('open-loop')).toBeInTheDocument()
    expect(summary.getByText('open-loop · Request rate 10')).toBeInTheDocument()
    expect(summary.queryByText(/Concurrency -1|closed-loop/)).not.toBeInTheDocument()
  })

  it('disables input-length trends for non-random runs and keeps default chart URLs unchanged', async () => {
    mockDetails({
      'run-a': makePerfDetail({ path: 'run-a' }),
      'run-b': makePerfDetail({ path: 'run-b', generated_at: '2026-06-02T00:00:00Z' }),
    })
    await renderLoaded([compareUrl(['run-a', 'run-b'])])

    expect(screen.getByRole('button', { name: 'Input length' })).toBeDisabled()
    expect(screen.getByTestId('input-length-unavailable')).toHaveTextContent('fixed-length random datasets')
    expect(screen.getByTestId('chart-Latency (s)')).toHaveAttribute('data-src', expect.not.stringContaining('x_axis='))
  })

  it('enables input-length trends for concurrency runs with a finite configured rate', async () => {
    const ratedClosedLoop = (path: string, inputLength: number) => {
      const base = makeRandomDetail(path, inputLength)
      return {
        ...base,
        configuration: { ...base.configuration, parallel: '8', rate: '10' },
        summary_rows: [{
          ...base.summary_rows[0],
          values: { ...base.summary_rows[0].values, concurrency: 8, request_rate: 10 },
        }],
      }
    }
    mockDetails({
      'run-a': ratedClosedLoop('run-a', 1024),
      'run-b': ratedClosedLoop('run-b', 2048),
    })
    await renderLoaded([compareUrl(['run-a', 'run-b'])])

    const inputLength = screen.getByRole('button', { name: 'Input length' })
    expect(inputLength).toBeEnabled()
    expect(screen.queryByTestId('input-length-unavailable')).not.toBeInTheDocument()
    fireEvent.click(inputLength)
    expect(screen.getByTestId('chart-Latency (s)')).toHaveAttribute('data-src', expect.stringContaining('x_axis=input_length'))
  })

  it('shows reliable input labels and switches every chart to the input-length axis', async () => {
    mockDetails(Object.fromEntries([1024, 2048, 4096, 8192].map((length, index) => [
      `run-${index}`,
      makeRandomDetail(`run-${index}`, length),
    ])))
    await renderLoaded([compareUrl(['run-0', 'run-1', 'run-2', 'run-3'])])

    expect(screen.getByTestId('run-heading-1')).toHaveTextContent('Input 1,024')
    expect(screen.getByTestId('run-heading-4')).toHaveTextContent('Input 8,192')
    fireEvent.click(screen.getByRole('button', { name: 'Input length' }))

    expect(currentLocation()).toContain('x_axis=input_length')
    for (const title of ['Latency (s)', 'TTFT (ms)', 'TPOT (ms)', 'Requests/sec', 'Tokens/sec', 'Success Rate (%)']) {
      expect(screen.getByTestId(`chart-${title}`)).toHaveAttribute('data-src', expect.stringContaining('x_axis=input_length'))
    }
    const fallback = within(screen.getByTestId('chart-Latency (s)'))
    expect(fallback.getByText('Configured input length')).toBeInTheDocument()
    expect(fallback.getByText('1024')).toBeInTheDocument()
    expect(fallback.getByText('8192')).toBeInTheDocument()
  })

  it('describes sparse trends without prescribing marker rendering', async () => {
    mockDetails({
      'run-a': makeRandomDetail('run-a', 1024),
      'run-b': makeRandomDetail('run-b', 2048),
    })
    await renderLoaded([compareUrl(['run-a', 'run-b'])])

    expect(screen.getByTestId('sparse-hint')).toHaveTextContent('interpret the trend with caution')
    expect(screen.getByTestId('sparse-hint')).not.toHaveTextContent('discrete point markers')
  })

  it('uses actual summary points for the sparse hint', async () => {
    const rows = [1, 2, 4].map((concurrency) => ({
      values: { concurrency, request_rate: -1, avg_latency: concurrency },
      sample_counts: { avg_latency: 200 },
    }))
    mockDetails({
      'run-a': makePerfDetail({ path: 'run-a', summary_rows: rows }),
      'run-b': makePerfDetail({ path: 'run-b', generated_at: '2026-06-02T00:00:00Z', summary_rows: rows }),
    })
    await renderLoaded([compareUrl(['run-a', 'run-b'])])
    expect(screen.queryByTestId('sparse-hint')).not.toBeInTheDocument()
  })

  it('provides chart-specific all-run fallback data instead of pair deltas', async () => {
    mockDetails({
      'run-a': makePerfDetail({ path: 'run-a' }),
      'run-b': makePerfDetail({ path: 'run-b', generated_at: '2026-06-02T00:00:00Z' }),
    })
    await renderLoaded([compareUrl(['run-a', 'run-b'])])

    const chart = within(screen.getByTestId('chart-Latency (s)'))
    expect(chart.getByText('Concurrency / Rate')).toBeInTheDocument()
    expect(chart.getByText('Requests')).toBeInTheDocument()
    expect(chart.getByText('Value')).toBeInTheDocument()
    expect(chart.getAllByText('1.2 s')).toHaveLength(2)
    expect(chart.queryByText('Absolute delta')).not.toBeInTheDocument()
    expect(chart.queryByText('Candidate')).not.toBeInTheDocument()
  })

  it('uses input throughput in Tokens/sec fallback tables for SystemOne and AutoTrust', async () => {
    const decision = (path: string, apiType: 'systemone' | 'autotrust', inputThroughput: number) => {
      const base = makePerfDetail({ path, api_type: apiType, dataset: 'line_by_line', is_embedding: false })
      const columns = base.summary_columns.map((column) => column.key === 'output_token_throughput'
        ? { ...column, key: 'input_token_throughput', label: 'Input throughput (tokens/s)' }
        : column)
      const values: Record<string, number | null> = { ...base.summary_rows[0].values, input_token_throughput: inputThroughput }
      const sampleCounts: Record<string, number> = { ...base.summary_rows[0].sample_counts, input_token_throughput: 200 }
      delete values.output_token_throughput
      delete sampleCounts.output_token_throughput
      return {
        ...base,
        summary_columns: columns,
        summary_rows: [{ values, sample_counts: sampleCounts }],
      }
    }
    mockDetails({
      'run-a': decision('run-a', 'systemone', 500),
      'run-b': decision('run-b', 'autotrust', 700),
    })
    await renderLoaded([compareUrl(['run-a', 'run-b'])])

    const chart = within(screen.getByTestId('chart-Tokens/sec'))
    expect(chart.getByText('500 tok/s')).toBeInTheDocument()
    expect(chart.getByText('700 tok/s')).toBeInTheDocument()
  })
})
