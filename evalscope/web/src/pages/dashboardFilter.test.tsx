// The dashboard's ALL / Eval / Perf tabs.
//
// The table mixes two kinds of run -- evaluations and performance sweeps -- whose metrics sit on
// unrelated scales, so being able to look at one kind alone is the point of the control. These pin
// the part that is easy to get wrong: the tabs must actually narrow the rows (not just restyle a
// button), `All` must restore everything, and a kind with no rows must say so rather than render a
// bare table header.
//
// Both API modules are mocked so the page resolves fixtures instead of hitting the network; the
// providers supply the default `./outputs` root that triggers the load effect.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'

import { LocaleProvider } from '@/contexts/LocaleContext'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { ReportsProvider, useScan } from '@/contexts/ReportsContext'
import type { MetricSemantics } from '@/domain/metric'
import type { PerfRunSummary, ReportSummary } from '@/api/types'

import DashboardPage from './DashboardPage'

vi.mock('@/api/reports', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/api/reports')>()),
  listReports: vi.fn(),
}))

vi.mock('@/api/perf', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/api/perf')>()),
  listPerfRuns: vi.fn(),
}))

import * as perfApi from '@/api/perf'
import * as reportsApi from '@/api/reports'

const ACCURACY: MetricSemantics = {
  semantic_id: 'quality.accuracy.ratio',
  metric_name: 'Accuracy',
  kind: 'quality',
  direction: 'higher_is_better',
  value_range: { min: 0, max: 1 },
  display_kind: 'percent',
  display_multiplier: 100,
  display_unit: '%',
  display_precision: 1,
}

const RPS: MetricSemantics = {
  semantic_id: 'perf.throughput.requests_per_second',
  metric_name: 'Best RPS',
  kind: 'quality',
  direction: 'higher_is_better',
  raw_unit: 'req/s',
  display_kind: 'number',
  display_unit: '',
  display_precision: 2,
}

const EVAL_REPORT = {
  run_id: '20260810_112700',
  model_id: 'qwen-plus',
  model_name: 'qwen-plus',
  dataset_name: 'gsm8k',
  num_samples: 3,
  timestamp: '2026-08-10T11:27:00',
  primary_metrics: [{ dataset_name: 'gsm8k', identity: { name: 'accuracy', aggregation: 'mean', dimensions: {} }, score: 0.6, semantics: ACCURACY }],
} as ReportSummary

const PERF_RUN = {
  path: 'perf/qwen-plus/20260807_094400',
  model: 'qwen-plus',
  api_type: 'openai_api',
  dataset: 'openqa',
  num_runs: 1,
  total_requests: 10,
  success_rate: 100,
  best_rps: 0.12,
  best_latency: 1.2,
  is_embedding: false,
  has_html: true,
  timestamp: '2026-08-07T09:44:00',
  concurrency: [1],
} as PerfRunSummary

/** Flush the chained promise resolutions and effect re-renders of the page's async load. */
async function settle(): Promise<void> {
  for (let i = 0; i < 8; i++) {
    await act(async () => {
      await Promise.resolve()
    })
  }
}

async function renderDashboard(): Promise<void> {
  render(
    <LocaleProvider>
      <ThemeProvider>
        <ReportsProvider>
          <MemoryRouter initialEntries={['/dashboard']}>
            <DashboardPage />
            <NavigationProbe />
          </MemoryRouter>
        </ReportsProvider>
      </ThemeProvider>
    </LocaleProvider>,
  )
  await settle()
}

function NavigationProbe() {
  const location = useLocation()
  const { setRootPath, triggerScan } = useScan()
  return <>
    <output data-testid="location">{location.pathname}{location.search}</output>
    <button onClick={() => setRootPath('./other-outputs')}>Change test root</button>
    <button onClick={() => triggerScan()}>Rescan test root</button>
  </>
}

function mockReports(reports: ReportSummary[]) {
  vi.mocked(reportsApi.listReports).mockResolvedValue({
    reports, total: reports.length, page: 1, page_size: 100,
    filters: { available_models: [], available_datasets: [] },
  })
}

function selectRun(ref: string) {
  fireEvent.click(screen.getByRole('checkbox', { name: `Select run ${ref}` }))
}

/** Click one of the kind tabs by its visible label. */
function selectTab(label: string): void {
  fireEvent.click(screen.getByRole('tab', { name: label }))
}

beforeEach(() => {
  vi.mocked(reportsApi.listReports).mockResolvedValue({
    reports: [EVAL_REPORT],
    total: 1,
    page: 1,
    page_size: 1000,
    filters: { available_models: ['qwen-plus'], available_datasets: ['gsm8k'] },
  })
  vi.mocked(perfApi.listPerfRuns).mockResolvedValue({
    runs: [PERF_RUN],
    total: 1,
    metric_semantics: { best_rps: RPS },
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('dashboard kind tabs', () => {
  it('loads every evaluation report page', async () => {
    vi.mocked(reportsApi.listReports)
      .mockResolvedValueOnce({
        reports: [EVAL_REPORT],
        total: 2,
        page: 1,
        page_size: 100,
        filters: { available_models: [], available_datasets: [] },
      })
      .mockResolvedValueOnce({
        reports: [{ ...EVAL_REPORT, run_id: '20260809_112700' }],
        total: 2,
        page: 2,
        page_size: 100,
        filters: { available_models: [], available_datasets: [] },
      })

    await renderDashboard()

    expect(reportsApi.listReports).toHaveBeenNthCalledWith(1, expect.objectContaining({ page: 1, pageSize: 100 }))
    expect(reportsApi.listReports).toHaveBeenNthCalledWith(2, expect.objectContaining({ page: 2, pageSize: 100 }))
  })

  it('shows both kinds under All', async () => {
    await renderDashboard()

    expect(screen.getByText('gsm8k')).toBeInTheDocument()
    expect(screen.getByText('openqa')).toBeInTheDocument()
  })

  it('narrows the table to evaluations', async () => {
    await renderDashboard()

    selectTab('Eval')

    expect(screen.getByText('gsm8k')).toBeInTheDocument()
    expect(screen.queryByText('openqa')).not.toBeInTheDocument()
  })

  it('narrows the table to performance runs', async () => {
    await renderDashboard()

    selectTab('Perf')

    expect(screen.getByText('openqa')).toBeInTheDocument()
    expect(screen.queryByText('gsm8k')).not.toBeInTheDocument()
  })

  it('restores every row when All is selected again', async () => {
    await renderDashboard()

    selectTab('Perf')
    selectTab('All')

    expect(screen.getByText('gsm8k')).toBeInTheDocument()
    expect(screen.getByText('openqa')).toBeInTheDocument()
  })

  it('keeps exactly one tab selected', async () => {
    await renderDashboard()

    selectTab('Eval')

    const selected = screen.getAllByRole('tab').filter((tab) => tab.getAttribute('aria-selected') === 'true')
    expect(selected).toHaveLength(1)
    expect(selected[0]).toHaveTextContent('Eval')
  })

  it('says a kind is empty instead of rendering a table with no rows', async () => {
    // No perf runs recorded at all, so the Perf tab has nothing to show.
    vi.mocked(perfApi.listPerfRuns).mockResolvedValue({ runs: [], total: 0, metric_semantics: {} })
    await renderDashboard()

    selectTab('Perf')

    // The column headers of the results table are gone, replaced by the empty state.
    expect(screen.queryByRole('columnheader', { name: /Benchmark/ })).not.toBeInTheDocument()
    expect(screen.queryByText('gsm8k')).not.toBeInTheDocument()
  })

  it('filters rows by model or benchmark from the toolbar', async () => {
    await renderDashboard()

    fireEvent.change(screen.getByPlaceholderText('Search model or benchmark'), {
      target: { value: 'openqa' },
    })

    expect(screen.getByText('openqa')).toBeInTheDocument()
    expect(screen.queryByText('gsm8k')).not.toBeInTheDocument()
  })

  it('changes the result order from the toolbar sort control', async () => {
    await renderDashboard()

    fireEvent.change(screen.getByLabelText('Sort results'), { target: { value: 'lastRun-asc' } })

    const bodyRows = screen.getAllByRole('row').slice(1)
    expect(bodyRows[0]).toHaveTextContent('openqa')
    expect(bodyRows[1]).toHaveTextContent('gsm8k')
  })

  it('surfaces the most recent changed result without inventing an alert system', async () => {
    const older = {
      ...EVAL_REPORT,
      name: 'qwen-plus_gsm8k_20260809_112700',
      timestamp: '2026-08-09T11:27:00',
      primary_metrics: [{ ...EVAL_REPORT.primary_metrics[0], score: 1 }],
    } as ReportSummary
    vi.mocked(reportsApi.listReports).mockResolvedValue({
      reports: [EVAL_REPORT, older],
      total: 2,
      page: 1,
      page_size: 1000,
      filters: { available_models: ['qwen-plus'], available_datasets: ['gsm8k'] },
    })

    await renderDashboard()

    expect(screen.getByText('Recent change')).toBeInTheDocument()
    expect(screen.getByText(/a change of -40 pp from the previous run/)).toBeInTheDocument()
  })
})

describe('dashboard run comparison', () => {
  const oldRun = { ...EVAL_REPORT, run_id: '20260809_112700', timestamp: '2026-08-09T11:27:00' }
  const otherModel = { ...EVAL_REPORT, model_id: 'other-model', model_name: 'other-model' }

  it('compares the exact selected runs across model groups and filters', async () => {
    mockReports([EVAL_REPORT, oldRun, otherModel])
    await renderDashboard()

    fireEvent.click(screen.getAllByRole('button', { name: 'Show history' })[0])
    selectRun('20260809_112700/qwen-plus')
    expect(screen.getByRole('button', { name: 'Compare' })).toBeDisabled()

    fireEvent.change(screen.getByPlaceholderText('Search model or benchmark'), { target: { value: 'other-model' } })
    fireEvent.click(screen.getByRole('button', { name: 'Show history' }))
    selectRun('20260810_112700/other-model')
    fireEvent.click(screen.getByRole('button', { name: 'Compare' }))

    const location = new URL(screen.getByTestId('location').textContent!, 'http://localhost')
    expect(location.pathname).toBe('/compare')
    expect(location.searchParams.get('root_path')).toBe('./outputs')
    expect(location.searchParams.getAll('report')).toEqual([
      '20260809_112700/qwen-plus', '20260810_112700/other-model',
    ])
  })

  it('keeps repeated runs of the same model distinct', async () => {
    mockReports([EVAL_REPORT, oldRun])
    await renderDashboard()
    selectTab('Eval')
    fireEvent.click(screen.getByRole('button', { name: 'Show history' }))
    selectRun('20260809_112700/qwen-plus')
    selectRun('20260810_112700/qwen-plus')
    fireEvent.click(screen.getByRole('button', { name: 'Compare' }))

    const location = new URL(screen.getByTestId('location').textContent!, 'http://localhost')
    expect(location.searchParams.getAll('report')).toEqual([
      '20260809_112700/qwen-plus', '20260810_112700/qwen-plus',
    ])
  })

  it('shares one selection for a run recorded under multiple datasets', async () => {
    mockReports([EVAL_REPORT, {
      ...EVAL_REPORT, dataset_name: 'iquiz',
      primary_metrics: [{ ...EVAL_REPORT.primary_metrics[0], dataset_name: 'iquiz' }],
    }])
    await renderDashboard()
    selectTab('Eval')
    fireEvent.click(screen.getAllByRole('button', { name: 'Show history' })[0])
    selectRun('20260810_112700/qwen-plus')
    fireEvent.click(screen.getByRole('button', { name: 'Show history' }))

    expect(screen.getByRole('checkbox', { name: 'Select run 20260810_112700/qwen-plus' }))
      .toHaveAttribute('aria-checked', 'true')
    expect(screen.getByText('1 selected')).toBeInTheDocument()
    selectRun('20260810_112700/qwen-plus')
    expect(screen.queryByText('1 selected')).not.toBeInTheDocument()
  })

  it.each(['Change test root', 'Rescan test root'])('clears the selection on %s', async (button) => {
    await renderDashboard()
    selectTab('Eval')
    fireEvent.click(screen.getByRole('button', { name: 'Show history' }))
    selectRun('20260810_112700/qwen-plus')
    fireEvent.click(screen.getByRole('button', { name: button }))
    await settle()

    expect(screen.queryByText('1 selected')).not.toBeInTheDocument()
  })

  it('does not offer evaluation comparison for performance runs', async () => {
    await renderDashboard()
    selectTab('Perf')
    fireEvent.click(screen.getByRole('button', { name: 'Show history' }))

    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument()
  })

  it('can clear selected runs without changing the displayed history', async () => {
    await renderDashboard()
    selectTab('Eval')
    fireEvent.click(screen.getByRole('button', { name: 'Show history' }))
    selectRun('20260810_112700/qwen-plus')
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))

    expect(screen.getByRole('checkbox', { name: 'Select run 20260810_112700/qwen-plus' }))
      .toHaveAttribute('aria-checked', 'false')
    expect(screen.queryByText('1 selected')).not.toBeInTheDocument()
  })
})
