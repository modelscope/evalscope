// Component tests for the Performance_View surfaces (Task 12.6).
//
// These cover the display behaviour that lives in the migrated Performance
// pages (task 12.5) rather than in the pure `providerResolution` /
// `perfWorkload` domain logic (which have their own property tests):
//   - Provider and Protocol render as two independent, individually-labelled
//     fields and never collapse into one combined field (Req 8.1);
//   - a single-run report uses single-run wording (`Run Summary` /
//     `Run Configuration`) and never the cross-run "best configuration" /
//     "Cross-Run" phrasing, while a multi-run report does the opposite
//     (Req 8.8);
//   - the run identity is the model alias, not the raw path or timestamp
//     (Req 8.9);
//   - the per-run workload context (concurrency, number of requests, request
//     rate) is surfaced and never shows `INF` (Req 8.4).
//
// The suite runs under the global deterministic setup (fake timers, fixed
// system time, network disabled). The perf API module is mocked so the pages
// resolve deterministic fixtures instead of hitting the network; async data
// loads are flushed with `settle()`.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import { LocaleProvider } from '@/contexts/LocaleContext'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { ReportsProvider } from '@/contexts/ReportsContext'
import { useReports } from '@/contexts/ReportsContext'
import { DomainError } from '@/api/errors'
import { loadFixture } from '@/test/loadFixture'
import type {
  PerfDetailResponse,
  PerfRequestsResponse,
  PerfRunsListResponse,
  PerfRunSummary,
} from '@/api/types'

import PerfReportDetailPage from './PerfReportDetailPage'
import PerfRunsTab from './PerfRunsTab'
import PerfReportsPage from './PerfReportsPage'

// The perf API module is entirely mocked: the URL builders return stable
// strings and the data loaders are configured per test to resolve fixtures.
vi.mock('@/api/perf', () => ({
  getPerfDetail: vi.fn(),
  listPerfRunDetails: vi.fn(),
  getPerfRequests: vi.fn(),
  listPerfRuns: vi.fn(),
  getPerfChartUrl: vi.fn(() => '/api/v1/perf/chart'),
  getPerfHistoryReportUrl: vi.fn(() => '/api/v1/perf/history/report'),
}))

import * as perfApi from '@/api/perf'

const detailFixture = loadFixture<PerfDetailResponse>('perf-detail')
const singleRunFixture = loadFixture<PerfRunsListResponse>('perf-single-run')

const RUN_PATH = 'perf/test-model-a/20240101_000000'

/** An empty per-request page so the Runs tab has no rows to render. */
const EMPTY_REQUESTS: PerfRequestsResponse = {
  columns: [],
  rows: [],
  total: 0,
  page: 1,
  page_size: 50,
  has_db: true,
}

/** A representative perf-run summary for the list view. */
const RUN_SUMMARY: PerfRunSummary = {
  path: RUN_PATH,
  model: 'test-model-a',
  api_type: 'openai_api',
  dataset: 'openqa',
  num_runs: 1,
  total_requests: 100,
  success_rate: 100.0,
  best_rps: 5.43,
  best_latency: 1.842,
  is_embedding: false,
  has_html: true,
  timestamp: '2024-01-01T00:00:00',
  api_host: 'dashscope.aliyuncs.com',
  concurrency: [10],
}

/**
 * Flush the chained promise resolutions and effect re-renders produced by the
 * pages' async data loads. Mocked promises settle on microtasks (no real
 * timers), so a handful of act flushes drains the load → state → effect chain.
 */
async function settle(): Promise<void> {
  for (let i = 0; i < 8; i++) {
    await act(async () => {
      await Promise.resolve()
    })
  }
}

/** Text of the value span rendered next to a labelled identity/workload field. */
function labelledValue(label: string): string {
  const labelEl = screen.getByText(label)
  return labelEl.nextElementSibling?.textContent ?? ''
}

beforeEach(() => {
  vi.mocked(perfApi.getPerfDetail).mockResolvedValue(detailFixture)
  vi.mocked(perfApi.listPerfRunDetails).mockResolvedValue(singleRunFixture)
  vi.mocked(perfApi.getPerfRequests).mockResolvedValue(EMPTY_REQUESTS)
  vi.mocked(perfApi.listPerfRuns).mockResolvedValue({ runs: [RUN_SUMMARY], total: 1 })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

// ------------------------------------------------------------------ //
// PerfReportDetailPage                                                //
// ------------------------------------------------------------------ //

/** Render the detail page for the given run path within the required providers. */
async function renderDetail(detail: PerfDetailResponse = detailFixture) {
  vi.mocked(perfApi.getPerfDetail).mockResolvedValue(detail)
  const entry =
    `/perf-report?path=${encodeURIComponent(RUN_PATH)}&root_path=${encodeURIComponent('./outputs')}`
  render(
    <LocaleProvider>
      <ThemeProvider>
        <ReportsProvider>
          <MemoryRouter initialEntries={[entry]}>
            <PerfReportDetailPage />
          </MemoryRouter>
        </ReportsProvider>
      </ThemeProvider>
    </LocaleProvider>,
  )
  await settle()
}

describe('PerfReportDetailPage', () => {
  it('renders Provider and Protocol as two independent labelled fields (Req 8.1)', async () => {
    await renderDetail()

    // Both labels are present as distinct fields ...
    expect(screen.getByText('Provider')).toBeInTheDocument()
    expect(screen.getByText('Protocol')).toBeInTheDocument()

    // ... each carrying its own, independent value (not merged into one field).
    expect(labelledValue('Provider')).toBe('Custom')
    expect(labelledValue('Protocol')).toBe('OpenAI-compatible')
    expect(labelledValue('Provider')).not.toBe(labelledValue('Protocol'))
  })

  it('detects DashScope from a sanitized archive endpoint host', async () => {
    await renderDetail({
      ...detailFixture,
      basic_info: { ...detailFixture.basic_info, Provider: '', 'API Host': 'dashscope.aliyuncs.com' },
    })

    expect(labelledValue('Provider')).toBe('DashScope')
    expect(screen.queryByText('dashscope.aliyuncs.com')).not.toBeInTheDocument()
  })

  it('uses the model alias as the primary identity, not the path/timestamp (Req 8.9)', async () => {
    await renderDetail()

    const heading = screen.getByRole('heading', { level: 1 })
    expect(heading).toHaveTextContent('test-model-a')
    // The raw path and timestamp must not be the primary identity.
    expect(heading.textContent).not.toContain('20240101_000000')
    expect(heading.textContent).not.toContain('perf/')
  })

  it('uses single-run wording and avoids cross-run "best configuration" phrasing (Req 8.8)', async () => {
    await renderDetail() // detailFixture has num_runs === 1

    // A single-run report front-loads the Runs tab; switch to Overview to
    // inspect the summary/configuration section wording.
    fireEvent.click(screen.getByRole('tab', { name: 'Overview' }))

    expect(screen.getByText('Run Summary')).toBeInTheDocument()
    expect(screen.getByText('Run Configuration')).toBeInTheDocument()
    expect(screen.getByText(/Single-run benchmark/)).toBeInTheDocument()

    // The cross-run phrasing must not appear for a single-run report.
    expect(screen.queryByText('Cross-Run Summary')).not.toBeInTheDocument()
    expect(screen.queryByText('Best Configuration')).not.toBeInTheDocument()
  })

  it('replaces the unlimited-rate sentinel in the overview summary', async () => {
    await renderDetail({
      ...detailFixture,
      summary_columns: ['Conc.', 'Rate', 'RPS'],
      summary_rows: [['2', 'INF', '1.9256']],
    })
    fireEvent.click(screen.getByRole('tab', { name: 'Overview' }))

    expect(screen.getAllByText('closed-loop')).not.toHaveLength(0)
    expect(screen.queryByText(/^INF$/i)).not.toBeInTheDocument()
  })

  it('uses cross-run wording for a multi-run report (Req 8.8)', async () => {
    await renderDetail({ ...detailFixture, num_runs: 3 })

    // A multi-run report stays on the Overview tab by default.
    expect(screen.getByText('Cross-Run Summary')).toBeInTheDocument()
    expect(screen.getByText('Best Configuration')).toBeInTheDocument()

    // The single-run phrasing must not appear for a multi-run report.
    expect(screen.queryByText('Run Summary')).not.toBeInTheDocument()
    expect(screen.queryByText('Run Configuration')).not.toBeInTheDocument()
    expect(screen.queryByText(/Single-run benchmark/)).not.toBeInTheDocument()
  })
})

// ------------------------------------------------------------------ //
// PerfRunsTab                                                         //
// ------------------------------------------------------------------ //

/** Render the per-run tab in isolation with the required locale provider. */
async function renderRunsTab() {
  render(
    <LocaleProvider>
      <PerfRunsTab rootPath="./outputs" path={RUN_PATH} isEmbedding={false} />
    </LocaleProvider>,
  )
  await settle()
}

describe('PerfRunsTab', () => {
  it('surfaces the workload parameters for the selected run (Req 8.4)', async () => {
    await renderRunsTab()

    // The single-run fixture is parallel=10, number=100, rate=null.
    expect(labelledValue('Concurrency')).toBe('10')
    expect(labelledValue('Number of requests')).toBe('100')
    // A null rate under a concurrency limit is a closed-loop workload ...
    expect(labelledValue('Request rate')).toBe('closed-loop')
    // ... and never the raw `INF` sentinel (Req 8.6).
    expect(screen.queryByText(/INF/i)).not.toBeInTheDocument()
  })
})

// ------------------------------------------------------------------ //
// PerfReportsPage                                                     //
// ------------------------------------------------------------------ //

/** Render the perf-run list page within the required providers. */
async function renderReports() {
  render(
    <LocaleProvider>
      <ReportsProvider>
        <MemoryRouter initialEntries={['/performance']}>
          <PerfReportsPage />
        </MemoryRouter>
      </ReportsProvider>
    </LocaleProvider>,
  )
  await settle()
}

function RescanHarness() {
  const { triggerScan } = useReports()
  return <button onClick={() => triggerScan()}>Rescan fixture</button>
}

describe('PerfReportsPage', () => {
  it('lists a run by its model alias, not the raw path (Req 8.9)', async () => {
    await renderReports()

    // The model alias is the primary identity in the card ...
    expect(screen.getByText('test-model-a')).toBeInTheDocument()
    // ... and the raw archive path is never shown as the identity.
    expect(screen.queryByText(RUN_PATH)).not.toBeInTheDocument()
    expect(screen.getByText(/Provider: DashScope/)).toBeInTheDocument()
    expect(screen.getAllByText(/Concurrency: 10/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Number of requests: 100/).length).toBeGreaterThan(0)
  })

  it('searches provider and protocol independently from the API type', async () => {
    await renderReports()
    const search = screen.getByPlaceholderText('Search model / api / dataset...')

    fireEvent.change(search, { target: { value: 'DashScope' } })
    expect(screen.getByText('test-model-a')).toBeInTheDocument()

    fireEvent.change(search, { target: { value: 'OpenAI-compatible' } })
    expect(screen.getByText('test-model-a')).toBeInTheDocument()
  })

  it('keeps the existing run visible when a refreshed response fails schema validation (Req 13.4)', async () => {
    render(
      <LocaleProvider>
        <ReportsProvider>
          <MemoryRouter initialEntries={['/performance']}>
            <RescanHarness />
            <PerfReportsPage />
          </MemoryRouter>
        </ReportsProvider>
      </LocaleProvider>,
    )
    await settle()
    expect(screen.getByText('test-model-a')).toBeInTheDocument()

    vi.mocked(perfApi.listPerfRuns).mockRejectedValueOnce(
      new DomainError('validation', 'Performance response did not match its schema'),
    )
    fireEvent.click(screen.getByRole('button', { name: 'Rescan fixture' }))
    await settle()

    expect(screen.getByText('test-model-a')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('Performance response did not match its schema')
  })
})
