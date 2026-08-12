// Evaluation history (Reports) page.
//
// It lists reports for the active root, paginates and filters them, and drives
// the compare-selection tray. `listReports` is mocked so the page resolves a
// fixed page instead of the network; these pin the load path, the empty state
// and the load-failure surface. Filter/selection minutiae live in the reports
// component tests.

import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import { LocaleProvider } from '@/contexts/LocaleContext'
import { ReportsProvider } from '@/contexts/ReportsContext'
import type { ListReportsResponse, ReportSummary } from '@/api/types'
import type { MetricSemantics } from '@/domain/metric'

vi.mock('@/api/reports', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/api/reports')>()),
  listReports: vi.fn(),
}))

import * as reportsApi from '@/api/reports'
import ReportsPage from './ReportsPage'

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

const REPORT = {
  run_id: '20260810_112700',
  model_id: 'qwen-plus',
  model_name: 'qwen-plus',
  dataset_name: 'gsm8k',
  num_samples: 3,
  timestamp: '2026-08-10T11:27:00',
  primary_metrics: [{ dataset_name: 'gsm8k', identity: { name: 'accuracy', aggregation: 'mean', dimensions: {} }, score: 0.6, semantics: ACCURACY }],
} as ReportSummary

function page(reports: ReportSummary[]): ListReportsResponse {
  return {
    reports,
    total: reports.length,
    page: 1,
    page_size: 20,
    filters: { available_models: ['qwen-plus'], available_datasets: ['gsm8k'] },
  }
}

async function settle(): Promise<void> {
  for (let i = 0; i < 8; i++) {
    await act(async () => { await Promise.resolve() })
  }
}

async function renderReports(): Promise<void> {
  render(
    <LocaleProvider>
      <ReportsProvider>
        <MemoryRouter initialEntries={['/reports']}>
          <ReportsPage />
        </MemoryRouter>
      </ReportsProvider>
    </LocaleProvider>,
  )
  await settle()
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('ReportsPage', () => {
  it('loads reports for the active root and lists them', async () => {
    vi.mocked(reportsApi.listReports).mockResolvedValue(page([REPORT]))
    await renderReports()

    expect(reportsApi.listReports).toHaveBeenCalled()
    // Rendered on both the table and card surfaces, so more than one match is expected.
    expect(screen.getAllByText('qwen-plus').length).toBeGreaterThan(0)
  })

  it('shows an empty state when the root has no reports', async () => {
    vi.mocked(reportsApi.listReports).mockResolvedValue(page([]))
    await renderReports()

    expect(screen.queryAllByText('qwen-plus')).toHaveLength(0)
    expect(reportsApi.listReports).toHaveBeenCalled()
  })

  it('surfaces a load failure', async () => {
    vi.mocked(reportsApi.listReports).mockRejectedValue(new Error('list down'))
    await renderReports()

    expect(screen.getByText(/list down/i)).toBeInTheDocument()
  })

  it('does not expose global score filters or sorting', async () => {
    vi.mocked(reportsApi.listReports).mockResolvedValue(page([REPORT]))
    await renderReports()

    expect(screen.queryByRole('spinbutton')).not.toBeInTheDocument()
    expect(screen.queryByRole('option', { name: 'Score' })).not.toBeInTheDocument()
  })
})
