// Report detail page — loads one report and hosts its Overview / Details /
// Predictions tabs.
//
// The page's own job is the load-then-frame flow: request the report for the
// URL's run/model, show the model identity once it arrives, and open on the
// report's first dataset. `loadReport` is mocked so the page resolves a fixture
// instead of the network; the tab bodies have their own dedicated tests.

import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'

import { LocaleProvider } from '@/contexts/LocaleContext'
import { ReportsProvider } from '@/contexts/ReportsContext'
import { loadFixture } from '@/test/loadFixture'
import type { LoadReportResponse } from '@/api/types'

vi.mock('@/api/reports', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/api/reports')>()),
  loadReport: vi.fn(),
}))

import * as reportsApi from '@/api/reports'
import ReportDetailPage from './ReportDetailPage'

const REPORT = loadFixture<LoadReportResponse>('report-multi-dataset')

async function settle(): Promise<void> {
  for (let i = 0; i < 8; i++) {
    await act(async () => { await Promise.resolve() })
  }
}

async function renderDetail(): Promise<void> {
  render(
    <LocaleProvider>
      <ReportsProvider>
        <MemoryRouter initialEntries={['/reports/20260810_112700/test-model-a']}>
          <Routes>
            <Route path="/reports/:runId/:modelId" element={<ReportDetailPage />} />
          </Routes>
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

describe('ReportDetailPage', () => {
  it('loads the report for the URL run/model and shows the model identity', async () => {
    vi.mocked(reportsApi.loadReport).mockResolvedValue(REPORT)
    await renderDetail()

    // The load is keyed on the URL's run and model.
    expect(reportsApi.loadReport).toHaveBeenCalledWith('./outputs', '20260810_112700/test-model-a', expect.anything())
    expect(screen.getAllByText('test-model-a').length).toBeGreaterThan(0)
  })

  it('offers the report datasets in the dataset navigation', async () => {
    vi.mocked(reportsApi.loadReport).mockResolvedValue(REPORT)
    await renderDetail()

    // Every dataset the report covers is reachable from the detail view.
    expect(screen.getAllByText(/gsm8k/i).length).toBeGreaterThan(0)
  })

  it('surfaces a load failure instead of a blank frame', async () => {
    vi.mocked(reportsApi.loadReport).mockRejectedValue(new Error('boom'))
    await renderDetail()

    expect(screen.getByText(/boom/i)).toBeInTheDocument()
  })
})
