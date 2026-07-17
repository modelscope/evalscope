import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ReportData } from '@/api/types'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { loadFixture } from '@/test/loadFixture'
import OverviewTab from './OverviewTab'

vi.mock('@/components/charts/PlotlyChart', () => ({
  default: ({ src }: { src: string }) => <div data-testid="radar-chart" data-src={src} />,
}))

afterEach(cleanup)

function renderOverview(reports: ReportData[]) {
  return render(
    <LocaleProvider>
      <OverviewTab reports={reports} reportName="fixture-report" rootPath="/outputs" />
    </LocaleProvider>,
  )
}

describe('OverviewTab dataset score view', () => {
  const multi = loadFixture<{ report_list: ReportData[] }>('report-multi-dataset').report_list

  it('renders a single dataset in the score table without a duplicate visualization', () => {
    renderOverview(multi.slice(0, 1))

    expect(screen.getByRole('progressbar', { name: 'gsm8k Score' })).toBeInTheDocument()
    expect(screen.queryByText('Dataset Score Visualization')).not.toBeInTheDocument()
    expect(screen.queryByTestId('radar-chart')).not.toBeInTheDocument()
  })

  it('keeps two datasets in one sortable table', () => {
    renderOverview(multi.slice(0, 2))

    expect(screen.getByRole('progressbar', { name: 'gsm8k Score' })).toBeInTheDocument()
    expect(screen.getByRole('progressbar', { name: 'arc_challenge Score' })).toBeInTheDocument()
    expect(screen.getByText('81.5%')).toBeInTheDocument()
    expect(screen.queryByText('8150.0%')).not.toBeInTheDocument()
    expect(screen.queryByTestId('radar-chart')).not.toBeInTheDocument()
  })

  it('keeps unbounded metrics in their native unit without a percentage bar', () => {
    renderOverview([multi[2]])

    expect(screen.getAllByText('512.00 tokens/s')).not.toHaveLength(0)
    expect(screen.queryByText('51200.0%')).not.toBeInTheDocument()
    expect(screen.queryByRole('progressbar', { name: 'throughput_suite Score' })).not.toBeInTheDocument()
  })

  it('offers radar only for three or more comparable bounded metrics', () => {
    const comparable = [0, 1, 2].map((index) => ({
      ...multi[0],
      name: `accuracy-${index}`,
      dataset_name: `accuracy_${index}`,
      score: 0.7 + index * 0.1,
    }))
    renderOverview(comparable)

    expect(screen.queryByTestId('radar-chart')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Radar' }))
    expect(screen.getByTestId('radar-chart')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Table' }))
    expect(screen.queryByTestId('radar-chart')).not.toBeInTheDocument()
  })

  it('does not compare heterogeneous metrics in a radar chart', () => {
    renderOverview(multi)

    expect(screen.queryByRole('button', { name: 'Radar' })).not.toBeInTheDocument()
  })
})
