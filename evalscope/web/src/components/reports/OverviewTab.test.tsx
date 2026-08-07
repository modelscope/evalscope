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

    expect(screen.getAllByText('gsm8k').length).toBeGreaterThan(0)
    expect(screen.queryByText('Dataset Score Visualization')).not.toBeInTheDocument()
    expect(screen.queryByTestId('radar-chart')).not.toBeInTheDocument()
  })

  it('keeps two datasets in one table, each with its own metric column', () => {
    // The rows of this table are different datasets, so no bar is drawn: a bar length would imply
    // a shared axis that an accuracy and a WER do not have. The metric moved to its own column.
    renderOverview(multi.slice(0, 2))

    expect(screen.getAllByText('gsm8k').length).toBeGreaterThan(0)
    expect(screen.getAllByText('arc_challenge').length).toBeGreaterThan(0)
    expect(screen.queryAllByRole('progressbar')).toHaveLength(0)
    expect(screen.getAllByText('81.5%').length).toBeGreaterThan(0)
    expect(screen.getAllByText(/^Score ↑$/).length).toBeGreaterThan(0)
    expect(screen.queryByText(/8150/)).not.toBeInTheDocument()
    expect(screen.queryByTestId('radar-chart')).not.toBeInTheDocument()
  })

  it('keeps an unbounded metric in its native unit', () => {
    renderOverview([multi[2]])

    // An unbounded throughput keeps its native unit and is never rescaled to a percentage.
    expect(screen.getAllByText(/^Token Throughput ↑$/)).not.toHaveLength(0)
    expect(screen.getAllByText('512 tok/s').length).toBeGreaterThan(0)
    expect(screen.queryByText(/51200/)).not.toBeInTheDocument()
    expect(screen.queryAllByRole('progressbar')).toHaveLength(0)
  })

  it('never states that metrics could not be merged', () => {
    // Two benchmarks simply have two results; that is not a condition to warn about, and a note
    // in place of the numbers hides what the run actually produced.
    renderOverview(multi.slice(0, 2))

    expect(screen.queryByText(/cannot be merged/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/primary metrics/i)).not.toBeInTheDocument()
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
