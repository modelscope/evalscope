import { cleanup, render, screen } from '@testing-library/react'
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

describe('OverviewTab adaptive visualization', () => {
  const multi = loadFixture<{ report_list: ReportData[] }>('report-multi-dataset').report_list

  it('renders a labelled single value instead of a one-axis radar', () => {
    renderOverview(multi.slice(0, 1))

    const visualization = screen.getByRole('img', { name: 'Single metric value' })
    expect(visualization).toBeInTheDocument()
    expect(visualization).toHaveTextContent('gsm8k')
    expect(screen.queryByTestId('radar-chart')).not.toBeInTheDocument()
  })

  it('renders a grouped comparison instead of a two-axis radar', () => {
    renderOverview(multi.slice(0, 2))

    expect(screen.getByRole('img', { name: 'Grouped bar chart' })).toBeInTheDocument()
    expect(screen.queryByTestId('radar-chart')).not.toBeInTheDocument()
  })

  it('renders the radar chart only for three or more dimensions', () => {
    renderOverview(multi)

    expect(screen.getByTestId('radar-chart')).toBeInTheDocument()
  })
})
