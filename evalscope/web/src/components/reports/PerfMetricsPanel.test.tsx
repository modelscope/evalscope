import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'

import { loadReportResponseSchema } from '@/api/schemas/reports.schema'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { loadFixture } from '@/test/loadFixture'
import PerfMetricsPanel from './PerfMetricsPanel'

afterEach(cleanup)

describe('PerfMetricsPanel missing statistics', () => {
  it('renders undefined single-sample standard deviations as placeholders', () => {
    const fixture = loadReportResponseSchema.parse(loadFixture<unknown>('report-real-single-sample'))
    const perfMetrics = fixture.report_list[0].perf_metrics
    expect(perfMetrics).toBeDefined()
    expect(perfMetrics).not.toBeNull()
    if (!perfMetrics) return

    const { container } = render(
      <LocaleProvider>
        <PerfMetricsPanel perfMetrics={perfMetrics} />
      </LocaleProvider>,
    )

    expect(screen.getAllByText('—')).toHaveLength(6)
    expect(container).not.toHaveTextContent('NaN')
    expect(container).not.toHaveTextContent('null')

    const kpiLabel = screen.getByText('Requests')
    const kpiStrip = kpiLabel.parentElement?.parentElement
    expect(kpiStrip).toHaveClass('grid', 'grid-cols-2', 'sm:grid-cols-3', 'xl:grid-cols-4')
    expect(kpiLabel).toHaveClass('break-words')
    expect(kpiLabel).not.toHaveClass('whitespace-nowrap')
  })
})
