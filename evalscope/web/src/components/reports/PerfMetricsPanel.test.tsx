import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import type { LoadReportResponse } from '@/api/types'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { loadFixture } from '@/test/loadFixture'
import PerfMetricsPanel from './PerfMetricsPanel'

afterEach(cleanup)

describe('PerfMetricsPanel missing statistics', () => {
  it('renders undefined single-sample standard deviations as placeholders', () => {
    const fixture = loadFixture<LoadReportResponse>('report-real-single-sample')
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

  it('formats values from the backend semantics map', () => {
    const fixture = loadFixture<LoadReportResponse>('report-real-single-sample')
    const perfMetrics = fixture.report_list[0].perf_metrics
    expect(perfMetrics).toBeDefined()
    expect(perfMetrics).not.toBeNull()
    if (!perfMetrics) return
    perfMetrics.metric_semantics!.latency = {
        semantic_id: 'perf.latency.seconds',
        metric_name: 'Latency',
        kind: 'quality',
        direction: 'lower_is_better',
        raw_unit: 's',
        display_kind: 'number',
        display_multiplier: 1000,
        display_unit: 'ms',
        display_precision: 1,
    }

    render(
      <LocaleProvider>
        <PerfMetricsPanel perfMetrics={perfMetrics} />
      </LocaleProvider>,
    )

    expect(screen.getAllByText('706.4 ms').length).toBeGreaterThan(0)
  })
})

it('shortens token counts and preserves exact values in both summary and details', () => {
  const perf = loadFixture<LoadReportResponse>('report-real-single-sample').report_list[0].perf_metrics!
  perf.summary!.usage.total_input_tokens = 2146277
  perf.summary!.usage.total_output_tokens = 50164
  perf.summary!.usage.input_tokens.mean = 1000
  perf.summary!.usage.output_tokens.mean = 0
  render(
    <LocaleProvider>
      <PerfMetricsPanel perfMetrics={perf} compact />
    </LocaleProvider>,
  )
  for (const value of screen.getAllByText('2.15M')) expect(value).toHaveAttribute('title', '2,146,277 tokens')
  for (const value of screen.getAllByText('50.16K')) expect(value).toHaveAttribute('title', '50,164 tokens')
  const toggle = screen.getByText('Latency and token details')
  expect(toggle.closest('details')).toHaveAttribute('open')
  fireEvent.click(toggle)
  expect(toggle.closest('details')).not.toHaveAttribute('open')
  expect(screen.getByText('1K')).toHaveAttribute('title', '1,000 tokens')
  expect(screen.getAllByText('0').length).toBeGreaterThan(0)
})
