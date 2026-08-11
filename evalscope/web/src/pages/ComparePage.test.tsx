import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import type { LoadReportResponse, ReportData } from '@/api/types'
import { LocaleProvider } from '@/contexts/LocaleContext'

const useReportsMock = vi.hoisted(() => vi.fn())

vi.mock('@/contexts/ReportsContext', () => ({
  useReports: useReportsMock,
}))

vi.mock('@/components/charts/PlotlyChart', () => ({
  default: ({ src, title }: { src: string; title?: string }) => (
    <div data-testid="comparison-chart" data-src={src}>{title}</div>
  ),
}))

vi.mock('@/components/single/ChatView', () => ({
  default: () => <div>Prediction</div>,
}))

import ComparePage from './ComparePage'

const runNames = [
  '20260811_144944@@Qwen3-Max-Instruct::gsm8k',
  '20260811_143817@@Qwen3-235B-A22B-Thinking-2507::gsm8k',
  '20260811_141102@@DeepSeek-R1-Distill-Qwen-32B::gsm8k',
  '20260811_140000@@Llama-3.1-70B-Instruct::gsm8k',
]

function makeReport(runName: string, score: number): ReportData {
  const modelName = runName.split('@@')[1].split('::')[0]
  const identity = { name: 'accuracy', aggregation: 'mean', dimensions: {} }
  return {
    schema_version: 2,
    name: runName,
    dataset_name: 'gsm8k',
    model_name: modelName,
    analysis: '',
    metrics: [{
      identity,
      num: 100,
      score,
      categories: [],
      semantics: {
        semantic_id: 'quality.accuracy.ratio',
        metric_name: 'Accuracy',
        role: 'primary',
        direction: 'higher_is_better',
        value_range: { min: 0, max: 1 },
        display_kind: 'percent',
        display_multiplier: 100,
        display_unit: '%',
        display_precision: 1,
        contract_version: 1,
      },
    }],
    primary_metric_identity: identity,
  }
}

const reports = [0.8, 0.9, 0.7, 0.85].map((score, index) => ({
  ...makeReport(runNames[index], score),
  _reportName: runNames[index],
}))

const reportCache = Object.fromEntries(
  reports.map((report) => [report._reportName, {
    report_list: [report],
    datasets: ['gsm8k'],
    task_config: {},
  } satisfies LoadReportResponse]),
)

async function renderPage() {
  const reportsParam = encodeURIComponent(runNames.join(';'))
  const view = render(
    <MemoryRouter initialEntries={[`/compare?reports=${reportsParam}&root_path=outputs`]}>
      <LocaleProvider>
        <ComparePage />
      </LocaleProvider>
    </MemoryRouter>,
  )
  await act(async () => {
    await Promise.resolve()
  })
  return view
}

beforeEach(() => {
  localStorage.setItem('evalscope-locale', 'en')
  useReportsMock.mockReturnValue({
    rootPath: 'outputs',
    setRootPath: vi.fn(),
    loadMultiReports: vi.fn(async () => reports),
    loading: false,
    reportCache,
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  localStorage.clear()
})

describe('ComparePage', () => {
  it('keeps every selected report in score comparison and always requests radar', async () => {
    await renderPage()

    expect(screen.getByRole('tab', { name: 'Score Comparison · 4 reports' })).toBeInTheDocument()
    const chart = screen.getByTestId('comparison-chart')
    const chartUrl = new URL(chart.getAttribute('data-src')!, 'http://localhost')
    expect(chartUrl.searchParams.get('chart_type')).toBe('radar')
    expect(chartUrl.searchParams.get('report_names')?.split(';')).toEqual(runNames)
    expect(screen.queryByRole('columnheader', { name: 'Average' })).not.toBeInTheDocument()
  })

  it('lets users choose a separate maximum of three reports for prediction comparison', async () => {
    await renderPage()
    fireEvent.click(screen.getByRole('tab', { name: 'Prediction Comparison · choose up to 3' }))

    const controls = screen.getByTestId('prediction-controls')
    expect(controls).toContainElement(screen.getByLabelText('Select Dataset'))
    expect(controls).toContainElement(screen.getByText('Filter'))
    expect(screen.getByTestId('prediction-model-filters').children).toHaveLength(3)
    expect(screen.getAllByRole('group', { name: /^Filter: / })).toHaveLength(3)
    expect(screen.getByRole('button', { name: 'All' })).toHaveAttribute('aria-pressed', 'true')

    fireEvent.click(screen.getByRole('button', { name: 'All Below' }))
    expect(screen.getByRole('button', { name: 'All Below' })).toHaveAttribute('aria-pressed', 'true')
    screen.getAllByRole('button', { name: 'Below filter' }).forEach((button) => {
      expect(button).toHaveAttribute('aria-pressed', 'true')
    })

    const first = screen.getByRole('checkbox', { name: /Prediction Comparison: Qwen3-Max-Instruct/ })
    const fourth = screen.getByRole('checkbox', { name: /Prediction Comparison: Llama-3.1-70B-Instruct/ })
    const slotStyle = (checkbox: HTMLElement) => (
      checkbox.querySelector<HTMLElement>('span[aria-hidden="true"][style*="--compare-"]')?.getAttribute('style') ?? ''
    )
    expect(first).toHaveAttribute('aria-checked', 'true')
    expect(fourth).toBeDisabled()
    expect(slotStyle(first)).toContain('var(--compare-0-dot)')
    expect(slotStyle(fourth)).toBe('')

    fireEvent.click(first)
    expect(fourth).toBeEnabled()
    fireEvent.click(fourth)

    expect(first).toHaveAttribute('aria-checked', 'false')
    expect(fourth).toHaveAttribute('aria-checked', 'true')
    expect(slotStyle(first)).toBe('')
    expect(slotStyle(fourth)).toContain('var(--compare-2-dot)')
    expect(screen.getAllByRole('checkbox').filter((checkbox) => checkbox.getAttribute('aria-checked') === 'true')).toHaveLength(3)
  })

  it('shows full model labels and recomputes signed baseline deltas', async () => {
    await renderPage()

    const longLabel = 'Qwen3-235B-A22B-Thinking-2507 · gsm8k'
    expect(screen.getAllByTitle(longLabel).length).toBeGreaterThan(0)
    const positiveDelta = screen.getAllByText('+10 pp')[0]
    expect(positiveDelta.closest('td')?.getAttribute('style')).toContain('var(--success)')
    expect(screen.getByText('Baseline mode')).toBeInTheDocument()
    expect(screen.getByLabelText('Higher is better')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('Baseline'), { target: { value: runNames[1] } })
    expect(screen.getAllByText('-10 pp').length).toBeGreaterThan(0)

    fireEvent.click(screen.getByRole('button', { name: 'Absolute' }))
    expect(screen.getByLabelText('Baseline')).toBeDisabled()
  })
})
