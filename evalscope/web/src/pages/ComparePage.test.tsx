import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import type { LoadReportResponse, ReportData } from '@/api/types'
import { LocaleProvider } from '@/contexts/LocaleContext'

const useScanMock = vi.hoisted(() => vi.fn())
const useReportCacheMock = vi.hoisted(() => vi.fn())

vi.mock('@/contexts/ReportsContext', () => ({
  useScan: useScanMock,
  useReportCache: useReportCacheMock,
}))

vi.mock('@/components/charts/PlotlyChart', () => ({
  default: ({ src, title }: { src: string; title?: string }) => (
    <div data-testid="comparison-chart" data-src={src}>{title}</div>
  ),
}))

vi.mock('@/components/chat/ChatView', () => ({
  default: () => <div>Prediction</div>,
}))

import ComparePage from './ComparePage'

const runNames = [
  '20260811_144944/Qwen3-Max-Instruct',
  '20260811_143817/Qwen3-235B-A22B-Thinking-2507',
  '20260811_141102/DeepSeek-R1-Distill-Qwen-32B',
  '20260811_140000/Llama-3.1-70B-Instruct',
]

function makeReport(
  runName: string,
  score: number,
  direction: 'higher_is_better' | 'lower_is_better' = 'higher_is_better',
): ReportData {
  const modelName = runName.split('/')[1]
  const identity = { name: direction === 'lower_is_better' ? 'wer' : 'accuracy', aggregation: 'mean', dimensions: {} }
  return {
    schema_version: 2,
    name: `${modelName}@gsm8k`,
    dataset_name: 'gsm8k',
    model_name: modelName,
    analysis: '',
    metrics: [{
      identity,
      num: 100,
      score,
      categories: [],
      semantics: {
        semantic_id: direction === 'lower_is_better' ? 'quality.wer.ratio' : 'quality.accuracy.ratio',
        metric_name: direction === 'lower_is_better' ? 'WER' : 'Accuracy',
        kind: 'quality',
        direction,
        value_range: { min: 0, max: 1 },
        display_kind: 'percent',
        display_multiplier: 100,
        display_unit: '%',
        display_precision: 1,
      },
    }],
    primary_metric_identity: identity,
  }
}

const reports = [0.8, 0.9, 0.7, 0.85].map((score, index) => ({
  ...makeReport(runNames[index], score),
  _reportRef: runNames[index],
}))

const reportCache = Object.fromEntries(
  reports.map((report) => [report._reportRef, {
    report_list: [report],
    datasets: ['gsm8k'],
    task_config: {},
  } satisfies LoadReportResponse]),
)

async function renderPage() {
  const params = new URLSearchParams()
  for (const ref of runNames) params.append('report', ref)
  params.set('root_path', 'outputs')
  const view = render(
    <MemoryRouter initialEntries={[`/compare?${params.toString()}`]}>
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
  useScanMock.mockReturnValue({ rootPath: 'outputs', scanToken: 0, setRootPath: vi.fn(), triggerScan: vi.fn() })
  useReportCacheMock.mockReturnValue({
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
    expect(chartUrl.pathname.endsWith('/charts/radar')).toBe(true)
    expect(chartUrl.searchParams.getAll('report')).toEqual(runNames)
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

  it('colors a lower-is-better decrease as an improvement', async () => {
    const lowerReports = [0.8, 0.7, 0.9, 0.75].map((score, index) => ({
      ...makeReport(runNames[index], score, 'lower_is_better'),
      _reportRef: runNames[index],
    }))
    useReportCacheMock.mockReturnValue({
      loadMultiReports: vi.fn(async () => lowerReports),
      loading: false,
      reportCache: Object.fromEntries(lowerReports.map((report) => [report._reportRef, {
        report_list: [report],
        datasets: ['gsm8k'],
        task_config: {},
      }])),
    })

    await renderPage()

    const improvedDelta = screen.getAllByText('-10 pp')[0]
    expect(improvedDelta).toHaveClass('text-[var(--success)]')
    expect(improvedDelta.closest('td')?.getAttribute('style')).toContain('var(--success)')
  })

  it('separates different primary metrics for the same dataset', async () => {
    const mixedReports = [
      { ...makeReport(runNames[0], 0.8), _reportRef: runNames[0] },
      { ...makeReport(runNames[1], 0.9), _reportRef: runNames[1] },
      { ...makeReport(runNames[2], 0.2, 'lower_is_better'), _reportRef: runNames[2] },
      { ...makeReport(runNames[3], 0.1, 'lower_is_better'), _reportRef: runNames[3] },
    ]
    useReportCacheMock.mockReturnValue({
      loadMultiReports: vi.fn(async () => mixedReports),
      loading: false,
      reportCache: Object.fromEntries(mixedReports.map((report) => [report._reportRef, {
        report_list: [report],
        datasets: ['gsm8k'],
        task_config: {},
      }])),
    })

    await renderPage()

    expect(screen.getByRole('columnheader', { name: /gsm8k.*Accuracy ↑/ })).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: /gsm8k.*WER ↓/ })).toBeInTheDocument()
    expect(screen.getByText('20%')).toBeInTheDocument()
    expect(screen.getByText('10%')).toBeInTheDocument()
    expect(screen.queryByText(/80 pp/)).not.toBeInTheDocument()
    expect(screen.queryByText('0 pp')).not.toBeInTheDocument()
  })
})
