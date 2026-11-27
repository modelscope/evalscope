import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { loadFixture } from '@/test/loadFixture'
import type { DiagnosisReport, LoadReportResponse, PredictionsResponse, SampleDiagnosesResponse } from '@/api/types'
vi.mock('@/api/reports', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/api/reports')>()),
  loadReport: vi.fn(),
  getDiagnosis: vi.fn(),
  getDataFrame: vi.fn(),
  getPredictions: vi.fn(),
  getSampleDiagnoses: vi.fn(),
}))
import * as api from '@/api/reports'
import ReportDetailPage from './ReportDetailPage'
const REPORT = loadFixture<LoadReportResponse>('report-multi-dataset')
const PREDICTIONS = loadFixture<PredictionsResponse>('predictions-tool-trace')
const DIAGNOSIS: DiagnosisReport = {
  schema_version: 7,
  analyzer_version: '1',
  taxonomy_version: '1',
  run_dir: '/outputs/run',
  model: 'm',
  benchmark: 'gsm8k',
  level: 'facts',
  diagnosis_model: null,
  self_evaluation: false,
  narrative: null,
  narrative_samples: {}, narrative_status: 'not_requested',
  generated_at: '2026-11-17T00:00:00Z',
  config_fingerprint: 'c',
  data_fingerprint: 'd',
  samples_sha256: 's',
  coverage: {
    samples: 2,
    task_failed: 1,
    task_success_known: 2,
    outcome_unknown: 0,
    excluded: 0,
    errored_without_record: 0,
    unreadable: 0,
    unexplained_failures: 0,
    risk_samples: 0,
    proposals_without_rule_clues: 0,
    skipped_facts: {},
    analysis_errors: {},
    selection_scope: null,
    diagnosis_population: 1,
    selected: 0,
    diagnosed_ok: 0,
    diagnosis_statuses: {},
    citations_total: 0,
    citations_valid: 0,
  },
  cost: {
    cached_calls: 0,
    estimated_calls: 0,
    estimated_input_tokens: 0,
    usage_missing_calls: 0,
    calls: 0,
    input_tokens: 0,
    output_tokens: 0,
  },
  facts: [],
  limitations: [],
  patterns: [
    {
      failure_type: 'tool_error',
      failed: 1,
      unscored: 0,
      tools: {},
      examples: [{ subset: 'example', index: 7 }],
    },
  ],
}
const URL = '/reports/20260810_112700/test-model-a'
function Probe() {
  const location = useLocation(),
    navigate = useNavigate()
  return (
    <>
      <output data-testid="location">{location.search}</output>
      <output data-testid="pathname">{location.pathname}</output>
      <button onClick={() => navigate(-1)}>Browser back</button>
    </>
  )
}
async function settle() {
  for (let i = 0; i < 12; i++)
    await act(async () => {
      await Promise.resolve()
    })
}
async function renderDetail(search = '') {
  render(
    <LocaleProvider>
      <MemoryRouter initialEntries={[URL + search]}>
        <Probe />
        <Routes>
          <Route path="/reports/:runId/:modelId" element={<ReportDetailPage />} />
          <Route path="/reports" element={<p>Report list</p>} />
        </Routes>
      </MemoryRouter>
    </LocaleProvider>,
  )
  await settle()
}
function params() {
  return new URLSearchParams(screen.getByTestId('location').textContent ?? '')
}
function dataset(name: string) {
  return within(screen.getByRole('navigation', { name: 'Datasets' })).getByRole('button', { name: new RegExp(name) })
}
beforeEach(() => {
  vi.mocked(api.loadReport).mockResolvedValue(REPORT)
  vi.mocked(api.getDiagnosis).mockResolvedValue({ status: 'missing', report: null })
  vi.mocked(api.getDataFrame).mockImplementation(async (_root, _report, _view, name) => ({
    columns: ['Subset', 'Score', 'Num'],
    data: [{ Subset: name === 'arc_challenge' ? 'arc' : 'example', Score: 0.5, Num: 2 }],
  }))
  vi.mocked(api.getPredictions).mockResolvedValue({
    ...PREDICTIONS,
    predictions: [
      { ...PREDICTIONS.predictions[0], Index: '1' },
      { ...PREDICTIONS.predictions[0], Index: '7' },
    ],
  })
  vi.mocked(api.getSampleDiagnoses).mockImplementation(async (_root, _report, _dataset, filters) =>
    filters?.index !== undefined
      ? { status: 'missing', items: [], total: 0, page: 1, page_size: 200 }
      : ({
          status: 'ready',
          items: [{ key: { subset: 'example', index: 7 }, findings: [], task_success: false, score_status: 'success', llm_status: 'not_selected', safety_violation: null, stop_reason: null, needs_attention: true, observed_types: ['tool_error'], llm: null, sources: [], context_omitted: false, invalid_evidence: 0, recorded_error: null, evidence_errors: [] }],
          total: 1,
          page: 1,
          page_size: 200,
        } as SampleDiagnosesResponse),
  )
})
afterEach(() => {
  cleanup()
  vi.resetAllMocks()
  localStorage.removeItem('evalscope-locale')
})
describe('report workspace', () => {
  it.each(['', '?tab=overview', '?tab=details', '?tab=results'])('merges the old score tabs at %s', async (search) => {
    await renderDetail(search)
    expect(screen.getAllByRole('tab').map((tab) => tab.textContent)).toEqual(['Results', 'Diagnosis', 'Samples'])
    expect(screen.getByRole('tab', { name: 'Results', selected: true })).toBeInTheDocument()
    expect(screen.queryByText('Average Score')).not.toBeInTheDocument()
    expect(screen.getByText('Task Configuration').closest('details')).toHaveAttribute('open')
  })
  it('keeps dataset context outside all tabs', async () => {
    await renderDetail()
    expect(screen.getByRole('navigation', { name: 'Datasets' }).closest('[role="tabpanel"]')).toBeNull()
    const header = screen.getByRole('heading', { level: 1 })
    fireEvent.click(dataset('arc_challenge'))
    await settle()
    expect(screen.getByRole('tab', { name: 'Results', selected: true })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: 'Diagnosis' }))
    await settle()
    expect(dataset('arc_challenge')).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('heading', { level: 1 })).toBe(header)
    expect(api.getDiagnosis).toHaveBeenLastCalledWith(
      './outputs',
      '20260810_112700/test-model-a',
      'arc_challenge',
      expect.anything(),
    )
  })
  it('keeps benchmark selection in the shared navigator rather than duplicating header links', async () => {
    await renderDetail()
    expect(screen.queryByRole('button', { name: 'arc_challenge' })).not.toBeInTheDocument()
    expect(dataset('arc_challenge')).toBeInTheDocument()
  })
  it('clears old dataset filters but keeps the tab and root', async () => {
    await renderDetail(
      '?tab=predictions&dataset=gsm8k&subset=example&sample=7&pattern=P-a&failure_type=tool_error&step=0&root_path=/saved',
    )
    fireEvent.click(dataset('arc_challenge'))
    await settle()
    expect(params().get('tab')).toBe('predictions')
    expect(params().get('root_path')).toBe('/saved')
    for (const key of ['subset', 'sample', 'pattern', 'failure_type', 'step']) expect(params().has(key)).toBe(false)
    expect(api.getPredictions).toHaveBeenLastCalledWith(
      '/saved',
      '20260810_112700/test-model-a',
      'arc_challenge',
      'arc',
      expect.anything(),
    )
  })
  it('opens a pattern collection and restores diagnosis with browser back', async () => {
    vi.mocked(api.getDiagnosis).mockResolvedValue({ status: 'ready', report: DIAGNOSIS })
    await renderDetail('?tab=diagnosis&dataset=gsm8k')
    await settle()
    fireEvent.click(screen.getByRole('button', { name: 'Open in sample details' }))
    await settle()
    expect(params().get('failure_type')).toBe('tool_error')
    expect(params().get('outcome')).toBe('failed')
    expect(params().get('dataset')).toBe('gsm8k')
    expect(screen.getByText('Sample 1 / 1')).toBeInTheDocument()
    expect(screen.getByText('(index: 7)')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Browser back' }))
    await settle()
    expect(screen.getByRole('tab', { name: 'Diagnosis', selected: true })).toBeInTheDocument()
  })
  it('opens representative samples and preserves direct links', async () => {
    vi.mocked(api.getDiagnosis).mockResolvedValue({ status: 'ready', report: DIAGNOSIS })
    await renderDetail('?tab=diagnosis&dataset=gsm8k')
    await settle()
    fireEvent.click(screen.getByRole('button', { name: 'Open in sample details' }))
    await settle()
    expect(params().get('subset')).toBe('example')
    expect(params().get('sample')).toBe('7')
    expect(screen.getByText('(index: 7)')).toBeInTheDocument()
    const link = screen.getByTestId('location').textContent!
    cleanup()
    await renderDetail(link)
    expect(screen.getByText('(index: 7)')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Back to diagnosis' }))
    await settle()
    expect(screen.getByRole('tab', { name: 'Diagnosis', selected: true })).toBeInTheDocument()
  })
  it('restores the issue and the second sample from a diagnosis URL', async () => {
    vi.mocked(api.getDiagnosis).mockResolvedValue({ status: 'ready', report: DIAGNOSIS })
    const first = await api.getSampleDiagnoses('', '', '')
    vi.mocked(api.getSampleDiagnoses).mockResolvedValue({
      ...first, total: 2, items: [{ ...first.items[0], key: { subset: 'example', index: 1 } }, first.items[0]],
    })
    await renderDetail('?tab=diagnosis&dataset=gsm8k&failure_type=tool_error&outcome=failed&subset=example&sample=7&position=1')
    const panel = screen.getByRole('complementary', { name: 'Related sample' })
    expect(within(panel).getByText('example · Sample 7')).toBeInTheDocument()
    expect(within(panel).getByText('2 / 2')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Tool errors during execution/ })).toHaveAttribute('aria-pressed', 'true')
  })
  it('opens the exact cited sample and step, preserves it in Samples, and restores Diagnosis', async () => {
    const item = { key: { subset: 'example', index: 7 }, findings: [], task_success: false, score_status: 'success', llm_status: 'not_selected', safety_violation: null, stop_reason: null, needs_attention: true, observed_types: ['tool_error'], llm: null, sources: [], context_omitted: false, invalid_evidence: 0, recorded_error: null, evidence_errors: [] } as SampleDiagnosesResponse['items'][number]
    vi.mocked(api.getSampleDiagnoses).mockResolvedValue({ status: 'ready', items: [item], total: 1, page: 1, page_size: 200 })
    vi.mocked(api.getDiagnosis).mockResolvedValue({ status: 'ready', report: { ...DIAGNOSIS, level: 'diagnose',
      narrative_status: 'ok', narrative_samples: { 'S-exact': { sample: item.key, step: 0 } },
      narrative: { findings: [{ text: 'Inspect this sample.', refs: ['S-exact'] }], recommendations: [], limitations: [] },
    } })
    await renderDetail('?tab=diagnosis&dataset=gsm8k')
    fireEvent.click(screen.getByRole('button', { name: '[example · Sample 7 · Step 0]' }))
    await settle()
    expect(api.getSampleDiagnoses).toHaveBeenLastCalledWith('./outputs', '20260810_112700/test-model-a', 'gsm8k',
      { subset_name: 'example', index: 7 }, 1, expect.any(AbortSignal))
    expect(screen.getByRole('combobox', { name: 'Locate step' })).toHaveValue('0')
    fireEvent.click(screen.getByRole('button', { name: 'Open in sample details' }))
    await settle()
    expect(params().get('subset')).toBe('example')
    expect(params().get('sample')).toBe('7')
    expect(params().get('step')).toBe('0')
    expect(params().has('pattern')).toBe(false)
    expect(params().has('failure_type')).toBe(false)
    expect(screen.getByRole('navigation', { name: 'Sample contents' }).querySelector('[aria-current="step"]')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Browser back' }))
    await settle()
    expect(screen.getByRole('tab', { name: 'Diagnosis', selected: true })).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: 'Locate step' })).toHaveValue('0')
  })
  it('opens a subset without an old diagnostic filter', async () => {
    await renderDetail('?tab=results&dataset=gsm8k&pattern=old&sample=99')
    fireEvent.click(screen.getByRole('button', { name: 'example' }))
    await settle()
    expect(params().get('subset')).toBe('example')
    expect(params().has('pattern')).toBe(false)
    expect(params().has('sample')).toBe(false)
    expect(screen.getByText('Sample 1 / 2')).toBeInTheDocument()
  })
  it('hides single-dataset navigation and moves saved analysis to Diagnosis', async () => {
    vi.mocked(api.loadReport).mockResolvedValue({
      ...REPORT,
      datasets: ['gsm8k'],
      report_list: [{ ...REPORT.report_list[0], analysis: 'Saved analysis text' }],
    })
    await renderDetail()
    expect(screen.queryByRole('navigation', { name: 'Datasets' })).not.toBeInTheDocument()
    expect(screen.getByText('Saved analysis text')).not.toBeVisible()
    fireEvent.click(screen.getByRole('tab', { name: 'Diagnosis' }))
    await settle()
    expect(screen.getByText('Saved analysis text')).toBeInTheDocument()
  })
  it('rejects late responses from a previous dataset', async () => {
    let resolveOld!: (frame: { columns: string[]; data: Record<string, unknown>[] }) => void
    vi.mocked(api.getDataFrame).mockImplementation((_root, _report, _view, name) =>
      name === 'gsm8k'
        ? new Promise((resolve) => {
            resolveOld = resolve
          })
        : Promise.resolve({ columns: ['Subset'], data: [{ Subset: 'new-dataset' }] }),
    )
    await renderDetail()
    fireEvent.click(dataset('arc_challenge'))
    await settle()
    await act(async () => resolveOld({ columns: ['Subset'], data: [{ Subset: 'old-dataset' }] }))
    expect(screen.getByText('new-dataset')).toBeInTheDocument()
    expect(screen.queryByText('old-dataset')).not.toBeInTheDocument()
  })
  it('shows load failures', async () => {
    vi.mocked(api.loadReport).mockRejectedValue(new Error('boom'))
    await renderDetail()
    expect(screen.getByRole('alert')).toHaveTextContent('boom')
  })
  it('returns to the report list instead of undoing a dataset selection', async () => {
    await renderDetail('?root_path=/saved')
    fireEvent.click(dataset('arc_challenge'))
    await settle()
    fireEvent.click(screen.getByRole('button', { name: 'Back to Reports' }))
    await settle()
    expect(screen.getByTestId('pathname')).toHaveTextContent('/reports')
    expect(screen.getByText('Report list')).toBeInTheDocument()
    expect(params().get('root_path')).toBe('/saved')
  })
})
