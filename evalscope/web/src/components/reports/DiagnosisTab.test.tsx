import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { LocaleProvider } from '@/contexts/LocaleContext'
import type { DiagnosisReport } from '@/api/types'
import DiagnosisTab from './DiagnosisTab'

const { getDiagnosis, getSampleDiagnoses, getPredictions } = vi.hoisted(() => ({ getDiagnosis: vi.fn(), getSampleDiagnoses: vi.fn(), getPredictions: vi.fn() }))
vi.mock('@/api/reports', () => ({ getDiagnosis, getSampleDiagnoses, getPredictions }))
beforeEach(() => {
  getSampleDiagnoses.mockResolvedValue({ status: 'ready', items: [], total: 0, page: 1, page_size: 200 })
})
afterEach(() => {
  cleanup()
  vi.resetAllMocks()
  localStorage.removeItem('evalscope-locale')
})

const report: DiagnosisReport = {
  schema_version: 1,
  analyzer_version: '1',
  taxonomy_version: '1',
  run_dir: '/outputs/run',
  model: 'm',
  benchmark: 'demo',
  level: 'facts',
  diagnosis_model: null, self_evaluation: false, narrative: null, narrative_status: 'not_requested',
  generated_at: '2026-11-16T00:00:00Z',
  config_fingerprint: 'c',
  data_fingerprint: 'd',
  samples_sha256: 's',
  coverage: {
    diagnosis_population: 0, selected: 0, diagnosed_ok: 0, diagnosis_statuses: {}, strata: [], failed_unexplained_estimate: null, rule_reviews: 0, rule_agreements: 0, citations_total: 0, citations_valid: 0,
    samples: 3,
    task_failed: 1,
    task_success_known: 2,
    outcome_unknown: 1,
    excluded: 1,
    errored_without_record: 0,
    unreadable: 0,
    needs_attention: 2,
    unexplained_failures: 1,
    skipped_facts: {},
    analysis_errors: {},
  },
  cost: { cached_calls: 0, estimated_calls: 0, estimated_input_tokens: 0, usage_missing_calls: 0, calls: 0, input_tokens: 0, output_tokens: 0 },
  facts: [],
  limitations: [],
  patterns: [
    {
      id: 'P-a',
      failure_type: 'tool_error',
      attribution: 'unknown',
      detail: '',
      count: 1,
      denominator: 2,
      share: 0.5,
      by_subset: { default: 1 },
      examples: [{ subset: 'default', index: 7 }],
      source: 'rule', estimated_share: null,
    },
  ],
}

it('offers a free, quoted command when diagnostics are missing or stale', async () => {
  getDiagnosis.mockResolvedValue({ status: 'stale', report })
  render(
    <LocaleProvider>
      <DiagnosisTab
        rootPath="/outputs with spaces"
        reportName="run/m"
        datasetName="demo"
        onSamples={vi.fn()}
      />
    </LocaleProvider>,
  )
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })
  expect(screen.getByRole('status')).toHaveTextContent('outdated')
  expect(screen.getByText(/evalscope analyze/)).toHaveTextContent("--run-dir '/outputs with spaces/run'")
  expect(screen.queryByRole('button', { name: 'Sample 7' })).not.toBeInTheDocument()
})

async function settle() {
  await act(async () => { for (let i = 0; i < 8; i++) await Promise.resolve() })
}

it('selects the largest failed issue immediately and reuses the persistent sample panel', async () => {
  getDiagnosis.mockResolvedValue({ status: 'ready', report, issues: [
    { failure_type: 'tool_error', failed: 1, unscored: 0, tools: { search: 1 } },
    { failure_type: 'truncated', failed: 2, unscored: 0, tools: {} },
  ] })
  render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await settle()
  const issues = screen.getByRole('region', { name: 'Problem list' })
  expect(within(issues).getAllByRole('button')[0]).toHaveTextContent('Output truncated')
  expect(within(issues).getAllByRole('button')[0]).toHaveAttribute('aria-pressed', 'true')
  expect(getSampleDiagnoses).toHaveBeenLastCalledWith('/outputs', 'run/m', 'demo', { failure_type: 'truncated', outcome: 'failed' }, 1, expect.any(AbortSignal))
  expect(screen.getByRole('complementary', { name: 'Related sample' })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Close sample' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'View example' })).not.toBeInTheDocument()
  expect(screen.getByText('Analysis record', { selector: 'summary' }).closest('details')).not.toHaveAttribute('open')
  fireEvent.click(within(issues).getByRole('button', { name: /Tool errors during execution/ }))
  await settle()
  expect(getSampleDiagnoses).toHaveBeenLastCalledWith('/outputs', 'run/m', 'demo', { failure_type: 'tool_error', outcome: 'failed' }, 1, expect.any(AbortSignal))
  expect(screen.getByText(/Tools involved/)).toHaveTextContent('search')
  expect(screen.queryByRole('img', { name: /Model:/ })).not.toBeInTheDocument()
})

it('keeps readable analysis records available without exposing them as primary issues', async () => {
  localStorage.setItem('evalscope-locale', 'zh')
  getDiagnosis.mockResolvedValue({ status: 'ready', report: {
    ...report, coverage: { ...report.coverage, skipped_facts: { 'F-agent': 'No trace.' } },
    facts: [{ id: 'F-truncation', kind: 'generation', severity: 'warn', title: 'Developer title', value: 0.5,
      unit: 'fraction', definition: 'internal.stop_reason', numerator: 1, denominator: 2, samples: [], data: {} }],
  } })
  const { container } = render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await settle()
  const record = screen.getByText('分析记录', { selector: 'summary' }).closest('details')!
  expect(record).not.toHaveAttribute('open')
  fireEvent.click(screen.getByRole('link', { name: '分析记录' }))
  expect(record).toHaveAttribute('open')
  expect(screen.getByText(/回答被截断/)).toBeInTheDocument()
  expect(screen.getByText('没有工具调用记录')).toBeInTheDocument()
  expect(container.textContent).not.toMatch(/internal.stop_reason|F-agent|Developer title/)
  expect(screen.queryByText('相关环节')).not.toBeInTheDocument()
  expect(screen.queryByText('统计与分析详情')).not.toBeInTheDocument()
})

it.each([true, false])('distinguishes all-pass and unavailable outcomes without an empty evidence panel (%s)', async (known) => {
  getDiagnosis.mockResolvedValue({ status: 'ready', report: { ...report, patterns: [], coverage: {
    ...report.coverage, task_failed: 0, task_success_known: known ? 3 : 0, outcome_unknown: known ? 0 : 3,
    unexplained_failures: 0, needs_attention: 0, excluded: 0,
  } } })
  render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await settle()
  expect(screen.getAllByText(known ? 'No failed samples' : 'Task outcomes unavailable').length).toBeGreaterThan(0)
  expect(screen.queryByRole('complementary')).not.toBeInTheDocument()
  expect(getSampleDiagnoses).not.toHaveBeenCalled()
  if (!known) expect(screen.getByText('Analyzed 3 samples; task outcomes cannot currently be determined.')).toBeInTheDocument()
})

it('provides an immediate unexplained-case entry instead of inventing an ordinary QA cause', async () => {
  getDiagnosis.mockResolvedValue({ status: 'ready', report: { ...report, patterns: [], coverage: {
    ...report.coverage, task_success_known: 3, outcome_unknown: 0, excluded: 0,
  } }, issues: [{ failure_type: 'unexplained', failed: 1 }] })
  render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await settle()
  expect(screen.getByRole('button', { name: /No explanation yet/ })).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByText('0 have diagnostic clues; 1 still have no explanation.')).toBeInTheDocument()
  expect(getSampleDiagnoses).toHaveBeenLastCalledWith('/outputs', 'run/m', 'demo', { failure_type: 'unexplained', outcome: 'failed' }, 1, expect.any(AbortSignal))
  expect(screen.queryByText('Reasoning or calculation error')).not.toBeInTheDocument()
})

it('keeps risk, unscored and passed observations separate from failed issue counts', async () => {
  getDiagnosis.mockResolvedValue({ status: 'ready', report, risk_samples: 2, passed_observations: 3, issues: [
    { failure_type: 'tool_error', failed: 1, unscored: 0, tools: {} },
    { failure_type: 'unexplained', failed: 22, unscored: 0, tools: {} },
    { failure_type: 'judge_failure', failed: 0, unscored: 1, tools: {} },
  ] })
  render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await settle()
  const main = screen.getByRole('region', { name: 'Problem list' })
  expect(within(main).queryByText('22 samples')).not.toBeInTheDocument()
  expect(within(main).queryByText('Scoring unavailable')).not.toBeInTheDocument()
  fireEvent.click(within(main).getByRole('button', { name: /Not scored/ }))
  await settle()
  expect(getSampleDiagnoses).toHaveBeenLastCalledWith('/outputs', 'run/m', 'demo', { outcome: 'unscored' }, 1, expect.any(AbortSignal))
  fireEvent.click(screen.getByRole('button', { name: 'Safety signals in 2 samples' }))
  await settle()
  expect(getSampleDiagnoses).toHaveBeenLastCalledWith('/outputs', 'run/m', 'demo', { outcome: 'risk' }, 1, expect.any(AbortSignal))
  fireEvent.click(screen.getByRole('button', { name: 'View process observations in 3 passed samples' }))
  await settle()
  expect(getSampleDiagnoses).toHaveBeenLastCalledWith('/outputs', 'run/m', 'demo', { outcome: 'passed' }, 1, expect.any(AbortSignal))
})

it('keeps model analysis, references and cached usage accessible', async () => {
  getDiagnosis.mockResolvedValue({ status: 'ready', report: { ...report, level: 'full', cost: { ...report.cost, cached_calls: 16 },
    narrative: { findings: [{ text: 'Inspect the tool reply.', refs: ['P-a'] }], recommendations: [] },
  } })
  render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await settle()
  expect(screen.getByText(/16 cached analyses reused/)).toBeInTheDocument()
  expect(screen.getByText(/0 new calls/)).toBeInTheDocument()
  expect(screen.getByRole('region', { name: 'Model summary and suggestions' })).toHaveTextContent('Inspect the tool reply.')
  fireEvent.click(screen.getByRole('button', { name: '[P1]' }))
  await settle()
  expect(getSampleDiagnoses).toHaveBeenLastCalledWith('/outputs', 'run/m', 'demo', { pattern_id: 'P-a' }, 1, expect.any(AbortSignal))
})
