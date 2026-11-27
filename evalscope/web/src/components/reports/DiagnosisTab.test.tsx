import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { LocaleProvider } from '@/contexts/LocaleContext'
import type { DiagnosisReport } from '@/api/types'
import { useState, type ComponentProps } from 'react'
import DiagnosisTabView from './DiagnosisTab'

function DiagnosisTab(props: Omit<ComponentProps<typeof DiagnosisTabView>, 'location' | 'onLocate'>) {
  const [location, setLocation] = useState<ComponentProps<typeof DiagnosisTabView>['location']>({ filters: {} })
  return <DiagnosisTabView {...props} location={location}
    onLocate={(filters, sample, step, position) => setLocation({ filters, sample, step, position })} />
}

const { getDiagnosis, getSampleDiagnoses, getPredictions, runDiagnosis } = vi.hoisted(() => ({ getDiagnosis: vi.fn(), getSampleDiagnoses: vi.fn(), getPredictions: vi.fn(), runDiagnosis: vi.fn() }))
vi.mock('@/api/reports', () => ({ getDiagnosis, getSampleDiagnoses, getPredictions, runDiagnosis }))
beforeEach(() => {
  getPredictions.mockResolvedValue({ predictions: [] })
  getSampleDiagnoses.mockResolvedValue({ status: 'ready', items: [], total: 0, page: 1, page_size: 200 })
})
afterEach(() => {
  cleanup()
  vi.resetAllMocks()
  localStorage.removeItem('evalscope-locale')
})

const report: DiagnosisReport = {
  narrative_samples: {},
  schema_version: 7,
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
    selection_scope: null, diagnosis_population: 0, selected: 0, diagnosed_ok: 0, diagnosis_statuses: {},     citations_total: 0, citations_valid: 0,
    samples: 3,
    task_failed: 1,
    task_success_known: 2,
    outcome_unknown: 1,
    excluded: 1,
    errored_without_record: 0,
    unreadable: 0,
    unexplained_failures: 1,
    risk_samples: 0,
    proposals_without_rule_clues: 0,
    skipped_facts: {},
    analysis_errors: {},
  },
  cost: { cached_calls: 0, estimated_calls: 0, estimated_input_tokens: 0, usage_missing_calls: 0, calls: 0, input_tokens: 0, output_tokens: 0 },
  facts: [],
  limitations: [],
  patterns: [
    {
      failure_type: 'tool_error',
      failed: 1,
      unscored: 0,
      tools: {},
      examples: [{ subset: 'default', index: 7 }],
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
  getDiagnosis.mockResolvedValue({ status: 'ready', report: { ...report, patterns: [
    { failure_type: 'tool_error', failed: 1, unscored: 0, tools: { search: 1 }, examples: [] },
    { failure_type: 'truncated', failed: 2, unscored: 0, tools: {}, examples: [] },
  ] } })
  render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await settle()
  const issues = screen.getByRole('region', { name: 'Problem list' })
  expect(screen.getByText('1 of 2 samples with known outcomes did not pass')).toBeInTheDocument()
  expect(within(issues).getAllByRole('button')[0]).toHaveTextContent('Output truncated')
  expect(within(issues).getAllByRole('button')[0]).toHaveAttribute('aria-pressed', 'true')
  expect(getSampleDiagnoses).toHaveBeenLastCalledWith('/outputs', 'run/m', 'demo', { failure_type: 'truncated', outcome: 'failed' }, 1, expect.any(AbortSignal))
  expect(screen.getByRole('complementary', { name: 'Related sample' })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Close sample' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'View example' })).not.toBeInTheDocument()
  expect(screen.getByText('Analysis information', { selector: 'summary span' }).closest('details')).not.toHaveAttribute('open')
  fireEvent.click(within(issues).getByRole('button', { name: /Tool errors during execution/ }))
  await settle()
  expect(getSampleDiagnoses).toHaveBeenLastCalledWith('/outputs', 'run/m', 'demo', { failure_type: 'tool_error', outcome: 'failed' }, 1, expect.any(AbortSignal))
  expect(screen.getByText(/Tools involved/)).toHaveTextContent('search')
  expect(screen.queryByRole('img', { name: /Model:/ })).not.toBeInTheDocument()
  expect(within(issues).queryByText('Tools returned errors during execution. Their effect on the final result still needs review.')).not.toBeInTheDocument()
})

it('keeps facts information compact and excludes statistics dumps, inapplicable items and model fields', async () => {
  localStorage.setItem('evalscope-locale', 'zh')
  getDiagnosis.mockResolvedValue({ status: 'ready', report: {
    ...report, coverage: { ...report.coverage, skipped_facts: { 'F-agent': 'No trace.' } },
    facts: [{ id: 'F-truncation', kind: 'generation', severity: 'warn', title: 'Developer title', value: 0.5,
      unit: 'fraction', definition: 'internal.stop_reason', numerator: 1, denominator: 2, samples: [], data: {} }],
  } })
  const { container } = render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await settle()
  const info = screen.getByText('分析信息', { selector: 'summary span' }).closest('details')!
  expect(info).not.toHaveAttribute('open')
  fireEvent.click(screen.getByRole('link', { name: '分析信息' }))
  expect(info).toHaveAttribute('open')
  expect(within(info).getByText('基础分析')).toBeInTheDocument()
  expect(within(info).getByText('3 条样本')).toBeInTheDocument()
  expect(within(info).getByText('未调用模型')).toBeInTheDocument()
  expect(within(info).queryByText('Token 用量')).not.toBeInTheDocument()
  expect(within(info).queryByText('分析模型')).not.toBeInTheDocument()
  expect(within(info).queryByText('总结状态')).not.toBeInTheDocument()
  expect(container.textContent).not.toMatch(/internal.stop_reason|F-agent|Developer title|没有工具调用记录/)
  expect(screen.queryByText(/达到输出上限/)).not.toBeInTheDocument()
})

it.each([true, false])('distinguishes all-pass and unavailable outcomes without an empty evidence panel (%s)', async (known) => {
  getDiagnosis.mockResolvedValue({ status: 'ready', report: { ...report, patterns: [], coverage: {
    ...report.coverage, task_failed: 0, task_success_known: known ? 3 : 0, outcome_unknown: known ? 0 : 3,
    unexplained_failures: 0, excluded: 0,
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
    ...report.coverage, task_success_known: 3, outcome_unknown: 0, excluded: 0, proposals_without_rule_clues: 1,
  } } })
  render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await settle()
  expect(screen.getByRole('button', { name: /Other failed samples/ })).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByText('0 have record anomalies; 1 are other failed samples.')).toBeInTheDocument()
  expect(getSampleDiagnoses).toHaveBeenLastCalledWith('/outputs', 'run/m', 'demo', { failure_type: 'unexplained', outcome: 'failed' }, 1, expect.any(AbortSignal))
  expect(screen.getByText('Model proposals are available for 1 of these samples.')).toBeInTheDocument()
  expect(screen.queryByText('Reasoning or calculation error')).not.toBeInTheDocument()
})

it('keeps risk and unscored cases separate and removes the passed-observation feature', async () => {
  getDiagnosis.mockResolvedValue({ status: 'ready', report: { ...report, coverage: { ...report.coverage, risk_samples: 2 } } })
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
  expect(screen.queryByRole('button', { name: /process observations.*passed/i })).not.toBeInTheDocument()
})

it('keeps model analysis, references and cached usage accessible', async () => {
  getDiagnosis.mockResolvedValue({ status: 'ready', report: { ...report, level: 'diagnose', narrative_status: 'ok', cost: { ...report.cost, cached_calls: 16 },
    narrative_samples: { 'S-a': { sample: { subset: '跨子集', index: 42 }, step: 3 } },
    narrative: { findings: [{ text: 'Inspect the tool reply.', refs: ['S-a'] }], recommendations: [] },
  } })
  render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await settle()
  const info = screen.getByText('Analysis information', { selector: 'summary span' }).closest('details')!
  expect(within(info).getByText('16')).toBeInTheDocument()
  expect(within(info).getByText('No new token usage')).toBeInTheDocument()
  expect(screen.getByRole('region', { name: 'Model summary and suggestions' })).toHaveTextContent('Inspect the tool reply.')
  const workspace = document.getElementById('diagnosis-issues')!
  const scroll = vi.fn()
  workspace.scrollIntoView = scroll
  fireEvent.click(screen.getByRole('button', { name: '[跨子集 · Sample 42 · Step 3]' }))
  expect(scroll).toHaveBeenCalledWith({ block: 'start', behavior: 'smooth' })
  await settle()
  expect(getSampleDiagnoses).toHaveBeenLastCalledWith('/outputs', 'run/m', 'demo', { subset_name: '跨子集', index: 42 }, 1, expect.any(AbortSignal))
  expect(screen.queryByRole('button', { name: '[Other failed samples]' })).not.toBeInTheDocument()
})


it('shows model-analysis scope, citation counts, usage and summary status', async () => {
  getDiagnosis.mockResolvedValue({ status: 'ready', report: { ...report, level: 'diagnose', diagnosis_model: 'reviewer',
    coverage: { ...report.coverage, selection_scope: 'failed', selected: 2, diagnosis_population: 5,
      diagnosed_ok: 1, citations_valid: 3, citations_total: 4, diagnosis_statuses: { ok: 1, invalid_evidence: 1 } },
    cost: { ...report.cost, calls: 2, input_tokens: 1500000, output_tokens: 2300 },
  } })
  render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await settle()
  const info = screen.getByText('Analysis information', { selector: 'summary span' }).closest('details')!
  fireEvent.click(screen.getByRole('link', { name: 'Analysis information' }))
  expect(within(info).getByText('Model analysis')).toBeInTheDocument()
  expect(within(info).getByText('Failed tasks')).toBeInTheDocument()
  expect(within(info).getByText('reviewer')).toBeInTheDocument()
  expect(within(info).getByText('2 / 5')).toBeInTheDocument()
  expect(within(info).getByText('3 / 4')).toBeInTheDocument()
  expect(within(info).getByText('Input 1.5M · output 2.3K')).toHaveAttribute('title', 'Input tokens: 1,500,000 · Output tokens: 2,300')
  expect(screen.queryByRole('region', { name: 'Model summary and suggestions' })).not.toBeInTheDocument()
  expect(within(info).getByText('Summary status')).toBeInTheDocument()
  expect(within(info).getByText(/Samples with unknown outcomes are excluded/)).toBeInTheDocument()
})

it.each([
  ['no_findings', 'Nothing to summarize'],
  ['transport_error', 'Summary request failed'],
  ['parse_error', 'Invalid summary response'],
  ['invalid_references', 'Summary references unavailable'],
  ['not_requested', 'Not generated'],
])('shows the correct model-summary state without a fake narrative (%s)', async (status, text) => {
  getDiagnosis.mockResolvedValue({ status: 'ready', report: { ...report, level: 'diagnose', narrative_status: status } })
  render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await settle()
  const info = screen.getByText('Analysis information', { selector: 'summary span' }).closest('details')!
  expect(within(info).getAllByText(text).length).toBeGreaterThan(0)
  expect(within(info).getByText('No citations to check')).toBeInTheDocument()
  expect(screen.queryByRole('region', { name: 'Model summary and suggestions' })).not.toBeInTheDocument()
})

it('keeps model findings and recommendations distinct and opens only the cited statistic', async () => {
  getDiagnosis.mockResolvedValue({ status: 'ready', report: { ...report, level: 'diagnose', narrative_status: 'partial_references',
    narrative: { findings: [{ text: 'One answer was cut off.', refs: ['F-truncation'] }],
      recommendations: [{ text: 'Check the output limit.', refs: ['F-truncation'] }], limitations: [] },
    facts: [
      { id: 'F-truncation', kind: 'generation', severity: 'warn', title: 'Truncation', value: 0.5, unit: 'fraction',
        definition: 'Known budget endings.', numerator: 1, denominator: 2, samples: [], data: {} },
      { id: 'F-perf', kind: 'perf', severity: 'info', title: 'Hidden performance dump', value: null, unit: '',
        definition: 'Not cited.', numerator: null, denominator: null, samples: [], data: {} },
    ],
  } })
  render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await settle()
  const summary = screen.getByRole('region', { name: 'Model summary and suggestions' })
  expect(within(summary).getByRole('heading', { name: 'Observations' })).toBeInTheDocument()
  expect(within(summary).getByRole('heading', { name: 'Suggested next checks' })).toBeInTheDocument()
  expect(screen.queryByRole('complementary', { name: 'Referenced statistic' })).not.toBeInTheDocument()
  fireEvent.click(within(summary).getAllByRole('button', { name: '[Output limit reached]' })[0])
  expect(screen.getByRole('complementary', { name: 'Referenced statistic' })).toHaveTextContent('1 / 2')
  expect(screen.queryByText('Hidden performance dump')).not.toBeInTheDocument()
})

it('does not present missing provider usage as zero tokens', async () => {
  getDiagnosis.mockResolvedValue({ status: 'ready', report: { ...report, level: 'diagnose',
    cost: { ...report.cost, calls: 1, usage_missing_calls: 1 } } })
  render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await settle()
  expect(screen.getByText('Some usage was not returned')).toBeInTheDocument()
  expect(screen.queryByText('Input 0 · output 0')).not.toBeInTheDocument()
})

it('runs the free analysis for a missing report and shows the returned diagnostics', async () => {
  getDiagnosis.mockResolvedValue({ status: 'missing', report: null })
  runDiagnosis.mockResolvedValue({ status: 'ready', report })
  render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await settle()
  fireEvent.click(screen.getByRole('button', { name: 'Generate basic analysis' }))
  await settle()
  expect(runDiagnosis).toHaveBeenCalledWith('/outputs', 'run/m', 'demo', { level: 'facts' })
  expect(screen.getByRole('region', { name: 'Problem list' })).toBeInTheDocument()
  expect(screen.getByText('Set up model analysis')).toBeInTheDocument()
})
