import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { LocaleProvider } from '@/contexts/LocaleContext'
import type { DiagnosisReport } from '@/api/types'
import DiagnosisTab from './DiagnosisTab'

const { getDiagnosis } = vi.hoisted(() => ({ getDiagnosis: vi.fn() }))
vi.mock('@/api/reports', () => ({ getDiagnosis }))
afterEach(() => {
  cleanup()
  getDiagnosis.mockReset()
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

it('shows denominators and opens the precise example or heatmap subset', async () => {
  getDiagnosis.mockResolvedValue({ status: 'ready', report })
  const onSamples = vi.fn()
  render(
    <LocaleProvider>
      <DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={onSamples} />
    </LocaleProvider>,
  )
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })
  expect(screen.getByText('1 / 2')).toBeInTheDocument()
  expect(screen.getByText('Issues to review · 2 samples')).toBeInTheDocument()
  expect(screen.getByRole('progressbar', { name: 'Tool error' })).toHaveAttribute('aria-valuenow', '50')
  fireEvent.click(screen.getByRole('button', { name: 'Sample 7' }))
  expect(onSamples).toHaveBeenLastCalledWith({ pattern_id: 'P-a' }, { subset: 'default', index: 7 })
  expect(onSamples).toHaveBeenCalledTimes(1)
  for (const name of ['Task failure rate', 'Rule coverage', 'Analysis usage']) {
    expect(screen.getByRole('heading', { name })).toBeInTheDocument()
  }
  expect(screen.getByRole('columnheader', { name: 'Source' })).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'default · Tool error: 1' }))
  expect(onSamples).toHaveBeenLastCalledWith({ subset_name: 'default', failure_type: 'tool_error' })
})

it('uses readable labels without exposing rule IDs or implementation notes', async () => {
  localStorage.setItem('evalscope-locale', 'zh')
  getDiagnosis.mockResolvedValue({
    status: 'ready',
    report: {
      ...report,
      coverage: { ...report.coverage, skipped_facts: { 'F-agent': 'No agent trace or tool calls.' } },
      facts: [
        {
          id: 'F-truncation',
          kind: 'generation',
          severity: 'warn',
          title: 'Developer title',
          value: 0.5,
          unit: 'fraction',
          definition: 'predictions.model_output.choices[0].stop_reason',
          numerator: 1,
          denominator: 2,
          samples: [],
          data: {},
        },
      ],
    },
  })
  const { container } = render(
    <LocaleProvider>
      <DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} />
    </LocaleProvider>,
  )
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })
  expect(screen.getByText('待排查问题 · 2 条样本')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: '样本 7' })).toBeInTheDocument()
  const fact = within(container.querySelector('#F-truncation') as HTMLElement)
  const title = fact.getByText(/回答被截断/)
  fireEvent.mouseEnter(title.closest('span[tabindex]') ?? title)
  expect(screen.getByRole('tooltip')).toHaveTextContent('优先看回答的结束原因')
  expect(screen.getByText('没有工具调用记录')).toBeInTheDocument()
  expect(container.textContent).not.toMatch(/stop_reason|F-agent|Developer title|不代表|分母/)
  expect(container.querySelector('meter')).toBeNull()
  localStorage.removeItem('evalscope-locale')
})

it.each([true, false])('distinguishes unknown outcomes from an all-pass run (known: %s)', async (known) => {
  getDiagnosis.mockResolvedValue({
    status: 'ready',
    report: {
      ...report,
      coverage: {
        ...report.coverage,
        task_failed: 0,
        task_success_known: known ? 3 : 0,
        outcome_unknown: known ? 0 : 3,
        unexplained_failures: 0,
        needs_attention: 0,
        excluded: 0,
      },
      patterns: [],
    },
  })
  render(
    <LocaleProvider>
      <DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} />
    </LocaleProvider>,
  )
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })
  if (known) {
    expect(screen.getByText('0 / 3')).toBeInTheDocument()
    expect(screen.getByText('No failed samples')).toBeInTheDocument()
  } else {
    expect(screen.getAllByText('Task outcomes unavailable')).toHaveLength(2)
    expect(screen.getByText('Analyzed 3 samples; task outcomes cannot currently be determined.')).toBeInTheDocument()
    expect(screen.queryByText('0 / 0')).not.toBeInTheDocument()
    expect(screen.queryByText(/known results did not pass/)).not.toBeInTheDocument()
  }
})

it('limits the summary ranking to identified issues when unexplained failures dominate', async () => {
  getDiagnosis.mockResolvedValue({
    status: 'ready',
    report: {
      ...report,
      patterns: [
        {
          ...report.patterns[0],
          id: 'P-unknown',
          failure_type: 'unexplained',
          count: 22,
          denominator: 23,
          share: 22 / 23,
          source: 'none',
        },
        { ...report.patterns[0], failure_type: 'judge_failure', denominator: 23, share: 1 / 23 },
      ],
    },
  })
  render(
    <LocaleProvider>
      <DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} />
    </LocaleProvider>,
  )
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })
  expect(screen.getByText(/Among identified issues, Scoring unavailable is the most common, found in 1 samples/)).toBeInTheDocument()
  expect(screen.queryByText(/^The most common observation/)).not.toBeInTheDocument()
})


it('distinguishes cached model analysis from no model usage', async () => {
  getDiagnosis.mockResolvedValue({ status: 'ready', report: { ...report, level: 'full', cost: { ...report.cost, cached_calls: 16 } } })
  render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })
  expect(screen.getByText('16 cached analyses reused')).toBeInTheDocument()
  expect(screen.getByText('0 new calls')).toBeInTheDocument()
  expect(screen.queryByText('No model usage')).not.toBeInTheDocument()
})


it('scales the heatmap by combined cell counts across different attributions', async () => {
  getDiagnosis.mockResolvedValue({ status: 'ready', report: { ...report, patterns: [
    { ...report.patterns[0], count: 2, by_subset: { default: 2 } },
    { ...report.patterns[0], id: 'P-b', count: 3, attribution: 'model', by_subset: { default: 3 } },
  ] } })
  render(<LocaleProvider><DiagnosisTab rootPath="/outputs" reportName="run/m" datasetName="demo" onSamples={vi.fn()} /></LocaleProvider>)
  await act(async () => { await Promise.resolve(); await Promise.resolve() })
  const cell = screen.getByRole('button', { name: 'default · Tool error: 5' })
  expect(cell.getAttribute('style')).toContain('100%')
  expect(cell).toHaveTextContent('5')
})
