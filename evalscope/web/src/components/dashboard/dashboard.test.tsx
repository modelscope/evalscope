/**
 * Dashboard surfaces: actions above, results grouped by what they measure below.
 *
 * These tests pin the decisions that make the page different from the list pages it used to
 * duplicate: repeated runs of one benchmark collapse into a single row carrying its spread, a spread
 * is reported in the right unit and never labelled good or bad, and the repeat-run link carries only
 * non-secret fields.
 */

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import type { ReactNode } from 'react'

import QuickActions from './QuickActions'
import { repeatableRuns } from '@/domain/report/repeatRun'
import AggregatedResults from './AggregatedResults'
import ActivityStrip from './ActivityStrip'
import { aggregateRuns } from '@/domain/report/runAggregation'
import { LocaleProvider } from '@/contexts/LocaleContext'
import type { MetricSemantics } from '@/domain/metric'
import type { PerfRunSummary, ReportSummary } from '@/api/types'

afterEach(() => {
  cleanup()
})

const ACCURACY: MetricSemantics = {
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
}

const RPS: MetricSemantics = {
  semantic_id: 'perf.throughput.requests_per_second',
  metric_name: 'Best RPS',
  role: 'primary',
  direction: 'higher_is_better',
  display_kind: 'number',
  display_unit: 'req/s',
  display_precision: 4,
  contract_version: 1,
}

function report(name: string, timestamp: string, score: number, dataset = 'iquiz'): ReportSummary {
  return {
    name,
    model_name: 'qwen-plus',
    dataset_name: dataset,
    score,
    num_samples: 3,
    timestamp,
    primary_metrics: [{ dataset_name: dataset, metric_name: 'mean_acc', score, semantics: ACCURACY }],
  } as ReportSummary
}

function perfRun(timestamp: string, bestRps: number): PerfRunSummary {
  return {
    path: `perf/${timestamp}`,
    model: 'qwen-plus',
    api_type: 'openai',
    dataset: 'openqa',
    num_runs: 1,
    total_requests: 10,
    success_rate: 1,
    best_rps: bestRps,
    best_latency: 1.2,
    is_embedding: false,
    has_html: false,
    timestamp,
  } as PerfRunSummary
}

function renderWith(node: ReactNode) {
  return render(
    <LocaleProvider>
      <MemoryRouter>{node}</MemoryRouter>
    </LocaleProvider>,
  )
}

describe('QuickActions', () => {
  it('offers the four entry points as links', () => {
    renderWith(<QuickActions reports={[]} perfRuns={[]} />)

    expect(screen.getByRole('link', { name: /Run evaluation/ })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /Run benchmark/ })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /Compare models/ })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /Browse benchmarks/ })).toBeInTheDocument()
  })

  it('prefills a repeat with the model and every dataset of the run', () => {
    const multi = {
      ...report('run-a', '2026-08-07T11:37:00', 1, 'general_mcq'),
      primary_metrics: [
        { dataset_name: 'general_mcq', metric_name: 'mean_acc', score: 1, semantics: ACCURACY },
        { dataset_name: 'iquiz', metric_name: 'mean_acc', score: 0.5, semantics: ACCURACY },
      ],
    } as ReportSummary

    const [run] = repeatableRuns([multi], [])

    expect(run.href).toContain('model=qwen-plus')
    expect(run.href).toContain(encodeURIComponent('general_mcq,iquiz'))
  })

  it('never puts a credential in the repeat URL', () => {
    // The link goes into browser history and into every proxy log on the way, so only the model and
    // the dataset list may travel this way. The task form still asks for the key.
    const runs = repeatableRuns([report('run-a', '2026-08-07T11:37:00', 1)], [perfRun('2026-08-07T09:00:00', 0.12)])

    expect(runs).not.toHaveLength(0)
    for (const run of runs) {
      expect(run.href).not.toMatch(/api[_-]?key|token|secret|password/i)
    }
  })

  it('orders the repeat offers by recency across eval and perf', () => {
    const runs = repeatableRuns(
      [report('older', '2026-08-07T08:00:00', 1)],
      [perfRun('2026-08-07T12:00:00', 0.23)],
    )

    expect(runs[0].kind).toBe('perf')
  })
})

describe('AggregatedResults', () => {
  /** Three runs of one benchmark swinging the full range, plus a steady one. */
  function rows() {
    return aggregateRuns(
      [
        report('a', '2026-08-07T08:00:00', 0, 'automation_bench'),
        report('b', '2026-08-07T09:00:00', 1, 'automation_bench'),
        report('c', '2026-08-07T10:00:00', 0.5, 'automation_bench'),
        report('d', '2026-08-07T10:30:00', 1, 'general_mcq'),
        report('e', '2026-08-07T11:00:00', 1, 'general_mcq'),
      ],
      [],
    )
  }

  it('collapses repeated runs into one row and counts them', () => {
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)

    // Two benchmarks behind five runs, so two rows rather than five.
    expect(screen.getAllByText('automation_bench')).toHaveLength(1)
    expect(screen.getAllByText('general_mcq')).toHaveLength(1)
    expect(screen.getByText('3')).toBeInTheDocument()
  })

  it('reports a spread of percentages in points, not as a percentage', () => {
    // The gap between 0% and 100% is 100 percentage points. Calling it "100%" would describe it as a
    // ratio of something, which it is not.
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)

    expect(screen.getByText('100 pp')).toBeInTheDocument()
  })

  it('shows the latest value on the metric own scale, with its label', () => {
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)

    expect(screen.getAllByText('50%').length).toBeGreaterThan(0)
    expect(screen.getAllByText(/^Accuracy ↑$/).length).toBeGreaterThan(0)
    // The native ratio is never shown re-scaled twice.
    expect(screen.queryByText(/5000/)).not.toBeInTheDocument()
  })

  it('puts the widest spread first and never labels it', () => {
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)

    const benchmarks = screen.getAllByRole('row').slice(1).map((row) => row.textContent ?? '')
    expect(benchmarks[0]).toContain('automation_bench')
    // No warning glyph anywhere: the page reports the quantity and lets the reader judge.
    expect(screen.queryByText(/⚠/)).not.toBeInTheDocument()
  })

  it('reveals full statistics only when a row is expanded', () => {
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)

    expect(screen.queryByText('Std. dev.')).not.toBeInTheDocument()
    fireEvent.click(screen.getAllByRole('button', { name: 'Show history' })[0])

    expect(screen.getByText('Std. dev.')).toBeInTheDocument()
    expect(screen.getByText('Mean')).toBeInTheDocument()
    expect(screen.getByText('Range')).toBeInTheDocument()
  })

  it('opens the run behind a selected history point', () => {
    const onOpenRun = vi.fn()
    renderWith(<AggregatedResults rows={rows()} onOpenRun={onOpenRun} />)
    fireEvent.click(screen.getAllByRole('button', { name: 'Show history' })[0])

    // Every recorded run is selectable, so a suspicious point can be inspected directly.
    const bars = screen.getAllByRole('button').filter((node) => /^08-07/.test(node.getAttribute('title') ?? ''))
    expect(bars.length).toBeGreaterThan(0)
    fireEvent.click(bars[0])

    expect(onOpenRun).toHaveBeenCalledTimes(1)
    expect(onOpenRun.mock.calls[0][1].runId).toBe('a')
  })

  it('says a single run has nothing to compare against', () => {
    const single = aggregateRuns([report('only', '2026-08-07T08:00:00', 0.6)], [])
    renderWith(<AggregatedResults rows={single} onOpenRun={() => {}} />)
    fireEvent.click(screen.getByRole('button', { name: 'Show history' }))

    expect(screen.getByText(/Measured once/)).toBeInTheDocument()
  })
})

describe('ActivityStrip', () => {
  it('lists recent runs newest first, across eval and perf', () => {
    renderWith(
      <ActivityStrip
        reports={[report('a', '2026-08-07T10:00:00', 0.5)]}
        perfRuns={[perfRun('2026-08-07T11:00:00', 0.23)]}
        perfSemantics={{ best_rps: RPS }}
        rootPath="/tmp/out"
      />,
    )

    const links = screen.getAllByRole('link').filter((node) => node.textContent?.includes('08-07'))
    expect(links[0].textContent).toContain('openqa')
    expect(links[1].textContent).toContain('iquiz')
  })

  it('formats each run with its own metric semantics', () => {
    renderWith(
      <ActivityStrip
        reports={[report('a', '2026-08-07T10:00:00', 0.5)]}
        perfRuns={[perfRun('2026-08-07T11:00:00', 0.23)]}
        perfSemantics={{ best_rps: RPS }}
        rootPath="/tmp/out"
      />,
    )

    // A ratio renders as a percentage; an unbounded throughput keeps its own unit.
    expect(screen.getByText('50%')).toBeInTheDocument()
    expect(screen.getByText('0.23 req/s')).toBeInTheDocument()
  })

  it('renders nothing when there is no activity', () => {
    const { container } = renderWith(
      <ActivityStrip reports={[]} perfRuns={[]} perfSemantics={{}} rootPath="/tmp/out" />,
    )

    expect(container.textContent).toBe('')
  })
})
