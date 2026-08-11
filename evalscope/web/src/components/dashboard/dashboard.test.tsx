/**
 * Dashboard surfaces: four counters above, results grouped by what they measure below.
 *
 * These tests pin the decisions that make the table different from the list pages it used to
 * duplicate: repeated runs of one benchmark collapse into a single row, the whole row is the
 * disclosure control rather than a chevron at its edge, the default order is the one the reader can
 * verify from the timestamp column, and a spread is reported in the right unit and never labelled
 * good or bad.
 */

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import type { ReactNode } from 'react'

import AggregatedResults from './AggregatedResults'
import { aggregateRuns } from '@/domain/report/runAggregation'
import { LocaleProvider } from '@/contexts/LocaleContext'
import type { MetricSemantics } from '@/domain/metric'
import type { ReportSummary } from '@/api/types'

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

function report(name: string, timestamp: string, score: number, dataset = 'iquiz'): ReportSummary {
  return {
    run_id: name,
    model_id: 'qwen-plus',
    model_name: 'qwen-plus',
    dataset_name: dataset,
    num_samples: 3,
    timestamp,
    primary_metrics: [{ dataset_name: dataset, identity: { name: 'accuracy', aggregation: 'mean', dimensions: {} }, score, semantics: ACCURACY }],
    quality_ratio: score,
  }
}

function renderWith(node: ReactNode) {
  return render(
    <LocaleProvider>
      <MemoryRouter>{node}</MemoryRouter>
    </LocaleProvider>,
  )
}

/** Three runs of one benchmark swinging the full range, plus a steadier, more recent one. */
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

/** Row text in render order, header excluded. */
function bodyRows(): string[] {
  return screen.getAllByRole('row').slice(1).map((row) => row.textContent ?? '')
}

describe('AggregatedResults', () => {
  it('uses the full table width without a trailing spacer column', () => {
    const { container } = renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)

    expect(screen.getByRole('table')).toHaveClass('w-full', 'table-fixed')
    expect(container.querySelectorAll('colgroup col')).toHaveLength(9)
    expect(container.querySelector('thead tr')?.children).toHaveLength(9)
  })

  it('collapses repeated runs into one row and counts them', () => {
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)

    // Two benchmarks behind five runs, so two rows rather than five.
    expect(screen.getAllByText('automation_bench')).toHaveLength(1)
    expect(screen.getAllByText('general_mcq')).toHaveLength(1)
    expect(screen.getByText('3')).toBeInTheDocument()
  })

  it('shows the latest value on the metric own scale, with its label', () => {
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)

    expect(screen.getAllByText('50%').length).toBeGreaterThan(0)
    expect(screen.getAllByText(/^Accuracy ↑$/).length).toBeGreaterThan(0)
    // The native ratio is never shown re-scaled twice.
    expect(screen.queryByText(/5000/)).not.toBeInTheDocument()
  })

  it('shows the latest change without inventing a stability label', () => {
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)

    expect(screen.getByText('-50 pp')).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: 'Trend' })).toBeInTheDocument()
    expect(screen.queryByText('Variable')).not.toBeInTheDocument()
    expect(screen.queryByText('Stable')).not.toBeInTheDocument()
  })

  it('opens most recent first, and never labels a result', () => {
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)

    // general_mcq last ran at 11:00, automation_bench at 10:00.
    expect(bodyRows()[0]).toContain('general_mcq')
    // No warning glyph anywhere: the page reports the quantity and lets the reader judge.
    expect(screen.queryByText(/⚠/)).not.toBeInTheDocument()
  })

  it('reverses the order when the active sort header is clicked again', () => {
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)

    fireEvent.click(screen.getByRole('button', { name: /Last run/ }))

    expect(bodyRows()[0]).toContain('automation_bench')
  })

  it('sorts by another column when its header is clicked', () => {
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)

    fireEvent.click(screen.getByRole('button', { name: /Benchmark/ }))

    expect(bodyRows()[0]).toContain('automation_bench')
  })

  it('offers no sort on Latest, whose values sit on different scales', () => {
    // Ranking 0.12 req/s against 0.95 accuracy would be a comparison across two rulers, which is
    // exactly what the aggregation refuses to do everywhere else.
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)

    expect(screen.queryByRole('button', { name: /Sort by Latest/ })).not.toBeInTheDocument()
  })

  it('expands from anywhere on the row, not only from the chevron', () => {
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)

    expect(screen.queryByText('Std. dev.')).not.toBeInTheDocument()
    // The row itself is the control: clicking the benchmark name has to open it.
    fireEvent.click(screen.getByText('automation_bench'))

    expect(screen.getByText('Std. dev.')).toBeInTheDocument()
    expect(screen.getByText('Mean')).toBeInTheDocument()
    expect(screen.getByText('Range')).toBeInTheDocument()
  })

  it('opens and closes a row from its chevron button too', () => {
    // A real button keeps the disclosure reachable by keyboard and announced by a screen reader,
    // which a click handler on the row alone would not be.
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)

    fireEvent.click(screen.getAllByRole('button', { name: 'Show history' })[0])
    expect(screen.getByText('Std. dev.')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Hide history' }))
    expect(screen.queryByText('Std. dev.')).not.toBeInTheDocument()
  })

  it('reports a spread of percentages in points, not as a percentage', () => {
    // The gap between 0% and 100% is 100 percentage points. Calling it "100%" would describe it as a
    // ratio of something, which it is not.
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)
    fireEvent.click(screen.getByText('automation_bench'))

    expect(screen.getByText('100 pp')).toBeInTheDocument()
  })

  it('opens the run behind a selected history point', () => {
    const onOpenRun = vi.fn()
    renderWith(<AggregatedResults rows={rows()} onOpenRun={onOpenRun} />)
    fireEvent.click(screen.getByText('automation_bench'))

    // Every recorded run is selectable, so a suspicious point can be inspected directly.
    const bars = screen.getAllByRole('button', { name: /^08-07/ })
    expect(bars.length).toBeGreaterThan(0)
    fireEvent.click(bars[0])

    expect(onOpenRun).toHaveBeenCalledTimes(1)
    expect(onOpenRun.mock.calls[0][1].runId).toBe('a/qwen-plus')
  })

  it('opens the most recent run from the expanded summary action', () => {
    const onOpenRun = vi.fn()
    renderWith(<AggregatedResults rows={rows()} onOpenRun={onOpenRun} />)
    fireEvent.click(screen.getByText('automation_bench'))

    fireEvent.click(screen.getByRole('button', { name: 'Open latest run' }))

    expect(onOpenRun).toHaveBeenCalledTimes(1)
    expect(onOpenRun.mock.calls[0][1].runId).toBe('c/qwen-plus')
  })

  it('reads out a run as soon as the pointer reaches its bar', () => {
    // A native `title` tooltip only appears after the browser's own hover delay, which is too slow
    // to scrub a row of bars with, so the value is reported in a line that is always present.
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)
    fireEvent.click(screen.getByText('automation_bench'))

    const bars = screen.getAllByRole('button', { name: /^08-07/ })
    // Before any hover the readout describes the latest run, which here scored 50%.
    expect(screen.getAllByText('50%').length).toBeGreaterThan(0)

    fireEvent.mouseEnter(bars[0])

    // The first run scored 0%, and no tooltip delay stands between the pointer and that fact.
    expect(screen.getByText('0%')).toBeInTheDocument()
    // The bars carry no `title`, so no delayed native tooltip competes with the readout.
    expect(bars[0]).not.toHaveAttribute('title')
  })

  it('names every statistic instead of leaving a bare glyph', () => {
    // Six unlabelled icons are only readable by someone who already knows the statistics.
    renderWith(<AggregatedResults rows={rows()} onOpenRun={() => {}} />)
    fireEvent.click(screen.getByText('automation_bench'))

    for (const label of ['Latest', 'Mean', 'Range', 'Spread', 'Std. dev.', 'Runs']) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0)
    }
  })

  it('says a single run has nothing to compare against', () => {
    const single = aggregateRuns([report('only', '2026-08-07T08:00:00', 0.6)], [])
    renderWith(<AggregatedResults rows={single} onOpenRun={() => {}} />)
    fireEvent.click(screen.getByRole('button', { name: 'Show history' }))

    expect(screen.getByText(/Measured once/)).toBeInTheDocument()
    expect(screen.queryByText('New')).not.toBeInTheDocument()
  })
})
