import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { LocaleProvider } from '@/contexts/LocaleContext'
import type { MetricSemantics } from '@/domain/metric'
import DetailsTab from './DetailsTab'

vi.mock('@/api/reports', () => ({
  getAnalysis: vi.fn().mockResolvedValue(''),
  getDataFrame: vi.fn().mockResolvedValue({
    columns: ['Subset', 'Metric', 'Score', 'Num'],
    data: [
      { Subset: 'percentage', Metric: 'WeightedScorePercent', Score: 81.5, Num: 10 },
      { Subset: 'throughput', Metric: 'AverageOutputTps', Score: 512, Num: 10 },
    ],
  }),
}))

afterEach(cleanup)

/** Official 0-100 scale: already a percentage, so the multiplier is 1. */
const POINTS_100: MetricSemantics = {
  semantic_id: 'quality.score.points_100',
  metric_name: 'Score',
  role: 'auxiliary',
  direction: 'higher_is_better',
  value_range: { min: 0, max: 100 },
  display_kind: 'percent',
  display_multiplier: 1,
  display_unit: '%',
  display_precision: 1,
  contract_version: 1,
}

/** Unbounded throughput in its native unit. */
const THROUGHPUT: MetricSemantics = {
  semantic_id: 'perf.throughput.tokens_per_second',
  metric_name: 'Token Throughput',
  role: 'primary',
  direction: 'higher_is_better',
  raw_unit: 'tok/s',
  value_range: null,
  display_kind: 'number',
  display_unit: 'tok/s',
  display_precision: 2,
  contract_version: 1,
}

describe('DetailsTab metric semantics', () => {
  it('formats each score from its metric contract instead of its magnitude', async () => {
    render(
      <LocaleProvider>
        <DetailsTab
          reportName="fixture-report"
          datasetName="fixture-dataset"
          rootPath="/outputs"
          overallScore={512}
          semantics={THROUGHPUT}
          semanticsByMetric={{ WeightedScorePercent: POINTS_100, AverageOutputTps: THROUGHPUT }}
        />
      </LocaleProvider>,
    )

    await act(async () => {
      await Promise.resolve()
    })

    // The headline and the matching row both render the throughput in its native unit.
    expect(screen.getAllByText('512 tok/s')).toHaveLength(2)
    // An official 0-100 score keeps its own scale instead of being multiplied again.
    expect(screen.getByText('81.5%')).toBeInTheDocument()
    expect(screen.queryByText(/8150/)).not.toBeInTheDocument()
    expect(screen.queryByText(/51200/)).not.toBeInTheDocument()
  })

  it('labels the headline with the metric name and its direction', async () => {
    render(
      <LocaleProvider>
        <DetailsTab
          reportName="fixture-report"
          datasetName="fixture-dataset"
          rootPath="/outputs"
          overallScore={512}
          semantics={THROUGHPUT}
        />
      </LocaleProvider>,
    )

    await act(async () => {
      await Promise.resolve()
    })

    expect(screen.getByText('Token Throughput ↑')).toBeInTheDocument()
  })
})
