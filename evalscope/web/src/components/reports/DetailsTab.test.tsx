import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { LocaleProvider } from '@/contexts/LocaleContext'
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

describe('DetailsTab metric semantics', () => {
  it('formats each score from its metric contract instead of its magnitude', async () => {
    render(
      <LocaleProvider>
        <DetailsTab
          reportName="fixture-report"
          datasetName="fixture-dataset"
          rootPath="/outputs"
          overallScore={512}
          metricName="AverageOutputTps"
        />
      </LocaleProvider>,
    )

    await act(async () => {
      await Promise.resolve()
    })

    expect(screen.getAllByText('512.00 tokens/s')).toHaveLength(2)
    expect(screen.getByText('81.5%')).toBeInTheDocument()
    expect(screen.queryByText('8150.0%')).not.toBeInTheDocument()
    expect(screen.queryByText('51200.0%')).not.toBeInTheDocument()
  })
})
