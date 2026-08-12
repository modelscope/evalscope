import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { MetricSemantics } from '@/domain/metric'
import ScoreBar from './ScoreBar'

afterEach(cleanup)

/** Bounded 0-1 quality metric: has a range, so a bar can place the value in it. */
const RATIO: MetricSemantics = {
  semantic_id: 'quality.score.ratio',
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

/** Bounded metric where a low value is the good one. */
const ERROR_RATE: MetricSemantics = {
  ...RATIO,
  semantic_id: 'quality.error.ratio',
  metric_name: 'WER',
  direction: 'lower_is_better',
}

/** Unbounded metric: no range, so no proportion exists to draw. */
const THROUGHPUT: MetricSemantics = {
  semantic_id: 'perf.throughput.tokens_per_second',
  metric_name: 'Token Throughput',
  role: 'primary',
  direction: 'higher_is_better',
  value_range: null,
  display_kind: 'number',
  display_multiplier: 1,
  display_unit: 't/s',
  display_precision: 1,
  contract_version: 1,
}

describe('ScoreBar', () => {
  // Both layouts render the same element tree, so neither surface can quietly
  // lose the bar's programmatic name the way a hand-written copy did.
  it.each(['fill', 'fixed'] as const)('exposes the bar as a named progressbar in the %s layout', (track) => {
    render(<ScoreBar score={0.912} semantics={RATIO} ariaLabel="gsm8k accuracy" track={track} />)

    const bar = screen.getByRole('progressbar', { name: 'gsm8k accuracy' })
    expect(bar).toHaveAttribute('aria-valuenow', '91')
    expect(bar).toHaveAttribute('aria-valuemin', '0')
    expect(bar).toHaveAttribute('aria-valuemax', '100')
  })

  it('sizes the bar by the value position, not by its quality', () => {
    // A 4.3% error rate is a *short* bar even though it is a good result: length
    // is the value's own place in its range, and colour carries the quality.
    render(<ScoreBar score={0.043} semantics={ERROR_RATE} ariaLabel="wer" />)

    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '4')
  })

  it('omits the bar for a metric with no bounded range', () => {
    render(<ScoreBar score={512} semantics={THROUGHPUT} ariaLabel="throughput" />)

    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
    expect(screen.getByText('512 t/s')).toBeInTheDocument()
  })

  it('renders the formatted value through the metric domain', () => {
    render(<ScoreBar score={0.912} semantics={RATIO} ariaLabel="accuracy" />)

    expect(screen.getByText('91.2%')).toBeInTheDocument()
  })
})
