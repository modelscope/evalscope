/**
 * The result cells always show concrete metric/score pairs, aligned row by row.
 *
 * Feature: metric-semantics-governance. These pin two contracts that regressed before:
 * a run's numbers are never replaced by a note about them, and the dataset name is never repeated
 * inside the Result column when a Dataset column already carries it. The renderers emit one line
 * per dataset at a shared line height, which is what keeps the columns aligned.
 */

import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { DatasetLines, MetricLines, ScoreLines } from './metricCells'
import type { PrimaryMetricRef } from '@/domain/report/primaryMetrics'
import type { MetricSemantics } from '@/domain/metric'

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

const F1: MetricSemantics = {
  ...ACCURACY,
  semantic_id: 'quality.f1.ratio',
  metric_name: 'F1',
}

const WER: MetricSemantics = {
  ...ACCURACY,
  semantic_id: 'quality.wer.ratio',
  metric_name: 'WER',
  direction: 'lower_is_better',
}

const LATENCY: MetricSemantics = {
  semantic_id: 'perf.latency.seconds',
  metric_name: 'Latency',
  role: 'primary',
  direction: 'lower_is_better',
  display_kind: 'number',
  display_unit: 's',
  display_precision: 3,
  contract_version: 1,
}

function ref(overrides: Partial<PrimaryMetricRef> = {}): PrimaryMetricRef {
  return {
    dataset_name: 'gsm8k',
    identity: { name: 'accuracy', aggregation: 'mean', dimensions: {} },
    score: 0.6,
    semantics: ACCURACY,
    ...overrides,
  }
}

const MIXED = [
  ref({ dataset_name: 'conll2003', identity: { name: 'f1', aggregation: 'mean', dimensions: {} }, score: 0.912, semantics: F1 }),
  ref({ dataset_name: 'torgo', identity: { name: 'wer', aggregation: 'mean', dimensions: {} }, score: 0.0432, semantics: WER }),
]

afterEach(cleanup)

describe('ScoreLines', () => {
  it('shows the value of a single dataset', () => {
    const { container } = render(<ScoreLines refs={[ref()]} emptyLabel="none" />)

    expect(container.textContent).toContain('60%')
  })

  it('renders each dataset on its own scale', () => {
    const { container } = render(<ScoreLines refs={MIXED} emptyLabel="none" />)

    expect(container.textContent).toContain('91.2%')
    expect(container.textContent).toContain('4.3%')
  })

  it('keeps each metric in its own unit rather than one shared one', () => {
    const refs = [ref({ score: 0.5 }), ref({ dataset_name: 'b', identity: { name: 'latency', aggregation: 'mean', dimensions: {} }, score: 1.25, semantics: LATENCY })]

    const { container } = render(<ScoreLines refs={refs} emptyLabel="none" />)

    expect(container.textContent).toContain('50%')
    expect(container.textContent).toContain('1.25 s')
  })

  it('never repeats the dataset name, which the Dataset column already carries', () => {
    const { container } = render(<ScoreLines refs={MIXED} emptyLabel="none" />)

    expect(container.textContent).not.toContain('conll2003')
    expect(container.textContent).not.toContain('torgo')
  })

  it('carries the metric label inline when asked, for a layout with no Metric column', () => {
    const { container } = render(<ScoreLines refs={MIXED} emptyLabel="none" inlineMetricClass="" />)

    expect(container.textContent).toContain('F1 ↑')
    expect(container.textContent).toContain('WER ↓')
  })

  it('omits the inline label by default, so a Metric column is not duplicated', () => {
    const { container } = render(<ScoreLines refs={MIXED} emptyLabel="none" />)

    expect(container.textContent).not.toContain('F1 ↑')
  })

  it('collapses the tail once there are many datasets', () => {
    const refs = ['a', 'b', 'c', 'd', 'e'].map((name) => ref({ dataset_name: name }))

    const { container } = render(<ScoreLines refs={refs} emptyLabel="none" />)

    // Three lines plus an ellipsis marker, matching DatasetLines' own cap.
    expect(container.textContent).toContain('…')
  })

  it('falls back to the placeholder when a run reports nothing', () => {
    const { container } = render(<ScoreLines refs={[]} emptyLabel="none" />)

    expect(container.textContent).toBe('none')
  })
})

describe('DatasetLines', () => {
  it('lists one dataset per line', () => {
    const { container } = render(<DatasetLines refs={MIXED} fallback="unused" />)

    expect(container.textContent).toContain('conll2003')
    expect(container.textContent).toContain('torgo')
  })

  it('shows the joined fallback when the response carries no refs', () => {
    const { container } = render(<DatasetLines refs={[]} fallback="a, b" />)

    expect(container.textContent).toBe('a, b')
  })

  it('caps the visible lines and reports the remainder', () => {
    const refs = ['a', 'b', 'c', 'd', 'e'].map((name) => ref({ dataset_name: name }))

    const { container } = render(<DatasetLines refs={refs} fallback="unused" />)

    expect(container.textContent).toContain('a')
    expect(container.textContent).not.toContain('d')
    expect(container.textContent).toContain('+2')
  })
})

describe('MetricLines', () => {
  it('labels each dataset with its metric and direction', () => {
    const { container } = render(<MetricLines refs={MIXED} />)

    expect(container.textContent).toContain('F1 ↑')
    expect(container.textContent).toContain('WER ↓')
  })

  it('puts the complete identity in the tooltip', () => {
    const { container } = render(<MetricLines refs={[ref()]} />)
    expect(container.querySelector('[title="accuracy:mean"]')).not.toBeNull()
  })
})
