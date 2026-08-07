/**
 * The result surface always shows concrete metric/score pairs.
 *
 * Feature: metric-semantics-governance. A run may cover datasets with different primary metrics.
 * Averaging them would be a fake total, but the earlier behaviour swung the other way and replaced
 * the numbers with a note ("Multiple metrics"), which hid the very results the run produced. These
 * tests pin the contract: every dataset is listed with its own metric and value, formatted by its
 * own semantics, and an inferred primary metric is marked rather than silently presented as
 * declared.
 */

import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import PrimaryMetricResult from './PrimaryMetricResult'
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
  return { dataset_name: 'gsm8k', metric_name: 'mean_acc', score: 0.6, semantics: ACCURACY, ...overrides }
}

function textOf(refs: PrimaryMetricRef[]): string {
  const { container } = render(
    <PrimaryMetricResult refs={refs} emptyLabel="none" inferredHint="inferred hint" />,
  )
  return container.textContent ?? ''
}

afterEach(cleanup)

describe('PrimaryMetricResult', () => {
  it('shows the metric and its value for a single dataset', () => {
    const text = textOf([ref()])

    expect(text).toContain('Accuracy')
    expect(text).toContain('60%')
  })

  it('lists every dataset when their metrics differ, with each on its own scale', () => {
    const text = textOf([
      ref({ dataset_name: 'conll2003', metric_name: 'f1_score', score: 0.912 }),
      ref({ dataset_name: 'torgo', metric_name: 'wer', score: 0.0432, semantics: WER }),
    ])

    expect(text).toContain('conll2003')
    expect(text).toContain('91.2%')
    expect(text).toContain('torgo')
    expect(text).toContain('WER')
    expect(text).toContain('4.3%')
  })

  it('renders each metric with its own unit rather than one shared one', () => {
    const text = textOf([
      ref({ dataset_name: 'a', score: 0.5 }),
      ref({ dataset_name: 'b', metric_name: 'latency', score: 1.25, semantics: LATENCY }),
    ])

    expect(text).toContain('50%')
    expect(text).toContain('1.25 s')
  })

  it('marks an inferred primary metric and explains it in the tooltip', () => {
    const { container } = render(
      <PrimaryMetricResult
        refs={[ref({ metric_name: 'avg@1_all/success_rate', semantics: null, inferred: true })]}
        emptyLabel="none"
        inferredHint="inferred hint"
      />,
    )

    expect(container.textContent).toContain('*')
    expect(container.querySelector('[title*="inferred hint"]')).not.toBeNull()
  })

  it('does not mark a declared primary metric', () => {
    const { container } = render(
      <PrimaryMetricResult refs={[ref()]} emptyLabel="none" inferredHint="inferred hint" />,
    )

    expect(container.textContent).not.toContain('*')
    expect(container.querySelector('[title*="inferred hint"]')).toBeNull()
  })

  it('collapses the tail into a count once there are many datasets', () => {
    const refs = ['a', 'b', 'c', 'd', 'e'].map((name) => ref({ dataset_name: name }))

    const text = textOf(refs)

    expect(text).toContain('a')
    expect(text).toContain('c')
    // Only the first three are listed; the rest are summarised.
    expect(text).not.toContain('d')
    expect(text).toContain('+2')
  })

  it('falls back to the placeholder when a run reports nothing', () => {
    expect(textOf([])).toBe('none')
  })
})
