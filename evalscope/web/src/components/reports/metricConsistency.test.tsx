// Cross-surface metric display consistency.
//
// The same metric must render identically wherever it appears. Both the card view (`ReportCard`,
// narrow screens) and the desktop table (`ReportsTable`) derive their score from the backend's
// `primary_metrics` through one shared helper, so a single report must produce byte-for-byte
// identical display text on both surfaces — including the "cannot be merged" case, where neither
// surface may invent a single number.
//
// Feature: metric-semantics-governance, Property 41 (frontend side): every view shows the same
// metric name, direction and formatted value.

import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render } from '@testing-library/react'

import ReportCard from './ReportCard'
import ReportsTable from './ReportsTable'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { formatMetric } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'
import type { ReportSummary } from '@/api/types'

afterEach(cleanup)

/** Accuracy as the backend declares it: a bounded ratio rendered as a percentage. */
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

/** WER: a bounded ratio where lower is better. */
const WER: MetricSemantics = { ...ACCURACY, semantic_id: 'quality.wer.ratio', metric_name: 'WER', direction: 'lower_is_better' }

/** Build a report summary carrying one primary metric. */
function makeReport(score: number, semantics: MetricSemantics = ACCURACY): ReportSummary {
  return {
    name: 'Qwen2.5-0.5B_gsm8k_20260701_120000',
    model_name: 'Qwen2.5-0.5B',
    dataset_name: 'gsm8k',
    dataset_pretty_name: 'GSM8K Pretty',
    num_samples: 128,
    timestamp: '2026-07-01T12:00:00',
    primary_metrics: [{
      dataset_name: 'gsm8k', dataset_pretty_name: 'GSM8K Pretty',
      identity: { name: 'accuracy', aggregation: 'mean', dimensions: {} },
      score,
      semantics,
    }],
    quality_ratio: score,
  }
}

/** Build a report whose datasets report different metrics, so no single score exists. */
function makeMixedReport(): ReportSummary {
  return {
    ...makeReport(0.5),
    primary_metrics: [
      { dataset_name: 'gsm8k', identity: { name: 'accuracy', aggregation: 'mean', dimensions: {} }, score: 0.9, semantics: ACCURACY },
      { dataset_name: 'librispeech', identity: { name: 'wer', aggregation: 'mean', dimensions: {} }, score: 0.07, semantics: WER },
    ],
    quality_ratio: 0.915,
  }
}

/** Render the card surface and return its displayed score text. */
function cardScoreText(report: ReportSummary): string {
  const { container } = render(
    <LocaleProvider>
      <ReportCard report={report} selected={false} onSelect={() => {}} onClick={() => {}} />
    </LocaleProvider>,
  )
  const badge = container.querySelector('.font-mono.font-semibold')
  expect(badge).not.toBeNull()
  return badge!.textContent ?? ''
}

/** Render the table surface and return its displayed score text. */
function tableScoreText(report: ReportSummary): string {
  const { container } = render(
    <LocaleProvider>
      <ReportsTable
        reports={[report]}
        selected={[]}
        allSelected={false}
        onToggleSelectAll={() => {}}
        onToggleSelect={() => {}}
        onRowClick={() => {}}
      />
    </LocaleProvider>,
  )
  const badge = container.querySelector('.font-mono.font-semibold')
  expect(badge).not.toBeNull()
  return badge!.textContent ?? ''
}

describe('metric display consistency across surfaces', () => {
  it('shows the pretty dataset name while retaining the stable id in data', () => {
    const report = makeReport(0.5)

    for (const render_ of [
      () => <ReportCard report={report} selected={false} onSelect={() => {}} onClick={() => {}} />,
      () => (
        <ReportsTable
          reports={[report]}
          selected={[]}
          allSelected={false}
          onToggleSelectAll={() => {}}
          onToggleSelect={() => {}}
          onRowClick={() => {}}
        />
      ),
    ]) {
      const { container } = render(<LocaleProvider>{render_()}</LocaleProvider>)
      expect(container.textContent).toContain('GSM8K Pretty')
      expect(report.dataset_name).toBe('gsm8k')
      cleanup()
    }
  })

  it('renders the same formatted score in the card and the table', () => {
    for (const score of [0, 0.0721, 0.5, 0.8567, 1]) {
      const report = makeReport(score)
      const expected = formatMetric(score, ACCURACY).primary

      expect(cardScoreText(report).trim()).toBe(expected)
      cleanup()
      expect(tableScoreText(report).trim()).toBe(expected)
      cleanup()
    }
  })

  it('honours a lower-is-better metric identically on both surfaces', () => {
    const report = makeReport(0.0721, WER)
    const expected = formatMetric(0.0721, WER).primary

    expect(cardScoreText(report).trim()).toBe(expected)
    cleanup()
    expect(tableScoreText(report).trim()).toBe(expected)
  })

  it('shows every dataset as metric and score when the metrics cannot be merged', () => {
    // Averaging an accuracy with a WER would be a fake total, but hiding both numbers behind a
    // note is worse: each dataset is listed with its own metric, formatted by its own semantics.
    const report = makeMixedReport()
    const expectedAccuracy = formatMetric(0.9, ACCURACY).primary
    const expectedWer = formatMetric(0.07, WER).primary

    for (const [surface, render_] of [
      ['card', () => (
        <LocaleProvider>
          <ReportCard report={report} selected={false} onSelect={() => {}} onClick={() => {}} />
        </LocaleProvider>
      )],
      ['table', () => (
        <LocaleProvider>
          <ReportsTable
            reports={[report]}
            selected={[]}
            allSelected={false}
            onToggleSelectAll={() => {}}
            onToggleSelect={() => {}}
            onRowClick={() => {}}
          />
        </LocaleProvider>
      )],
    ] as const) {
      const { container } = render(render_())
      const text = container.textContent ?? ''

      // Both metrics and both values are present on either surface. The dataset names live in the
      // table's own Dataset column and on the card's per-line prefix, so only the metric and the
      // value are asserted here -- what matters is that no number is hidden.
      expect(text, surface).toContain('Accuracy')
      expect(text, surface).toContain(expectedAccuracy)
      expect(text, surface).toContain('WER')
      expect(text, surface).toContain(expectedWer)
      // No merged total, and no placeholder standing in for the numbers.
      expect(text, surface).not.toContain('cannot be merged')
      cleanup()
    }
  })
})
