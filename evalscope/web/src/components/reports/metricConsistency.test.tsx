// Cross-surface metric display consistency.
//
// The same metric must render identically wherever it appears. Both the card
// view (`ReportCard`, narrow screens) and the desktop table (`ReportsTable`)
// funnel their score through the centralized `formatMetricByKey` entry point, so
// a single score value must produce byte-for-byte identical display text on both
// surfaces. This test renders both components against the same
// `ReportSummary` and asserts the rendered score text matches, and matches the
// value the centralized formatter produces.

import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render } from '@testing-library/react'

import ReportCard from './ReportCard'
import ReportsTable from './ReportsTable'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { formatMetricByKey } from '@/domain/metric/registry'
import type { ReportSummary } from '@/api/types'

afterEach(cleanup)

/** Identity translate used only to compute the expected display value. */
const identity = (key: string): string => key

/** Build a representative report summary with the given score. */
function makeReport(score: number): ReportSummary {
  return {
    name: 'Qwen2.5-0.5B_gsm8k_20260701_120000',
    model_name: 'Qwen2.5-0.5B',
    dataset_name: 'gsm8k',
    score,
    num_samples: 128,
    timestamp: '2026-07-01T12:00:00',
  }
}

/** Render the card surface and return its displayed score text. */
function cardScoreText(report: ReportSummary): string {
  const { container } = render(
    <LocaleProvider>
      <ReportCard report={report} selected={false} onSelect={() => {}} onClick={() => {}} />
    </LocaleProvider>,
  )
  // The score badge is the only font-mono semibold pill in the card.
  const badge = container.querySelector('span.font-mono.font-semibold')
  expect(badge).not.toBeNull()
  return badge!.textContent ?? ''
}

/** Render the table surface and return its displayed score cell text. */
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
  // The score cell is the only font-mono semibold pill in the table body.
  const badge = container.querySelector('tbody span.font-mono.font-semibold')
  expect(badge).not.toBeNull()
  return badge!.textContent ?? ''
}

describe('metric display consistency across surfaces', () => {
  // A score whose 4-decimal round-half-up representation is stable and
  // unambiguous, plus a legitimate zero and a rounding boundary.
  const scores = [0.8567, 0.92005, 0, 0.123456, 1]

  it.each(scores)('renders score %s identically in card and table', (score) => {
    const report = makeReport(score)

    const cardText = cardScoreText(report)
    cleanup()
    const tableText = tableScoreText(report)

    // Both surfaces must show the same text ...
    expect(cardText).toBe(tableText)
    // ... and it must be exactly what the centralized formatter produces.
    const expected = formatMetricByKey('score', score, identity).primary
    expect(cardText).toBe(expected)
  })
})
