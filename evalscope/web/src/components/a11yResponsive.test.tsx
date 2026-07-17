// Accessibility / touch-target / responsive-wrapping component tests + axe.
//
// This suite covers the accessibility and responsive contracts that the pure
// logic layer cannot express — they live in the rendered DOM (class names,
// ARIA, computed structure) rather than in a pure function:
//
//   - Responsive wrapping, not truncation: the elements that
//     carry latest-run metadata (model, dataset, …) wrap on word boundaries via
//     `break-words` + `min-w-0` and never use `truncate`, so no characters are
//     dropped at 390px. jsdom has no layout engine, so we assert the wrapping
//     *strategy* is present in the class list (and truncation absent) rather
//     than measuring pixels.
//
//   - Touch targets ≥ 44×44: the primary navigation / disclosure
//     / compare-selection controls carry the 44px guarantee through either the
//     `coarse-target` utility class or explicit `min-w-[44px] min-h-[44px]`
//     hit-area padding. jsdom cannot report real pixel sizes, so we assert the
//     class/attribute that encodes the 44px guarantee is present.
//
//   - Structural a11y: a representative metadata component is run through axe.
//     The `color-contrast` rule cannot run under jsdom because there is no
//     layout/paint, so contrast is intentionally outside this suite's claims.
//
// The suite runs under the global deterministic setup (fake timers, fixed
// system time, network disabled). The axe assertion temporarily restores real
// timers because axe-core schedules its analysis on real timers, which fake
// timers would otherwise stall (mirrors Field.test.tsx / Tabs.test.tsx).

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { axe } from 'jest-axe'

import ReportCard from './reports/ReportCard'
import ReportsTable from './reports/ReportsTable'
import TopNav from './nav/TopNav'
import { LocaleProvider } from '@/contexts/LocaleContext'
import type { ReportSummary } from '@/api/types'

const FIXED_SYSTEM_TIME = new Date('2026-07-01T00:00:00.000Z')

/**
 * A representative latest-run summary with long, wrap-prone model and dataset
 * names (no natural break opportunities) so the wrapping contract is meaningful.
 */
function makeReport(overrides: Partial<ReportSummary> = {}): ReportSummary {
  return {
    name: 'Qwen2.5-72B-Instruct-Long-Model-Identifier_gsm8k_20260701_120000',
    model_name: 'Qwen2.5-72B-Instruct-Long-Model-Identifier-That-Cannot-Break',
    dataset_name: 'gsm8k-extended-reasoning-benchmark-suite-verylongname',
    score: 0.9234,
    num_samples: 128,
    timestamp: '2026-07-01T12:00:00',
    ...overrides,
  }
}

/** Render helper: wraps children in the app's Locale provider. */
function renderWithLocale(ui: React.ReactNode) {
  return render(<LocaleProvider>{ui}</LocaleProvider>)
}

/** Render helper: Locale provider + a router (needed by TopNav's NavLink). */
function renderWithRouter(ui: React.ReactNode) {
  return render(
    <LocaleProvider>
      <MemoryRouter>{ui}</MemoryRouter>
    </LocaleProvider>,
  )
}

afterEach(() => {
  cleanup()
})

describe('Responsive wrapping — metadata wraps, never truncates', () => {
  it('ReportCard renders model and dataset with break-words + min-w-0 and no truncate', () => {
    const report = makeReport()
    const { container } = renderWithLocale(
      <ReportCard report={report} selected={false} onSelect={() => {}} onClick={() => {}} />,
    )

    // The model name and dataset name are the primary metadata fields.
    const model = screen.getByText(report.model_name)
    const dataset = screen.getByText(report.dataset_name)

    for (const el of [model, dataset]) {
      expect(el.className).toContain('break-words')
      expect(el.className).toContain('min-w-0')
      expect(el.className).not.toContain('truncate')
    }

    // No element anywhere in the card relies on truncation to hide overflow.
    expect(container.querySelector('.truncate')).toBeNull()
  })

  it('ReportsTable renders model and dataset cells with break-words + min-w-0 and no truncate', () => {
    const report = makeReport()
    const { container } = renderWithLocale(
      <ReportsTable
        reports={[report]}
        selected={[]}
        allSelected={false}
        onToggleSelectAll={() => {}}
        onToggleSelect={() => {}}
        onRowClick={() => {}}
      />,
    )

    const model = screen.getByText(report.model_name)
    const dataset = screen.getByText(report.dataset_name)

    for (const el of [model, dataset]) {
      expect(el.className).toContain('break-words')
      expect(el.className).toContain('min-w-0')
      expect(el.className).not.toContain('truncate')
    }

    // The tabular metadata region must not truncate any run field.
    expect(container.querySelector('tbody .truncate')).toBeNull()
  })
})

describe('Touch targets — primary controls carry the 44px guarantee', () => {
  it('TopNav navigation links carry the coarse-target utility', () => {
    const { container } = renderWithRouter(<TopNav />)

    const links = Array.from(container.querySelectorAll('a'))
    expect(links.length).toBeGreaterThan(0)
    for (const link of links) expect(link.className).toContain('coarse-target')

    // The mobile menu toggle button is also a coarse target.
    const menuButton = screen.getByLabelText('Toggle menu')
    expect(menuButton.className).toContain('coarse-target')

  })

  it('ReportCard compare-selection control has a >=44x44 hit area', () => {
    const report = makeReport()
    renderWithLocale(
      <ReportCard report={report} selected={false} onSelect={() => {}} onClick={() => {}} />,
    )

    // The tappable wrapper around the checkbox pads the hit area to 44x44.
    const checkbox = screen.getByRole('checkbox')
    expect(checkbox.className).toContain('min-w-[44px]')
    expect(checkbox.className).toContain('min-h-[44px]')
  })

  it('ReportsTable compare-selection control has a >=44x44 hit area', () => {
    const report = makeReport()
    renderWithLocale(
      <ReportsTable
        reports={[report]}
        selected={[]}
        allSelected={false}
        onToggleSelectAll={() => {}}
        onToggleSelect={() => {}}
        onRowClick={() => {}}
      />,
    )

    const checkbox = screen.getByRole('checkbox', { name: /Select report/ })
    expect(checkbox.className).toContain('min-w-[44px]')
    expect(checkbox.className).toContain('min-h-[44px]')
  })
})

describe('Structural accessibility (axe)', () => {
  it('ReportCard has no jsdom-supported axe violations', async () => {
    const report = makeReport()
    const { container } = renderWithLocale(
      <ReportCard report={report} selected={false} onSelect={() => {}} onClick={() => {}} />,
    )

    // axe-core schedules work on real timers; the global setup installs fake
    // timers, so restore real timers for the duration of the analysis.
    vi.useRealTimers()
    try {
      const results = await axe(container)
      expect(results.violations).toEqual([])
    } finally {
      vi.useFakeTimers()
      vi.setSystemTime(FIXED_SYSTEM_TIME)
    }
  })
})
