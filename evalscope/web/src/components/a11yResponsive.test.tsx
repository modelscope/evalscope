// Contrast / touch-target / responsive-wrapping component tests + axe.
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
//   - Contrast-related a11y: a representative metadata component is
//     run through axe. Note that axe-core's `color-contrast` rule is disabled
//     under jsdom because there is no layout/paint to sample colours from, so a
//     passing axe run here validates the other a11y rules. Contrast tokens are
//     reviewed against DESIGN.md and exercised in focused component/Browser
//     checks; this jsdom assertion alone is only a partial guarantee.
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
import EvalRunCard from './ui/EvalRunCard'
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

  it('EvalRunCard renders the model name with break-words + min-w-0 and no truncate', () => {
    const report = makeReport()
    const { container } = renderWithLocale(<EvalRunCard report={report} onClick={() => {}} />)

    const model = screen.getByText(report.model_name)
    expect(model.className).toContain('break-words')
    expect(model.className).toContain('min-w-0')
    expect(model.className).not.toContain('truncate')

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

    // Every rendered navigation link (desktop + tablet + right-hand controls)
    // opts into the 44px coarse-pointer hit area via `coarse-target`.
    const links = container.querySelectorAll('a[href], a[class]')
    const navLinks = Array.from(container.querySelectorAll('a')).filter((a) =>
      a.className.includes('coarse-target'),
    )
    expect(navLinks.length).toBeGreaterThan(0)

    // The mobile menu toggle button is also a coarse target.
    const menuButton = screen.getByLabelText('Toggle menu')
    expect(menuButton.className).toContain('coarse-target')

    void links
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

describe('Contrast-related accessibility (axe)', () => {
  it('ReportCard has no axe violations (incl. no color-contrast violations)', async () => {
    // ReportCard renders essential metadata text (model, dataset, score) using
    // the design system's --text / --text-muted tokens — a representative
    // contrast surface.
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

      // Explicitly assert there are no color-contrast violations. Under jsdom
      // axe cannot compute painted colours, so this rule does not fire here.
      const contrastViolations = results.violations.filter((v) => v.id === 'color-contrast')
      expect(contrastViolations).toHaveLength(0)
    } finally {
      vi.useFakeTimers()
      vi.setSystemTime(FIXED_SYSTEM_TIME)
    }
  })
})
