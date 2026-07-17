// Component and accessibility unit tests for the Tabs component.
//
// Covers the WAI-ARIA tablist behaviour that the pure roving-index logic cannot
// express on its own:
//   - ARIA roles and non-orphan aria-controls / aria-labelledby wiring;
//   - exactly one selected tab;
//   - roving focus movement with wrap-around via arrow / Home / End keys;
//   - manual activation of the focused tab on Enter / Space;
//   - exactly one visible tabpanel in managed-panels mode;
//   - orphan tabs (missing panel) are dropped and reported via role="alert";
//   - a rendered tablist has no axe accessibility violations.
//
// The suite runs under the global deterministic setup (fake timers, fixed
// system time, network disabled). Keyboard interactions use synchronous
// `fireEvent.keyDown`, so no async flushing is required. The axe assertion
// temporarily restores real timers because axe-core schedules its analysis on
// real timers, which fake timers would otherwise stall.

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { axe } from 'jest-axe'

import Tabs, { type TabItem, type TabsProps } from './Tabs'
import { LocaleProvider } from '@/contexts/LocaleContext'

const FIXED_SYSTEM_TIME = new Date('2026-07-01T00:00:00.000Z')

/** Render Tabs inside the app's Locale provider (locale = en). */
function renderTabs(overrides: Partial<TabsProps> = {}) {
  const props: TabsProps = {
    tabs: [
      { key: 'a', label: 'Alpha', panelId: 'panel-a' },
      { key: 'b', label: 'Beta', panelId: 'panel-b' },
      { key: 'c', label: 'Gamma', panelId: 'panel-c' },
    ],
    activeKey: 'a',
    onChange: vi.fn(),
    panels: {
      'panel-a': <p>Alpha body</p>,
      'panel-b': <p>Beta body</p>,
      'panel-c': <p>Gamma body</p>,
    },
    ...overrides,
  }
  const utils = render(
    <LocaleProvider>
      <Tabs {...props} />
    </LocaleProvider>,
  )
  return { ...utils, props }
}

afterEach(() => {
  cleanup()
})

describe('Tabs — ARIA roles and relationships', () => {
  it('exposes tablist / tab roles and applies the orientation', () => {
    renderTabs({ orientation: 'horizontal' })

    const tablist = screen.getByRole('tablist')
    expect(tablist).toHaveAttribute('aria-orientation', 'horizontal')

    const tabs = screen.getAllByRole('tab')
    expect(tabs).toHaveLength(3)
    expect(tabs.map((tab) => tab.textContent)).toEqual(['Alpha', 'Beta', 'Gamma'])
  })

  it('renders tabpanels and wires aria-controls / aria-labelledby with no orphan references', () => {
    const tabs: TabItem[] = [
      { key: 'a', label: 'Alpha', panelId: 'panel-a' },
      { key: 'b', label: 'Beta', panelId: 'panel-b' },
    ]
    const panels = { 'panel-a': <p>Alpha body</p>, 'panel-b': <p>Beta body</p> }
    const { container } = renderTabs({ tabs, panels, activeKey: 'a' })

    const tabEls = screen.getAllByRole('tab')
    // Every tab points at a panel id that actually exists in the document.
    for (const tab of tabEls) {
      const controlled = tab.getAttribute('aria-controls')
      expect(controlled).toBeTruthy()
      expect(container.querySelector(`#${controlled}`)).not.toBeNull()
    }

    // Every panel is labelled by a tab id that actually exists (getAllByRole
    // returns hidden tabpanels too when explicitly requested).
    const panelEls = screen.getAllByRole('tabpanel', { hidden: true })
    expect(panelEls).toHaveLength(2)
    for (const panel of panelEls) {
      const labelledBy = panel.getAttribute('aria-labelledby')
      expect(labelledBy).toBeTruthy()
      const label = container.querySelector(`#${labelledBy}`)
      expect(label).not.toBeNull()
      expect(label).toHaveAttribute('role', 'tab')
    }

    // The relationship is a genuine round-trip: the selected tab's controlled
    // panel points back to that same tab.
    const selected = screen.getByRole('tab', { selected: true })
    const controlledPanel = container.querySelector(`#${selected.getAttribute('aria-controls')}`)
    expect(controlledPanel?.getAttribute('aria-labelledby')).toBe(selected.getAttribute('id'))
  })
})

describe('Tabs — single selection', () => {
  it('keeps exactly one tab selected', () => {
    renderTabs({ activeKey: 'b' })

    const selected = screen.getAllByRole('tab').filter((tab) => tab.getAttribute('aria-selected') === 'true')
    expect(selected).toHaveLength(1)
    expect(selected[0]).toHaveTextContent('Beta')
  })

  it('falls back to the first valid tab when activeKey matches nothing', () => {
    renderTabs({ activeKey: 'does-not-exist' })

    const selected = screen.getAllByRole('tab').filter((tab) => tab.getAttribute('aria-selected') === 'true')
    expect(selected).toHaveLength(1)
    expect(selected[0]).toHaveTextContent('Alpha')
  })
})

describe('Tabs — keyboard activation', () => {
  it('activates the focused tab on Enter', () => {
    const onChange = vi.fn()
    renderTabs({ onChange })

    const beta = screen.getByRole('tab', { name: 'Beta' })
    beta.focus()
    fireEvent.keyDown(beta, { key: 'Enter' })

    expect(onChange).toHaveBeenCalledTimes(1)
    expect(onChange).toHaveBeenCalledWith('b')
  })

  it('activates the focused tab on Space', () => {
    const onChange = vi.fn()
    renderTabs({ onChange })

    const gamma = screen.getByRole('tab', { name: 'Gamma' })
    gamma.focus()
    fireEvent.keyDown(gamma, { key: ' ' })

    expect(onChange).toHaveBeenCalledTimes(1)
    expect(onChange).toHaveBeenCalledWith('c')
  })
})

describe('Tabs — roving focus', () => {
  it('applies roving tabindex: selected tab is 0, others -1', () => {
    renderTabs({ activeKey: 'b' })

    const [alpha, beta, gamma] = screen.getAllByRole('tab')
    expect(beta).toHaveAttribute('tabindex', '0')
    expect(alpha).toHaveAttribute('tabindex', '-1')
    expect(gamma).toHaveAttribute('tabindex', '-1')
  })

  it('moves focus to the next / previous tab with arrow keys', () => {
    renderTabs({ activeKey: 'a' })
    const [alpha, beta, gamma] = screen.getAllByRole('tab')

    alpha.focus()
    fireEvent.keyDown(alpha, { key: 'ArrowRight' })
    expect(document.activeElement).toBe(beta)

    fireEvent.keyDown(beta, { key: 'ArrowLeft' })
    expect(document.activeElement).toBe(alpha)

    // Home / End jump to the first / last tab.
    fireEvent.keyDown(alpha, { key: 'End' })
    expect(document.activeElement).toBe(gamma)
    fireEvent.keyDown(gamma, { key: 'Home' })
    expect(document.activeElement).toBe(alpha)
  })

  it('wraps focus around at both ends', () => {
    renderTabs({ activeKey: 'a' })
    const [alpha, beta, gamma] = screen.getAllByRole('tab')

    // Moving left from the first tab wraps to the last.
    alpha.focus()
    fireEvent.keyDown(alpha, { key: 'ArrowLeft' })
    expect(document.activeElement).toBe(gamma)

    // Moving right from the last tab wraps back to the first.
    fireEvent.keyDown(gamma, { key: 'ArrowRight' })
    expect(document.activeElement).toBe(alpha)

    void beta
  })

  it('does not activate the tab while only moving focus', () => {
    const onChange = vi.fn()
    renderTabs({ activeKey: 'a', onChange })
    const [alpha] = screen.getAllByRole('tab')

    alpha.focus()
    fireEvent.keyDown(alpha, { key: 'ArrowRight' })

    expect(onChange).not.toHaveBeenCalled()
  })
})

describe('Tabs — single visible panel', () => {
  it('shows only the selected panel and hides the rest', () => {
    const tabs: TabItem[] = [
      { key: 'a', label: 'Alpha', panelId: 'panel-a' },
      { key: 'b', label: 'Beta', panelId: 'panel-b' },
      { key: 'c', label: 'Gamma', panelId: 'panel-c' },
    ]
    const panels = {
      'panel-a': <p>Alpha body</p>,
      'panel-b': <p>Beta body</p>,
      'panel-c': <p>Gamma body</p>,
    }
    const { container } = renderTabs({ tabs, panels, activeKey: 'b' })

    const allPanels = screen.getAllByRole('tabpanel', { hidden: true })
    const visible = allPanels.filter((panel) => !panel.hasAttribute('hidden'))
    expect(visible).toHaveLength(1)
    expect(within(visible[0]).getByText('Beta body')).toBeInTheDocument()

    // The non-selected panels are hidden and render no body content.
    const panelA = container.querySelector('#panel-a')
    const panelC = container.querySelector('#panel-c')
    expect(panelA).toHaveAttribute('hidden')
    expect(panelC).toHaveAttribute('hidden')
    expect(panelA).toBeEmptyDOMElement()
    expect(panelC).toBeEmptyDOMElement()
  })
})

describe('Tabs — orphan tab handling', () => {
  it('drops tabs whose panel is missing and reports them via role="alert"', () => {
    const tabs: TabItem[] = [
      { key: 'a', label: 'Alpha', panelId: 'panel-a' },
      { key: 'b', label: 'Beta', panelId: 'panel-missing' },
    ]
    const panels = { 'panel-a': <p>Alpha body</p> }
    renderTabs({ tabs, panels, activeKey: 'a' })

    // Only the valid tab is rendered.
    const renderedTabs = screen.getAllByRole('tab')
    expect(renderedTabs).toHaveLength(1)
    expect(renderedTabs[0]).toHaveTextContent('Alpha')
    expect(screen.queryByRole('tab', { name: 'Beta' })).toBeNull()

    // The orphan is surfaced to assistive tech and names the offending key.
    const alert = screen.getByRole('alert')
    expect(alert).toBeInTheDocument()
    expect(alert).toHaveTextContent('b')
  })

  it('renders no alert when every tab has a matching panel', () => {
    const tabs: TabItem[] = [{ key: 'a', label: 'Alpha', panelId: 'panel-a' }]
    const panels = { 'panel-a': <p>Alpha body</p> }
    renderTabs({ tabs, panels, activeKey: 'a' })

    expect(screen.queryByRole('alert')).toBeNull()
  })
})

describe('Tabs — accessibility (axe)', () => {
  it('has no axe violations in managed-panels mode', async () => {
    const tabs: TabItem[] = [
      { key: 'a', label: 'Alpha', panelId: 'panel-a' },
      { key: 'b', label: 'Beta', panelId: 'panel-b' },
    ]
    const panels = { 'panel-a': <p>Alpha body</p>, 'panel-b': <p>Beta body</p> }
    const { container } = renderTabs({ tabs, panels, activeKey: 'a' })

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
