// Component and accessibility unit tests for the Combobox primitive (Task 5.8).
//
// Combobox implements the WAI-ARIA APG "combobox with listbox popup" pattern.
// These tests cover the interactive accessibility behaviour that pure logic
// cannot express:
//   - the trigger exposes role="combobox" with aria-expanded and aria-controls,
//     and opening reveals a role="listbox" with role="option" items whose
//     highlight is tracked by aria-activedescendant (Req 10.5);
//   - keyboard: ArrowDown/ArrowUp move the highlight, Enter commits the
//     highlighted option through onChange, Escape closes the popup (Req 10.5);
//   - an open combobox has no axe accessibility violations.
//
// The suite runs under the global deterministic setup (fake timers, fixed
// system time, network disabled). Keyboard interactions use synchronous
// `fireEvent.keyDown`; the axe assertion temporarily restores real timers
// because axe-core schedules its analysis on real timers (mirrors
// Tabs.test.tsx).

import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { axe } from 'jest-axe'

import Combobox, { type ComboboxOption, type ComboboxProps } from './Combobox'
import { LocaleProvider } from '@/contexts/LocaleContext'

const FIXED_SYSTEM_TIME = new Date('2026-07-01T00:00:00.000Z')

const OPTIONS: ComboboxOption[] = [
  { value: 'gsm8k', labelKey: 'datasets.gsm8k.label' },
  { value: 'mmlu', labelKey: 'datasets.mmlu.label' },
  { value: 'arc', labelKey: 'datasets.arc.label' },
]

beforeAll(() => {
  // jsdom does not implement scrollIntoView, which Combobox calls to keep the
  // highlighted option in view. Provide a no-op so those effects don't throw.
  if (!Element.prototype.scrollIntoView) {
    Element.prototype.scrollIntoView = vi.fn()
  }
})

/** Render Combobox inside the app's Locale provider (locale = en). */
function renderCombobox(overrides: Partial<ComboboxProps> = {}) {
  const props: ComboboxProps = {
    options: OPTIONS,
    value: 'gsm8k',
    onChange: vi.fn(),
    labelKey: 'fields.dataset.label',
    name: 'dataset',
    ...overrides,
  }
  const utils = render(
    <LocaleProvider>
      <Combobox {...props} />
    </LocaleProvider>,
  )
  return { ...utils, props }
}

afterEach(() => {
  cleanup()
})

describe('Combobox — ARIA roles and relationships (Req 10.5)', () => {
  it('exposes role=combobox with aria-expanded=false and aria-controls when closed', () => {
    renderCombobox()

    const combobox = screen.getByRole('combobox')
    expect(combobox).toHaveAttribute('aria-expanded', 'false')
    expect(combobox).toHaveAttribute('aria-controls')
    expect(combobox).toHaveAccessibleName()
    // The popup is not present until opened.
    expect(screen.queryByRole('listbox')).toBeNull()
  })

  it('opens a listbox of options and points aria-activedescendant at the highlight', () => {
    renderCombobox({ value: 'gsm8k' })

    const combobox = screen.getByRole('combobox')
    fireEvent.click(combobox)

    expect(combobox).toHaveAttribute('aria-expanded', 'true')

    const listbox = screen.getByRole('listbox')
    // aria-controls references the actual listbox element.
    expect(combobox.getAttribute('aria-controls')).toBe(listbox.getAttribute('id'))

    const options = within(listbox).getAllByRole('option')
    expect(options).toHaveLength(3)

    // Opening highlights the current selection (index 0 = gsm8k).
    const active = combobox.getAttribute('aria-activedescendant')
    expect(active).toBe(options[0].getAttribute('id'))
    expect(options[0]).toHaveAttribute('aria-selected', 'true')
  })
})

describe('Combobox — keyboard navigation (Req 10.5)', () => {
  it('moves the highlight down and up with the arrow keys', () => {
    renderCombobox({ value: 'gsm8k' })

    const combobox = screen.getByRole('combobox')
    // First ArrowDown opens the popup (highlight on current selection, index 0).
    fireEvent.keyDown(combobox, { key: 'ArrowDown' })
    const options = within(screen.getByRole('listbox')).getAllByRole('option')
    expect(combobox.getAttribute('aria-activedescendant')).toBe(options[0].getAttribute('id'))

    // Subsequent ArrowDown moves the highlight to the next option.
    fireEvent.keyDown(combobox, { key: 'ArrowDown' })
    expect(combobox.getAttribute('aria-activedescendant')).toBe(options[1].getAttribute('id'))

    // ArrowUp moves it back.
    fireEvent.keyDown(combobox, { key: 'ArrowUp' })
    expect(combobox.getAttribute('aria-activedescendant')).toBe(options[0].getAttribute('id'))
  })

  it('selects the highlighted option on Enter and fires onChange', () => {
    const onChange = vi.fn()
    renderCombobox({ value: 'gsm8k', onChange })

    const combobox = screen.getByRole('combobox')
    fireEvent.keyDown(combobox, { key: 'ArrowDown' }) // open, highlight index 0
    fireEvent.keyDown(combobox, { key: 'ArrowDown' }) // move to index 1 (mmlu)
    fireEvent.keyDown(combobox, { key: 'Enter' })

    expect(onChange).toHaveBeenCalledTimes(1)
    expect(onChange).toHaveBeenCalledWith('mmlu')
    // The popup closes after a selection.
    expect(screen.queryByRole('listbox')).toBeNull()
  })

  it('closes the popup on Escape without changing the value', () => {
    const onChange = vi.fn()
    renderCombobox({ value: 'gsm8k', onChange })

    const combobox = screen.getByRole('combobox')
    fireEvent.click(combobox)
    expect(screen.getByRole('listbox')).toBeInTheDocument()

    fireEvent.keyDown(combobox, { key: 'Escape' })
    expect(screen.queryByRole('listbox')).toBeNull()
    expect(combobox).toHaveAttribute('aria-expanded', 'false')
    expect(onChange).not.toHaveBeenCalled()
  })
})

describe('Combobox — pointer selection (Req 10.5)', () => {
  it('selects an option on click', () => {
    const onChange = vi.fn()
    renderCombobox({ value: 'gsm8k', onChange })

    const combobox = screen.getByRole('combobox')
    fireEvent.click(combobox)

    const options = within(screen.getByRole('listbox')).getAllByRole('option')
    fireEvent.click(options[2])

    expect(onChange).toHaveBeenCalledWith('arc')
  })
})

describe('Combobox — accessibility (axe)', () => {
  it('has no axe violations while open', async () => {
    const { container } = renderCombobox({ value: 'gsm8k' })

    // Open so the listbox referenced by aria-controls / aria-activedescendant
    // exists in the DOM during analysis.
    fireEvent.click(screen.getByRole('combobox'))
    expect(screen.getByRole('listbox')).toBeInTheDocument()

    // axe-core schedules work on real timers; restore them for the analysis.
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
