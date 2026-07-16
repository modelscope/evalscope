// Component and accessibility unit tests for the Field primitive (Task 5.8).
//
// Field is a control-agnostic form primitive: it renders a localized <label>
// linked to the control supplied through a render prop, and reserves a live
// region for validation errors. These tests cover the parts of the
// accessibility contract that the pure validation logic cannot express:
//   - every control exposes a non-empty, programmatically associated
//     accessible name;
//   - when an error is present the control is marked aria-invalid, associated
//     with the error element via aria-describedby, and the error container is a
//     polite live region with role="alert";
//   - the autoComplete hint is forwarded to the control, including API-key
//     scenarios;
//   - a rendered field has no axe accessibility violations.
//
// The suite runs under the global deterministic setup (fake timers, fixed
// system time, network disabled). The axe assertion temporarily restores real
// timers because axe-core schedules its analysis on real timers, which fake
// timers would otherwise stall (mirrors Tabs.test.tsx).

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { axe } from 'jest-axe'

import Field, { type FieldAriaProps } from './Field'
import { LocaleProvider } from '@/contexts/LocaleContext'

const FIXED_SYSTEM_TIME = new Date('2026-07-01T00:00:00.000Z')

/** Render a Field wrapping a text input inside the app's Locale provider. */
function renderTextField(
  props: Partial<React.ComponentProps<typeof Field>> = {},
  renderControl: (aria: FieldAriaProps) => React.ReactNode = (aria) => <input type="text" {...aria} />,
) {
  const merged = {
    id: 'model-name',
    labelKey: 'fields.model.label',
    name: 'model_name',
    ...props,
  } as React.ComponentProps<typeof Field>

  return render(
    <LocaleProvider>
      <Field {...merged}>{renderControl}</Field>
    </LocaleProvider>,
  )
}

afterEach(() => {
  cleanup()
})

describe('Field — accessible name', () => {
  it('gives the control a non-empty accessible name via aria-labelledby', () => {
    renderTextField()

    const input = screen.getByRole('textbox')
    // The control's name is derived programmatically from the visible label.
    expect(input).toHaveAccessibleName()
    expect(input.getAttribute('aria-labelledby')).toBeTruthy()

    // The referenced label element exists and carries the (non-empty) label text.
    const labelId = input.getAttribute('aria-labelledby') as string
    const label = document.getElementById(labelId)
    expect(label).not.toBeNull()
    expect(label?.textContent?.trim().length ?? 0).toBeGreaterThan(0)
  })

  it('links the label to the control with matching htmlFor / id', () => {
    renderTextField()

    const input = screen.getByRole('textbox')
    const label = document.getElementById('model-name-label') as HTMLLabelElement
    expect(label).not.toBeNull()
    expect(label.getAttribute('for')).toBe(input.getAttribute('id'))
    expect(input).toHaveAttribute('id', 'model-name')
    expect(input).toHaveAttribute('name', 'model_name')
  })
})

describe('Field — error association and live region', () => {
  it('has no aria-invalid=true and no describedby when there is no error', () => {
    renderTextField()

    const input = screen.getByRole('textbox')
    expect(input).toHaveAttribute('aria-invalid', 'false')
    expect(input).not.toHaveAttribute('aria-describedby')
  })

  it('marks the control invalid and associates it with the error element', () => {
    renderTextField({ error: 'Model name is required' })

    const input = screen.getByRole('textbox')
    expect(input).toHaveAttribute('aria-invalid', 'true')

    const describedBy = input.getAttribute('aria-describedby')
    expect(describedBy).toBe('model-name-error')

    const errorEl = document.getElementById(describedBy as string)
    expect(errorEl).not.toBeNull()
    expect(errorEl).toHaveTextContent('Model name is required')
  })

  it('renders the error container as a polite live region (role=alert, aria-live=polite)', () => {
    renderTextField({ error: 'Model name is required' })

    const alert = screen.getByRole('alert')
    expect(alert).toHaveAttribute('aria-live', 'polite')
    expect(alert).toHaveAttribute('id', 'model-name-error')
    expect(alert).toHaveTextContent('Model name is required')
  })
})

describe('Field — autoComplete passthrough', () => {
  it('forwards the autoComplete hint to the control', () => {
    renderTextField({ autoComplete: 'off' })

    expect(screen.getByRole('textbox')).toHaveAttribute('autocomplete', 'off')
  })

  it('forwards an API-key autocomplete hint and accepts typed input', () => {
    // API key fields opt out of browser autofill of stored credentials.
    renderTextField(
      { id: 'api-key', labelKey: 'fields.apiKey.label', name: 'api_key', autoComplete: 'off' },
      (aria) => <input type="password" {...aria} />,
    )

    const input = document.getElementById('api-key') as HTMLInputElement
    expect(input).toHaveAttribute('autocomplete', 'off')
    expect(input).toHaveAttribute('name', 'api_key')

    fireEvent.change(input, { target: { value: 'sk-secret-123' } })
    expect(input).toHaveValue('sk-secret-123')
  })
})

describe('Field — accessibility (axe)', () => {
  it('has no axe violations with an associated error', async () => {
    const { container } = renderTextField({ error: 'Model name is required' })

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
