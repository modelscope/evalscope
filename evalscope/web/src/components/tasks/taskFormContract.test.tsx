// Shared-contract tests for the task configuration forms.
//
// `useTaskForm` + `TaskFormShell` own the submit protocol for both the Eval and
// the Performance form: collect the errors, block submission, focus the first
// invalid field in DOM order and expand the "More Parameters" disclosure when
// that field is hidden inside it. Those behaviours are asserted once here and
// parameterized over both forms, so the two cannot drift apart again — and so
// the Performance form is covered by the same suite as its Eval twin rather
// than being left untested because its code was a copy.
//
// Form-specific payload assembly stays in the per-form suites
// (see EvalConfigForm.test.tsx for the sandbox payload).

import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ReactElement } from 'react'
import { LocaleProvider } from '@/contexts/LocaleContext'

vi.mock('@/api/eval', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/eval')>()
  return { ...actual, listBenchmarks: vi.fn(() => new Promise(() => {})) }
})

import EvalConfigForm from '@/components/eval/EvalConfigForm'
import PerfConfigForm from '@/components/perf/PerfConfigForm'

afterEach(cleanup)

interface FormCase {
  name: string
  render: (onSubmit: (config: Record<string, unknown>) => void) => ReactElement
  /** Accessible name of the submit button. */
  submitLabel: RegExp
  /** Fields that must be filled for submission to be attempted. */
  fillRequired: () => void
  /** A field that lives inside the "More Parameters" disclosure. */
  hiddenFieldLabel: string
  /** A visible numeric field, and a value outside its constraints. */
  visibleInvalid: { label: RegExp; value: string }
  /** Label of the field expected to be reported first when everything is blank. */
  firstRequiredLabel: RegExp
}

const CASES: FormCase[] = [
  {
    name: 'EvalConfigForm',
    render: (onSubmit) => <EvalConfigForm onSubmit={onSubmit} />,
    submitLabel: /Start Evaluation/i,
    fillRequired: () => {
      fireEvent.change(screen.getByLabelText(/Model Name/), { target: { value: 'qwen-plus' } })
      fireEvent.change(screen.getByLabelText(/^Datasets/), { target: { value: 'gsm8k' } })
    },
    // Temperature is capped at 2 and lives behind the disclosure.
    hiddenFieldLabel: 'Temperature',
    visibleInvalid: { label: /^Limit/, value: '0' },
    firstRequiredLabel: /Model Name/,
  },
  {
    name: 'PerfConfigForm',
    render: (onSubmit) => <PerfConfigForm onSubmit={onSubmit} />,
    submitLabel: /Start Performance Test/i,
    fillRequired: () => {
      fireEvent.change(screen.getByLabelText(/Model Name/), { target: { value: 'qwen-plus' } })
    },
    // The tokenizer path is the only field behind the perf disclosure.
    hiddenFieldLabel: 'Tokenizer Path',
    visibleInvalid: { label: /Rate Limit/, value: '-1' },
    firstRequiredLabel: /Model Name/,
  },
]

describe.each(CASES)('$name submit contract', (formCase) => {
  const renderForm = () => {
    const onSubmit = vi.fn()
    render(<LocaleProvider>{formCase.render(onSubmit)}</LocaleProvider>)
    return onSubmit
  }

  const submit = () => fireEvent.click(screen.getByRole('button', { name: formCase.submitLabel }))

  it('submits once every required field is filled', () => {
    const onSubmit = renderForm()
    formCase.fillRequired()

    submit()

    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('blocks submission and reports the missing required field', () => {
    const onSubmit = renderForm()

    submit()

    expect(onSubmit).not.toHaveBeenCalled()
    expect(screen.getByLabelText(formCase.firstRequiredLabel)).toHaveAttribute('aria-invalid', 'true')
  })

  it('moves focus to the first invalid field in DOM order', async () => {
    renderForm()

    submit()

    await act(async () => {
      vi.runOnlyPendingTimers()
      await Promise.resolve()
    })

    expect(screen.getByLabelText(formCase.firstRequiredLabel)).toHaveFocus()
  })

  it('blocks submission when a visible numeric field violates its constraint', () => {
    const onSubmit = renderForm()
    formCase.fillRequired()
    fireEvent.change(screen.getByLabelText(formCase.visibleInvalid.label), {
      target: { value: formCase.visibleInvalid.value },
    })

    submit()

    expect(onSubmit).not.toHaveBeenCalled()
    expect(screen.getByLabelText(formCase.visibleInvalid.label)).toHaveAttribute('aria-invalid', 'true')
  })

  it('clears a field error as the user edits the field', () => {
    renderForm()

    submit()
    const field = screen.getByLabelText(formCase.firstRequiredLabel)
    expect(field).toHaveAttribute('aria-invalid', 'true')

    fireEvent.change(field, { target: { value: 'qwen-plus' } })
    expect(field).not.toHaveAttribute('aria-invalid', 'true')
  })

  it('toggles the More Parameters disclosure', () => {
    renderForm()
    const toggle = screen.getByRole('button', { name: /More Parameters/i })

    expect(screen.queryByLabelText(formCase.hiddenFieldLabel)).not.toBeInTheDocument()
    fireEvent.click(toggle)
    expect(screen.getByLabelText(formCase.hiddenFieldLabel)).toBeInTheDocument()
    fireEvent.click(toggle)
    expect(screen.queryByLabelText(formCase.hiddenFieldLabel)).not.toBeInTheDocument()
  })
})

describe('EvalConfigForm disclosure expansion', () => {
  it('expands More Parameters when the first invalid field is hidden inside it', () => {
    const onSubmit = vi.fn()
    render(
      <LocaleProvider>
        <EvalConfigForm onSubmit={onSubmit} />
      </LocaleProvider>,
    )

    fireEvent.change(screen.getByLabelText(/Model Name/), { target: { value: 'qwen-plus' } })
    fireEvent.change(screen.getByLabelText(/^Datasets/), { target: { value: 'gsm8k' } })

    // Set an out-of-range temperature, then collapse the section again so the
    // offending field is hidden when submission is attempted.
    const toggle = screen.getByRole('button', { name: /More Parameters/i })
    fireEvent.click(toggle)
    fireEvent.change(screen.getByLabelText('Temperature'), { target: { value: '5' } })
    fireEvent.click(toggle)
    expect(screen.queryByLabelText('Temperature')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /Start Evaluation/i }))

    expect(onSubmit).not.toHaveBeenCalled()
    // Re-opened so the user can see and fix the field being complained about.
    expect(screen.getByLabelText('Temperature')).toHaveAttribute('aria-invalid', 'true')
  })
})
