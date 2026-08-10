// Component tests for the sandbox section of EvalConfigForm.
//
// The sandbox payload-assembly logic went through several review rounds on
// PR #1545: the bug that actually shipped was that optional fields (manager
// URL, docker image, pool size) were sent as empty/NaN values instead of
// being omitted, which silently overrides backend defaults (see
// CodeExecutionSandboxMixin._resolve_sandbox_config_dict, which merges
// default_config on top of BenchmarkMeta.sandbox_config via dict.update).
// These tests lock in the corrected behaviour so it doesn't regress:
//   - sandbox disabled -> no `sandbox` key in the submitted config;
//   - enabled with all optional fields blank -> only `enabled` / `engine`;
//   - enabled with every field filled -> nested `manager_config` /
//     `default_config` / `pool_size` appear;
//   - an invalid pool size (0) blocks submit and marks the field invalid.

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { LocaleProvider } from '@/contexts/LocaleContext'

vi.mock('@/api/eval', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/eval')>()
  return { ...actual, listBenchmarks: vi.fn().mockResolvedValue({ text: [], multimodal: [] }) }
})

import EvalConfigForm from './EvalConfigForm'

afterEach(cleanup)

function renderForm(onSubmit = vi.fn()) {
  render(
    <LocaleProvider>
      <EvalConfigForm onSubmit={onSubmit} />
    </LocaleProvider>,
  )
  return onSubmit
}

/** Fills the two required fields and opens the "More Parameters" section where sandbox lives. */
function fillRequiredAndExpand() {
  fireEvent.change(screen.getByLabelText(/Model Name/), { target: { value: 'qwen-plus' } })
  fireEvent.change(screen.getByLabelText(/^Datasets/), { target: { value: 'gsm8k' } })
  fireEvent.click(screen.getByRole('button', { name: /More Parameters/i }))
}

function submit() {
  fireEvent.click(screen.getByRole('button', { name: /Start Evaluation/i }))
}

describe('EvalConfigForm sandbox payload', () => {
  it('omits the sandbox key entirely when sandbox is disabled', () => {
    const onSubmit = renderForm()
    fillRequiredAndExpand()

    submit()

    expect(onSubmit).toHaveBeenCalledTimes(1)
    const config = onSubmit.mock.calls[0][0]
    expect(config).not.toHaveProperty('sandbox')
  })

  it('sends only enabled/engine when optional fields are left blank', () => {
    const onSubmit = renderForm()
    fillRequiredAndExpand()
    fireEvent.click(screen.getByLabelText(/Enable Sandbox/))

    submit()

    expect(onSubmit).toHaveBeenCalledTimes(1)
    const config = onSubmit.mock.calls[0][0]
    expect(config.sandbox).toEqual({ enabled: true, engine: 'docker' })
  })

  it('nests manager_config/default_config/pool_size when every field is filled', () => {
    const onSubmit = renderForm()
    fillRequiredAndExpand()
    fireEvent.click(screen.getByLabelText(/Enable Sandbox/))
    fireEvent.change(screen.getByLabelText('Engine'), { target: { value: 'volcengine' } })
    fireEvent.change(screen.getByLabelText('Manager URL'), { target: { value: 'https://sandbox.example.com' } })
    fireEvent.change(screen.getByLabelText('Docker Image'), { target: { value: 'my-image:latest' } })
    fireEvent.change(screen.getByLabelText('Pool Size'), { target: { value: '3' } })

    submit()

    expect(onSubmit).toHaveBeenCalledTimes(1)
    const config = onSubmit.mock.calls[0][0]
    expect(config.sandbox).toEqual({
      enabled: true,
      engine: 'volcengine',
      manager_config: { base_url: 'https://sandbox.example.com' },
      default_config: { image: 'my-image:latest' },
      pool_size: 3,
    })
  })

  it('blocks submit and marks the pool size field invalid when pool size is 0', () => {
    const onSubmit = renderForm()
    fillRequiredAndExpand()
    fireEvent.click(screen.getByLabelText(/Enable Sandbox/))
    fireEvent.change(screen.getByLabelText('Pool Size'), { target: { value: '0' } })

    submit()

    expect(onSubmit).not.toHaveBeenCalled()
    expect(screen.getByLabelText('Pool Size')).toHaveAttribute('aria-invalid', 'true')
  })
})
