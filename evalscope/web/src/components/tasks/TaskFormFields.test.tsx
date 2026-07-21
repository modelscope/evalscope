import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { ApiKeyField, ApiUrlField, ModelField } from './TaskFormFields'

afterEach(cleanup)

describe('TaskFormFields', () => {
  it('preserves task-specific names while sharing connection-field behavior', () => {
    const onModelChange = vi.fn()
    render(
      <LocaleProvider>
        <ModelField id="model" value="" error="Required" onChange={onModelChange} />
        <ApiUrlField id="url" name="url" value="" onChange={() => {}} />
        <ApiKeyField id="key" value="" onChange={() => {}} />
      </LocaleProvider>,
    )

    const model = screen.getByLabelText(/Model Name/)
    fireEvent.change(model, { target: { value: 'qwen-plus' } })
    expect(onModelChange).toHaveBeenCalledWith('qwen-plus')
    expect(screen.getByLabelText(/API URL/)).toHaveAttribute('name', 'url')
    expect(screen.getByLabelText(/API Key/)).toHaveAttribute('type', 'password')
  })
})
