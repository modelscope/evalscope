import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import PromptDialog from './PromptDialog'

afterEach(cleanup)

const baseProps = {
  title: 'Rename report',
  label: 'Model name',
  value: 'model-a',
  confirmLabel: 'Rename',
  cancelLabel: 'Cancel',
  onValueChange: () => {},
  onConfirm: () => {},
  onCancel: () => {},
}

describe('PromptDialog', () => {
  it('renders nothing when closed', () => {
    render(<PromptDialog {...baseProps} open={false} />)
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('shows title, label and the current value, focused and selected', () => {
    render(<PromptDialog {...baseProps} open />)
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    expect(screen.getByText('Rename report')).toBeInTheDocument()
    const input = screen.getByLabelText('Model name') as HTMLInputElement
    expect(input).toHaveValue('model-a')
    expect(input).toHaveFocus()
  })

  it('reports edits via onValueChange', () => {
    const onValueChange = vi.fn()
    render(<PromptDialog {...baseProps} open onValueChange={onValueChange} />)
    fireEvent.change(screen.getByLabelText('Model name'), { target: { value: 'model-a-v2' } })
    expect(onValueChange).toHaveBeenCalledWith('model-a-v2')
  })

  it('confirms on click and on Enter', () => {
    const onConfirm = vi.fn()
    render(<PromptDialog {...baseProps} open onConfirm={onConfirm} />)
    fireEvent.click(screen.getByRole('button', { name: 'Rename' }))
    expect(onConfirm).toHaveBeenCalledTimes(1)
    fireEvent.keyDown(document, { key: 'Enter' })
    expect(onConfirm).toHaveBeenCalledTimes(2)
  })

  it('cancels on click and on Escape, but not while busy', () => {
    const onCancel = vi.fn()
    const { rerender } = render(<PromptDialog {...baseProps} open onCancel={onCancel} />)
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onCancel).toHaveBeenCalledTimes(1)

    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onCancel).toHaveBeenCalledTimes(2)

    rerender(<PromptDialog {...baseProps} open busy onCancel={onCancel} />)
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onCancel).toHaveBeenCalledTimes(2)
  })

  it('shows a field error and marks the input invalid', () => {
    render(<PromptDialog {...baseProps} open error="Name already in use" />)
    expect(screen.getByText('Name already in use')).toBeInTheDocument()
    expect(screen.getByLabelText('Model name')).toHaveAttribute('aria-invalid', 'true')
  })

  it('disables the input and both actions while busy', () => {
    render(<PromptDialog {...baseProps} open busy />)
    expect(screen.getByLabelText('Model name')).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Rename' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled()
  })
})
