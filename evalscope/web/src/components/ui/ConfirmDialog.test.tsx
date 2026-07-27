import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import ConfirmDialog from './ConfirmDialog'

afterEach(cleanup)

const baseProps = {
  title: 'Delete runs',
  message: 'This cannot be undone.',
  confirmLabel: 'Delete',
  cancelLabel: 'Cancel',
  onConfirm: () => {},
  onCancel: () => {},
}

describe('ConfirmDialog', () => {
  it('renders nothing when closed', () => {
    render(<ConfirmDialog {...baseProps} open={false} />)
    expect(screen.queryByRole('alertdialog')).toBeNull()
  })

  it('shows title, message and affected items when open', () => {
    render(<ConfirmDialog {...baseProps} open items={['run-a', 'run-b']} />)
    expect(screen.getByRole('alertdialog')).toBeInTheDocument()
    expect(screen.getByText('Delete runs')).toBeInTheDocument()
    expect(screen.getByText('This cannot be undone.')).toBeInTheDocument()
    expect(screen.getByText('run-a')).toBeInTheDocument()
    expect(screen.getByText('run-b')).toBeInTheDocument()
  })

  it('focuses cancel first and fires the matching callbacks', () => {
    const onConfirm = vi.fn()
    const onCancel = vi.fn()
    render(<ConfirmDialog {...baseProps} open onConfirm={onConfirm} onCancel={onCancel} />)

    expect(screen.getByRole('button', { name: 'Cancel' })).toHaveFocus()
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onConfirm).toHaveBeenCalledTimes(1)
    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it('cancels on Escape but not while busy', () => {
    const onCancel = vi.fn()
    const { rerender } = render(<ConfirmDialog {...baseProps} open onCancel={onCancel} />)
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onCancel).toHaveBeenCalledTimes(1)

    rerender(<ConfirmDialog {...baseProps} open busy onCancel={onCancel} />)
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it('disables both actions while busy', () => {
    render(<ConfirmDialog {...baseProps} open busy />)
    expect(screen.getByRole('button', { name: 'Delete' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled()
  })
})
