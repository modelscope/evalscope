import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { LocaleProvider } from '@/contexts/LocaleContext'
import SelectionCheckbox from '@/components/ui/SelectionCheckbox'
import SelectionTray from './SelectionTray'

afterEach(cleanup)

describe('selection controls', () => {
  it('exposes checkbox state and forwards clicks', () => {
    const onClick = vi.fn()
    render(<SelectionCheckbox checked label="Select run" onClick={onClick} />)

    const checkbox = screen.getByRole('checkbox', { name: 'Select run' })
    expect(checkbox).toHaveAttribute('aria-checked', 'true')
    expect(checkbox.className).toContain('min-h-[44px]')
    fireEvent.click(checkbox)
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it('prevents selection changes when disabled', () => {
    const onClick = vi.fn()
    render(<SelectionCheckbox checked={false} disabled label="Unavailable run" onClick={onClick} />)

    const checkbox = screen.getByRole('checkbox', { name: 'Unavailable run' })
    expect(checkbox).toBeDisabled()
    fireEvent.click(checkbox)
    expect(onClick).not.toHaveBeenCalled()
  })

  it('shares the view, compare, compare-limit notice, and clear behavior', () => {
    const onViewHtml = vi.fn()
    const onCompare = vi.fn()
    const onClear = vi.fn()
    render(
      <LocaleProvider>
        <SelectionTray
          count={4}
          compareLimit={3}
          canViewHtml={false}
          onViewHtml={onViewHtml}
          onCompare={onCompare}
          onClear={onClear}
        />
      </LocaleProvider>,
    )

    expect(screen.getByText('Only the first 3 will be compared.')).toHaveAttribute('role', 'status')
    expect(screen.getByRole('button', { name: /View HTML/ })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: /Compare/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Clear' }))
    expect(onCompare).toHaveBeenCalledTimes(1)
    expect(onClear).toHaveBeenCalledTimes(1)
  })

  it('omits the compare-limit notice when the surface compares any number of runs', () => {
    render(
      <LocaleProvider>
        <SelectionTray
          count={7}
          canViewHtml={false}
          onViewHtml={vi.fn()}
          onCompare={vi.fn()}
          onClear={vi.fn()}
        />
      </LocaleProvider>,
    )

    expect(screen.queryByRole('status')).toBeNull()
    expect(screen.getByRole('button', { name: /Compare/ })).toBeEnabled()
  })

  it('omits Delete/Rename/Merge when their handlers are not provided', () => {
    render(
      <LocaleProvider>
        <SelectionTray count={1} canViewHtml onViewHtml={() => {}} onCompare={() => {}} onClear={() => {}} />
      </LocaleProvider>,
    )
    expect(screen.queryByRole('button', { name: /Delete/ })).toBeNull()
    expect(screen.queryByRole('button', { name: /Rename/ })).toBeNull()
    expect(screen.queryByRole('button', { name: /Merge/ })).toBeNull()
  })

  it('wires Delete, Rename and Merge when provided, respecting their enabled flags', () => {
    const onDelete = vi.fn()
    const onRename = vi.fn()
    const onMerge = vi.fn()
    render(
      <LocaleProvider>
        <SelectionTray
          count={2}
          canViewHtml={false}
          onViewHtml={() => {}}
          onCompare={() => {}}
          onClear={() => {}}
          onDelete={onDelete}
          onRename={onRename}
          canRename={false}
          onMerge={onMerge}
          canMerge={true}
        />
      </LocaleProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: /Delete/ }))
    expect(onDelete).toHaveBeenCalledTimes(1)

    // Rename is disabled (canRename=false, e.g. more than one report selected).
    expect(screen.getByRole('button', { name: /Rename/ })).toBeDisabled()

    const mergeButton = screen.getByRole('button', { name: /Merge/ })
    expect(mergeButton).not.toBeDisabled()
    fireEvent.click(mergeButton)
    expect(onMerge).toHaveBeenCalledTimes(1)
  })

  it('disables Merge while merging and Delete while deleting', () => {
    render(
      <LocaleProvider>
        <SelectionTray
          count={2}
          canViewHtml={false}
          onViewHtml={() => {}}
          onCompare={() => {}}
          onClear={() => {}}
          onDelete={() => {}}
          deleting
          onMerge={() => {}}
          canMerge
          merging
        />
      </LocaleProvider>,
    )
    expect(screen.getByRole('button', { name: /Delete/ })).toBeDisabled()
    expect(screen.getByRole('button', { name: /Merge/ })).toBeDisabled()
  })
})
