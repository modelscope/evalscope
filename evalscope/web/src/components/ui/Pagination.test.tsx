import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import Pagination from './Pagination'

afterEach(cleanup)

describe('Pagination', () => {
  it('renders nothing when there is only one page', () => {
    const { container } = render(<Pagination page={1} totalPages={1} onPageChange={() => {}} />)
    expect(container).toBeEmptyDOMElement()
  })

  it('moves to adjacent and explicit pages', () => {
    const onPageChange = vi.fn()
    render(<Pagination page={4} totalPages={8} onPageChange={onPageChange} />)

    fireEvent.click(screen.getByRole('button', { name: '←' }))
    fireEvent.click(screen.getByRole('button', { name: '6' }))
    fireEvent.click(screen.getByRole('button', { name: '→' }))

    expect(onPageChange.mock.calls).toEqual([[3], [6], [5]])
    expect(screen.getAllByText('...')).toHaveLength(1)
  })
})
