import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { useBatchDelete } from './useBatchDelete'

afterEach(cleanup)

function Probe({
  items,
  deleteItem,
  onSettled,
  reload,
}: {
  items: string[]
  deleteItem: (item: string) => Promise<unknown>
  onSettled: (remaining: string[]) => void
  reload: () => void
}) {
  const deletion = useBatchDelete({
    items,
    deleteItem,
    onSettled,
    reload,
    formatError: (message) => `failed: ${message}`,
  })
  return (
    <div>
      <span data-testid="error">{deletion.error ?? '-'}</span>
      <button onClick={() => void deletion.confirm()}>confirm</button>
    </div>
  )
}

it('clears the selection and reloads after deleting every item', async () => {
  const deleteItem = vi.fn(async () => {})
  const onSettled = vi.fn()
  const reload = vi.fn()
  render(<Probe items={['a', 'b']} deleteItem={deleteItem} onSettled={onSettled} reload={reload} />)

  await act(async () => { screen.getByText('confirm').click() })

  expect(deleteItem).toHaveBeenCalledTimes(2)
  expect(onSettled).toHaveBeenCalledWith([])
  expect(reload).toHaveBeenCalledOnce()
})

it('stops on failure and preserves the unprocessed selection', async () => {
  const deleteItem = vi.fn(async (item: string) => {
    if (item === 'b') throw new Error('server said no')
  })
  const onSettled = vi.fn()
  render(<Probe items={['a', 'b', 'c']} deleteItem={deleteItem} onSettled={onSettled} reload={() => {}} />)

  await act(async () => { screen.getByText('confirm').click() })

  expect(deleteItem.mock.calls.map(([item]) => item)).toEqual(['a', 'b'])
  expect(onSettled).toHaveBeenCalledWith(['b', 'c'])
  expect(screen.getByTestId('error')).toHaveTextContent('failed: server said no')
})
