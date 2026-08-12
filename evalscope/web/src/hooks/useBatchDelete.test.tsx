import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useBatchDelete } from './useBatchDelete'

afterEach(cleanup)

/** Exposes the hook's surface so each branch is assertable from the DOM. */
function Probe({
  items,
  deleteItem,
  onSettled = () => {},
  reload = () => {},
}: {
  items: string[]
  deleteItem: (item: string) => Promise<unknown>
  onSettled?: (remaining: string[]) => void
  reload?: () => void
}) {
  const deletion = useBatchDelete<string>({
    items,
    deleteItem,
    onSettled,
    reload,
    formatError: (msg) => `failed: ${msg}`,
  })
  return (
    <div>
      <span data-testid="deleting">{String(deletion.deleting)}</span>
      <span data-testid="open">{String(deletion.confirmOpen)}</span>
      <span data-testid="error">{deletion.error ?? '-'}</span>
      <button onClick={deletion.request}>request</button>
      <button onClick={deletion.cancel}>cancel</button>
      <button onClick={() => void deletion.confirm()}>confirm</button>
    </div>
  )
}

const read = (id: string) => screen.getByTestId(id).textContent

describe('useBatchDelete', () => {
  it('opens confirmation only when something is selected', () => {
    const view = render(<Probe items={[]} deleteItem={async () => {}} />)
    act(() => { screen.getByText('request').click() })
    expect(read('open')).toBe('false')

    view.rerender(<Probe items={['a']} deleteItem={async () => {}} />)
    act(() => { screen.getByText('request').click() })
    expect(read('open')).toBe('true')
  })

  it('clears the selection and closes after deleting everything', async () => {
    const deleteItem = vi.fn(async () => {})
    const onSettled = vi.fn()
    render(<Probe items={['a', 'b']} deleteItem={deleteItem} onSettled={onSettled} />)

    await act(async () => { await screen.getByText('confirm').click() })

    expect(deleteItem).toHaveBeenCalledTimes(2)
    expect(onSettled).toHaveBeenCalledWith([])
    expect(read('open')).toBe('false')
    expect(read('deleting')).toBe('false')
    expect(read('error')).toBe('-')
  })

  it('deletes sequentially so a failure leaves a known boundary', async () => {
    const order: string[] = []
    const deleteItem = vi.fn(async (item: string) => {
      order.push(item)
      if (item === 'b') throw new Error('server said no')
    })
    render(<Probe items={['a', 'b', 'c']} deleteItem={deleteItem} />)

    await act(async () => { await screen.getByText('confirm').click() })

    // `c` is never attempted: the loop stops at the first rejection.
    expect(order).toEqual(['a', 'b'])
  })

  it('keeps the items that were not deleted and surfaces the failure', async () => {
    const deleteItem = vi.fn(async (item: string) => {
      if (item === 'b') throw new Error('server said no')
    })
    const onSettled = vi.fn()
    render(<Probe items={['a', 'b', 'c']} deleteItem={deleteItem} onSettled={onSettled} />)

    await act(async () => { await screen.getByText('confirm').click() })

    // `a` succeeded and is dropped; `b` and `c` are still on the server.
    expect(onSettled).toHaveBeenCalledWith(['b', 'c'])
    expect(read('error')).toBe('failed: server said no')
  })

  it('re-reads the list even when the attempt failed', async () => {
    // A partial deletion changed the server, so the stale list must not stand.
    const reload = vi.fn()
    render(
      <Probe
        items={['a']}
        deleteItem={async () => { throw new Error('nope') }}
        reload={reload}
      />,
    )

    await act(async () => { await screen.getByText('confirm').click() })

    expect(reload).toHaveBeenCalledTimes(1)
  })

  it('ignores a confirm with nothing selected', async () => {
    const deleteItem = vi.fn(async () => {})
    const onSettled = vi.fn()
    render(<Probe items={[]} deleteItem={deleteItem} onSettled={onSettled} />)

    await act(async () => { await screen.getByText('confirm').click() })

    expect(deleteItem).not.toHaveBeenCalled()
    expect(onSettled).not.toHaveBeenCalled()
  })

  it('closes the confirmation without deleting on cancel', () => {
    const deleteItem = vi.fn(async () => {})
    render(<Probe items={['a']} deleteItem={deleteItem} />)

    act(() => { screen.getByText('request').click() })
    act(() => { screen.getByText('cancel').click() })

    expect(read('open')).toBe('false')
    expect(deleteItem).not.toHaveBeenCalled()
  })
})
