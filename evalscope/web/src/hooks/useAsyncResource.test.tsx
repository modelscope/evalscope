import { describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach } from 'vitest'

import { useAsyncResource } from './useAsyncResource'
import { DomainError } from '@/api/errors'

afterEach(cleanup)

/** Renders a resource's whole surface so each branch is assertable from the DOM. */
function Probe({
  fetcher,
  deps = [],
  enabled,
  fallbackMessage,
}: {
  fetcher: (signal: AbortSignal) => Promise<string>
  deps?: readonly unknown[]
  enabled?: boolean
  fallbackMessage?: string
}) {
  const { data, loading, error, reload } = useAsyncResource(fetcher, deps, { enabled, fallbackMessage })
  return (
    <div>
      <span data-testid="data">{data ?? '-'}</span>
      <span data-testid="loading">{String(loading)}</span>
      <span data-testid="error">{error || '-'}</span>
      <button onClick={reload}>reload</button>
    </div>
  )
}

const read = (id: string) => screen.getByTestId(id).textContent

async function flush() {
  await act(async () => { await Promise.resolve() })
}

describe('useAsyncResource', () => {
  it('exposes the resolved value and clears the loading flag', async () => {
    render(<Probe fetcher={async () => 'first'} />)
    expect(read('loading')).toBe('true')

    await flush()
    expect(read('data')).toBe('first')
    expect(read('loading')).toBe('false')
    expect(read('error')).toBe('-')
  })

  it('surfaces a rejection message and recovers on reload', async () => {
    let attempt = 0
    const fetcher = async () => {
      attempt += 1
      if (attempt === 1) throw new Error('boom')
      return 'recovered'
    }

    render(<Probe fetcher={fetcher} />)
    await flush()
    expect(read('error')).toBe('boom')
    expect(read('loading')).toBe('false')

    await act(async () => { screen.getByText('reload').click() })
    await flush()
    expect(read('error')).toBe('-')
    expect(read('data')).toBe('recovered')
    expect(attempt).toBe(2)
  })

  it('falls back to the supplied message when the rejection carries none', async () => {
    render(<Probe fetcher={async () => { throw 'no message' }} fallbackMessage="could not load" />)
    await flush()
    expect(read('error')).toBe('could not load')
  })

  it('drops an aborted read instead of reporting it as a failure', async () => {
    const fetcher = async (signal: AbortSignal) => {
      await Promise.resolve()
      // Model the API layer: a superseded request rejects with an `aborted` DomainError.
      if (signal.aborted) throw new DomainError('aborted', 'aborted')
      return 'value'
    }

    const view = render(<Probe fetcher={fetcher} deps={['a']} />)
    // Re-key the read before the first attempt settles: the first is aborted.
    view.rerender(<Probe fetcher={fetcher} deps={['b']} />)
    await flush()

    expect(read('error')).toBe('-')
    expect(read('data')).toBe('value')
    expect(read('loading')).toBe('false')
  })

  it('does not read at all while disabled, then reads once enabled', async () => {
    const fetcher = vi.fn(async () => 'value')

    const view = render(<Probe fetcher={fetcher} enabled={false} />)
    await flush()
    expect(fetcher).not.toHaveBeenCalled()
    expect(read('loading')).toBe('false')
    expect(read('data')).toBe('-')

    view.rerender(<Probe fetcher={fetcher} enabled />)
    await flush()
    expect(fetcher).toHaveBeenCalledTimes(1)
    expect(read('data')).toBe('value')
  })

  it('leaves loading and error state when disabled during an active read', async () => {
    let rejectRead: ((reason: Error) => void) | undefined
    const fetcher = vi.fn(
      () => new Promise<string>((_resolve, reject) => { rejectRead = reject }),
    )

    const view = render(<Probe fetcher={fetcher} enabled />)
    expect(read('loading')).toBe('true')

    view.rerender(<Probe fetcher={fetcher} enabled={false} />)
    await flush()

    expect(read('loading')).toBe('false')
    expect(read('error')).toBe('-')

    // A late rejection from the disabled attempt must remain inert.
    rejectRead?.(new Error('late failure'))
    await flush()
    expect(read('loading')).toBe('false')
    expect(read('error')).toBe('-')
  })

  it('re-reads when the dependency list changes and keeps the previous value meanwhile', async () => {
    const fetcher = vi.fn(async () => 'value')

    const view = render(<Probe fetcher={fetcher} deps={['a']} />)
    await flush()
    expect(fetcher).toHaveBeenCalledTimes(1)

    // Same deps: no extra read.
    view.rerender(<Probe fetcher={fetcher} deps={['a']} />)
    await flush()
    expect(fetcher).toHaveBeenCalledTimes(1)

    view.rerender(<Probe fetcher={fetcher} deps={['b']} />)
    // The prior value stays on screen while the new read is in flight.
    expect(read('data')).toBe('value')
    await flush()
    expect(fetcher).toHaveBeenCalledTimes(2)
  })
})
