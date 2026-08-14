import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { useScopedState } from './useScopedState'

afterEach(cleanup)

/** Exposes the whole hook surface so each branch is assertable from the DOM. */
function Probe({ scope, fallback = 'none' }: { scope: string; fallback?: string }) {
  const [value, setValue] = useScopedState<string>(scope, fallback)
  return (
    <div>
      <span data-testid="value">{value}</span>
      <button onClick={() => setValue('picked')}>pick</button>
      <button onClick={() => setValue((current) => `${current}+`)}>append</button>
    </div>
  )
}

const read = () => screen.getByTestId('value').textContent

describe('useScopedState', () => {
  it('starts at the fallback and holds a value written in the same scope', () => {
    render(<Probe scope="a" />)
    expect(read()).toBe('none')

    act(() => { screen.getByText('pick').click() })
    expect(read()).toBe('picked')
  })

  it('abandons a value once the scope changes', () => {
    const view = render(<Probe scope="a" />)
    act(() => { screen.getByText('pick').click() })
    expect(read()).toBe('picked')

    view.rerender(<Probe scope="b" />)
    expect(read()).toBe('none')
  })

  it('reveals the earlier value again when the original scope comes back', () => {
    // One value is held, keyed by the scope it was written in — it is not cleared
    // on the way out. Returning to a scope therefore restores the choice made
    // there, which is what lets a user navigate away from a report and back
    // without losing the dataset they were reading. Only one scope is remembered:
    // a second scope's value overwrites the first.
    const view = render(<Probe scope="a" />)
    act(() => { screen.getByText('pick').click() })

    view.rerender(<Probe scope="b" />)
    expect(read()).toBe('none')

    view.rerender(<Probe scope="a" />)
    expect(read()).toBe('picked')
  })

  it('forgets the first scope once another scope writes a value', () => {
    const view = render(<Probe scope="a" />)
    act(() => { screen.getByText('pick').click() })

    view.rerender(<Probe scope="b" />)
    act(() => { screen.getByText('append').click() })

    view.rerender(<Probe scope="a" />)
    expect(read()).toBe('none')
  })

  it('resolves an updater against the in-scope value', () => {
    render(<Probe scope="a" />)
    act(() => { screen.getByText('pick').click() })
    act(() => { screen.getByText('append').click() })

    expect(read()).toBe('picked+')
  })

  it('resolves an updater against the fallback after a scope change', () => {
    // The updater must not see the previous scope's value, which is what keeps a
    // stale selection from being merged into a new scope.
    const view = render(<Probe scope="a" />)
    act(() => { screen.getByText('pick').click() })

    view.rerender(<Probe scope="b" />)
    act(() => { screen.getByText('append').click() })

    expect(read()).toBe('none+')
  })

  it('keeps the setter stable while the scope holds', () => {
    // Callers put the setter in effect dependency arrays, so a new identity per
    // render would re-fire those effects. Every call site passes a stable
    // fallback (a module constant or a literal `null`), which is what the hook's
    // contract requires.
    const setters: unknown[] = []
    function IdentityProbe({ scope }: { scope: string }) {
      const [, setValue] = useScopedState<string>(scope, 'none')
      setters.push(setValue)
      return null
    }

    const view = render(<IdentityProbe scope="a" />)
    view.rerender(<IdentityProbe scope="a" />)
    view.rerender(<IdentityProbe scope="a" />)

    expect(new Set(setters).size).toBe(1)
  })
})
