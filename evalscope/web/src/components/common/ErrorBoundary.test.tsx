import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import ErrorBoundary from './ErrorBoundary'

afterEach(cleanup)

/** Throws on demand so the boundary can be driven from a prop. */
function Bomb({ throws, message = 'kaboom' }: { throws: boolean; message?: string }) {
  if (throws) throw new Error(message)
  return <span>page content</span>
}

const LABELS = { title: 'Page failed', body: 'Nothing to say', action: 'Try again' }

describe('ErrorBoundary', () => {
  beforeEach(() => {
    // The boundary logs the caught error by design; keep the run output readable.
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
  })

  it('renders its children while nothing throws', () => {
    render(
      <ErrorBoundary>
        <Bomb throws={false} />
      </ErrorBoundary>,
    )

    expect(screen.getByText('page content')).toBeInTheDocument()
  })

  it('shows the error message and the supplied labels after a throw', () => {
    render(
      <ErrorBoundary labels={LABELS}>
        <Bomb throws message="schema exploded" />
      </ErrorBoundary>,
    )

    expect(screen.getByText('Page failed')).toBeInTheDocument()
    expect(screen.getByText('schema exploded')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument()
  })

  it('falls back to the supplied body when the error carries no message', () => {
    render(
      <ErrorBoundary labels={LABELS}>
        <Bomb throws message="" />
      </ErrorBoundary>,
    )

    expect(screen.getByText('Nothing to say')).toBeInTheDocument()
  })

  it('recovers through onRecover instead of reloading the document', () => {
    const onRecover = vi.fn()
    render(
      <ErrorBoundary labels={LABELS} onRecover={onRecover}>
        <Bomb throws />
      </ErrorBoundary>,
    )

    screen.getByRole('button', { name: 'Try again' }).click()

    // A route-level boundary remounts the page in place; the rest of the app is
    // still mounted, so a full document reload would be a regression.
    expect(onRecover).toHaveBeenCalledTimes(1)
  })

  it('prefers an explicit fallback node over the default surface', () => {
    render(
      <ErrorBoundary fallback={<span>custom fallback</span>}>
        <Bomb throws />
      </ErrorBoundary>,
    )

    expect(screen.getByText('custom fallback')).toBeInTheDocument()
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('uses English defaults when mounted outside the locale provider', () => {
    // The root boundary has to catch failures in the providers themselves, so it
    // cannot translate and must still say something.
    render(
      <ErrorBoundary>
        <Bomb throws message="" />
      </ErrorBoundary>,
    )

    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reload' })).toBeInTheDocument()
  })
})
