// Component tests for the Empty_State_System (Task 10.5).
//
// Covers the behaviour that lives in the component layer rather than in the pure
// `emptyState` domain logic:
//   - the three distinguishable reason messages render their localized text;
//   - the empty state is gated on loading and appears within the 300ms reveal
//     budget once a load completes;
//   - each recovery action navigates to its in-product route, while an
//     `onAction` handler that returns `true` intercepts the action in-view and
//     suppresses navigation;
//   - a Performance view with no data offers an action that routes into the
//     Tasks flow rather than relying on CLI text.
//
// The suite runs under the global deterministic setup (fake timers, fixed system
// time, network disabled). Because `useNavigate` is used, the component is
// wrapped in a MemoryRouter; a small LocationDisplay surfaces the current route
// so navigation can be asserted without mocking the router. LocaleProvider
// supplies the (default `en`) locale strings.

import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'

import EmptyStateSystem, {
  MAX_REVEAL_DELAY_MS,
  type EmptyStateSystemProps,
} from './EmptyStateSystem'
import { LocaleProvider } from '@/contexts/LocaleContext'

afterEach(() => {
  cleanup()
})

/** Surfaces the current router location so navigation is observable in the DOM. */
function LocationDisplay() {
  const location = useLocation()
  return <div data-testid="location">{`${location.pathname}${location.search}`}</div>
}

/**
 * Render EmptyStateSystem inside a LocaleProvider (locale = en) and a
 * MemoryRouter, exposing the active route through {@link LocationDisplay}.
 */
function renderEmptyState(
  props: EmptyStateSystemProps,
  { initialEntries = ['/reports'] }: { initialEntries?: string[] } = {},
) {
  return render(
    <LocaleProvider>
      <MemoryRouter initialEntries={initialEntries}>
        <Routes>
          <Route
            path="*"
            element={
              <>
                <EmptyStateSystem {...props} />
                <LocationDisplay />
              </>
            }
          />
        </Routes>
      </MemoryRouter>
    </LocaleProvider>,
  )
}

/** The current router location text, as rendered by {@link LocationDisplay}. */
function currentLocation(): string {
  return screen.getByTestId('location').textContent ?? ''
}

describe('EmptyStateSystem', () => {
  describe('reason messages', () => {
    it.each([
      ['no-data', 'Nothing here yet'],
      ['load-error', 'Something went wrong while loading'],
      ['no-match', 'No results match your filters'],
    ] as const)('renders the localized message for reason=%s', (reason, message) => {
      renderEmptyState({ reason })
      expect(screen.getByText(message)).toBeInTheDocument()
    })
  })

  describe('reveal timing', () => {
    it('renders nothing while loading', () => {
      renderEmptyState({ reason: 'no-data', loading: true })
      expect(screen.queryByText('Nothing here yet')).not.toBeInTheDocument()
    })

    it('reveals within the 300ms budget once loading completes', () => {
      renderEmptyState({ reason: 'no-data', loading: false, revealDelayMs: MAX_REVEAL_DELAY_MS })

      // Not visible immediately: the reveal is deferred by the delay.
      expect(screen.queryByText('Nothing here yet')).not.toBeInTheDocument()

      // Still hidden just before the budget elapses.
      act(() => {
        vi.advanceTimersByTime(MAX_REVEAL_DELAY_MS - 1)
      })
      expect(screen.queryByText('Nothing here yet')).not.toBeInTheDocument()

      // Shown at the 300ms boundary.
      act(() => {
        vi.advanceTimersByTime(1)
      })
      expect(screen.getByText('Nothing here yet')).toBeInTheDocument()
    })

    it('clamps an over-budget reveal delay to 300ms', () => {
      renderEmptyState({ reason: 'no-data', loading: false, revealDelayMs: 5000 })

      // Advancing by the max budget is sufficient because the delay is clamped.
      act(() => {
        vi.advanceTimersByTime(MAX_REVEAL_DELAY_MS)
      })
      expect(screen.getByText('Nothing here yet')).toBeInTheDocument()
    })
  })

  describe('action navigation', () => {
    it('navigates to the action route when clicked', () => {
      renderEmptyState({ reason: 'no-data', context: { view: 'reports' } })

      expect(currentLocation()).toBe('/reports')
      fireEvent.click(screen.getByRole('button', { name: 'Browse benchmarks' }))
      expect(currentLocation()).toBe('/benchmarks')
    })

    it('suppresses navigation when onAction returns true (in-view recovery)', () => {
      const onAction = vi.fn(() => true)
      renderEmptyState({ reason: 'no-match', context: { view: 'reports' }, onAction })

      const clearFilters = screen.getByRole('button', { name: 'Clear filters' })
      fireEvent.click(clearFilters)

      expect(onAction).toHaveBeenCalledTimes(1)
      expect(onAction).toHaveBeenCalledWith(
        expect.objectContaining({ label: 'Clear filters' }),
      )
      // Route is unchanged because the handler claimed the action.
      expect(currentLocation()).toBe('/reports')
    })

    it('falls through to navigation when onAction returns a non-true value', () => {
      const onAction = vi.fn(() => false)
      renderEmptyState({ reason: 'no-data', context: { view: 'reports' }, onAction })

      fireEvent.click(screen.getByRole('button', { name: 'Browse benchmarks' }))

      expect(onAction).toHaveBeenCalledTimes(1)
      expect(currentLocation()).toBe('/benchmarks')
    })
  })

  describe('Performance no-data entry into Tasks', () => {
    it('offers an action that routes into the Tasks flow', () => {
      renderEmptyState(
        { reason: 'no-data', context: { view: 'performance' } },
        { initialEntries: ['/performance'] },
      )

      expect(currentLocation()).toBe('/performance')
      fireEvent.click(screen.getByRole('button', { name: 'Create a task' }))
      expect(currentLocation()).toBe('/tasks?tab=perf')
    })
  })
})
