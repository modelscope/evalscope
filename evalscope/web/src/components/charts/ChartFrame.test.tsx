// Component and boundary unit tests for the Chart_Renderer (Task 6.7).
//
// Covers ChartFrame behaviour that cannot be expressed as a pure property:
//   - a light theme never renders a dark chart, because the active theme is
//     carried on the iframe request path;
//   - switching the theme re-renders the chart against the new theme;
//   - a failed preflight (404/500/timeout) shows visible error text plus a retry
//     control and never mounts a blank iframe;
//   - a visible loading state is shown while the preflight is in flight, ahead
//     of any error state;
//   - the preflight timeout branch is classified deterministically.
//
// The suite runs under the global deterministic setup (fake timers, fixed
// system time, network disabled). Each test installs its own controllable
// `fetch` stub — returning a fixed Response, or a promise that only settles when
// the request's AbortSignal fires — so the state machine is driven without any
// real network access. React's `waitFor` is intentionally avoided: it does not
// detect Vitest fake timers, so transitions are flushed manually inside `act`.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'

import ChartFrame from './ChartFrame'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { LocaleProvider } from '@/contexts/LocaleContext'
import type { DataTableModel } from '@/components/common/DataTableFallback'

/** Minimal authoritative fallback table shared by the component cases. */
const FALLBACK_TABLE: DataTableModel = {
  columns: ['metric', 'value'],
  rows: [{ metric: 'accuracy', value: 0.92 }],
}

/** Render ChartFrame inside the app's Theme and Locale providers (locale = en). */
function renderChart(overrides: Partial<React.ComponentProps<typeof ChartFrame>> = {}) {
  const props: React.ComponentProps<typeof ChartFrame> = {
    baseSrc: '/api/charts/report',
    theme: 'light',
    fallbackTable: FALLBACK_TABLE,
    ...overrides,
  }
  return render(
    <ThemeProvider>
      <LocaleProvider>
        <ChartFrame {...props} />
      </LocaleProvider>
    </ThemeProvider>,
  )
}

/**
 * Flush pending microtasks (the async preflight continuation) inside `act` so
 * React applies the resulting state update. Two ticks cover the chained
 * `await fetch(...)` continuation and its follow-up dispatch.
 */
async function flushAsync(): Promise<void> {
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })
}

/** A `fetch` stub that resolves immediately with the given HTTP status. */
function stubFetchStatus(status: number): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(() => Promise.resolve({ status } as Response)),
  )
}

/**
 * A `fetch` stub that never resolves on its own and only rejects with an
 * AbortError when the request's signal is aborted — mirroring the browser's
 * behaviour so the timeout path can be exercised with fake timers.
 */
function stubFetchPending(): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(
      (_input: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => {
            reject(new DOMException('The operation was aborted.', 'AbortError'))
          })
        }),
    ),
  )
}

afterEach(() => {
  cleanup()
})

describe('ChartFrame — theme injection on the request path', () => {
  beforeEach(() => {
    stubFetchStatus(200)
  })

  it('carries theme=light on the iframe request and never renders a dark chart', async () => {
    const { container } = renderChart({ theme: 'light' })
    await flushAsync()

    const iframe = container.querySelector('iframe')
    expect(iframe).not.toBeNull()
    const src = iframe?.getAttribute('src') ?? ''
    expect(src).toContain('theme=light')
    expect(src).not.toContain('theme=dark')
  })

  it('re-renders the chart against the new theme when the theme changes', async () => {
    const { container, rerender } = renderChart({ theme: 'light' })
    await flushAsync()
    expect(container.querySelector('iframe')?.getAttribute('src')).toContain('theme=light')

    // Switch the theme prop; the themed URL is an effect dependency, so the
    // iframe reloads against the new theme.
    rerender(
      <ThemeProvider>
        <LocaleProvider>
          <ChartFrame baseSrc="/api/charts/report" theme="dark" fallbackTable={FALLBACK_TABLE} />
        </LocaleProvider>
      </ThemeProvider>,
    )
    await flushAsync()

    const src = container.querySelector('iframe')?.getAttribute('src') ?? ''
    expect(src).toContain('theme=dark')
    expect(src).not.toContain('theme=light')
  })
})

describe('ChartFrame — failure states', () => {
  it.each([
    { status: 404, kind: '4xx', detail: 'The chart could not be found.' },
    { status: 500, kind: '5xx', detail: 'The chart service encountered an error.' },
  ])('shows error text + retry and no iframe for a $status preflight', async ({ status, detail }) => {
    stubFetchStatus(status)
    const { container } = renderChart()
    await flushAsync()

    const alert = screen.getByRole('alert')
    expect(alert).toBeInTheDocument()
    expect(screen.getByText('Failed to load chart')).toBeInTheDocument()
    expect(screen.getByText(detail)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Retry/i })).toBeInTheDocument()

    // No blank iframe is mounted in the error state.
    expect(container.querySelector('iframe')).toBeNull()

    // The authoritative data-table fallback is offered instead.
    expect(screen.getByText('Data table')).toBeInTheDocument()
  })

  it('invokes onRetry and re-runs the preflight when retry is clicked', async () => {
    stubFetchStatus(404)
    const onRetry = vi.fn()
    const { container } = renderChart({ onRetry })
    await flushAsync()
    expect(screen.getByRole('alert')).toBeInTheDocument()

    // A subsequent attempt succeeds; clicking retry recovers to the iframe.
    stubFetchStatus(200)
    fireEvent.click(screen.getByRole('button', { name: /Retry/i }))
    await flushAsync()

    expect(onRetry).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('alert')).toBeNull()
    expect(container.querySelector('iframe')).not.toBeNull()
  })
})

describe('ChartFrame — loading precedes error', () => {
  it('shows a visible loading state while the preflight is in flight, before any error', async () => {
    stubFetchPending()
    const { container } = renderChart()

    // While the request is pending: loading is visible, no error, no iframe.
    expect(screen.getByRole('status')).toBeInTheDocument()
    expect(screen.getByText(/Loading chart/)).toBeInTheDocument()
    expect(screen.queryByRole('alert')).toBeNull()
    expect(container.querySelector('iframe')).toBeNull()
  })

  it('classifies a 10s non-responding preflight as a timeout error', async () => {
    stubFetchPending()
    renderChart({ preflightTimeoutMs: 10000 })

    // Still loading just before the timeout budget elapses.
    expect(screen.getByRole('status')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).toBeNull()

    // Advance to the 10s deadline: the request is aborted and classified as a
    // timeout failure, moving the frame into the error state.
    await act(async () => {
      vi.advanceTimersByTime(10000)
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(screen.getByRole('alert')).toBeInTheDocument()
    expect(screen.getByText('The chart request timed out.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Retry/i })).toBeInTheDocument()
  })
})
