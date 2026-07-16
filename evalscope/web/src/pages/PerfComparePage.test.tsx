// Component tests for the Performance_Compare_View (Task 13.11).
//
// These cover the behaviour that lives in the PerfComparePage component layer
// (rather than in the pure `compareModel` domain logic, which is covered by the
// property tests in tasks 13.2-13.7):
//   - the baseline swap is persisted to the URL `baseline` query param so it
//     survives re-render and the effective-baseline marker follows it (Req 9.3);
//   - low sample sizes (n < 30) surface a strong warning and de-emphasize the
//     P90 / P95 / P99 delta rows while leaving other rows emphasized (Req 9.6);
//   - runs with mismatched workloads show a non-blocking mismatch hint while the
//     delta table still renders (Req 9.10);
//   - a run missing performance data surfaces the missing-data hint while the
//     available metrics are still compared (Req 9.14).
//
// The suite runs under the global deterministic setup (fake timers, fixed system
// time, network disabled). `@/api/perf` is mocked so `getPerfDetail` resolves
// from in-memory desensitized fixtures; the rest of the module (URL builders) is
// preserved. The page reads `paths` / `baseline` / `root_path` from the query
// string, so it is wrapped in a MemoryRouter with `initialEntries` and a small
// LocationDisplay surfaces the active route so persistence can be asserted.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'

import PerfComparePage from './PerfComparePage'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { ReportsProvider } from '@/contexts/ReportsContext'
import type { PerfDetailResponse } from '@/api/types'

// Mock only `getPerfDetail`; keep the real URL builders (pure string helpers, no
// network) so the chart iframes render without hitting the disabled fetch guard.
vi.mock('@/api/perf', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/perf')>()
  return { ...actual, getPerfDetail: vi.fn() }
})

import { getPerfDetail } from '@/api/perf'

// ------------------------------------------------------------------ //
// Fixtures                                                            //
// ------------------------------------------------------------------ //

/**
 * Build a desensitized `PerfDetailResponse`. Sample count is driven through the
 * `Number of requests` summary row (mirrors {@link getSampleCount}); percentile
 * rows are always present so the low-sample de-emphasis can be exercised.
 */
type PerfDetailOverrides = Partial<PerfDetailResponse> & { __sampleCount?: number }

function makePerfDetail(overrides: PerfDetailOverrides = {}): PerfDetailResponse {
  const sampleCount = overrides.__sampleCount ?? 200
  const rows: PerfDetailResponse['summary_rows'] = [
    ['Number of requests', sampleCount],
    ['Average latency (s)', 1.2],
    ['P90 latency (s)', 2.0],
    ['P95 latency (s)', 2.5],
    ['P99 latency (s)', 3.0],
    ['Output throughput (tokens/s)', 500],
  ]
  const base: PerfDetailResponse = {
    path: 'run-a',
    model: 'model-a',
    api_type: 'openai_api',
    dataset: 'openqa',
    generated_at: '2026-06-01T00:00:00Z',
    basic_info: { 'Total requests': String(sampleCount) },
    summary_columns: ['Metric', 'Value'],
    summary_rows: rows,
    best_config: { concurrency: '10' },
    recommendations: [],
    num_runs: 1,
    is_embedding: false,
    has_html: false,
  }
  // `__sampleCount` is a test-only helper; strip it before returning.
  const merged = { ...base, ...overrides }
  delete merged.__sampleCount
  return merged as PerfDetailResponse
}

/** Resolve `getPerfDetail(root, path)` from an in-memory map; reject unknown/failed paths. */
function mockDetails(map: Record<string, PerfDetailResponse>, failed: string[] = []) {
  vi.mocked(getPerfDetail).mockImplementation(async (_root: string, path: string) => {
    if (failed.includes(path)) throw new Error('missing run')
    const detail = map[path]
    if (!detail) throw new Error('unknown run')
    return detail
  })
}

// ------------------------------------------------------------------ //
// Render helpers                                                      //
// ------------------------------------------------------------------ //

/** Surfaces the active router location so URL persistence is observable. */
function LocationDisplay() {
  const location = useLocation()
  return <div data-testid="location">{`${location.pathname}${location.search}`}</div>
}

function renderPage(entries: string[]) {
  return render(
    <ThemeProvider>
      <LocaleProvider>
        <ReportsProvider>
          <MemoryRouter initialEntries={entries}>
            <Routes>
              <Route
                path="*"
                element={
                  <>
                    <PerfComparePage />
                    <LocationDisplay />
                  </>
                }
              />
            </Routes>
          </MemoryRouter>
        </ReportsProvider>
      </LocaleProvider>
    </ThemeProvider>,
  )
}

/** Render and flush the async `getPerfDetail` load (fixed timers + microtasks). */
async function renderLoaded(entries: string[]) {
  const utils = renderPage(entries)
  await act(async () => {
    await vi.runAllTimersAsync()
  })
  return utils
}

/** Build a compare-view entry URL for the given run paths. */
function compareUrl(paths: string[], extra: Record<string, string> = {}): string {
  const params = new URLSearchParams({ root_path: './outputs', paths: paths.join(';'), ...extra })
  return `/performance/compare?${params.toString()}`
}

function currentLocation(): string {
  return screen.getByTestId('location').textContent ?? ''
}

// ------------------------------------------------------------------ //
// Tests                                                               //
// ------------------------------------------------------------------ //

beforeEach(() => {
  vi.mocked(getPerfDetail).mockReset()
})

afterEach(() => {
  cleanup()
})

describe('PerfComparePage', () => {
  describe('baseline swap persistence (Req 9.3)', () => {
    it('persists the swapped baseline to the URL and updates the effective-baseline marker', async () => {
      const older = makePerfDetail({ path: 'run-a', model: 'model-a', generated_at: '2026-06-01T00:00:00Z' })
      const newer = makePerfDetail({ path: 'run-b', model: 'model-b', generated_at: '2026-06-02T00:00:00Z' })
      mockDetails({ 'run-a': older, 'run-b': newer })

      await renderLoaded([compareUrl(['run-a', 'run-b'])])

      // Default baseline is the oldest run (model-a); candidate is the newest (model-b).
      expect(screen.getByTestId('baseline-label')).toHaveTextContent('model-a')
      expect(screen.getByTestId('candidate-label')).toHaveTextContent('model-b')
      expect(screen.getByTestId('compare-run-labels')).toHaveTextContent('model-a · openqa')
      expect(screen.getByTestId('compare-run-labels')).not.toHaveTextContent('run-a')
      expect(screen.getByTestId('compare-run-labels')).toHaveAttribute('title', 'run-a\nrun-b')
      expect(currentLocation()).not.toContain('baseline=')

      // Swap the baseline.
      await act(async () => {
        fireEvent.click(screen.getByTestId('swap-baseline'))
        await vi.runAllTimersAsync()
      })

      // The candidate run id is persisted as the effective baseline in the URL,
      // and the labels swap to reflect the new effective baseline.
      expect(currentLocation()).toContain('baseline=run-b')
      expect(screen.getByTestId('baseline-label')).toHaveTextContent('model-b')
      expect(screen.getByTestId('candidate-label')).toHaveTextContent('model-a')
    })

    it('honours a baseline provided in the initial query (persisted selection survives load)', async () => {
      const older = makePerfDetail({ path: 'run-a', model: 'model-a', generated_at: '2026-06-01T00:00:00Z' })
      const newer = makePerfDetail({ path: 'run-b', model: 'model-b', generated_at: '2026-06-02T00:00:00Z' })
      mockDetails({ 'run-a': older, 'run-b': newer })

      await renderLoaded([compareUrl(['run-a', 'run-b'], { baseline: 'run-b' })])

      // The persisted baseline (run-b / model-b) is honoured over the default oldest run.
      expect(screen.getByTestId('baseline-label')).toHaveTextContent('model-b')
      expect(screen.getByTestId('candidate-label')).toHaveTextContent('model-a')
    })
  })

  describe('low-sample de-emphasis (Req 9.6)', () => {
    it('shows the critical warning and de-emphasizes P90/P95/P99 rows while keeping others emphasized', async () => {
      const older = makePerfDetail({ path: 'run-a', generated_at: '2026-06-01T00:00:00Z', __sampleCount: 10 })
      const newer = makePerfDetail({
        path: 'run-b',
        model: 'model-b',
        generated_at: '2026-06-02T00:00:00Z',
        __sampleCount: 12,
      })
      mockDetails({ 'run-a': older, 'run-b': newer })

      await renderLoaded([compareUrl(['run-a', 'run-b'])])

      // Strong low-sample warning is present; the sample count is still shown.
      expect(screen.getByTestId('low-sample-critical')).toBeInTheDocument()
      expect(screen.queryByTestId('low-sample-warn')).not.toBeInTheDocument()

      // Percentile deltas are de-emphasized (raw values preserved via title tooltips).
      for (const key of ['P90 latency (s)', 'P95 latency (s)', 'P99 latency (s)']) {
        expect(screen.getByTestId(`delta-row-${key}`)).toHaveAttribute('data-deemphasized', 'true')
      }

      // A non-percentile metric stays emphasized.
      expect(screen.getByTestId('delta-row-Average latency (s)')).toHaveAttribute('data-deemphasized', 'false')
    })
  })

  describe('workload mismatch hint (Req 9.10)', () => {
    it('shows a non-blocking mismatch hint and still renders the delta table', async () => {
      const older = makePerfDetail({ path: 'run-a', dataset: 'openqa', generated_at: '2026-06-01T00:00:00Z' })
      const newer = makePerfDetail({
        path: 'run-b',
        model: 'model-b',
        dataset: 'longalpaca',
        generated_at: '2026-06-02T00:00:00Z',
      })
      mockDetails({ 'run-a': older, 'run-b': newer })

      await renderLoaded([compareUrl(['run-a', 'run-b'])])

      // Mismatch hint appears...
      expect(screen.getByTestId('workload-mismatch')).toBeInTheDocument()
      // ...but the comparison is not blocked: the delta table still renders.
      expect(screen.getByTestId('delta-table')).toBeInTheDocument()
      expect(screen.getByTestId('delta-row-Average latency (s)')).toBeInTheDocument()
    })
  })

  describe('missing performance data hint (Req 9.14)', () => {
    it('shows the missing-data hint while still comparing the available metrics', async () => {
      const withData = makePerfDetail({ path: 'run-a', generated_at: '2026-06-01T00:00:00Z' })
      // Candidate is missing its performance summary rows entirely.
      const missing = makePerfDetail({
        path: 'run-b',
        model: 'model-b',
        generated_at: '2026-06-02T00:00:00Z',
        summary_rows: [],
      })
      mockDetails({ 'run-a': withData, 'run-b': missing })

      await renderLoaded([compareUrl(['run-a', 'run-b'])])

      // Missing-data hint appears and the delta table is still rendered.
      expect(screen.getByTestId('missing-perf-data')).toBeInTheDocument()
      expect(screen.getByTestId('delta-table')).toBeInTheDocument()

      // The metric present on the baseline but missing on the candidate is
      // de-emphasized (incomputable) rather than dropped.
      expect(screen.getByTestId('delta-row-Average latency (s)')).toHaveAttribute('data-deemphasized', 'true')
    })
  })
})
