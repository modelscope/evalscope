// The unified Tasks page — a two-tab shell over the Eval and Perf task runners.
//
// Its own responsibility is narrow: pick the sub-tab from `?tab=`, default to
// eval, and reflect a switch back into the query param so a refresh or shared
// link lands on the same tab. The runners themselves are mocked out so this
// exercises the routing, not their networked internals.

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import { LocaleProvider } from '@/contexts/LocaleContext'

vi.mock('@/components/tasks/EvalTaskPanel', () => ({
  default: () => <div data-testid="eval-panel">eval runner</div>,
}))
vi.mock('@/components/tasks/PerfTaskPanel', () => ({
  default: () => <div data-testid="perf-panel">perf runner</div>,
}))

import TasksPage from './TasksPage'

afterEach(cleanup)

function renderAt(search = '') {
  render(
    <LocaleProvider>
      <MemoryRouter initialEntries={[`/tasks${search}`]}>
        <TasksPage />
      </MemoryRouter>
    </LocaleProvider>,
  )
}

describe('TasksPage', () => {
  it('defaults to the evaluation runner', () => {
    renderAt()
    expect(screen.getByTestId('eval-panel')).toBeInTheDocument()
    expect(screen.getByRole('tab', { selected: true })).toHaveTextContent(/eval/i)
  })

  it('opens the performance runner when the query param asks for it', () => {
    renderAt('?tab=perf')
    expect(screen.getByTestId('perf-panel')).toBeInTheDocument()
  })

  it('switches tabs on click', () => {
    renderAt()
    fireEvent.click(screen.getByRole('tab', { name: /perf/i }))
    expect(screen.getByTestId('perf-panel')).toBeInTheDocument()
  })
})
