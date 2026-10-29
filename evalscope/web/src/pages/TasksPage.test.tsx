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
  default: ({ selectedTaskId }: { selectedTaskId: string | null }) => <div data-testid="eval-panel">{selectedTaskId || 'eval runner'}<input aria-label="Eval draft" defaultValue="" /></div>
}))
vi.mock('@/components/tasks/PerfTaskPanel', () => ({
  default: ({ selectedTaskId }: { selectedTaskId: string | null }) => <div data-testid="perf-panel">{selectedTaskId || 'perf runner'}<input aria-label="Perf draft" defaultValue="" /></div>
}))
vi.mock('@/components/tasks/ActiveTasksPanel', () => ({
  default: ({ onAdopt }: { onAdopt: (kind: 'eval' | 'perf', taskId: string) => void }) => (
    <button data-testid="adopt-perf" onClick={() => onAdopt('perf', 'perf_1')} />
  ),
}))

import TasksPage from './TasksPage'

afterEach(() => { cleanup(); sessionStorage.clear() })

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
  it('keeps drafts across type switches and monitoring selection', () => {
    renderAt()
    fireEvent.change(screen.getByLabelText('Eval draft'), { target: { value: 'draft' } })
    fireEvent.click(screen.getByTestId('adopt-perf'))
    fireEvent.click(screen.getByRole('tab', { name: /eval/i }))
    expect(screen.getByLabelText('Eval draft')).toHaveValue('draft')
  })
  it('reads separate monitoring slots on refresh', () => {
    sessionStorage.setItem('evalscope-active-task-eval', 'eval_saved')
    sessionStorage.setItem('evalscope-active-task-perf', 'perf_saved')
    renderAt('?tab=perf')
    expect(screen.getByTestId('perf-panel')).toHaveTextContent('perf_saved')
    fireEvent.click(screen.getByRole('tab', { name: /eval/i }))
    expect(screen.getByTestId('eval-panel')).toHaveTextContent('eval_saved')
  })
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

  it('selects an active task without remounting the forms', () => {
    renderAt()
    fireEvent.click(screen.getByTestId('adopt-perf'))

    expect(screen.getByTestId('perf-panel')).toHaveTextContent('perf_1')
    expect(screen.getByTestId('perf-panel')).toBeInTheDocument()
    expect(screen.getByRole('tab', { selected: true })).toHaveTextContent(/perf/i)
  })
})
