// Component tests for the active-tasks panel: it lists the service's running
// tasks (from any browser session) and hands one over to its runner panel.

import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'

import { LocaleProvider } from '@/contexts/LocaleContext'

vi.mock('@/api/tasks', () => ({
  listActiveTasks: vi.fn(),
}))

import { listActiveTasks } from '@/api/tasks'
import ActiveTasksPanel from './ActiveTasksPanel'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function settle(): Promise<void> {
  for (let i = 0; i < 8; i++) {
    await act(async () => {
      await Promise.resolve()
    })
  }
}

const RUNNING = {
  tasks: [
    {
      task_id: 'perf_1',
      kind: 'perf',
      model: 'Qwen3.8-27B',
      endpoint: '10.0.0.2:8000',
      alias: 'server-A',
      started_at: '2026-06-30T23:00:00',
      percent: 42,
      status: 'running' as const,
    },
  ],
  total: 1,
}

function renderPanel(onAdopt = vi.fn()) {
  render(
    <LocaleProvider>
      <ActiveTasksPanel onAdopt={onAdopt} />
    </LocaleProvider>,
  )
  return onAdopt
}

describe('ActiveTasksPanel', () => {
  it('distinguishes loading, failure and retry recovery', async () => {
    vi.mocked(listActiveTasks).mockRejectedValueOnce(new Error('network')).mockResolvedValue(RUNNING)
    renderPanel()
    expect(screen.getByText('Loading...')).toBeInTheDocument()
    await settle()
    expect(screen.getByText(/Cannot reach the task service/)).toBeInTheDocument()
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Retry' })) })
    expect(screen.getByText('server-A')).toBeInTheDocument()
  })
  it('highlights the selected task and keeps unknown progress unknown', async () => {
    vi.mocked(listActiveTasks).mockResolvedValue({ ...RUNNING, tasks: [{ ...RUNNING.tasks[0], percent: null }] })
    render(<LocaleProvider><ActiveTasksPanel selectedTaskId="perf_1" onAdopt={vi.fn()} /></LocaleProvider>)
    await settle()
    expect(screen.getByRole('button', { name: 'View monitor: server-A' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText('Running')).toBeInTheDocument()
    expect(screen.queryByText('0%')).not.toBeInTheDocument()
  })
  it('renders nothing when no task is active', async () => {
    vi.mocked(listActiveTasks).mockResolvedValue({ tasks: [], total: 0 })
    const { container } = render(
      <LocaleProvider>
        <ActiveTasksPanel onAdopt={vi.fn()} />
      </LocaleProvider>,
    )
    await settle()
    expect(container).toHaveTextContent('No running tasks on this service.')
  })

  it('lists running tasks with alias, endpoint and progress', async () => {
    vi.mocked(listActiveTasks).mockResolvedValue(RUNNING)
    renderPanel()
    await settle()

    expect(screen.getByText('server-A')).toBeInTheDocument()
    expect(screen.getByText('10.0.0.2:8000 · perf_1')).toBeInTheDocument()
    expect(screen.getByText('42%')).toBeInTheDocument()
  })

  it('hands the task over to its runner panel on Monitor', async () => {
    vi.mocked(listActiveTasks).mockResolvedValue(RUNNING)
    const onAdopt = renderPanel()
    await settle()

    fireEvent.click(screen.getByRole('button', { name: 'View monitor: server-A' }))
    expect(onAdopt).toHaveBeenCalledWith('perf', 'perf_1')
  })
})
