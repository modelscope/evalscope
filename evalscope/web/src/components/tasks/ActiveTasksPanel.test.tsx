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
  it('renders nothing when no task is active', async () => {
    vi.mocked(listActiveTasks).mockResolvedValue({ tasks: [], total: 0 })
    const { container } = render(
      <LocaleProvider>
        <ActiveTasksPanel onAdopt={vi.fn()} />
      </LocaleProvider>,
    )
    await settle()
    expect(container).toBeEmptyDOMElement()
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

    fireEvent.click(screen.getByRole('button', { name: 'Monitor' }))
    expect(onAdopt).toHaveBeenCalledWith('perf', 'perf_1')
  })
})
