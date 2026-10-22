import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState, type ComponentProps } from 'react'
import { MemoryRouter } from 'react-router-dom'
import type { ActiveTaskInfo, EvalInvokeResponse } from '@/api/types'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { DomainError } from '@/api/errors'
import TaskRunnerShell from './TaskRunnerShell'

vi.mock('@/api/tasks', () => ({ getTask: vi.fn() }))
import { getTask } from '@/api/tasks'

const LOG = { text: '', head_line: 0, tail_line: 0, total_lines: 0 }
const TASK: ActiveTaskInfo = { task_id: 'eval_A', kind: 'eval', alias: 'server-A', model: 'same-model', endpoint: 'host:8000', started_at: '', percent: 10, status: 'running', config: { eval_batch_size: 1 } }

beforeEach(() => { vi.mocked(getTask).mockImplementation(async (id) => ({ ...TASK, task_id: id })) })
afterEach(() => { cleanup(); vi.resetAllMocks(); sessionStorage.clear(); localStorage.clear() })

function Form({ onSubmit, disabled }: { onSubmit: (config: Record<string, unknown>) => Promise<void>; disabled: boolean }) {
  const [draft, setDraft] = useState('')
  return <><input aria-label="Draft" value={draft} onChange={(e) => setDraft(e.target.value)} /><button disabled={disabled} onClick={() => onSubmit({ model: draft })}>Submit</button></>
}
function Runner(props: Partial<ComponentProps<typeof TaskRunnerShell>> = {}) {
  const [taskId, setTaskId] = useState<string | null>(props.selectedTaskId ?? null)
  return <LocaleProvider><MemoryRouter>
    <button onClick={() => setTaskId('eval_A')}>View A</button><button onClick={() => setTaskId('eval_B')}>View B</button>
    <TaskRunnerShell idPrefix="eval" title="Evaluation" configTitle="Configuration" statusTitle="Status" readyLabel="Ready"
      selectedTaskId={taskId} onSelect={setTaskId} submitTask={vi.fn().mockImplementation(() => new Promise(() => {}))}
      stopTask={vi.fn().mockResolvedValue({})} getLog={vi.fn().mockResolvedValue(LOG)} renderForm={(form) => <Form {...form} />} {...props} />
  </MemoryRouter></LocaleProvider>
}
async function click(name: string) { await act(async () => { fireEvent.click(screen.getByRole('button', { name })) }) }

 describe('TaskRunnerShell and selected task monitor', () => {
  it('keeps the form draft and task identity when monitoring another task', async () => {
    render(<Runner />)
    fireEvent.change(screen.getByLabelText('Draft'), { target: { value: 'my draft' } })
    await click('View A')
    expect(screen.getByRole('heading', { name: 'Viewing task: server-A' })).toBeInTheDocument()
    expect(screen.getByText('Running 10%')).toBeInTheDocument()
    expect(screen.getByText('ID: eval_A')).toBeInTheDocument()
    expect(screen.getByText(/Endpoint: host:8000/)).toBeInTheDocument()
    await click('View B')
    expect(screen.getByLabelText('Draft')).toHaveValue('my draft')
    expect(screen.getByText('ID: eval_B')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Submit' })).toBeEnabled()
    expect(sessionStorage.getItem('evalscope-active-task-eval')).toBe('eval_B')
    expect(sessionStorage.length).toBe(1)
  })
  it('selects and monitors a task when browser storage is blocked', async () => {
    const storage = vi.spyOn(window, 'sessionStorage', 'get').mockImplementation(() => { throw new DOMException('blocked') })
    try {
      render(<Runner />)
      await click('View A')
      expect(screen.getByText('Running 10%')).toBeInTheDocument()
      await click('Stop')
      expect(screen.getByText('Stopped')).toBeInTheDocument()
    } finally { storage.mockRestore() }
  })
  it('does not show stopped or stop polling after a failed stop', async () => {
    render(<Runner stopTask={vi.fn().mockRejectedValue(new Error('network'))} />)
    await click('View A')
    await click('Stop')
    expect(screen.queryByText('Stopped')).not.toBeInTheDocument()
    expect(screen.getByText(/Stop was not confirmed/)).toBeInTheDocument()
    await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
    expect(getTask).toHaveBeenCalledTimes(2)
  })
  it('shows stopping while the request is pending', async () => {
    let finish!: () => void
    const stopTask = vi.fn().mockImplementation(() => new Promise<void>((resolve) => { finish = resolve }))
    render(<Runner stopTask={stopTask} />)
    await click('View A'); await click('Stop')
    expect(screen.getByRole('button', { name: 'Stopping…' })).toBeDisabled()
    await act(async () => { finish() })
    expect(screen.getByText('Stopped')).toBeInTheDocument()
  })
  it('discards late status, logs and stop responses after selecting B', async () => {
    let status!: (value: ActiveTaskInfo) => void
    let log!: (value: typeof LOG) => void
    let stop!: () => void
    vi.mocked(getTask).mockImplementationOnce(() => new Promise((resolve) => { status = resolve }))
    const getLog = vi.fn().mockImplementationOnce(() => new Promise((resolve) => { log = resolve })).mockResolvedValue(LOG)
    render(<Runner getLog={getLog} stopTask={vi.fn().mockImplementation(() => new Promise<void>((resolve) => { stop = resolve }))} />)
    await click('View A'); await click('Stop'); await click('View B')
    await act(async () => { status({ ...TASK, status: 'completed' }); log({ ...LOG, text: 'Old log' }); stop() })
    expect(screen.getByText('ID: eval_B')).toBeInTheDocument()
    expect(screen.getByText('Running 10%')).toBeInTheDocument()
    expect(screen.queryByText('Stopped')).not.toBeInTheDocument()
    expect(screen.queryByText('Old log')).not.toBeInTheDocument()
  })
  it('does not let a late invoke result replace the selected task', async () => {
    let finish!: (value: EvalInvokeResponse) => void
    render(<Runner submitTask={vi.fn().mockImplementation(() => new Promise((resolve) => { finish = resolve }))} />)
    await click('Submit'); await click('View B')
    await act(async () => { finish({ task_id: 'eval_1782864000000', status: 'completed' }) })
    expect(screen.getByText('ID: eval_B')).toBeInTheDocument()
    expect(screen.getByText('Running 10%')).toBeInTheDocument()
  })
  it('does not mark a 100 percent active task as completed', async () => {
    vi.mocked(getTask).mockResolvedValue({ ...TASK, percent: 100, status: 'finalizing' })
    render(<Runner />); await click('View A')
    expect(screen.getByText('Finalizing report')).toBeInTheDocument()
    expect(screen.queryByText('Completed')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled()
  })
  it('renders unknown progress without falsely showing zero', async () => {
    vi.mocked(getTask).mockResolvedValue({ ...TASK, percent: null, status: 'preparing' })
    render(<Runner />); await click('View A')
    expect(screen.getByText('Preparing')).toBeInTheDocument()
    expect(screen.queryByText(/0%/)).not.toBeInTheDocument()
  })
  it.each(['completed', 'error', 'unknown'] as const)('restores authoritative %s without invoking again', async (status) => {
    const submitTask = vi.fn()
    vi.mocked(getTask).mockResolvedValue({ ...TASK, status, report_url: status === 'completed' ? '/reports/run/model' : null })
    await act(async () => { render(<Runner selectedTaskId="eval_A" submitTask={submitTask} />) })
    expect(submitTask).not.toHaveBeenCalled()
    if (status === 'completed') expect(screen.getByRole('link', { name: 'View report' })).toHaveAttribute('href', '/reports/run/model')
    if (status === 'error') expect(screen.getByText('Task failed. See the task log for details.')).toBeInTheDocument()
    if (status === 'unknown') expect(screen.getByText('Running state unknown')).toBeInTheDocument()
  })
  it('classifies missing tasks by HTTP status rather than error wording', async () => {
    vi.mocked(getTask).mockRejectedValueOnce(new DomainError('http-4xx', 'Different server wording', 404))
    render(<Runner />)
    await click('View A')
    expect(screen.getByText(/This task is not available yet/)).toBeInTheDocument()
    expect(screen.queryByText(/Cannot reach the task service/)).not.toBeInTheDocument()
  })
  it('reports a connection problem and recovers on retry', async () => {
    vi.mocked(getTask).mockRejectedValueOnce(new Error('network'))
    render(<Runner />); await click('View A')
    expect(screen.getByText(/Cannot reach the task service/)).toBeInTheDocument()
    await click('Retry')
    expect(screen.queryByText(/Cannot reach the task service/)).not.toBeInTheDocument()
    expect(screen.getByText('Running 10%')).toBeInTheDocument()
  })
  it('localizes restored terminal outcomes', async () => {
    localStorage.setItem('evalscope-locale', 'zh')
    vi.mocked(getTask).mockResolvedValue({ ...TASK, status: 'error' })
    await act(async () => { render(<Runner selectedTaskId="eval_A" />) })
    expect(screen.getByText('任务失败，详情见任务日志。')).toBeInTheDocument()
  })
})
