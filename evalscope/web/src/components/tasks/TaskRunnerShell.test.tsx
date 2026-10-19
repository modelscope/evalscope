import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { ComponentProps } from 'react'
import type { EvalInvokeResponse, LogResponse, ProgressResponse } from '@/api/types'
import { LocaleProvider } from '@/contexts/LocaleContext'
import TaskRunnerShell from './TaskRunnerShell'

afterEach(() => {
  cleanup()
  sessionStorage.clear()
  localStorage.clear()
})

function taskRunner(props: Partial<ComponentProps<typeof TaskRunnerShell>> = {}) {
  return (
    <LocaleProvider>
      <TaskRunnerShell
        idPrefix="eval"
        title="Evaluation"
        configTitle="Configuration"
        statusTitle="Status"
        readyLabel="Ready"
        submitTask={vi.fn().mockImplementation(() => new Promise(() => {}))}
        stopTask={vi.fn()}
        getProgress={vi.fn().mockResolvedValue({ percent: 10, status: 'running' })}
        getLog={vi.fn().mockResolvedValue({ text: '', head_line: 0, tail_line: 0, total_lines: 0 })}
        getReportUrl={(taskId) => `/report/${taskId}`}
        renderForm={({ onSubmit, disabled }) => (
          <button type="button" disabled={disabled} onClick={() => onSubmit({ model: 'qwen-plus' })}>
            Submit
          </button>
        )}
        {...props}
      />
    </LocaleProvider>
  )
}

describe('TaskRunnerShell', () => {
  it('can run a task when session storage is unavailable', async () => {
    const storage = vi.spyOn(window, 'sessionStorage', 'get').mockImplementation(() => {
      throw new DOMException('Storage blocked', 'SecurityError')
    })
    try {
      render(taskRunner())
      await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Submit' })) })
      expect(screen.getByText('Running 10%')).toBeInTheDocument()
      await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Stop' })) })
      expect(screen.getByText('Stopped')).toBeInTheDocument()
    } finally {
      storage.mockRestore()
    }
  })

  it('ignores late progress, logs and invoke responses from a previous task', async () => {
    let finishProgress!: (value: ProgressResponse) => void
    let finishLog!: (value: LogResponse) => void
    let finishInvoke!: (value: EvalInvokeResponse) => void
    const getProgress = vi.fn()
      .mockImplementationOnce(() => new Promise((resolve) => { finishProgress = resolve }))
      .mockResolvedValue({ percent: 10, status: 'running' })
    const getLog = vi.fn()
      .mockImplementationOnce(() => new Promise((resolve) => { finishLog = resolve }))
      .mockResolvedValue({ text: '', head_line: 0, tail_line: 0, total_lines: 0 })
    const submitTask = vi.fn()
      .mockImplementationOnce(() => new Promise((resolve) => { finishInvoke = resolve }))
      .mockImplementation(() => new Promise(() => {}))
    render(taskRunner({ getProgress, getLog, submitTask }))
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Submit' })) })
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Stop' })) })
    await act(async () => { await vi.advanceTimersByTimeAsync(1) })
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Submit' })) })
    expect(screen.getByText('Running 10%')).toBeInTheDocument()

    await act(async () => {
      finishProgress({ percent: 100, status: 'completed' })
      finishLog({ text: 'Old task log', head_line: 0, tail_line: 1, total_lines: 1 })
      finishInvoke({ status: 'completed', task_id: 'eval_1782864000000' })
    })
    expect(screen.getByText('Running 10%')).toBeInTheDocument()
    expect(screen.queryByText('Old task log')).not.toBeInTheDocument()
    expect(sessionStorage.getItem('evalscope-active-task-eval')).toBe('eval_1782864000001')
  })

  it('preserves the invoke error when completed progress arrives later', async () => {
    let finishProgress!: (value: ProgressResponse) => void
    let failInvoke!: (reason: Error) => void
    const getProgress = vi.fn().mockImplementation(() => new Promise((resolve) => { finishProgress = resolve }))
    const submitTask = vi.fn().mockImplementation(() => new Promise((_, reject) => { failInvoke = reject }))
    render(taskRunner({ getProgress, submitTask }))
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Submit' })) })
    await act(async () => { failInvoke(new Error('Evaluation completed but no results were produced')) })
    expect(screen.getByText('Error: Evaluation completed but no results were produced')).toBeInTheDocument()

    await act(async () => { finishProgress({ percent: 100, status: 'completed' }) })
    expect(screen.getByText('Error: Evaluation completed but no results were produced')).toBeInTheDocument()
    expect(screen.queryByText('Completed')).not.toBeInTheDocument()
  })

  it.each(['eval', 'perf'])('restores and can stop the active %s task after remounting', async (idPrefix) => {
    const submitTask = vi.fn().mockImplementation(() => new Promise(() => {}))
    const getProgress = vi.fn().mockResolvedValue({ percent: 40, status: 'running' })
    const getLog = vi.fn().mockResolvedValue({ text: 'Still evaluating', head_line: 0, tail_line: 1, total_lines: 1 })
    const runner = (
      <LocaleProvider>
        <TaskRunnerShell
          idPrefix={idPrefix}
          title="Evaluation"
          configTitle="Configuration"
          statusTitle="Status"
          readyLabel="Ready"
          submitTask={submitTask}
          stopTask={vi.fn()}
          getProgress={getProgress}
          getLog={getLog}
          getReportUrl={(taskId) => `/report/${taskId}`}
          renderForm={({ onSubmit, disabled }) => (
            <button type="button" disabled={disabled} onClick={() => onSubmit({ model: 'qwen-plus' })}>
              Submit
            </button>
          )}
        />
      </LocaleProvider>
    )
    const firstMount = render(runner)
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Submit' })) })
    expect(screen.getByText('Running 40%')).toBeInTheDocument()

    firstMount.unmount()
    getProgress.mockClear()
    getLog.mockClear()
    await act(async () => { render(runner) })

    expect(getProgress).toHaveBeenCalledWith(`${idPrefix}_1782864000000`)
    expect(getLog).toHaveBeenCalledWith(`${idPrefix}_1782864000000`, 0)
    expect(screen.getByText('Running 40%')).toBeInTheDocument()
    expect(screen.getByText('Still evaluating')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Submit' })).toBeDisabled()
    expect(submitTask).toHaveBeenCalledTimes(1)

    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Stop' })) })
    expect(screen.getByText('Stopped')).toBeInTheDocument()
    cleanup()
    getProgress.mockClear()
    render(runner)
    expect(screen.getByText('Ready')).toBeInTheDocument()
    expect(getProgress).not.toHaveBeenCalled()
  })

  it.each(['completed', 'error'])('renders the restored task outcome when progress reports %s', async (status) => {
    sessionStorage.setItem('evalscope-active-task-eval', 'eval_1782864000000')
    const getProgress = vi.fn().mockResolvedValue({ percent: status === 'completed' ? 100 : 40, status })
    const submitTask = vi.fn()
    const runner = (
      <LocaleProvider>
        <TaskRunnerShell
          idPrefix="eval"
          title="Evaluation"
          configTitle="Configuration"
          statusTitle="Status"
          readyLabel="Ready"
          submitTask={submitTask}
          stopTask={vi.fn()}
          getProgress={getProgress}
          getLog={vi.fn().mockResolvedValue({ text: '', head_line: 0, tail_line: 0, total_lines: 0 })}
          getReportUrl={(taskId) => `/report/${taskId}`}
          renderForm={() => null}
        />
      </LocaleProvider>
    )
    await act(async () => { render(runner) })
    expect(screen.getByText(status === 'completed' ? 'Completed' : 'Task failed. See the task log for details.')).toBeInTheDocument()
    expect(submitTask).not.toHaveBeenCalled()
    if (status === 'completed') {
      const open = vi.spyOn(window, 'open').mockImplementation(() => null)
      fireEvent.click(screen.getByRole('button', { name: 'Open in New Tab' }))
      expect(open).toHaveBeenCalledWith('/report/eval_1782864000000', '_blank')
      open.mockRestore()
    }
    cleanup()
    getProgress.mockClear()
    render(runner)
    expect(screen.getByText('Ready')).toBeInTheDocument()
    expect(getProgress).not.toHaveBeenCalled()
  })

  it('keeps evaluation and performance task identities separate', async () => {
    sessionStorage.setItem('evalscope-active-task-perf', 'perf_1782864000000')
    const getProgress = vi.fn()
    render(
      <LocaleProvider>
        <TaskRunnerShell
          idPrefix="eval"
          title="Evaluation"
          configTitle="Configuration"
          statusTitle="Status"
          readyLabel="Ready"
          submitTask={vi.fn()}
          stopTask={vi.fn()}
          getProgress={getProgress}
          getLog={vi.fn()}
          getReportUrl={(taskId) => `/report/${taskId}`}
          renderForm={() => null}
        />
      </LocaleProvider>,
    )
    expect(screen.getByText('Ready')).toBeInTheDocument()
    expect(getProgress).not.toHaveBeenCalled()
    expect(sessionStorage.getItem('evalscope-active-task-perf')).toBe('perf_1782864000000')
  })

  it('clears a restored task when stopping returns a not-found error', async () => {
    sessionStorage.setItem('evalscope-active-task-eval', 'eval_1782864000000')
    const stopTask = vi.fn().mockRejectedValue(new Error('No running task found'))
    await act(async () => {
      render(taskRunner({ stopTask, getProgress: vi.fn().mockResolvedValue({ percent: 0 }) }))
    })
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Stop' })) })

    expect(stopTask).toHaveBeenCalledWith('eval_1782864000000')
    expect(screen.getByText('Stopped')).toBeInTheDocument()
    expect(sessionStorage.getItem('evalscope-active-task-eval')).toBeNull()
    expect(screen.getByRole('button', { name: 'Submit' })).toBeEnabled()
  })

  it('renders the restored failure in the active locale', async () => {
    localStorage.setItem('evalscope-locale', 'zh')
    sessionStorage.setItem('evalscope-active-task-eval', 'eval_1782864000000')
    const getProgress = vi.fn().mockResolvedValue({ percent: 40, status: 'error' })
    await act(async () => {
      render(taskRunner({ getProgress }))
    })
    expect(screen.getByText('任务失败，详情见任务日志。')).toBeInTheDocument()
  })

  it('warns when progress polling keeps failing, and Stop clears the warning', async () => {
    sessionStorage.setItem('evalscope-active-task-eval', 'eval_1782864000000')
    const getProgress = vi.fn().mockRejectedValue(new Error('404 task not found'))
    await act(async () => {
      render(taskRunner({ getProgress }))
    })
    expect(screen.queryByText(/Progress updates are failing/)).not.toBeInTheDocument()

    await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
    await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
    expect(screen.getByText(/Progress updates are failing/)).toBeInTheDocument()

    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Stop' })) })
    expect(screen.queryByText(/Progress updates are failing/)).not.toBeInTheDocument()
    expect(screen.getByText('Stopped')).toBeInTheDocument()
  })

  it('resets the failure count when a poll succeeds', async () => {
    sessionStorage.setItem('evalscope-active-task-eval', 'eval_1782864000000')
    const getProgress = vi.fn()
      .mockRejectedValueOnce(new Error('x'))
      .mockRejectedValueOnce(new Error('x'))
      .mockResolvedValueOnce({ percent: 10, status: 'running' })
      .mockRejectedValue(new Error('x'))
    await act(async () => {
      render(taskRunner({ getProgress }))
    })

    // fail, fail, success (resets the count), fail -> still below the threshold
    await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
    await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
    await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
    expect(screen.queryByText(/Progress updates are failing/)).not.toBeInTheDocument()

    // fail, fail -> the count reaches the threshold
    await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
    await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
    expect(screen.getByText(/Progress updates are failing/)).toBeInTheDocument()
  })

  it('submits through the injected API and renders the completed state', async () => {
    const submitTask = vi.fn().mockResolvedValue({ status: 'ok', task_id: 'eval_1782864000000' })
    const getProgress = vi.fn().mockResolvedValue({ percent: 100 })
    const getLog = vi.fn().mockResolvedValue({ text: '', head_line: 0, tail_line: 0, total_lines: 0 })

    render(
      <LocaleProvider>
        <TaskRunnerShell
          idPrefix="eval"
          title="Evaluation"
          configTitle="Configuration"
          statusTitle="Status"
          readyLabel="Ready"
          submitTask={submitTask}
          stopTask={vi.fn()}
          getProgress={getProgress}
          getLog={getLog}
          getReportUrl={(taskId) => `/report/${taskId}`}
          renderForm={({ onSubmit, disabled }) => (
            <button type="button" disabled={disabled} onClick={() => onSubmit({ model: 'qwen-plus' })}>
              Submit
            </button>
          )}
        />
      </LocaleProvider>,
    )

    expect(screen.getByText('Ready')).toBeInTheDocument()
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Submit' }))
      await Promise.resolve()
    })

    expect(submitTask).toHaveBeenCalledWith({ model: 'qwen-plus' }, 'eval_1782864000000')
    expect(screen.getByText('Completed')).toBeInTheDocument()
  })
})
