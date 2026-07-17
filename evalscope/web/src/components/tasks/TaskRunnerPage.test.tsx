import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { LocaleProvider } from '@/contexts/LocaleContext'
import TaskRunnerPage from './TaskRunnerPage'

afterEach(cleanup)

describe('TaskRunnerPage', () => {
  it('submits through the injected API and renders the completed state', async () => {
    const submitTask = vi.fn().mockResolvedValue({ status: 'ok', task_id: 'eval_1782864000000' })
    const getProgress = vi.fn().mockResolvedValue({ percent: 100 })
    const getLog = vi.fn().mockResolvedValue({ text: '', head_line: 0, tail_line: 0, total_lines: 0 })

    render(
      <LocaleProvider>
        <TaskRunnerPage
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
