import { beforeEach, describe, expect, it, vi } from 'vitest'
import { apiPostValidated, apiValidated } from './client'
import { createTaskApi } from './task'

vi.mock('./client', () => ({
  apiPostValidated: vi.fn(),
  apiValidated: vi.fn(),
}))

const post = vi.mocked(apiPostValidated)
const get = vi.mocked(apiValidated)

beforeEach(() => {
  post.mockResolvedValue({ status: 'ok', task_id: 'task-1' })
  get.mockResolvedValue({ percent: 50 })
})

describe.each(['eval', 'perf'] as const)('createTaskApi(%s)', (scope) => {
  it('keeps all task lifecycle requests under the selected API scope', async () => {
    const api = createTaskApi(scope)

    await api.submit({ model: 'qwen-plus' }, 'task-1')
    await api.progress('task-1')
    await api.log('task-1', 12, 100)
    await api.stop('task-1')

    expect(post).toHaveBeenNthCalledWith(
      1,
      `/api/v1/${scope}/invoke`,
      { model: 'qwen-plus' },
      expect.anything(),
      expect.objectContaining({ headers: { 'EvalScope-Task-Id': 'task-1' } }),
    )
    expect(get).toHaveBeenNthCalledWith(
      1,
      `/api/v1/${scope}/progress`,
      expect.anything(),
      expect.objectContaining({ params: { task_id: 'task-1' } }),
    )
    expect(get).toHaveBeenNthCalledWith(
      2,
      `/api/v1/${scope}/log`,
      expect.anything(),
      expect.objectContaining({ params: { task_id: 'task-1', start_line: '12', page: '100' } }),
    )
    expect(post).toHaveBeenNthCalledWith(
      2,
      `/api/v1/${scope}/stop`,
      {},
      expect.anything(),
      expect.objectContaining({ params: { task_id: 'task-1' } }),
    )
    expect(api.reportUrl('task 1')).toBe(`/api/v1/${scope}/report?task_id=task%201`)
  })
})
