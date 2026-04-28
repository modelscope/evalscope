import { apiPost, api } from './client'
import type { EvalInvokeResponse, LogResponse, ProgressResponse } from './types'

export async function submitPerfTask(
  payload: Record<string, unknown>,
  taskId: string,
): Promise<EvalInvokeResponse> {
  return apiPost<EvalInvokeResponse>('/api/v1/perf/invoke', payload, { 'EvalScope-Task-Id': taskId })
}

export async function getPerfProgress(taskId: string): Promise<ProgressResponse> {
  return api<ProgressResponse>('/api/v1/perf/progress', { task_id: taskId })
}

export async function getPerfLog(taskId: string, startLine?: number, page = 500): Promise<LogResponse> {
  const params: Record<string, string> = { task_id: taskId, page: String(page) }
  if (startLine !== undefined) params.start_line = String(startLine)
  return api<LogResponse>('/api/v1/perf/log', params)
}

export function getPerfReportUrl(taskId: string): string {
  return `/api/v1/perf/report?task_id=${encodeURIComponent(taskId)}`
}
