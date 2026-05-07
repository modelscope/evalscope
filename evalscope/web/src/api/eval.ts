import { apiPost, api } from './client'
import type { BenchmarksResponse, EvalInvokeResponse, LogResponse, ProgressResponse } from './types'

export async function submitEvalTask(
  payload: Record<string, unknown>,
  taskId: string,
): Promise<EvalInvokeResponse> {
  return apiPost<EvalInvokeResponse>('/api/v1/eval/invoke', payload, { 'EvalScope-Task-Id': taskId })
}

export async function getEvalProgress(taskId: string): Promise<ProgressResponse> {
  return api<ProgressResponse>('/api/v1/eval/progress', { task_id: taskId })
}

export async function getEvalLog(taskId: string, startLine?: number, page = 500): Promise<LogResponse> {
  const params: Record<string, string> = { task_id: taskId, page: String(page) }
  if (startLine !== undefined) params.start_line = String(startLine)
  return api<LogResponse>('/api/v1/eval/log', params)
}

export function getEvalReportUrl(taskId: string): string {
  return `/api/v1/eval/report?task_id=${encodeURIComponent(taskId)}`
}

export async function stopEvalTask(taskId: string): Promise<{ status: string; task_id: string }> {
  return apiPost<{ status: string; task_id: string }>(`/api/v1/eval/stop?task_id=${encodeURIComponent(taskId)}`, {})
}

export async function listBenchmarks(type?: 'text' | 'multimodal', all?: boolean): Promise<BenchmarksResponse> {
  const params: Record<string, string> = {}
  if (type) params.type = type
  if (all) params.all = 'true'
  return api<BenchmarksResponse>('/api/v1/eval/benchmarks', params)
}
