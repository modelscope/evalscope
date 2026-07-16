import { apiValidated, apiPostValidated } from './client'
import {
  benchmarksResponseSchema,
  evalInvokeResponseSchema,
  logResponseSchema,
  progressResponseSchema,
  taskStatusResponseSchema,
} from './schemas'
import type { BenchmarksResponse, EvalInvokeResponse, LogResponse, ProgressResponse } from './types'

export async function submitEvalTask(
  payload: Record<string, unknown>,
  taskId: string,
  signal?: AbortSignal,
): Promise<EvalInvokeResponse> {
  return apiPostValidated('/api/v1/eval/invoke', payload, evalInvokeResponseSchema, {
    headers: { 'EvalScope-Task-Id': taskId },
    signal,
  })
}

export async function getEvalProgress(taskId: string, signal?: AbortSignal): Promise<ProgressResponse> {
  return apiValidated('/api/v1/eval/progress', progressResponseSchema, {
    params: { task_id: taskId },
    signal,
  })
}

export async function getEvalLog(
  taskId: string,
  startLine?: number,
  page = 500,
  signal?: AbortSignal,
): Promise<LogResponse> {
  const params: Record<string, string> = { task_id: taskId, page: String(page) }
  if (startLine !== undefined) params.start_line = String(startLine)
  return apiValidated('/api/v1/eval/log', logResponseSchema, { params, signal })
}

export function getEvalReportUrl(taskId: string): string {
  return `/api/v1/eval/report?task_id=${encodeURIComponent(taskId)}`
}

export async function stopEvalTask(taskId: string, signal?: AbortSignal): Promise<{ status: string; task_id: string }> {
  return apiPostValidated(
    '/api/v1/eval/stop',
    {},
    taskStatusResponseSchema,
    { params: { task_id: taskId }, signal },
  )
}

export async function listBenchmarks(
  type?: 'text' | 'multimodal',
  all?: boolean,
  signal?: AbortSignal,
): Promise<BenchmarksResponse> {
  const params: Record<string, string> = {}
  if (type) params.type = type
  if (all) params.all = 'true'
  return apiValidated('/api/v1/eval/benchmarks', benchmarksResponseSchema, { params, signal })
}
