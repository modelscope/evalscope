import { apiPost, api } from './client'
import type {
  EvalInvokeResponse,
  ListPerfRunsResponse,
  LogResponse,
  PerfDetailResponse,
  ProgressResponse,
} from './types'

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

export async function stopPerfTask(taskId: string): Promise<{ status: string; task_id: string }> {
  return apiPost<{ status: string; task_id: string }>(`/api/v1/perf/stop?task_id=${encodeURIComponent(taskId)}`, {})
}

// ------------------------------------------------------------------ //
// Historical perf-run archive                                         //
// ------------------------------------------------------------------ //

export async function listPerfRuns(rootPath: string): Promise<ListPerfRunsResponse> {
  return api<ListPerfRunsResponse>('/api/v1/perf/list', { root_path: rootPath })
}

export async function getPerfDetail(rootPath: string, path: string): Promise<PerfDetailResponse> {
  return api<PerfDetailResponse>('/api/v1/perf/detail', { root_path: rootPath, path })
}

export function getPerfChartUrl(rootPath: string, path: string, chartType: string): string {
  const params = new URLSearchParams({ root_path: rootPath, path, chart_type: chartType })
  return `/api/v1/perf/chart?${params.toString()}`
}

export function getPerfHistoryReportUrl(rootPath: string, path: string): string {
  const params = new URLSearchParams({ root_path: rootPath, path })
  return `/api/v1/perf/history/report?${params.toString()}`
}
