import { apiValidated, apiPostValidated } from './client'
import {
  evalInvokeResponseSchema,
  logResponseSchema,
  listPerfRunsResponseSchema,
  perfDetailResponseSchema,
  perfRequestsResponseSchema,
  perfRunsListResponseSchema,
  progressResponseSchema,
  taskStatusResponseSchema,
} from './schemas'
import type {
  EvalInvokeResponse,
  ListPerfRunsResponse,
  LogResponse,
  PerfDetailResponse,
  PerfRequestsResponse,
  PerfRunsListResponse,
  ProgressResponse,
} from './types'

export async function submitPerfTask(
  payload: Record<string, unknown>,
  taskId: string,
  signal?: AbortSignal,
): Promise<EvalInvokeResponse> {
  return apiPostValidated('/api/v1/perf/invoke', payload, evalInvokeResponseSchema, {
    headers: { 'EvalScope-Task-Id': taskId },
    signal,
  })
}

export async function getPerfProgress(taskId: string, signal?: AbortSignal): Promise<ProgressResponse> {
  return apiValidated('/api/v1/perf/progress', progressResponseSchema, {
    params: { task_id: taskId },
    signal,
  })
}

export async function getPerfLog(
  taskId: string,
  startLine?: number,
  page = 500,
  signal?: AbortSignal,
): Promise<LogResponse> {
  const params: Record<string, string> = { task_id: taskId, page: String(page) }
  if (startLine !== undefined) params.start_line = String(startLine)
  return apiValidated('/api/v1/perf/log', logResponseSchema, { params, signal })
}

export function getPerfReportUrl(taskId: string): string {
  return `/api/v1/perf/report?task_id=${encodeURIComponent(taskId)}`
}

export async function stopPerfTask(taskId: string, signal?: AbortSignal): Promise<{ status: string; task_id: string }> {
  return apiPostValidated(
    '/api/v1/perf/stop',
    {},
    taskStatusResponseSchema,
    { params: { task_id: taskId }, signal },
  )
}

// ------------------------------------------------------------------ //
// Historical perf-run archive                                         //
// ------------------------------------------------------------------ //

export async function listPerfRuns(rootPath: string, signal?: AbortSignal): Promise<ListPerfRunsResponse> {
  return apiValidated('/api/v1/perf/list', listPerfRunsResponseSchema, {
    params: { root_path: rootPath },
    signal,
  })
}

export async function getPerfDetail(
  rootPath: string,
  path: string,
  signal?: AbortSignal,
): Promise<PerfDetailResponse> {
  return apiValidated('/api/v1/perf/detail', perfDetailResponseSchema, {
    params: { root_path: rootPath, path },
    signal,
  })
}

export async function listPerfRunDetails(
  rootPath: string,
  path: string,
  signal?: AbortSignal,
): Promise<PerfRunsListResponse> {
  return apiValidated('/api/v1/perf/runs', perfRunsListResponseSchema, {
    params: { root_path: rootPath, path },
    signal,
  })
}

export async function getPerfRequests(params: {
  rootPath: string
  path: string
  run: string
  status?: 'success' | 'failed'
  page?: number
  pageSize?: number
  signal?: AbortSignal
}): Promise<PerfRequestsResponse> {
  return apiValidated('/api/v1/perf/requests', perfRequestsResponseSchema, {
    params: {
      root_path: params.rootPath,
      path: params.path,
      run: params.run,
      status: params.status,
      page: params.page,
      page_size: params.pageSize,
    },
    signal: params.signal,
  })
}

export function getPerfChartUrl(
  rootPath: string,
  path: string,
  chartType: string,
  opts: { run?: string; theme?: string } = {},
): string {
  const params = new URLSearchParams({ root_path: rootPath, path, chart_type: chartType })
  if (opts.run) params.set('run', opts.run)
  if (opts.theme) params.set('theme', opts.theme)
  return `/api/v1/perf/chart?${params.toString()}`
}

export function getPerfHistoryReportUrl(rootPath: string, path: string): string {
  const params = new URLSearchParams({ root_path: rootPath, path })
  return `/api/v1/perf/history/report?${params.toString()}`
}

export function getPerfCompareChartUrl(
  rootPath: string,
  paths: string[],
  chartType: string,
  theme?: string,
): string {
  const params = new URLSearchParams({ root_path: rootPath, paths: paths.join(';'), chart_type: chartType })
  if (theme) params.set('theme', theme)
  return `/api/v1/perf/compare/chart?${params.toString()}`
}
