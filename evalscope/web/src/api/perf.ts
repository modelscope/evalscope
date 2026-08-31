import { apiDeleteValidated, apiValidated } from './client'
import type {
  DeletePerfRunResponse,
  ListPerfRunsResponse,
  PerfDetailResponse,
  PerfRequestsResponse,
  PerfRunsListResponse,
} from './types'
import { createTaskApi } from './task'

const perfTaskApi = createTaskApi('perf')

export const submitPerfTask = perfTaskApi.submit
export const getPerfProgress = perfTaskApi.progress
export const getPerfLog = perfTaskApi.log
export const getPerfReportUrl = perfTaskApi.reportUrl
export const stopPerfTask = perfTaskApi.stop

// ------------------------------------------------------------------ //
// Historical perf-run archive                                         //
// ------------------------------------------------------------------ //

export async function listPerfRuns(rootPath: string, signal?: AbortSignal): Promise<ListPerfRunsResponse> {
  return apiValidated<ListPerfRunsResponse>('/api/v1/perf/list', {
    params: { root_path: rootPath },
    signal,
  })
}

export async function deletePerfRun(
  rootPath: string,
  path: string,
  signal?: AbortSignal,
): Promise<DeletePerfRunResponse> {
  return apiDeleteValidated<DeletePerfRunResponse>('/api/v1/perf/run', {
    params: { root_path: rootPath, path },
    signal,
  })
}

export async function getPerfDetail(
  rootPath: string,
  path: string,
  signal?: AbortSignal,
): Promise<PerfDetailResponse> {
  return apiValidated<PerfDetailResponse>('/api/v1/perf/detail', {
    params: { root_path: rootPath, path },
    signal,
  })
}

export async function listPerfRunDetails(
  rootPath: string,
  path: string,
  signal?: AbortSignal,
): Promise<PerfRunsListResponse> {
  return apiValidated<PerfRunsListResponse>('/api/v1/perf/runs', {
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
  return apiValidated<PerfRequestsResponse>('/api/v1/perf/requests', {
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
