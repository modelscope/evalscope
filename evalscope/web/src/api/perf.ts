import { apiValidated } from './client'
import {
  listPerfRunsResponseSchema,
  perfDetailResponseSchema,
  perfRequestsResponseSchema,
  perfRunsListResponseSchema,
} from './schemas'
import type {
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
