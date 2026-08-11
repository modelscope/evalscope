import { apiDeleteValidated, apiPostValidated, apiValidated } from './client'
import {
  analysisResponseSchema,
  dataFrameResponseSchema,
  deleteReportResponseSchema,
  listReportsResponseSchema,
  loadReportResponseSchema,
  mergeReportResponseSchema,
  predictionsResponseSchema,
  renameReportResponseSchema,
} from './schemas'
import type {
  AnalysisResponse,
  DataFrameResponse,
  DeleteReportResponse,
  ListReportsResponse,
  LoadReportResponse,
  MergeReportResponse,
  PredictionsResponse,
  RenameReportResponse,
} from './types'
import { parseReportRef } from '@/domain/report/reportRef'

const BASE = '/api/v1/reports'

/** Path to one model report resource: `/api/v1/reports/runs/{runId}/models/{modelId}`. */
function reportPath(ref: string): string {
  const { runId, modelId } = parseReportRef(ref)
  return `${BASE}/runs/${encodeURIComponent(runId)}/models/${encodeURIComponent(modelId)}`
}

export async function listReports(params: {
  rootPath: string
  search?: string
  models?: string[]
  datasets?: string[]
  sortBy?: 'model' | 'dataset' | 'time'
  sortOrder?: 'asc' | 'desc'
  page?: number
  pageSize?: number
  /** Optional signal to cancel a superseded list/search request. */
  signal?: AbortSignal
}): Promise<ListReportsResponse> {
  return apiValidated(BASE, listReportsResponseSchema, {
    params: {
      root_path: params.rootPath,
      search: params.search,
      models: params.models?.join(';'),
      datasets: params.datasets?.join(';'),
      sort_by: params.sortBy,
      sort_order: params.sortOrder,
      page: params.page,
      page_size: params.pageSize,
    },
    signal: params.signal,
  })
}

export async function deleteReport(
  rootPath: string,
  ref: string,
  signal?: AbortSignal,
): Promise<DeleteReportResponse> {
  return apiDeleteValidated(reportPath(ref), deleteReportResponseSchema, {
    params: { root_path: rootPath },
    signal,
  })
}

export async function mergeReports(
  rootPath: string,
  refs: string[],
  signal?: AbortSignal,
): Promise<MergeReportResponse> {
  return apiPostValidated(
    `${BASE}/merge`,
    { root_path: rootPath, refs },
    mergeReportResponseSchema,
    { signal },
  )
}

export async function renameReport(
  rootPath: string,
  ref: string,
  newModelName: string,
  signal?: AbortSignal,
): Promise<RenameReportResponse> {
  return apiPostValidated(
    `${reportPath(ref)}/rename`,
    { root_path: rootPath, new_model_name: newModelName },
    renameReportResponseSchema,
    { signal },
  )
}

export async function loadReport(
  rootPath: string,
  ref: string,
  signal?: AbortSignal,
): Promise<LoadReportResponse> {
  return apiValidated(reportPath(ref), loadReportResponseSchema, {
    params: { root_path: rootPath },
    signal,
  })
}

export async function getDataFrame(
  rootPath: string,
  ref: string,
  view: 'acc' | 'compare' | 'dataset' = 'acc',
  datasetName?: string,
  signal?: AbortSignal,
): Promise<DataFrameResponse> {
  return apiValidated(`${reportPath(ref)}/table`, dataFrameResponseSchema, {
    params: {
      root_path: rootPath,
      view,
      dataset_name: datasetName,
    },
    signal,
  })
}

export async function getPredictions(
  rootPath: string,
  ref: string,
  datasetName: string,
  subsetName: string,
  signal?: AbortSignal,
): Promise<PredictionsResponse> {
  return apiValidated(`${reportPath(ref)}/predictions`, predictionsResponseSchema, {
    params: {
      root_path: rootPath,
      dataset_name: datasetName,
      subset_name: subsetName,
    },
    signal,
  })
}

export async function getAnalysis(
  rootPath: string,
  ref: string,
  datasetName: string,
  signal?: AbortSignal,
): Promise<string> {
  const res: AnalysisResponse = await apiValidated(`${reportPath(ref)}/analysis`, analysisResponseSchema, {
    params: {
      root_path: rootPath,
      dataset_name: datasetName,
    },
    signal,
  })
  return res.analysis
}

export function getHtmlReportUrl(rootPath: string, ref: string): string {
  const { runId } = parseReportRef(ref)
  return `${BASE}/runs/${encodeURIComponent(runId)}/html?root_path=${encodeURIComponent(rootPath)}`
}

/** URL of a multi-report comparison chart (`radar` | `grouped_bar`), one `report=` per reference. */
export function getCompareChartUrl(
  rootPath: string,
  refs: string[],
  chartType: 'radar' | 'grouped_bar',
): string {
  const params = new URLSearchParams({ root_path: rootPath })
  for (const ref of refs) params.append('report', ref)
  return `${BASE}/charts/${chartType}?${params.toString()}`
}
