import { apiValidated } from './client'
import {
  analysisResponseSchema,
  dataFrameResponseSchema,
  listReportsResponseSchema,
  loadReportResponseSchema,
  predictionsResponseSchema,
  scanResponseSchema,
} from './schemas'
import type {
  AnalysisResponse,
  DataFrameResponse,
  ListReportsResponse,
  LoadReportResponse,
  PredictionsResponse,
  ScanResponse,
} from './types'

const BASE = '/api/v1/reports'

export async function listReports(params: {
  rootPath: string
  search?: string
  models?: string[]
  datasets?: string[]
  scoreMin?: number
  scoreMax?: number
  sortBy?: 'score' | 'model' | 'dataset' | 'time'
  sortOrder?: 'asc' | 'desc'
  page?: number
  pageSize?: number
  /** Optional signal to cancel a superseded list/search request (Req 13.3, 13.5). */
  signal?: AbortSignal
}): Promise<ListReportsResponse> {
  return apiValidated(`${BASE}/list`, listReportsResponseSchema, {
    params: {
      root_path: params.rootPath,
      search: params.search,
      models: params.models?.join(';'),
      datasets: params.datasets?.join(';'),
      score_min: params.scoreMin,
      score_max: params.scoreMax,
      sort_by: params.sortBy,
      sort_order: params.sortOrder,
      page: params.page,
      page_size: params.pageSize,
    },
    signal: params.signal,
  })
}

export async function scanReports(rootPath: string, signal?: AbortSignal): Promise<string[]> {
  const res: ScanResponse = await apiValidated(`${BASE}/scan`, scanResponseSchema, {
    params: { root_path: rootPath },
    signal,
  })
  return res.reports
}

export async function loadReport(
  rootPath: string,
  reportName: string,
  signal?: AbortSignal,
): Promise<LoadReportResponse> {
  return apiValidated(`${BASE}/load`, loadReportResponseSchema, {
    params: { root_path: rootPath, report_name: reportName },
    signal,
  })
}

export async function getDataFrame(
  rootPath: string,
  reportName: string,
  type: 'acc' | 'compare' | 'dataset' = 'acc',
  datasetName?: string,
  signal?: AbortSignal,
): Promise<DataFrameResponse> {
  return apiValidated(`${BASE}/dataframe`, dataFrameResponseSchema, {
    params: {
      root_path: rootPath,
      report_name: reportName,
      type,
      dataset_name: datasetName,
    },
    signal,
  })
}

export async function getPredictions(
  rootPath: string,
  reportName: string,
  datasetName: string,
  subsetName: string,
  signal?: AbortSignal,
): Promise<PredictionsResponse> {
  return apiValidated(`${BASE}/predictions`, predictionsResponseSchema, {
    params: {
      root_path: rootPath,
      report_name: reportName,
      dataset_name: datasetName,
      subset_name: subsetName,
    },
    signal,
  })
}

export async function getAnalysis(
  rootPath: string,
  reportName: string,
  datasetName: string,
  signal?: AbortSignal,
): Promise<string> {
  const res: AnalysisResponse = await apiValidated(`${BASE}/analysis`, analysisResponseSchema, {
    params: {
      root_path: rootPath,
      report_name: reportName,
      dataset_name: datasetName,
    },
    signal,
  })
  return res.analysis
}

export function getHtmlReportUrl(rootPath: string, reportName: string): string {
  return `${BASE}/html?root_path=${encodeURIComponent(rootPath)}&report_name=${encodeURIComponent(reportName)}`
}

export function getChartUrl(
  rootPath: string,
  chartType: 'scores' | 'sunburst' | 'dataset_scores' | 'radar' | 'histogram' | 'grouped_bar',
  opts: { reportName?: string; reportNames?: string[]; datasetName?: string; subsetName?: string } = {},
): string {
  const params = new URLSearchParams({ root_path: rootPath, chart_type: chartType })
  if (opts.reportName) params.set('report_name', opts.reportName)
  if (opts.reportNames?.length) params.set('report_names', opts.reportNames.join(';'))
  if (opts.datasetName) params.set('dataset_name', opts.datasetName)
  if (opts.subsetName) params.set('subset_name', opts.subsetName)
  return `${BASE}/chart?${params.toString()}`
}
