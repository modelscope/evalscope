import { api } from './client'
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
}): Promise<ListReportsResponse> {
  return api<ListReportsResponse>(`${BASE}/list`, {
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
  })
}

export async function scanReports(rootPath: string): Promise<string[]> {
  const res = await api<ScanResponse>(`${BASE}/scan`, { root_path: rootPath })
  return res.reports
}

export async function loadReport(rootPath: string, reportName: string): Promise<LoadReportResponse> {
  return api<LoadReportResponse>(`${BASE}/load`, { root_path: rootPath, report_name: reportName })
}

export async function getDataFrame(
  rootPath: string,
  reportName: string,
  type: 'acc' | 'compare' | 'dataset' = 'acc',
  datasetName?: string,
): Promise<DataFrameResponse> {
  return api<DataFrameResponse>(`${BASE}/dataframe`, {
    root_path: rootPath,
    report_name: reportName,
    type,
    dataset_name: datasetName,
  })
}

export async function getPredictions(
  rootPath: string,
  reportName: string,
  datasetName: string,
  subsetName: string,
): Promise<PredictionsResponse> {
  return api<PredictionsResponse>(`${BASE}/predictions`, {
    root_path: rootPath,
    report_name: reportName,
    dataset_name: datasetName,
    subset_name: subsetName,
  })
}

export async function getAnalysis(rootPath: string, reportName: string, datasetName: string): Promise<string> {
  const res = await api<AnalysisResponse>(`${BASE}/analysis`, {
    root_path: rootPath,
    report_name: reportName,
    dataset_name: datasetName,
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
