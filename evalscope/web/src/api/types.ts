// ------------------------------------------------------------------ //
// Shared TypeScript types for API responses                           //
// ------------------------------------------------------------------ //

export interface SubsetData {
  name: string
  score: number
  num: number
}

export interface CategoryData {
  name: string[]
  num: number
  score: number
  subsets: SubsetData[]
}

export interface MetricData {
  name: string
  num: number
  score: number
  categories: CategoryData[]
}

export interface PercentileStats {
  mean: number
  std: number
  min: number
  '25%': number
  '50%': number
  '75%': number
  '90%': number
  '99%': number
  max: number
}

export interface PerfMetricsSummary {
  n_samples: number
  latency: PercentileStats
  throughput: {
    avg_output_tps: number
    avg_req_ps: number
  }
  usage: {
    input_tokens: PercentileStats
    output_tokens: PercentileStats
    total_tokens: PercentileStats
    total_input_tokens?: number
    total_output_tokens?: number
    total_tokens_count?: number
  }
  ttft?: PercentileStats
  tpot?: PercentileStats
}

export interface PerfMetrics {
  summary: PerfMetricsSummary
}

export interface ReportData {
  name: string
  dataset_name: string
  model_name: string
  score: number
  analysis: string
  metrics: MetricData[]
  perf_metrics?: PerfMetrics
}

export interface LoadReportResponse {
  report_list: ReportData[]
  datasets: string[]
  task_config: Record<string, unknown>
}

export interface ScanResponse {
  reports: string[]
}

export interface DataFrameResponse {
  columns: string[]
  data: Record<string, unknown>[]
}

export interface SamplePerfMetrics {
  latency: number
  ttft?: number | null
  tpot?: number | null
  input_tokens: number
  output_tokens: number
}

export interface PredictionRow {
  Index: string
  Input: string
  Metadata: unknown
  Generated: string
  Gold: string
  Pred: string
  Score: Record<string, unknown>
  NScore: number
  PerfMetrics?: SamplePerfMetrics | null
}

export interface PredictionsResponse {
  predictions: PredictionRow[]
}

export interface AnalysisResponse {
  analysis: string
}

export interface BenchmarkEntry {
  name: string
  metrics: string[]
  meta: Record<string, unknown>
  description: {
    en?: { full: string; sections: Record<string, string> }
    zh?: { full: string; sections: Record<string, string> }
  }
}

export interface BenchmarksResponse {
  text?: BenchmarkEntry[]
  multimodal?: BenchmarkEntry[]
}

export interface EvalInvokeResponse {
  status: string
  task_id: string
  result?: unknown
  table?: string
  error?: string
}

export interface LogResponse {
  text: string
  head_line: number
  tail_line: number
  total_lines: number
}

export interface ProgressResponse {
  percent: number
  current_step?: string
  [key: string]: unknown
}

export interface ReportSummary {
  name: string
  model_name: string
  dataset_name: string
  score: number
  dataset_scores?: Record<string, number>
  num_samples: number
  timestamp: string
}

export interface ListReportsResponse {
  reports: ReportSummary[]
  total: number
  page: number
  page_size: number
  filters: {
    available_models: string[]
    available_datasets: string[]
  }
}
