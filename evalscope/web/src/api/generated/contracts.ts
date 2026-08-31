/* eslint-disable */
/** Generated from Pydantic Web API response models. Do not edit. */

/**
 * Whether a score is usable, and why it is not.
 *
 * A failed judge is not a zero: only ``SUCCESS`` and ``FALLBACK`` carry a score that may
 * enter aggregation. The remaining values mean the score is unavailable and the sample
 * must be excluded from the affected metric rather than counted as 0.
 */
export type ScoreStatus =
  "success" | "transport_error" | "parse_error" | "invalid_session" | "fallback" | "degraded" | "excluded";
/**
 * Optimization direction of a metric.
 */
export type MetricDirection = "higher_is_better" | "lower_is_better" | "none";
/**
 * How a metric value is rendered.
 */
export type MetricDisplayKind = "number" | "percent";
/**
 * Whether a metric grades quality or only describes a run.
 */
export type MetricKind = "quality" | "diagnostic";
/**
 * Canonical event kinds emitted by the AgentLoop or external-agent bridge.
 */
export type EventType =
  | "model_generate"
  | "tool_call"
  | "tool_result"
  | "env_exec"
  | "env_reset"
  | "error"
  | "submit"
  | "nudge"
  | "run_start"
  | "run_end";

/**
 * Schema-only root that keeps every public Web API type reachable for code generation.
 */
export interface WebApiContracts {
  analysis_response: AnalysisResponse;
  benchmark_entry: BenchmarkEntry;
  benchmarks_response: BenchmarksResponse;
  chat_message: ChatMessage;
  config_response: ConfigResponse;
  content_block: ContentBlock;
  data_frame_response: DataFrameResponse;
  delete_perf_run_response: DeletePerfRunResponse;
  delete_report_response: DeleteReportResponse;
  eval_invoke_response: EvalInvokeResponse;
  execution_summary: ExecutionSummary;
  judge_attempt: JudgeAttempt;
  judge_summary: JudgeSummary;
  list_perf_runs_response: ListPerfRunsResponse;
  list_reports_response: ListReportsResponse;
  load_report_response: LoadReportResponse;
  log_response: LogResponse;
  metric_display_kind: MetricDisplayKind;
  metric_identity: MetricIdentity;
  metric_semantics: MetricSemantics;
  percentile_stats: PercentileStats;
  perf_detail_response: PerfDetailResponse;
  perf_metrics: PerfMetrics;
  perf_requests_response: PerfRequestsResponse;
  perf_run_item: PerfRunItem;
  perf_run_summary: PerfRunSummary;
  perf_runs_list_response: PerfRunsListResponse;
  prediction_row: PredictionRow;
  prediction_score: PredictionScore;
  predictions_response: PredictionsResponse;
  progress_response: ProgressResponse;
  report_data: ReportData;
  report_metric: ReportMetric;
  report_summary: ReportSummary;
  task_status_response: TaskStatusResponse;
  tool_call: PredictionToolCall;
  value_range: ValueRange;
  [k: string]: any;
}
export interface AnalysisResponse {
  analysis: string;
}
export interface BenchmarkEntry {
  category: "llm" | "vlm" | "agent" | "aigc";
  dataset_id: string;
  description: BenchmarkDescription;
  few_shot_num: number;
  meta: {
    [k: string]: any;
  };
  metrics: string[];
  name: string;
  paper_url?: string | null;
  pretty_name: string;
  subset_list: string[];
  tags: string[];
  total_samples: number;
}
export interface BenchmarkDescription {
  en?: BenchmarkDescriptionLocale | null;
  zh?: BenchmarkDescriptionLocale | null;
}
export interface BenchmarkDescriptionLocale {
  full: string;
  sections: {
    [k: string]: string;
  };
}
export interface BenchmarksResponse {
  agent?: BenchmarkEntry[] | null;
  aigc?: BenchmarkEntry[] | null;
  multimodal?: BenchmarkEntry[] | null;
  text?: BenchmarkEntry[] | null;
}
export interface ChatMessage {
  content: string | ContentBlock[];
  error?: ToolMessageError | null;
  function?: string | null;
  id?: string | null;
  metadata?: {
    [k: string]: any;
  } | null;
  model?: string | null;
  perf_metrics?: SamplePerfMetrics | null;
  role: "system" | "user" | "assistant" | "tool";
  source?: ("input" | "generate") | null;
  tool_call_id?: string | string[] | null;
  tool_calls?: PredictionToolCall[] | null;
}
export interface ContentBlock {
  audio?: string | null;
  data?: {
    [k: string]: any;
  } | null;
  detail?: string | null;
  format?: string | null;
  image?: string | null;
  reasoning?: string | null;
  reasoning_tokens?: number | null;
  text?: string | null;
  type: "text" | "reasoning" | "image" | "audio" | "video" | "data";
  video?: string | null;
  [k: string]: any;
}
export interface ToolMessageError {
  message: string;
  type?: string | null;
}
export interface SamplePerfMetrics {
  input_tokens: number;
  latency: number;
  output_tokens: number;
  tpot?: number | null;
  ttft?: number | null;
}
export interface PredictionToolCall {
  arguments: {
    [k: string]: any;
  };
  function: string;
  id: string;
}
export interface ConfigResponse {
  outputs_root: string;
}
export interface DataFrameResponse {
  columns: string[];
  data: {
    [k: string]: any;
  }[];
}
export interface DeletePerfRunResponse {
  path: string;
  success: boolean;
}
export interface DeleteReportResponse {
  model_id: string;
  run_id: string;
  success: boolean;
}
export interface EvalInvokeResponse {
  error?: string | null;
  result?: {
    [k: string]: any;
  };
  status: "ok" | "completed" | "error" | "stopped";
  table?: string | null;
  task_id: string;
}
/**
 * Run completeness kept separate from aggregation-specific metric counts.
 */
export interface ExecutionSummary {
  errored?: number;
  incomplete?: boolean;
  requested?: number;
  subsets?: {
    [k: string]: ExecutionSubset;
  };
  succeeded?: number;
  [k: string]: any;
}
/**
 * Completion counts for one evaluated subset.
 */
export interface ExecutionSubset {
  errored?: number;
  requested?: number;
  succeeded?: number;
  [k: string]: any;
}
export interface JudgeAttempt {
  case_id: string;
  error?: string | null;
  judge_id: string;
  latency?: number | null;
  messages?: any[];
  model_output?: {
    [k: string]: any;
  };
  parsed_value?: {
    [k: string]: any;
  };
  placement?: "original" | "swapped";
  raw_response?: string | null;
  repeat_id?: number;
  status: ScoreStatus;
}
/**
 * First-class summary of a judge session, for reports and offline inspection.
 */
export interface JudgeSummary {
  coverage?: number;
  disagreement?: {
    [k: string]: any;
  };
  error?: string | null;
  failures?: {
    [k: string]: number;
  };
  judge_models?: string[];
  scored?: number;
  /**
   * Whether a score is usable, and why it is not.
   *
   * A failed judge is not a zero: only ``SUCCESS`` and ``FALLBACK`` carry a score that may
   * enter aggregation. The remaining values mean the score is unavailable and the sample
   * must be excluded from the affected metric rather than counted as 0.
   */
  status?: "success" | "transport_error" | "parse_error" | "invalid_session" | "fallback" | "degraded" | "excluded";
  total?: number;
  total_observations?: number;
  valid_observations?: number;
  [k: string]: any;
}
export interface ListPerfRunsResponse {
  metric_semantics?: {
    [k: string]: MetricSemantics;
  };
  runs: PerfRunSummary[];
  total: number;
}
export interface MetricSemantics {
  direction: MetricDirection;
  display_kind: MetricDisplayKind;
  display_multiplier?: number | null;
  display_name?: string | null;
  display_precision: number;
  display_unit?: string | null;
  kind: MetricKind;
  metric_name: string;
  raw_unit?: string | null;
  semantic_id: string;
  value_range?: ValueRange | null;
}
export interface ValueRange {
  max: number;
  min: number;
}
export interface PerfRunSummary {
  api_host?: string | null;
  api_type: string;
  avg_input_tokens?: number | null;
  avg_output_tokens?: number | null;
  best_latency: number;
  best_rps: number;
  concurrency?: number[] | null;
  dataset: string;
  has_html: boolean;
  is_embedding: boolean;
  model: string;
  num_runs: number;
  path: string;
  protocol?: string | null;
  provider?: string | null;
  success_rate: number;
  timestamp: string;
  total_requests: number;
}
export interface ListReportsResponse {
  filters: ReportFilters;
  page: number;
  page_size: number;
  reports: ReportSummary[];
  total: number;
}
export interface ReportFilters {
  available_datasets: string[];
  available_models: string[];
}
export interface ReportSummary {
  dataset_name: string;
  dataset_pretty_name?: string;
  model_id: string;
  model_name: string;
  num_samples: number;
  primary_metrics: PrimaryMetricRef[];
  run_id: string;
  timestamp: string;
}
export interface PrimaryMetricRef {
  dataset_name: string;
  dataset_pretty_name?: string;
  identity: MetricIdentity;
  score: number;
  semantics: MetricSemantics;
}
export interface MetricIdentity {
  aggregation: string;
  dimensions: {
    [k: string]: string | number | boolean;
  };
  name: string;
}
export interface LoadReportResponse {
  datasets: string[];
  report_list: ReportData[];
  task_config: {
    [k: string]: any;
  };
}
/**
 * Canonical report payload emitted by ``Report.to_dict``.
 */
export interface ReportData {
  analysis: string;
  dataset_description?: string;
  dataset_name: string;
  dataset_pretty_name?: string;
  execution_summary?: ExecutionSummary | null;
  judge_summary?: JudgeSummary | null;
  metrics: ReportMetric[];
  model_name: string;
  name: string;
  num?: number;
  perf_metrics?: PerfMetrics | null;
  primary_metric_identity?: MetricIdentity | null;
  primary_metric_unavailable_reason?: string | null;
  schema_version: 2;
}
export interface ReportMetric {
  categories: ReportCategory[];
  identity: MetricIdentity;
  legacy_name?: string | null;
  macro_score?: number;
  num: number;
  score: number;
  semantics: MetricSemantics;
}
export interface ReportCategory {
  macro_score?: number;
  name: string[];
  num: number;
  score: number;
  subsets: ReportSubset[];
}
export interface ReportSubset {
  is_aggregate?: boolean;
  name: string;
  num: number;
  score: number;
}
export interface PerfMetrics {
  coverage?: PerfCoverage | null;
  metric_semantics?: {
    [k: string]: MetricSemantics;
  };
  summary?: PerfMetricsSummary | null;
}
export interface PerfCoverage {
  requests_with_metrics: number;
  total_requests: number;
}
export interface PerfMetricsSummary {
  latency: PercentileStats;
  n_samples: number;
  throughput: ThroughputSummary;
  tpot?: PercentileStats | null;
  ttft?: PercentileStats | null;
  usage: UsageSummary;
}
export interface PercentileStats {
  "25%": number;
  "50%": number;
  "75%": number;
  "90%": number;
  "99%": number;
  max: number;
  mean: number;
  min: number;
  std: number | null;
}
export interface ThroughputSummary {
  avg_output_tps: number;
  avg_req_ps: number;
}
export interface UsageSummary {
  input_tokens: PercentileStats;
  output_tokens: PercentileStats;
  total_input_tokens?: number | null;
  total_output_tokens?: number | null;
  total_tokens: PercentileStats;
  total_tokens_count?: number | null;
}
export interface LogResponse {
  head_line: number;
  tail_line: number;
  text: string;
  total_lines: number;
}
export interface PerfDetailResponse {
  api_type: string;
  basic_info: {
    [k: string]: string;
  };
  best_config: {
    [k: string]: string;
  };
  dataset: string;
  generated_at: string;
  has_html: boolean;
  is_embedding: boolean;
  model: string;
  num_runs: number;
  path: string;
  recommendations: string[];
  summary_columns: PerfSummaryColumn[];
  summary_rows: PerfSummaryRow[];
  total_requests: number;
}
export interface PerfSummaryColumn {
  key: string;
  label: string;
  semantics: MetricSemantics | null;
}
export interface PerfSummaryRow {
  sample_counts: {
    [k: string]: number;
  };
  values: {
    [k: string]: number;
  };
}
export interface PerfRequestsResponse {
  columns: string[];
  has_db: boolean;
  page: number;
  page_size: number;
  rows: {
    [k: string]: any;
  }[];
  total: number;
}
export interface PerfRunItem {
  dir_name: string;
  has_requests: boolean;
  name: string;
  num_requests: number;
  number: number;
  parallel: number;
  percentile_columns: string[];
  percentile_rows: (string | number)[][];
  rate: number | null;
  succeed_requests: number;
  success_rate: number;
  total_requests: number;
}
export interface PerfRunsListResponse {
  runs: PerfRunItem[];
  total: number;
}
export interface PredictionRow {
  AgentTrace?: AgentTrace | null;
  Generated: string;
  Gold: string;
  Index: string;
  Input: string;
  Messages?: ChatMessage[] | null;
  Metadata: any;
  NScore: number | null;
  PerfMetrics?: SamplePerfMetrics | null;
  Pred: string;
  Score: PredictionScore;
  Status?: string | null;
}
export interface AgentTrace {
  environment?: string | null;
  events: AgentTraceEvent[];
  max_steps: number;
  strategy?: string | null;
}
export interface AgentTraceEvent {
  latency_ms?: number | null;
  message_id?: string | null;
  payload: {
    [k: string]: any;
  };
  step: number;
  timestamp: number;
  token_usage?: {
    [k: string]: number;
  } | null;
  type: EventType;
}
/**
 * Score fields rendered by the Web UI; metadata remains benchmark-defined.
 */
export interface PredictionScore {
  explanation?: string | null;
  judge_summary?: JudgeSummary | null;
  main_score_name?: string | null;
  metadata?: PredictionMetadata | null;
  /**
   * Whether a score is usable, and why it is not.
   *
   * A failed judge is not a zero: only ``SUCCESS`` and ``FALLBACK`` carry a score that may
   * enter aggregation. The remaining values mean the score is unavailable and the sample
   * must be excluded from the affected metric rather than counted as 0.
   */
  status?: "success" | "transport_error" | "parse_error" | "invalid_session" | "fallback" | "degraded" | "excluded";
  value?: {
    [k: string]: number | boolean;
  };
  [k: string]: any;
}
export interface PredictionMetadata {
  judge_attempts?: JudgeAttempt[] | null;
  judge_skip_reason?: string | null;
  judge_skipped?: boolean | null;
  [k: string]: any;
}
export interface PredictionsResponse {
  predictions: PredictionRow[];
}
/**
 * Progress tracker payload with stable fields and tracker-specific metadata.
 */
export interface ProgressResponse {
  current_step?: string | null;
  percent: number;
  status?: string | null;
  [k: string]: any;
}
export interface TaskStatusResponse {
  status: string;
  task_id: string;
}
