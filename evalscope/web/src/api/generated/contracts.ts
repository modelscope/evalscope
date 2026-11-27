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
 * Top-level kind of a finding.
 */
export type FindingCategory =
  | "tool_failure"
  | "repetition"
  | "termination"
  | "side_effect"
  | "tainted_side_effect"
  | "attack_goal_action"
  | "risky_command"
  | "refusal";
/**
 * How much a finding can be trusted.
 */
export type Confidence = "high" | "medium" | "low";
/**
 * Who is responsible for a finding.
 */
export type Attribution = "model" | "guard" | "environment" | "benchmark" | "unknown";

/**
 * Schema-only root that keeps every public Web API type reachable for code generation.
 */
export interface WebApiContracts {
  active_task_info: ActiveTaskInfo;
  analysis_response: AnalysisResponse;
  benchmark_entry: BenchmarkEntry;
  benchmarks_response: BenchmarksResponse;
  chat_message: ChatMessage;
  config_response: ConfigResponse;
  content_block: ContentBlock;
  data_frame_response: DataFrameResponse;
  delete_perf_run_response: DeletePerfRunResponse;
  delete_report_response: DeleteReportResponse;
  diagnosis_response: DiagnosisResponse;
  diagnosis_run_request: DiagnosisRunRequest;
  eval_invoke_response: EvalInvokeResponse;
  execution_summary: ExecutionSummary;
  judge_attempt: JudgeAttempt;
  judge_summary: JudgeSummary;
  list_active_tasks_response: ListActiveTasksResponse;
  list_perf_runs_response: ListPerfRunsResponse;
  list_reports_grouped_response: ListReportsGroupedResponse;
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
  report_group: ReportGroup;
  report_metric: ReportMetric;
  report_summary: ReportSummary;
  sample_diagnoses_response: SampleDiagnosesResponse;
  task_status_response: TaskStatusResponse;
  tool_call: PredictionToolCall;
  value_range: ValueRange;
  [k: string]: any;
}
/**
 * Display identity of a task whose subprocess is currently registered.
 */
export interface ActiveTaskInfo {
  alias?: string | null;
  config?: TaskConfigSummary | null;
  endpoint?: string | null;
  html_report_url?: string | null;
  kind: string;
  model?: string;
  percent?: number | null;
  report_url?: string | null;
  started_at?: string;
  status?: "preparing" | "running" | "finalizing" | "completed" | "stopped" | "error" | "unknown";
  task_id: string;
}
export interface TaskConfigSummary {
  api?: string | null;
  dataset?: string | null;
  datasets?: string[] | null;
  eval_batch_size?: number | null;
  limit?: number | null;
  max_prompt_length?: number | null;
  max_tokens?: number | number[] | null;
  min_prompt_length?: number | null;
  min_tokens?: number | null;
  number?: number | number[] | null;
  parallel?: number | number[] | null;
  rate?: number | number[] | null;
  repeats?: number | null;
  temperature?: number | null;
  top_k?: number | null;
  top_p?: number | null;
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
  version: string;
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
/**
 * Diagnostic artifact and freshness state.
 */
export interface DiagnosisResponse {
  report?: DiagnosisReport | null;
  status: "ready" | "missing" | "stale" | "error";
}
/**
 * One analysis of a persisted (run, model, benchmark), with cache provenance.
 */
export interface DiagnosisReport {
  analyzer_version: string;
  benchmark: string;
  config_fingerprint: string;
  cost: DiagnosisCost;
  coverage: DiagnosisCoverage;
  data_fingerprint: string;
  diagnosis_model: string | null;
  facts: Fact[];
  generated_at: string;
  level: "facts" | "diagnose";
  limitations: string[];
  model: string;
  narrative: Narrative | null;
  narrative_samples: {
    [k: string]: NarrativeSampleReference;
  };
  narrative_status: string;
  patterns: FailurePattern[];
  run_dir: string;
  samples_sha256: string;
  schema_version: number;
  self_evaluation: boolean;
  taxonomy_version: string;
}
/**
 * Analysis usage only, separate from evaluation and scoring usage.
 */
export interface DiagnosisCost {
  cached_calls: number;
  calls: number;
  estimated_calls: number;
  estimated_input_tokens: number;
  input_tokens: number;
  output_tokens: number;
  usage_missing_calls: number;
}
/**
 * Do not equate unavailable outcomes or absent facts with success or zero events.
 */
export interface DiagnosisCoverage {
  analysis_errors: {
    [k: string]: number;
  };
  citations_total: number;
  citations_valid: number;
  diagnosed_ok: number;
  diagnosis_population: number;
  diagnosis_statuses: {
    [k: string]: number;
  };
  errored_without_record: number;
  excluded: number;
  outcome_unknown: number;
  proposals_without_rule_clues: number;
  risk_samples: number;
  samples: number;
  selected: number;
  selection_scope: ("failed" | "unscored" | "risk" | "process" | "all" | "specified") | null;
  skipped_facts: {
    [k: string]: string;
  };
  task_failed: number;
  task_success_known: number;
  unexplained_failures: number;
  unreadable: number;
}
/**
 * An observation with an explicit denominator and computation definition.
 */
export interface Fact {
  data: {
    [k: string]: any;
  };
  definition: string;
  denominator: number | null;
  id: string;
  kind: "generic" | "generation" | "mcq" | "repeats" | "agent" | "judge" | "perf";
  numerator: number | null;
  samples: SampleKey[];
  severity: "info" | "warn" | "critical";
  title: string;
  unit: string;
  value: number | null;
}
/**
 * A persisted sample identity within one model and benchmark.
 */
export interface SampleKey {
  index: number;
  subset: string;
}
/**
 * Grounded summary and suggestions, without an unreferenced free-form summary field.
 */
export interface Narrative {
  /**
   * @maxItems 3
   */
  findings: [] | [NarrativeItem] | [NarrativeItem, NarrativeItem] | [NarrativeItem, NarrativeItem, NarrativeItem];
  /**
   * @maxItems 5
   */
  limitations:
    | []
    | [string]
    | [string, string]
    | [string, string, string]
    | [string, string, string, string]
    | [string, string, string, string, string];
  /**
   * @maxItems 3
   */
  recommendations:
    [] | [NarrativeItem] | [NarrativeItem, NarrativeItem] | [NarrativeItem, NarrativeItem, NarrativeItem];
}
/**
 * Every sentence has resolvable fact/sample references; it remains model-generated text.
 */
export interface NarrativeItem {
  /**
   * @minItems 1
   * @maxItems 8
   */
  refs:
    | [string]
    | [string, string]
    | [string, string, string]
    | [string, string, string, string]
    | [string, string, string, string, string]
    | [string, string, string, string, string, string]
    | [string, string, string, string, string, string, string]
    | [string, string, string, string, string, string, string, string];
  text: string;
}
/**
 * A summary citation resolved by the analyzer to one inspected sample and optional evidence step.
 */
export interface NarrativeSampleReference {
  sample: SampleKey;
  step: number | null;
}
/**
 * Observed issue types among failed or unscored samples; groups may overlap.
 */
export interface FailurePattern {
  examples: SampleKey[];
  failed: number;
  failure_type: string;
  tools: {
    [k: string]: number;
  };
  unscored: number;
}
/**
 * Options for analyzing one existing dataset report.
 */
export interface DiagnosisRunRequest {
  api_key?: string | null;
  api_url?: string | null;
  include_trace?: boolean;
  level?: "facts" | "diagnose";
  max_diagnosed_samples?: number;
  model_id?: string | null;
  scope?: "failed" | "unscored" | "risk" | "all";
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
export interface ListActiveTasksResponse {
  tasks: ActiveTaskInfo[];
  total: number;
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
  alias?: string | null;
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
export interface ListReportsGroupedResponse {
  filters: ReportFilters;
  page: number;
  page_size: number;
  reports: ReportGroup[];
  total: number;
}
export interface ReportFilters {
  available_datasets: string[];
  available_models: string[];
}
/**
 * One row per model in a `group_by=model` listing - display-only rollup, never a merged report.
 */
export interface ReportGroup {
  children: ReportSummary[];
  dataset_count: number;
  dataset_name: string;
  model_name: string;
  num_samples: number;
  refs: string[];
  report_count: number;
  timestamp: string;
}
export interface ReportSummary {
  alias?: string | null;
  api_host?: string | null;
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
export interface ListReportsResponse {
  filters: ReportFilters;
  page: number;
  page_size: number;
  reports: ReportSummary[];
  total: number;
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
  num_correct?: number | null;
  score: number;
  semantics: MetricSemantics;
}
export interface ReportCategory {
  macro_score?: number;
  name: string[];
  num: number;
  num_correct?: number | null;
  score: number;
  subsets: ReportSubset[];
}
export interface ReportSubset {
  is_aggregate?: boolean;
  metadata?: {
    [k: string]: any;
  } | null;
  name: string;
  num: number;
  num_correct?: number | null;
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
  alias?: string | null;
  api_type: string;
  basic_info: {
    [k: string]: string;
  };
  best_config: {
    [k: string]: string;
  };
  configuration?: {
    [k: string]: string;
  } | null;
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
    [k: string]: number | null;
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
  percentile_rows: (string | number | null)[][];
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
  Gold: string | string[];
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
  framework?: string | null;
  max_steps: number;
  strategy?: string | null;
  total_usage?: TraceUsage | null;
  trial_id?: string | null;
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
 * Mirrors `evalscope.api.model.model_output.ModelUsage`'s shape for the response contract.
 */
export interface TraceUsage {
  input_tokens?: number;
  input_tokens_cache_read?: number | null;
  input_tokens_cache_write?: number | null;
  output_tokens?: number;
  reasoning_tokens?: number | null;
  total_tokens?: number;
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
/**
 * Filtered sample observations, paged after matching.
 */
export interface SampleDiagnosesResponse {
  items: SampleDiagnosis[];
  page: number;
  page_size: number;
  status: "ready" | "missing" | "stale" | "error";
  total: number;
}
/**
 * Official outcome and observed findings; a finding is not a proved root cause.
 */
export interface SampleDiagnosis {
  context_omitted: boolean;
  evidence_errors: string[];
  findings: Finding[];
  invalid_evidence: number;
  key: SampleKey;
  llm: LLMDiagnosis | null;
  llm_status: "not_selected" | "ok" | "parse_error" | "transport_error" | "invalid_evidence" | "unexplained";
  needs_attention: boolean;
  observed_types: string[];
  recorded_error: string | null;
  safety_violation: boolean | null;
  score_status: string;
  sources: DiagnosticSource[];
  stop_reason: string | null;
  task_success: boolean | null;
}
/**
 * One problem detected in one sample.
 */
export interface Finding {
  /**
   * Who is responsible for a finding.
   */
  attribution?: "model" | "guard" | "environment" | "benchmark" | "unknown";
  call_ids?: string[];
  category: FindingCategory;
  confidence: Confidence;
  data?: {
    [k: string]: any;
  };
  detector: string;
  evidence?: Evidence[];
  first_position?: number | null;
  positions?: number[];
  rule_id?: string;
  subtype: string;
  text?: string;
  [k: string]: any;
}
/**
 * A short excerpt supporting a finding.
 */
export interface Evidence {
  excerpt: string;
  message_index?: number | null;
  position?: number | null;
  [k: string]: any;
}
/**
 * A model hypothesis; valid citations do not establish semantic correctness.
 */
export interface LLMDiagnosis {
  attribution: Attribution;
  confidence: "low" | "medium" | "high";
  /**
   * @maxItems 3
   */
  evidence:
    | []
    | [EvidenceReference]
    | [EvidenceReference, EvidenceReference]
    | [EvidenceReference, EvidenceReference, EvidenceReference];
  explanation: string;
  failure_type:
    | "knowledge_error"
    | "reasoning_error"
    | "instruction_violation"
    | "incomplete_answer"
    | "hallucination"
    | "missing_step"
    | "ignored_tool_error"
    | "followed_injection"
    | "unrequested_action"
    | "harmful_compliance"
    | "benchmark_issue"
    | "tool_error"
    | "wrong_tool"
    | "bad_arguments"
    | "environment_failure"
    | "guard_block"
    | "truncated"
    | "empty_output"
    | "extraction_failure"
    | "premature_stop"
    | "loop_or_budget"
    | "attack_goal_action"
    | "safety_violation"
    | "unsafe_command"
    | "refusal"
    | "judge_failure"
    | "unexplained"
    | "other";
  gold_suspect: boolean;
  root_cause_step: number | null;
  /**
   * @maxItems 3
   */
  secondary_types:
    | []
    | [
        | "knowledge_error"
        | "reasoning_error"
        | "instruction_violation"
        | "incomplete_answer"
        | "hallucination"
        | "missing_step"
        | "ignored_tool_error"
        | "followed_injection"
        | "unrequested_action"
        | "harmful_compliance"
        | "benchmark_issue"
        | "tool_error"
        | "wrong_tool"
        | "bad_arguments"
        | "environment_failure"
        | "guard_block"
        | "truncated"
        | "empty_output"
        | "extraction_failure"
        | "premature_stop"
        | "loop_or_budget"
        | "attack_goal_action"
        | "safety_violation"
        | "unsafe_command"
        | "refusal"
        | "judge_failure"
        | "unexplained"
        | "other"
      ]
    | [
        (
          | "knowledge_error"
          | "reasoning_error"
          | "instruction_violation"
          | "incomplete_answer"
          | "hallucination"
          | "missing_step"
          | "ignored_tool_error"
          | "followed_injection"
          | "unrequested_action"
          | "harmful_compliance"
          | "benchmark_issue"
          | "tool_error"
          | "wrong_tool"
          | "bad_arguments"
          | "environment_failure"
          | "guard_block"
          | "truncated"
          | "empty_output"
          | "extraction_failure"
          | "premature_stop"
          | "loop_or_budget"
          | "attack_goal_action"
          | "safety_violation"
          | "unsafe_command"
          | "refusal"
          | "judge_failure"
          | "unexplained"
          | "other"
        ),
        (
          | "knowledge_error"
          | "reasoning_error"
          | "instruction_violation"
          | "incomplete_answer"
          | "hallucination"
          | "missing_step"
          | "ignored_tool_error"
          | "followed_injection"
          | "unrequested_action"
          | "harmful_compliance"
          | "benchmark_issue"
          | "tool_error"
          | "wrong_tool"
          | "bad_arguments"
          | "environment_failure"
          | "guard_block"
          | "truncated"
          | "empty_output"
          | "extraction_failure"
          | "premature_stop"
          | "loop_or_budget"
          | "attack_goal_action"
          | "safety_violation"
          | "unsafe_command"
          | "refusal"
          | "judge_failure"
          | "unexplained"
          | "other"
        )
      ]
    | [
        (
          | "knowledge_error"
          | "reasoning_error"
          | "instruction_violation"
          | "incomplete_answer"
          | "hallucination"
          | "missing_step"
          | "ignored_tool_error"
          | "followed_injection"
          | "unrequested_action"
          | "harmful_compliance"
          | "benchmark_issue"
          | "tool_error"
          | "wrong_tool"
          | "bad_arguments"
          | "environment_failure"
          | "guard_block"
          | "truncated"
          | "empty_output"
          | "extraction_failure"
          | "premature_stop"
          | "loop_or_budget"
          | "attack_goal_action"
          | "safety_violation"
          | "unsafe_command"
          | "refusal"
          | "judge_failure"
          | "unexplained"
          | "other"
        ),
        (
          | "knowledge_error"
          | "reasoning_error"
          | "instruction_violation"
          | "incomplete_answer"
          | "hallucination"
          | "missing_step"
          | "ignored_tool_error"
          | "followed_injection"
          | "unrequested_action"
          | "harmful_compliance"
          | "benchmark_issue"
          | "tool_error"
          | "wrong_tool"
          | "bad_arguments"
          | "environment_failure"
          | "guard_block"
          | "truncated"
          | "empty_output"
          | "extraction_failure"
          | "premature_stop"
          | "loop_or_budget"
          | "attack_goal_action"
          | "safety_violation"
          | "unsafe_command"
          | "refusal"
          | "judge_failure"
          | "unexplained"
          | "other"
        ),
        (
          | "knowledge_error"
          | "reasoning_error"
          | "instruction_violation"
          | "incomplete_answer"
          | "hallucination"
          | "missing_step"
          | "ignored_tool_error"
          | "followed_injection"
          | "unrequested_action"
          | "harmful_compliance"
          | "benchmark_issue"
          | "tool_error"
          | "wrong_tool"
          | "bad_arguments"
          | "environment_failure"
          | "guard_block"
          | "truncated"
          | "empty_output"
          | "extraction_failure"
          | "premature_stop"
          | "loop_or_budget"
          | "attack_goal_action"
          | "safety_violation"
          | "unsafe_command"
          | "refusal"
          | "judge_failure"
          | "unexplained"
          | "other"
        )
      ];
}
/**
 * Offsets are zero-based Python Unicode character offsets, end-exclusive, in the supplied source.
 */
export interface EvidenceReference {
  end: number | null;
  quote: string;
  source_id: string;
  start: number | null;
}
/**
 * Redacted exact text supplied to the model, with an optional original message/step identity.
 */
export interface DiagnosticSource {
  excerpted: boolean;
  id: string;
  message_index: number | null;
  original_start: number;
  step: number | null;
  text: string;
}
export interface TaskStatusResponse {
  status: string;
  task_id: string;
}
