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

/** A single content block inside a chat message.
 *
 * Supported block types (mirrors the backend `Content` union in `content.py`):
 *  - text      : plain/markdown text
 *  - reasoning : model chain-of-thought / thinking block
 *  - image     : image URL or base64 data-URI
 *  - audio     : audio URL or base64 data-URI
 *  - video     : video URL or base64 data-URI
 *  - data      : opaque provider-specific payload (not rendered directly)
 */
export interface ContentBlock {
  type: 'text' | 'reasoning' | 'image' | 'audio' | 'video' | 'data'
  // text / reasoning fields
  text?: string
  reasoning?: string
  /** Number of reasoning tokens reported by the model API. */
  reasoning_tokens?: number
  // multimodal fields (present when type === 'image' | 'audio' | 'video')
  image?: string
  audio?: string
  video?: string
  /** Audio/video format hint, e.g. 'mp3', 'wav', 'mp4'. */
  format?: string
  /** Image detail level hint ('auto' | 'low' | 'high'). */
  detail?: string
  /** Opaque payload for type === 'data'. */
  data?: Record<string, unknown>
}

/** A single tool call invocation emitted by an assistant message.
 *
 * Mirrors the backend ``ToolCall`` model (``api/tool/tool_call.py``),
 * flattened for the wire format – ``function`` is the plain function name
 * and ``arguments`` is the parsed JSON object.
 */
export interface ToolCall {
  id: string
  function: string
  arguments: Record<string, unknown>
}

/** Error payload attached to a tool message on failure. */
export interface ToolMessageError {
  type?: string | null
  message: string
}

/** A single chat message in a conversation (system / user / assistant / tool). */
export interface ChatMessage {
  id?: string
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string | ContentBlock[]
  perf_metrics?: SamplePerfMetrics | null
  // Assistant-only ---------------------------------------------------------
  /** Tool calls emitted by the model in this turn. */
  tool_calls?: ToolCall[] | null
  /** Model identifier that produced this assistant message. */
  model?: string | null
  // Tool-only --------------------------------------------------------------
  /** ID of the originating tool_call this message is a response for. */
  tool_call_id?: string | null
  /** Name of the function this tool observation is for. */
  function?: string | null
  /** Error info, when the tool invocation failed. */
  error?: ToolMessageError | null
}

// ------------------------------------------------------------------ //
// Agent Trace types (mirrors Python api/agent/trace.py)               //
// ------------------------------------------------------------------ //

/** Canonical event kinds emitted by the AgentLoop. */
export type AgentTraceEventType =
  | 'model_generate'
  | 'tool_call'
  | 'tool_result'
  | 'env_exec'
  | 'error'
  | 'nudge'
  | 'submit'
  | 'run_start'
  | 'run_end'

/** Single structured event in an agent trajectory. */
export interface AgentTraceEvent {
  step: number
  timestamp: number
  type: AgentTraceEventType
  message_id?: string | null
  latency_ms?: number | null
  token_usage?: { input?: number; output?: number; total?: number } | null
  payload: Record<string, unknown>
}

/** Complete agent trajectory attached to a sample. */
export interface AgentTrace {
  strategy?: string | null
  environment?: string | null
  max_steps: number
  events: AgentTraceEvent[]
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
  /** Structured message list; present for all new-format caches. */
  Messages?: ChatMessage[] | null
  /** Agent trajectory; only populated for agent-mode runs. */
  AgentTrace?: AgentTrace | null
}

export interface PredictionsResponse {
  predictions: PredictionRow[]
}

export interface AnalysisResponse {
  analysis: string
}

export interface BenchmarkEntry {
  name: string
  pretty_name: string
  tags: string[]
  category: 'llm' | 'vlm'
  subset_list: string[]
  total_samples: number
  few_shot_num: number
  dataset_id: string
  paper_url: string | null
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

export type InvokeStatus = 'ok' | 'error' | 'stopped'

export interface EvalInvokeResponse {
  status: InvokeStatus
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
