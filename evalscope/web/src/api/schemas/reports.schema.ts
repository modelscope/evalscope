/**
 * Runtime (zod) schemas for the "reports" API domain.
 *
 * These schemas are the runtime and TypeScript source of truth for report,
 * prediction, comparison, and analysis endpoint responses. Public type import
 * paths remain stable through the domain barrels while the types use `z.infer`.
 *
 * Covers: LoadReportResponse, ListReportsResponse, PredictionsResponse.
 */
import { z } from 'zod'
import { metricSemanticsSchema } from '@/domain/metric/MetricSemantics'

export { metricSemanticsSchema, valueRangeSchema } from '@/domain/metric/MetricSemantics'

export const metricIdentitySchema = z.object({
  name: z.string().regex(/^[a-z][a-z0-9_]*$/),
  aggregation: z.string().regex(/^[a-z][a-z0-9_]*$/),
  dimensions: z.record(z.string(), z.union([z.string(), z.number(), z.boolean()])),
})

// ------------------------------------------------------------------ //
// Report score tree                                                   //
// ------------------------------------------------------------------ //

/** Runtime contract for a subset score. */
export const subsetDataSchema = z.object({
  name: z.string(),
  score: z.number(),
  num: z.number(),
})

/** Runtime contract for a category score. */
export const categoryDataSchema = z.object({
  name: z.array(z.string()),
  num: z.number(),
  score: z.number(),
  subsets: z.array(subsetDataSchema),
})

/** Runtime contract for a metric score tree. */
export const metricDataSchema = z.object({
  identity: metricIdentitySchema,
  legacy_name: z.string().nullable().optional(),
  num: z.number(),
  score: z.number(),
  categories: z.array(categoryDataSchema),
  semantics: metricSemanticsSchema,
})

// ------------------------------------------------------------------ //
// Performance metrics embedded on a report                            //
// ------------------------------------------------------------------ //

/** Runtime contract for report-level percentile statistics. */
export const percentileStatsSchema = z.object({
  mean: z.number(),
  // pandas uses sample standard deviation (ddof=1), which is undefined for a
  // single observation. The backend serializes that NaN as JSON null, which the
  // API client normalizes to undefined before validation - hence nullish.
  std: z.number().nullish(),
  min: z.number(),
  '25%': z.number(),
  '50%': z.number(),
  '75%': z.number(),
  '90%': z.number(),
  '99%': z.number(),
  max: z.number(),
})

/** Runtime contract for embedded performance summary data. */
export const perfMetricsSummarySchema = z.object({
  n_samples: z.number(),
  latency: percentileStatsSchema,
  throughput: z.object({
    avg_output_tps: z.number(),
    avg_req_ps: z.number(),
  }),
  usage: z.object({
    input_tokens: percentileStatsSchema,
    output_tokens: percentileStatsSchema,
    total_tokens: percentileStatsSchema,
    total_input_tokens: z.number().optional(),
    total_output_tokens: z.number().optional(),
    total_tokens_count: z.number().optional(),
  }),
  ttft: percentileStatsSchema.optional(),
  tpot: percentileStatsSchema.optional(),
})

/** Runtime contract for embedded performance metrics. */
export const perfMetricsSchema = z.object({
  summary: perfMetricsSummarySchema,
  // Field key -> persisted semantics. Report v2 always carries this map.
  metric_semantics: z.record(z.string(), metricSemanticsSchema),
})

/** First-class judge execution summary. Mirrors `JudgeSummary`. */
export const judgeSummarySchema = z.object({
  status: z.string().optional(),
  scored: z.number().optional(),
  total: z.number().optional(),
  coverage: z.number().optional(),
  judge_models: z.array(z.string()).optional(),
  valid_observations: z.number().optional(),
  total_observations: z.number().optional(),
  /** Attempt counts keyed by `ScoreStatus` value. */
  failures: z.record(z.string(), z.number()).optional(),
  disagreement: z.record(z.string(), z.unknown()).optional(),
  error: z.string().optional(),
})

/** Runtime contract for one dataset report. */
export const reportDataSchema = z.object({
  schema_version: z.literal(2),
  name: z.string(),
  dataset_name: z.string(),
  dataset_pretty_name: z.string().optional(),
  model_name: z.string(),
  analysis: z.string(),
  metrics: z.array(metricDataSchema),
  // Reports created without collect_perf persist this field as null rather
  // than omitting it. Both shapes are part of the backend contract.
  perf_metrics: perfMetricsSchema.nullable().optional(),
  primary_metric_identity: metricIdentitySchema.nullable(),
  judge_summary: judgeSummarySchema.optional(),
})

/** Runtime contract for a report detail response. */
export const loadReportResponseSchema = z.object({
  report_list: z.array(reportDataSchema),
  datasets: z.array(z.string()),
  task_config: z.record(z.string(), z.unknown()),
})

// ------------------------------------------------------------------ //
// Report list / summary                                               //
// ------------------------------------------------------------------ //

/** Runtime contract for one dataset's primary metric in a report-list item. */
export const primaryMetricRefSchema = z.object({
  dataset_name: z.string(),
  dataset_pretty_name: z.string().optional(),
  identity: metricIdentitySchema,
  score: z.number(),
  semantics: metricSemanticsSchema,
})

/** Runtime contract for one report-list item. */
export const reportSummarySchema = z.object({
  run_id: z.string(),
  model_id: z.string(),
  model_name: z.string(),
  dataset_name: z.string(),
  dataset_pretty_name: z.string().optional(),
  num_samples: z.number(),
  timestamp: z.string(),
  primary_metrics: z.array(primaryMetricRefSchema),
})

/** Runtime contract for a paginated report list. */
export const listReportsResponseSchema = z.object({
  reports: z.array(reportSummarySchema),
  total: z.number(),
  page: z.number(),
  page_size: z.number(),
  filters: z.object({
    available_models: z.array(z.string()),
    available_datasets: z.array(z.string()),
  }),
})

// ------------------------------------------------------------------ //
// Prediction rows (chat messages + agent trace)                       //
// ------------------------------------------------------------------ //

/** Runtime contract for message/sample performance metadata. */
export const samplePerfMetricsSchema = z.object({
  latency: z.number(),
  ttft: z.number().nullable().optional(),
  tpot: z.number().nullable().optional(),
  input_tokens: z.number(),
  output_tokens: z.number(),
})

/** Runtime contract for a structured message content block. */
export const contentBlockSchema = z.object({
  type: z.enum(['text', 'reasoning', 'image', 'audio', 'video', 'data']),
  text: z.string().optional(),
  reasoning: z.string().optional(),
  reasoning_tokens: z.number().optional(),
  image: z.string().optional(),
  audio: z.string().optional(),
  video: z.string().optional(),
  format: z.string().optional(),
  detail: z.string().optional(),
  data: z.record(z.string(), z.unknown()).optional(),
})

/** Runtime contract for a tool call. */
export const toolCallSchema = z.object({
  id: z.string(),
  function: z.string(),
  arguments: z.record(z.string(), z.unknown()),
})

/** Runtime contract for a tool-result error. */
export const toolMessageErrorSchema = z.object({
  type: z.string().nullable().optional(),
  message: z.string(),
})

/** Runtime contract for a chronological chat message. */
export const chatMessageSchema = z.object({
  id: z.string().optional(),
  role: z.enum(['system', 'user', 'assistant', 'tool']),
  content: z.union([z.string(), z.array(contentBlockSchema)]),
  source: z.enum(['input', 'generate']).nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).nullable().optional(),
  perf_metrics: samplePerfMetricsSchema.nullable().optional(),
  tool_calls: z.array(toolCallSchema).nullable().optional(),
  model: z.string().nullable().optional(),
  tool_call_id: z.union([z.string(), z.array(z.string())]).nullable().optional(),
  function: z.string().nullable().optional(),
  error: toolMessageErrorSchema.nullable().optional(),
})

/** Accepted agent trace event types. */
export const agentTraceEventTypeSchema = z.enum([
  'model_generate',
  'tool_call',
  'tool_result',
  'env_exec',
  'env_reset',
  'error',
  'nudge',
  'submit',
  'run_start',
  'run_end',
])

/** Runtime contract for one chronological agent trace event. */
export const agentTraceEventSchema = z.object({
  step: z.number(),
  timestamp: z.number(),
  type: agentTraceEventTypeSchema,
  message_id: z.string().nullable().optional(),
  latency_ms: z.number().nullable().optional(),
  token_usage: z
    .object({
      input: z.number().optional(),
      output: z.number().optional(),
      total: z.number().optional(),
    })
    .nullable()
    .optional(),
  payload: z.record(z.string(), z.unknown()),
})

/** Runtime contract for an agent trace. */
export const agentTraceSchema = z.object({
  strategy: z.string().nullable().optional(),
  environment: z.string().nullable().optional(),
  max_steps: z.number(),
  events: z.array(agentTraceEventSchema),
})

/** One judge request/response round trip, including the ones that failed. Mirrors `JudgeAttempt`. */
export const judgeAttemptSchema = z.object({
  /** Mirrors `ScoreStatus`: success, transport_error, parse_error, invalid_session, fallback, excluded. */
  status: z.string(),
  case_id: z.string(),
  judge_id: z.string(),
  /** Which planned observation this attempt belongs to. */
  repeat_id: z.number().optional(),
  /** `original` or `swapped` for a pairwise benchmark that judges both orders. */
  placement: z.string().optional(),
  messages: z.array(z.unknown()).optional(),
  model_output: z.unknown().optional(),
  raw_response: z.string().optional(),
  /** The parsed verdict; its shape is the benchmark's own `schema_model`. */
  parsed_value: z.unknown().optional(),
  error: z.string().optional(),
  latency: z.number().optional(),
})

/**
 * Mirrors `Score`. Left open with `loose` because `value` and `metadata` keys are
 * benchmark-defined; only the fields the UI reads are typed.
 */
export const predictionScoreSchema = z.looseObject({
  value: z.record(z.string(), z.unknown()).optional(),
  /** Mirrors `ScoreStatus`; absent in reports produced before it existed. */
  status: z.string().optional(),
  judge_summary: judgeSummarySchema.optional(),
  explanation: z.string().optional(),
  main_score_name: z.string().optional(),
  metadata: z
    .looseObject({
      judge_attempts: z.array(judgeAttemptSchema).optional(),
    })
    .optional(),
})

/** Mirrors `PredictionRow`. */
export const predictionRowSchema = z.object({
  Index: z.string(),
  Input: z.string(),
  Metadata: z.unknown(),
  Generated: z.string(),
  Gold: z.string(),
  Pred: z.string(),
  Score: predictionScoreSchema,
  /** ``null`` when the sample has no usable score; it is excluded from the metric, not 0. */
  NScore: z.number().nullable(),
  /** Mirrors ``ScoreStatus``; absent in reports produced before it existed. */
  Status: z.string().optional(),
  PerfMetrics: samplePerfMetricsSchema.nullable().optional(),
  Messages: z.array(chatMessageSchema).nullable().optional(),
  AgentTrace: agentTraceSchema.nullable().optional(),
})

/** Mirrors `PredictionsResponse`. */
export const predictionsResponseSchema = z.object({
  predictions: z.array(predictionRowSchema),
})

/** Runtime contract for DELETE /api/v1/reports/runs/{run_id}/models/{model_id}. */
export const deleteReportResponseSchema = z.object({
  success: z.boolean(),
  run_id: z.string(),
  model_id: z.string(),
})

export const analysisResponseSchema = z.object({
  analysis: z.string(),
})

// ------------------------------------------------------------------ //
// Inferred types (schema-as-source-of-truth)                          //
// ------------------------------------------------------------------ //

export type PercentileStats = z.infer<typeof percentileStatsSchema>
export type PerfMetrics = z.infer<typeof perfMetricsSchema>
export type ReportData = z.infer<typeof reportDataSchema>
export type LoadReportResponse = z.infer<typeof loadReportResponseSchema>
export type ReportSummary = z.infer<typeof reportSummarySchema>
export type ListReportsResponse = z.infer<typeof listReportsResponseSchema>
export type ContentBlock = z.infer<typeof contentBlockSchema>
export type ToolCall = z.infer<typeof toolCallSchema>
export type ChatMessage = z.infer<typeof chatMessageSchema>
export type AgentTraceEvent = z.infer<typeof agentTraceEventSchema>
export type AgentTrace = z.infer<typeof agentTraceSchema>
export type JudgeAttempt = z.infer<typeof judgeAttemptSchema>
export type JudgeSummary = z.infer<typeof judgeSummarySchema>
export type PredictionScore = z.infer<typeof predictionScoreSchema>
export type PredictionRow = z.infer<typeof predictionRowSchema>
export type PredictionsResponse = z.infer<typeof predictionsResponseSchema>
export type DeleteReportResponse = z.infer<typeof deleteReportResponseSchema>
export type AnalysisResponse = z.infer<typeof analysisResponseSchema>
