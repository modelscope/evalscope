/**
 * Runtime (zod) schemas for the "performance" API domain.
 *
 * These schemas are the runtime and TypeScript source of truth for all
 * performance archive endpoint responses. Public type import paths remain
 * stable through the domain barrels while the types use `z.infer`.
 *
 * Covers: PerfDetailResponse, ListPerfRunsResponse, PerfRunsListResponse.
 */
import { z } from 'zod'

/** A row/table cell value used by perf summary and percentile tables. */
const tableCellSchema = z.union([z.string(), z.number()])

// ------------------------------------------------------------------ //
// Perf run archive (GET /api/v1/perf/list)                            //
// ------------------------------------------------------------------ //

/** Runtime contract for an archived performance-run summary. */
export const perfRunSummarySchema = z.object({
  path: z.string(),
  model: z.string(),
  api_type: z.string(),
  dataset: z.string(),
  num_runs: z.number(),
  total_requests: z.number(),
  success_rate: z.number(),
  best_rps: z.number(),
  best_latency: z.number(),
  is_embedding: z.boolean(),
  has_html: z.boolean(),
  timestamp: z.string(),
  provider: z.string().optional(),
  protocol: z.string().optional(),
  api_host: z.string().optional(),
  concurrency: z.array(z.number()).optional(),
})

/** Runtime contract for the performance archive list. */
export const listPerfRunsResponseSchema = z.object({
  runs: z.array(perfRunSummarySchema),
  total: z.number(),
})

// ------------------------------------------------------------------ //
// Perf run native-render detail (GET /api/v1/perf/detail)             //
// ------------------------------------------------------------------ //

/** Runtime contract for native-render performance details. */
export const perfDetailResponseSchema = z.object({
  path: z.string(),
  model: z.string(),
  api_type: z.string(),
  dataset: z.string(),
  generated_at: z.string(),
  basic_info: z.record(z.string(), z.string()),
  summary_columns: z.array(z.string()),
  summary_rows: z.array(z.array(tableCellSchema)),
  best_config: z.record(z.string(), z.string()),
  recommendations: z.array(z.string()),
  num_runs: z.number(),
  is_embedding: z.boolean(),
  has_html: z.boolean(),
})

// ------------------------------------------------------------------ //
// Individual runs within a perf-run directory                         //
// ------------------------------------------------------------------ //

/** Runtime contract for one workload configuration inside an archive. */
export const perfRunItemSchema = z.object({
  dir_name: z.string(),
  name: z.string(),
  parallel: z.number(),
  number: z.number(),
  rate: z.number().nullable(),
  total_requests: z.number(),
  succeed_requests: z.number(),
  success_rate: z.number(),
  num_requests: z.number(),
  has_requests: z.boolean(),
  percentile_columns: z.array(z.string()),
  percentile_rows: z.array(z.array(tableCellSchema)),
})

/** Runtime contract for the workload-configuration list. */
export const perfRunsListResponseSchema = z.object({
  runs: z.array(perfRunItemSchema),
  total: z.number(),
})

export const perfRequestsResponseSchema = z.object({
  columns: z.array(z.string()),
  rows: z.array(z.record(z.string(), z.unknown())),
  total: z.number(),
  page: z.number(),
  page_size: z.number(),
  has_db: z.boolean(),
})

// ------------------------------------------------------------------ //
// Inferred types (schema-as-source-of-truth)                          //
// ------------------------------------------------------------------ //

export type PerfRunSummary = z.infer<typeof perfRunSummarySchema>
export type ListPerfRunsResponse = z.infer<typeof listPerfRunsResponseSchema>
export type PerfDetailResponse = z.infer<typeof perfDetailResponseSchema>
export type PerfRunItem = z.infer<typeof perfRunItemSchema>
export type PerfRunsListResponse = z.infer<typeof perfRunsListResponseSchema>
export type PerfRequestsResponse = z.infer<typeof perfRequestsResponseSchema>
