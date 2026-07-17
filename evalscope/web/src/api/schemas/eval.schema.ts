/**
 * Runtime (zod) schemas for the "eval" API domain.
 *
 * These schemas are the runtime and TypeScript source of truth for eval-invoke,
 * progress, and benchmark endpoint responses. Public type import paths remain
 * stable through the domain barrels while the types themselves use `z.infer`.
 *
 * Covers: EvalInvokeResponse, ProgressResponse.
 */
import { z } from 'zod'

/** Accepted invoke status values. */
export const invokeStatusSchema = z.enum(['ok', 'error', 'stopped'])

/** Runtime contract for an evaluation invoke response. */
export const evalInvokeResponseSchema = z.object({
  status: invokeStatusSchema,
  task_id: z.string(),
  result: z.unknown().optional(),
  table: z.string().optional(),
  error: z.string().optional(),
})

/**
 * Runtime contract for progress responses. A catchall preserves backend
 * progress metadata in addition to the stable percent/current-step fields.
 */
export const progressResponseSchema = z
  .object({
    percent: z.number(),
    current_step: z.string().optional(),
  })
  .catchall(z.unknown())

const benchmarkDescriptionLocaleSchema = z.object({
  full: z.string(),
  sections: z.record(z.string(), z.string()),
})

export const benchmarkEntrySchema = z.object({
  name: z.string(),
  pretty_name: z.string(),
  tags: z.array(z.string()),
  category: z.enum(['llm', 'vlm', 'agent', 'aigc']),
  subset_list: z.array(z.string()),
  total_samples: z.number(),
  few_shot_num: z.number(),
  dataset_id: z.string(),
  paper_url: z.string().nullable(),
  metrics: z.array(z.string()),
  meta: z.record(z.string(), z.unknown()),
  description: z.object({
    en: benchmarkDescriptionLocaleSchema.optional(),
    zh: benchmarkDescriptionLocaleSchema.optional(),
  }),
})

export const benchmarksResponseSchema = z.object({
  text: z.array(benchmarkEntrySchema).optional(),
  multimodal: z.array(benchmarkEntrySchema).optional(),
  agent: z.array(benchmarkEntrySchema).optional(),
  aigc: z.array(benchmarkEntrySchema).optional(),
})

// ------------------------------------------------------------------ //
// Inferred types (schema-as-source-of-truth)                          //
// ------------------------------------------------------------------ //

export type EvalInvokeResponse = z.infer<typeof evalInvokeResponseSchema>
export type ProgressResponse = z.infer<typeof progressResponseSchema>
export type BenchmarkEntry = z.infer<typeof benchmarkEntrySchema>
export type BenchmarksResponse = z.infer<typeof benchmarksResponseSchema>
