import { z } from 'zod'

/** Runtime and TypeScript mirror of the backend `MetricSemantics` contract. */

/** Display tier of a metric and whether it may take part in verdicts. */
export const metricRoleSchema = z.enum(['primary', 'auxiliary', 'diagnostic'])
export type MetricRole = z.infer<typeof metricRoleSchema>

/** Optimization direction of a metric. */
export const metricDirectionSchema = z.enum(['higher_is_better', 'lower_is_better', 'none'])
export type MetricDirection = z.infer<typeof metricDirectionSchema>

/** How a metric value is rendered. */
export const metricDisplayKindSchema = z.enum(['number', 'percent'])
export type MetricDisplayKind = z.infer<typeof metricDisplayKindSchema>

/** Closed value range of a bounded metric. */
export const valueRangeSchema = z.object({ min: z.number(), max: z.number() })
export type ValueRange = z.infer<typeof valueRangeSchema>

/** Single source of truth for validating and typing metric semantics on the frontend. */
export const metricSemanticsSchema = z.object({
  semantic_id: z.string(),
  metric_name: z.string(),
  role: metricRoleSchema,
  direction: metricDirectionSchema,
  raw_unit: z.string().nullable().optional(),
  value_range: valueRangeSchema.nullable().optional(),
  display_kind: metricDisplayKindSchema,
  display_multiplier: z.number().nullable().optional(),
  display_unit: z.string().nullable().optional(),
  display_precision: z.number(),
  contract_version: z.number(),
})

export type MetricSemantics = z.infer<typeof metricSemanticsSchema>
