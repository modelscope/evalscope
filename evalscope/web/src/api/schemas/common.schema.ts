import { z } from 'zod'

export const dataFrameResponseSchema = z.object({
  columns: z.array(z.string()),
  data: z.array(z.record(z.string(), z.unknown())),
})

export const logResponseSchema = z.object({
  text: z.string(),
  head_line: z.number(),
  tail_line: z.number(),
  total_lines: z.number(),
})

export const taskStatusResponseSchema = z.object({
  status: z.string(),
  task_id: z.string(),
})

export const configResponseSchema = z.object({
  outputs_root: z.string(),
})

export type DataFrameResponse = z.infer<typeof dataFrameResponseSchema>
export type LogResponse = z.infer<typeof logResponseSchema>
export type TaskStatusResponse = z.infer<typeof taskStatusResponseSchema>
export type ConfigResponse = z.infer<typeof configResponseSchema>
