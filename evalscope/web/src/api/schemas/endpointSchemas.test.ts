import { describe, expect, it } from 'vitest'
import type { ZodType } from 'zod'

import {
  analysisResponseSchema,
  benchmarksResponseSchema,
  configResponseSchema,
  dataFrameResponseSchema,
  evalInvokeResponseSchema,
  listPerfRunsResponseSchema,
  listReportsResponseSchema,
  loadReportResponseSchema,
  logResponseSchema,
  perfDetailResponseSchema,
  perfRequestsResponseSchema,
  perfRunsListResponseSchema,
  predictionsResponseSchema,
  progressResponseSchema,
  scanResponseSchema,
  taskStatusResponseSchema,
} from './index'

const ENDPOINT_CASES: Array<{ name: string; schema: ZodType; valid: unknown }> = [
  { name: 'config', schema: configResponseSchema, valid: { outputs_root: './outputs' } },
  {
    name: 'dataframe',
    schema: dataFrameResponseSchema,
    valid: { columns: ['score'], data: [{ score: 1 }] },
  },
  { name: 'log', schema: logResponseSchema, valid: { text: '', head_line: 0, tail_line: 0, total_lines: 0 } },
  { name: 'task status', schema: taskStatusResponseSchema, valid: { status: 'running', task_id: 'task-1' } },
  { name: 'eval invoke', schema: evalInvokeResponseSchema, valid: { status: 'completed', task_id: 'task-1' } },
  { name: 'progress', schema: progressResponseSchema, valid: { percent: 50, phase: 'evaluate' } },
  { name: 'benchmarks', schema: benchmarksResponseSchema, valid: { text: [] } },
  { name: 'report scan', schema: scanResponseSchema, valid: { reports: ['run-a'] } },
  { name: 'analysis', schema: analysisResponseSchema, valid: { analysis: 'complete' } },
  {
    name: 'report list',
    schema: listReportsResponseSchema,
    valid: {
      reports: [],
      total: 0,
      page: 1,
      page_size: 20,
      filters: { available_models: [], available_datasets: [] },
    },
  },
  {
    name: 'report detail',
    schema: loadReportResponseSchema,
    valid: {
      report_list: [
        {
          name: 'run-a',
          dataset_name: 'gsm8k',
          model_name: 'model-a',
          score: 0.5,
          analysis: '',
          metrics: [],
          perf_metrics: null,
        },
      ],
      datasets: ['gsm8k'],
      task_config: {},
    },
  },
  {
    name: 'predictions',
    schema: predictionsResponseSchema,
    valid: {
      predictions: [
        {
          Index: '0',
          Input: 'question',
          Metadata: {},
          Generated: 'answer',
          Gold: 'answer',
          Pred: 'answer',
          Score: { acc: 1 },
          NScore: 1,
        },
      ],
    },
  },
  {
    name: 'performance list',
    schema: listPerfRunsResponseSchema,
    valid: { runs: [], total: 0 },
  },
  {
    name: 'performance detail',
    schema: perfDetailResponseSchema,
    valid: {
      path: 'run-a',
      model: 'model-a',
      api_type: 'openai',
      dataset: 'openqa',
      generated_at: '2026-07-01T00:00:00Z',
      basic_info: {},
      summary_columns: [],
      summary_rows: [],
      best_config: {},
      recommendations: [],
      num_runs: 0,
      is_embedding: false,
      has_html: false,
    },
  },
  { name: 'performance runs', schema: perfRunsListResponseSchema, valid: { runs: [], total: 0 } },
  {
    name: 'performance requests',
    schema: perfRequestsResponseSchema,
    valid: { columns: [], rows: [], total: 0, page: 1, page_size: 20, has_db: false },
  },
]

describe('endpoint response schemas', () => {
  it.each(ENDPOINT_CASES)('accepts a valid $name response', ({ schema, valid }) => {
    expect(schema.safeParse(valid).success).toBe(true)
  })

  it.each(ENDPOINT_CASES)('rejects a non-object $name response', ({ schema }) => {
    expect(schema.safeParse(null).success).toBe(false)
  })
})

describe('invoke response status schema', () => {
  it.each(['ok', 'completed', 'error', 'stopped'])('accepts the supported status %s', (status) => {
    expect(evalInvokeResponseSchema.safeParse({ status, task_id: 'task-1' }).success).toBe(true)
  })

  it('rejects an unknown status', () => {
    expect(evalInvokeResponseSchema.safeParse({ status: 'unknown', task_id: 'task-1' }).success).toBe(false)
  })
})
