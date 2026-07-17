import { apiValidated } from './client'
import { benchmarksResponseSchema } from './schemas'
import type { BenchmarksResponse } from './types'
import { createTaskApi } from './task'

const evalTaskApi = createTaskApi('eval')

export const submitEvalTask = evalTaskApi.submit
export const getEvalProgress = evalTaskApi.progress
export const getEvalLog = evalTaskApi.log
export const getEvalReportUrl = evalTaskApi.reportUrl
export const stopEvalTask = evalTaskApi.stop

export async function listBenchmarks(
  type?: 'text' | 'multimodal',
  all?: boolean,
  signal?: AbortSignal,
): Promise<BenchmarksResponse> {
  const params: Record<string, string> = {}
  if (type) params.type = type
  if (all) params.all = 'true'
  return apiValidated('/api/v1/eval/benchmarks', benchmarksResponseSchema, { params, signal })
}
