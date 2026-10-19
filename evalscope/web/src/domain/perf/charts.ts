// Shared perf-report chart groupings, reused across the perf archive pages
// (detail / compare / list) so the chart ids and titles stay in a single place.

import type { PerfDetailResponse } from '@/api/types'

export type PerfRunMode = 'generation' | 'embedding' | 'decision'

/** Keep timing availability separate from the model/provider identity. */
export function perfRunMode(run?: Pick<PerfDetailResponse, 'api_type' | 'is_embedding'> | null): PerfRunMode {
  if (run?.is_embedding) return 'embedding'
  if (['systemone', 'autotrust'].includes(run?.api_type ?? '')) return 'decision'
  return 'generation'
}

/** Latency-group sweep chart ids for LLM runs. Embedding runs drop TTFT/TPOT. */
export const LATENCY_CHARTS = ['latency', 'ttft', 'tpot'] as const

/** Throughput-group sweep chart ids (apply to every run mode). */
export const THROUGHPUT_CHARTS = ['rps', 'throughput', 'success'] as const

/** Display titles keyed by sweep chart id. */
export const CHART_TITLES: Record<string, string> = {
  latency: 'Latency (s)',
  ttft: 'TTFT (ms)',
  tpot: 'TPOT (ms)',
  rps: 'Requests/sec',
  throughput: 'Tokens/sec',
  success: 'Success Rate (%)',
}
