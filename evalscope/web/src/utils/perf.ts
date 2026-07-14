// Shared perf-report chart groupings + helpers, reused across the perf archive
// pages (detail / compare / list) so the chart ids, titles and timestamp
// formatting stay in a single place.

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

/** Format an ISO timestamp as ``YYYY-MM-DD HH:MM:SS`` (empty string when absent). */
export function formatFull(ts: string): string {
  return ts ? ts.replace('T', ' ').slice(0, 19) : ''
}
