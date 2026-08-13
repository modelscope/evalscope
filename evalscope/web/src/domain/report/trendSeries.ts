import type { ReportSummary } from '@/api/types'
import type { MetricIdentity, MetricSemantics } from '@/domain/metric'
import { metricIdentityKey } from '@/domain/metric'

export interface TrendPoint {
  runId: string
  modelId: string
  timestamp: string
  score: number
}

export interface TrendSeries {
  modelId: string
  modelLabel: string
  benchmark: string
  benchmarkLabel: string
  identity: MetricIdentity
  semantics: MetricSemantics
  points: TrendPoint[]
}

/** Build comparable histories without joining different models, benchmarks, or metric identities. */
export function buildTrendSeries(reports: ReportSummary[]): TrendSeries[] {
  const grouped = new Map<string, TrendSeries>()

  for (const report of reports) {
    for (const metric of report.primary_metrics) {
      if (!Number.isFinite(metric.score)) continue
      const key = [
        report.model_id,
        metric.dataset_name,
        metricIdentityKey(metric.identity),
        metric.semantics.semantic_id,
      ].join('\0')
      const existing = grouped.get(key)
      const point = {
        runId: report.run_id,
        modelId: report.model_id,
        timestamp: report.timestamp,
        score: metric.score,
      }
      if (existing) {
        existing.points.push(point)
        continue
      }
      grouped.set(key, {
        modelId: report.model_id,
        modelLabel: report.model_name || report.model_id,
        benchmark: metric.dataset_name,
        benchmarkLabel: metric.dataset_pretty_name?.trim() || metric.dataset_name,
        identity: metric.identity,
        semantics: metric.semantics,
        points: [point],
      })
    }
  }

  return [...grouped.values()]
    .map((series) => ({
      ...series,
      points: series.points.sort((left, right) => left.timestamp.localeCompare(right.timestamp)),
    }))
    .filter((series) => series.points.length >= 2)
    .sort((left, right) => latestTimestamp(right).localeCompare(latestTimestamp(left)))
}

export function latestTimestamp(series: TrendSeries): string {
  return series.points[series.points.length - 1]?.timestamp ?? ''
}
