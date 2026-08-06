/**
 * Report-summary helpers built on the backend's primary-metric contract.
 *
 * The report list, the cards and the page header all need the same three answers: which metric
 * represents a run, how to render it, and whether a set of runs may be compared or sorted at all.
 * Deriving that once here keeps those surfaces consistent and keeps metric-name guessing out of
 * the components.
 */

import { formatMetric, getBoundedQualityRatio } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'
import type { ReportSummary } from '@/api/types'

/** One dataset's primary metric as reported by the API. */
export interface PrimaryMetricRef {
  dataset_name: string
  metric_name: string
  score: number | null
  semantics?: MetricSemantics | null
}

/**
 * Semantics of a plain `[0, 1]` ratio rendered as a percentage.
 *
 * Used by the aggregate comparison matrix, whose cells are normalized scores computed by the
 * backend DataFrame rather than individual metrics, so they carry no per-cell semantics. This is
 * an explicit statement about those cells, not a lookup by metric name.
 */
export const RATIO_PERCENT_SEMANTICS: MetricSemantics = {
  semantic_id: 'quality.score.ratio',
  metric_name: 'Score',
  role: 'auxiliary',
  direction: 'higher_is_better',
  value_range: { min: 0, max: 1 },
  display_kind: 'percent',
  display_multiplier: 100,
  display_unit: '%',
  display_precision: 1,
  contract_version: 1,
}

/** Arrow indicating the optimization direction, empty when the metric carries none. */
export function directionArrow(semantics: MetricSemantics | null | undefined): string {
  if (!semantics) return ''
  if (semantics.direction === 'higher_is_better') return '↑'
  if (semantics.direction === 'lower_is_better') return '↓'
  return ''
}

/**
 * i18n key describing the direction, for a tooltip and an `aria-label`.
 *
 * Returns `null` when the metric has no direction, so callers omit the hint entirely rather than
 * claiming a neutral metric is better in some direction.
 */
export function directionHintKey(semantics: MetricSemantics | null | undefined): string | null {
  if (!semantics) return null
  if (semantics.direction === 'higher_is_better') return 'metric.higherIsBetter'
  if (semantics.direction === 'lower_is_better') return 'metric.lowerIsBetter'
  return null
}

/** Label of a metric: its display name plus the direction arrow, e.g. `Accuracy ↑`. */
export function metricLabel(ref: PrimaryMetricRef | null | undefined): string {
  if (!ref) return ''
  const name = ref.semantics?.metric_name || ref.metric_name
  const arrow = directionArrow(ref.semantics)
  return arrow ? `${name} ${arrow}` : name
}

/** Primary metrics of a run, falling back to an empty list on an older backend response. */
export function primaryMetricsOf(report: ReportSummary): PrimaryMetricRef[] {
  return (report as { primary_metrics?: PrimaryMetricRef[] }).primary_metrics ?? []
}

/** Summary status of a run; `undefined` on an older backend response. */
export function summaryStatusOf(report: ReportSummary): string | undefined {
  return (report as { summary_status?: string }).summary_status
}

/**
 * Whether a single number may represent this run.
 *
 * Only a run with exactly one semantic primary metric has one. Several datasets are never
 * averaged, even when they share a metric: `no_aggregate` and `mixed_metrics` both mean
 * "show the metrics individually".
 */
export function hasSingleScore(report: ReportSummary): boolean {
  const status = summaryStatusOf(report)
  return status === undefined ? true : status === 'single_metric'
}

/** Display text of a run's score, or `null` when the run has no single representative score. */
export function summaryScoreDisplay(report: ReportSummary): string | null {
  if (!hasSingleScore(report)) {
    return null
  }
  const refs = primaryMetricsOf(report)
  if (refs.length === 1) {
    return formatMetric(refs[0].score, refs[0].semantics).primary
  }
  // Older backend: fall back to the legacy score, which was already a single number.
  return formatMetric(report.score, null).primary
}

/** Colour-scale ratio of a run's score, or `null` when a scale would be meaningless. */
export function summaryScoreRatio(report: ReportSummary): number | null {
  const refs = primaryMetricsOf(report)
  if (!hasSingleScore(report) || refs.length !== 1) {
    return null
  }
  return getBoundedQualityRatio(refs[0].score, refs[0].semantics)
}

/** Number of distinct semantic identifiers among a run's primary metrics. */
export function distinctSemanticCount(refs: PrimaryMetricRef[]): number {
  return new Set(refs.map((ref) => ref.semantics?.semantic_id ?? ref.metric_name)).size
}

/**
 * Whether a set of runs may be sorted or compared by score.
 *
 * Allowed only when every run exposes exactly one primary metric and all of them share a
 * `semantic_id`: comparing an accuracy against a WER, or against a judge score on an unknown
 * scale, would rank incomparable numbers.
 */
export function isScoreComparable(reports: ReportSummary[]): boolean {
  const ids = new Set<string>()
  for (const report of reports) {
    const refs = primaryMetricsOf(report)
    if (refs.length === 0) {
      // Older backend response: no semantics to disagree about.
      continue
    }
    if (refs.length > 1) {
      return false
    }
    ids.add(refs[0].semantics?.semantic_id ?? refs[0].metric_name)
  }
  return ids.size <= 1
}
