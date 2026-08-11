/**
 * Report-summary helpers built on the backend's primary-metric contract.
 *
 * The report list, the cards and the page header all need the same three answers: which metric
 * represents a run, how to render it, and whether a set of runs may be compared or sorted at all.
 * Deriving that once here keeps those surfaces consistent and keeps metric-name guessing out of
 * the components.
 */

import { formatMetricLabel } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'
import type { ReportData, ReportSummary } from '@/api/types'

/** One metric of a dataset report, as returned by the API. */
export type ReportMetric = ReportData['metrics'][number]

/**
 * The metric that represents one dataset report, selected by role rather than by position.
 *
 * `primary_metric_name` names it explicitly; otherwise the metric whose semantics say
 * `role === 'primary'` is used. A report with neither yields `null`, which the UI shows as an
 * absent score instead of borrowing another metric's number.
 *
 * Every surface that needs "the score of this run" must go through here, so the report list, the
 * detail page, the overview tab and the compare page can never disagree about which metric that is.
 */
export function primaryMetricOf(report: ReportData): ReportMetric | null {
  const named = report.primary_metric_name
    ? report.metrics.find((metric) => metric.name === report.primary_metric_name)
    : undefined
  return named ?? report.metrics.find((metric) => metric.semantics?.role === 'primary') ?? null
}

/** One dataset's primary metric as reported by the API. */
export interface PrimaryMetricRef {
  dataset_name: string
  metric_name: string
  score: number | null
  semantics?: MetricSemantics | null
  /** Whether the benchmark declared this metric as primary, or one was inferred to show a value. */
  inferred?: boolean
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

/**
 * i18n key describing the direction, for a tooltip and an `aria-label`.
 *
 * Returns `null` when the metric has no direction, so callers omit the hint entirely rather than
 * claiming a neutral metric is better in some direction.
 */
export function directionHintKey(semantics: MetricSemantics | null | undefined): string | null {
  if (!semantics) return null
  if (semantics.direction === 'higher_is_better') return 'metrics.higherIsBetter'
  if (semantics.direction === 'lower_is_better') return 'metrics.lowerIsBetter'
  return null
}

/** Label of a metric: its display name plus the direction arrow, e.g. `Accuracy ↑`. */
export function metricLabel(ref: PrimaryMetricRef | null | undefined): string {
  if (!ref) return ''
  return formatMetricLabel(ref.metric_name, ref.semantics)
}

/** Primary metrics of a run, falling back to an empty list on an older backend response. */
export function primaryMetricsOf(report: ReportSummary): PrimaryMetricRef[] {
  return (report as { primary_metrics?: PrimaryMetricRef[] }).primary_metrics ?? []
}
