/**
 * Report-summary helpers built on the backend's primary-metric contract.
 *
 * The report list, the cards and the page header all need the same three answers: which metric
 * represents a run, how to render it, and whether a set of runs may be compared or sorted at all.
 * Deriving that once here keeps those surfaces consistent and keeps metric-name guessing out of
 * the components.
 */

import { formatMetricIdentityLabel, metricIdentityKey } from '@/domain/metric'
import type { MetricIdentity, MetricSemantics } from '@/domain/metric'
import type { ReportData } from '@/api/types'

/** One metric of a dataset report, as returned by the API. */
export type ReportMetric = ReportData['metrics'][number]

/**
 * The metric that represents one dataset report, selected by role rather than by position.
 *
 * ``primary_metric_identity`` names it explicitly. The API migrates old reports before they
 * reach the frontend, so this function never guesses from metric order or role.
 *
 * Every surface that needs "the score of this run" must go through here, so the report list, the
 * detail page, the overview tab and the compare page can never disagree about which metric that is.
 */
export function primaryMetricOf(report: ReportData): ReportMetric | null {
  if (!report.primary_metric_identity) return null
  return report.metrics.find(
    (metric) => metricIdentityKey(metric.identity) === metricIdentityKey(report.primary_metric_identity!),
  ) ?? null
}

/** One dataset's primary metric as reported by the API. */
export interface PrimaryMetricRef {
  dataset_name: string
  dataset_pretty_name?: string
  identity: MetricIdentity
  score: number
  semantics: MetricSemantics
}

/** Human-readable dataset label, with the stable registry name as a lossless fallback. */
export function datasetLabel(dataset: { dataset_name: string; dataset_pretty_name?: string | null }): string {
  return dataset.dataset_pretty_name?.trim() || dataset.dataset_name
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
  kind: 'quality',
  direction: 'higher_is_better',
  value_range: { min: 0, max: 1 },
  display_kind: 'percent',
  display_multiplier: 100,
  display_unit: '%',
  display_precision: 1,
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
  return formatMetricIdentityLabel(ref.identity, ref.semantics)
}
