/**
 * Report-summary helpers built on the backend's primary-metric contract.
 *
 * The report list, the cards and the page header all need the same three answers: which metric
 * represents a run, how to render it, and whether a set of runs may be compared or sorted at all.
 * Deriving that once here keeps those surfaces consistent and keeps metric-name guessing out of
 * the components.
 */

import type { MetricSemantics } from '@/domain/metric'
import type { ReportSummary } from '@/api/types'

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
  if (semantics.direction === 'higher_is_better') return 'metrics.higherIsBetter'
  if (semantics.direction === 'lower_is_better') return 'metrics.lowerIsBetter'
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

/**
 * The metric label shared by every ref, or `null` when they differ.
 *
 * When a whole table measures one metric — by far the common case — the label belongs in the
 * column header rather than repeated down every row, and the Metric column can disappear
 * entirely. Returns `null` for an empty input, since there is nothing to hoist.
 */
export function uniformMetricLabel(refGroups: PrimaryMetricRef[][]): string | null {
  const labels = new Set<string>()
  for (const refs of refGroups) {
    for (const ref of refs) {
      labels.add(metricLabel(ref))
    }
  }
  return labels.size === 1 ? [...labels][0] : null
}
