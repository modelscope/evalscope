/**
 * Public surface of the metric domain: four pure functions plus the mirrored contract types.
 *
 * Nothing here resolves a metric by name. Semantics come from the backend, which is what keeps
 * the direction, unit and precision of a metric defined in exactly one place.
 */

export type {
  MetricDirection,
  MetricDisplayKind,
  MetricRole,
  MetricSemantics,
  ValueRange,
} from './MetricSemantics'

export type { ComparisonVerdict, FormattedMetric } from './metricFormat'
export {
  formatMetric,
  getBoundedQualityRatio,
  getComparisonVerdict,
  getValuePosition,
  MISSING_PLACEHOLDER,
  roundHalfUp,
} from './metricFormat'
