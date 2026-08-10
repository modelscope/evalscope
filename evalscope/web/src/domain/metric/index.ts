/**
 * Public surface of the metric domain: five pure functions plus the mirrored contract types.
 *
 * Nothing here resolves a metric by name. Semantics come from the backend, which is what keeps
 * the direction, unit and precision of a metric defined in exactly one place.
 *
 * `MISSING_PLACEHOLDER` and `roundHalfUp` stay unexported: they are internals of `formatMetric`
 * with no consumer outside this folder, and tests import them from `./metricFormat` directly.
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
  formatDifference,
  getValuePosition,
} from './metricFormat'
