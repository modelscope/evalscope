/** Small reusable metric-semantics fixtures. */

import type { MetricSemantics } from './MetricSemantics'

/** A bounded `[0, 1]` ratio rendered as a percentage, e.g. accuracy. */
export function ratioSemantics(): MetricSemantics {
  return {
    semantic_id: 'quality.accuracy.ratio',
    metric_name: 'Accuracy',
    kind: 'quality',
    direction: 'higher_is_better',
    value_range: { min: 0, max: 1 },
    display_kind: 'percent',
    display_multiplier: 100,
    display_unit: '%',
    display_precision: 1,
  }
}

/** An unbounded latency in seconds, where lower is better. */
export function secondsSemantics(): MetricSemantics {
  return {
    semantic_id: 'perf.latency.seconds',
    metric_name: 'Latency',
    kind: 'quality',
    direction: 'lower_is_better',
    raw_unit: 's',
    value_range: null,
    display_kind: 'number',
    display_unit: 's',
    display_precision: 3,
  }
}

/** A diagnostic count that carries no direction and no comparison group. */
export function diagnosticSemantics(): MetricSemantics {
  return {
    semantic_id: 'diagnostic.count.items',
    metric_name: 'Count',
    kind: 'diagnostic',
    direction: 'none',
    value_range: null,
    display_kind: 'number',
    display_unit: null,
    display_precision: 0,
  }
}
