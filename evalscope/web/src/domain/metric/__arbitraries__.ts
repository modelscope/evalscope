/**
 * fast-check generators for `MetricSemantics`.
 *
 * Generators constrain the space to *valid* contracts instead of filtering, so shrinking stays
 * meaningful. The invariants mirror the backend validation: a scored metric carries a direction,
 * a diagnostic one does not, and a percent metric always declares a range and a multiplier.
 */

import fc from 'fast-check'
import type { MetricDirection, MetricDisplayKind, MetricRole, MetricSemantics } from './MetricSemantics'

/** A bounded `[0, 1]` ratio rendered as a percentage, e.g. accuracy. */
export function ratioSemantics(): MetricSemantics {
  return {
    semantic_id: 'quality.accuracy.ratio',
    metric_name: 'Accuracy',
    role: 'primary',
    direction: 'higher_is_better',
    value_range: { min: 0, max: 1 },
    display_kind: 'percent',
    display_multiplier: 100,
    display_unit: '%',
    display_precision: 1,
    comparison_group: 'quality.accuracy',
    contract_version: 1,
  }
}

/** An unbounded latency in seconds, where lower is better. */
export function secondsSemantics(): MetricSemantics {
  return {
    semantic_id: 'perf.latency.seconds',
    metric_name: 'Latency',
    role: 'primary',
    direction: 'lower_is_better',
    raw_unit: 's',
    value_range: null,
    display_kind: 'number',
    display_unit: 's',
    display_precision: 3,
    comparison_group: 'perf.latency',
    contract_version: 1,
  }
}

/** A diagnostic count that carries no direction and no comparison group. */
export function diagnosticSemantics(): MetricSemantics {
  return {
    semantic_id: 'diagnostic.count.items',
    metric_name: 'Count',
    role: 'diagnostic',
    direction: 'none',
    value_range: null,
    display_kind: 'number',
    display_unit: null,
    display_precision: 0,
    comparison_group: null,
    contract_version: 1,
  }
}

/** Generate a valid `MetricSemantics` across the whole legal space. */
export function arbSemantics(): fc.Arbitrary<MetricSemantics> {
  const scoredRole: fc.Arbitrary<MetricRole> = fc.constantFrom('primary', 'auxiliary')
  const scoredDirection: fc.Arbitrary<MetricDirection> = fc.constantFrom('higher_is_better', 'lower_is_better')

  const scored = fc.record({
    role: scoredRole,
    direction: scoredDirection,
    displayKind: fc.constantFrom<MetricDisplayKind>('number', 'percent'),
    bounded: fc.boolean(),
    precision: fc.integer({ min: 0, max: 6 }),
    unit: fc.option(fc.constantFrom('%', 's', 'ms', 'tok/s'), { nil: null }),
  }).map(({ role, direction, displayKind, bounded, precision, unit }): MetricSemantics => {
    // A percent metric must declare both a range and a multiplier.
    const isPercent = displayKind === 'percent'
    const hasRange = isPercent || bounded
    return {
      semantic_id: 'quality.generated.ratio',
      metric_name: 'Generated',
      role,
      direction,
      raw_unit: unit,
      value_range: hasRange ? { min: 0, max: 1 } : null,
      display_kind: displayKind,
      display_multiplier: isPercent ? 100 : null,
      display_unit: isPercent ? '%' : unit,
      display_precision: precision,
      comparison_group: 'quality.generated',
      contract_version: 1,
    }
  })

  const diagnostic = fc.integer({ min: 0, max: 6 }).map((precision): MetricSemantics => ({
    ...diagnosticSemantics(),
    display_precision: precision,
  }))

  return fc.oneof(scored, diagnostic)
}
