/**
 * Metric formatting primitives.
 *
 * These are the only metric-related decisions the frontend makes, and each is a pure function of
 * the value plus the backend-provided `MetricSemantics`. There is no metric-name table, no alias
 * matching and no inference from a value's magnitude: the direction, unit, scale and precision of
 * a metric come from the backend contract, so every surface renders the same value identically
 * and a catalog fix reaches the UI without a frontend change.
 */

import type { MetricDirection, MetricDisplayKind, MetricSemantics } from './MetricSemantics'

export interface MetricIdentity {
  name: string
  aggregation: string
  dimensions: Record<string, string | number | boolean>
}

export function metricIdentityKey(identity: MetricIdentity): string {
  const dimensions = Object.entries(identity.dimensions)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(',')
  return `${identity.name}:${identity.aggregation}${dimensions ? `[${dimensions}]` : ''}`
}

/**
 * Placeholder shown for a missing metric value. Intentionally distinct from a legitimate `0` or
 * an empty string so a missing value can never be mistaken for a real zero.
 */
export const MISSING_PLACEHOLDER = '—'

/** Decimals used when the backend provided no semantics at all. */
const DIAGNOSTIC_FALLBACK_PRECISION = 4

/**
 * Decimals of the raw text.
 *
 * The raw text never applies `display_multiplier`: it shows the stored value with its `raw_unit`,
 * so a tooltip or an export exposes what was actually recorded. It is a frontend-only concern --
 * no backend surface renders an unscaled value -- which is why this constant has no backend
 * counterpart and the golden samples pin it through `expected_raw`.
 */
const RAW_VALUE_PRECISION = 4

/** Display-ready representation of a single metric value. */
export interface FormattedMetric {
  /** Primary display text, e.g. `"85.7%"`, `"1.235 s"` or the missing placeholder. */
  primary: string
  /** Unrounded value for tooltips, detail views and copy actions. */
  raw: string
  /** Unit appended to the displayed value; empty when the metric has none. */
  unitLabel: string
  /** `true` when the value is `null`, `undefined`, `NaN` or otherwise non-finite. */
  isMissing: boolean
  /**
   * `true` when no semantics were provided, or the metric is diagnostic. The UI uses this to
   * suppress colour scales, progress bars and verdicts.
   */
  isDiagnosticFallback: boolean
}

/** Arrow indicating the optimization direction, empty when the metric carries none. */
export function directionArrow(semantics: MetricSemantics | null | undefined): string {
  if (!semantics) return ''
  if (semantics.direction === 'higher_is_better') return '↑'
  if (semantics.direction === 'lower_is_better') return '↓'
  return ''
}

/** Label one final report metric without inferring anything from its name. */
export function formatMetricLabel(
  finalMetricName: string,
  semantics: MetricSemantics | null | undefined,
): string {
  if (!semantics || semantics.role === 'diagnostic') return finalMetricName
  return `${semantics.metric_name} ${directionArrow(semantics)}`.trimEnd()
}

/** Label a v2 identity; default mean stays in the tooltip/key, while dimensions disambiguate rows. */
export function formatMetricIdentityLabel(
  identity: MetricIdentity,
  semantics: MetricSemantics | null | undefined,
  legacyName?: string | null,
): string {
  if (!semantics || semantics.role === 'diagnostic') return legacyName || metricIdentityKey(identity)
  const base = formatMetricLabel(identity.name, semantics)
  const dimensionOrder: Record<string, number> = {
    target: 0,
    level: 1,
    scope: 2,
    ngram: 3,
    variant: 3,
    statistic: 4,
  }
  const dimensions = Object.entries(identity.dimensions)
    .sort(([left], [right]) => (dimensionOrder[left] ?? 3) - (dimensionOrder[right] ?? 3) || left.localeCompare(right))
    .map(([, value]) => {
    if (typeof value === 'boolean') return value ? 'Yes' : 'No'
    return String(value)
      .replaceAll('_', ' ')
      .replace(/\b\w/g, (character) => character.toUpperCase())
    })
  return dimensions.length > 0 ? `${base} · ${dimensions.join(' · ')}` : base
}

/** Label all metrics of one report and disambiguate repeated semantic display names. */
export function formatMetricLabels(
  metrics: ReadonlyArray<{ metricName: string; semantics: MetricSemantics | null | undefined }>,
): Record<string, string> {
  const labels = Object.fromEntries(
    metrics.map(({ metricName, semantics }) => [metricName, formatMetricLabel(metricName, semantics)]),
  )
  const counts = new Map<string, number>()
  for (const label of Object.values(labels)) counts.set(label, (counts.get(label) ?? 0) + 1)
  return Object.fromEntries(
    metrics.map(({ metricName }) => [
      metricName,
      counts.get(labels[metricName])! > 1 ? `${labels[metricName]} (${metricName})` : labels[metricName],
    ]),
  )
}

/**
 * Round `value` to `precision` decimal places with ties toward positive infinity
 * (`0.5 → 1`, `-0.5 → 0`, `2.5 → 3`). Decimal-point shifting avoids the binary drift of
 * `toFixed` at values such as `1.005`.
 *
 * The implementation shifts the decimal point through exponential-notation string parsing
 * (`"1.005e2"` parses to the nearest double of `100.5`, avoiding the `1.005 * 100 = 100.4999…`
 * error) and only falls back to arithmetic scaling for values that stringify exponentially.
 *
 * @param value The number to round.
 * @param precision Number of decimal places to keep (>= 0).
 * @returns The rounded number.
 */
export function roundHalfUp(value: number, precision: number): number {
  if (!Number.isFinite(value)) {
    return value
  }
  const safePrecision = precision >= 0 ? precision : 0
  const shifted = Number(`${value}e${safePrecision}`)
  if (Number.isFinite(shifted)) {
    const rounded = Number(`${Math.round(shifted)}e${-safePrecision}`)
    if (Number.isFinite(rounded)) {
      return rounded
    }
  }
  // Fallback for values that stringify in exponential notation (e.g. 1e-7).
  const factor = 10 ** safePrecision
  return Math.round(value * factor) / factor
}

/**
 * Render a rounded number in its shortest exact decimal form, matching the backend formatter
 * (`50` rather than `50.0`), so both sides agree character for character.
 */
function formatNumber(value: number, precision: number): string {
  const rounded = roundHalfUp(value, precision)
  // Collapse -0 so it stringifies like 0 on both sides.
  return String(rounded === 0 ? 0 : rounded)
}

/**
 * Join a numeric string with its unit, using the same rule as the backend `_join_unit`: a
 * `percent` value is glued to its unit (`85.7%`), any other kind gets one space (`1.235 s`).
 *
 * The separator is derived from `displayKind`, never from the unit string, so the two
 * implementations cannot drift for an unusual combination such as `percent` + `pp`.
 */
function joinUnit(text: string, unit: string, displayKind: MetricDisplayKind): string {
  if (unit.length === 0) {
    return text
  }
  return displayKind === 'percent' ? `${text}${unit}` : `${text} ${unit}`
}

function isMissingValue(value: number | null | undefined): boolean {
  return value === null || value === undefined || !Number.isFinite(value)
}

/**
 * Format a metric value for display, driven only by the semantics' display fields.
 *
 * - Missing value → the placeholder in both `primary` and `raw`, `isMissing = true`.
 * - No semantics (backend did not provide any) → the raw number at the fallback precision, with
 *   no unit and no percentage conversion, `isDiagnosticFallback = true`.
 * - `display_kind === 'percent'` → `value * (display_multiplier ?? 1)` at `display_precision`,
 *   followed by `display_unit` with no separator.
 * - `display_kind === 'number'` → `value * (display_multiplier ?? 1)` at `display_precision`,
 *   then a space and the unit.
 *
 * @param value Raw metric value as stored in the report.
 * @param semantics Backend semantics of the metric; `null` triggers the diagnostic fallback.
 * @returns The display-ready `FormattedMetric`.
 */
export function formatMetric(
  value: number | null | undefined,
  semantics: MetricSemantics | null | undefined,
): FormattedMetric {
  const isDiagnosticFallback = !semantics || semantics.role === 'diagnostic'

  if (isMissingValue(value)) {
    return {
      primary: MISSING_PLACEHOLDER,
      raw: MISSING_PLACEHOLDER,
      unitLabel: '',
      isMissing: true,
      isDiagnosticFallback,
    }
  }

  const numeric = value as number
  if (!semantics) {
    const text = formatNumber(numeric, DIAGNOSTIC_FALLBACK_PRECISION)
    return { primary: text, raw: text, unitLabel: '', isMissing: false, isDiagnosticFallback }
  }

  const unitLabel = semantics.display_unit ?? ''
  const raw = joinUnit(formatNumber(numeric, RAW_VALUE_PRECISION), semantics.raw_unit ?? '', 'number')
  const scaled = numeric * (semantics.display_multiplier ?? 1)

  if (semantics.display_kind === 'percent') {
    return {
      primary: joinUnit(formatNumber(scaled, semantics.display_precision), unitLabel, 'percent'),
      raw,
      unitLabel,
      isMissing: false,
      isDiagnosticFallback,
    }
  }

  return {
    primary: joinUnit(formatNumber(scaled, semantics.display_precision), unitLabel, 'number'),
    raw,
    unitLabel,
    isMissing: false,
    isDiagnosticFallback,
  }
}

/**
 * Normalize a bounded quality metric into `[0, 1]` for colour scales and progress bars, so that
 * "fuller is better" always holds.
 *
 * This is the *quality* of a value, so it drives colour, not length. A `lower_is_better` metric is
 * inverted here, which is exactly why it must not size a bar -- see {@link getValuePosition}.
 *
 * Returns `null` — meaning "do not render a scale" — for a diagnostic metric, a metric without a
 * `value_range`, or one with `direction === 'none'`.
 *
 * @param value Raw metric value.
 * @param semantics Backend semantics of the metric.
 * @returns The ratio in `[0, 1]`, or `null` when a scale would be meaningless.
 */
export function getBoundedQualityRatio(
  value: number | null | undefined,
  semantics: MetricSemantics | null | undefined,
): number | null {
  const position = getValuePosition(value, semantics)
  if (position == null) {
    return null
  }
  return semantics!.direction === 'lower_is_better' ? 1 - position : position
}

/**
 * Where a value sits within its own range, as a ratio in `[0, 1]`.
 *
 * This is the magnitude of the value, never its quality: it is *not* inverted for a
 * `lower_is_better` metric. Use it to size anything whose length stands for "how much" -- a bar, a
 * track, a donut -- and use {@link getBoundedQualityRatio} for the colour that says "how good".
 *
 * Keeping the two apart matters. Sizing a bar by quality makes `WER 4.3%` draw a 95.7% full bar,
 * which sits next to `F1 91.2%` looking almost identical while describing a completely different
 * number. Sized by position instead, a low error rate is a short bar that is coloured green: the
 * length reads as "little error" and the colour reads as "good".
 *
 * Returns `null` -- meaning "do not render a scale" -- for a diagnostic metric, a metric without a
 * `value_range`, or one with `direction === 'none'`.
 *
 * @param value Raw metric value.
 * @param semantics Backend semantics of the metric.
 * @returns The ratio in `[0, 1]`, or `null` when no scale applies.
 */
export function getValuePosition(
  value: number | null | undefined,
  semantics: MetricSemantics | null | undefined,
): number | null {
  if (isMissingValue(value) || !semantics) {
    return null
  }
  if (semantics.role === 'diagnostic' || semantics.direction === 'none') {
    return null
  }
  const range = semantics.value_range
  if (!range || !(range.max > range.min)) {
    return null
  }

  const normalized = ((value as number) - range.min) / (range.max - range.min)
  return Math.min(1, Math.max(0, normalized))
}

/**
 * Format a *difference* between two values of a metric.
 *
 * A difference is not the same kind of quantity as the values it came from. Subtracting two
 * percentages gives percentage points, so rendering `0.5` through the original percent semantics
 * would claim "50%" where the truth is "50 pp" -- the gap between 50% and 100% is not itself half of
 * anything. Every other display kind keeps its own unit, since a difference of seconds is seconds.
 *
 * The value is pre-scaled and the derived semantics reset the multiplier to 1 because the result
 * renders as a plain `pp` number rather than as a percent.
 *
 * @param value Difference in the metric's native scale, e.g. `0.5` for a gap of 50 points.
 * @param semantics Semantics of the metric the difference was taken from.
 * @returns Formatted difference; diagnostic, so callers apply no colour scale or verdict to it.
 */
export function formatDifference(
  value: number | null | undefined,
  semantics: MetricSemantics | null | undefined,
): FormattedMetric {
  if (!semantics || semantics.display_kind !== 'percent') {
    return formatMetric(value, semantics)
  }
  const scaled = isMissingValue(value) ? value : (value as number) * (semantics.display_multiplier ?? 1)
  return formatMetric(scaled, {
    ...semantics,
    metric_name: 'Change',
    role: 'diagnostic',
    direction: 'none',
    display_kind: 'number',
    display_unit: 'pp',
    display_multiplier: 1,
    value_range: null,
  })
}

/** Outcome of comparing two values of the same metric. */
export type ComparisonVerdict = 'better' | 'worse' | 'equal' | 'incomparable'

/**
 * Decide whether a delta is an improvement, based only on the metric's direction.
 *
 * A diagnostic metric or one without a direction is `incomparable`: its change may be shown, but
 * never as a win or a loss.
 *
 * @param delta Difference between the two values (candidate minus baseline).
 * @param semantics Backend semantics of the metric.
 * @returns The verdict.
 */
export function getComparisonVerdict(delta: number, semantics: MetricSemantics | null | undefined): ComparisonVerdict {
  if (!semantics || semantics.role === 'diagnostic') {
    return 'incomparable'
  }
  const direction: MetricDirection = semantics.direction
  if (direction === 'none') {
    return 'incomparable'
  }
  if (!Number.isFinite(delta) || delta === 0) {
    return Number.isFinite(delta) ? 'equal' : 'incomparable'
  }
  const isImprovement = direction === 'higher_is_better' ? delta > 0 : delta < 0
  return isImprovement ? 'better' : 'worse'
}
