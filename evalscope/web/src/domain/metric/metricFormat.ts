/**
 * Metric formatting primitives.
 *
 * These are the only metric-related decisions the frontend makes, and each is a pure function of
 * the value plus the backend-provided `MetricSemantics`. There is no metric-name table, no alias
 * matching and no inference from a value's magnitude: the direction, unit, scale and precision of
 * a metric come from the backend contract, so every surface renders the same value identically
 * and a catalog fix reaches the UI without a frontend change.
 */

import type { MetricDirection, MetricSemantics } from './MetricSemantics'

/**
 * Placeholder shown for a missing metric value. Intentionally distinct from a legitimate `0` or
 * an empty string so a missing value can never be mistaken for a real zero.
 */
export const MISSING_PLACEHOLDER = '—'

/** Decimals used when the backend provided no semantics at all. */
const DIAGNOSTIC_FALLBACK_PRECISION = 4

/**
 * Decimals of the raw text, matching the backend's `RAW_VALUE_PRECISION`.
 *
 * The raw text never applies `display_multiplier`: it shows the stored value with its `raw_unit`,
 * so a tooltip or an export exposes what was actually recorded.
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

/**
 * Round `value` to `precision` decimal places using round-half-up semantics: a tie (`.5`) always
 * rounds toward positive infinity (`0.5 → 1`, `-0.5 → 0`, `2.5 → 3`). This differs from the
 * binary floating point drift of `toFixed`, which can misround values such as `1.005`.
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

/** Join a numeric string with its unit. `%` is attached directly, other units after a space. */
function joinUnit(text: string, unit: string): string {
  if (unit.length === 0) {
    return text
  }
  return unit === '%' ? `${text}${unit}` : `${text} ${unit}`
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
 * - `display_kind === 'number'` → the value at `display_precision`, then a space and the unit.
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
  const raw = joinUnit(formatNumber(numeric, RAW_VALUE_PRECISION), semantics.raw_unit ?? '')

  if (semantics.display_kind === 'percent') {
    const scaled = numeric * (semantics.display_multiplier ?? 1)
    return {
      primary: joinUnit(formatNumber(scaled, semantics.display_precision), unitLabel),
      raw,
      unitLabel,
      isMissing: false,
      isDiagnosticFallback,
    }
  }

  return {
    primary: joinUnit(formatNumber(numeric, semantics.display_precision), unitLabel),
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
