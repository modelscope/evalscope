/**
 * Metric formatting: the single entry point that turns a raw metric value plus
 * its `MetricDisplaySpec` into display-ready strings.
 *
 * Every surface (lists, page headers, overviews, detail views, predictions,
 * comparison tables and exports) formats metrics through `formatMetric`, so the
 * same value renders with identical precision and rounding everywhere
 * (Req 1.6, 8.10, 15.4). This module is pure logic: it has no DOM, network,
 * clock or randomness dependency, which makes it the target of property tests.
 */

import type { MetricDisplaySpec } from './MetricDisplaySpec'
import { DEFAULT_METRIC_SPEC } from './MetricDisplaySpec'

/**
 * Placeholder shown for a missing metric value. It is intentionally distinct
 * from a legitimate `0` or an empty string so a missing value can never be
 * mistaken for a real zero (Req 1.8).
 */
export const MISSING_PLACEHOLDER = '—'

/** Display-ready representation of a single metric value. */
export interface FormattedMetric {
  /**
   * Primary display text: a percentage for bounded ratios, the raw value plus
   * its unit for unbounded metrics, and the missing placeholder when the value
   * is absent.
   */
  primary: string
  /**
   * Raw value for tooltips, detail views and copy actions; the missing
   * placeholder when the value is absent.
   */
  raw: string
  /** Localized unit label; an empty string for bounded ratios. */
  unitLabel: string
  /** `true` when the value is `null`, `undefined`, `NaN` or otherwise non-finite. */
  isMissing: boolean
  /**
   * `true` when the metric has no registered spec and the default fallback is
   * used. Detected by reference-equality against `DEFAULT_METRIC_SPEC`, which
   * `resolveSpec` returns on a registry miss (its `isFallback` flag). The UI
   * uses this to signal that the display form is undefined (Req 1.13).
   */
  isSpecUndefined: boolean
}

/**
 * Round `value` to `precision` decimal places using round-half-up semantics:
 * a tie (`.5`) always rounds toward positive infinity (e.g. `0.5 → 1`,
 * `-0.5 → 0`, `2.5 → 3`). This differs from the binary floating point drift of
 * `toFixed`, which can misround values such as `1.005`.
 *
 * The implementation shifts the decimal point via exponential-notation string
 * parsing (`"1.005e2"` parses to the nearest double of `100.5`, avoiding the
 * `1.005 * 100 = 100.4999…` error) and only falls back to arithmetic scaling
 * for values that stringify in exponential notation.
 *
 * @param value The number to round.
 * @param precision Number of decimal places to keep (>= 0).
 * @returns The rounded number.
 */
export function roundHalfUp(value: number, precision: number): number {
  if (!Number.isFinite(value)) {
    return value
  }
  const shifted = Number(`${value}e${precision}`)
  if (Number.isFinite(shifted)) {
    const rounded = Number(`${Math.round(shifted)}e${-precision}`)
    if (Number.isFinite(rounded)) {
      return rounded
    }
  }
  // Fallback for values that stringify in exponential notation (e.g. 1e-7).
  const factor = 10 ** precision
  return Math.round(value * factor) / factor
}

/** Translate function contract used to localize unit labels. */
type Translate = (key: string) => string

/** Format a rounded number to a fixed number of decimals, padding trailing zeros. */
function toFixedString(value: number, precision: number): string {
  const safePrecision = precision >= 0 ? precision : 0
  return roundHalfUp(value, safePrecision).toFixed(safePrecision)
}

/**
 * Join a numeric string with its unit label. `%` is attached without a space
 * (e.g. `"12.35%"`); other units are separated by a space (e.g. `"123 ms"`).
 */
function joinUnit(numberStr: string, spec: MetricDisplaySpec, unitLabel: string): string {
  if (unitLabel.length === 0) {
    return numberStr
  }
  const separator = spec.unit === '%' ? '' : ' '
  return `${numberStr}${separator}${unitLabel}`
}

/**
 * Format a metric value for display according to its `MetricDisplaySpec`.
 *
 * Behaviour:
 * - Missing value (`null` / `undefined` / `NaN` / non-finite) → `primary` and
 *   `raw` are the missing placeholder, `isMissing = true` (Req 1.8).
 * - Bounded ratio (`spec.boundedness === 'bounded'`) → `primary` is a percentage
 *   with `percentPrecision` decimals (round half up, e.g. `"92.0%"`) and `raw`
 *   is the 0-1 ratio with `rawPrecision` decimals (e.g. `"0.9200"`). When
 *   `spec.storedAsHundred` is set the raw value is stored as 0-100 and is
 *   normalized to 0-1 before both computations (Req 1.2, 1.3).
 * - Unbounded / benchmark-native metric → `primary` is the raw value plus its
 *   unit at `rawPrecision`; no percentage conversion happens, regardless of
 *   whether the value exceeds 1 (Req 1.4, 1.5, 1.7).
 * - Missing spec (the shared `DEFAULT_METRIC_SPEC`, detected by reference) →
 *   `isSpecUndefined = true`; the value is shown as a raw number with the default
 *   precision, without inferring a percentage or unit (Req 1.13).
 *
 * The function is pure and deterministic: the same `(value, spec)` always
 * produces field-for-field identical output (Property 1).
 *
 * @param value Raw metric value; `null` / `undefined` / `NaN` mean "missing".
 * @param spec Display spec describing how to render the metric.
 * @param t Locale translate function used to resolve the unit label.
 * @returns The display-ready `FormattedMetric`.
 */
export function formatMetric(
  value: number | null | undefined,
  spec: MetricDisplaySpec,
  t: Translate,
): FormattedMetric {
  const isSpecUndefined = spec === DEFAULT_METRIC_SPEC

  // Missing value: distinct placeholder, never rendered as 0 or blank (Req 1.8).
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return {
      primary: MISSING_PLACEHOLDER,
      raw: MISSING_PLACEHOLDER,
      unitLabel: '',
      isMissing: true,
      isSpecUndefined,
    }
  }

  const rawPrecision = spec.rawPrecision
  const percentPrecision = spec.percentPrecision

  // Bounded ratio → percentage primary + 0-1 raw (Req 1.2, 1.3).
  if (spec.boundedness === 'bounded') {
    const ratio = spec.storedAsHundred ? value / 100 : value
    const primary = `${toFixedString(ratio * 100, percentPrecision)}%`
    const raw = toFixedString(ratio, rawPrecision)
    return {
      primary,
      raw,
      unitLabel: '',
      isMissing: false,
      isSpecUndefined,
    }
  }

  // Unbounded / native metric → keep unit, no percentage conversion
  // (Req 1.4, 1.5, 1.7). Missing spec falls through here as a plain raw value
  // with default precision and no unit (Req 1.13).
  const rawStr = toFixedString(value, rawPrecision)
  const unitLabel = spec.unit ? t(spec.unit) : ''
  return {
    primary: joinUnit(rawStr, spec, unitLabel),
    raw: rawStr,
    unitLabel,
    isMissing: false,
    isSpecUndefined,
  }
}
