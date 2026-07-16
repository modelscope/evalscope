/**
 * Metric display contract.
 *
 * `MetricDisplaySpec` is the single source of truth for how a given metric is
 * presented across every surface (lists, page headers, overviews, detail views,
 * predictions, comparison tables and exports). Display form is decided from this
 * metadata rather than inferred from the magnitude of a value, so the same score
 * always renders with consistent semantics and units.
 */

/** Whether a metric has a fixed value range (e.g. a 0-1 ratio) or is open-ended. */
export type MetricBoundedness = 'bounded' | 'unbounded'

/** Direction in which "better" moves for a metric. */
export type MetricDirection = 'higher-is-better' | 'lower-is-better'

/** Metadata describing how a single metric should be displayed. */
export interface MetricDisplaySpec {
  /**
   * Implementation-level metric name. Hidden from users as a primary label and
   * only surfaced as secondary information in detail views / tooltips.
   */
  key: string
  /** Localized label key resolved through the locale system. */
  labelKey: string
  boundedness: MetricBoundedness
  direction: MetricDirection
  /**
   * Unit of the raw value. Preserved for unbounded metrics (e.g. `'ms'`,
   * `'tokens'`); `null` for bounded ratios that display as percentages.
   */
  unit: string | null
  /** Raw-value precision (round half up). Defaults to 4. */
  rawPrecision: number
  /** Percentage precision for bounded ratios (round half up). Defaults to 1. */
  percentPrecision: number
  /**
   * Whether the raw value is stored as 0-100 rather than 0-1. Used to normalize
   * bounded ratios before formatting.
   */
  storedAsHundred?: boolean
}

/** Default raw-value precision applied when a spec does not override it. */
export const DEFAULT_RAW_PRECISION = 4

/** Default percentage precision applied when a spec does not override it. */
export const DEFAULT_PERCENT_PRECISION = 1

/**
 * Fallback spec used when a metric has no registered `MetricDisplaySpec`.
 *
 * It represents an "undefined display form": the metric is treated as unbounded,
 * its raw value is shown with 4 decimal places, and no unit is assumed. The
 * display form is never guessed to be a percentage and the unit is never
 * converted.
 */
export const DEFAULT_METRIC_SPEC: MetricDisplaySpec = {
  key: '',
  labelKey: '',
  boundedness: 'unbounded',
  direction: 'higher-is-better',
  unit: null,
  rawPrecision: DEFAULT_RAW_PRECISION,
  percentPrecision: DEFAULT_PERCENT_PRECISION,
}

/** Registry mapping metric keys to their display specs. */
export type MetricRegistry = Record<string, MetricDisplaySpec>

/** Result of resolving a metric key against a registry. */
export interface ResolvedMetricSpec {
  /** The resolved spec, or `DEFAULT_METRIC_SPEC` when the key is not registered. */
  spec: MetricDisplaySpec
  /** `true` when the key was not found and the default fallback was used. */
  isFallback: boolean
}

/**
 * Resolve a metric's display spec from a registry.
 *
 * On a registry hit the registered spec is returned with `isFallback: false`.
 * On a miss `DEFAULT_METRIC_SPEC` is returned with `isFallback: true`, signalling
 * to the UI that the display form is undefined and the raw value should be shown
 * with default precision without any percentage/unit inference.
 *
 * @param key Implementation-level metric key to resolve.
 * @param registry Registry of known metric display specs.
 * @returns The resolved spec together with a fallback indicator.
 */
export function resolveSpec(key: string, registry: MetricRegistry): ResolvedMetricSpec {
  const spec = registry[key]
  if (spec === undefined) {
    return { spec: DEFAULT_METRIC_SPEC, isFallback: true }
  }
  return { spec, isFallback: false }
}
