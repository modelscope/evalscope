/**
 * TypeScript mirror of the backend `MetricSemantics` contract.
 *
 * Field names stay snake_case so the shape is byte-for-byte the wire format the API sends: no
 * renaming layer, no adapter, nothing to keep in sync beyond this file. The backend definition
 * lives in `evalscope/api/metric/semantics.py`.
 */

/** Display tier of a metric and whether it may take part in verdicts. */
export type MetricRole = 'primary' | 'auxiliary' | 'diagnostic'

/** Optimization direction of a metric. */
export type MetricDirection = 'higher_is_better' | 'lower_is_better' | 'none'

/** How a metric value is rendered. */
export type MetricDisplayKind = 'number' | 'percent'

/** Closed value range of a bounded metric. */
export interface ValueRange {
  min: number
  max: number
}

/** Single source of truth for how one final report metric is interpreted and displayed. */
export interface MetricSemantics {
  /** Unique semantic identifier, named `{domain}.{concept}.{unit}`. */
  semantic_id: string
  /** Display name of the metric; may differ from the final report metric name. */
  metric_name: string
  role: MetricRole
  direction: MetricDirection
  /** Unit of the raw stored value (`s`, `ms`, `tok/s`, ...). */
  raw_unit?: string | null
  /** Value range for bounded metrics; `null` means unbounded. */
  value_range?: ValueRange | null
  display_kind: MetricDisplayKind
  /** Display multiplier; `null` means undeclared, consumers use 1. */
  display_multiplier?: number | null
  /** Unit appended to the displayed value (`%`, `s`, `ms`, ...). */
  display_unit?: string | null
  /** Number of decimals; ties are rounded toward positive infinity. */
  display_precision: number
  /** Version of the backend contract this declaration follows. */
  contract_version: number
}
