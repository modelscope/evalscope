/**
 * Performance workload normalization — pure logic.
 *
 * This module turns a raw `PerfRunItem` into the workload context a reader needs
 * to interpret a performance run: concurrency, number of requests, and a request
 * rate label. It has no dependency on React, the DOM, the network, the system
 * clock, or randomness, which makes it the target of a property test
 * (Property 14: workload never renders `INF` and missing params show a placeholder).
 *
 * Rendering (list/detail layout, labels, columns) lives in the component layer;
 * this module only produces the data contract.
 */

import type { PerfRunItem } from '../../api/types'

/** Placeholder shown for a missing workload parameter. */
export const MISSING_WORKLOAD = 'N/A'

/** Semantic label for a concurrency-bound run with no rate limit. */
export const CLOSED_LOOP_LABEL = 'closed-loop'
/** Semantic label for an unbounded run with no rate limit. */
export const OPEN_LOOP_LABEL = 'open-loop'

/**
 * Normalized workload context for a single performance run.
 *
 * Every field is a display-ready string. Missing numeric parameters collapse to
 * {@link MISSING_WORKLOAD}; `rateLabel` never contains the literal `INF`.
 */
export interface PerfWorkload {
  /** Concurrency (parallel clients); missing → {@link MISSING_WORKLOAD}. */
  concurrency: string
  /** Number of requests; missing → {@link MISSING_WORKLOAD}. */
  numberOfRequests: string
  /**
   * Request rate label:
   * - truly unlimited → a semantic label ({@link CLOSED_LOOP_LABEL} /
   *   {@link OPEN_LOOP_LABEL}), never `INF`;
   * - finite → the finite value (e.g. `5/s`), never `INF`.
   */
  rateLabel: string
}

/** Return true only for a finite, real number (excludes NaN, ±Infinity). */
function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

/**
 * Format a non-negative count for display.
 * Returns {@link MISSING_WORKLOAD} when the value is not a usable finite number.
 */
function formatCount(value: unknown): string {
  if (!isFiniteNumber(value) || value < 0) return MISSING_WORKLOAD
  return String(value)
}

/**
 * Format a finite request rate without any trailing-zero noise (e.g. `5/s`,
 * `2.5/s`). The unit suffix mirrors the form's `req/s` convention.
 */
function formatRate(rate: number): string {
  // String() already normalizes `5.0` -> `"5"` while preserving `2.5`.
  return `${String(rate)}/s`
}

/**
 * Decide whether a request rate represents "truly unlimited" throughput.
 *
 * A rate is unlimited when it is absent (`null`/`undefined`), non-finite
 * (`Infinity`/`NaN`), a sentinel `<= 0` value (EvalScope's perf `--rate -1`
 * default means "no rate limit"), or a textual `INF`/`inf` marker. Accepts
 * `unknown` so the check stays robust to raw upstream payloads.
 */
function isUnlimitedRate(rate: unknown): boolean {
  if (rate == null) return true
  if (typeof rate === 'string') return true // any textual rate (e.g. "INF") is treated as unlimited
  if (typeof rate !== 'number') return true
  if (!Number.isFinite(rate)) return true
  return rate <= 0
}

/**
 * Build the workload context for a performance run.
 *
 * Invariants (Property 14):
 * - `concurrency` / `numberOfRequests` are the formatted values, or
 *   {@link MISSING_WORKLOAD} when missing;
 * - `rateLabel` is a semantic loop label when the rate is unlimited and the
 *   finite value when it is limited;
 * - `rateLabel` never contains the literal `INF`.
 */
export function normalizeWorkload(run: PerfRunItem): PerfWorkload {
  const concurrency = formatCount(run?.parallel)
  const numberOfRequests = formatCount(run?.number)

  let rateLabel: string
  if (isUnlimitedRate(run?.rate)) {
    // No rate limit: the run is either concurrency-bound (closed-loop) or,
    // without a concurrency limit, unbounded (open-loop).
    const hasConcurrencyLimit = isFiniteNumber(run?.parallel) && run.parallel > 0
    rateLabel = hasConcurrencyLimit ? CLOSED_LOOP_LABEL : OPEN_LOOP_LABEL
  } else {
    // Limited rate: show the finite value, never an `INF` label.
    rateLabel = formatRate(run.rate as number)
  }

  return { concurrency, numberOfRequests, rateLabel }
}
