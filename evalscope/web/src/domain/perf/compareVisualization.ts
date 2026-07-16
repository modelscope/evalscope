/**
 * Performance comparison visualization selection — pure logic, no rendering
 * (Req 9.11, 9.12, task 13.8).
 *
 * The Performance_Compare_View must not overstate a trend when there are too
 * few data points to support one. This module encodes that single decision:
 * given the number of data points `n` participating in a comparison it returns
 * whether the view should render a sparse, non-trending form (discrete point
 * markers) or a distribution / trend form.
 *
 * It has no dependency on React, the DOM, the network, the system clock or
 * randomness, so it is the target of the property test in task 13.9 (Property
 * 19). The actual chart rendering lives in the component layer (task 13.10);
 * this module only makes the choice.
 */

/**
 * Comparison visualization mode (Req 9.11, 9.12):
 * - `sparse` — non-trending sparse form (discrete point markers, no
 *   continuous curve / trend line), used when `n <= 2`;
 * - `trend` — distribution / trend form, used when `n > 2`.
 */
export type CompareVisualization = 'sparse' | 'trend'

/**
 * Upper bound (inclusive) for the sparse form: comparisons with at most this
 * many data points do not emphasize a trend (Req 9.11).
 */
const SPARSE_MAX_POINTS = 2

/**
 * Select the comparison visualization mode for `n` data points.
 *
 * `n > 2` selects `'trend'` (a distribution / trend form, Req 9.12); every
 * other input — `n <= 2`, as well as invalid inputs such as `n <= 0` or a
 * non-finite value (`NaN`, `Infinity`) — collapses to the conservative
 * `'sparse'` form (Req 9.11), so the view never overstates a trend it cannot
 * support. Non-integer counts are compared as-is (e.g. `2.5 → 'trend'`).
 *
 * @param n - Number of data points participating in the comparison.
 * @returns `'trend'` when `n > 2`, otherwise `'sparse'`.
 */
export function selectCompareVisualization(n: number): CompareVisualization {
  if (!Number.isFinite(n)) return 'sparse'
  return n > SPARSE_MAX_POINTS ? 'trend' : 'sparse'
}
