/**
 * Pure logic for adaptive visualization selection (Chart_Renderer).
 *
 * The renderer must match the visualization form to the number of normalized
 * dimensions in a report rather than always drawing a radar chart. This module
 * holds the deterministic, side-effect-free mapping so it can be exercised
 * directly by property-based tests; `AdaptiveVisualization` consumes it to pick
 * a concrete rendering.
 */

/**
 * Visualization forms selected by dimension count.
 *
 * - `'empty'`        — no dimensions; render an empty state, no chart (Req 3.5).
 * - `'single-value'` — one dimension; render its normalized value plus category
 *   label, never a single-axis radar (Req 3.1).
 * - `'grouped-bar'`  — two dimensions; render a grouped bar / dot plot, never a
 *   radar (Req 3.2).
 * - `'radar'`        — three or more dimensions; radar is the preferred chart
 *   with one axis per normalized dimension (Req 3.3).
 */
export type VizKind = 'empty' | 'single-value' | 'grouped-bar' | 'radar'

/**
 * Select a visualization form from the number of normalized dimensions.
 *
 * The mapping is total and deterministic (Property 6): `0 → 'empty'`,
 * `1 → 'single-value'`, `2 → 'grouped-bar'`, `>= 3 → 'radar'`. Non-integer or
 * negative inputs are normalized (floored, clamped at 0) so the function never
 * returns an out-of-range value.
 *
 * @param dimensionCount - Count of normalized dimensions in the report.
 * @returns The visualization form the renderer should use.
 */
export function selectVisualization(dimensionCount: number): VizKind {
  // Defensive normalization: treat NaN / negatives / fractions as their floored,
  // clamped integer so the selection stays total for any numeric input.
  const count = Number.isFinite(dimensionCount) ? Math.max(0, Math.floor(dimensionCount)) : 0

  if (count === 0) return 'empty'
  if (count === 1) return 'single-value'
  if (count === 2) return 'grouped-bar'
  return 'radar'
}
