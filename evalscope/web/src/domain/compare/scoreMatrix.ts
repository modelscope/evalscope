/**
 * Score-matrix data logic (no rendering).
 *
 * The Compare view's score tab renders one row per dataset and one column per
 * report, optionally expressed as a delta against a chosen baseline. Deciding
 * how strong a delta is, how it reads as text and which direction counts as an
 * improvement is pure computation, so it lives here and is exercised directly by
 * unit tests rather than through the table.
 */

import { formatDifference, getComparisonVerdict } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'

/** Weakest and strongest background tint applied to a non-zero delta. */
const MIN_TINT = 0.06
const MAX_EXTRA_TINT = 0.24

/**
 * Background tint for a delta cell, scaled by how large the delta is relative to
 * the largest one in its row.
 *
 * A metric with no direction (or an equal value) carries no verdict, so it gets
 * the plain surface rather than a colour that would imply one. `maxAbsoluteDelta`
 * of zero means every report scored the same and there is nothing to scale.
 *
 * @param delta - Candidate minus baseline.
 * @param maxAbsoluteDelta - Largest absolute delta in the same row.
 * @param semantics - Backend semantics of the row's metric.
 * @returns A CSS colour, always a `color-mix` against the deep surface.
 */
export function comparisonDeltaBackground(
  delta: number,
  maxAbsoluteDelta: number,
  semantics: MetricSemantics | undefined,
): string {
  const verdict = getComparisonVerdict(delta, semantics)
  if (verdict === 'equal' || verdict === 'incomparable' || maxAbsoluteDelta === 0) return 'var(--bg-deep)'
  const intensity = Math.min(1, Math.abs(delta) / maxAbsoluteDelta)
  const weight = Math.round((MIN_TINT + intensity * MAX_EXTRA_TINT) * 100)
  const semanticColor = verdict === 'better' ? 'var(--success)' : 'var(--danger)'
  return `color-mix(in srgb, ${semanticColor} ${weight}%, var(--bg-deep))`
}

/**
 * Format a delta with an explicit sign for the positive case.
 *
 * The metric's own formatter decides the unit and precision; only the leading `+`
 * is added here, because a negative value already carries its sign.
 */
export function signedDifference(delta: number, semantics: MetricSemantics | undefined): string {
  const formatted = formatDifference(delta, semantics).primary
  return delta > 0 ? `+${formatted}` : formatted
}

/**
 * Largest absolute delta per row, used to scale each row's tint independently.
 *
 * Scaling per row rather than across the whole table keeps a dataset with a wide
 * spread from flattening the colour of every other dataset.
 *
 * @param rows - Score rows, each keyed by report reference plus a `dataset_id`.
 * @param reportKeys - Report references that form the columns.
 * @param baselineKey - Report reference the deltas are measured against.
 * @returns Max absolute delta keyed by `dataset_id`; `0` when a row has no spread.
 */
export function computeDeltaRanges(
  rows: Record<string, unknown>[],
  reportKeys: string[],
  baselineKey: string,
): Record<string, number> {
  const ranges: Record<string, number> = {}
  for (const row of rows) {
    const baseline = Number(row[baselineKey])
    ranges[String(row.dataset_id)] = Math.max(
      0,
      ...reportKeys.map((report) => Math.abs(Number(row[report]) - baseline)),
    )
  }
  return ranges
}
