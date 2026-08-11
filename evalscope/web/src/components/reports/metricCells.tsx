import { scoreColor } from '@/utils/colorScale'
import { formatMetric, getBoundedQualityRatio } from '@/domain/metric'
import { metricIdentityKey } from '@/domain/metric'
import { datasetLabel, metricLabel } from '@/domain/report/primaryMetrics'
import type { PrimaryMetricRef } from '@/domain/report/primaryMetrics'

/**
 * Row-aligned cells for a run's per-dataset results.
 *
 * A run may cover several datasets, each measured by its own metric. Packing
 * `dataset + metric + value` into one cell made the text collide with the neighbouring Dataset
 * column, which already named the datasets. Instead the three facts are split across three
 * columns and rendered one line per dataset, so line N of Dataset, Metric and Score all describe
 * the same dataset. Every renderer here emits the same number of lines at the same line height,
 * which is what keeps them aligned.
 */

/**
 * Shared line box, so the three columns line up row by row.
 *
 * `min-h-5` rather than a fixed height: tabular metadata in this project wraps and is never
 * truncated, so an unusually long name grows its line instead of being cut off.
 */
const LINE = 'flex min-h-5 items-center leading-5 break-words min-w-0'

/** How many dataset lines to show before collapsing the rest into a count. */
export const MAX_VISIBLE_LINES = 3

/** Refs to render, plus how many were dropped. */
function visibleOf(refs: PrimaryMetricRef[]): { visible: PrimaryMetricRef[]; hidden: number } {
  const visible = refs.slice(0, MAX_VISIBLE_LINES)
  return { visible, hidden: refs.length - visible.length }
}

/** Stable key for a ref, which is unique per dataset within a run. */
function keyOf(ref: PrimaryMetricRef): string {
  return `${ref.dataset_name}:${metricIdentityKey(ref.identity)}`
}

interface LinesProps {
  refs: PrimaryMetricRef[]
  className?: string
}

/** One dataset name per line. */
export function DatasetLines({ refs, fallback, className }: LinesProps & { fallback: string }) {
  if (refs.length === 0) {
    // Same wrapping guarantee as a rendered line, for a response that carries no per-dataset refs.
    return <span className={`break-words min-w-0 ${className ?? ''}`.trim()}>{fallback}</span>
  }
  const { visible, hidden } = visibleOf(refs)
  return (
    <div className={className}>
      {visible.map((ref) => (
        <div key={keyOf(ref)} className={LINE} title={ref.dataset_name}>{datasetLabel(ref)}</div>
      ))}
      {hidden > 0 && <div className={`${LINE} text-[var(--text-dim)]`}>+{hidden}</div>}
    </div>
  )
}

/**
 * One metric label per line.
 *
 * A column of its own, so line N here names the metric behind line N of Score. Do not also pass
 * `inlineMetricClass` to {@link ScoreLines} in a layout that renders this column: the label would
 * appear twice.
 */
export function MetricLines({ refs, className }: LinesProps) {
  const { visible, hidden } = visibleOf(refs)
  return (
    <div className={className}>
      {visible.map((ref) => {
        const label = metricLabel(ref)
        return (
          <div
            key={keyOf(ref)}
            className={LINE}
            title={metricIdentityKey(ref.identity)}
          >
            {label}
          </div>
        )
      })}
      {hidden > 0 && <div className={LINE} />}
    </div>
  )
}

/**
 * One value per line, right-aligned and coloured by its own metric's quality scale.
 *
 * `inlineMetricClass` and `inlineDatasetClass` add the metric label and the dataset name to each
 * line. Both default to off, because in a table those facts have their own columns and repeating
 * them there is what made the cell collide with its neighbour. Pass a class to hide the inline
 * copy at the width where the column exists, or an empty string to always show it -- which is what
 * a layout with no such column, like the mobile card, needs in order to stay unambiguous.
 */
export function ScoreLines(
  { refs, emptyLabel, inlineMetricClass, inlineDatasetClass, className }:
    LinesProps & { emptyLabel: string; inlineMetricClass?: string; inlineDatasetClass?: string },
) {
  if (refs.length === 0) {
    return <span className={`text-xs text-[var(--text-muted)] ${className ?? ''}`}>{emptyLabel}</span>
  }
  const { visible, hidden } = visibleOf(refs)
  return (
    <div className={className}>
      {visible.map((ref) => {
        // Colour says how good the value is; the arrow in the metric label says which way is
        // better. No bar: the lines of this column can be different metrics on different scales.
        const quality = getBoundedQualityRatio(ref.score, ref.semantics)
        return (
          <div key={keyOf(ref)} className={`${LINE} justify-end gap-1.5`}>
            {inlineDatasetClass !== undefined && (
              <span className={`text-[11px] text-[var(--text-dim)] ${inlineDatasetClass}`}>
                {datasetLabel(ref)}
              </span>
            )}
            {inlineMetricClass !== undefined && (
              <span className={`text-[11px] font-normal text-[var(--text-muted)] ${inlineMetricClass}`}>
                {metricLabel(ref)}
              </span>
            )}
            <span
              className="font-mono font-semibold tabular-nums"
              style={{ color: quality == null ? 'var(--text)' : scoreColor(quality) }}
            >
              {formatMetric(ref.score, ref.semantics).primary}
            </span>
          </div>
        )
      })}
      {hidden > 0 && <div className={`${LINE} justify-end text-[var(--text-dim)]`}>…</div>}
    </div>
  )
}
