import { scoreColor } from '@/utils/colorScale'
import { formatMetric, getBoundedQualityRatio } from '@/domain/metric'
import { metricLabel } from '@/domain/report/primaryMetrics'
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
  return `${ref.dataset_name}:${ref.metric_name}`
}

/**
 * The metric label shared by every ref, or `null` when they differ.
 *
 * When a whole table measures one metric — by far the common case — the label belongs in the
 * column header rather than repeated down every row, and the Metric column can disappear
 * entirely. Returns `null` for an empty input, since there is nothing to hoist.
 */
export function uniformMetricLabel(refGroups: PrimaryMetricRef[][]): string | null {
  const labels = new Set<string>()
  for (const refs of refGroups) {
    for (const ref of refs) {
      labels.add(metricLabel(ref))
    }
  }
  return labels.size === 1 ? [...labels][0] : null
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
        <div key={keyOf(ref)} className={LINE}>{ref.dataset_name}</div>
      ))}
      {hidden > 0 && <div className={`${LINE} text-[var(--text-dim)]`}>+{hidden}</div>}
    </div>
  )
}

/**
 * One metric label per line.
 *
 * Render this column only when {@link uniformMetricLabel} returns `null`; otherwise the label is
 * in the header and this column is redundant.
 */
export function MetricLines({ refs, inferredHint, className }: LinesProps & { inferredHint?: string }) {
  const { visible, hidden } = visibleOf(refs)
  return (
    <div className={className}>
      {visible.map((ref) => {
        const label = metricLabel(ref)
        return (
          <div
            key={keyOf(ref)}
            className={LINE}
            title={ref.inferred ? `${ref.metric_name} — ${inferredHint ?? ''}` : ref.metric_name}
          >
            {label}
            {ref.inferred && <span className="ml-0.5 opacity-60">*</span>}
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
                {ref.dataset_name}
              </span>
            )}
            {inlineMetricClass !== undefined && (
              <span className={`text-[11px] font-normal text-[var(--text-muted)] ${inlineMetricClass}`}>
                {metricLabel(ref)}
                {ref.inferred && <span className="ml-0.5 opacity-60">*</span>}
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
