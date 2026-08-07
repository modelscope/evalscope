import { cn } from '@/lib/utils'
import { scoreColor } from '@/utils/colorScale'
import { formatMetric, getBoundedQualityRatio } from '@/domain/metric'
import { metricLabel } from '@/domain/report/primaryMetrics'
import type { PrimaryMetricRef } from '@/domain/report/primaryMetrics'

/**
 * A run's result, always shown as concrete `metric -> score` values.
 *
 * A run may cover several datasets, and their primary metrics are frequently different (an
 * accuracy here, a WER there). Collapsing that into one number would be a fake total, but
 * replacing it with a note like "multiple metrics" hides the very numbers the run produced. So
 * every dataset gets its own line, labelled with its metric and formatted with that metric's own
 * semantics; only a single-dataset run renders as one badge.
 */

/** How many dataset lines to show before collapsing the rest into a count. */
const MAX_VISIBLE_ROWS = 3

/**
 * Label styling for a metric name.
 *
 * Deliberately not upper-cased: a declared metric has a short display name (`Accuracy`, `WER`),
 * but an inferred one falls back to its raw name, and `AVG@1_ALL/SUCCESS_RATE` is markedly harder
 * to read than `avg@1_all/success_rate`. Long names truncate with the full text in the tooltip.
 */
const METRIC_LABEL_CLASS = 'text-[11px] text-[var(--text-muted)] truncate max-w-[13rem]'

interface PrimaryMetricResultProps {
  /** One reference per dataset, in display order. */
  refs: PrimaryMetricRef[]
  /** Layout: `stacked` for a table cell, `inline` for a compact card row. */
  variant?: 'stacked' | 'inline'
  /** Placeholder for a run that reports no metric at all. */
  emptyLabel: string
  /** Tooltip explaining that a metric was inferred rather than declared. */
  inferredHint?: string
  className?: string
}

/** Colour a value only when its metric defines a bounded quality scale. */
function valueStyle(ref: PrimaryMetricRef): React.CSSProperties {
  const ratio = getBoundedQualityRatio(ref.score, ref.semantics)
  return ratio == null
    ? { color: 'var(--text)' }
    : { color: scoreColor(ratio) }
}

export default function PrimaryMetricResult({
  refs,
  variant = 'stacked',
  emptyLabel,
  inferredHint,
  className,
}: PrimaryMetricResultProps) {
  if (refs.length === 0) {
    return <span className={cn('text-xs text-[var(--text-muted)]', className)}>{emptyLabel}</span>
  }

  // A single dataset needs no dataset column: the row already names it.
  if (refs.length === 1) {
    const ref = refs[0]
    const formatted = formatMetric(ref.score, ref.semantics)
    const label = metricLabel(ref)
    return (
      <span
        className={cn('inline-flex items-baseline gap-1.5 whitespace-nowrap', className)}
        title={ref.inferred ? `${label} — ${inferredHint ?? ''}` : label}
      >
        <span className={METRIC_LABEL_CLASS}>
          {label}
          {ref.inferred && <span className="ml-0.5 opacity-60">*</span>}
        </span>
        <span className="text-sm font-mono font-semibold tabular-nums" style={valueStyle(ref)}>
          {formatted.primary}
        </span>
      </span>
    )
  }

  const visible = refs.slice(0, MAX_VISIBLE_ROWS)
  const hidden = refs.length - visible.length

  return (
    <div
      className={cn(
        'flex gap-x-3 gap-y-0.5',
        variant === 'inline' ? 'flex-row flex-wrap items-baseline' : 'flex-col items-end',
        className,
      )}
    >
      {visible.map((ref) => {
        const formatted = formatMetric(ref.score, ref.semantics)
        const label = metricLabel(ref)
        return (
          <span
            key={ref.dataset_name + ref.metric_name}
            className="inline-flex items-baseline gap-1.5 whitespace-nowrap"
            title={ref.inferred ? `${label} — ${inferredHint ?? ''}` : label}
          >
            <span className="text-[11px] text-[var(--text-dim)]">{ref.dataset_name}</span>
            <span className={METRIC_LABEL_CLASS}>
              {label}
              {ref.inferred && <span className="ml-0.5 opacity-60">*</span>}
            </span>
            <span className="text-xs font-mono font-semibold tabular-nums" style={valueStyle(ref)}>
              {formatted.primary}
            </span>
          </span>
        )
      })}
      {hidden > 0 && (
        <span className="text-[11px] text-[var(--text-dim)]">+{hidden}</span>
      )}
    </div>
  )
}
