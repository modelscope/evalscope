import { cn } from '@/lib/utils'
import { scoreColor } from '@/utils/colorScale'
import { formatMetric, getBoundedQualityRatio, getValuePosition } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'

/**
 * How wide the track is allowed to be.
 *
 * `fill` grows to the column width, which is what makes the length difference
 * between two rows readable in a main results table. `fixed` keeps a short
 * constant track for a dense secondary table, where the column is narrow and a
 * growing bar would crowd its neighbour.
 */
type TrackWidth = 'fill' | 'fixed'

interface ScoreBarProps {
  score: number | null | undefined
  /** Backend semantics of this row's own metric. Decides formatting, range and direction. */
  semantics?: MetricSemantics | null
  /** Programmatic name of what is measured, announced together with the value. */
  ariaLabel: string
  track?: TrackWidth
  className?: string
}

const ROW: Record<TrackWidth, string> = {
  fill: 'flex items-center justify-end gap-3',
  fixed: 'flex items-center gap-2',
}

const TRACK: Record<TrackWidth, string> = {
  // Hidden below `sm`: at that width the value alone has to carry the cell.
  fill: 'hidden h-1.5 min-w-9 flex-1 overflow-hidden rounded-full bg-[var(--border)] sm:block',
  fixed: 'h-1.5 w-[60px] min-w-[60px] overflow-hidden rounded-full bg-[var(--border)]',
}

const VALUE: Record<TrackWidth, string> = {
  fill: 'min-w-14 shrink-0 text-right font-mono text-xs font-semibold tabular-nums sm:text-sm',
  fixed: 'font-mono font-medium tabular-nums',
}

/**
 * One metric value, coloured by its own quality scale and optionally preceded by
 * a proportional bar.
 *
 * The bar length is the value's own position in its own range and is never
 * inverted, so two different metrics never draw the same length: an F1 of 91.2%
 * is long, a WER of 4.3% is short. The colour is what carries the quality, which
 * is why that short WER bar is green. Sizing by quality instead is what used to
 * make those two bars look identical.
 *
 * The bar is omitted entirely when the metric has no bounded range to place the
 * value in, because a bar with no range would imply a proportion that does not
 * exist.
 */
export default function ScoreBar({
  score,
  semantics,
  ariaLabel,
  track = 'fill',
  className,
}: ScoreBarProps) {
  const position = getValuePosition(score, semantics)
  const quality = getBoundedQualityRatio(score, semantics)

  return (
    <div className={cn(ROW[track], className)}>
      {position != null && (
        <div className={TRACK[track]}>
          <div
            role="progressbar"
            aria-label={ariaLabel}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(position * 100)}
            className="h-full rounded-full transition-all duration-300"
            style={{ width: `${position * 100}%`, background: scoreColor(quality ?? position) }}
          />
        </div>
      )}
      <span
        className={VALUE[track]}
        style={{ color: quality == null ? 'var(--text)' : scoreColor(quality) }}
      >
        {formatMetric(score, semantics).primary}
      </span>
    </div>
  )
}
