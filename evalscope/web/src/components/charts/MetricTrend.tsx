import { useState } from 'react'
import { getBoundedQualityRatio, formatMetric } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'
import { scoreColor } from '@/utils/colorScale'
import { trendBounds, trendPosition } from '@/domain/report/runAggregation'
import type { CellPoint } from '@/domain/report/runAggregation'

/**
 * A metric's history, drawn as bars.
 *
 * Built from the same primitive the score bars elsewhere in this app use -- a positioned `div`
 * sized by percentage -- rather than an SVG path or a charting library. This project deliberately
 * ships no client-side plotting library: charts are rendered by the backend and embedded through
 * `ChartFrame`, which is an iframe and therefore unusable once per table row.
 *
 * Bar height is magnitude, never quality. A `lower_is_better` series is not flipped, so a falling
 * error rate draws falling bars, while the colour says those low values are good. Sizing by quality
 * instead is what once made a 4.3% WER draw a 95.7% full bar.
 *
 * The `detail` variant reads out the pointed-at run in a fixed line above the bars instead of a
 * `title` tooltip. A native tooltip only appears after the browser's own hover delay -- around a
 * second, not adjustable from CSS or JS -- which is far too slow for scrubbing across a row of
 * thirty bars looking for the odd one out.
 */

interface MetricTrendProps {
  /** Points to draw, oldest first. */
  history: CellPoint[]
  /** Metric contract; supplies the extent and the colour scale. */
  semantics: MetricSemantics | null | undefined
  /** `inline` fits a table cell; `detail` is the expanded view with a readout and clickable bars. */
  variant?: 'inline' | 'detail'
  /** Called with the point a user clicks, in the `detail` variant. */
  onSelect?: (point: CellPoint) => void
  /** Accessible summary, since the shape itself carries the meaning. */
  label?: string
  className?: string
}

/** Bars never collapse to nothing: a floor keeps a zero-valued run visible as a run. */
const MIN_BAR_HEIGHT = 8

/** Beyond this many runs the inline variant shows only the most recent ones. */
const INLINE_MAX_BARS = 12

/**
 * Widest a single bar may grow in the expanded view.
 *
 * Bars share the width, so without a cap a history of three runs would draw three blocks hundreds of
 * pixels across -- which reads as a filled panel rather than as a trend. Capped, few runs stay a
 * short row of bars and many runs still spread across the panel.
 */
const DETAIL_MAX_BAR_WIDTH = 'max-w-6'

function formatShortTime(timestamp: string): string {
  return timestamp ? timestamp.replace('T', ' ').slice(5, 16) : ''
}

export default function MetricTrend({
  history,
  semantics,
  variant = 'inline',
  onSelect,
  label,
  className,
}: MetricTrendProps) {
  // Index of the bar being pointed at, in the `shown` slice. `null` falls back to the latest run,
  // so the readout is never blank and never needs a "hover a bar" instruction.
  const [hovered, setHovered] = useState<number | null>(null)

  const points = history.filter((point) => Number.isFinite(point.score))
  if (points.length === 0) {
    return null
  }

  // The inline variant keeps the newest runs, since the question it answers is "is this settling
  // down"; the detail variant shows everything.
  const shown = variant === 'inline' ? points.slice(-INLINE_MAX_BARS) : points
  const bounds = trendBounds(points.map((point) => point.score), semantics)
  const isDetail = variant === 'detail'
  const trackHeight = isDetail ? 'h-24' : 'h-[18px]'
  // A metric that declares a range whose series does not fit inside it is being reported on some
  // other scale, so its own display conversion does not apply either: 96.7749 is not "9677.5%".
  const offScale = semantics?.value_range != null && !bounds.declared
  const extentSemantics = offScale ? null : semantics

  const readoutIndex = hovered != null && hovered < shown.length ? hovered : shown.length - 1
  const readout = shown[readoutIndex]

  return (
    <div className={className}>
      {isDetail && (
        // Fixed position, so scrubbing the bars changes the text without anything moving or
        // covering the chart.
        <div className="mb-1.5 flex items-baseline gap-2 type-caption-mono">
          <span className="font-semibold tabular-nums text-[var(--text)]">
            {formatMetric(readout.score, extentSemantics).primary}
          </span>
          <span className="text-[var(--text-muted)]">{formatShortTime(readout.timestamp)}</span>
        </div>
      )}

      <div
        role="img"
        aria-label={label}
        onMouseLeave={isDetail ? () => setHovered(null) : undefined}
        className={`flex ${trackHeight} items-end gap-px ${isDetail ? 'gap-0.5' : ''}`}
      >
        {shown.map((point, index) => {
          const position = trendPosition(point.score, bounds)
          const quality = getBoundedQualityRatio(point.score, semantics)
          const color = quality == null ? 'var(--text-muted)' : scoreColor(quality)
          const isLatest = index === shown.length - 1
          const description = `${formatShortTime(point.timestamp)}  ${formatMetric(point.score, extentSemantics).primary}`
          const bar = (
            <span
              className="block w-full rounded-sm transition-[height] duration-300"
              style={{
                height: `calc(${MIN_BAR_HEIGHT}px + ${position * 100}% * ${(100 - MIN_BAR_HEIGHT) / 100})`,
                background: color,
                // The latest value is the one the adjacent number column shows; the earlier ones are
                // context, so they recede. The pointed-at bar comes forward to say which one the
                // readout above is describing.
                opacity: index === readoutIndex || isLatest ? 1 : 0.45,
              }}
            />
          )
          return onSelect && isDetail ? (
            <button
              key={`${point.timestamp}-${index}`}
              type="button"
              onClick={() => onSelect(point)}
              onMouseEnter={() => setHovered(index)}
              onFocus={() => setHovered(index)}
              aria-label={description}
              className={`flex h-full min-w-[6px] flex-1 ${DETAIL_MAX_BAR_WIDTH} items-end rounded-sm focus-visible:outline-2 focus-visible:outline-[var(--accent)] hover:opacity-80`}
            >
              {bar}
            </button>
          ) : (
            <span
              key={`${point.timestamp}-${index}`}
              // The inline sparkline has no room for a readout line, so it keeps the native tooltip.
              title={isDetail ? undefined : description}
              onMouseEnter={isDetail ? () => setHovered(index) : undefined}
              className={`flex h-full items-end ${isDetail ? `min-w-[6px] flex-1 ${DETAIL_MAX_BAR_WIDTH}` : 'w-[4px]'}`}
            >
              {bar}
            </span>
          )
        })}
      </div>

      {isDetail && (
        <div className="mt-1.5 flex items-baseline justify-between gap-2 type-caption-mono text-[var(--text-dim)]">
          <span>{formatShortTime(shown[0].timestamp)}</span>
          {/* Whether heights mean an absolute position or only a position within this series is not
              something the reader can infer from the bars, so it is stated. */}
          {!bounds.declared && (
            <span className="text-[10px]">
              {`${formatMetric(bounds.low, extentSemantics).primary} – ${formatMetric(bounds.high, extentSemantics).primary}`}
            </span>
          )}
          {shown.length > 1 && <span>{formatShortTime(shown[shown.length - 1].timestamp)}</span>}
        </div>
      )}
    </div>
  )
}
