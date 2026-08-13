import { formatDifference, formatMetric, formatMetricIdentityLabel } from '@/domain/metric'
import type { TrendSeries } from '@/domain/report/trendSeries'
import { useLocale } from '@/contexts/LocaleContext'

interface ScopedTrendProps {
  series: TrendSeries
}

const WIDTH = 1000
const HEIGHT = 230
const PADDING = { top: 18, right: 22, bottom: 34, left: 76 }

/** One primary-metric history, already scoped to an exact model and benchmark. */
export default function ScopedTrend({ series }: ScopedTrendProps) {
  const { t } = useLocale()
  const { points, semantics } = series
  const scores = points.map((point) => point.score)
  const bounds = chartBounds(scores, semantics.value_range)
  const plotWidth = WIDTH - PADDING.left - PADDING.right
  const plotHeight = HEIGHT - PADDING.top - PADDING.bottom
  const xValues = timePositions(points.map((point) => point.timestamp))
  const x = (position: number) => PADDING.left + position * plotWidth
  const y = (score: number) => PADDING.top + ((bounds.high - score) / (bounds.high - bounds.low)) * plotHeight
  const path = points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${x(xValues[index])} ${y(point.score)}`).join(' ')
  const latest = points[points.length - 1]
  const previous = points[points.length - 2]
  const delta = latest.score - previous.score
  const deltaText = formatDifference(delta, semantics).primary
  const yTicks = [bounds.high, (bounds.high + bounds.low) / 2, bounds.low]

  return (
    <div>
      <div className="mb-3 flex items-end justify-between gap-4">
        <div className="min-w-0">
          <div className="type-label-xs text-[var(--text-muted)]">
            {formatMetricIdentityLabel(series.identity, semantics)}
          </div>
          <div className="mt-1 type-title-md tabular-nums text-[var(--text)]">
            {formatMetric(latest.score, semantics).primary}
          </div>
        </div>
        <div className="shrink-0 text-right type-caption-mono text-[var(--text-muted)]">
          <div>{t('dashboard.trendRunCount', { count: points.length })}</div>
          <div>{t('dashboard.trendPrevious', { delta: `${delta > 0 ? '+' : ''}${deltaText}` })}</div>
        </div>
      </div>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`${series.modelLabel} ${series.benchmarkLabel} ${semantics.metric_name} trend`}
        className="h-[230px] w-full overflow-visible"
      >
        {yTicks.map((tick) => {
          const position = y(tick)
          return (
            <g key={tick}>
              <line
                x1={PADDING.left}
                x2={WIDTH - PADDING.right}
                y1={position}
                y2={position}
                stroke="var(--border)"
                strokeDasharray="4 6"
              />
              <text
                x={PADDING.left - 12}
                y={position + 4}
                textAnchor="end"
                fill="var(--text-muted)"
                fontSize="12"
              >
                {formatMetric(tick, semantics).primary}
              </text>
            </g>
          )
        })}
        <path d={path} fill="none" stroke="var(--accent)" strokeWidth="3" strokeLinejoin="round" />
        {points.map((point, index) => (
          <circle
            key={`${point.runId}-${point.timestamp}`}
            cx={x(xValues[index])}
            cy={y(point.score)}
            r={index === points.length - 1 ? 6 : 4}
            fill="var(--bg-card)"
            stroke="var(--accent)"
            strokeWidth="3"
          >
            <title>{`${shortTimestamp(point.timestamp)} · ${formatMetric(point.score, semantics).primary}`}</title>
          </circle>
        ))}
        <text x={PADDING.left} y={HEIGHT - 8} fill="var(--text-muted)" fontSize="12">
          {shortTimestamp(points[0].timestamp)}
        </text>
        <text
          x={WIDTH - PADDING.right}
          y={HEIGHT - 8}
          textAnchor="end"
          fill="var(--text-muted)"
          fontSize="12"
        >
          {shortTimestamp(latest.timestamp)}
        </text>
      </svg>
    </div>
  )
}

function chartBounds(scores: number[], declared?: { min: number; max: number } | null) {
  if (declared && scores.every((score) => score >= declared.min && score <= declared.max)) {
    return { low: declared.min, high: declared.max }
  }
  const low = Math.min(...scores)
  const high = Math.max(...scores)
  if (low !== high) {
    const padding = (high - low) * 0.1
    return { low: low - padding, high: high + padding }
  }
  const padding = Math.abs(low) * 0.1 || 1
  return { low: low - padding, high: high + padding }
}

function timePositions(timestamps: string[]): number[] {
  const values = timestamps.map((timestamp) => Date.parse(timestamp))
  const low = Math.min(...values)
  const high = Math.max(...values)
  if (values.every(Number.isFinite) && low !== high) {
    return values.map((value) => (value - low) / (high - low))
  }
  return values.map((_, index) => index / Math.max(1, values.length - 1))
}

function shortTimestamp(timestamp: string): string {
  return timestamp.replace('T', ' ').slice(5, 16)
}
