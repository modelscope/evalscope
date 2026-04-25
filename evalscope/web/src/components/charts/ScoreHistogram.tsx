import { useMemo } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { rdYlGn } from '@/utils/colorScale'
import type { PredictionRow } from '@/api/types'

interface Props {
  predictions: PredictionRow[]
  bins?: number
}

const DEFAULT_BINS = 10

export default function ScoreHistogram({ predictions, bins = DEFAULT_BINS }: Props) {
  const { t } = useLocale()

  const { buckets, maxCount } = useMemo(() => {
    if (!predictions.length) return { buckets: [], maxCount: 0 }

    const step = 1 / bins
    const counts = Array(bins).fill(0) as number[]

    for (const p of predictions) {
      const idx = Math.min(Math.floor(p.NScore / step), bins - 1)
      counts[idx]++
    }

    const buckets = counts.map((count, i) => ({
      start: i * step,
      end: (i + 1) * step,
      count,
      midpoint: (i + 0.5) * step,
    }))

    return { buckets, maxCount: Math.max(...counts, 1) }
  }, [predictions, bins])

  if (!predictions.length) {
    return (
      <div className="flex items-center justify-center h-24 text-[var(--color-ink-faint)] text-sm">
        {t('single.noScoreData')}
      </div>
    )
  }

  // SVG dimensions
  const svgW = 480
  const svgH = 180
  const padL = 36
  const padR = 16
  const padT = 10
  const padB = 36
  const chartW = svgW - padL - padR
  const chartH = svgH - padT - padB
  const barGap = 2
  const barW = (chartW - barGap * (bins - 1)) / bins

  // Y-axis ticks
  const yTicks = 4
  const yTickValues = Array.from({ length: yTicks + 1 }, (_, i) => Math.round((i / yTicks) * maxCount))

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs text-[var(--color-ink-muted)] font-medium">{t('single.scoreDistribution')}</span>
        <span className="text-[10px] text-[var(--color-ink-faint)]">
          {predictions.length} {t('single.samples')}
        </span>
      </div>
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        style={{ maxWidth: '100%', overflow: 'visible' }}
        aria-label={t('charts.scoreHistogramAriaLabel')}
      >
        {/* Y grid lines + labels */}
        {yTickValues.map((val, i) => {
          const y = padT + chartH - (val / maxCount) * chartH
          return (
            <g key={i}>
              <line
                x1={padL}
                y1={y}
                x2={padL + chartW}
                y2={y}
                stroke="var(--color-border)"
                strokeWidth="1"
                strokeDasharray={i === 0 ? 'none' : '3 3'}
                opacity="0.5"
              />
              <text x={padL - 4} y={y} textAnchor="end" dominantBaseline="middle" fontSize="9" fill="var(--color-ink-faint)">
                {val}
              </text>
            </g>
          )
        })}

        {/* Bars */}
        {buckets.map((b, i) => {
          const barH = (b.count / maxCount) * chartH
          const x = padL + i * (barW + barGap)
          const y = padT + chartH - barH
          const color = rdYlGn(b.midpoint)
          const pct = ((b.count / predictions.length) * 100).toFixed(1)

          return (
            <g key={i}>
              <rect
                x={x}
                y={y}
                width={barW}
                height={Math.max(barH, b.count > 0 ? 2 : 0)}
                rx="2"
                fill={color}
                opacity="0.85"
              >
                <title>{`${(b.start * 100).toFixed(0)}–${(b.end * 100).toFixed(0)}%: ${b.count} (${pct}%)`}</title>
              </rect>
              {/* X label: only first, middle, last */}
              {(i === 0 || i === bins - 1 || i === Math.floor(bins / 2)) && (
                <text
                  x={x + barW / 2}
                  y={padT + chartH + 14}
                  textAnchor="middle"
                  fontSize="9"
                  fill="var(--color-ink-faint)"
                >
                  {(b.start * 100).toFixed(0)}
                </text>
              )}
            </g>
          )
        })}

        {/* Axes */}
        <line x1={padL} y1={padT} x2={padL} y2={padT + chartH} stroke="var(--color-border)" strokeWidth="1" />
        <line x1={padL} y1={padT + chartH} x2={padL + chartW} y2={padT + chartH} stroke="var(--color-border)" strokeWidth="1" />

        {/* X-axis title */}
        <text
          x={padL + chartW / 2}
          y={svgH - 2}
          textAnchor="middle"
          fontSize="9"
          fill="var(--color-ink-faint)"
        >
          {t('charts.scoreAxisLabel')}
        </text>
      </svg>
    </div>
  )
}
