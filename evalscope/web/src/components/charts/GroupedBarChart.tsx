import { useMemo } from 'react'
import { useLocale } from '@/contexts/LocaleContext'

interface Props {
  /** datasets (x-axis groups) */
  datasets: string[]
  /** models with their per-dataset scores */
  models: { name: string; scores: Record<string, number> }[]
  height?: number
}

// Distinct color palette for models (accessible, dark-friendly)
const MODEL_COLORS = [
  '#6366f1', // indigo
  '#10b981', // emerald
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // violet
  '#06b6d4', // cyan
  '#f97316', // orange
  '#ec4899', // pink
  '#14b8a6', // teal
  '#a3e635', // lime
]

function getModelColor(i: number): string {
  return MODEL_COLORS[i % MODEL_COLORS.length]
}

export default function GroupedBarChart({ datasets, models, height = 280 }: Props) {
  const { t } = useLocale()
  const svgW = 640
  const svgH = height
  const padL = 40
  const padR = 16
  const padT = 12
  const padB = 80 // space for angled labels + legend
  const chartW = svgW - padL - padR
  const chartH = svgH - padT - padB

  const { maxScore, barW, xOffsets } = useMemo(() => {
    const nModels = models.length
    const nDatasets = datasets.length
    if (!nDatasets || !nModels) return { maxScore: 1, groupW: 0, barW: 0, xOffsets: [] }

    const allScores = models.flatMap((m) => Object.values(m.scores))
    const maxScore = Math.max(...allScores, 0.001)

    const groupW = chartW / nDatasets
    const barPad = groupW * 0.15
    const barW = (groupW - barPad * 2) / nModels

    // x offset of first bar in each group
    const xOffsets = datasets.map((_, gi) => padL + gi * groupW + groupW * 0.15)

    return { maxScore, groupW, barW, xOffsets }
  }, [datasets, models, chartW])

  if (!datasets.length || !models.length) return null

  // Y-axis ticks
  const yTicks = 5
  const isPercent = maxScore <= 1
  const yTickValues = Array.from({ length: yTicks + 1 }, (_, i) => (i / yTicks) * maxScore)

  return (
    <div className="flex flex-col gap-3">
      {/* Legend */}
      <div className="flex flex-wrap gap-3">
        {models.map((m, i) => (
          <div key={m.name} className="flex items-center gap-1.5">
            <span
              className="inline-block w-3 h-3 rounded-sm flex-shrink-0"
              style={{ background: getModelColor(i) }}
            />
            <span className="text-[11px] text-[var(--color-ink-muted)] truncate max-w-[160px]" title={m.name}>
              {m.name}
            </span>
          </div>
        ))}
      </div>

      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
        style={{ maxWidth: '100%', overflow: 'visible' }}
        aria-label={t('charts.groupedBarAriaLabel')}
      >
        {/* Y grid lines */}
        {yTickValues.map((val, i) => {
          const y = padT + chartH - (val / maxScore) * chartH
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
              <text
                x={padL - 4}
                y={y}
                textAnchor="end"
                dominantBaseline="middle"
                fontSize="9"
                fill="var(--color-ink-faint)"
              >
                {isPercent ? (val * 100).toFixed(0) : val.toFixed(1)}
              </text>
            </g>
          )
        })}

        {/* Grouped bars */}
        {datasets.map((ds, gi) => {
          const x0 = xOffsets[gi]
          return (
            <g key={ds}>
              {models.map((m, mi) => {
                const score = m.scores[ds] ?? 0
                const norm = maxScore > 0 ? score / maxScore : 0
                const bh = norm * chartH
                const bx = x0 + mi * barW
                const by = padT + chartH - bh
                const color = getModelColor(mi)
                const displayScore = isPercent ? (score * 100).toFixed(1) + '%' : score.toFixed(2)

                return (
                  <g key={m.name}>
                    <rect
                      x={bx}
                      y={by}
                      width={Math.max(barW - 1, 1)}
                      height={Math.max(bh, score > 0 ? 2 : 0)}
                      rx="2"
                      fill={color}
                      opacity="0.85"
                    >
                      <title>{`${m.name} / ${ds}: ${displayScore}`}</title>
                    </rect>
                  </g>
                )
              })}

              {/* Group (dataset) label — angled for compactness */}
              <text
                x={x0 + (barW * models.length) / 2 - barW / 2}
                y={padT + chartH + 14}
                textAnchor="end"
                fontSize="10"
                fill="var(--color-ink-muted)"
                transform={`rotate(-35, ${x0 + (barW * models.length) / 2 - barW / 2}, ${padT + chartH + 14})`}
              >
                <title>{ds}</title>
                {ds.length > 16 ? ds.slice(0, 15) + '…' : ds}
              </text>
            </g>
          )
        })}

        {/* Axes */}
        <line x1={padL} y1={padT} x2={padL} y2={padT + chartH} stroke="var(--color-border)" strokeWidth="1" />
        <line x1={padL} y1={padT + chartH} x2={padL + chartW} y2={padT + chartH} stroke="var(--color-border)" strokeWidth="1" />
      </svg>
    </div>
  )
}
