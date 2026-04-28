import { useMemo } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import type { ReportData } from '@/api/types'
import { scoreColor } from '@/components/ui/Table'

interface Props {
  reports: ReportData[]
}

/** SVG circular progress ring */
function ScoreRing({ score, size = 80 }: { score: number; size?: number }) {
  const r = (size - 8) / 2
  const circ = 2 * Math.PI * r
  const offset = circ * (1 - Math.min(1, score))
  const color = scoreColor(score)
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ transform: 'rotate(-90deg)', flexShrink: 0 }}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--border)" strokeWidth="5" />
      <circle
        cx={size / 2} cy={size / 2} r={r} fill="none"
        stroke={color}
        strokeWidth="5"
        strokeDasharray={circ}
        strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: 'stroke-dashoffset 0.7s ease' }}
      />
    </svg>
  )
}

export default function ReportSummaryStats({ reports }: Props) {
  const { t } = useLocale()

  const stats = useMemo(() => {
    if (!reports.length) return null

    const scores = reports.map((r) => r.score)
    const avg = scores.reduce((s, v) => s + v, 0) / scores.length

    const bestIdx = scores.indexOf(Math.max(...scores))
    const worstIdx = scores.indexOf(Math.min(...scores))

    const totalSamples = reports.reduce((sum, r) => {
      return sum + (r.metrics[0]?.categories?.reduce((s, c) => s + c.num, 0) ?? 0)
    }, 0)

    return {
      avg,
      best: { name: reports[bestIdx].dataset_name, score: scores[bestIdx] },
      worst: { name: reports[worstIdx].dataset_name, score: scores[worstIdx] },
      totalSamples,
    }
  }, [reports])

  if (!stats) return null

  const toNorm = (s: number) => (s > 1 ? s / 100 : s)
  const formatPct = (s: number) => (s > 1 ? s.toFixed(1) : (s * 100).toFixed(1)) + '%'

  const scoreCards = [
    { label: t('reportDetail.avgScore'), norm: toNorm(stats.avg), pct: formatPct(stats.avg) },
    { label: t('reportDetail.bestDataset'), norm: toNorm(stats.best.score), pct: formatPct(stats.best.score), sub: stats.best.name },
    { label: t('reportDetail.worstDataset'), norm: toNorm(stats.worst.score), pct: formatPct(stats.worst.score), sub: stats.worst.name },
  ]

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Score ring cards */}
      {scoreCards.map((card, i) => (
        <div
          key={i}
          className="flex items-center gap-3 p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]"
        >
          <ScoreRing score={card.norm} size={72} />
          <div className="flex flex-col gap-0.5 min-w-0">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-dim)]">
              {card.label}
            </span>
            <span
              className="text-xl font-bold font-mono tabular-nums leading-tight"
              style={{ color: scoreColor(card.norm) }}
            >
              {card.pct}
            </span>
            {card.sub && (
              <span className="text-xs text-[var(--text-muted)] truncate" title={card.sub}>
                {card.sub}
              </span>
            )}
          </div>
        </div>
      ))}

      {/* Total Samples — plain number card */}
      <div className="flex flex-col justify-center gap-1 p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-dim)]">
          {t('reportDetail.totalSamples')}
        </span>
        <span className="text-2xl font-bold font-mono tabular-nums" style={{ color: 'var(--accent)' }}>
          {stats.totalSamples.toLocaleString()}
        </span>
      </div>
    </div>
  )
}
