import { useMemo } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import type { ReportData } from '@/api/types'
import { rdYlGn } from '@/utils/colorScale'

interface Props {
  reports: ReportData[]
}

interface StatCard {
  label: string
  value: string
  sub?: string
  gradient: string
  iconPath: string
  score?: number
}

export default function SummaryStats({ reports }: Props) {
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

  const formatScore = (s: number) => {
    if (s > 1) return s.toFixed(1)
    return (s * 100).toFixed(1) + '%'
  }

  const cards: StatCard[] = [
    {
      label: t('single.avgScore'),
      value: formatScore(stats.avg),
      gradient: 'var(--gradient-primary)',
      score: stats.avg > 1 ? stats.avg / 100 : stats.avg,
      iconPath: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
    },
    {
      label: t('single.bestDataset'),
      value: formatScore(stats.best.score),
      sub: stats.best.name,
      gradient: 'var(--gradient-accent)',
      score: stats.best.score > 1 ? stats.best.score / 100 : stats.best.score,
      iconPath: 'M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z',
    },
    {
      label: t('single.weakestDataset'),
      value: formatScore(stats.worst.score),
      sub: stats.worst.name,
      gradient: 'var(--gradient-warm)',
      score: stats.worst.score > 1 ? stats.worst.score / 100 : stats.worst.score,
      iconPath: 'M13 17h8m0 0V9m0 8l-8-8-4 4-6-6',
    },
    {
      label: t('single.totalSamples'),
      value: stats.totalSamples.toLocaleString(),
      gradient: 'linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%)',
      iconPath: 'M4 6h16M4 10h16M4 14h16M4 18h16',
    },
  ]

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 stagger-children">
      {cards.map((card, i) => (
        <div key={i} className="kpi-card p-4" style={{ background: 'var(--color-surface)' }}>
          {/* Icon + gradient accent bar */}
          <div className="flex items-start justify-between mb-3">
            <div
              className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ background: card.gradient + '22', border: `1px solid ${card.gradient.includes('accent') ? 'var(--color-accent-muted)' : 'var(--color-primary-muted)'}` }}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="w-4 h-4"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.8}
                stroke="currentColor"
                style={{ color: 'var(--color-primary)' }}
              >
                <path strokeLinecap="round" strokeLinejoin="round" d={card.iconPath} />
              </svg>
            </div>
            {card.score !== undefined && (
              <div
                className="w-2 h-2 rounded-full flex-shrink-0 mt-1"
                style={{ background: rdYlGn(card.score), boxShadow: `0 0 6px ${rdYlGn(card.score)}88` }}
              />
            )}
          </div>

          {/* Value */}
          <div
            className="text-2xl font-bold mb-0.5"
            style={{ background: card.gradient, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}
          >
            {card.value}
          </div>

          {/* Label */}
          <div className="text-xs font-medium text-[var(--color-ink-muted)]">{card.label}</div>

          {/* Sub label (dataset name) */}
          {card.sub && (
            <div className="mt-1 text-[10px] text-[var(--color-ink-faint)] truncate" title={card.sub}>
              {card.sub}
            </div>
          )}

          {/* Score bar */}
          {card.score !== undefined && (
            <div className="mt-3 h-1 rounded-full bg-[var(--color-border)] overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{ width: `${Math.min(100, card.score * 100)}%`, background: rdYlGn(card.score) }}
              />
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
