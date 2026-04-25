import { useMemo } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import type { ReportData } from '@/api/types'
import Card from '@/components/ui/Card'
import { scoreColor } from '@/components/ui/Table'

interface Props {
  reports: ReportData[]
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

  const formatVal = (s: number) => (s > 1 ? s.toFixed(1) : (s * 100).toFixed(1) + '%')

  const cards = [
    {
      label: t('reportDetail.avgScore'),
      value: formatVal(stats.avg),
      score: stats.avg > 1 ? stats.avg / 100 : stats.avg,
    },
    {
      label: t('reportDetail.bestDataset'),
      value: formatVal(stats.best.score),
      sub: stats.best.name,
      score: stats.best.score > 1 ? stats.best.score / 100 : stats.best.score,
    },
    {
      label: t('reportDetail.worstDataset'),
      value: formatVal(stats.worst.score),
      sub: stats.worst.name,
      score: stats.worst.score > 1 ? stats.worst.score / 100 : stats.worst.score,
    },
    {
      label: t('reportDetail.totalSamples'),
      value: stats.totalSamples.toLocaleString(),
    },
  ]

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card, i) => (
        <Card key={i}>
          <div className="flex flex-col gap-1">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-dim)]">
              {card.label}
            </span>
            <span
              className="text-2xl font-bold font-mono"
              style={card.score !== undefined ? { color: scoreColor(card.score) } : { color: 'var(--accent)' }}
            >
              {card.value}
            </span>
            {card.sub && (
              <span className="text-xs text-[var(--text-muted)] truncate" title={card.sub}>
                {card.sub}
              </span>
            )}
            {card.score !== undefined && (
              <div
                className="mt-1 h-1 rounded-full overflow-hidden"
                style={{ background: 'var(--bg-deep)' }}
              >
                <div
                  className="h-full rounded-full transition-all duration-700"
                  style={{
                    width: `${Math.min(100, card.score * 100)}%`,
                    background: scoreColor(card.score),
                  }}
                />
              </div>
            )}
          </div>
        </Card>
      ))}
    </div>
  )
}
