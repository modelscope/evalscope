import { cn } from '@/lib/utils'
import { scoreColor, scoreBg } from '@/utils/colorScale'

interface ScoreChipProps {
  score: number
  label?: string
  /** Format the score: 'percent' renders `87.3`, 'raw' renders `0.873`, 'none' suppresses it. */
  format?: 'percent' | 'raw' | 'none'
  className?: string
}

/**
 * Dynamic-score pill — DESIGN.md `{components.score-chip}`.
 * Small caption-mono chip with HSL-computed fg/bg. Use in chip rows (dataset chips, leaderboard).
 * For the larger body-sm bold percentage badge atop an eval row, use `ScoreBadge`.
 */
export default function ScoreChip({ score, label, format = 'percent', className }: ScoreChipProps) {
  const fg = scoreColor(score)
  const bg = scoreBg(score)
  const text =
    format === 'percent' ? (score * 100).toFixed(1) : format === 'raw' ? score.toFixed(3) : ''

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded-full type-caption-mono whitespace-nowrap',
        className,
      )}
      style={{ background: bg, color: fg }}
    >
      {label && <span>{label}</span>}
      {text && <span>{text}</span>}
    </span>
  )
}
