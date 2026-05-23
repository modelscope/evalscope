import { cn } from '@/lib/utils'
import { scoreColor } from '@/utils/colorScale'

interface ScoreChipProps {
  score: number
  label?: string
  /** Format the score: 'percent' renders `87.3`, 'raw' renders `0.873`, 'none' suppresses it. */
  format?: 'percent' | 'raw' | 'none'
  className?: string
}

/**
 * Dynamic-score pill — DESIGN.md `{components.score-chip}`.
 * Outline chip: transparent bg + 1px scoreColor border + scoreColor text.
 * Avoids the high-luminance yellow-fill problem at hue ≈ 60 and reads cleanly
 * on both warm-cream and near-black canvases. Hue still carries the data; the
 * chip just stops shouting it.
 * For the larger body-sm bold percentage badge atop an eval row, use `ScoreBadge`.
 */
export default function ScoreChip({ score, label, format = 'percent', className }: ScoreChipProps) {
  const fg = scoreColor(score)
  const text =
    format === 'percent' ? (score * 100).toFixed(1) : format === 'raw' ? score.toFixed(3) : ''

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded-full type-caption-mono whitespace-nowrap border',
        className,
      )}
      style={{ borderColor: fg, color: fg }}
    >
      {label && <span>{label}</span>}
      {text && <span>{text}</span>}
    </span>
  )
}
