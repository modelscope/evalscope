import { cn } from '@/lib/utils'
import { scoreColor, scoreBg } from '@/utils/colorScale'

interface ScoreBadgeProps {
  score: number
  /** When provided, score is rendered as `score >= threshold ? PASS : FAIL` with the deep pass/fail color. */
  threshold?: number
  /** Override the displayed text. Defaults: percent if no threshold, `score.toFixed(4)` if threshold. */
  label?: string
  className?: string
}

/**
 * Bold percentage / boolean pill — DESIGN.md `{components.score-badge}`.
 * Larger and heavier than `ScoreChip`: body-sm bold + 10/2 padding.
 * Two modes:
 *   - no threshold: dynamic HSL fg/bg, renders `(score*100).toFixed(1)%`.
 *   - threshold:    deep `--pass` / `--fail` background with white text, renders raw score.
 */
export default function ScoreBadge({ score, threshold, label, className }: ScoreBadgeProps) {
  if (threshold !== undefined) {
    const pass = score >= threshold
    const text = label ?? score.toFixed(4)
    return (
      <span
        className={cn(
          'inline-block px-2.5 py-0.5 rounded-full text-sm font-bold tabular-nums',
          className,
        )}
        style={{
          backgroundColor: pass ? 'var(--pass)' : 'var(--fail)',
          color: 'var(--text-on-filled)',
        }}
      >
        {text}
      </span>
    )
  }

  const fg = scoreColor(score)
  const bg = scoreBg(score)
  const text = label ?? `${(score * 100).toFixed(1)}%`
  return (
    <span
      className={cn(
        'inline-block px-2.5 py-0.5 rounded-full text-sm font-bold tabular-nums',
        className,
      )}
      style={{ background: bg, color: fg }}
    >
      {text}
    </span>
  )
}
