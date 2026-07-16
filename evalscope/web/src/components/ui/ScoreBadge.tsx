import { cn } from '@/lib/utils'
import { scoreColor, scoreBg } from '@/utils/colorScale'
import { useLocale } from '@/contexts/LocaleContext'
import { formatMetricByKey } from '@/domain/metric/registry'

interface ScoreBadgeProps {
  score: number
  /** When provided, score is rendered as `score >= threshold ? PASS : FAIL` with the deep pass/fail color. */
  threshold?: number
  /** Override the displayed text. Defaults come from the metric display contract. */
  label?: string
  className?: string
}

/**
 * Bold percentage / boolean pill — DESIGN.md `{components.score-badge}`.
 * Larger and heavier than `ScoreChip`: body-sm bold + 10/2 padding.
 * Two modes:
 *   - no threshold: dynamic HSL fg/bg, renders the canonical percentage.
 *   - threshold: deep `--pass` / `--fail` background with white text, renders the canonical raw score.
 */
export default function ScoreBadge({ score, threshold, label, className }: ScoreBadgeProps) {
  const { t } = useLocale()
  const formatted = formatMetricByKey('score', score, t)
  if (threshold !== undefined) {
    const pass = score >= threshold
    const text = label ?? formatted.raw
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
  const text = label ?? formatted.primary
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
