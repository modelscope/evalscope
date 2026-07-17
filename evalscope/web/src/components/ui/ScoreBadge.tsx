import { cn } from '@/lib/utils'
import { scoreColor, scoreBg } from '@/utils/colorScale'
import { useLocale } from '@/contexts/LocaleContext'
import { formatMetricByKey } from '@/domain/metric/registry'

interface ScoreBadgeProps {
  score: number
  /** When provided, use the threshold's semantic success/danger treatment. */
  threshold?: number
  /** Override the displayed text. Defaults come from the metric display contract. */
  label?: string
  className?: string
}

/**
 * Bold percentage / boolean pill — DESIGN.md `{components.score-badge}`.
 * Uses body-sm bold text with compact pill padding.
 * Two modes:
 *   - no threshold: dynamic HSL fg/bg, renders the canonical percentage.
 *   - threshold: restrained semantic background/border, renders the canonical percentage.
 */
export default function ScoreBadge({ score, threshold, label, className }: ScoreBadgeProps) {
  const { t } = useLocale()
  const formatted = formatMetricByKey('score', score, t)
  if (threshold !== undefined) {
    const pass = score >= threshold
    const text = label ?? formatted.primary
    return (
      <span
        className={cn(
          'inline-block px-2.5 py-0.5 rounded-full border text-sm font-bold tabular-nums',
          className,
        )}
        style={{
          backgroundColor: pass ? 'var(--success-bg)' : 'var(--danger-bg)',
          borderColor: pass ? 'var(--success-border)' : 'var(--danger-border)',
          color: pass ? 'var(--success)' : 'var(--danger)',
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
