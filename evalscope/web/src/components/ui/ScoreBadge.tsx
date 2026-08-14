import { cn } from '@/lib/utils'
import { scoreColor, scoreBg } from '@/utils/colorScale'
import { formatMetric, getBoundedQualityRatio } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'

interface ScoreBadgeProps {
  score: number
  /** Backend semantics of the metric. Decides the formatting and whether a colour scale applies. */
  semantics?: MetricSemantics | null
  /** When provided, use the threshold's semantic success/danger treatment. */
  threshold?: number
  /** Override the displayed text. Defaults to the formatted value. */
  label?: string
  className?: string
}

/**
 * Bold percentage / boolean pill — DESIGN.md `{components.score-badge}`.
 * Uses body-sm bold text with compact pill padding.
 *
 * Three modes:
 *   - threshold: restrained semantic background/border.
 *   - bounded quality metric: dynamic HSL fg/bg driven by the normalized ratio, so "fuller is
 *     better" holds for a low-is-better metric too.
 *   - anything else (latency, throughput, counts, unresolved semantics): no colour scale, since
 *     colouring those would imply a verdict the metric does not carry.
 */
export default function ScoreBadge({ score, semantics, threshold, label, className }: ScoreBadgeProps) {
  const formatted = formatMetric(score, semantics)
  const text = label ?? formatted.primary

  if (threshold !== undefined) {
    const pass = score >= threshold
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

  const ratio = getBoundedQualityRatio(score, semantics)
  if (ratio === null) {
    return (
      <span
        className={cn('inline-block px-2.5 py-0.5 rounded-full text-sm font-bold tabular-nums', className)}
      >
        {text}
      </span>
    )
  }

  return (
    <span
      className={cn(
        'inline-block px-2.5 py-0.5 rounded-full text-sm font-bold tabular-nums',
        className,
      )}
      style={{ background: scoreBg(ratio), color: scoreColor(ratio) }}
    >
      {text}
    </span>
  )
}
