import { ChevronDown, ChevronRight } from 'lucide-react'
import { scoreColor } from '@/utils/colorScale'

interface ModelGroupHeaderProps {
  /** Title text (model name or dataset name) */
  title: string
  /** Run count shown after the title */
  count: number
  /** "runs" label (i18n-translated by caller) */
  runsLabel: string
  /** Best score across the group (0-1) */
  bestScore: number
  /** "Best score" label (i18n-translated by caller) */
  bestScoreLabel: string
  expanded: boolean
  onToggle: () => void
}

/**
 * ModelGroupHeader — collapsible row-card header for the dashboard grouped-view.
 * DESIGN.md `{ex-model-group-header}`: row-card chrome with chevron, title, count, and
 * a best-score callout in dynamic HSL color. Click toggles expansion.
 */
export default function ModelGroupHeader({
  title,
  count,
  runsLabel,
  bestScore,
  bestScoreLabel,
  expanded,
  onToggle,
}: ModelGroupHeaderProps) {
  return (
    <button
      onClick={onToggle}
      className="flex items-center gap-2 w-full px-4 py-3 hover:bg-[var(--bg-card2)] transition-colors text-left"
    >
      {expanded ? (
        <ChevronDown size={14} className="text-[var(--text-muted)] shrink-0" />
      ) : (
        <ChevronRight size={14} className="text-[var(--text-muted)] shrink-0" />
      )}
      <span className="type-title-md text-[var(--text)]">{title}</span>
      <span className="type-caption-mono text-[var(--text-muted)]">
        ({count} {runsLabel})
      </span>
      <span
        className="ml-auto type-body-sm font-mono tabular-nums"
        style={{ color: scoreColor(bestScore) }}
      >
        {bestScoreLabel}: {(bestScore * 100).toFixed(1)}%
      </span>
    </button>
  )
}
