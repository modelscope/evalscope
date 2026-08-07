import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface KpiCardProps {
  icon: ReactNode
  value: string
  label: string
  /** Stagger-delay in ms — feeds into the fadeInUp animation via inline style (legitimate dynamic value). */
  delay?: number
  onClick?: () => void
  className?: string
}

/**
 * Dashboard hero KPI tile — DESIGN.md `{components.kpi-card}`.
 * Uses the `.kpi-card` CSS class for hover lift + violet wash via `::before`.
 * Renders display-xl value + eyebrow label + a single-hue accent icon tile.
 *
 * The icon tile carries no per-card hue: the four counters are the same kind of quantity, so a
 * distinct hue per tile would assert a distinction that does not exist. See DESIGN.md
 * `#### KPI Icon Tile`.
 */
export default function KpiCard({
  icon,
  value,
  label,
  delay = 0,
  onClick,
  className,
}: KpiCardProps) {
  return (
    <div
      className={cn(
        'kpi-card group bg-[var(--bg-card)] p-5',
        onClick && 'cursor-pointer',
        className,
      )}
      onClick={onClick}
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-start justify-between mb-3 relative">
        <div className="w-10 h-10 rounded-[var(--radius)] flex items-center justify-center bg-[var(--accent-dim)] text-[var(--accent)]">
          {icon}
        </div>
      </div>
      <div className="type-display-xl text-[var(--text)] relative break-words">{value}</div>
      <div className="type-body-xs text-[var(--text-muted)] mt-0.5 font-medium relative">{label}</div>
    </div>
  )
}
