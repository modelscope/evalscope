import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

/** One counter in a KPI strip. */
export interface KpiItem {
  /** Caption under the value. */
  label: string
  /** Pre-formatted value — formatting stays with the caller's metric layer. */
  value: string
  /** Accent colour for the value; defaults to the plain text colour. */
  color?: string
  /** Leading icon tile. Rendered by the `hero` layout only. */
  icon?: ReactNode
  /** Full text for the native tooltip when the displayed value is shortened. */
  title?: string
  /** Makes the tile a button. Display-only tiles stay non-interactive. */
  onClick?: () => void
}

/**
 * Visual density of the strip:
 * - `hero`     — landing-page counters: 2-up grid, 4-up from `lg`, 20-px padding, icon tile.
 * - `dense`    — metric overview inside a panel: 2/3/4-up grid of hairline-separated cells.
 * - `inline`   — identity/config strip: one flexible row that wraps, no icons.
 */
export type KpiStripLayout = 'hero' | 'dense' | 'inline'

/**
 * Container chrome of the `hero` layout, exported so a loading skeleton can
 * occupy the exact same box as the resolved strip.
 */
export const KPI_HERO_CONTAINER =
  'grid grid-cols-2 overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] lg:grid-cols-4'

/** Padding of a `hero` cell, exported for the same skeleton alignment reason. */
export const KPI_HERO_CELL = 'border-[var(--border)] p-5 lg:border-r lg:last:border-r-0'

const CONTAINER: Record<KpiStripLayout, string> = {
  hero: cn(KPI_HERO_CONTAINER, 'shadow-[var(--shadow-sm)]'),
  dense:
    'grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-4 gap-px overflow-hidden rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--border)]',
  inline:
    'flex flex-wrap overflow-hidden rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-card)]',
}

interface KpiStripProps {
  items: KpiItem[]
  layout?: KpiStripLayout
  className?: string
}

function HeroCell({ item }: { item: KpiItem }) {
  const content = (
    <>
      {item.icon && (
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[var(--radius-sm)] bg-[var(--accent-dim)] text-[var(--accent)]">
          {item.icon}
        </span>
      )}
      <span className="min-w-0">
        <span className="type-title-md block truncate text-[var(--text)]" title={item.title}>
          {item.value}
        </span>
        <span className="block truncate type-body-xs text-[var(--text-muted)]">{item.label}</span>
      </span>
    </>
  )

  const className =
    'flex min-w-0 items-center gap-3 border-b border-r border-[var(--border)] p-5 text-left transition-colors even:border-r-0 lg:border-b-0 lg:even:border-r lg:last:border-r-0'

  if (item.onClick) {
    return (
      <button type="button" onClick={item.onClick} className={cn(className, 'hover:bg-[var(--bg-card2)]')}>
        {content}
      </button>
    )
  }
  return <div className={className}>{content}</div>
}

function DenseCell({ item }: { item: KpiItem }) {
  return (
    <div className="min-w-0 bg-[var(--bg-card)] px-3 py-2.5">
      <div
        className="text-lg font-semibold leading-tight tabular-nums"
        style={{ color: item.color ?? 'var(--text)' }}
        title={item.title}
      >
        {item.value}
      </div>
      <div className="mt-0.5 break-words text-[10px] text-[var(--text-muted)]">{item.label}</div>
    </div>
  )
}

function InlineCell({ item, last }: { item: KpiItem; last: boolean }) {
  return (
    <div className={cn('min-w-[140px] flex-1 px-4 py-3', !last && 'border-r border-[var(--border)]')}>
      <div
        className="type-body-sm-strong tabular-nums text-[var(--text)]"
        style={item.color ? { color: item.color } : undefined}
        title={item.title}
      >
        {item.value}
      </div>
      <div className="type-table-xs mt-0.5">{item.label}</div>
    </div>
  )
}

/**
 * Row of counters — DESIGN.md `{components.kpi-strip}`.
 *
 * One joined surface with hairline-separated cells rather than free-floating
 * tiles: the counters are the same kind of quantity, so separating them into
 * individually lifting cards would assert a distinction that does not exist.
 * Values arrive pre-formatted; this component never rounds or scales a metric.
 */
export default function KpiStrip({ items, layout = 'hero', className }: KpiStripProps) {
  if (items.length === 0) return null

  return (
    <div className={cn(CONTAINER[layout], className)}>
      {items.map((item, index) =>
        layout === 'hero' ? (
          <HeroCell key={item.label} item={item} />
        ) : layout === 'dense' ? (
          <DenseCell key={item.label} item={item} />
        ) : (
          <InlineCell key={item.label} item={item} last={index === items.length - 1} />
        ),
      )}
    </div>
  )
}
