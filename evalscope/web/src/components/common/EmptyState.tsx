import type { ReactNode } from 'react'
import { Inbox } from 'lucide-react'
import { cn } from '@/lib/utils'

interface EmptyStateProps {
  /** Lucide icon node (rendered at 28px). Defaults to `<Inbox />` for 'empty' variant. */
  icon?: ReactNode
  title?: string
  hint?: string
  /**
   * 'empty'   — no-data state, icon in text-dim, title in text-muted (default).
   * 'welcome' — first-contact state, icon in accent, title in text.
   */
  variant?: 'empty' | 'welcome'
  className?: string
}

/**
 * Empty / welcome state — DESIGN.md `{components.empty-state}`.
 * 64×64 deep-well rounded-lg tile holding a 28px Lucide icon, followed by a 2-line message.
 */
export default function EmptyState({
  icon,
  title,
  hint,
  variant = 'empty',
  className,
}: EmptyStateProps) {
  const iconNode = icon ?? <Inbox size={28} strokeWidth={1.5} />
  const isWelcome = variant === 'welcome'

  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3 py-12',
        className,
      )}
    >
      <div
        className="w-16 h-16 rounded-[var(--radius-lg)] bg-[var(--bg-deep)] border border-[var(--border)] flex items-center justify-center"
        style={{ color: isWelcome ? 'var(--accent)' : 'var(--text-dim)' }}
      >
        {iconNode}
      </div>
      {(title || hint) && (
        <div className="text-center flex flex-col gap-1">
          {title && (
            <p
              className={cn(
                'type-body-sm font-semibold',
                isWelcome ? 'text-[var(--text)]' : 'text-[var(--text-muted)]',
              )}
            >
              {title}
            </p>
          )}
          {/* text-dim allowed: non-essential hint per DESIGN.md §empty-state */}
          {hint && <p className="type-body-xs text-[var(--text-dim)]">{hint}</p>}
        </div>
      )}
    </div>
  )
}
