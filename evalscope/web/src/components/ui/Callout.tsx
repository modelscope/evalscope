import type { HTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/utils'

/**
 * Severity of a callout, which decides both its colours and how it is announced.
 *
 * `danger` is a failure the user has to deal with and is announced as an alert.
 * `warning` and `info` qualify what is already on screen — a workload mismatch, a
 * low sample count — and are announced as status, because interrupting a screen
 * reader for a note that changes nothing is worse than letting it be read in turn.
 */
export type CalloutVariant = 'danger' | 'warning' | 'info'

interface CalloutProps extends Omit<HTMLAttributes<HTMLDivElement>, 'role'> {
  variant?: CalloutVariant
  /** Leading icon. Sized and coloured by the caller. */
  icon?: ReactNode
}

const SURFACE: Record<CalloutVariant, string> = {
  danger: 'border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger)]',
  warning: 'border-[var(--warning-border)] bg-[var(--warning-bg)] text-[var(--text)]',
  info: 'border-[var(--border)] bg-[var(--bg-card2)] text-[var(--text-muted)]',
}

const ROLE: Record<CalloutVariant, 'alert' | 'status'> = {
  danger: 'alert',
  warning: 'status',
  info: 'status',
}

/**
 * Bordered notice strip for a failure, a caveat or a hint.
 *
 * This is the single surface for all three severities so a new caveat cannot
 * quietly arrive with its own padding, its own radius, or no announcement at all
 * — which is what happened while each page hand-rolled its own warning strip.
 *
 * Without an `icon` the children are laid out by the caller: callers pass their
 * own `flex` / `justify-between` to put a retry button opposite the message, and
 * an inner wrapper would swallow that. With an `icon` the strip owns the row so
 * the icon aligns to the first line of a message of any length.
 */
export default function Callout({
  variant = 'danger',
  icon,
  className,
  children,
  ...props
}: CalloutProps) {
  const base = 'rounded-[var(--radius)] border px-4 py-3 type-body-sm'

  if (!icon) {
    return (
      <div role={ROLE[variant]} className={cn(base, SURFACE[variant], className)} {...props}>
        {children}
      </div>
    )
  }

  return (
    <div
      role={ROLE[variant]}
      className={cn(base, 'flex items-start gap-2', SURFACE[variant], className)}
      {...props}
    >
      <span className="mt-0.5 shrink-0">{icon}</span>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  )
}
