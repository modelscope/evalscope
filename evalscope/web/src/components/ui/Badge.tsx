import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

type BadgeVariant = 'default' | 'success' | 'warning' | 'danger'

interface BadgeProps {
  children: ReactNode
  variant?: BadgeVariant
  className?: string
}

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-[var(--accent-dim)] text-[var(--accent)]',
  success: 'bg-[var(--color-accent-muted)] text-[var(--green)]',
  warning: 'bg-[var(--warning-bg)] text-[var(--yellow)]',
  danger: 'bg-[var(--danger-bg)] text-[var(--danger)]',
}

export default function Badge({ children, variant = 'default', className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
        variantStyles[variant],
        className,
      )}
    >
      {children}
    </span>
  )
}
