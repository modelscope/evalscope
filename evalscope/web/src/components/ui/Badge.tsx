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
  success: 'bg-[rgba(15,156,126,0.12)] text-[var(--green)]',
  warning: 'bg-[rgba(251,191,36,0.12)] text-[var(--yellow)]',
  danger: 'bg-[rgba(239,68,68,0.12)] text-[#ef4444]',
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
