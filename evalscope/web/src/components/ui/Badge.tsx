import type { ComponentPropsWithoutRef, ReactNode } from 'react'
import { cn } from '@/lib/utils'

type BadgeVariant = 'default' | 'success' | 'warning' | 'danger'

interface BadgeProps extends ComponentPropsWithoutRef<'span'> {
  children: ReactNode
  variant?: BadgeVariant
}

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-[var(--accent-dim)] text-[var(--accent)]',
  success: 'bg-[var(--success-bg)] text-[var(--success)]',
  warning: 'bg-[var(--warning-bg)] text-[var(--yellow)]',
  danger: 'bg-[var(--danger-bg)] text-[var(--danger)]',
}

export default function Badge({ children, variant = 'default', className, ...rest }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
        variantStyles[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </span>
  )
}
