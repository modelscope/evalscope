import type { HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

export default function ErrorAlert({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      role="alert"
      className={cn(
        'rounded-[var(--radius)] border border-[var(--danger-border)] bg-[var(--danger-bg)] px-4 py-3 text-sm text-[var(--danger)]',
        className,
      )}
      {...props}
    />
  )
}
