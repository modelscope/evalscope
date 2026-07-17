import type { MouseEvent, ReactNode } from 'react'
import { Check } from 'lucide-react'
import { cn } from '@/lib/utils'

interface SelectionCheckboxProps {
  checked: boolean
  label: string
  onClick: (event: MouseEvent<HTMLButtonElement>) => void
  children?: ReactNode
  className?: string
}

export default function SelectionCheckbox({
  checked,
  label,
  onClick,
  children,
  className,
}: SelectionCheckboxProps) {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={checked}
      aria-label={label}
      onClick={onClick}
      className={cn(
        'inline-flex min-h-[44px] min-w-[44px] items-center justify-center',
        children && 'gap-2',
        className,
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          'flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded-[var(--radius-xs)] border-2 transition-all duration-150',
          checked && 'text-[var(--text-on-filled)]',
        )}
        style={{
          borderColor: checked ? 'var(--accent)' : 'var(--border-strong)',
          background: checked ? 'var(--accent)' : 'transparent',
        }}
      >
        {checked && <Check size={10} strokeWidth={3} />}
      </span>
      {children}
    </button>
  )
}
