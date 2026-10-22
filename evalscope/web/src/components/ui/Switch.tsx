import type { ButtonHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

interface SwitchProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'onChange' | 'type' | 'role'> {
  checked: boolean
  onCheckedChange: (checked: boolean) => void
}

/**
 * Compact toggle for boolean settings, aligned with adjacent form inputs.
 * The coarse-pointer target is expanded independently of the visible track.
 * Field ARIA properties keep the visible label associated with the control.
 */
export default function Switch({ checked, onCheckedChange, disabled, className, ...rest }: SwitchProps) {
  return (
    <button
      {...rest}
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onCheckedChange(!checked)}
      className={cn(
        'coarse-target group flex h-[38px] items-center rounded-[var(--radius-sm)]',
        disabled ? 'cursor-not-allowed opacity-45' : 'cursor-pointer',
        className,
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          'inline-flex h-5 w-9 shrink-0 items-center rounded-full border px-0.5 transition-colors duration-150',
          checked
            ? 'border-[var(--accent)] bg-[var(--accent)]'
            : 'border-[var(--border-strong)] bg-[var(--bg-deep)] group-hover:border-[var(--accent-dim)]',
        )}
      >
        <span
          className={cn(
            'size-3.5 rounded-full shadow-sm transition-transform duration-150',
            checked ? 'translate-x-4 bg-[var(--text-on-filled)]' : 'translate-x-0 bg-[var(--text-muted)]',
          )}
        />
      </span>
    </button>
  )
}
