import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

/** One option in a segmented control. */
export interface SegmentedOption<T extends string> {
  value: T
  label: string
  /** Programmatic name when the visible label is an abbreviation. Defaults to `label`. */
  accessibleLabel?: string
  /** Optional leading icon. */
  icon?: ReactNode
  /** Optional trailing count, rendered de-emphasized. */
  count?: number
}

interface SegmentedControlProps<T extends string> {
  options: SegmentedOption<T>[]
  value: T
  onChange: (value: T) => void
  /** Group label, announced to assistive technology. */
  ariaLabel: string
  /** `md` matches a form control's height; `sm` is for dense toolbars. */
  size?: 'sm' | 'md'
  /** Fill the available width, dividing it evenly between the options. */
  fullWidth?: boolean
  className?: string
}

const SIZE: Record<'sm' | 'md', string> = {
  sm: 'px-2 py-1.5 type-button-sm',
  md: 'px-4 py-2 type-button-md',
}

/**
 * Mutually-exclusive choice rendered as one joined row of buttons.
 *
 * Used wherever a small, fixed set of view modes is chosen: prediction filters,
 * absolute-vs-baseline score comparison, per-model tri-state filters. Selection
 * is published through `aria-pressed` on each button rather than a `radiogroup`,
 * because these switch the view rather than submit a value.
 */
export default function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
  size = 'md',
  fullWidth = false,
  className,
}: SegmentedControlProps<T>) {
  return (
    <div
      role="group"
      aria-label={ariaLabel}
      className={cn(
        'inline-flex overflow-hidden rounded-[var(--radius-sm)] border border-[var(--border)]',
        fullWidth && 'flex w-full',
        className,
      )}
    >
      {options.map((option, index) => {
        const isActive = option.value === value
        return (
          <button
            key={option.value}
            type="button"
            aria-label={option.accessibleLabel ?? option.label}
            aria-pressed={isActive}
            onClick={() => onChange(option.value)}
            className={cn(
              'flex min-w-0 cursor-pointer items-center justify-center gap-1.5 whitespace-nowrap transition-colors',
              SIZE[size],
              fullWidth && 'flex-1',
              isActive
                ? 'bg-[var(--accent)] text-[var(--text-on-filled)]'
                : 'bg-transparent text-[var(--text-muted)] hover:text-[var(--text)]',
              index < options.length - 1 && 'border-r border-[var(--border)]',
            )}
          >
            {option.icon}
            <span>{option.label}</span>
            {option.count !== undefined && (
              <span className="text-[0.8rem] opacity-65 tabular-nums">{option.count}</span>
            )}
          </button>
        )
      })}
    </div>
  )
}
