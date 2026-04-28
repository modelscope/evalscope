import { cn } from '@/lib/utils'
import { ChevronDown } from 'lucide-react'

interface SelectOption {
  value: string
  label: string
}

interface SelectProps {
  options: SelectOption[]
  value?: string | string[]
  onChange?: (value: string) => void
  label?: string
  placeholder?: string
  className?: string
}

export default function Select({
  options,
  value,
  onChange,
  label,
  placeholder,
  className,
}: SelectProps) {
  const selectId = label?.toLowerCase().replace(/\s+/g, '-')

  return (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label
          htmlFor={selectId}
          className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]"
        >
          {label}
        </label>
      )}
      <div className="relative">
        <select
          id={selectId}
          value={typeof value === 'string' ? value : value?.[0] ?? ''}
          onChange={(e) => onChange?.(e.target.value)}
          className={cn(
            'w-full appearance-none px-3 py-2 pr-8 text-sm rounded-[var(--radius-sm)]',
            'bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)]',
            'focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent-dim)]',
            'transition-all duration-[var(--transition)]',
            !value && 'text-[var(--text-dim)]',
            className,
          )}
        >
          {placeholder && (
            <option value="" disabled>
              {placeholder}
            </option>
          )}
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <ChevronDown
          size={14}
          className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[var(--text-dim)] pointer-events-none"
        />
      </div>
    </div>
  )
}
