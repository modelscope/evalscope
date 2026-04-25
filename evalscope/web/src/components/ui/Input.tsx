import { forwardRef, type InputHTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, className, id, ...props }, ref) => {
    const inputId = id || label?.toLowerCase().replace(/\s+/g, '-')

    return (
      <div className="flex flex-col gap-1.5">
        {label && (
          <label
            htmlFor={inputId}
            className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]"
          >
            {label}
          </label>
        )}
        <input
          ref={ref}
          id={inputId}
          className={cn(
            'w-full px-3 py-2 text-sm rounded-[var(--radius-sm)]',
            'bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)]',
            'placeholder:text-[var(--text-dim)]',
            'focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent-dim)]',
            'transition-all duration-[var(--transition)]',
            error && 'border-[#ef4444] focus:border-[#ef4444] focus:ring-[rgba(239,68,68,0.12)]',
            className,
          )}
          {...props}
        />
        {error && <p className="text-xs text-[#ef4444]">{error}</p>}
      </div>
    )
  },
)

Input.displayName = 'Input'
export default Input
