import { type ReactNode } from 'react'
import { FolderInput, Search } from 'lucide-react'
import Button from '@/components/ui/Button'

interface PathBarProps {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  placeholder?: string
  submitLabel: ReactNode
  scanningLabel?: ReactNode
  scanning?: boolean
  disabled?: boolean
  icon?: ReactNode
}

/**
 * PathBar — the dashboard "scan this directory" input row.
 * DESIGN.md `{components.path-bar}`: L2 card chrome hosting icon + input + primary button, 12-px flex gap.
 */
export default function PathBar({
  value,
  onChange,
  onSubmit,
  placeholder,
  submitLabel,
  scanningLabel,
  scanning,
  disabled,
  icon,
}: PathBarProps) {
  return (
    <div className="flex items-center gap-3 p-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] shadow-[var(--shadow-sm)]">
      <span className="text-[var(--accent)] shrink-0 flex items-center">
        {icon ?? <FolderInput size={18} />}
      </span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && onSubmit()}
        placeholder={placeholder}
        className="flex-1 min-w-0 px-3 py-2 type-body-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent-dim)] transition-all duration-150"
      />
      <Button onClick={onSubmit} disabled={disabled || scanning} size="md">
        {scanning ? (
          <span className="flex items-center gap-1.5">
            <svg className="animate-spin w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={3}>
              <circle cx="12" cy="12" r="9" strokeOpacity={0.25} />
              <path d="M21 12a9 9 0 11-9-9" />
            </svg>
            {scanningLabel ?? submitLabel}
          </span>
        ) : (
          <span className="flex items-center gap-1.5">
            <Search size={14} />
            {submitLabel}
          </span>
        )}
      </Button>
    </div>
  )
}
