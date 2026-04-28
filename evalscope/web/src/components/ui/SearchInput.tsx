import { cn } from '@/lib/utils'
import { Search, X } from 'lucide-react'

interface SearchInputProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  className?: string
}

export default function SearchInput({ value, onChange, placeholder, className }: SearchInputProps) {
  return (
    <div className={cn('relative', className)}>
      <Search
        size={14}
        className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-dim)]"
      />
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder ?? 'Search...'}
        className={cn(
          'w-full pl-9 pr-8 py-2 text-sm rounded-[var(--radius-sm)]',
          'bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)]',
          'placeholder:text-[var(--text-dim)]',
          'focus:outline-none focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent-dim)]',
          'transition-all duration-[var(--transition)]',
        )}
      />
      {value && (
        <button
          onClick={() => onChange('')}
          className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[var(--text-dim)] hover:text-[var(--text)] transition-colors cursor-pointer"
          aria-label="Clear search"
        >
          <X size={14} />
        </button>
      )}
    </div>
  )
}
