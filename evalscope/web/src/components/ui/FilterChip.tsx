import { cn } from '@/lib/utils'
import { X } from 'lucide-react'

interface FilterChipProps {
  label: string
  onRemove?: () => void
  className?: string
}

export default function FilterChip({ label, onRemove, className }: FilterChipProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium',
        'bg-[var(--accent-dim)] text-[var(--accent)]',
        className,
      )}
    >
      {label}
      {onRemove && (
        <button
          onClick={onRemove}
          className="flex items-center justify-center w-3.5 h-3.5 rounded-full hover:bg-[rgba(129,109,248,0.2)] transition-colors cursor-pointer"
          aria-label={`Remove ${label}`}
        >
          <X size={10} />
        </button>
      )}
    </span>
  )
}
