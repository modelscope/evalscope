import { ChevronLeft, ChevronRight } from 'lucide-react'

interface Props {
  page: number
  total: number
  onChange: (p: number) => void
}

export default function Pagination({ page, total, onChange }: Props) {
  if (total <= 1) return null
  return (
    <div className="flex items-center gap-2 text-sm">
      <button
        disabled={page <= 1}
        onClick={() => onChange(page - 1)}
        className="p-1 rounded hover:bg-[var(--color-surface-hover)] disabled:opacity-30"
      >
        <ChevronLeft size={16} />
      </button>
      <span className="text-[var(--color-ink-muted)] min-w-[5rem] text-center">
        {page} / {total}
      </span>
      <button
        disabled={page >= total}
        onClick={() => onChange(page + 1)}
        className="p-1 rounded hover:bg-[var(--color-surface-hover)] disabled:opacity-30"
      >
        <ChevronRight size={16} />
      </button>
    </div>
  )
}
