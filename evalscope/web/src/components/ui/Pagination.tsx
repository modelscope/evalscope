import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'

interface PaginationProps {
  page: number
  totalPages: number
  onPageChange: (page: number) => void
  className?: string
}

function visiblePages(page: number, totalPages: number): (number | 'ellipsis')[] {
  const pages = Array.from({ length: totalPages }, (_, index) => index + 1)
    .filter((candidate) => candidate === 1 || candidate === totalPages || Math.abs(candidate - page) <= 2)

  return pages.reduce<(number | 'ellipsis')[]>((result, candidate, index) => {
    if (index > 0 && candidate - pages[index - 1] > 1) result.push('ellipsis')
    result.push(candidate)
    return result
  }, [])
}

export default function Pagination({ page, totalPages, onPageChange, className }: PaginationProps) {
  if (totalPages <= 1) return null

  return (
    <div className={cn('flex items-center justify-center gap-2 pt-2', className)}>
      <Button
        variant="ghost"
        size="sm"
        disabled={page <= 1}
        onClick={() => onPageChange(Math.max(1, page - 1))}
      >
        ←
      </Button>
      {visiblePages(page, totalPages).map((item, index) =>
        item === 'ellipsis' ? (
          // text-dim allowed: decorative pagination ellipsis (DESIGN.md §Text)
          <span key={`ellipsis-${index}`} className="px-1 text-[var(--text-dim)]">
            ...
          </span>
        ) : (
          <Button
            key={item}
            variant={item === page ? 'primary' : 'ghost'}
            size="sm"
            onClick={() => onPageChange(item)}
            className="!min-w-[32px]"
          >
            {item}
          </Button>
        ),
      )}
      <Button
        variant="ghost"
        size="sm"
        disabled={page >= totalPages}
        onClick={() => onPageChange(Math.min(totalPages, page + 1))}
      >
        →
      </Button>
    </div>
  )
}
