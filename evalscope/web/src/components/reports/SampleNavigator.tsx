import { useEffect, type ReactNode } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useLocale } from '@/contexts/LocaleContext'

interface SampleNavigatorProps {
  /** 1-based index of the sample on screen. */
  page: number
  /** Total number of samples in the current filter. */
  total: number
  onPageChange: (page: number) => void
  /** Extra content between the arrows, e.g. the sample's own index. */
  children?: ReactNode
  className?: string
}

/**
 * Previous / next navigation for a one-sample-at-a-time view.
 *
 * Both prediction surfaces — the single-report tab and the side-by-side compare
 * tab — step through samples one at a time, so the arrows, the `n / total`
 * readout and the Left/Right keyboard shortcuts live here once. The shortcuts are
 * ignored while a text field has focus, otherwise typing in a search box would
 * page the view out from under the user.
 *
 * The arrows are primary navigation controls, so they carry the R-TOUCH 44 × 44
 * hit-area floor at every size.
 */
export default function SampleNavigator({
  page,
  total,
  onPageChange,
  children,
  className,
}: SampleNavigatorProps) {
  const { t } = useLocale()

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Never steal arrow keys from a field the user is typing in.
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return
      if (event.key === 'ArrowLeft' && page > 1) onPageChange(page - 1)
      else if (event.key === 'ArrowRight' && page < total) onPageChange(page + 1)
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [page, total, onPageChange])

  const atStart = page <= 1
  const atEnd = page >= total
  const arrowClass =
    'flex min-h-[44px] min-w-[44px] items-center justify-center rounded-full border border-[var(--border)] bg-transparent text-[var(--text)] transition-colors'

  return (
    <div className={cn('flex items-center gap-3', className)}>
      <button
        type="button"
        aria-label={t('prediction.previousSample')}
        onClick={() => onPageChange(Math.max(1, page - 1))}
        disabled={atStart}
        className={cn(arrowClass, atStart ? 'cursor-not-allowed opacity-30' : 'cursor-pointer hover:bg-[var(--bg-card2)]')}
      >
        <ChevronLeft size={16} />
      </button>

      {children}

      <button
        type="button"
        aria-label={t('prediction.nextSample')}
        onClick={() => onPageChange(Math.min(total, page + 1))}
        disabled={atEnd}
        className={cn(arrowClass, atEnd ? 'cursor-not-allowed opacity-30' : 'cursor-pointer hover:bg-[var(--bg-card2)]')}
      >
        <ChevronRight size={16} />
      </button>
    </div>
  )
}
