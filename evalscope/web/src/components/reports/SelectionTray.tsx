import { Eye, GitCompareArrows, Trash2, X } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { MAX_COMPARE_SELECTION } from '@/domain/compare/compareModel'
import Button from '@/components/ui/Button'

interface SelectionTrayProps {
  count: number
  capNotice: boolean
  canViewHtml: boolean
  onViewHtml: () => void
  onCompare: () => void
  onClear: () => void
  onDelete?: () => void
  deleting?: boolean
}

export default function SelectionTray({
  count,
  capNotice,
  canViewHtml,
  onViewHtml,
  onCompare,
  onClear,
  onDelete,
  deleting,
}: SelectionTrayProps) {
  const { t } = useLocale()

  if (count < 1) return null

  return (
    <div className="sticky bottom-0 z-30 mt-2 -mx-1 px-1">
      <div className="flex flex-wrap items-center gap-3 rounded-[var(--radius)] border border-[var(--accent-dim)] bg-[var(--bg-card)] px-4 py-3 shadow-[var(--shadow-lg)]">
        <span className="text-sm font-semibold text-[var(--text)]">
          {count} {t('reports.selected')}
          <span className="ml-1 text-xs font-normal text-[var(--text-muted)]">
            / {MAX_COMPARE_SELECTION}
          </span>
        </span>

        {capNotice && (
          <span className="text-xs text-[var(--warning-color)]" role="status" aria-live="polite">
            {t('reports.capReached')}
          </span>
        )}
        {!capNotice && count > 3 && (
          <span className="text-xs text-[var(--warning-color)]">{t('compare.maxThreeSelected')}</span>
        )}

        <div className="ml-auto flex items-center gap-2">
          {onDelete && (
            <Button
              variant="outline"
              size="sm"
              disabled={deleting}
              onClick={onDelete}
              className="border-[var(--danger-border)] text-[var(--danger)] hover:border-[var(--danger)] hover:text-[var(--danger)]"
            >
              <Trash2 size={14} />
              {t('reports.delete')}
            </Button>
          )}
          <Button variant="outline" size="sm" disabled={!canViewHtml} onClick={onViewHtml}>
            <Eye size={14} />
            {t('reports.viewHtml')}
          </Button>
          <Button variant="primary" size="sm" disabled={count < 2} onClick={onCompare}>
            <GitCompareArrows size={14} />
            {t('reports.compare')}
          </Button>
          <button
            type="button"
            aria-label={t('reports.clearSelection')}
            onClick={onClear}
            className="flex min-h-[44px] min-w-[44px] cursor-pointer items-center justify-center rounded-[var(--radius-sm)] text-[var(--text-muted)] transition-colors hover:bg-[var(--bg-card2)] hover:text-[var(--text)]"
          >
            <X size={16} />
          </button>
        </div>
      </div>
    </div>
  )
}
