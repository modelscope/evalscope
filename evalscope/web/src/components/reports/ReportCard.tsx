import { ChevronRight } from 'lucide-react'
import type { MouseEvent } from 'react'
import { cn } from '@/lib/utils'
import { useLocale } from '@/contexts/LocaleContext'
import type { ReportSummary } from '@/api/types'
import { scoreColor } from '@/utils/colorScale'
import { formatMetricByKey } from '@/domain/metric/registry'

interface ReportCardProps {
  report: ReportSummary
  selected: boolean
  onSelect: (name: string) => void
  /** Navigate to report detail */
  onClick: (name: string) => void
}

function formatTimestamp(ts: string): string {
  return ts.replace('T', ' ').slice(0, 16)
}

function Checkbox({ checked, label, onClick }: { checked: boolean; label: string; onClick: (event: MouseEvent<HTMLButtonElement>) => void }) {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={checked}
      aria-label={label}
      onClick={onClick}
      className="flex min-h-[44px] min-w-[44px] items-center justify-center"
    >
      <span
        aria-hidden="true"
        className="flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded-[var(--radius-xs)] border-2 transition-all duration-150"
        style={{
          borderColor: checked ? 'var(--accent)' : 'var(--border-strong)',
          background: checked ? 'var(--accent)' : 'transparent',
        }}
      >
        {checked && (
          <svg
            width="10"
            height="10"
            viewBox="0 0 12 12"
            fill="none"
            stroke="white"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="2,6 5,9 10,3" />
          </svg>
        )}
      </span>
    </button>
  )
}

export default function ReportCard({ report, selected, onSelect, onClick }: ReportCardProps) {
  const { t } = useLocale()

  const formattedDate = report.timestamp ? formatTimestamp(report.timestamp) : ''

  const handleDetailClick = (e: MouseEvent) => {
    e.stopPropagation()
    onClick(report.name)
  }

  return (
    <div
      className={cn(
        'group flex items-center gap-3 px-4 py-3 rounded-[var(--radius)] border bg-[var(--bg-card)]',
        'transition-all duration-[var(--transition)]',
        selected
          ? 'border-[var(--accent)] shadow-[0_0_0_1px_var(--accent-dim)]'
          : 'border-[var(--border)] hover:border-[var(--border-md)]',
      )}
    >
      <Checkbox
        checked={selected}
        label={`${t('reports.selectReport')}: ${report.model_name}`}
        onClick={(e) => {
          e.stopPropagation()
          onSelect(report.name)
        }}
      />

      {/* Content — clicking navigates to detail; selection stays on the checkbox. */}
      <button
        type="button"
        className="flex-1 min-w-0 min-h-11 flex items-center gap-4 cursor-pointer text-left"
        onClick={() => onClick(report.name)}
      >
        {/* Model + Dataset */}
        <div className="flex-1 min-w-0">
          {/* Primary row: model name + timestamp for disambiguation */}
          <div className="flex items-baseline gap-2 flex-wrap">
            <span className="font-bold text-base text-[var(--text)] break-words min-w-0">
              {report.model_name}
            </span>
            {formattedDate && (
              <span className="text-xs text-[var(--text-muted)] font-mono shrink-0">
                {formattedDate}
              </span>
            )}
          </div>
          {/* Secondary row: dataset + sample count */}
          <div className="flex items-center gap-3 mt-0.5">
            <span className="text-sm text-[var(--text-muted)] break-words min-w-0">
              {report.dataset_name}
            </span>
            <span className="text-xs text-[var(--text-muted)] shrink-0">
              {t('reports.samples')}: {report.num_samples}
            </span>
            {/* Status — keeps card fields consistent with the desktop table. */}
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-[var(--success-bg)] text-[var(--success)] shrink-0">
              {t('reports.status.completed')}
            </span>
          </div>
        </div>

        {/* Score badge */}
        <span
          className="inline-flex items-center px-2.5 py-1 rounded-full text-sm font-mono font-semibold shrink-0"
          style={{ backgroundColor: `${scoreColor(report.score)}20`, color: scoreColor(report.score) }}
        >
          {formatMetricByKey('score', report.score, t).primary}
        </span>
      </button>

      {/* Chevron — dedicated detail navigation button */}
      <button
        type="button"
        aria-label="View report detail"
        onClick={handleDetailClick}
        className="shrink-0 flex min-h-11 min-w-11 items-center justify-center rounded transition-colors cursor-pointer opacity-60 group-hover:opacity-100 hover:bg-[var(--bg-card2)]"
      >
        {/* text-dim allowed: detail-nav chevron icon (DESIGN.md §Text) */}
        <ChevronRight
          size={16}
          className="text-[var(--text-dim)] transition-colors"
        />
      </button>
    </div>
  )
}
