import { ChevronRight } from 'lucide-react'
import type { MouseEvent } from 'react'
import { cn } from '@/lib/utils'
import { useLocale } from '@/contexts/LocaleContext'
import type { ReportSummary } from '@/api/types'
import { scoreColor } from '@/components/ui/Table'

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

function Checkbox({ checked }: { checked: boolean }) {
  return (
    <div
      role="checkbox"
      aria-checked={checked}
      className="w-4.5 h-4.5 rounded-[var(--radius-xs)] border-2 flex items-center justify-center transition-all duration-150 shrink-0"
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
    </div>
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
      {/* Checkbox — always visible; dimmed when unchecked, full opacity on hover or when checked */}
      <div
        onClick={(e) => {
          e.stopPropagation()
          onSelect(report.name)
        }}
        className={cn(
          'transition-opacity duration-150 cursor-pointer',
          selected ? 'opacity-100' : 'opacity-30 group-hover:opacity-70',
        )}
      >
        <Checkbox checked={selected} />
      </div>

      {/* Content — clicking navigates to detail */}
      <div
        className="flex-1 min-w-0 flex items-center gap-4 cursor-pointer"
        onClick={() => onClick(report.name)}
      >
        {/* Model + Dataset */}
        <div className="flex-1 min-w-0">
          {/* Primary row: model name + timestamp for disambiguation */}
          <div className="flex items-baseline gap-2 flex-wrap">
            <span className="font-bold text-base text-[var(--text)] truncate">
              {report.model_name}
            </span>
            {formattedDate && (
              <span className="text-xs text-[var(--text-dim)] font-mono shrink-0">
                {formattedDate}
              </span>
            )}
          </div>
          {/* Secondary row: dataset + sample count */}
          <div className="flex items-center gap-3 mt-0.5">
            <span className="text-sm text-[var(--text-muted)] truncate">
              {report.dataset_name}
            </span>
            <span className="text-xs text-[var(--text-dim)] shrink-0">
              {t('reports.samples')}: {report.num_samples}
            </span>
          </div>
        </div>

        {/* Score badge */}
        <span
          className="inline-flex items-center px-2.5 py-1 rounded-full text-sm font-mono font-semibold shrink-0"
          style={{ backgroundColor: `${scoreColor(report.score)}20`, color: scoreColor(report.score) }}
        >
          {report.score.toFixed(4)}
        </span>
      </div>

      {/* Chevron — dedicated detail navigation button */}
      <button
        type="button"
        aria-label="View report detail"
        onClick={handleDetailClick}
        className="shrink-0 flex items-center justify-center rounded p-0.5 transition-colors cursor-pointer opacity-40 group-hover:opacity-100 hover:bg-[var(--bg-card2)]"
      >
        <ChevronRight
          size={16}
          className="text-[var(--text-dim)] transition-colors"
        />
      </button>
    </div>
  )
}
