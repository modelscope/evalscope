import { ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useLocale } from '@/contexts/LocaleContext'
import type { ReportSummary } from '@/api/types'

interface ReportCardProps {
  report: ReportSummary
  selected: boolean
  onSelect: (name: string) => void
  onClick: (name: string) => void
}

function scoreColor(score: number): string {
  return `hsl(${score * 120}, 70%, 45%)`
}

function formatTimestamp(ts: string): string {
  return ts.replace('T', ' ').slice(0, 19)
}

function Checkbox({ checked, onChange }: { checked: boolean; onChange: () => void }) {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={checked}
      onClick={onChange}
      className="w-4.5 h-4.5 rounded-[var(--radius-xs)] border-2 flex items-center justify-center transition-all duration-150 cursor-pointer shrink-0"
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
    </button>
  )
}

export default function ReportCard({ report, selected, onSelect, onClick }: ReportCardProps) {
  const { t } = useLocale()

  const formattedDate = report.timestamp ? formatTimestamp(report.timestamp) : '—'

  return (
    <div
      className={cn(
        'group flex items-center gap-3 px-4 py-3 rounded-[var(--radius)] border bg-[var(--bg-card)]',
        'transition-all duration-[var(--transition)] cursor-pointer',
        selected
          ? 'border-[var(--accent)] shadow-[0_0_0_1px_var(--accent-dim)]'
          : 'border-[var(--border)] hover:border-[var(--border-md)]',
      )}
      onClick={() => onClick(report.name)}
    >
      {/* Checkbox */}
      <div
        onClick={(e) => {
          e.stopPropagation()
        }}
      >
        <Checkbox
          checked={selected}
          onChange={() => onSelect(report.name)}
        />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0 flex items-center gap-4">
        {/* Model + Dataset */}
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-2 flex-wrap">
            <span className="font-bold text-base text-[var(--text)] truncate">
              {report.model_name}
            </span>
            <span className="text-[var(--text-dim)] text-xs">·</span>
            <span className="text-sm text-[var(--text-muted)] truncate">
              {report.dataset_name}
            </span>
          </div>
          <div className="flex items-center gap-3 mt-0.5">
            <span className="text-xs text-[var(--text-dim)]">
              {t('reports.samples')}: {report.num_samples}
            </span>
            <span className="text-xs text-[var(--text-dim)]">{formattedDate}</span>
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

      {/* Chevron */}
      <ChevronRight
        size={16}
        className="text-[var(--text-dim)] group-hover:text-[var(--text-muted)] transition-colors shrink-0"
      />
    </div>
  )
}
