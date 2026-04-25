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

export default function ReportCard({ report, selected, onSelect, onClick }: ReportCardProps) {
  const { t } = useLocale()

  const formattedDate = report.timestamp
    ? new Date(report.timestamp).toLocaleDateString(undefined, {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
      })
    : '—'

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
      <input
        type="checkbox"
        checked={selected}
        onChange={(e) => {
          e.stopPropagation()
          onSelect(report.name)
        }}
        onClick={(e) => e.stopPropagation()}
        className="accent-[var(--accent)] w-4 h-4 shrink-0 cursor-pointer"
      />

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-semibold text-sm text-[var(--text)] truncate">
            {report.model_name}
          </span>
          <span className="text-[var(--text-dim)] text-xs">::</span>
          <span className="text-sm text-[var(--text-muted)] truncate">
            {report.dataset_name}
          </span>
        </div>
        <div className="flex items-center gap-3 mt-1 text-xs text-[var(--text-muted)]">
          <span
            className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-mono font-medium"
            style={{ backgroundColor: `${scoreColor(report.score)}20`, color: scoreColor(report.score) }}
          >
            {report.score.toFixed(4)}
          </span>
          <span>{t('reports.samples')}: {report.num_samples}</span>
          <span>{formattedDate}</span>
        </div>
      </div>

      {/* Chevron */}
      <ChevronRight
        size={16}
        className="text-[var(--text-dim)] group-hover:text-[var(--text-muted)] transition-colors shrink-0"
      />
    </div>
  )
}
