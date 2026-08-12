import { ChevronRight } from 'lucide-react'
import type { MouseEvent } from 'react'
import { cn } from '@/lib/utils'
import SelectionCheckbox from '@/components/ui/SelectionCheckbox'
import { ScoreLines } from '@/components/reports/metricCells'
import { useLocale } from '@/contexts/LocaleContext'
import { formatTimestamp } from '@/utils/formatUtils'
import type { ReportSummary } from '@/api/types'
import { datasetLabel, primaryMetricsOf } from '@/domain/report/primaryMetrics'
import { formatReportRef, reportRefFromSummary } from '@/domain/report/reportRef'

interface ReportCardProps {
  report: ReportSummary
  selected: boolean
  onSelect: (ref: string) => void
  /** Navigate to report detail by reference */
  onClick: (ref: string) => void
}

export default function ReportCard({ report, selected, onSelect, onClick }: ReportCardProps) {
  const { t } = useLocale()

  const ref = formatReportRef(reportRefFromSummary(report))
  const formattedDate = formatTimestamp(report.timestamp)
  const metricRefs = primaryMetricsOf(report)

  const handleDetailClick = (e: MouseEvent) => {
    e.stopPropagation()
    onClick(ref)
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
      <SelectionCheckbox
        checked={selected}
        label={`${t('reports.selectReport')}: ${report.model_name}`}
        onClick={(e) => {
          e.stopPropagation()
          onSelect(ref)
        }}
      />

      {/* Content — clicking navigates to detail; selection stays on the checkbox. */}
      <button
        type="button"
        className="flex-1 min-w-0 min-h-11 flex items-center gap-4 cursor-pointer text-left"
        onClick={() => onClick(ref)}
      >
        {/* Model + Dataset */}
        <span className="block flex-1 min-w-0">
          {/* Primary row: model name + timestamp for disambiguation */}
          <span className="flex items-baseline gap-2 flex-wrap">
            <span className="font-bold text-base text-[var(--text)] break-words min-w-0">
              {report.model_name}
            </span>
            {formattedDate && (
              <span className="text-xs text-[var(--text-muted)] font-mono shrink-0">
                {formattedDate}
              </span>
            )}
          </span>
          {/* Secondary row: dataset + sample count */}
          <span className="flex items-center gap-3 mt-0.5">
            <span className="text-sm text-[var(--text-muted)] break-words min-w-0" title={report.dataset_name}>
              {datasetLabel(report)}
            </span>
            <span className="text-xs text-[var(--text-muted)] shrink-0">
              {t('reports.samples')}: {report.num_samples}
            </span>
            {/* Status — keeps card fields consistent with the desktop table. */}
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-[var(--success-bg)] text-[var(--success)] shrink-0">
              {t('reports.status.completed')}
            </span>
          </span>
        </span>

        {/* Every dataset's metric and score, so the card never hides the run's numbers. */}
        {metricRefs.length > 0 ? (
          <ScoreLines
            refs={metricRefs}
            emptyLabel={t('metrics.noPrimaryMetric')}
            inlineMetricClass=""
            inlineDatasetClass={metricRefs.length > 1 ? '' : undefined}
            className="shrink-0"
          />
        ) : <span className="text-sm text-[var(--text-muted)]">{t('metrics.noPrimaryMetric')}</span>}
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
