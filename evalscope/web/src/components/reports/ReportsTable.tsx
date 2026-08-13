import { cn } from '@/lib/utils'
import { useLocale } from '@/contexts/LocaleContext'
import { formatTimestamp } from '@/utils/formatUtils'
import SelectionCheckbox from '@/components/ui/SelectionCheckbox'
import { DatasetLines, MetricLines, ScoreLines } from '@/components/reports/metricCells'
import type { ReportSummary } from '@/api/types'
import { datasetLabel } from '@/domain/report/primaryMetrics'
import { formatReportRef, reportRefFromSummary } from '@/domain/report/reportRef'

interface ReportsTableProps {
  reports: ReportSummary[]
  /** Report references (`{runId}/{modelId}`) currently selected for compare. */
  selected: string[]
  /** Whether every run on the current page is selected. */
  allSelected: boolean
  /** Toggle a run's selection by reference. */
  onToggleSelectAll: () => void
  /** Toggle a run's selection by reference. */
  onToggleSelect: (ref: string) => void
  /** Navigate to a run's detail view by reference. */
  onRowClick: (ref: string) => void
}

/**
 * Desktop (>=1024px) tabular view of the evaluation history.
 *
 * Columns are fixed and ordered: model, dataset, time, samples, score, status.
 * The model name and dataset come straight from the summary; the row is keyed and
 * selected by its report reference. A leading selection column is always visible
 * while row clicks continue to open the report detail.
 */
export default function ReportsTable({
  reports,
  selected,
  allSelected,
  onToggleSelectAll,
  onToggleSelect,
  onRowClick,
}: ReportsTableProps) {
  const { t } = useLocale()
  const selectedSet = new Set(selected)

  return (
    <div className="overflow-x-auto rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-[var(--border)] text-left">
            <th scope="col" className="w-10 px-4 py-3">
              <SelectionCheckbox checked={allSelected} label={t('reports.selectAll')} onClick={onToggleSelectAll} />
            </th>
            {/* Fixed, ordered columns: model, dataset, metric, score, samples, status, time.
                The result reads left to right as "what was measured, then how it did"; the
                timestamp is bookkeeping and sits last. Metric is a column of its own at every
                width -- naming it in the Score header instead only worked while every row shared
                one metric, and made the header change as the list was filtered or paged. */}
            <th scope="col" className="px-4 py-3 text-xs font-semibold text-[var(--text-muted)]">
              {t('reports.columns.model')}
            </th>
            <th scope="col" className="px-4 py-3 text-xs font-semibold text-[var(--text-muted)]">
              {t('reports.columns.dataset')}
            </th>
            <th scope="col" className="px-4 py-3 text-xs font-semibold text-[var(--text-muted)]">
              {t('reportDetail.metric')}
            </th>
            <th scope="col" className="px-4 py-3 text-xs font-semibold text-[var(--text-muted)] text-right whitespace-nowrap">
              {t('reports.columns.score')}
            </th>
            <th scope="col" className="px-4 py-3 text-xs font-semibold text-[var(--text-muted)] text-right">
              {t('reports.columns.samples')}
            </th>
            <th scope="col" className="px-4 py-3 text-xs font-semibold text-[var(--text-muted)]">
              {t('reports.columns.status')}
            </th>
            <th scope="col" className="px-4 py-3 text-xs font-semibold text-[var(--text-muted)]">
              {t('reports.columns.time')}
            </th>
          </tr>
        </thead>
        <tbody>
          {reports.map((report) => {
            const ref = formatReportRef(reportRefFromSummary(report))
            const isSelected = selectedSet.has(ref)
            const model = report.model_name || report.model_id
            const dataset = datasetLabel(report)
            const metricRefs = report.primary_metrics
            return (
              <tr
                key={ref}
                onClick={() => onRowClick(ref)}
                className={cn(
                  'border-b border-[var(--border)] last:border-b-0 cursor-pointer transition-colors',
                  isSelected ? 'bg-[var(--accent-dim)]' : 'hover:bg-[var(--bg-card2)]',
                )}
              >
                <td className="px-4 py-3">
                  <SelectionCheckbox
                    checked={isSelected}
                    label={`${t('reports.selectReport')}: ${model}`}
                    onClick={(e) => {
                      e.stopPropagation()
                      onToggleSelect(ref)
                    }}
                  />
                </td>
                <td className="px-4 py-3 font-semibold text-[var(--text)] break-words min-w-0">
                  {model}
                </td>
                <td className="px-4 py-3 text-[var(--text-muted)] min-w-0">
                  <DatasetLines refs={metricRefs} fallback={dataset} />
                </td>
                <td className="px-4 py-3 text-[var(--text-muted)] text-xs min-w-0">
                  <MetricLines refs={metricRefs} />
                </td>
                <td className="px-4 py-3 text-right">
                  {metricRefs.length > 0 ? (
                    <ScoreLines refs={metricRefs} emptyLabel={t('metrics.noPrimaryMetric')} />
                  ) : <span className="text-sm text-[var(--text-muted)]">{t('metrics.noPrimaryMetric')}</span>}
                </td>
                <td className="px-4 py-3 text-[var(--text-muted)] text-right tabular-nums">
                  {report.num_samples}
                </td>
                <td className="px-4 py-3">
                  <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-[var(--success-bg)] text-[var(--success)]">
                    {t('reports.status.completed')}
                  </span>
                </td>
                <td className="px-4 py-3 text-[var(--text-muted)] font-mono text-xs whitespace-nowrap">
                  {formatTimestamp(report.timestamp) || '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
