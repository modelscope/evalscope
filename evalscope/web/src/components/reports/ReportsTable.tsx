import { cn } from '@/lib/utils'
import { useLocale } from '@/contexts/LocaleContext'
import type { ReportSummary } from '@/api/types'
import { scoreColor } from '@/utils/colorScale'
import { formatMetricByKey } from '@/domain/metric/registry'
import { buildDisplayLabel } from '@/domain/compare/compareModel'

interface ReportsTableProps {
  reports: ReportSummary[]
  /** Names currently selected for compare. */
  selected: string[]
  /** Whether the explicit compare-selection mode is active. */
  compareMode: boolean
  /** Toggle a run's selection (only meaningful while in compare mode). */
  onToggleSelect: (name: string) => void
  /** Navigate to a run's detail view. */
  onRowClick: (name: string) => void
}

function formatTimestamp(ts: string): string {
  return ts.replace('T', ' ').slice(0, 16)
}

function Checkbox({ checked }: { checked: boolean }) {
  return (
    <span
      role="checkbox"
      aria-checked={checked}
      className="w-4.5 h-4.5 rounded-[var(--radius-xs)] border-2 flex items-center justify-center transition-all duration-150 shrink-0"
      style={{
        borderColor: checked ? 'var(--accent)' : 'var(--border-strong)',
        background: checked ? 'var(--accent)' : 'transparent',
      }}
    >
      {checked && (
        <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="2,6 5,9 10,3" />
        </svg>
      )}
    </span>
  )
}

/**
 * Desktop (>=1024px) tabular view of the evaluation history.
 *
 * Columns are fixed and ordered: model, dataset, time, samples, score, status
 * (Req 5.1, 5.2). Each run's model/dataset are derived through
 * `buildDisplayLabel` so the row shows a meaningful label rather than the raw
 * timestamped run name (Req 5.6). When compare mode is active a leading
 * selection column is shown and row clicks toggle selection instead of
 * navigating.
 */
export default function ReportsTable({
  reports,
  selected,
  compareMode,
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
            {compareMode && <th scope="col" className="w-10 px-4 py-3" />}
            {/* Fixed, ordered columns: model, dataset, time, samples, score, status */}
            <th scope="col" className="px-4 py-3 text-xs font-semibold text-[var(--text-muted)]">
              {t('reports.columns.model')}
            </th>
            <th scope="col" className="px-4 py-3 text-xs font-semibold text-[var(--text-muted)]">
              {t('reports.columns.dataset')}
            </th>
            <th scope="col" className="px-4 py-3 text-xs font-semibold text-[var(--text-muted)]">
              {t('reports.columns.time')}
            </th>
            <th scope="col" className="px-4 py-3 text-xs font-semibold text-[var(--text-muted)] text-right">
              {t('reports.columns.samples')}
            </th>
            <th scope="col" className="px-4 py-3 text-xs font-semibold text-[var(--text-muted)] text-right">
              {t('reports.columns.score')}
            </th>
            <th scope="col" className="px-4 py-3 text-xs font-semibold text-[var(--text-muted)]">
              {t('reports.columns.status')}
            </th>
          </tr>
        </thead>
        <tbody>
          {reports.map((report) => {
            const isSelected = selectedSet.has(report.name)
            const parsed = buildDisplayLabel(report.name)
            const model = report.model_name || parsed.model || report.name
            const dataset = report.dataset_name || parsed.dataset
            const score = formatMetricByKey('score', report.score, t)
            const handleClick = () => {
              if (compareMode) onToggleSelect(report.name)
              else onRowClick(report.name)
            }
            return (
              <tr
                key={report.name}
                onClick={handleClick}
                className={cn(
                  'border-b border-[var(--border)] last:border-b-0 cursor-pointer transition-colors',
                  isSelected ? 'bg-[var(--accent-dim)]' : 'hover:bg-[var(--bg-card2)]',
                )}
              >
                {compareMode && (
                  <td className="px-4 py-3">
                    <span
                      onClick={(e) => {
                        e.stopPropagation()
                        onToggleSelect(report.name)
                      }}
                      className="inline-flex items-center justify-center min-w-[44px] min-h-[44px] -m-3 p-3 cursor-pointer"
                    >
                      <Checkbox checked={isSelected} />
                    </span>
                  </td>
                )}
                <td className="px-4 py-3 font-semibold text-[var(--text)] break-words min-w-0">
                  {model}
                </td>
                <td className="px-4 py-3 text-[var(--text-muted)] break-words min-w-0">
                  {dataset}
                </td>
                <td className="px-4 py-3 text-[var(--text-muted)] font-mono text-xs whitespace-nowrap">
                  {report.timestamp ? formatTimestamp(report.timestamp) : '—'}
                </td>
                <td className="px-4 py-3 text-[var(--text-muted)] text-right tabular-nums">
                  {report.num_samples}
                </td>
                <td className="px-4 py-3 text-right">
                  <span
                    className="inline-flex items-center px-2.5 py-1 rounded-full text-sm font-mono font-semibold"
                    style={{ backgroundColor: `${scoreColor(report.score)}20`, color: scoreColor(report.score) }}
                  >
                    {score.primary}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-[var(--success-bg)] text-[var(--success)]">
                    {t('reports.status.completed')}
                  </span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
