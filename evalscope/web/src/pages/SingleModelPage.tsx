import { useEffect, useMemo, useState } from 'react'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import DatasetsOverview from '@/components/single/DatasetsOverview'
import DatasetDetails from '@/components/single/DatasetDetails'
import SummaryStats from '@/components/single/SummaryStats'
import EmptyState from '@/components/common/EmptyState'
import type { ReportData } from '@/api/types'
import { prettyJson } from '@/utils/formatUtils'

export default function SingleModelPage() {
  const { t } = useLocale()
  const { selectedReports, reportCache } = useReports()
  const [activeReport, setActiveReport] = useState<string>(selectedReports[0] ?? '')

  // Sync activeReport when selectedReports changes
  useEffect(() => {
    if (selectedReports.length && !selectedReports.includes(activeReport)) {
      setActiveReport(selectedReports[0])
    }
  }, [selectedReports, activeReport])

  const cached = reportCache[activeReport]
  const reportList = useMemo<ReportData[]>(() => cached?.report_list ?? [], [cached])
  const taskConfig = cached?.task_config

  if (selectedReports.length === 0) {
    return <EmptyState text={t('sidebar.note')} />
  }

  return (
    <div className="space-y-6">
      {/* Report selector */}
      {selectedReports.length > 1 && (
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-[var(--color-ink-muted)]">{t('single.selectReport')}</label>
          <select
            value={activeReport}
            onChange={(e) => setActiveReport(e.target.value)}
            className="px-2 py-1.5 text-sm rounded-md bg-[var(--color-surface)] border border-[var(--color-border)] focus:outline-none focus:border-[var(--color-primary)]"
          >
            {selectedReports.map((r) => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
        </div>
      )}

      {/* Summary stats */}
      {reportList.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold mb-3">{t('single.summaryStats')}</h2>
          <SummaryStats reports={reportList} />
        </section>
      )}

      {/* Task config */}
      {taskConfig && Object.keys(taskConfig).length > 0 && (
        <details className="rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)]">
          <summary className="px-4 py-2 text-sm font-medium cursor-pointer hover:bg-[var(--color-surface-hover)]">
            {t('single.taskConfig')}
          </summary>
          <pre className="px-4 py-3 text-xs overflow-auto max-h-60">{prettyJson(taskConfig)}</pre>
        </details>
      )}

      {/* Overview */}
      <section>
        <h2 className="text-lg font-semibold mb-3">{t('single.datasetsOverview')}</h2>
        <DatasetsOverview reports={reportList} reportName={activeReport} />
      </section>

      {/* Details */}
      <section>
        <h2 className="text-lg font-semibold mb-3">{t('single.datasetDetails')}</h2>
        <DatasetDetails reports={reportList} reportName={activeReport} />
      </section>
    </div>
  )
}
