import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import Sidebar from '@/components/sidebar/Sidebar'
import DatasetsOverview from '@/components/single/DatasetsOverview'
import ModelsOverview from '@/components/multi/ModelsOverview'
import EmptyState from '@/components/common/EmptyState'
import LoadingSpinner from '@/components/common/LoadingSpinner'
import type { ReportData } from '@/api/types'
import { parseReportName } from '@/utils/reportParser'

type Tab = 'single' | 'multi'

export default function DashboardPage() {
  const { t } = useLocale()
  const { selectedReports, loading, loadReport, loadMultiReports, reportCache, multiReportList } = useReports()
  const navigate = useNavigate()
  const [tab, setTab] = useState<Tab>('single')
  const [loaded, setLoaded] = useState(false)

  const uniqueModels = useMemo(() => {
    const models = new Set(selectedReports.map((r) => parseReportName(r).model))
    return models.size
  }, [selectedReports])

  const handleLoadView = async () => {
    if (selectedReports.length === 0) return
    setLoaded(false)
    if (uniqueModels > 1) {
      await loadMultiReports(selectedReports)
      setTab('multi')
    } else if (selectedReports.length > 0) {
      await loadReport(selectedReports[0])
      setTab('single')
    }
    setLoaded(true)
  }

  const singleReportData = useMemo<ReportData[]>(() => {
    if (selectedReports.length === 0) return []
    const cached = reportCache[selectedReports[0]]
    return cached?.report_list ?? []
  }, [selectedReports, reportCache])

  return (
    <div className="flex gap-4">
      {/* Left sidebar */}
      <aside className="w-72 shrink-0 p-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)]">
        <Sidebar />
        <button
          onClick={handleLoadView}
          disabled={loading || selectedReports.length === 0}
          className="mt-3 w-full px-3 py-2 text-sm font-medium rounded-md bg-[var(--color-primary)] text-white disabled:opacity-50 hover:opacity-90 transition-opacity"
        >
          {loading ? t('common.loading') : t('sidebar.loadBtn')}
        </button>
      </aside>

      {/* Main content */}
      <div className="flex-1 min-w-0">
        {!loaded && !loading && (
          <EmptyState text={t('sidebar.note')} />
        )}
        {loading && <LoadingSpinner text={t('common.loading')} />}
        {loaded && !loading && (
          <>
            {/* Tabs */}
            <div className="flex gap-1 mb-4 border-b border-[var(--color-border)]">
              <button
                onClick={() => setTab('single')}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                  tab === 'single'
                    ? 'border-[var(--color-primary)] text-[var(--color-primary)]'
                    : 'border-transparent text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]'
                }`}
              >
                {t('visualization.singleModel')}
              </button>
              <button
                onClick={() => setTab('multi')}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                  tab === 'multi'
                    ? 'border-[var(--color-primary)] text-[var(--color-primary)]'
                    : 'border-transparent text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]'
                }`}
              >
                {t('visualization.multiModel')}
              </button>
            </div>

            {/* Tab content */}
            {tab === 'single' && singleReportData.length > 0 && (
              <div className="space-y-4">
                <DatasetsOverview reports={singleReportData} reportName={selectedReports[0] ?? ''} />
                <div className="flex justify-end">
                  <button
                    onClick={() => navigate('/dashboard/single')}
                    className="text-sm text-[var(--color-primary)] hover:underline"
                  >
                    {t('single.datasetDetails')} &rarr;
                  </button>
                </div>
              </div>
            )}
            {tab === 'multi' && multiReportList.length > 0 && (
              <div className="space-y-4">
                <ModelsOverview reports={multiReportList} />
                <div className="flex justify-end">
                  <button
                    onClick={() => navigate('/dashboard/multi')}
                    className="text-sm text-[var(--color-primary)] hover:underline"
                  >
                    {t('multi.modelComparisonDetails')} &rarr;
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
