import { useEffect, useMemo, useState } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import { loadReport as apiLoadReport, getHtmlReportUrl } from '@/api/reports'
import type { LoadReportResponse, ReportData } from '@/api/types'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Tabs from '@/components/ui/Tabs'
import Skeleton from '@/components/ui/Skeleton'
import ReportHeader from '@/components/reports/ReportHeader'
import DatasetNav from '@/components/reports/DatasetNav'
import OverviewTab from '@/components/reports/OverviewTab'
import DetailsTab from '@/components/reports/DetailsTab'
import PredictionsTab from '@/components/reports/PredictionsTab'

type TabKey = 'overview' | 'details' | 'predictions'

export default function ReportDetailPage() {
  const { reportId } = useParams<{ reportId: string }>()
  const [searchParams] = useSearchParams()
  const { t } = useLocale()

  const rootPath = searchParams.get('root_path') || './outputs'
  const reportName = decodeURIComponent(reportId ?? '')

  const [data, setData] = useState<LoadReportResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState<TabKey>('overview')
  const [activeDataset, setActiveDataset] = useState('')

  // Load report on mount
  useEffect(() => {
    if (!reportName) return
    let cancelled = false
    setLoading(true)
    setError('')

    apiLoadReport(rootPath, reportName)
      .then((res) => {
        if (cancelled) return
        setData(res)
        if (res.datasets.length > 0) {
          setActiveDataset(res.datasets[0])
        }
      })
      .catch((err) => {
        if (!cancelled) setError(String(err))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => { cancelled = true }
  }, [rootPath, reportName])

  const reportList = useMemo<ReportData[]>(() => data?.report_list ?? [], [data])

  // Derive overall info from report list
  const modelName = reportList[0]?.model_name ?? reportName
  const primaryDataset = reportList[0]?.dataset_name ?? ''
  const overallScore = reportList.length > 0
    ? reportList.reduce((s, r) => s + r.score, 0) / reportList.length
    : 0
  const totalSamples = reportList.reduce((sum, r) => {
    return sum + (r.metrics[0]?.categories?.reduce((s, c) => s + c.num, 0) ?? 0)
  }, 0)

  const datasets = data?.datasets ?? []
  const htmlReportUrl = getHtmlReportUrl(rootPath, reportName)

  const tabs = [
    { key: 'overview', label: t('reportDetail.overview') },
    { key: 'details', label: t('reportDetail.details') },
    { key: 'predictions', label: t('reportDetail.predictions') },
  ]

  if (loading) {
    return (
      <div className="page-enter p-6 flex flex-col gap-4">
        <Skeleton width={300} height={20} />
        <Skeleton width="100%" height={100} />
        <Skeleton lines={6} />
      </div>
    )
  }

  if (error) {
    return (
      <div className="page-enter p-6">
        <Breadcrumb
          items={[
            { label: 'Reports', href: '/reports' },
            { label: reportName || 'Detail' },
          ]}
        />
        <div className="mt-6 p-6 rounded-[var(--radius)] border border-[#ef4444] bg-[rgba(239,68,68,0.08)] text-[#ef4444]">
          <p className="text-sm">Failed to load report: {error}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="page-enter flex flex-col gap-4 p-6">
      {/* Breadcrumb */}
      <Breadcrumb
        items={[
          { label: 'Reports', href: '/reports' },
          { label: reportName },
        ]}
      />

      {/* Report Header */}
      <ReportHeader
        modelName={modelName}
        datasetName={primaryDataset}
        score={overallScore}
        totalSamples={totalSamples}
        htmlReportUrl={htmlReportUrl}
      />

      {/* Main body: Dataset Nav + Tabbed Content */}
      <div className="flex flex-col md:flex-row gap-0 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden">
        {/* Dataset nav - horizontal scroll on mobile, vertical sidebar on desktop */}
        {datasets.length > 0 && (
          <>
            {/* Mobile horizontal dataset nav */}
            <div className="md:hidden flex items-center gap-1 px-4 py-2 border-b border-[var(--border)] overflow-x-auto">
              {datasets.map((ds) => (
                <button
                  key={ds}
                  onClick={() => setActiveDataset(ds)}
                  className={`whitespace-nowrap px-3 py-1.5 text-xs rounded-full transition-all duration-150 ${
                    ds === activeDataset
                      ? 'bg-[var(--accent-dim)] text-[var(--accent)] font-medium'
                      : 'text-[var(--text-muted)] hover:bg-[var(--bg-card2)]'
                  }`}
                >
                  {ds}
                </button>
              ))}
            </div>
            {/* Desktop vertical dataset nav */}
            <div className="hidden md:block">
              <DatasetNav
                datasets={datasets}
                active={activeDataset}
                onChange={setActiveDataset}
              />
            </div>
          </>
        )}

        {/* Right content area */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Tabs bar */}
          <div className="px-5 pt-4 pb-2 border-b border-[var(--border)]">
            <Tabs tabs={tabs} activeKey={activeTab} onChange={(k) => setActiveTab(k as TabKey)} />
          </div>

          {/* Tab content */}
          <div className="p-5">
            {activeTab === 'overview' && (
              <OverviewTab
                reports={reportList}
                reportName={reportName}
                rootPath={rootPath}
                taskConfig={data?.task_config}
              />
            )}
            {activeTab === 'details' && (
              <DetailsTab
                reportName={reportName}
                datasetName={activeDataset}
                rootPath={rootPath}
              />
            )}
            {activeTab === 'predictions' && (
              <PredictionsTab
                reportName={reportName}
                datasetName={activeDataset}
                rootPath={rootPath}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
