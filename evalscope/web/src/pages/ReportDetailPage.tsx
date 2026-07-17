import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import { loadReport as apiLoadReport, getHtmlReportUrl } from '@/api/reports'
import { isDomainError } from '@/api/errors'
import type { LoadReportResponse, ReportData } from '@/api/types'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Tabs from '@/components/ui/Tabs'
import Skeleton from '@/components/ui/Skeleton'
import ErrorAlert from '@/components/ui/ErrorAlert'
import ReportHeader from '@/components/reports/ReportHeader'
import DatasetNav from '@/components/reports/DatasetNav'
import OverviewTab from '@/components/reports/OverviewTab'
import DetailsTab from '@/components/reports/DetailsTab'
import PredictionsTab from '@/components/reports/PredictionsTab'
import { resolveMetricKey } from '@/domain/metric/registry'

type TabKey = 'overview' | 'details' | 'predictions'

export default function ReportDetailPage() {
  const { reportId } = useParams<{ reportId: string }>()
  const [searchParams] = useSearchParams()
  const { t } = useLocale()

  const rootPath = searchParams.get('root_path') || './outputs'
  const reportName = decodeURIComponent(reportId ?? '')

  // Parse model name from reportName format: {timestamp}@@{model_name}::{datasets}
  const breadcrumbLabel = useMemo(() => {
    const atIdx = reportName.indexOf('@@')
    if (atIdx === -1) return reportName
    const afterAt = reportName.slice(atIdx + 2)
    const colonIdx = afterAt.indexOf('::')
    return colonIdx !== -1 ? afterAt.slice(0, colonIdx) : afterAt
  }, [reportName])

  const [data, setData] = useState<LoadReportResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState<TabKey>('overview')
  const [activeDataset, setActiveDataset] = useState('')
  const [initialSubset, setInitialSubset] = useState<string | undefined>(undefined)

  // Load report when the detail inputs change. A change aborts the previous
  // request and drops its late/aborted response so only the newest
  // request updates the view.
  useEffect(() => {
    if (!reportName) return
    const controller = new AbortController()
    const load = async () => {
      setLoading(true)
      setError('')
      try {
        const res = await apiLoadReport(rootPath, reportName, controller.signal)
        if (controller.signal.aborted) return
        setData(res)
        if (res.datasets.length > 0) {
          setActiveDataset(res.datasets[0])
        }
      } catch (err) {
        if (controller.signal.aborted || (isDomainError(err) && err.kind === 'aborted')) return
        setError(String(err))
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }
    load()
    return () => controller.abort()
  }, [rootPath, reportName])

  const reportList = useMemo<ReportData[]>(() => data?.report_list ?? [], [data])

  // Derive overall info from report list
  const modelName = reportList[0]?.model_name ?? reportName
  const primaryDataset = reportList[0]?.dataset_name ?? ''
  const overallMetric = useMemo(() => {
    if (reportList.length === 0) return { score: null, metricName: '' }
    const metricNames = reportList.map((report) => report.metrics[0]?.name ?? 'score')
    const firstKey = resolveMetricKey(metricNames[0])
    if (!metricNames.every((name) => resolveMetricKey(name) === firstKey)) {
      return { score: null, metricName: '' }
    }
    return {
      score: reportList.reduce((sum, report) => sum + report.score, 0) / reportList.length,
      metricName: metricNames[0],
    }
  }, [reportList])
  const totalSamples = reportList.reduce((sum, r) => {
    return sum + (r.metrics[0]?.categories?.reduce((s, c) => s + c.num, 0) ?? 0)
  }, 0)

  const datasets = data?.datasets ?? []
  const htmlReportUrl = getHtmlReportUrl(rootPath, reportName)

  // Handler: switch dataset and auto-navigate to details tab
  const handleDatasetChange = (ds: string) => {
    setActiveDataset(ds)
    setInitialSubset(undefined)
    if (activeTab === 'overview') {
      setActiveTab('details')
    }
  }

  // Handler: click a subset name in DetailsTab → jump to Predictions with that subset pre-selected
  const handleSubsetClick = (subset: string) => {
    setInitialSubset(subset)
    setActiveTab('predictions')
  }

  const tabs = [
    { key: 'overview', label: t('reportDetail.overview'), panelId: 'report-overview-panel' },
    { key: 'details', label: t('reportDetail.details'), panelId: 'report-details-panel' },
    { key: 'predictions', label: t('reportDetail.predictions'), panelId: 'report-predictions-panel' },
  ]

  const renderDatasetPanel = (content: ReactNode) => (
    <div className="flex flex-col md:flex-row gap-0 rounded-b-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden">
      {datasets.length > 0 && (
        <>
          <div className="md:hidden flex items-center gap-1 px-4 py-2 border-b border-[var(--border)] overflow-x-auto">
            {datasets.map((ds) => (
              <button
                key={ds}
                type="button"
                onClick={() => handleDatasetChange(ds)}
                className={`min-h-11 whitespace-nowrap px-3 py-1.5 text-xs rounded-full transition-all duration-150 ${
                  ds === activeDataset
                    ? 'bg-[var(--accent-dim)] text-[var(--accent)] font-medium'
                    : 'text-[var(--text-muted)] hover:bg-[var(--bg-card2)]'
                }`}
              >
                {ds}
              </button>
            ))}
          </div>
          <div className="hidden md:block">
            <DatasetNav datasets={datasets} active={activeDataset} onChange={handleDatasetChange} />
          </div>
        </>
      )}
      <div className="flex-1 min-w-0 p-5">{content}</div>
    </div>
  )

  if (loading && !data) {
    return (
      <div className="page-enter p-6 flex flex-col gap-4">
        <Skeleton width={300} height={20} />
        <Skeleton width="100%" height={100} />
        <Skeleton lines={6} />
      </div>
    )
  }

  if (error && !data) {
    return (
      <div className="page-enter p-6">
        <Breadcrumb
          items={[
            { label: 'Reports', href: `/reports?root_path=${encodeURIComponent(rootPath)}` },
            { label: breadcrumbLabel || 'Detail' },
          ]}
        />
        <ErrorAlert className="mt-6 p-6 border-[var(--danger)]">
          <p className="text-sm">Failed to load report: {error}</p>
        </ErrorAlert>
      </div>
    )
  }

  return (
    <div className="page-enter flex flex-col gap-4 p-6">
      {/* Breadcrumb */}
      <Breadcrumb
        items={[
          { label: 'Reports', href: `/reports?root_path=${encodeURIComponent(rootPath)}` },
          { label: breadcrumbLabel },
        ]}
      />

      {error && (
        <ErrorAlert>{error}</ErrorAlert>
      )}

      {/* Report Header */}
      <ReportHeader
        modelName={modelName}
        datasetName={primaryDataset}
        datasets={datasets}
        score={overallMetric.score}
        metricName={overallMetric.metricName}
        totalSamples={totalSamples}
        htmlReportUrl={htmlReportUrl}
        onDatasetClick={handleDatasetChange}
      />

      <Tabs
        tabs={tabs}
        activeKey={activeTab}
        onChange={(k) => setActiveTab(k as TabKey)}
        className="w-full justify-start rounded-b-none border-b-0 bg-[var(--bg-card)] px-5 pt-4 pb-2"
        panels={{
          'report-overview-panel': (
            <div className="rounded-b-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-5">
              <OverviewTab
                reports={reportList}
                reportName={reportName}
                rootPath={rootPath}
                taskConfig={data?.task_config}
                onDatasetClick={handleDatasetChange}
              />
            </div>
          ),
          'report-details-panel': renderDatasetPanel(
            <DetailsTab
              key={activeDataset}
              reportName={reportName}
              datasetName={activeDataset}
              rootPath={rootPath}
              perfMetrics={reportList.find((r) => r.dataset_name === activeDataset)?.perf_metrics}
              overallScore={reportList.find((r) => r.dataset_name === activeDataset)?.score}
              metricName={reportList.find((r) => r.dataset_name === activeDataset)?.metrics[0]?.name}
              onSubsetClick={handleSubsetClick}
            />,
          ),
          'report-predictions-panel': renderDatasetPanel(
            <PredictionsTab
              key={`${activeDataset}-${initialSubset ?? ''}`}
              reportName={reportName}
              datasetName={activeDataset}
              rootPath={rootPath}
              initialSubset={initialSubset}
            />,
          ),
        }}
      />
    </div>
  )
}
