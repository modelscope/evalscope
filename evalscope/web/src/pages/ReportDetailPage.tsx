import { useMemo, useState, type ReactNode } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { loadReport as apiLoadReport, getHtmlReportUrl } from '@/api/reports'
import type { ReportData } from '@/api/types'
import { formatReportRef } from '@/domain/report/reportRef'
import { datasetLabel, primaryMetricOf } from '@/domain/report/primaryMetrics'
import { formatMetricIdentityLabel, metricIdentityKey } from '@/domain/metric'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Tabs from '@/components/ui/Tabs'
import Skeleton from '@/components/ui/Skeleton'
import ErrorAlert from '@/components/ui/ErrorAlert'
import ReportHeader from '@/components/reports/ReportHeader'
import DatasetNav from '@/components/reports/DatasetNav'
import OverviewTab from '@/components/reports/OverviewTab'
import DetailsTab from '@/components/reports/DetailsTab'
import PredictionsTab from '@/components/reports/PredictionsTab'

type TabKey = 'overview' | 'details' | 'predictions'

export default function ReportDetailPage() {
  const { runId, modelId } = useParams<{ runId: string; modelId: string }>()
  const [searchParams] = useSearchParams()
  const { t } = useLocale()

  const rootPath = searchParams.get('root_path') || './outputs'
  const reportName = useMemo(
    () => formatReportRef({ runId: runId ?? '', modelId: modelId ?? '' }),
    [runId, modelId],
  )

  const [activeTab, setActiveTab] = useState<TabKey>('overview')
  const [pickedDataset, setPickedDataset] = useState<{ scope: string; name: string } | null>(null)
  const [initialSubset, setInitialSubset] = useState<string | undefined>(undefined)

  // A change of inputs aborts the previous request and drops its late response,
  // so only the newest one updates the view.
  const report = useAsyncResource(
    (signal) => apiLoadReport(rootPath, reportName, signal),
    [rootPath, reportName],
    { enabled: Boolean(reportName), fallbackMessage: t('common.loadError') },
  )
  const data = report.data ?? null
  const loading = report.loading
  const error = report.error

  // Open on the report's first dataset, while still letting the user switch; the
  // pick is scoped to the report it was made on.
  const datasetScope = `${rootPath}\0${reportName}`
  const pickIsLoaded = pickedDataset?.scope === datasetScope
    && Boolean(data?.datasets.includes(pickedDataset.name))
  const activeDataset = pickIsLoaded ? pickedDataset.name : (data?.datasets[0] ?? '')
  const setActiveDataset = (name: string) => setPickedDataset({ scope: datasetScope, name })

  const reportList = useMemo<ReportData[]>(() => data?.report_list ?? [], [data])

  // Derive overall info from report list
  const modelName = reportList[0]?.model_name ?? modelId ?? ''
  // Prefer the loaded model name; fall back to the URL model id while the report loads.
  const breadcrumbLabel = reportList[0]?.model_name ?? modelId ?? ''
  const primaryDataset = reportList[0] ? datasetLabel(reportList[0]) : ''
  const overallMetric = useMemo(() => {
    if (reportList.length !== 1) return { score: null, semantics: null, metricName: '' }
    const primary = primaryMetricOf(reportList[0])
    return {
      score: primary?.score ?? null,
      semantics: primary?.semantics ?? null,
      metricName: primary ? formatMetricIdentityLabel(primary.identity, primary.semantics, primary.legacy_name) : '',
    }
  }, [reportList])
  const totalSamples = reportList.reduce((sum, r) => {
    const primary = primaryMetricOf(r)
    return sum + (primary?.categories?.reduce((s, c) => s + c.num, 0) ?? 0)
  }, 0)

  const datasets = data?.datasets ?? []
  const datasetLabels = useMemo(
    () => Object.fromEntries(reportList.map((report) => [report.dataset_name, datasetLabel(report)])),
    [reportList],
  )
  const htmlReportUrl = getHtmlReportUrl(rootPath, reportName)

  // Semantics of the dataset currently shown in the details panel: its primary metric drives the
  // headline number, and the per-metric map lets each row format itself.
  const activeReport = useMemo(() => {
    const report = reportList.find((r) => r.dataset_name === activeDataset)
    if (!report) return undefined
    const primaryMetric = primaryMetricOf(report)
    const semanticsByMetric = Object.fromEntries(
      report.metrics.map((metric) => [metricIdentityKey(metric.identity), metric.semantics]),
    )
    return { primaryMetric, semanticsByMetric }
  }, [reportList, activeDataset])

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
                <span title={ds}>{datasetLabels[ds] || ds}</span>
              </button>
            ))}
          </div>
          <div className="hidden md:block">
            <DatasetNav datasets={datasets} labels={datasetLabels} active={activeDataset} onChange={handleDatasetChange} />
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
        datasetLabels={datasetLabels}
        score={overallMetric.score}
        metricName={overallMetric.metricName}
        semantics={overallMetric.semantics}
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
              overallScore={activeReport?.primaryMetric?.score}
              metricName={activeReport?.primaryMetric
                ? formatMetricIdentityLabel(
                    activeReport.primaryMetric.identity,
                    activeReport.primaryMetric.semantics,
                    activeReport.primaryMetric.legacy_name,
                  )
                : undefined}
              semantics={activeReport?.primaryMetric?.semantics}
              semanticsByMetric={activeReport?.semanticsByMetric}
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
