import { useMemo } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { loadReport as apiLoadReport, getHtmlReportUrl } from '@/api/reports'
import type { ReportData } from '@/api/types'
import { formatReportRef } from '@/domain/report/reportRef'
import { datasetLabel, primaryMetricOf, reportSampleCount } from '@/domain/report/primaryMetrics'
import { formatMetricLabels, metricIdentityKey } from '@/domain/metric'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Tabs from '@/components/ui/Tabs'
import Skeleton from '@/components/ui/Skeleton'
import Select from '@/components/ui/Select'
import { ArrowLeft, ArrowRight } from 'lucide-react'
import ErrorAlert from '@/components/ui/ErrorAlert'
import ReportHeader from '@/components/reports/ReportHeader'
import DatasetNav from '@/components/reports/DatasetNav'
import ResultsTab from '@/components/reports/ResultsTab'
import PredictionsTab from '@/components/reports/PredictionsTab'
import DiagnosisTab from '@/components/reports/DiagnosisTab'
import type { DiagnosisFilters } from '@/api/reports'
import type { SampleKey } from '@/api/types'

type TabKey = 'results' | 'diagnosis' | 'predictions'

const SAMPLE_PARAMS = ['subset', 'pattern', 'sample', 'failure_type', 'attribution', 'finding_category', 'diagnosed', 'outcome', 'step', 'position']

function sampleNumber(value: string | null): number | undefined {
  return value !== null && /^\d+$/.test(value) && Number.isSafeInteger(Number(value)) ? Number(value) : undefined
}

export default function ReportDetailPage() {
  const { runId, modelId } = useParams<{ runId: string; modelId: string }>()
  const [searchParams, setSearchParams] = useSearchParams()
  const { t } = useLocale()

  const rootPath = searchParams.get('root_path') || './outputs'
  const reportName = useMemo(
    () => formatReportRef({ runId: runId ?? '', modelId: modelId ?? '' }),
    [runId, modelId],
  )

  const tab = searchParams.get('tab')
  const activeTab: TabKey = tab === 'diagnosis' || tab === 'predictions' ? tab : 'results'
  const focusStep = sampleNumber(searchParams.get('step'))
  const sampleIndex = sampleNumber(searchParams.get('sample'))
  const samplePosition = sampleNumber(searchParams.get('position'))
  const initialSubset = searchParams.get('subset') || undefined
  const setActiveTab = (value: TabKey) => setSearchParams(previous => {
    const next = new URLSearchParams(previous); next.set('tab', value); return next
  })
  const outcome = searchParams.get('outcome')
  const diagnosisFilters: DiagnosisFilters = {
    outcome: outcome === 'failed' || outcome === 'unscored' || outcome === 'risk' ? outcome : undefined,
    failure_type: searchParams.get('failure_type') || undefined,
    attribution: searchParams.get('attribution') || undefined,
    finding_category: searchParams.get('finding_category') || undefined,
  }

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
  // pick is scoped to the report it was made on, and only holds while it still
  // names one of the datasets the report carries.
  const pickedDataset = searchParams.get('dataset')
  const activeDataset = pickedDataset !== null && Boolean(data?.datasets.includes(pickedDataset))
    ? pickedDataset
    : (data?.datasets[0] ?? '')

  const reportList = useMemo<ReportData[]>(() => data?.report_list ?? [], [data])

  // Derive overall info from report list
  const modelName = reportList[0]?.model_name ?? modelId ?? ''
  // Prefer the loaded model name; fall back to the URL model id while the report loads.
  const breadcrumbLabel = reportList[0]?.model_name ?? modelId ?? ''
  const totalSamples = reportList.reduce((sum, report) => sum + reportSampleCount(report), 0)

  const datasets = data?.datasets ?? []
  const datasetLabels = useMemo(
    () => Object.fromEntries(reportList.map((report) => [report.dataset_name, datasetLabel(report)])),
    [reportList],
  )
  const htmlReportUrl = getHtmlReportUrl(rootPath, reportName)

  const activeReport = useMemo(() => {
    const report = reportList.find((r) => r.dataset_name === activeDataset)
    if (!report) return undefined
    const primaryMetric = primaryMetricOf(report)
    const semanticsByMetric = Object.fromEntries(
      report.metrics.map((metric) => [metricIdentityKey(metric.identity), metric.semantics]),
    )
    const labelsByMetric = formatMetricLabels(
      report.metrics.map((metric) => ({
        identity: metric.identity,
        semantics: metric.semantics,
        legacyName: metric.legacy_name,
      })),
    )
    return { report, primaryMetric, semanticsByMetric, labelsByMetric }
  }, [reportList, activeDataset])

  const handleDatasetChange = (ds: string) => {
    if (ds === activeDataset) return
    const next = new URLSearchParams(searchParams)
    next.set('dataset', ds)
    for (const key of SAMPLE_PARAMS) next.delete(key)
    setSearchParams(next)
  }

  const handleSubsetClick = (subset: string) => {
    const next = new URLSearchParams(searchParams)
    for (const key of SAMPLE_PARAMS) next.delete(key)
    next.set('dataset', activeDataset)
    next.set('subset', subset)
    next.set('tab', 'predictions')
    setSearchParams(next)
  }

  const locateSample = (target: TabKey, filters: DiagnosisFilters, sample?: SampleKey, step?: number, position?: number) => {
    const next = new URLSearchParams(searchParams)
    for (const key of SAMPLE_PARAMS) next.delete(key)
    next.set('tab', target)
    next.set('dataset', activeDataset)
    for (const [key,value] of Object.entries(filters)) {
      if (value !== undefined && value !== '') next.set(key === 'index' ? 'sample' : key === 'subset_name' ? 'subset' : key, String(value))
    }
    if (sample) { next.set('subset',sample.subset); next.set('sample',String(sample.index)) }
    if (step !== undefined) next.set('step', String(step))
    if (position !== undefined) next.set('position', String(position))
    if (next.toString() !== searchParams.toString()) setSearchParams(next)
  }

  const handleDiagnosisSamples = (filters: DiagnosisFilters, sample?: SampleKey, step?: number, position?: number) =>
    locateSample('predictions', filters, sample, step, position)

  const tabs = [
    { key: 'results', label: t('reportDetail.results'), panelId: 'report-results-panel' },
    { key: 'diagnosis', label: t('diagnosis.title'), panelId: 'report-diagnosis-panel' },
    { key: 'predictions', label: t('reportDetail.predictions'), panelId: 'report-predictions-panel' },
  ]

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
    <div className="page-enter flex flex-col gap-3 px-4">
      {/* Breadcrumb */}
      <Breadcrumb
        items={[
          { label: 'Reports', href: `/reports?root_path=${encodeURIComponent(rootPath)}` },
          { label: breadcrumbLabel },
        ]}
      />

      {error && <ErrorAlert>{error}</ErrorAlert>}

      <ReportHeader
        modelName={modelName}
        datasetCount={datasets.length}
        totalSamples={totalSamples}
        htmlReportUrl={htmlReportUrl}
        reportsUrl={`/reports?root_path=${encodeURIComponent(rootPath)}`}
      />
      <div className="flex min-w-0 items-start gap-4">
        {datasets.length > 1 && (
          <div className="hidden self-stretch lg:block">
            <DatasetNav reports={reportList} active={activeDataset} onChange={handleDatasetChange} />
          </div>
        )}
        <div className="min-w-0 flex-1 overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
          {datasets.length > 1 && (
            <div className="border-b border-[var(--border)] p-4 lg:hidden">
              <Select
                label={t('reportDetail.datasets')}
                value={activeDataset}
                options={datasets.map((dataset) => ({ value: dataset, label: datasetLabels[dataset] || dataset }))}
                onChange={handleDatasetChange}
              />
            </div>
          )}
          <Tabs
            keepMounted={['diagnosis']}
            tabs={tabs}
            activeKey={activeTab}
            onChange={(key) => setActiveTab(key as TabKey)}
            className="min-w-0 flex-1 justify-start overflow-x-auto rounded-b-none border-0 border-b border-[var(--border)] bg-[var(--bg-card)] px-2 py-2 sm:px-4 [&>button]:min-h-9 [&>button]:shrink-0 [&>button]:whitespace-nowrap [&>button]:px-3"
            actions={activeTab === 'predictions' ? <button type="button" onClick={() => setActiveTab('diagnosis')} className="inline-flex shrink-0 items-center gap-2 whitespace-nowrap px-4 text-sm text-[var(--accent)]"><ArrowLeft size={14} />{t('reportDetail.backToDiagnosis')}</button> : undefined}
            beforePanels={activeTab === 'predictions' || (activeTab === 'diagnosis' && datasets.length > 1) ? undefined :
              <div className="flex flex-wrap items-center justify-between gap-3 px-4 pt-3">
                <h2 className="break-words text-lg font-semibold" title={activeDataset}>
                  {datasetLabels[activeDataset] || activeDataset}
                </h2>
                {activeTab !== 'diagnosis' && (
                  <button
                    type="button"
                    onClick={() => setActiveTab('diagnosis')}
                    className="inline-flex min-h-11 items-center gap-2 text-sm text-[var(--accent)] hover:underline"
                  >
                    {t('diagnosis.open')}
                    <ArrowRight size={14} />
                  </button>
                )}
              </div>
            }
            panels={{
              'report-results-panel': (
                <div className="p-4">
                  <ResultsTab
                    key={`${rootPath}/${reportName}/${activeDataset}`}
                    reportName={reportName}
                    datasetName={activeDataset}
                    rootPath={rootPath}
                    report={activeReport?.report}
                    perfMetrics={activeReport?.report.perf_metrics}
                    semantics={activeReport?.primaryMetric?.semantics}
                    semanticsByMetric={activeReport?.semanticsByMetric}
                    labelsByMetric={activeReport?.labelsByMetric}
                    onSubsetClick={handleSubsetClick}
                    taskConfig={data?.task_config}
                  />
                </div>
              ),
              'report-diagnosis-panel': (
                <div className="p-4">
                  <DiagnosisTab
                    key={`${rootPath}/${reportName}/${activeDataset}`}
                    rootPath={rootPath}
                    reportName={reportName}
                    datasetName={activeDataset}
                    onSamples={handleDiagnosisSamples}
                    onLocate={(filters, sample, step, position) => locateSample('diagnosis', filters, sample, step, position)}
                    location={{
                      filters: diagnosisFilters,
                      sample: initialSubset && sampleIndex !== undefined ? { subset: initialSubset, index: sampleIndex } : undefined,
                      position: samplePosition,
                      step: focusStep,
                    }}
                    legacyAnalysis={activeReport?.report.analysis}
                    metrics={activeReport?.report.metrics}
                  />
                </div>
              ),
              'report-predictions-panel': (
                <div className="p-4">
                  <PredictionsTab
                    key={`${rootPath}/${reportName}/${activeDataset}/${initialSubset ?? ''}`}
                    reportName={reportName}
                    datasetName={activeDataset}
                    rootPath={rootPath}
                    initialSubset={initialSubset}
                    initialIndex={sampleIndex}
                    focusStep={focusStep}
                    diagnosisFilters={diagnosisFilters}
                    onDiagnosisFiltersChange={handleDiagnosisSamples}
                    report={activeReport?.report}
                  />
                </div>
              ),
            }}
          />
        </div>
      </div>
    </div>
  )
}
