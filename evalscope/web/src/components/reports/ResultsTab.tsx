import { useLocale } from '@/contexts/LocaleContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { getDataFrame } from '@/api/reports'
import Table from '@/components/ui/Table'
import { formatMetricLabel } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'
import JsonViewer from '@/components/ui/JsonViewer'
import ErrorAlert from '@/components/ui/ErrorAlert'
import ScoreBar from '@/components/ui/ScoreBar'
import Skeleton from '@/components/ui/Skeleton'
import PerfMetricsPanel from '@/components/reports/PerfMetricsPanel'
import type { PerfMetrics, ReportData } from '@/api/types'
import DecisionDiagnostics from './DecisionDiagnostics'

interface Props {
  reportName: string
  datasetName: string
  rootPath: string
  perfMetrics?: PerfMetrics | null
  report?: ReportData
  onSubsetClick?: (subset: string) => void
  taskConfig?: Record<string, unknown>
  /** Backend semantics of the primary metric, and of each metric by its final report name. */
  semantics?: MetricSemantics | null
  semanticsByMetric?: Record<string, MetricSemantics | null | undefined>
  /** Display label of each metric by its final report name, from `formatMetricLabels`. */
  labelsByMetric?: Record<string, string>
}

/** Stable placeholder so an unresolved frame does not produce a new object each render. */
const EMPTY_FRAME: { columns: string[]; data: Record<string, unknown>[] } = { columns: [], data: [] }

export default function ResultsTab({
  reportName,
  datasetName,
  rootPath,
  perfMetrics,
  report,
  onSubsetClick,
  taskConfig,
  semantics,
  semanticsByMetric = {},
  labelsByMetric = {},
}: Props) {
  const { t } = useLocale()

  const details = useAsyncResource(
    (signal) => getDataFrame(rootPath, reportName, 'dataset', datasetName, signal),
    [rootPath, reportName, datasetName],
    { enabled: Boolean(datasetName && reportName), fallbackMessage: t('common.loadError') },
  )
  const subsetData = details.data ?? EMPTY_FRAME

  // Detect whether data has Metric column
  const hasMetricCol = subsetData.data.length > 0 && 'Metric' in subsetData.data[0]

  const subsetColumns = [
    {
      key: 'Subset',
      label: t('reportDetail.subset'),
      sortable: true,
      render: (row: Record<string, unknown>) => {
        const name = String(row.Subset ?? '')
        if (onSubsetClick) {
          return (
            <button
              onClick={() => onSubsetClick(name)}
              className="text-[var(--accent)] hover:underline cursor-pointer bg-transparent border-none p-0 font-inherit text-left"
              title={t('reportDetail.viewPredictions')}
            >
              {name}
            </button>
          )
        }
        return <span>{name}</span>
      },
    },
    ...(hasMetricCol
      ? [
          {
            key: 'Metric',
            label: t('reportDetail.metric'),
            sortable: true,
            render: (row: Record<string, unknown>) => {
              const metricName = String(row.Metric ?? '')
              const rowSemantics = semanticsByMetric[metricName] ?? null
              // The metric's display name, direction and dimensions, the same label the CLI table and the
              // HTML report print. The raw name stays reachable through the tooltip.
              return (
                <span className="text-xs break-words text-[var(--text-muted)]" title={metricName}>
                  {labelsByMetric[metricName] ?? formatMetricLabel(metricName, rowSemantics)}
                </span>
              )
            },
          },
        ]
      : []),
    {
      key: 'Score',
      label: t('reportDetail.score'),
      sortable: true,
      render: (row: Record<string, unknown>) => {
        const score = Number(row.Score ?? 0)
        // Each row names its own metric, so a report mixing metrics formats each row correctly.
        const rowSemantics = semanticsByMetric[String(row.Metric ?? '')] ?? semantics
        // Subsets of one dataset share a metric, so a bar is comparable across these rows.
        return (
          <ScoreBar
            score={score}
            semantics={rowSemantics}
            ariaLabel={`${String(row.Subset ?? '')} ${String(row.Metric ?? '')}`.trim()}
            track="fixed"
          />
        )
      },
    },
    {
      key: 'Num',
      label: t('reportDetail.sampleCount'),
      sortable: true,
      render: (row: Record<string, unknown>) => (
        <span className="text-[var(--text-muted)]">{Number(row.Num ?? 0).toLocaleString()}</span>
      ),
    },
    // The backend emits this column only for binary, higher-is-better metrics; rows of other
    // metrics in the same frame carry '-'.
    ...(subsetData.columns.includes('Correct')
      ? [
          {
            key: 'Correct',
            label: t('reportDetail.correct'),
            sortable: true,
            render: (row: Record<string, unknown>) => (
              <span className="text-[var(--text-muted)]">
                {row.Correct === '-' ? '-' : Number(row.Correct ?? 0).toLocaleString()}
              </span>
            ),
          },
        ]
      : []),
  ]

  return (
    <div className="flex flex-col gap-6">
      {details.loading && <Skeleton lines={4} />}
      {details.error && <ErrorAlert>{details.error}</ErrorAlert>}
      {/* Subset Scores Table */}
      {subsetData.data.length > 0 && (
        <section className="space-y-4">
          <h3 className="text-sm font-semibold">{t('reportDetail.subsetScores')}</h3>
          <Table columns={subsetColumns} data={subsetData.data} defaultSort={{ key: 'Score', dir: 'desc' }} />
        </section>
      )}

      <DecisionDiagnostics report={report} />

      {/* Performance Metrics */}
      {perfMetrics?.summary && (
        <section className="space-y-4 border-t border-[var(--border)] pt-6">
          <h3 className="text-sm font-semibold">{t('reportDetail.perfMetrics')}</h3>
          <PerfMetricsPanel perfMetrics={perfMetrics} compact />
        </section>
      )}
      {taskConfig && Object.keys(taskConfig).length > 0 && (
        <details open className="rounded-[var(--radius)] border border-[var(--border)]">
          <summary className="cursor-pointer px-5 py-3 text-sm font-medium">{t('reportDetail.taskConfig')}</summary>
          <div className="border-t border-[var(--border)] p-5">
            <JsonViewer value={taskConfig} maxHeight={400} />
          </div>
        </details>
      )}
    </div>
  )
}
