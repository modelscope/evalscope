import { useEffect, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { getAnalysis, getDataFrame } from '@/api/reports'
import Card from '@/components/ui/Card'
import Table from '@/components/ui/Table'
import { scoreColor } from '@/utils/colorScale'
import {
  formatMetric,
  formatMetricLabel,
  formatMetricLabels,
  getBoundedQualityRatio,
  getValuePosition,
} from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import Skeleton from '@/components/ui/Skeleton'
import PerfMetricsPanel from '@/components/reports/PerfMetricsPanel'
import type { PerfMetrics } from '@/api/types'

interface Props {
  reportName: string
  datasetName: string
  rootPath: string
  perfMetrics?: PerfMetrics | null
  onSubsetClick?: (subset: string) => void
  overallScore?: number
  metricName?: string
  /** Backend semantics of the primary metric, and of each metric by its final report name. */
  semantics?: MetricSemantics | null
  semanticsByMetric?: Record<string, MetricSemantics | null | undefined>
}

export default function DetailsTab({
  reportName,
  datasetName,
  rootPath,
  perfMetrics,
  onSubsetClick,
  overallScore,
  metricName,
  semantics,
  semanticsByMetric = {},
}: Props) {
  const { t } = useLocale()
  const [analysis, setAnalysis] = useState('')
  const [analysisLoading, setAnalysisLoading] = useState(false)
  const [subsetData, setSubsetData] = useState<{ columns: string[]; data: Record<string, unknown>[] }>({
    columns: [],
    data: [],
  })

  useEffect(() => {
    if (!datasetName || !reportName) return
    const controller = new AbortController()

    const load = async () => {
      setAnalysisLoading(true)
      try {
        const [analysisText, dfRes] = await Promise.all([
          getAnalysis(rootPath, reportName, datasetName, controller.signal).catch(() => ''),
          getDataFrame(rootPath, reportName, 'dataset', datasetName, controller.signal).catch(() => ({ columns: [], data: [] })),
        ])
        if (controller.signal.aborted) return
        setAnalysis(analysisText)
        setSubsetData({ columns: dfRes.columns, data: dfRes.data })
      } finally {
        if (!controller.signal.aborted) setAnalysisLoading(false)
      }
    }
    load()
    return () => controller.abort()
  }, [datasetName, reportName, rootPath])

  // Detect whether data has Metric column
  const hasMetricCol = subsetData.data.length > 0 && 'Metric' in subsetData.data[0]
  const metricLabels = formatMetricLabels(
    Object.entries(semanticsByMetric).map(([name, metricSemantics]) => ({
      metricName: name,
      semantics: metricSemantics,
    })),
  )

  const subsetColumns = [
    {
      key: 'Subset',
      label: 'Subset',
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
    ...(hasMetricCol ? [{
      key: 'Metric',
      label: t('reportDetail.metric'),
      sortable: true,
      render: (row: Record<string, unknown>) => {
        const metricName = String(row.Metric ?? '')
        const rowSemantics = semanticsByMetric[metricName] ?? null
        // The metric's display name and direction, the same label the header card and every other
        // surface uses. The raw name stays reachable through the tooltip.
        return (
          <span className="text-xs text-[var(--text-muted)]" title={metricName}>
            {metricLabels[metricName] ?? formatMetricLabel(metricName, rowSemantics)}
          </span>
        )
      },
    }] : []),
    {
      key: 'Score',
      label: 'Score',
      sortable: true,
      render: (row: Record<string, unknown>) => {
        const score = Number(row.Score ?? 0)
        // Each row names its own metric, so a report mixing metrics formats each row correctly.
        const rowSemantics = semanticsByMetric[String(row.Metric ?? '')] ?? semantics
        // Subsets of one dataset share a metric, so a bar is comparable across these rows. Its
        // length is the value's own position in its range and is never inverted; the colour is
        // what says whether that value is good. A low error rate is therefore a short green bar.
        const position = getValuePosition(score, rowSemantics)
        const quality = getBoundedQualityRatio(score, rowSemantics)
        return (
          <div className="flex items-center gap-2">
            {position != null && (
              <div className="h-1.5 w-[60px] min-w-[60px] rounded-full bg-[var(--border)] overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{ width: `${position * 100}%`, background: scoreColor(quality ?? position) }}
                />
              </div>
            )}
            <span
              className="font-mono font-medium tabular-nums"
              style={{ color: quality == null ? 'var(--text)' : scoreColor(quality) }}
            >
              {formatMetric(score, rowSemantics).primary}
            </span>
          </div>
        )
      },
    },
    {
      key: 'Num',
      label: 'Num',
      sortable: true,
      render: (row: Record<string, unknown>) => (
        <span className="text-[var(--text-muted)]">{Number(row.Num ?? 0).toLocaleString()}</span>
      ),
    },
  ]

  // The arc length is the value's own position in its range; the colour says how good it is. A
  // 4.3% WER therefore draws a small green arc, rather than a near-full one.
  const overallPosition = getValuePosition(overallScore, semantics)
  const overallQuality = getBoundedQualityRatio(overallScore, semantics)

  return (
    <div className="flex flex-col gap-6">
      {/* Overall Score Stat */}
      {overallScore != null && (
        <div className="flex items-center gap-3 p-4 rounded-[var(--radius)] bg-[var(--bg-card2)] border border-[var(--border)]">
          <div className="flex flex-col gap-0.5">
            <span className="text-xs text-[var(--text-muted)] uppercase tracking-wide">
              {semantics
                ? metricName ?? formatMetricLabel(semantics.metric_name, semantics)
                : t('reportDetail.overallScore')}
            </span>
            <span
              className="text-3xl font-bold font-mono tabular-nums"
              style={{ color: overallQuality == null ? 'var(--text)' : scoreColor(overallQuality) }}
            >
              {formatMetric(overallScore, semantics).primary}
            </span>
          </div>
          {overallPosition != null && (
            <svg width="48" height="48" viewBox="0 0 48 48" style={{ transform: 'rotate(-90deg)' }}>
              <circle cx="24" cy="24" r="19" fill="none" stroke="var(--border)" strokeWidth="6" />
              <circle
                cx="24" cy="24" r="19" fill="none"
                stroke={scoreColor(overallQuality ?? overallPosition)}
                strokeWidth="6"
                strokeDasharray={`${2 * Math.PI * 19}`}
                strokeDashoffset={`${2 * Math.PI * 19 * (1 - overallPosition)}`}
                strokeLinecap="round"
              />
            </svg>
          )}
        </div>
      )}

      {/* Subset Scores Table */}
      {subsetData.data.length > 0 && (
        <Card title={t('reportDetail.subsetScores')}>
          <Table
            columns={subsetColumns}
            data={subsetData.data}
            defaultSort={{ key: 'Score', dir: 'desc' }}
          />
        </Card>
      )}

      {/* AI Analysis */}
      <Card title={t('reportDetail.analysis')}>
        {analysisLoading ? (
          <Skeleton lines={5} />
        ) : analysis && analysis !== 'N/A' ? (
          <MarkdownRenderer content={analysis} />
        ) : (
          <p className="text-sm text-[var(--text-muted)]">{t('common.noData')}</p>
        )}
      </Card>

      {/* Score Distribution Chart removed - info already visible in Subset Scores table */}

      {/* Performance Metrics */}
      {perfMetrics && (
        <Card title={t('reportDetail.perfMetrics')}>
          <PerfMetricsPanel perfMetrics={perfMetrics} />
        </Card>
      )}
    </div>
  )
}
