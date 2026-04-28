import { useEffect, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { getAnalysis, getDataFrame } from '@/api/reports'
import Card from '@/components/ui/Card'
import Table, { scoreColor } from '@/components/ui/Table'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import Skeleton from '@/components/ui/Skeleton'
import PerfMetricsPanel from '@/components/reports/PerfMetricsPanel'
import type { PerfMetrics } from '@/api/types'

interface Props {
  reportName: string
  datasetName: string
  rootPath: string
  perfMetrics?: PerfMetrics
  onSubsetClick?: (subset: string) => void
  overallScore?: number
}

export default function DetailsTab({ reportName, datasetName, rootPath, perfMetrics, onSubsetClick, overallScore }: Props) {
  const { t } = useLocale()
  const [analysis, setAnalysis] = useState('')
  const [analysisLoading, setAnalysisLoading] = useState(false)
  const [subsetData, setSubsetData] = useState<{ columns: string[]; data: Record<string, unknown>[] }>({
    columns: [],
    data: [],
  })

  useEffect(() => {
    if (!datasetName || !reportName) return
    let cancelled = false

    const load = async () => {
      setAnalysisLoading(true)
      try {
        const [analysisText, dfRes] = await Promise.all([
          getAnalysis(rootPath, reportName, datasetName).catch(() => ''),
          getDataFrame(rootPath, reportName, 'dataset', datasetName).catch(() => ({ columns: [], data: [] })),
        ])
        if (cancelled) return
        setAnalysis(analysisText)
        setSubsetData({ columns: dfRes.columns, data: dfRes.data })
      } finally {
        if (!cancelled) setAnalysisLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [datasetName, reportName, rootPath])

  // Detect whether data has Metric column
  const hasMetricCol = subsetData.data.length > 0 && 'Metric' in subsetData.data[0]

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
      render: (row: Record<string, unknown>) => (
        <span className="text-[var(--text-muted)] text-xs font-mono">{String(row.Metric ?? '')}</span>
      ),
    }] : []),
    {
      key: 'Score',
      label: 'Score',
      sortable: true,
      render: (row: Record<string, unknown>) => {
        const score = Number(row.Score ?? 0)
        const norm = score > 1 ? score / 100 : score
        // Inline score bar
        return (
          <div className="flex items-center gap-2">
            <div
              className="h-1.5 rounded-full bg-[var(--border)] overflow-hidden"
              style={{ width: '60px', minWidth: '60px' }}
            >
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: `${Math.min(100, norm * 100)}%`,
                  background: scoreColor(norm),
                }}
              />
            </div>
            <span className="font-mono font-medium tabular-nums" style={{ color: scoreColor(norm) }}>
              {score.toFixed(4)}
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

  const normOverall = overallScore != null ? (overallScore > 1 ? overallScore / 100 : overallScore) : null

  return (
    <div className="flex flex-col gap-6">
      {/* Overall Score Stat */}
      {normOverall != null && (
        <div className="flex items-center gap-3 p-4 rounded-[var(--radius)] bg-[var(--bg-card2)] border border-[var(--border)]">
          <div className="flex flex-col gap-0.5">
            <span className="text-xs text-[var(--text-muted)] uppercase tracking-wide">
              {t('reportDetail.overallScore')}
            </span>
            <span
              className="text-3xl font-bold font-mono tabular-nums"
              style={{ color: scoreColor(normOverall) }}
            >
              {(normOverall * 100).toFixed(2)}
            </span>
          </div>
          {/* mini progress ring */}
          <svg width="48" height="48" viewBox="0 0 48 48" style={{ transform: 'rotate(-90deg)' }}>
            <circle cx="24" cy="24" r="20" fill="none" stroke="var(--border)" strokeWidth="4" />
            <circle
              cx="24" cy="24" r="20" fill="none"
              stroke={scoreColor(normOverall)}
              strokeWidth="4"
              strokeDasharray={`${2 * Math.PI * 20}`}
              strokeDashoffset={`${2 * Math.PI * 20 * (1 - normOverall)}`}
              strokeLinecap="round"
            />
          </svg>
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
          <p className="text-sm text-[var(--text-dim)]">{t('common.noData')}</p>
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
