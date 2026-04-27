import { useEffect, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { getAnalysis, getDataFrame, getChartUrl } from '@/api/reports'
import Card from '@/components/ui/Card'
import Table, { scoreColor } from '@/components/ui/Table'
import PlotlyChart from '@/components/charts/PlotlyChart'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import Skeleton from '@/components/ui/Skeleton'
import PerfMetricsPanel from '@/components/reports/PerfMetricsPanel'
import type { PerfMetrics } from '@/api/types'

interface Props {
  reportName: string
  datasetName: string
  rootPath: string
  perfMetrics?: PerfMetrics
}

export default function DetailsTab({ reportName, datasetName, rootPath, perfMetrics }: Props) {
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

  const subsetColumns = [
    { key: 'Subset', label: 'Subset' },
    {
      key: 'Score',
      label: 'Score',
      render: (row: Record<string, unknown>) => {
        const score = Number(row.Score ?? 0)
        const norm = score > 1 ? score / 100 : score
        return (
          <span className="font-mono font-medium" style={{ color: scoreColor(norm) }}>
            {score.toFixed(4)}
          </span>
        )
      },
    },
    {
      key: 'Num',
      label: 'Samples',
      render: (row: Record<string, unknown>) => (
        <span className="text-[var(--text-muted)]">{Number(row.Num ?? 0).toLocaleString()}</span>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-6">
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

      {/* Subset Scores Table */}
      {subsetData.data.length > 0 && (
        <Card title={t('reportDetail.subsetScores')}>
          <Table columns={subsetColumns} data={subsetData.data} />
        </Card>
      )}

      {/* Dataset Score Chart */}
      <PlotlyChart
        src={getChartUrl(rootPath, 'dataset_scores', { reportName, datasetName })}
        height={350}
        title={t('single.datasetScores')}
      />

      {/* Performance Metrics */}
      {perfMetrics && (
        <Card title={t('reportDetail.perfMetrics')}>
          <PerfMetricsPanel perfMetrics={perfMetrics} />
        </Card>
      )}
    </div>
  )
}
