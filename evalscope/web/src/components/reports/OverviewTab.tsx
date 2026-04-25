import { useMemo } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import type { ReportData } from '@/api/types'
import { getChartUrl } from '@/api/reports'
import Card from '@/components/ui/Card'
import Table, { scoreColor } from '@/components/ui/Table'
import PlotlyChart from '@/components/charts/PlotlyChart'
import ReportSummaryStats from './ReportSummaryStats'
import { prettyJson } from '@/utils/formatUtils'

interface Props {
  reports: ReportData[]
  reportName: string
  rootPath: string
  taskConfig?: Record<string, unknown>
}

export default function OverviewTab({ reports, reportName, rootPath, taskConfig }: Props) {
  const { t } = useLocale()

  const tableData = useMemo(() => {
    return reports.map((r) => ({
      Dataset: r.dataset_name,
      Score: r.score,
      Samples: r.metrics[0]?.categories?.reduce((s, c) => s + c.num, 0) ?? 0,
    }))
  }, [reports])

  const columns = [
    { key: 'Dataset', label: 'Dataset' },
    {
      key: 'Score',
      label: 'Score',
      render: (row: Record<string, unknown>) => {
        const score = Number(row.Score)
        const norm = score > 1 ? score / 100 : score
        return (
          <span className="font-mono font-medium" style={{ color: scoreColor(norm) }}>
            {score.toFixed(4)}
          </span>
        )
      },
    },
    {
      key: 'Samples',
      label: 'Samples',
      render: (row: Record<string, unknown>) => (
        <span className="text-[var(--text-muted)]">{Number(row.Samples).toLocaleString()}</span>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-6">
      {/* Summary Stats */}
      <ReportSummaryStats reports={reports} />

      {/* Radar Chart */}
      <PlotlyChart
        src={getChartUrl(rootPath, 'radar', { reportName })}
        height={400}
        title={t('single.radarChart')}
      />

      {/* Scores Table */}
      <Card title={t('single.datasetScoresTable')}>
        <Table columns={columns} data={tableData} />
      </Card>

      {/* Task Config */}
      {taskConfig && Object.keys(taskConfig).length > 0 && (
        <Card title={t('reportDetail.taskConfig')} collapsible>
          <pre className="text-xs font-mono whitespace-pre-wrap break-all text-[var(--text-muted)] max-h-[400px] overflow-auto">
            {prettyJson(taskConfig)}
          </pre>
        </Card>
      )}
    </div>
  )
}
