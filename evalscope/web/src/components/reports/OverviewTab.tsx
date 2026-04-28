import { useMemo } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import type { ReportData } from '@/api/types'
import { getChartUrl } from '@/api/reports'
import Card from '@/components/ui/Card'
import Table, { scoreColor } from '@/components/ui/Table'
import PlotlyChart from '@/components/charts/PlotlyChart'
import ReportSummaryStats from './ReportSummaryStats'
import JsonViewer from '@/components/common/JsonViewer'

interface Props {
  reports: ReportData[]
  reportName: string
  rootPath: string
  taskConfig?: Record<string, unknown>
  onDatasetClick?: (dataset: string) => void
}

export default function OverviewTab({ reports, reportName, rootPath, taskConfig, onDatasetClick }: Props) {
  const { t } = useLocale()

  const tableData = useMemo(() => {
    return reports.map((r) => ({
      Dataset: r.dataset_name,
      Score: r.score,
      Samples: r.metrics[0]?.categories?.reduce((s, c) => s + c.num, 0) ?? 0,
    }))
  }, [reports])

  const columns = [
    {
      key: 'Dataset',
      label: 'Dataset',
      sortable: true,
      render: (row: Record<string, unknown>) => {
        const name = String(row.Dataset)
        if (onDatasetClick) {
          return (
            <button
              onClick={() => onDatasetClick(name)}
              className="text-[var(--accent)] hover:underline cursor-pointer bg-transparent border-none p-0 font-inherit text-left"
            >
              {name}
            </button>
          )
        }
        return name
      },
    },
    {
      key: 'Score',
      label: 'Score',
      sortable: true,
      render: (row: Record<string, unknown>) => {
        const score = Number(row.Score)
        const norm = score > 1 ? score / 100 : score
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
      key: 'Samples',
      label: 'Samples',
      sortable: true,
      render: (row: Record<string, unknown>) => (
        <span className="text-[var(--text-muted)]">{Number(row.Samples).toLocaleString()}</span>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-6">
      {/* Summary Stats */}
      <ReportSummaryStats reports={reports} />

      {/* Scores Table */}
      <Card title={t('single.datasetScoresTable')}>
        <Table
          columns={columns}
          data={tableData}
          defaultSort={{ key: 'Score', dir: 'desc' }}
        />
      </Card>

      {/* Radar Chart */}
      <PlotlyChart
        src={getChartUrl(rootPath, 'radar', { reportName })}
        height={400}
        title={t('single.radarChart')}
      />

      {/* Task Config */}
      {taskConfig && Object.keys(taskConfig).length > 0 && (
        <Card title={t('reportDetail.taskConfig')} collapsible>
          <JsonViewer value={taskConfig} maxHeight={400} />
        </Card>
      )}
    </div>
  )
}
