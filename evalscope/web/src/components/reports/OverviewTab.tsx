import { useMemo, useState } from 'react'
import { Radar, Table2 } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import type { ReportData } from '@/api/types'
import { getChartUrl } from '@/api/reports'
import Card from '@/components/ui/Card'
import Table from '@/components/ui/Table'
import { formatMetricByKey } from '@/domain/metric/registry'
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
  const [scoreView, setScoreView] = useState<'table' | 'radar'>('table')
  const canShowRadar = reports.length >= 3

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
        const content = (
          <>
            <span className="block max-w-[72px] break-words sm:max-w-none">{name}</span>
            <span className="mt-0.5 block text-[10px] text-[var(--text-muted)] sm:hidden">
              {Number(row.Samples).toLocaleString()} {t('single.samples')}
            </span>
          </>
        )
        if (onDatasetClick) {
          return (
            <button
              onClick={() => onDatasetClick(name)}
              className="text-[var(--accent)] hover:underline cursor-pointer bg-transparent border-none p-0 font-inherit text-left"
            >
              {content}
            </button>
          )
        }
        return content
      },
    },
    {
      key: 'Score',
      label: 'Score',
      sortable: true,
      render: (row: Record<string, unknown>) => {
        const score = Number(row.Score)
        const norm = Math.max(0, Math.min(1, score))
        return (
          <div className="flex min-w-[92px] items-center gap-1.5 sm:min-w-[240px] sm:gap-3">
            <div className="h-2 min-w-9 flex-1 overflow-hidden rounded-full border border-[var(--border)] bg-[var(--bg-deep)] sm:h-2.5">
              <div
                role="progressbar"
                aria-label={`${String(row.Dataset)} ${t('prediction.score')}`}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(norm * 100)}
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: `${Math.min(100, norm * 100)}%`,
                  background: 'var(--accent)',
                }}
              />
            </div>
            <span className="w-12 text-right font-mono text-xs font-semibold tabular-nums text-[var(--text)] sm:w-16 sm:text-sm">
              {formatMetricByKey('score', score, t).primary}
            </span>
          </div>
        )
      },
    },
    {
      key: 'Samples',
      label: 'Samples',
      sortable: true,
      headerClassName: 'hidden sm:table-cell',
      cellClassName: 'hidden sm:table-cell',
      render: (row: Record<string, unknown>) => (
        <span className="text-[var(--text-muted)]">{Number(row.Samples).toLocaleString()}</span>
      ),
    },
  ]

  return (
    <div className="flex flex-col gap-6">
      {/* Summary Stats */}
      <ReportSummaryStats reports={reports} />

      <Card title={t('single.datasetScores')}>
        {canShowRadar && (
          <div className="mb-4 flex justify-end">
            <div className="inline-flex rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-deep)] p-1">
              {([
                ['table', t('single.tableView'), Table2],
                ['radar', t('single.radarView'), Radar],
              ] as const).map(([view, label, Icon]) => (
                <button
                  key={view}
                  type="button"
                  aria-pressed={scoreView === view}
                  onClick={() => setScoreView(view)}
                  className={`inline-flex min-h-9 items-center gap-1.5 rounded-[var(--radius-xs)] px-3 type-button-sm transition-colors ${
                    scoreView === view
                      ? 'bg-[var(--bg-card)] text-[var(--text)] shadow-[var(--shadow-sm)]'
                      : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                  }`}
                >
                  <Icon size={14} aria-hidden="true" />
                  {label}
                </button>
              ))}
            </div>
          </div>
        )}

        {scoreView === 'radar' && canShowRadar ? (
          <PlotlyChart
            src={getChartUrl(rootPath, 'radar', { reportName })}
            height={400}
            fallbackTable={{
              columns: ['Dataset', 'Score', 'Samples'],
              rows: tableData,
              scoreColumns: ['Score'],
            }}
          />
        ) : (
          <Table
            columns={columns}
            data={tableData}
            defaultSort={{ key: 'Score', dir: 'desc' }}
            className="[&_th]:px-2 [&_td]:px-2 sm:[&_th]:px-4 sm:[&_td]:px-4"
          />
        )}
      </Card>

      {/* Task Config */}
      {taskConfig && Object.keys(taskConfig).length > 0 && (
        <Card title={t('reportDetail.taskConfig')} collapsible>
          <JsonViewer value={taskConfig} maxHeight={400} />
        </Card>
      )}
    </div>
  )
}
