import { useMemo, useState } from 'react'
import { Radar, Table2 } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import type { ReportData } from '@/api/types'
import { getChartUrl } from '@/api/reports'
import Card from '@/components/ui/Card'
import Table from '@/components/ui/Table'
import { formatMetric, getBoundedQualityRatio, getValuePosition } from '@/domain/metric'
import { scoreColor } from '@/utils/colorScale'
import type { MetricSemantics } from '@/domain/metric'
import { directionArrow, primaryMetricOf } from '@/domain/report/primaryMetrics'
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

/**
 * Headline figures of one dataset report: the primary metric's name, score, sample count and
 * semantics. `null` when the report declares no primary metric.
 */
function primarySummaryOf(report: ReportData): { name: string; score: number; num: number; semantics?: MetricSemantics | null } | null {
  const metric = primaryMetricOf(report)
  if (!metric) {
    return null
  }
  return {
    name: metric.name,
    score: metric.score,
    num: metric.categories?.reduce((sum, category) => sum + category.num, 0) ?? 0,
    semantics: metric.semantics,
  }
}

export default function OverviewTab({ reports, reportName, rootPath, taskConfig, onDatasetClick }: Props) {
  const { t } = useLocale()
  const [scoreView, setScoreView] = useState<'table' | 'radar'>('table')
  const primaries = reports.map(primaryMetricOf)
  const semanticIds = primaries.map((primary) => primary?.semantics?.semantic_id ?? null)
  const sameSemantics = semanticIds.length > 0 && semanticIds.every((id) => id !== null && id === semanticIds[0])
  // A radar chart puts every dataset on one axis scale, so it is only honest when the datasets
  // share a semantic identifier and the metric is a bounded quality metric.
  const canShowRadar = reports.length >= 3
    && sameSemantics
    && primaries[0]?.semantics?.value_range != null
    && primaries[0]?.semantics?.direction !== 'none'

  const tableData = useMemo(() => {
    // `primaries` is recomputed here rather than closed over: it is a fresh array on every render, so
    // listing it as a dependency would defeat the memo, and omitting it is what the exhaustive-deps
    // rule warns about. Deriving it inside keeps `reports` the only real input.
    return reports.map((r) => {
      const primary = primarySummaryOf(r)
      return {
        Dataset: r.dataset_name,
        Score: primary?.score ?? null,
        Metric: primary?.name ?? '',
        Samples: primary?.num ?? 0,
      }
    })
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
      key: 'Metric',
      label: t('reportDetail.metric'),
      sortable: true,
      render: (row: Record<string, unknown>) => {
        const metricName = String(row.Metric ?? '')
        const semantics = primaries.find((primary) => primary?.name === metricName)?.semantics
        return (
          <span
            className="truncate text-xs text-[var(--text-muted)] sm:text-sm"
            title={metricName}
          >
            {semantics ? `${semantics.metric_name} ${directionArrow(semantics)}`.trimEnd() : metricName}
          </span>
        )
      },
    },
    {
      key: 'Score',
      label: 'Score',
      sortable: true,
      render: (row: Record<string, unknown>) => {
        const score = row.Score == null ? null : Number(row.Score)
        const metricName = String(row.Metric ?? '')
        const semantics = primaries.find((primary) => primary?.name === metricName)?.semantics
        // The bar length is the value's own position in its own range, so two different metrics
        // never draw the same length: an F1 of 91.2% is long, a WER of 4.3% is short. The colour
        // carries the quality, so that short WER bar is green. Sizing by quality instead is what
        // used to make those two bars look identical.
        const position = getValuePosition(score, semantics)
        const quality = getBoundedQualityRatio(score, semantics)
        return (
          <div className="flex items-center justify-end gap-3">
            {position != null && (
              // The track grows to fill the column instead of leaving a gap beside a fixed-width
              // bar, which also makes the length difference between two rows easier to read.
              <div className="hidden h-1.5 min-w-9 flex-1 overflow-hidden rounded-full bg-[var(--border)] sm:block">
                <div
                  role="progressbar"
                  aria-label={`${String(row.Dataset)} ${metricName}`}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(position * 100)}
                  className="h-full rounded-full transition-all duration-300"
                  style={{ width: `${position * 100}%`, background: scoreColor(quality ?? position) }}
                />
              </div>
            )}
            <span
              className="min-w-14 shrink-0 text-right font-mono text-xs font-semibold tabular-nums sm:text-sm"
              style={{ color: quality == null ? 'var(--text)' : scoreColor(quality) }}
            >
              {formatMetric(score, semantics).primary}
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
