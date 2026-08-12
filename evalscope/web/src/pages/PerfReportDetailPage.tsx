import { useMemo } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useScan } from '@/contexts/ReportsContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { useScopedState } from '@/hooks/useScopedState'
import { useQueryParams } from '@/hooks/useQueryParams'
import { getPerfDetail, getPerfChartUrl, getPerfHistoryReportUrl } from '@/api/perf'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Tabs from '@/components/ui/Tabs'
import Card from '@/components/ui/Card'
import Skeleton from '@/components/ui/Skeleton'
import KpiStrip from '@/components/ui/KpiStrip'
import LabelledField from '@/components/ui/LabelledField'
import ErrorAlert from '@/components/ui/ErrorAlert'
import PerfChartGroup from '@/components/perf/PerfChartGroup'
import PerfRunsTab from '@/components/perf/PerfRunsTab'
import { LATENCY_CHARTS, THROUGHPUT_CHARTS } from '@/domain/perf/charts'
import { formatTimestamp } from '@/utils/formatUtils'
import { resolveProvider } from '@/domain/perf/providerResolution'
import { ExternalLink, Lightbulb } from 'lucide-react'

type TabKey = 'overview' | 'charts' | 'runs'

/** Return a shallow copy of `info` with the given keys removed. */
function omitKeys(info: Record<string, string>, keys: string[]): Record<string, string> {
  const drop = new Set(keys)
  return Object.fromEntries(Object.entries(info).filter(([k]) => !drop.has(k)))
}

// ------------------------------------------------------------------ //
// Overview building blocks                                            //
// ------------------------------------------------------------------ //
function formatSummaryCell(column: string, cell: string | number, t: (key: string) => string): string {
  if (column.trim().toLowerCase() === 'rate' && String(cell).trim().toUpperCase() === 'INF') {
    return t('perf.archive.closedLoop')
  }
  return String(cell)
}

function SummaryTable({ columns, rows, t }: { columns: string[]; rows: (string | number)[][]; t: (key: string) => string }) {
  if (columns.length === 0) return null
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c}
                className="type-table-xs px-3 py-2 text-right first:text-left whitespace-nowrap border-b border-[var(--border)]"
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} className={ri < rows.length - 1 ? 'border-b border-[var(--border)]' : ''}>
              {row.map((cell, ci) => (
                <td
                  key={ci}
                  className="type-body-sm tabular-nums px-3 py-2 text-right first:text-left whitespace-nowrap text-[var(--text)]"
                >
                  {formatSummaryCell(columns[ci] ?? '', cell, t)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function rowsToRecords(columns: string[], rows: (string | number)[][]): Record<string, unknown>[] {
  return rows.map((row) => Object.fromEntries(columns.map((column, index) => [column, row[index]])))
}

// ------------------------------------------------------------------ //
// Page                                                                //
// ------------------------------------------------------------------ //
export default function PerfReportDetailPage() {
  const { t } = useLocale()
  const { get } = useQueryParams()
  const { rootPath: ctxRoot } = useScan()

  const path = get('path') ?? ''
  const rootPath = get('root_path') ?? ctxRoot

  const detail = useAsyncResource(
    (signal) => getPerfDetail(rootPath, path, signal),
    [rootPath, path],
    { enabled: Boolean(path), fallbackMessage: t('common.loadError') },
  )
  const data = detail.data ?? null
  const loading = detail.loading
  const error = detail.error

  // Single-run sweeps have no meaningful trend curve; hide the Charts tab and
  // steer users to the per-run percentile / per-request (DB) views instead.
  const singleRun = (data?.num_runs ?? 0) <= 1

  // Resolve Provider and Protocol as two independent fields, applying the
  // metadata → known-host → Custom fallback priority.
  const identity = useMemo(() => resolveProvider(data ?? {}), [data])

  // Front-load the per-run (DB) views for single-run reports, while still letting
  // the user switch tabs; the pick is scoped to the report it was made on.
  const tabScope = `${rootPath}\0${path}\0${data?.num_runs ?? ''}`
  const [pickedTab, setActiveTab] = useScopedState<TabKey | null>(tabScope, null)
  const activeTab: TabKey = pickedTab ?? (singleRun && data ? 'runs' : 'overview')

  // Charts available for this run mode (embedding runs have no TTFT/TPOT).
  const latencyCharts = useMemo(
    () => (data?.is_embedding ? (['latency'] as const) : LATENCY_CHARTS),
    [data],
  )

  const htmlUrl = useMemo(
    () => (path ? getPerfHistoryReportUrl(rootPath, path) : ''),
    [rootPath, path],
  )

  if (!path) {
    return (
      <div className="flex items-center justify-center h-[60vh] text-[var(--text-muted)]">
        <p>No perf run specified.</p>
      </div>
    )
  }

  if (loading && !data) {
    return (
      <div className="page-enter p-6 flex flex-col gap-4">
        <Skeleton width={300} height={20} />
        <Skeleton width="100%" height={80} />
        <Skeleton lines={6} />
      </div>
    )
  }

  if (!data) {
    return (
      <div className="page-enter flex flex-col gap-4">
        <Breadcrumb
          items={[
            { label: t('nav.performance'), href: `/performance?root_path=${encodeURIComponent(rootPath)}` },
            { label: 'Detail' },
          ]}
        />
        <ErrorAlert className="p-6">
          <p className="text-sm">Failed to load perf report: {error || 'not found'}</p>
        </ErrorAlert>
      </div>
    )
  }

  const tabs = singleRun
    ? [
        { key: 'overview', label: t('perf.archive.overview'), panelId: 'perf-overview-panel' },
        { key: 'runs', label: t('perf.archive.runsTab'), panelId: 'perf-runs-panel' },
      ]
    : [
        { key: 'overview', label: t('perf.archive.overview'), panelId: 'perf-overview-panel' },
        { key: 'charts', label: t('perf.archive.charts'), panelId: 'perf-charts-panel' },
        { key: 'runs', label: t('perf.archive.runsTab'), panelId: 'perf-runs-panel' },
      ]

  const overviewPanel = (
    <div className="flex flex-col gap-4">
      {singleRun && (
        <div className="flex items-start gap-2 px-4 py-3 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-card2)] type-body-sm text-[var(--text-muted)]">
          <Lightbulb size={15} className="text-[var(--accent)] shrink-0 mt-0.5" />
          <span>{t('perf.archive.singleRunHint')}</span>
        </div>
      )}
      <KpiStrip
        layout="inline"
        items={Object.entries(
          omitKeys(data.basic_info, ['Provider', 'Protocol', 'API URL', 'API Host']),
        ).map(([label, value]) => ({ label, value }))}
      />
      <Card title={singleRun ? t('perf.archive.runSummary') : t('perf.archive.summaryTable')}>
        <SummaryTable columns={data.summary_columns} rows={data.summary_rows} t={t} />
      </Card>
      {Object.keys(data.best_config).length > 0 && (
        <Card title={singleRun ? t('perf.archive.runConfig') : t('perf.archive.bestConfig')}>
          <div className="flex flex-col gap-2">
            {Object.entries(data.best_config).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between gap-4 type-body-sm">
                <span className="text-[var(--text-muted)]">{k}</span>
                <span className="text-[var(--text)] tabular-nums text-right">{v}</span>
              </div>
            ))}
          </div>
        </Card>
      )}
      {data.recommendations.length > 0 && (
        <Card title={t('perf.archive.recommendations')}>
          <ul className="flex flex-col gap-2">
            {data.recommendations.map((rec, i) => (
              <li key={i} className="flex items-start gap-2 type-body-sm text-[var(--text)]">
                <Lightbulb size={15} className="text-[var(--accent)] shrink-0 mt-0.5" />
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  )

  const chartsPanel = (
    <div className="flex flex-col gap-4">
      <PerfChartGroup
        title={t('perf.archive.latencyGroup')}
        charts={latencyCharts}
        fallbackTable={{ columns: data.summary_columns, rows: rowsToRecords(data.summary_columns, data.summary_rows) }}
        getChartUrl={(chart) => getPerfChartUrl(rootPath, path, chart)}
      />
      <PerfChartGroup
        title={t('perf.archive.throughputGroup')}
        charts={THROUGHPUT_CHARTS}
        fallbackTable={{ columns: data.summary_columns, rows: rowsToRecords(data.summary_columns, data.summary_rows) }}
        getChartUrl={(chart) => getPerfChartUrl(rootPath, path, chart)}
      />
    </div>
  )

  return (
    <div className="page-enter flex flex-col gap-4">
      <Breadcrumb
        items={[
          { label: t('nav.performance'), href: `/performance?root_path=${encodeURIComponent(rootPath)}` },
          { label: data.model },
        ]}
      />

      {error && (
        <ErrorAlert>{error}</ErrorAlert>
      )}

      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-5">
        <div className="flex flex-col gap-1.5 min-w-0">
          {/* Model alias is the primary identity; fall back to dataset, never
              the raw path/timestamp, when the alias is absent. */}
          <h1 className="type-title-md text-[var(--text)] break-words">{data.model || data.dataset || '—'}</h1>
          {/* Provider and Protocol as two independent, individually-labelled fields. */}
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
            <LabelledField label={t('perf.archive.provider')} value={identity.provider} />
            <LabelledField label={t('perf.archive.protocol')} value={identity.protocol} />
          </div>
          <div className="type-caption-mono text-[var(--text-muted)]">
            {data.dataset} · {data.num_runs}{' '}
            {t(data.num_runs === 1 ? 'perf.archive.runSingular' : 'perf.archive.runs')} ·{' '}
            {formatTimestamp(data.generated_at, 'seconds')}
          </div>
        </div>
        <a
          href={htmlUrl}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-[var(--radius-sm)] border border-[var(--border-md)] text-sm text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card2)] transition-colors shrink-0"
        >
          <ExternalLink size={14} />
          {t('perf.archive.viewFullHtml')}
        </a>
      </div>

      <Tabs
        tabs={tabs}
        activeKey={activeTab}
        onChange={(k) => setActiveTab(k as TabKey)}
        panels={{
          'perf-overview-panel': overviewPanel,
          ...(!singleRun ? { 'perf-charts-panel': chartsPanel } : {}),
          'perf-runs-panel': <PerfRunsTab rootPath={rootPath} path={path} isEmbedding={data.is_embedding} />,
        }}
      />
    </div>
  )
}
