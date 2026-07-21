import { useEffect, useMemo, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import { getPerfDetail, getPerfChartUrl, getPerfHistoryReportUrl } from '@/api/perf'
import type { PerfDetailResponse } from '@/api/types'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Tabs from '@/components/ui/Tabs'
import Card from '@/components/ui/Card'
import Skeleton from '@/components/ui/Skeleton'
import ErrorAlert from '@/components/ui/ErrorAlert'
import PerfChartGroup from '@/components/perf/PerfChartGroup'
import PerfRunsTab from './PerfRunsTab'
import { LATENCY_CHARTS, THROUGHPUT_CHARTS, formatFull } from '@/utils/perf'
import { resolveProvider } from '@/domain/perf/providerResolution'
import { ExternalLink, Lightbulb } from 'lucide-react'

type TabKey = 'overview' | 'charts' | 'runs'

/**
 * A single, individually-labelled identity field. `Provider` and `Protocol`
 * are rendered as two of these so the two never collapse into one combined
 * field.
 */
function IdentityField({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-1.5 min-w-0">
      <span className="type-table-xs uppercase tracking-wider text-[var(--text-muted)]">{label}</span>
      <span className="type-body-sm text-[var(--text)] break-words">{value}</span>
    </div>
  )
}

/** Return a shallow copy of `info` with the given keys removed. */
function omitKeys(info: Record<string, string>, keys: string[]): Record<string, string> {
  const drop = new Set(keys)
  return Object.fromEntries(Object.entries(info).filter(([k]) => !drop.has(k)))
}

// ------------------------------------------------------------------ //
// Overview building blocks                                            //
// ------------------------------------------------------------------ //
function KpiStrip({ info }: { info: Record<string, string> }) {
  const entries = Object.entries(info)
  if (entries.length === 0) return null
  return (
    <div className="flex flex-wrap bg-[var(--bg-card)] border border-[var(--border)] rounded-[var(--radius-sm)] overflow-hidden">
      {entries.map(([k, v], i) => (
        <div
          key={k}
          className={`flex-1 min-w-[140px] px-4 py-3 ${i < entries.length - 1 ? 'border-r border-[var(--border)]' : ''}`}
        >
          <div className="type-body-sm-strong text-[var(--text)] tabular-nums">{v}</div>
          <div className="type-table-xs uppercase tracking-wider text-[var(--text-muted)] mt-0.5">{k}</div>
        </div>
      ))}
    </div>
  )
}

function formatSummaryCell(column: string, cell: string | number, t: (key: string) => string): string {
  if (column.trim().toLowerCase() === 'rate' && String(cell).trim().toUpperCase() === 'INF') {
    return t('performance.closedLoop')
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
                className="type-table-xs uppercase tracking-wider px-3 py-2 text-right first:text-left whitespace-nowrap border-b border-[var(--border)] text-[var(--text-muted)]"
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
  const { rootPath: ctxRoot } = useReports()

  const path = get('path') ?? ''
  const rootPath = get('root_path') ?? ctxRoot

  const [data, setData] = useState<PerfDetailResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState<TabKey>('overview')

  // Single-run sweeps have no meaningful trend curve; hide the Charts tab and
  // steer users to the per-run percentile / per-request (DB) views instead.
  const singleRun = (data?.num_runs ?? 0) <= 1

  // Resolve Provider and Protocol as two independent fields, applying the
  // metadata → known-host → Custom fallback priority.
  const identity = useMemo(() => resolveProvider(data ?? {}), [data])

  useEffect(() => {
    if (!path) return
    const controller = new AbortController()
    const load = async () => {
      setLoading(true)
      setError('')
      try {
        const res = await getPerfDetail(rootPath, path, controller.signal)
        if (!controller.signal.aborted) {
          setData(res)
          // Front-load the per-run (DB) views for single-run reports.
          if ((res.num_runs ?? 0) <= 1) setActiveTab('runs')
        }
      } catch (err) {
        if (!controller.signal.aborted) setError(String(err))
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }
    load()
    return () => {
      controller.abort()
    }
  }, [rootPath, path])

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
        { key: 'overview', label: t('performance.overview'), panelId: 'perf-overview-panel' },
        { key: 'runs', label: t('performance.runsTab'), panelId: 'perf-runs-panel' },
      ]
    : [
        { key: 'overview', label: t('performance.overview'), panelId: 'perf-overview-panel' },
        { key: 'charts', label: t('performance.charts'), panelId: 'perf-charts-panel' },
        { key: 'runs', label: t('performance.runsTab'), panelId: 'perf-runs-panel' },
      ]

  const overviewPanel = (
    <div className="flex flex-col gap-4">
      {singleRun && (
        <div className="flex items-start gap-2 px-4 py-3 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-card2)] type-body-sm text-[var(--text-muted)]">
          <Lightbulb size={15} className="text-[var(--accent)] shrink-0 mt-0.5" />
          <span>{t('performance.singleRunHint')}</span>
        </div>
      )}
      <KpiStrip info={omitKeys(data.basic_info, ['Provider', 'Protocol', 'API URL', 'API Host'])} />
      <Card title={singleRun ? t('performance.runSummary') : t('performance.summaryTable')}>
        <SummaryTable columns={data.summary_columns} rows={data.summary_rows} t={t} />
      </Card>
      {Object.keys(data.best_config).length > 0 && (
        <Card title={singleRun ? t('performance.runConfig') : t('performance.bestConfig')}>
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
        <Card title={t('performance.recommendations')}>
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
        title={t('performance.latencyGroup')}
        charts={latencyCharts}
        fallbackTable={{ columns: data.summary_columns, rows: rowsToRecords(data.summary_columns, data.summary_rows) }}
        getChartUrl={(chart) => getPerfChartUrl(rootPath, path, chart)}
      />
      <PerfChartGroup
        title={t('performance.throughputGroup')}
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
            <IdentityField label={t('performance.provider')} value={identity.provider} />
            <IdentityField label={t('performance.protocol')} value={identity.protocol} />
          </div>
          <div className="type-caption-mono text-[var(--text-muted)]">
            {data.dataset} · {data.num_runs}{' '}
            {t(data.num_runs === 1 ? 'performance.runSingular' : 'performance.runs')} ·{' '}
            {formatFull(data.generated_at)}
          </div>
        </div>
        <a
          href={htmlUrl}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-[var(--radius-sm)] border border-[var(--border-md)] text-sm text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card2)] transition-colors shrink-0"
        >
          <ExternalLink size={14} />
          {t('performance.viewFullHtml')}
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
