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
import Badge from '@/components/ui/Badge'
import PlotlyChart from '@/components/charts/PlotlyChart'
import { ExternalLink, Lightbulb } from 'lucide-react'

type TabKey = 'overview' | 'charts'

// Sweep charts grouped like the standalone HTML report.
const LATENCY_CHARTS = ['latency', 'ttft', 'tpot'] as const
const THROUGHPUT_CHARTS = ['rps', 'throughput', 'success'] as const

const CHART_TITLES: Record<string, string> = {
  latency: 'Latency',
  ttft: 'TTFT',
  tpot: 'TPOT',
  rps: 'Request Throughput',
  throughput: 'Token Throughput',
  success: 'Success Rate',
}

function formatFull(ts: string): string {
  return ts ? ts.replace('T', ' ').slice(0, 19) : ''
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

function SummaryTable({ columns, rows }: { columns: string[]; rows: (string | number)[][] }) {
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
                  {String(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
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

  useEffect(() => {
    if (!path) return
    let cancelled = false
    const load = async () => {
      setLoading(true)
      setError('')
      try {
        const res = await getPerfDetail(rootPath, path)
        if (!cancelled) setData(res)
      } catch (err) {
        if (!cancelled) setError(String(err))
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => {
      cancelled = true
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

  if (loading) {
    return (
      <div className="page-enter p-6 flex flex-col gap-4">
        <Skeleton width={300} height={20} />
        <Skeleton width="100%" height={80} />
        <Skeleton lines={6} />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="page-enter flex flex-col gap-4">
        <Breadcrumb
          items={[
            { label: t('nav.performance'), href: `/performance?root_path=${encodeURIComponent(rootPath)}` },
            { label: 'Detail' },
          ]}
        />
        <div className="p-6 rounded-[var(--radius)] border border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger)]">
          <p className="text-sm">Failed to load perf report: {error || 'not found'}</p>
        </div>
      </div>
    )
  }

  const tabs = [
    { key: 'overview', label: t('performance.overview') },
    { key: 'charts', label: t('performance.charts') },
  ]

  return (
    <div className="page-enter flex flex-col gap-4">
      <Breadcrumb
        items={[
          { label: t('nav.performance'), href: `/performance?root_path=${encodeURIComponent(rootPath)}` },
          { label: data.model },
        ]}
      />

      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-5">
        <div className="flex flex-col gap-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h1 className="type-title-md text-[var(--text)] truncate">{data.model}</h1>
            {data.api_type && <Badge>{data.api_type}</Badge>}
          </div>
          <div className="type-caption-mono text-[var(--text-muted)]">
            {data.dataset} · {data.num_runs} {t('performance.runs')} · {formatFull(data.generated_at)}
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

      {/* Tabs */}
      <Tabs tabs={tabs} activeKey={activeTab} onChange={(k) => setActiveTab(k as TabKey)} />

      {/* Overview */}
      {activeTab === 'overview' ? (
        <div className="flex flex-col gap-4">
          <KpiStrip info={data.basic_info} />

          <Card title={t('performance.summaryTable')}>
            <SummaryTable columns={data.summary_columns} rows={data.summary_rows} />
          </Card>

          {Object.keys(data.best_config).length > 0 && (
            <Card title={t('performance.bestConfig')}>
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
      ) : (
        <div className="flex flex-col gap-4">
          <Card title={t('performance.latencyGroup')}>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {latencyCharts.map((ct) => (
                <PlotlyChart key={ct} src={getPerfChartUrl(rootPath, path, ct)} title={CHART_TITLES[ct]} height={340} />
              ))}
            </div>
          </Card>
          <Card title={t('performance.throughputGroup')}>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {THROUGHPUT_CHARTS.map((ct) => (
                <PlotlyChart key={ct} src={getPerfChartUrl(rootPath, path, ct)} title={CHART_TITLES[ct]} height={340} />
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
