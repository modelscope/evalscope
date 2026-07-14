import { useEffect, useMemo, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { listPerfRunDetails, getPerfRequests, getPerfChartUrl } from '@/api/perf'
import type { PerfRunItem, PerfRequestsResponse } from '@/api/types'
import Card from '@/components/ui/Card'
import Skeleton from '@/components/ui/Skeleton'
import DataTable from '@/components/common/DataTable'
import EmptyState from '@/components/common/EmptyState'
import PlotlyChart from '@/components/charts/PlotlyChart'
import { Inbox, Database } from 'lucide-react'
import { cn } from '@/lib/utils'

const REQ_PAGE_SIZE = 50

type StatusFilter = 'all' | 'success' | 'failed'

interface Props {
  rootPath: string
  path: string
  isEmbedding: boolean
  theme: string
}

/** Convert a columns + row-arrays table into the record list DataTable expects. */
function toRecords(columns: string[], rows: (string | number)[][]): Record<string, unknown>[] {
  return rows.map((row) => Object.fromEntries(columns.map((c, i) => [c, row[i]])))
}

/**
 * Per-run drill-down: run selector + percentile table/charts + per-request
 * records (parsed from benchmark_data.db) as a table and Plotly charts.
 */
export default function PerfRunsTab({ rootPath, path, isEmbedding, theme }: Props) {
  const { t } = useLocale()

  const [runs, setRuns] = useState<PerfRunItem[]>([])
  const [loadingRuns, setLoadingRuns] = useState(true)
  const [selected, setSelected] = useState('')

  const [status, setStatus] = useState<StatusFilter>('all')
  const [page, setPage] = useState(1)
  const [requests, setRequests] = useState<PerfRequestsResponse | null>(null)
  const [loadingReq, setLoadingReq] = useState(false)

  // Load the list of individual runs for this perf directory.
  useEffect(() => {
    if (!path) return
    let cancelled = false
    const load = async () => {
      setLoadingRuns(true)
      try {
        const res = await listPerfRunDetails(rootPath, path)
        if (cancelled) return
        setRuns(res.runs)
        setSelected(res.runs[0]?.dir_name ?? '')
      } catch {
        if (!cancelled) setRuns([])
      } finally {
        if (!cancelled) setLoadingRuns(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [rootPath, path])

  const run = useMemo(() => runs.find((r) => r.dir_name === selected) ?? null, [runs, selected])

  // Load per-request records for the selected run / filter / page.
  useEffect(() => {
    let cancelled = false
    const load = async () => {
      if (!selected || !run?.has_requests) {
        setRequests(null)
        return
      }
      setLoadingReq(true)
      try {
        const res = await getPerfRequests({
          rootPath,
          path,
          run: selected,
          status: status === 'all' ? undefined : status,
          page,
          pageSize: REQ_PAGE_SIZE,
        })
        if (!cancelled) setRequests(res)
      } catch {
        if (!cancelled) setRequests(null)
      } finally {
        if (!cancelled) setLoadingReq(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [rootPath, path, selected, status, page, run?.has_requests])

  const handleSelectRun = (dirName: string) => {
    setSelected(dirName)
    setPage(1)
    setStatus('all')
  }

  const handleStatus = (s: StatusFilter) => {
    setStatus(s)
    setPage(1)
  }

  const reqUrl = (chartType: string) => getPerfChartUrl(rootPath, path, chartType, { run: selected, theme })
  const reqTotalPages = requests ? Math.max(1, Math.ceil(requests.total / REQ_PAGE_SIZE)) : 1

  if (loadingRuns) {
    return <Skeleton lines={6} height={16} />
  }

  if (runs.length === 0) {
    return (
      <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
        <EmptyState icon={<Inbox size={28} strokeWidth={1.5} />} title={t('performance.noRuns')} />
      </div>
    )
  }

  const perRequestCharts = isEmbedding
    ? (['req_latency', 'req_tokens', 'req_success'] as const)
    : (['req_latency', 'req_ttft_tpot', 'req_tokens', 'req_success'] as const)

  return (
    <div className="flex flex-col gap-4">
      {/* Run selector */}
      {runs.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {runs.map((r) => (
            <button
              key={r.dir_name}
              onClick={() => handleSelectRun(r.dir_name)}
              className={cn(
                'px-3 py-1.5 type-button-sm rounded-[var(--radius-sm)] border transition-colors',
                r.dir_name === selected
                  ? 'bg-[var(--accent)] text-[var(--text-on-filled)] border-[var(--accent)]'
                  : 'bg-[var(--bg-card)] text-[var(--text-muted)] border-[var(--border)] hover:bg-[var(--bg-card2)]',
              )}
            >
              {r.name}
            </button>
          ))}
        </div>
      )}

      {/* Percentile table + charts */}
      <Card title={t('performance.percentiles')}>
        {run && run.percentile_rows.length > 0 ? (
          <div className="flex flex-col gap-4">
            <DataTable columns={run.percentile_columns} data={toRecords(run.percentile_columns, run.percentile_rows)} />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <PlotlyChart src={reqUrl('percentile_latency')} title="Latency (s)" height={320} />
              {!isEmbedding && <PlotlyChart src={reqUrl('percentile_token')} title="TTFT / TPOT / ITL (ms)" height={320} />}
            </div>
          </div>
        ) : (
          <p className="type-body-sm text-[var(--text-muted)]">{t('performance.noPercentile')}</p>
        )}
      </Card>

      {/* Per-request records (DB) */}
      <Card title={t('performance.requests')}>
        {!run?.has_requests ? (
          <EmptyState icon={<Database size={28} strokeWidth={1.5} />} title={t('performance.noDb')} hint={t('performance.noDbHint')} />
        ) : (
          <div className="flex flex-col gap-4">
            {/* Status filter */}
            <div className="flex items-center gap-2">
              {(['all', 'success', 'failed'] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => handleStatus(s)}
                  className={cn(
                    'px-3 py-1 type-body-xs rounded-full border transition-colors',
                    status === s
                      ? 'bg-[var(--accent-dim)] text-[var(--accent)] border-[var(--accent)]'
                      : 'text-[var(--text-muted)] border-[var(--border)] hover:bg-[var(--bg-card2)]',
                  )}
                >
                  {t(`performance.status_${s}`)}
                </button>
              ))}
              {requests && (
                <span className="ml-auto type-caption-mono text-[var(--text-muted)]">{requests.total} reqs</span>
              )}
            </div>

            {/* Per-request charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {perRequestCharts.map((ct) => (
                <PlotlyChart key={ct} src={reqUrl(ct)} height={300} />
              ))}
            </div>

            {/* Records table */}
            {loadingReq ? (
              <Skeleton lines={6} height={16} />
            ) : requests && requests.rows.length > 0 ? (
              <>
                <DataTable columns={requests.columns} data={requests.rows} />
                {reqTotalPages > 1 && (
                  <div className="flex items-center justify-center gap-3 pt-1 type-body-sm">
                    <button
                      className="px-2 py-1 rounded-[var(--radius-sm)] disabled:opacity-40 hover:bg-[var(--bg-card2)]"
                      disabled={page <= 1}
                      onClick={() => setPage((p) => Math.max(1, p - 1))}
                    >
                      ←
                    </button>
                    <span className="type-caption-mono text-[var(--text-muted)]">{page} / {reqTotalPages}</span>
                    <button
                      className="px-2 py-1 rounded-[var(--radius-sm)] disabled:opacity-40 hover:bg-[var(--bg-card2)]"
                      disabled={page >= reqTotalPages}
                      onClick={() => setPage((p) => Math.min(reqTotalPages, p + 1))}
                    >
                      →
                    </button>
                  </div>
                )}
              </>
            ) : (
              <p className="type-body-sm text-[var(--text-muted)]">{t('performance.noRequests')}</p>
            )}
          </div>
        )}
      </Card>
    </div>
  )
}
