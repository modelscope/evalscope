import { useMemo, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { listPerfRunDetails, getPerfRequests, getPerfChartUrl } from '@/api/perf'
import type { PerfRunItem } from '@/api/types'
import Card from '@/components/ui/Card'
import Skeleton from '@/components/ui/Skeleton'
import DataTable from '@/components/common/DataTable'
import EmptyState from '@/components/common/EmptyState'
import EmptyStateSystem from '@/components/common/EmptyStateSystem'
import ErrorAlert from '@/components/ui/ErrorAlert'
import PlotlyChart from '@/components/charts/PlotlyChart'
import { normalizeWorkload } from '@/domain/perf/perfWorkload'
import { Database } from 'lucide-react'
import { cn } from '@/lib/utils'

const REQ_PAGE_SIZE = 50

type StatusFilter = 'all' | 'success' | 'failed'

/** Stable placeholder so an unresolved run list keeps a single identity. */
const EMPTY_RUNS: PerfRunItem[] = []

interface Props {
  rootPath: string
  path: string
  isEmbedding: boolean
}

/** Convert a columns + row-arrays table into the record list DataTable expects. */
function toRecords(columns: string[], rows: (string | number)[][]): Record<string, unknown>[] {
  return rows.map((row) => Object.fromEntries(columns.map((c, i) => [c, row[i]])))
}

/** A single labelled workload parameter (concurrency, request count, rate). */
function WorkloadField({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-1.5 min-w-0">
      <span className="type-table-xs uppercase tracking-wider text-[var(--text-muted)]">{label}</span>
      <span className="type-body-sm tabular-nums text-[var(--text)] break-words">{value}</span>
    </div>
  )
}

/**
 * Per-run drill-down: run selector + percentile table/charts + per-request
 * records (parsed from benchmark_data.db) as a table and Plotly charts.
 */
export default function PerfRunsTab({ rootPath, path, isEmbedding }: Props) {
  const { t } = useLocale()

  // The user's pick is remembered against the run list it was made in, so a new
  // list falls back to its own first run without an effect resetting state.
  const [pickedRun, setPickedRun] = useState<{ scope: string; dirName: string } | null>(null)
  const [status, setStatus] = useState<StatusFilter>('all')
  const [page, setPage] = useState(1)

  // The individual runs recorded under this perf directory.
  const runsResource = useAsyncResource(
    (signal) => listPerfRunDetails(rootPath, path, signal),
    [rootPath, path],
    { enabled: Boolean(path), fallbackMessage: t('common.loadError') },
  )
  const runs = runsResource.data?.runs ?? EMPTY_RUNS
  const loadingRuns = runsResource.loading
  const runError = runsResource.error

  // Validated against the loaded list, not just the scope: a pick that no longer
  // names an existing run falls back to the first one rather than leaving the
  // panel blank with no way to recover.
  const runScope = `${rootPath}\0${path}`
  const pickIsLoaded = pickedRun?.scope === runScope
    && runs.some((r) => r.dir_name === pickedRun.dirName)
  const selected = pickIsLoaded ? pickedRun.dirName : (runs[0]?.dir_name ?? '')

  const run = useMemo(() => runs.find((r) => r.dir_name === selected) ?? null, [runs, selected])

  // Workload context needed to interpret the selected run: concurrency, number
  // of requests and a request-rate label. Missing params show `N/A` and an
  // unlimited rate shows a semantic loop label rather than `INF`.
  const workload = useMemo(() => (run ? normalizeWorkload(run) : null), [run])

  // Per-request records for the selected run / filter / page.
  const hasRequests = Boolean(run?.has_requests)
  const requestsResource = useAsyncResource(
    (signal) => getPerfRequests({
      rootPath,
      path,
      run: selected,
      status: status === 'all' ? undefined : status,
      page,
      pageSize: REQ_PAGE_SIZE,
      signal,
    }),
    [rootPath, path, selected, status, page, hasRequests],
    { enabled: Boolean(selected) && hasRequests, fallbackMessage: t('common.loadError') },
  )
  // A run without a request database has no records to show at all.
  const requests = hasRequests ? (requestsResource.data ?? null) : null
  const loadingReq = requestsResource.loading
  const requestError = requestsResource.error

  const handleSelectRun = (dirName: string) => {
    setPickedRun({ scope: runScope, dirName })
    setPage(1)
    setStatus('all')
  }

  const handleStatus = (s: StatusFilter) => {
    setStatus(s)
    setPage(1)
  }

  const reqUrl = (chartType: string) => getPerfChartUrl(rootPath, path, chartType, { run: selected })
  const reqTotalPages = requests ? Math.max(1, Math.ceil(requests.total / REQ_PAGE_SIZE)) : 1

  if (loadingRuns) {
    return <Skeleton lines={6} height={16} />
  }

  if (runs.length === 0) {
    return (
      <div className="flex flex-col gap-3">
        {runError && <ErrorAlert className="rounded-[var(--radius-sm)] px-3 py-2">{runError}</ErrorAlert>}
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
          <EmptyStateSystem reason={runError ? 'load-error' : 'no-data'} context={{ view: 'performance' }} />
        </div>
      </div>
    )
  }

  const perRequestCharts = isEmbedding
    ? (['req_latency', 'req_tokens', 'req_success'] as const)
    : (['req_latency', 'req_ttft_tpot', 'req_tokens', 'req_success'] as const)

  return (
    <div className="flex flex-col gap-4">
      {(runError || requestError) && (
        <ErrorAlert className="rounded-[var(--radius-sm)] px-3 py-2">{runError || requestError}</ErrorAlert>
      )}
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

      {/* Workload context for the selected run. */}
      {workload && (
        <div className="flex flex-wrap gap-x-6 gap-y-2 px-4 py-3 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-card)]">
          <WorkloadField label={t('perf.archive.concurrency')} value={workload.concurrency} />
          <WorkloadField label={t('perf.archive.numberOfRequests')} value={workload.numberOfRequests} />
          <WorkloadField label={t('perf.archive.requestRate')} value={workload.rateLabel} />
        </div>
      )}

      {/* Percentile table + charts */}
      <Card title={t('perf.archive.percentiles')}>
        {run && run.percentile_rows.length > 0 ? (
          <div className="flex flex-col gap-4">
            <DataTable columns={run.percentile_columns} data={toRecords(run.percentile_columns, run.percentile_rows)} />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <PlotlyChart
                src={reqUrl('percentile_latency')}
                fallbackTable={{ columns: run.percentile_columns, rows: toRecords(run.percentile_columns, run.percentile_rows) }}
                title={t('perf.archive.latencyPercentiles')}
                height={320}
              />
              {!isEmbedding && (
                <PlotlyChart
                  src={reqUrl('percentile_token')}
                  fallbackTable={{ columns: run.percentile_columns, rows: toRecords(run.percentile_columns, run.percentile_rows) }}
                  title={t('perf.archive.tokenPercentiles')}
                  height={320}
                />
              )}
            </div>
          </div>
        ) : (
          <p className="type-body-sm text-[var(--text-muted)]">{t('perf.archive.noPercentile')}</p>
        )}
      </Card>

      {/* Per-request records (DB) */}
      <Card title={t('perf.archive.requests')}>
        {!run?.has_requests ? (
          <EmptyState icon={<Database size={28} strokeWidth={1.5} />} title={t('perf.archive.noDb')} hint={t('perf.archive.noDbHint')} />
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
                  {t(`perf.archive.status_${s}`)}
                </button>
              ))}
              {requests && (
                <span className="ml-auto type-caption-mono text-[var(--text-muted)]">{requests.total} reqs</span>
              )}
            </div>

            {/* Per-request charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {perRequestCharts.map((ct) => (
                <PlotlyChart
                  key={ct}
                  src={reqUrl(ct)}
                  fallbackTable={{ columns: requests?.columns ?? [], rows: requests?.rows ?? [] }}
                  height={300}
                />
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
              <p className="type-body-sm text-[var(--text-muted)]">{t('perf.archive.noRequests')}</p>
            )}
          </div>
        )}
      </Card>
    </div>
  )
}
