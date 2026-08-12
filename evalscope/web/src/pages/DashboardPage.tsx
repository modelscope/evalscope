import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowRight, Clock, Cpu, FileText, Gauge } from 'lucide-react'
import { useScan } from '@/contexts/ReportsContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { useLocale } from '@/contexts/LocaleContext'
import { listReports } from '@/api/reports'
import { listPerfRuns } from '@/api/perf'
import type { PerfRunSummary, ReportSummary } from '@/api/types'
import { formatDifference, formatMetric, type MetricSemantics } from '@/domain/metric'
import Skeleton from '@/components/ui/Skeleton'
import KpiStrip, { KPI_HERO_CONTAINER, KPI_HERO_CELL, type KpiItem } from '@/components/ui/KpiStrip'
import Tabs from '@/components/ui/Tabs'
import SearchInput from '@/components/ui/SearchInput'
import EmptyState from '@/components/common/EmptyState'
import EmptyStateSystem from '@/components/common/EmptyStateSystem'
import ErrorAlert from '@/components/ui/ErrorAlert'
import AggregatedResults from '@/components/dashboard/AggregatedResults'
import type { SortState } from '@/components/dashboard/AggregatedResults'
import { aggregateRuns } from '@/domain/report/runAggregation'
import { parseReportRef } from '@/domain/report/reportRef'
import type { AggregatedRow, CellKind, CellPoint } from '@/domain/report/runAggregation'
import { formatTimestamp } from '@/utils/formatUtils'

/**
 * Which kinds of run the table shows.
 *
 * `all` is not a kind, it is the absence of the filter, so it is kept out of `CellKind` rather than
 * added to it -- nothing produces a cell of kind "all".
 */
type KindFilter = CellKind | 'all'

/** Stable placeholders so an unresolved read keeps a single identity per collection. */
const EMPTY_REPORTS: ReportSummary[] = []
const EMPTY_PERF_RUNS: PerfRunSummary[] = []
const EMPTY_SEMANTICS: Record<string, MetricSemantics> = {}

/** Tab order, and the panel each one drives. */
const KIND_TABS: { key: KindFilter; labelKey: string; panelId: string }[] = [
  { key: 'all', labelKey: 'dashboard.tabAll', panelId: 'dashboard-results-all' },
  { key: 'eval', labelKey: 'dashboard.tabEval', panelId: 'dashboard-results-eval' },
  { key: 'perf', labelKey: 'dashboard.tabPerf', panelId: 'dashboard-results-perf' },
]

/**
 * Landing page: how much has been recorded here, then how the benchmarks are holding up.
 *
 * Four counters open the page, and below them the part no other page can do: results aggregated by
 * what they measure rather than by when they ran, because re-running a benchmark is the normal
 * workflow here and a flat feed renders those repeats as many identical-looking rows. Anything the
 * list pages do better is not duplicated -- no filter, no search, no pagination.
 */
export default function DashboardPage() {
  const { t } = useLocale()
  const { rootPath, scanToken } = useScan()
  const navigate = useNavigate()

  const [kindFilter, setKindFilter] = useState<KindFilter>('all')
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<SortState>({ key: 'lastRun', descending: true })

  // Fetch eval + perf whenever the global scan token or root changes. Settled,
  // not all-or-nothing: one side failing must not hide the other's runs.
  const overview = useAsyncResource(
    async (signal) => {
      const [evalRes, perfRes] = await Promise.allSettled([
        (async () => {
          const collected: ReportSummary[] = []
          let page = 1
          while (true) {
            const response = await listReports({
              rootPath,
              page,
              pageSize: 100,
              sortBy: 'time',
              sortOrder: 'desc',
              signal,
            })
            collected.push(...response.reports)
            if (collected.length >= response.total || response.reports.length === 0) return collected
            page += 1
          }
        })(),
        listPerfRuns(rootPath, signal),
      ])

      const failure = evalRes.status === 'rejected'
        ? evalRes.reason
        : perfRes.status === 'rejected' ? perfRes.reason : null

      return {
        reports: evalRes.status === 'fulfilled' ? evalRes.value : EMPTY_REPORTS,
        perfRuns: perfRes.status === 'fulfilled' ? perfRes.value.runs : EMPTY_PERF_RUNS,
        perfSemantics: perfRes.status === 'fulfilled' ? (perfRes.value.metric_semantics ?? {}) : {},
        failure: failure instanceof Error ? failure.message : failure ? t('common.loadError') : '',
      }
    },
    [rootPath, scanToken],
    { enabled: Boolean(rootPath), fallbackMessage: t('common.loadError') },
  )

  const reports = overview.data?.reports ?? EMPTY_REPORTS
  const perfRuns = overview.data?.perfRuns ?? EMPTY_PERF_RUNS
  const perfSemantics = overview.data?.perfSemantics ?? EMPTY_SEMANTICS
  const loading = overview.loading
  // A partial failure is reported by the resolved value; a total one by the hook.
  const loadError = overview.error || (overview.data?.failure ?? '')
  const scanned = overview.data !== undefined

  // The table is driven by this: every score ever recorded, grouped by what it measures.
  const rows = useMemo(
    () => aggregateRuns(reports, perfRuns, perfSemantics),
    [reports, perfRuns, perfSemantics],
  )

  // What the active tab admits. Filtering here rather than inside the table keeps the table a
  // renderer of whatever rows it is handed.
  const visibleRows = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase()
    return rows.filter((row) => {
      if (kindFilter !== 'all' && row.cell.kind !== kindFilter) return false
      if (!normalizedQuery) return true
      return [row.cell.model, row.cell.benchmark, row.cell.benchmarkLabel, row.cell.metricName]
        .filter((value): value is string => Boolean(value))
        .some((value) => value.toLocaleLowerCase().includes(normalizedQuery))
    })
  }, [rows, kindFilter, query])

  const kpi = useMemo(() => {
    const models = new Set<string>()
    reports.forEach((report) => report.model_name && models.add(report.model_name))
    perfRuns.forEach((run) => run.model && models.add(run.model))
    const timestamps = [
      ...reports.map((report) => report.timestamp || ''),
      ...perfRuns.map((run) => run.timestamp || ''),
    ].filter((timestamp): timestamp is string => Boolean(timestamp))
    const latest: string = timestamps.length > 0 ? timestamps.reduce((a, b) => (a > b ? a : b)) : ''
    return {
      evals: reports.length,
      perfs: perfRuns.length,
      models: models.size,
      latest,
    }
  }, [reports, perfRuns])

  const latestRunLabel = useMemo(() => {
    if (!kpi.latest) return t('dashboard.neverText')
    const value = new Date(kpi.latest)
    const today = new Date()
    const time = value.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    if (value.toDateString() === today.toDateString()) return `${t('dashboard.today')}, ${time}`
    return formatTimestamp(kpi.latest, 'seconds')
  }, [kpi.latest, t])

  const kpiItems = useMemo<KpiItem[]>(() => [
    {
      icon: <FileText size={17} strokeWidth={2} />,
      value: String(kpi.evals),
      label: t('dashboard.totalEvaluations'),
      onClick: () => navigate('/reports'),
    },
    {
      icon: <Gauge size={17} strokeWidth={2} />,
      value: String(kpi.perfs),
      label: t('dashboard.totalPerfRuns'),
      onClick: () => navigate('/performance'),
    },
    {
      icon: <Cpu size={17} strokeWidth={2} />,
      value: String(kpi.models),
      label: t('dashboard.modelsEvaluated'),
    },
    {
      icon: <Clock size={17} strokeWidth={2} />,
      value: latestRunLabel,
      label: t('dashboard.latestRun'),
      title: kpi.latest ? formatTimestamp(kpi.latest, 'seconds') : undefined,
    },
  ], [kpi, latestRunLabel, navigate, t])

  const recentChange = useMemo(() => {
    return visibleRows
      .filter((row) => row.cell.history.length > 1)
      .map((row) => {
        const latest = row.cell.history[row.cell.history.length - 1]
        const previous = row.cell.history
          .slice(0, -1)
          .reverse()
          .find((point) => point.score !== latest.score)
        if (!previous) return null
        return { row, latest, delta: latest.score - previous.score }
      })
      .filter((change): change is NonNullable<typeof change> => change !== null)
      .filter(({ delta }) => Number.isFinite(delta))
      .sort((a, b) => b.latest.timestamp.localeCompare(a.latest.timestamp))[0]
  }, [visibleRows])

  const openRun = (row: AggregatedRow, point: CellPoint) => {
    const root = encodeURIComponent(rootPath)
    if (row.cell.kind === 'eval') {
      const { runId, modelId } = parseReportRef(point.runId)
      navigate(`/reports/${encodeURIComponent(runId)}/${encodeURIComponent(modelId)}?root_path=${root}`)
      return
    }
    navigate(`/perf-report?path=${encodeURIComponent(point.runId)}&root_path=${root}`)
  }

  const hasData = scanned && rows.length > 0

  const recentChangeStrip = recentChange ? (
    <div className="flex flex-col gap-2 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--accent-dim)] px-4 py-2.5 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0">
        <span className="mr-3 type-label-xs text-[var(--accent)]">{t('dashboard.recentChange')}</span>
        <span className="type-body-sm text-[var(--text-muted)]">
          {t('dashboard.recentChangeSummary', {
            benchmark: recentChange.row.cell.benchmarkLabel || recentChange.row.cell.benchmark,
            metric: recentChange.row.cell.semantics?.metric_name || recentChange.row.cell.metricName,
            latest: formatMetric(recentChange.row.stats.latest, recentChange.row.cell.semantics).primary,
            change: formatSignedDifference(recentChange.delta, recentChange.row.cell.semantics),
          })}
        </span>
      </div>
      <button
        type="button"
        onClick={() => openRun(recentChange.row, recentChange.latest)}
        className="inline-flex shrink-0 items-center gap-1 type-body-xs font-medium text-[var(--accent)] transition-colors hover:text-[var(--accent-dark)]"
      >
        {t('dashboard.viewDetails')}
        <ArrowRight size={13} />
      </button>
    </div>
  ) : null

  // One node, handed to whichever panel is selected: the tab decides the rows, not the markup.
  const resultsPanel = visibleRows.length > 0 ? (
    <div className="mt-3 flex min-w-0 flex-col gap-3">
      {recentChangeStrip}
      <AggregatedResults rows={visibleRows} onOpenRun={openRun} sort={sort} onSortChange={setSort} />
    </div>
  ) : (
    <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
      <EmptyStateSystem reason="no-match" context={{ view: 'dashboard' }} />
    </div>
  )

  return (
    <div className="flex min-h-0 w-full flex-col gap-4">
      {loadError && <ErrorAlert className="rounded-[var(--radius-sm)]">{loadError}</ErrorAlert>}

      {loading && !scanned ? (
        <div className={KPI_HERO_CONTAINER}>
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className={KPI_HERO_CELL}>
              <Skeleton width={32} height={32} className="mb-2" />
              <Skeleton width={60} height={24} className="mb-1" />
              <Skeleton width={100} height={14} />
            </div>
          ))}
        </div>
      ) : (
        <KpiStrip items={kpiItems} />
      )}

      {loading && !scanned ? (
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-4">
          <Skeleton lines={8} height={14} />
        </div>
      ) : hasData ? (
        <div className="flex min-w-0 flex-col gap-3">
          <Tabs
            tabs={KIND_TABS}
            activeKey={kindFilter}
            onChange={(key) => setKindFilter(key as KindFilter)}
            panels={Object.fromEntries(KIND_TABS.map((tab) => [tab.panelId, resultsPanel]))}
            className="self-start"
            actions={
              <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row">
                <SearchInput
                  value={query}
                  onChange={setQuery}
                  placeholder={t('dashboard.searchPlaceholder')}
                  className="w-full sm:w-64"
                />
                <select
                  aria-label={t('dashboard.sortResults')}
                  value={`${sort.key}-${sort.descending ? 'desc' : 'asc'}`}
                  onChange={(event) => setSort(parseSort(event.target.value))}
                  className="rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-deep)] px-3 py-2 type-body-sm text-[var(--text-muted)] outline-none transition-colors focus:border-[var(--accent)] focus:ring-1 focus:ring-[var(--accent-dim)]"
                >
                  <option value="lastRun-desc">{t('dashboard.sortLastRunNewest')}</option>
                  <option value="lastRun-asc">{t('dashboard.sortLastRunOldest')}</option>
                  <option value="runs-desc">{t('dashboard.sortRunsMost')}</option>
                  <option value="runs-asc">{t('dashboard.sortRunsLeast')}</option>
                  <option value="model-asc">{t('dashboard.sortModelAsc')}</option>
                  <option value="model-desc">{t('dashboard.sortModelDesc')}</option>
                  <option value="benchmark-asc">{t('dashboard.sortBenchmarkAsc')}</option>
                  <option value="benchmark-desc">{t('dashboard.sortBenchmarkDesc')}</option>
                </select>
              </div>
            }
          />
        </div>
      ) : scanned ? (
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
          <EmptyStateSystem reason="no-data" context={{ view: 'dashboard' }} hint={t('dashboard.noReportsHint')} />
        </div>
      ) : (
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
          <EmptyState
            variant="welcome"
            icon={<FileText size={28} strokeWidth={1.5} />}
            title={t('dashboard.welcomeTitle')}
            hint={t('dashboard.welcomeDesc')}
          />
        </div>
      )}
    </div>
  )
}

function parseSort(value: string): SortState {
  const [key, direction] = value.split('-') as [SortState['key'], 'asc' | 'desc']
  return { key, descending: direction === 'desc' }
}

function formatSignedDifference(value: number, semantics: MetricSemantics | null): string {
  const formatted = formatDifference(value, semantics).primary
  return value > 0 ? `+${formatted}` : formatted
}
