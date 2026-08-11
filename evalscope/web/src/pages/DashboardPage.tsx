import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowRight, Clock, Cpu, FileText, Gauge } from 'lucide-react'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import { listReports } from '@/api/reports'
import { listPerfRuns } from '@/api/perf'
import type { PerfRunSummary, ReportSummary } from '@/api/types'
import { formatDifference, formatMetric, type MetricSemantics } from '@/domain/metric'
import Skeleton from '@/components/ui/Skeleton'
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
import { formatFull } from '@/utils/perf'

/**
 * Which kinds of run the table shows.
 *
 * `all` is not a kind, it is the absence of the filter, so it is kept out of `CellKind` rather than
 * added to it -- nothing produces a cell of kind "all".
 */
type KindFilter = CellKind | 'all'

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
  const { rootPath, scanToken } = useReports()
  const navigate = useNavigate()

  const [loading, setLoading] = useState(false)
  const [scanned, setScanned] = useState(false)
  const [reports, setReports] = useState<ReportSummary[]>([])
  const [perfRuns, setPerfRuns] = useState<PerfRunSummary[]>([])
  const [perfSemantics, setPerfSemantics] = useState<Record<string, MetricSemantics>>({})
  const [loadError, setLoadError] = useState('')
  const [kindFilter, setKindFilter] = useState<KindFilter>('all')
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<SortState>({ key: 'lastRun', descending: true })

  // Fetch eval + perf whenever the global scan token or root changes.
  useEffect(() => {
    if (!rootPath) return
    const controller = new AbortController()
    const load = async () => {
      setLoading(true)
      setLoadError('')
      const [evalRes, perfRes] = await Promise.allSettled([
        listReports({ rootPath, pageSize: 1000, sortBy: 'time', sortOrder: 'desc', signal: controller.signal }),
        listPerfRuns(rootPath, controller.signal),
      ])
      if (controller.signal.aborted) return
      if (evalRes.status === 'fulfilled') setReports(evalRes.value.reports)
      if (perfRes.status === 'fulfilled') {
        setPerfRuns(perfRes.value.runs)
        setPerfSemantics(perfRes.value.metric_semantics ?? {})
      }
      if (evalRes.status === 'rejected' || perfRes.status === 'rejected') {
        const reason = evalRes.status === 'rejected' ? evalRes.reason : perfRes.status === 'rejected' ? perfRes.reason : null
        setLoadError(reason instanceof Error ? reason.message : t('common.loadError'))
      }
      setScanned(true)
      setLoading(false)
    }
    load()
    return () => {
      controller.abort()
    }
  }, [rootPath, scanToken, t])

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
    return formatFull(kpi.latest)
  }, [kpi.latest, t])

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
        <div className="grid grid-cols-2 overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="border-[var(--border)] p-5 lg:border-r lg:last:border-r-0">
              <Skeleton width={32} height={32} className="mb-2" />
              <Skeleton width={60} height={24} className="mb-1" />
              <Skeleton width={100} height={14} />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] shadow-[var(--shadow-sm)] lg:grid-cols-4">
          <SummaryItem
            icon={<FileText size={17} strokeWidth={2} />}
            value={String(kpi.evals)}
            label={t('dashboard.totalEvaluations')}
            onClick={() => navigate('/reports')}
          />
          <SummaryItem
            icon={<Gauge size={17} strokeWidth={2} />}
            value={String(kpi.perfs)}
            label={t('dashboard.totalPerfRuns')}
            onClick={() => navigate('/performance')}
          />
          <SummaryItem
            icon={<Cpu size={17} strokeWidth={2} />}
            value={String(kpi.models)}
            label={t('dashboard.modelsEvaluated')}
          />
          <SummaryItem
            icon={<Clock size={17} strokeWidth={2} />}
            value={latestRunLabel}
            label={t('dashboard.latestRun')}
            title={kpi.latest ? formatFull(kpi.latest) : undefined}
          />
        </div>
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

function SummaryItem({
  icon,
  value,
  label,
  onClick,
  title,
}: {
  icon: ReactNode
  value: string
  label: string
  onClick?: () => void
  title?: string
}) {
  const content = (
    <>
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[var(--radius-sm)] bg-[var(--accent-dim)] text-[var(--accent)]">
        {icon}
      </span>
      <span className="min-w-0">
        <span className="block truncate type-title-sm font-semibold text-[var(--text)]" title={title}>
          {value}
        </span>
        <span className="block truncate type-body-xs text-[var(--text-muted)]">{label}</span>
      </span>
    </>
  )

  const className =
    'flex min-w-0 items-center gap-3 border-b border-r border-[var(--border)] p-5 text-left transition-colors even:border-r-0 lg:border-b-0 lg:even:border-r lg:last:border-r-0'

  if (onClick) {
    return (
      <button type="button" onClick={onClick} className={`${className} hover:bg-[var(--bg-card2)]`}>
        {content}
      </button>
    )
  }

  return <div className={className}>{content}</div>
}
