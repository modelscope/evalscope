import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Clock, Cpu, FileText, Gauge } from 'lucide-react'
import { useScan } from '@/contexts/ReportsContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { useLocale } from '@/contexts/LocaleContext'
import { listReports } from '@/api/reports'
import { listPerfRuns } from '@/api/perf'
import type { PerfRunSummary, ReportSummary } from '@/api/types'
import { formatMetric } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'
import Skeleton from '@/components/ui/Skeleton'
import KpiStrip, { KPI_HERO_CELL, KPI_HERO_CONTAINER, type KpiItem } from '@/components/ui/KpiStrip'
import Tabs from '@/components/ui/Tabs'
import SearchInput from '@/components/ui/SearchInput'
import Pagination from '@/components/ui/Pagination'
import EmptyState from '@/components/common/EmptyState'
import EmptyStateSystem from '@/components/common/EmptyStateSystem'
import ErrorAlert from '@/components/ui/ErrorAlert'
import { formatTimestamp } from '@/utils/formatUtils'

const RECENT_LIMIT = 15
const EMPTY_REPORTS: ReportSummary[] = []
const EMPTY_PERF_RUNS: PerfRunSummary[] = []

type RunItem =
  | { kind: 'eval'; timestamp: string; report: ReportSummary }
  | { kind: 'perf'; timestamp: string; run: PerfRunSummary }

type KindFilter = RunItem['kind'] | 'all'

const KIND_TABS: { key: KindFilter; labelKey: string; panelId: string }[] = [
  { key: 'all', labelKey: 'dashboard.tabAll', panelId: 'dashboard-results-all' },
  { key: 'eval', labelKey: 'dashboard.tabEval', panelId: 'dashboard-results-eval' },
  { key: 'perf', labelKey: 'dashboard.tabPerf', panelId: 'dashboard-results-perf' },
]

/** Landing page counters and a bounded feed of recent runs. */
export default function DashboardPage() {
  const { t } = useLocale()
  const { rootPath, scanToken } = useScan()
  const navigate = useNavigate()
  const [kindFilter, setKindFilter] = useState<KindFilter>('all')
  const [query, setQuery] = useState('')
  const [page, setPage] = useState(1)

  const overview = useAsyncResource(
    async (signal) => {
      const [evalResult, perfResult] = await Promise.allSettled([
        listReports({ rootPath, page: 1, pageSize: 100, sortBy: 'time', sortOrder: 'desc', signal }),
        listPerfRuns(rootPath, signal),
      ])
      const failure = evalResult.status === 'rejected'
        ? evalResult.reason
        : perfResult.status === 'rejected' ? perfResult.reason : null
      return {
        reports: evalResult.status === 'fulfilled' ? evalResult.value.reports : [],
        reportTotal: evalResult.status === 'fulfilled' ? evalResult.value.total : 0,
        reportModels: evalResult.status === 'fulfilled' ? evalResult.value.filters.available_models : [],
        perfRuns: perfResult.status === 'fulfilled' ? perfResult.value.runs : [],
        perfSemantics: perfResult.status === 'fulfilled' ? perfResult.value.metric_semantics : undefined,
        failure: failure instanceof Error ? failure.message : failure ? t('common.loadError') : '',
      }
    },
    [rootPath, scanToken],
    { enabled: Boolean(rootPath), fallbackMessage: t('common.loadError') },
  )

  const reports = overview.data?.reports ?? EMPTY_REPORTS
  const perfRuns = overview.data?.perfRuns ?? EMPTY_PERF_RUNS
  const items = useMemo<RunItem[]>(() => [
    ...reports.map((report): RunItem => ({ kind: 'eval', timestamp: report.timestamp || '', report })),
    ...perfRuns.map((run): RunItem => ({ kind: 'perf', timestamp: run.timestamp || '', run })),
  ].sort((left, right) => right.timestamp.localeCompare(left.timestamp)), [reports, perfRuns])

  const filteredItems = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase()
    return items.filter((item) => {
      if (kindFilter !== 'all' && item.kind !== kindFilter) return false
      if (!normalized) return true
      const values = item.kind === 'eval'
        ? [item.report.model_name, item.report.dataset_pretty_name, item.report.dataset_name]
        : [item.run.model, item.run.dataset, item.run.api_type]
      return values.some((value) => value?.toLocaleLowerCase().includes(normalized))
    })
  }, [items, kindFilter, query])

  const totalPages = Math.max(1, Math.ceil(filteredItems.length / RECENT_LIMIT))
  const safePage = Math.min(page, totalPages)
  const visibleItems = filteredItems.slice((safePage - 1) * RECENT_LIMIT, safePage * RECENT_LIMIT)
  const models = useMemo(() => new Set([
    ...(overview.data?.reportModels ?? []),
    ...perfRuns.map((run) => run.model).filter(Boolean),
  ]), [overview.data?.reportModels, perfRuns])
  const latestTimestamp = items[0]?.timestamp ?? ''
  const latestLabel = latestTimestamp ? formatTimestamp(latestTimestamp, 'seconds') : t('dashboard.neverText')
  const kpis: KpiItem[] = [
    {
      icon: <FileText size={17} strokeWidth={2} />,
      value: String(overview.data?.reportTotal ?? 0),
      label: t('dashboard.totalEvaluations'),
      onClick: () => navigate('/reports'),
    },
    {
      icon: <Gauge size={17} strokeWidth={2} />,
      value: String(perfRuns.length),
      label: t('dashboard.totalPerfRuns'),
      onClick: () => navigate('/performance'),
    },
    { icon: <Cpu size={17} strokeWidth={2} />, value: String(models.size), label: t('dashboard.modelsEvaluated') },
    { icon: <Clock size={17} strokeWidth={2} />, value: latestLabel, label: t('dashboard.latestRun') },
  ]

  const openItem = (item: RunItem) => {
    const root = encodeURIComponent(rootPath)
    if (item.kind === 'eval') {
      navigate(
        `/reports/${encodeURIComponent(item.report.run_id)}/${encodeURIComponent(item.report.model_id)}`
        + `?root_path=${root}`,
      )
      return
    }
    navigate(`/perf-report?path=${encodeURIComponent(item.run.path)}&root_path=${root}`)
  }

  const results = visibleItems.length > 0 ? (
    <div className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
      <div className="divide-y divide-[var(--border)]">
        {visibleItems.map((item) => (
          <RunRow
            key={runKey(item)}
            item={item}
            perfSemantics={overview.data?.perfSemantics?.best_rps}
            onClick={() => openItem(item)}
          />
        ))}
      </div>
      {filteredItems.length > RECENT_LIMIT && (
        <Pagination page={safePage} totalPages={totalPages} onPageChange={setPage} className="p-3" />
      )}
    </div>
  ) : (
    <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
      <EmptyStateSystem reason="no-match" context={{ view: 'dashboard' }} />
    </div>
  )

  const scanned = overview.data !== undefined
  const hasData = scanned && items.length > 0
  return (
    <div className="flex min-h-0 w-full flex-col gap-4">
      {(overview.error || overview.data?.failure) && (
        <ErrorAlert className="rounded-[var(--radius-sm)]">{overview.error || overview.data?.failure}</ErrorAlert>
      )}
      {overview.loading && !scanned ? <KpiSkeleton /> : <KpiStrip items={kpis} />}
      {overview.loading && !scanned ? (
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-4">
          <Skeleton lines={8} height={14} />
        </div>
      ) : hasData ? (
        <Tabs
          tabs={KIND_TABS}
          activeKey={kindFilter}
          onChange={(key) => {
            setKindFilter(key as KindFilter)
            setPage(1)
          }}
          panels={Object.fromEntries(KIND_TABS.map((tab) => [tab.panelId, results]))}
          actions={(
            <SearchInput
              value={query}
              onChange={(value) => {
                setQuery(value)
                setPage(1)
              }}
              placeholder={t('dashboard.searchPlaceholder')}
              className="w-full sm:w-64"
            />
          )}
        />
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

function RunRow({
  item,
  perfSemantics,
  onClick,
}: {
  item: RunItem
  perfSemantics?: MetricSemantics
  onClick: () => void
}) {
  const { t } = useLocale()
  const isEval = item.kind === 'eval'
  const model = isEval ? item.report.model_name : item.run.model
  const dataset = isEval
    ? item.report.dataset_pretty_name || item.report.dataset_name
    : item.run.dataset || item.run.api_type || 'perf'
  const result = isEval
    ? formatEvalResult(item.report, t('dashboard.datasets'))
    : formatMetric(item.run.best_rps, perfSemantics).primary
  const metadata = isEval
    ? `${item.report.num_samples} ${t('dashboard.samples')}`
    : `${item.run.num_runs} ${t('dashboard.runs')}`
  return (
    <button
      type="button"
      onClick={onClick}
      className="grid min-h-14 w-full grid-cols-[2.5rem_minmax(0,1fr)_auto] items-center gap-3 px-4 py-2 text-left transition-colors hover:bg-[var(--bg-card2)] md:grid-cols-[2.5rem_minmax(8rem,1fr)_minmax(10rem,1.5fr)_8rem_8rem]"
    >
      <span className="flex h-8 w-8 items-center justify-center rounded-[var(--radius-sm)] bg-[var(--accent-dim)] text-[var(--accent)]">
        {isEval ? <FileText size={15} /> : <Gauge size={15} />}
      </span>
      <span className="min-w-0">
        <span className="block truncate type-body-sm text-[var(--text)]">{model}</span>
        <span className="block truncate type-caption-mono text-[var(--text-muted)] md:hidden">{dataset}</span>
      </span>
      <span className="hidden min-w-0 md:block">
        <span className="block truncate type-body-sm text-[var(--text)]">{dataset}</span>
        <span className="type-caption-mono text-[var(--text-muted)]">{metadata}</span>
      </span>
      <span className="hidden type-caption-mono text-[var(--text-muted)] md:block">
        {item.timestamp.replace('T', ' ').slice(5, 16)}
      </span>
      <span className="type-caption-mono text-right text-[var(--text)]">{result}</span>
    </button>
  )
}

function formatEvalResult(report: ReportSummary, datasetsLabel: string): string {
  if (report.primary_metrics.length === 0) return '—'
  if (report.primary_metrics.length > 1) return `${report.primary_metrics.length} ${datasetsLabel}`
  const [metric] = report.primary_metrics
  return formatMetric(metric.score, metric.semantics).primary
}

function runKey(item: RunItem): string {
  return item.kind === 'eval'
    ? `eval\0${item.report.run_id}\0${item.report.model_id}`
    : `perf\0${item.run.path}`
}

function KpiSkeleton() {
  return (
    <div className={KPI_HERO_CONTAINER}>
      {Array.from({ length: 4 }).map((_, index) => (
        <div key={index} className={KPI_HERO_CELL}>
          <Skeleton width={32} height={32} className="mb-2" />
          <Skeleton width={60} height={24} className="mb-1" />
          <Skeleton width={100} height={14} />
        </div>
      ))}
    </div>
  )
}
