import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Clock, Cpu, FileText, Gauge } from 'lucide-react'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import { listReports } from '@/api/reports'
import { listPerfRuns } from '@/api/perf'
import type { PerfRunSummary, ReportSummary } from '@/api/types'
import type { MetricSemantics } from '@/domain/metric'
import Skeleton from '@/components/ui/Skeleton'
import KpiCard from '@/components/ui/KpiCard'
import Tabs from '@/components/ui/Tabs'
import EmptyState from '@/components/common/EmptyState'
import EmptyStateSystem from '@/components/common/EmptyStateSystem'
import ErrorAlert from '@/components/ui/ErrorAlert'
import AggregatedResults from '@/components/dashboard/AggregatedResults'
import { aggregateRuns } from '@/domain/report/runAggregation'
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
  const visibleRows = useMemo(
    () => (kindFilter === 'all' ? rows : rows.filter((row) => row.cell.kind === kindFilter)),
    [rows, kindFilter],
  )

  const kpi = useMemo(() => {
    const models = new Set<string>()
    reports.forEach((report) => report.model_name && models.add(report.model_name))
    perfRuns.forEach((run) => run.model && models.add(run.model))
    const timestamps = [
      ...reports.map((report) => report.timestamp || ''),
      ...perfRuns.map((run) => run.timestamp || ''),
    ].filter(Boolean)
    const latest = timestamps.length > 0 ? timestamps.reduce((a, b) => (a > b ? a : b)) : ''
    return {
      evals: reports.length,
      perfs: perfRuns.length,
      models: models.size,
      latest: latest ? formatFull(latest) : t('dashboard.neverText'),
    }
  }, [reports, perfRuns, t])

  const openRun = (row: AggregatedRow, point: CellPoint) => {
    const root = encodeURIComponent(rootPath)
    navigate(
      row.cell.kind === 'eval'
        ? `/reports/${encodeURIComponent(point.runId)}?root_path=${root}`
        : `/perf-report?path=${encodeURIComponent(point.runId)}&root_path=${root}`,
    )
  }

  const hasData = scanned && rows.length > 0

  // One node, handed to whichever panel is selected: the tab decides the rows, not the markup.
  const resultsPanel = visibleRows.length > 0 ? (
    <AggregatedResults rows={visibleRows} onOpenRun={openRun} />
  ) : (
    <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
      <EmptyStateSystem reason="no-match" context={{ view: 'dashboard' }} />
    </div>
  )

  return (
    <div className="mx-auto flex min-h-0 w-full max-w-7xl flex-col gap-4">
      {loadError && <ErrorAlert className="rounded-[var(--radius-sm)]">{loadError}</ErrorAlert>}

      {loading && !scanned ? (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-5">
              <Skeleton width={40} height={40} className="mb-3" />
              <Skeleton width={60} height={28} className="mb-1" />
              <Skeleton width={100} height={14} />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <KpiCard
            icon={<FileText size={18} strokeWidth={2} />}
            value={String(kpi.evals)}
            label={t('dashboard.totalEvaluations')}
            delay={0}
            onClick={() => navigate('/reports')}
          />
          <KpiCard
            icon={<Gauge size={18} strokeWidth={2} />}
            value={String(kpi.perfs)}
            label={t('dashboard.totalPerfRuns')}
            delay={60}
            onClick={() => navigate('/performance')}
          />
          <KpiCard
            icon={<Cpu size={18} strokeWidth={2} />}
            value={String(kpi.models)}
            label={t('dashboard.modelsEvaluated')}
            delay={120}
          />
          <KpiCard
            icon={<Clock size={18} strokeWidth={2} />}
            value={kpi.latest}
            label={t('dashboard.latestRun')}
            delay={180}
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
