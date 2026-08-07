import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { FileText } from 'lucide-react'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import { listReports } from '@/api/reports'
import { listPerfRuns } from '@/api/perf'
import type { PerfRunSummary, ReportSummary } from '@/api/types'
import type { MetricSemantics } from '@/domain/metric'
import Skeleton from '@/components/ui/Skeleton'
import EmptyState from '@/components/common/EmptyState'
import EmptyStateSystem from '@/components/common/EmptyStateSystem'
import ErrorAlert from '@/components/ui/ErrorAlert'
import QuickActions from '@/components/dashboard/QuickActions'
import AggregatedResults from '@/components/dashboard/AggregatedResults'
import ActivityStrip from '@/components/dashboard/ActivityStrip'
import { aggregateRuns, totalsOf } from '@/domain/report/runAggregation'
import type { AggregatedRow, CellPoint } from '@/domain/report/runAggregation'

/**
 * Landing page: what to do next, then how the benchmarks are holding up.
 *
 * This page used to open with four counters and a weaker copy of the Evaluations list -- same feed,
 * but with a filter, a search box and a pagination control that the dedicated list pages already
 * provide with sorting and comparison selection on top. Two of the counters existed only to link to
 * those pages, and a fourth put a timestamp where a comparable quantity belongs.
 *
 * What is left is the part no other page can do. Results are aggregated by what they measure rather
 * than by when they ran, because re-running a benchmark is the normal workflow here and a flat feed
 * renders those repeats as many identical-looking rows. Anything the list pages do better is not
 * duplicated: there is no filter, no search and no pagination, only a link across to them.
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

  // The whole page is driven by this: every score ever recorded, grouped by what it measures.
  const rows = useMemo(
    () => aggregateRuns(reports, perfRuns, perfSemantics),
    [reports, perfRuns, perfSemantics],
  )
  const totals = useMemo(() => totalsOf(rows), [rows])

  const openRun = (row: AggregatedRow, point: CellPoint) => {
    const root = encodeURIComponent(rootPath)
    navigate(
      row.cell.kind === 'eval'
        ? `/reports/${encodeURIComponent(point.runId)}?root_path=${root}`
        : `/perf-report?path=${encodeURIComponent(point.runId)}&root_path=${root}`,
    )
  }

  const hasData = scanned && rows.length > 0

  if (loading && !scanned) {
    return (
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-4">
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-4">
          <Skeleton lines={2} height={32} />
        </div>
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-4">
          <Skeleton lines={8} height={14} />
        </div>
      </div>
    )
  }

  return (
    <div className="mx-auto flex min-h-0 w-full max-w-7xl flex-col gap-4">
      {loadError && <ErrorAlert className="rounded-[var(--radius-sm)]">{loadError}</ErrorAlert>}

      {scanned && <QuickActions reports={reports} perfRuns={perfRuns} />}

      {hasData ? (
        <>
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="type-body text-[var(--text)]">{t('dashboard.resultsTitle')}</h2>
            {/* Replaces the four counter cards: the same facts, in one line instead of 120px. */}
            <span className="type-caption text-[var(--text-muted)]">
              {t('dashboard.totalsSummary', {
                models: totals.models,
                benchmarks: totals.benchmarks,
                cells: totals.cells,
                runs: totals.runs,
              })}
            </span>
          </div>
          <AggregatedResults rows={rows} onOpenRun={openRun} />
          <ActivityStrip
            reports={reports}
            perfRuns={perfRuns}
            perfSemantics={perfSemantics}
            rootPath={rootPath}
          />
        </>
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
