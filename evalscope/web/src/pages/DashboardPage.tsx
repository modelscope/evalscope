import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import { listReports } from '@/api/reports'
import { listPerfRuns } from '@/api/perf'
import type { PerfRunSummary, ReportSummary } from '@/api/types'
import Card from '@/components/ui/Card'
import Badge from '@/components/ui/Badge'
import Skeleton from '@/components/ui/Skeleton'
import KpiCard from '@/components/ui/KpiCard'
import ScoreBadge from '@/components/ui/ScoreBadge'
import EmptyState from '@/components/common/EmptyState'
import EmptyStateSystem from '@/components/common/EmptyStateSystem'
import SearchInput from '@/components/ui/SearchInput'
import Button from '@/components/ui/Button'
import { FileText, Gauge, Cpu, Clock, ChevronRight } from 'lucide-react'
import { formatMetricByKey } from '@/domain/metric/registry'

// Number of recent runs shown before the "view all" toggle.
const RECENT_LIMIT = 15

// ------------------------------------------------------------------ //
// Helpers                                                             //
// ------------------------------------------------------------------ //

/** Format ISO timestamp to short form MM-DD HH:MM. */
function formatShort(ts: string): string {
  return ts ? ts.replace('T', ' ').slice(5, 16) : ''
}

/** Format ISO timestamp to YYYY-MM-DD HH:MM:SS. */
function formatFull(ts: string): string {
  return ts ? ts.replace('T', ' ').slice(0, 19) : ''
}

// Unified recent-run item across eval + perf.
type RunItem =
  | { kind: 'eval'; ts: string; report: ReportSummary }
  | { kind: 'perf'; ts: string; run: PerfRunSummary }

// ------------------------------------------------------------------ //
// Recent run row                                                      //
// ------------------------------------------------------------------ //
function RunRow({ item, onClick }: { item: RunItem; onClick: () => void }) {
  const { t } = useLocale()
  const isEval = item.kind === 'eval'
  const model = isEval ? item.report.model_name : item.run.model
  const dataset = isEval ? item.report.dataset_name : item.run.dataset || item.run.api_type || 'perf'
  const meta = isEval
    ? `${item.report.num_samples} ${t('dashboard.samples')}`
    : `${item.run.num_runs} ${t('dashboard.runs')}`

  return (
    <button
      onClick={onClick}
      className="grid min-h-14 w-full grid-cols-[auto_minmax(0,1fr)_auto_auto] items-center gap-x-2 px-3 py-2 text-left transition-colors hover:bg-[var(--bg-card2)] focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-[var(--accent)] md:grid-cols-[4.5rem_minmax(8rem,1fr)_minmax(10rem,1.5fr)_8rem_7rem_1rem] md:gap-x-3"
    >
      <Badge
        variant="default"
        className={item.kind === 'perf' ? '!bg-[var(--bg-card2)] !text-[var(--text-muted)]' : undefined}
      >
        {isEval ? 'Eval' : 'Perf'}
      </Badge>
      <div className="flex flex-col min-w-0 flex-1">
        <span className="type-body-sm text-[var(--text)] break-words">{model}</span>
        <span className="type-caption-mono text-[var(--text-muted)] break-words md:hidden">{dataset}</span>
        <span className="type-caption-mono mt-0.5 text-[var(--text-dim)] md:hidden">{formatShort(item.ts)}</span>
      </div>
      <div className="hidden min-w-0 flex-col md:flex">
        <span className="type-body-sm break-words text-[var(--text)]">{dataset}</span>
        <span className="type-caption-mono text-[var(--text-muted)]">{meta}</span>
      </div>
      <span className="type-caption-mono hidden whitespace-nowrap text-[var(--text-muted)] md:block">
        {formatShort(item.ts)}
      </span>
      {isEval ? (
        <ScoreBadge score={item.report.score} className="shrink-0 !text-xs !px-2" />
      ) : (
        <span className="type-caption-mono text-[var(--text)] shrink-0">
          {formatMetricByKey('rps', item.run.best_rps, t).primary}
        </span>
      )}
      <ChevronRight size={14} className="text-[var(--text-dim)] shrink-0" />
    </button>
  )
}

// ------------------------------------------------------------------ //
// Dashboard (overview home)                                           //
// ------------------------------------------------------------------ //
export default function DashboardPage() {
  const { t } = useLocale()
  const { rootPath, scanToken } = useReports()
  const navigate = useNavigate()

  const [loading, setLoading] = useState(false)
  const [scanned, setScanned] = useState(false)
  const [reports, setReports] = useState<ReportSummary[]>([])
  const [perfRuns, setPerfRuns] = useState<PerfRunSummary[]>([])
  const [loadError, setLoadError] = useState('')

  // Recent-runs feed controls.
  const [query, setQuery] = useState('')
  const [typeFilter, setTypeFilter] = useState<'all' | 'eval' | 'perf'>('all')
  const [page, setPage] = useState(1)

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
      if (perfRes.status === 'fulfilled') setPerfRuns(perfRes.value.runs)
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

  // Merge into a single time-sorted feed (uncapped).
  const allItems = useMemo<RunItem[]>(() => {
    const items: RunItem[] = [
      ...reports.map((r): RunItem => ({ kind: 'eval', ts: r.timestamp || '', report: r })),
      ...perfRuns.map((r): RunItem => ({ kind: 'perf', ts: r.timestamp || '', run: r })),
    ]
    return items.sort((a, b) => b.ts.localeCompare(a.ts))
  }, [reports, perfRuns])

  // Apply the type filter + keyword search.
  const filteredItems = useMemo<RunItem[]>(() => {
    const q = query.trim().toLowerCase()
    return allItems.filter((it) => {
      if (typeFilter !== 'all' && it.kind !== typeFilter) return false
      if (!q) return true
      if (it.kind === 'eval') {
        return (
          (it.report.model_name || '').toLowerCase().includes(q) ||
          (it.report.dataset_name || '').toLowerCase().includes(q)
        )
      }
      return (
        (it.run.model || '').toLowerCase().includes(q) ||
        (it.run.dataset || '').toLowerCase().includes(q) ||
        (it.run.api_type || '').toLowerCase().includes(q)
      )
    })
  }, [allItems, typeFilter, query])

  // Paginate the filtered feed (page is reset to 1 by the filter/search handlers).
  const totalPages = Math.max(1, Math.ceil(filteredItems.length / RECENT_LIMIT))
  const safePage = Math.min(page, totalPages)
  const visibleItems = filteredItems.slice((safePage - 1) * RECENT_LIMIT, safePage * RECENT_LIMIT)

  const kpi = useMemo(() => {
    const models = new Set<string>()
    reports.forEach((r) => models.add(r.model_name))
    perfRuns.forEach((r) => r.model && models.add(r.model))
    const latestTs = allItems.length > 0 ? allItems[0].ts : ''
    return {
      evals: reports.length,
      perfs: perfRuns.length,
      models: models.size,
      latest: latestTs ? formatFull(latestTs) : t('dashboard.neverText'),
    }
  }, [reports, perfRuns, allItems, t])

  const openItem = (item: RunItem) => {
    if (item.kind === 'eval') {
      navigate(`/reports/${encodeURIComponent(item.report.name)}?root_path=${encodeURIComponent(rootPath)}`)
    } else {
      navigate(`/perf-report?path=${encodeURIComponent(item.run.path)}&root_path=${encodeURIComponent(rootPath)}`)
    }
  }

  const hasData = scanned && allItems.length > 0

  return (
    <div className="mx-auto flex min-h-0 w-full max-w-7xl flex-col gap-5">
      {loadError && (
        <div role="alert" className="rounded-[var(--radius-sm)] border border-[var(--danger-border)] bg-[var(--danger-bg)] px-4 py-3 text-sm text-[var(--danger)]">
          {loadError}
        </div>
      )}

      {/* ── KPI Cards ── */}
      {loading && !scanned ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-5">
              <Skeleton width={40} height={40} className="mb-3" />
              <Skeleton width={60} height={28} className="mb-1" />
              <Skeleton width={100} height={14} />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <KpiCard
            icon={<FileText size={18} strokeWidth={2} />}
            value={String(kpi.evals)}
            label={t('dashboard.totalEvaluations')}
            gradient="var(--kpi-grad-0)"
            delay={0}
            onClick={() => navigate('/reports')}
          />
          <KpiCard
            icon={<Gauge size={18} strokeWidth={2} />}
            value={String(kpi.perfs)}
            label={t('dashboard.totalPerfRuns')}
            gradient="var(--kpi-grad-1)"
            delay={60}
            onClick={() => navigate('/performance')}
          />
          <KpiCard
            icon={<Cpu size={18} strokeWidth={2} />}
            value={String(kpi.models)}
            label={t('dashboard.modelsEvaluated')}
            gradient="var(--kpi-grad-2)"
            delay={120}
          />
          <KpiCard
            icon={<Clock size={18} strokeWidth={2} />}
            value={kpi.latest}
            label={t('dashboard.latestRun')}
            gradient="var(--kpi-grad-3)"
            delay={180}
          />
        </div>
      )}

      {/* ── Recent Runs ── */}
      {loading && !scanned ? (
        <Card title={t('dashboard.recentRuns')}>
          <Skeleton lines={8} height={14} />
        </Card>
      ) : hasData ? (
        <Card title={t('dashboard.recentRuns')} badge={<Badge>{filteredItems.length}</Badge>}>
          {/* Filter controls */}
          <div className="mb-2 flex flex-col gap-2 sm:flex-row sm:items-center">
            <div className="flex items-center gap-1 p-0.5 rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] w-fit">
              {(['all', 'eval', 'perf'] as const).map((k) => (
                <button
                  key={k}
                  onClick={() => {
                    setTypeFilter(k)
                    setPage(1)
                  }}
                  className={[
                    'px-3 py-1 rounded-[var(--radius-sm)] type-body-xs transition-colors',
                    typeFilter === k
                      ? 'bg-[var(--accent)] text-[var(--text-on-filled)]'
                      : 'text-[var(--text-muted)] hover:text-[var(--text)]',
                  ].join(' ')}
                >
                  {t(`dashboard.filter_${k}`)}
                </button>
              ))}
            </div>
            <SearchInput
              value={query}
              onChange={(v) => {
                setQuery(v)
                setPage(1)
              }}
              placeholder={t('dashboard.searchPlaceholder')}
              className="w-full sm:ml-auto sm:w-72"
            />
          </div>

          {visibleItems.length > 0 ? (
            <div className="divide-y divide-[var(--border)] overflow-hidden rounded-[var(--radius-sm)]">
              <div className="hidden grid-cols-[4.5rem_minmax(8rem,1fr)_minmax(10rem,1.5fr)_8rem_7rem_1rem] items-center gap-x-3 border-b border-[var(--border)] px-3 py-3 text-xs font-semibold text-[var(--text-muted)] md:grid">
                <span />
                <span>{t('dashboard.model')}</span>
                <span>{t('dashboard.dataset')}</span>
                <span>{t('dashboard.date')}</span>
                <span>{t('dashboard.result')}</span>
                <span />
              </div>
              {visibleItems.map((item, i) => (
                <RunRow key={`${item.kind}-${i}`} item={item} onClick={() => openItem(item)} />
              ))}
            </div>
          ) : (
            <div className="py-8 text-center type-body-sm text-[var(--text-muted)]">{t('dashboard.noMatch')}</div>
          )}

          {filteredItems.length > RECENT_LIMIT && (
            <div className="flex items-center justify-center gap-2 mt-3">
              <Button variant="ghost" size="sm" disabled={safePage <= 1} onClick={() => setPage(Math.max(1, safePage - 1))}>
                ←
              </Button>
              {Array.from({ length: totalPages }, (_, i) => i + 1)
                .filter((p) => p === 1 || p === totalPages || Math.abs(p - safePage) <= 2)
                .reduce<(number | 'ellipsis')[]>((acc, p, idx, arr) => {
                  if (idx > 0 && p - (arr[idx - 1] as number) > 1) acc.push('ellipsis')
                  acc.push(p)
                  return acc
                }, [])
                .map((item, idx) =>
                  item === 'ellipsis' ? (
                    // text-dim allowed: decorative pagination ellipsis (DESIGN.md §Text)
                    <span key={`e${idx}`} className="px-1 text-[var(--text-dim)]">
                      ...
                    </span>
                  ) : (
                    <Button
                      key={item}
                      variant={item === safePage ? 'primary' : 'ghost'}
                      size="sm"
                      onClick={() => setPage(item as number)}
                      className="!min-w-[32px]"
                    >
                      {item}
                    </Button>
                  ),
                )}
              <Button variant="ghost" size="sm" disabled={safePage >= totalPages} onClick={() => setPage(Math.min(totalPages, safePage + 1))}>
                →
              </Button>
            </div>
          )}
        </Card>
      ) : scanned ? (
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
          <EmptyStateSystem
            reason="no-data"
            context={{ view: 'dashboard' }}
            hint={t('dashboard.noReportsHint')}
          />
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
