import { useEffect, useMemo, useState, type ReactNode } from 'react'
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
import SearchInput from '@/components/ui/SearchInput'
import Button from '@/components/ui/Button'
import { FileText, Gauge, Cpu, Clock, Inbox, FlaskConical, ChevronRight } from 'lucide-react'

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
  const isEval = item.kind === 'eval'
  const model = isEval ? item.report.model_name : item.run.model
  const sub = isEval
    ? item.report.dataset_name
    : `${item.run.dataset || item.run.api_type || 'perf'} · ${item.run.num_runs} runs`

  return (
    <button
      onClick={onClick}
      className="flex items-center gap-3 py-2.5 px-3 rounded-[var(--radius-sm)] hover:bg-[var(--bg-card2)] transition-colors w-full text-left"
    >
      <Badge
        variant="default"
        className={item.kind === 'perf' ? '!bg-[var(--bg-card2)] !text-[var(--text-muted)]' : undefined}
      >
        {isEval ? 'Eval' : 'Perf'}
      </Badge>
      <span className="type-caption-mono text-[var(--text-muted)] shrink-0 w-[100px]">
        {formatShort(item.ts)}
      </span>
      <div className="flex flex-col min-w-0 flex-1">
        <span className="type-body-sm text-[var(--text)] truncate">{model}</span>
        <span className="type-caption-mono text-[var(--text-muted)] truncate">{sub}</span>
      </div>
      {isEval ? (
        <ScoreBadge score={item.report.score} className="shrink-0 !text-xs !px-2" />
      ) : (
        <span className="type-caption-mono text-[var(--text)] shrink-0">
          {item.run.best_rps.toFixed(2)} req/s
        </span>
      )}
      <ChevronRight size={14} className="text-[var(--text-dim)] shrink-0" />
    </button>
  )
}

// ------------------------------------------------------------------ //
// Quick-link card                                                     //
// ------------------------------------------------------------------ //
function QuickLink({
  icon,
  title,
  desc,
  onClick,
}: {
  icon: ReactNode
  title: string
  desc: string
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-3 p-4 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] hover:bg-[var(--bg-card2)] hover:border-[var(--border-strong)] transition-colors text-left"
    >
      <span className="text-[var(--accent)] shrink-0">{icon}</span>
      <div className="min-w-0">
        <div className="type-body-sm-strong text-[var(--text)]">{title}</div>
        <div className="type-body-xs text-[var(--text-muted)] truncate">{desc}</div>
      </div>
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

  // Recent-runs feed controls.
  const [query, setQuery] = useState('')
  const [typeFilter, setTypeFilter] = useState<'all' | 'eval' | 'perf'>('all')
  const [page, setPage] = useState(1)

  // Fetch eval + perf whenever the global scan token or root changes.
  useEffect(() => {
    if (!rootPath) return
    let cancelled = false
    const load = async () => {
      setLoading(true)
      const [evalRes, perfRes] = await Promise.allSettled([
        listReports({ rootPath, pageSize: 1000, sortBy: 'time', sortOrder: 'desc' }),
        listPerfRuns(rootPath),
      ])
      if (cancelled) return
      setReports(evalRes.status === 'fulfilled' ? evalRes.value.reports : [])
      setPerfRuns(perfRes.status === 'fulfilled' ? perfRes.value.runs : [])
      setScanned(true)
      setLoading(false)
    }
    load()
    return () => {
      cancelled = true
    }
  }, [rootPath, scanToken])

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
    <div className="flex flex-col gap-5 min-h-0">
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
            value={kpi.latest.length > 20 ? kpi.latest.slice(0, 20) + '…' : kpi.latest}
            label={t('dashboard.latestRun')}
            gradient="var(--kpi-grad-3)"
            delay={180}
          />
        </div>
      )}

      {/* ── Quick Links ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <QuickLink
          icon={<FileText size={20} />}
          title={t('nav.evaluations')}
          desc={t('dashboard.browseReportsDesc')}
          onClick={() => navigate('/reports')}
        />
        <QuickLink
          icon={<Gauge size={20} />}
          title={t('nav.performance')}
          desc={t('dashboard.newPerfDesc')}
          onClick={() => navigate('/performance')}
        />
        <QuickLink
          icon={<FlaskConical size={20} />}
          title={t('nav.tasks')}
          desc={t('dashboard.newEvalDesc')}
          onClick={() => navigate('/tasks')}
        />
      </div>

      {/* ── Recent Runs ── */}
      {loading && !scanned ? (
        <Card title={t('dashboard.recentRuns')}>
          <Skeleton lines={8} height={14} />
        </Card>
      ) : hasData ? (
        <Card title={t('dashboard.recentRuns')} badge={<Badge>{filteredItems.length}</Badge>}>
          {/* Filter controls */}
          <div className="flex flex-col sm:flex-row sm:items-center gap-2 mb-3">
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
              className="sm:ml-auto w-full sm:w-64"
            />
          </div>

          {visibleItems.length > 0 ? (
            <div className="flex flex-col gap-1">
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
          <EmptyState
            icon={<Inbox size={28} strokeWidth={1.5} />}
            title={t('dashboard.noReportsYet')}
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
