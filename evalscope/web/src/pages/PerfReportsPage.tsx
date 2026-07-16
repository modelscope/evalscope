import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { Gauge, ChevronRight, Check, GitCompareArrows } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import { listPerfRuns } from '@/api/perf'
import { isDomainError } from '@/api/errors'
import type { PerfRunSummary } from '@/api/types'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Skeleton from '@/components/ui/Skeleton'
import EmptyStateSystem from '@/components/common/EmptyStateSystem'
import type { ResolvedEmptyStateAction } from '@/domain/empty/emptyState'
import SearchInput from '@/components/ui/SearchInput'
import { formatMetricByKey } from '@/domain/metric/registry'
import { formatFull } from '@/utils/perf'
import { resolveProvider } from '@/domain/perf/providerResolution'

/** Locale translate contract (kept minimal so cards can format metrics). */
type Translate = (key: string, vars?: Record<string, string | number>) => string

type SortKey = 'time' | 'rps' | 'latency'

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <span className="type-caption-mono text-[var(--text)] tabular-nums">{value}</span>
      <span className="type-table-xs uppercase tracking-wider text-[var(--text-muted)]">{label}</span>
    </div>
  )
}

function Checkbox({ checked }: { checked: boolean }) {
  return (
    <span
      role="checkbox"
      aria-checked={checked}
      className={[
        'flex items-center justify-center w-4 h-4 rounded-[4px] border transition-colors',
        checked
          ? 'bg-[var(--accent)] border-[var(--accent)] text-[var(--text-on-filled)]'
          : 'border-[var(--border-md)] text-transparent',
      ].join(' ')}
    >
      <Check size={12} />
    </span>
  )
}

function PerfRunCard({
  run,
  selected,
  onToggle,
  onClick,
  t,
}: {
  run: PerfRunSummary
  selected: boolean
  onToggle: () => void
  onClick: () => void
  t: Translate
}) {
  const identity = resolveProvider(run)
  const concurrency = run.concurrency?.length ? run.concurrency.join(', ') : 'N/A'

  return (
    <div
      className={[
        'flex items-center gap-3 p-4 rounded-[var(--radius)] border bg-[var(--bg-card)] transition-colors',
        selected ? 'border-[var(--accent)]' : 'border-[var(--border)] hover:border-[var(--border-strong)]',
      ].join(' ')}
    >
      <button type="button" onClick={onToggle} aria-label="Select for compare" className="shrink-0 flex min-h-11 min-w-11 items-center justify-center cursor-pointer">
        <Checkbox checked={selected} />
      </button>
      <button type="button" onClick={onClick} className="flex min-h-11 items-center gap-4 flex-1 min-w-0 text-left">
        <span className="text-[var(--accent)] shrink-0">
          <Gauge size={20} />
        </span>
        <div className="flex flex-col min-w-0 flex-1 gap-0.5">
          <div className="flex flex-wrap items-center gap-2 min-w-0">
            {/* Model alias is the primary identity; fall back to dataset, never
                the raw path/timestamp, when the alias is absent (Req 8.9). */}
            <span className="type-title-md text-[var(--text)] break-words min-w-0">{run.model || run.dataset || '—'}</span>
          </div>
          <span className="type-caption-mono text-[var(--text-muted)] break-words">
            {t('performance.provider')}: {identity.provider} · {t('performance.protocol')}: {identity.protocol}
          </span>
          <span className="type-caption-mono text-[var(--text-muted)] break-words">
            {t('performance.concurrency')}: {concurrency} · {t('performance.numberOfRequests')}: {run.total_requests}
          </span>
          <span className="type-caption-mono text-[var(--text-muted)] break-words">
            {(run.dataset || '—')} · {formatFull(run.timestamp)}
          </span>
        </div>
        <div className="hidden lg:flex items-center gap-6 shrink-0">
          <Stat label="Runs" value={String(run.num_runs)} />
          {/* Domain metrics render through the shared formatter so the same
              value rounds identically here, in the detail view and per-run
              tables (Req 8.10). */}
          <Stat label="Best RPS" value={formatMetricByKey('rps', run.best_rps, t).primary} />
          <Stat label="Min Lat" value={formatMetricByKey('latency', run.best_latency, t).primary} />
          <Stat label="Success" value={formatMetricByKey('success_rate', run.success_rate, t).primary} />
        </div>
        <ChevronRight size={16} className="text-[var(--text-dim)] shrink-0" />
      </button>
    </div>
  )
}

export default function PerfReportsPage() {
  const { t } = useLocale()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { rootPath, scanToken, setRootPath } = useReports()

  // Sync root_path from URL on mount (e.g. when navigating back from a detail
  // or compare page, which carry the active root in their breadcrumbs).
  useEffect(() => {
    const urlRoot = searchParams.get('root_path')
    if (urlRoot) setRootPath(urlRoot)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const [runs, setRuns] = useState<PerfRunSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [hasLoaded, setHasLoaded] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Bumped to re-trigger the fetch effect when the user retries from an empty state.
  const [reloadToken, setReloadToken] = useState(0)

  // List controls (symmetric with the Evaluations page).
  const [query, setQuery] = useState('')
  const [sortBy, setSortBy] = useState<SortKey>('time')

  // Multi-select for cross-run comparison (page-local; independent of eval Compare).
  const [selected, setSelected] = useState<string[]>([])

  const toggleSelect = (path: string) =>
    setSelected((prev) => (prev.includes(path) ? prev.filter((p) => p !== path) : [...prev, path]))

  const compareSelected = () => {
    if (selected.length < 2) return
    // Forward the first run's mode so the compare page can hide TTFT/TPOT for
    // embedding/rerank runs without an extra detail round-trip.
    const first = runs.find((r) => r.path === selected[0])
    const embedding = first?.is_embedding ? '1' : '0'
    navigate(
      `/perf-compare?paths=${encodeURIComponent(selected.join(';'))}`
        + `&embedding=${embedding}&root_path=${encodeURIComponent(rootPath)}`,
    )
  }

  useEffect(() => {
    if (!rootPath) return
    const controller = new AbortController()
    const load = async () => {
      setLoading(true)
      setError(null)
      try {
        const res = await listPerfRuns(rootPath, controller.signal)
        if (!controller.signal.aborted) {
          setRuns(res.runs)
          setSelected([])
        }
      } catch (err) {
        if (!controller.signal.aborted && !(isDomainError(err) && err.kind === 'aborted')) {
          setError(err instanceof Error ? err.message : 'Failed to load perf runs')
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false)
          setHasLoaded(true)
        }
      }
    }
    load()
    return () => controller.abort()
  }, [rootPath, scanToken, reloadToken])

  // In-view recovery: retry re-fetches, clear-filters resets the search query;
  // other empty-state actions (create task, browse benchmarks) navigate (Req 6.2, 6.3).
  const handleEmptyAction = useCallback((action: ResolvedEmptyStateAction) => {
    if (action.navigateTo === '#retry') {
      setReloadToken((n) => n + 1)
      return true
    }
    if (action.navigateTo === '#clear-filters') {
      setQuery('')
      return true
    }
    return false
  }, [])

  const openRun = (run: PerfRunSummary) => {
    navigate(`/perf-report?path=${encodeURIComponent(run.path)}&root_path=${encodeURIComponent(rootPath)}`)
  }

  // Apply keyword search + sort.
  const visibleRuns = useMemo(() => {
    const q = query.trim().toLowerCase()
    const filtered = q
      ? runs.filter(
          (r) => {
            const identity = resolveProvider(r)
            return (
              (r.model || '').toLowerCase().includes(q) ||
              (r.dataset || '').toLowerCase().includes(q) ||
              (r.api_type || '').toLowerCase().includes(q) ||
              identity.provider.toLowerCase().includes(q) ||
              identity.protocol.toLowerCase().includes(q)
            )
          },
        )
      : runs
    const sorted = [...filtered]
    if (sortBy === 'rps') sorted.sort((a, b) => b.best_rps - a.best_rps)
    else if (sortBy === 'latency') sorted.sort((a, b) => a.best_latency - b.best_latency)
    else sorted.sort((a, b) => (b.timestamp || '').localeCompare(a.timestamp || ''))
    return sorted
  }, [runs, query, sortBy])

  return (
    <div className="page-enter flex flex-col gap-5">
      <Breadcrumb items={[{ label: t('nav.performance') }]} />

      {error && (
        <div role="alert" className="px-4 py-3 rounded-[var(--radius)] bg-[var(--danger-bg)] border border-[var(--danger-border)] text-sm text-[var(--danger)]">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex flex-col gap-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} height={72} className="rounded-[var(--radius)]" />
          ))}
        </div>
      ) : runs.length === 0 ? (
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
          <EmptyStateSystem
            reason={error ? 'load-error' : 'no-data'}
            context={{ view: 'performance', retryTo: '#retry' }}
            hint={!error && hasLoaded ? t('performance.noRunsHint') : undefined}
            onAction={handleEmptyAction}
          />
        </div>
      ) : (
        <>
          {/* Controls */}
          <div className="flex flex-col sm:flex-row sm:items-center gap-2">
            <div className="flex items-center gap-1 p-0.5 rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] w-fit">
              {(['time', 'rps', 'latency'] as const).map((k) => (
                <button
                  key={k}
                  onClick={() => setSortBy(k)}
                  className={[
                    'min-h-11 px-3 py-1 rounded-[var(--radius-sm)] type-body-xs transition-colors',
                    sortBy === k
                      ? 'bg-[var(--accent)] text-[var(--text-on-filled)]'
                      : 'text-[var(--text-muted)] hover:text-[var(--text)]',
                  ].join(' ')}
                >
                  {t(`performance.sort_${k}`)}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2 sm:ml-auto w-full sm:w-auto">
              <button
                onClick={compareSelected}
                disabled={selected.length < 2}
                className={[
                  'inline-flex min-h-11 items-center gap-1.5 px-3 py-1.5 rounded-[var(--radius-sm)] type-body-sm border transition-colors shrink-0',
                  selected.length >= 2
                    ? 'border-[var(--accent)] text-[var(--accent)] hover:bg-[var(--bg-card2)] cursor-pointer'
                    : 'border-[var(--border)] text-[var(--text-dim)] cursor-not-allowed',
                ].join(' ')}
              >
                <GitCompareArrows size={14} />
                {t('performance.compareN', { n: selected.length })}
              </button>
              <SearchInput
                value={query}
                onChange={setQuery}
                placeholder={t('performance.searchPlaceholder')}
                className="w-full sm:w-64"
              />
            </div>
          </div>

          {visibleRuns.length > 0 ? (
            <div className="flex flex-col gap-2">
              {visibleRuns.map((run) => (
                <PerfRunCard
                  key={run.path}
                  run={run}
                  selected={selected.includes(run.path)}
                  onToggle={() => toggleSelect(run.path)}
                  onClick={() => openRun(run)}
                  t={t}
                />
              ))}
            </div>
          ) : (
            <EmptyStateSystem
              reason="no-match"
              context={{ view: 'performance', clearFiltersTo: '#clear-filters' }}
              onAction={handleEmptyAction}
            />
          )}
        </>
      )}
    </div>
  )
}
