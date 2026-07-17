import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { ChevronRight } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import { getPerfHistoryReportUrl, listPerfRuns } from '@/api/perf'
import { isDomainError } from '@/api/errors'
import type { PerfRunSummary } from '@/api/types'
import Skeleton from '@/components/ui/Skeleton'
import EmptyStateSystem, { type ResolvedEmptyStateAction } from '@/components/common/EmptyStateSystem'
import SearchInput from '@/components/ui/SearchInput'
import SelectionCheckbox from '@/components/ui/SelectionCheckbox'
import ErrorAlert from '@/components/ui/ErrorAlert'
import SelectionTray from '@/components/reports/SelectionTray'
import { formatMetricByKey } from '@/domain/metric/registry'
import { formatFull } from '@/utils/perf'
import { resolveProvider } from '@/domain/perf/providerResolution'
import { addToSelection, preserveSelectionAcrossReorder } from '@/domain/compare/compareModel'

/** Locale translate contract (kept minimal so cards can format metrics). */
type Translate = (key: string, vars?: Record<string, string | number>) => string

type SortKey = 'time' | 'rps' | 'latency'

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
        'flex items-center gap-1 px-3 py-2 transition-colors',
        selected ? 'bg-[var(--accent-dim)]' : 'hover:bg-[var(--bg-card2)]',
      ].join(' ')}
    >
      <SelectionCheckbox
        checked={selected}
        onClick={onToggle}
        label={`${t('performance.selectRun')}: ${run.model || run.dataset || '—'}`}
        className="shrink-0 cursor-pointer"
      />
      <button
        type="button"
        onClick={onClick}
        className="grid min-h-11 min-w-0 flex-1 grid-cols-[minmax(0,1fr)_auto] items-center gap-3 text-left lg:grid-cols-[minmax(11rem,1.5fr)_minmax(10rem,1.2fr)_9.5rem_7rem_6rem_6rem_auto]"
      >
        <div className="flex min-w-0 flex-col gap-0.5">
          {/* Model alias is the primary identity; fall back to dataset, never
              the raw path/timestamp, when the alias is absent. */}
          <span className="type-body-sm font-semibold text-[var(--text)] break-words min-w-0">{run.model || run.dataset || '—'}</span>
          <span className="type-caption-mono text-[var(--text-muted)] break-words">
            {identity.provider} · {identity.protocol}
          </span>
          <span className="type-caption-mono text-[var(--text-muted)] break-words lg:hidden">
            {t('performance.runMeta', { concurrency, requests: run.total_requests, runs: run.num_runs })}
          </span>
          <span className="type-caption-mono text-[var(--text-muted)] break-words lg:hidden">
            {(run.dataset || '—')} · {formatFull(run.timestamp)}
          </span>
        </div>
        <div className="hidden min-w-0 flex-col gap-0.5 lg:flex">
          <span className="type-body-sm text-[var(--text)] break-words">{run.dataset || '—'}</span>
          <span className="type-caption-mono text-[var(--text-muted)] break-words">
            {t('performance.runMeta', { concurrency, requests: run.total_requests, runs: run.num_runs })}
          </span>
        </div>
        <span className="type-caption-mono hidden whitespace-nowrap text-[var(--text-muted)] lg:block">
          {formatFull(run.timestamp)}
        </span>
        {/* Domain metrics render through the shared formatter so the same
            value rounds identically here, in the detail view and per-run
            tables. */}
        <span className="type-caption-mono hidden whitespace-nowrap text-[var(--text)] lg:block">
          {formatMetricByKey('rps', run.best_rps, t).primary}
        </span>
        <span className="type-caption-mono hidden whitespace-nowrap text-[var(--text)] lg:block">
          {formatMetricByKey('latency', run.best_latency, t).primary}
        </span>
        <span className="type-caption-mono hidden whitespace-nowrap text-[var(--text)] lg:block">
          {formatMetricByKey('success_rate', run.success_rate, t).primary}
        </span>
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
  const [capNotice, setCapNotice] = useState(false)
  const capTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  const selectionScope = useRef('')

  const toggleSelect = (path: string) => {
    if (selected.includes(path)) {
      setSelected(selected.filter((p) => p !== path))
      return
    }
    const { next, rejected } = addToSelection(selected, path)
    if (rejected) {
      setCapNotice(true)
      clearTimeout(capTimer.current)
      capTimer.current = setTimeout(() => setCapNotice(false), 3000)
      return
    }
    setSelected(next)
  }

  const compareSelected = () => {
    if (selected.length < 2) return
    // Forward the first run's mode so the compare page can hide TTFT/TPOT for
    // embedding/rerank runs without an extra detail round-trip.
    const first = runs.find((r) => r.path === selected[0])
    const embedding = first?.is_embedding ? '1' : '0'
    navigate(
      `/perf-compare?paths=${encodeURIComponent(selected.slice(0, 3).join(';'))}`
        + `&embedding=${embedding}&root_path=${encodeURIComponent(rootPath)}`,
    )
  }

  const selectedRun = selected.length === 1 ? runs.find((run) => run.path === selected[0]) : undefined

  const viewSelectedHtml = () => {
    if (!selectedRun?.has_html) return
    window.open(getPerfHistoryReportUrl(rootPath, selectedRun.path), '_blank')
  }

  useEffect(() => () => clearTimeout(capTimer.current), [])

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
          const nextScope = `${rootPath}\0${scanToken}`
          if (selectionScope.current !== nextScope) {
            selectionScope.current = nextScope
            setSelected([])
          }
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
  // other empty-state actions (create task, browse benchmarks) navigate.
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

  const orderedSelection = useMemo(
    () => preserveSelectionAcrossReorder(selected, visibleRuns.map((run) => run.path)),
    [selected, visibleRuns],
  )

  return (
    <div className="page-enter mx-auto flex w-full max-w-7xl flex-col gap-5">
      {error && (
        <ErrorAlert>{error}</ErrorAlert>
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
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
            <SearchInput
              value={query}
              onChange={setQuery}
              placeholder={t('performance.searchPlaceholder')}
              className="w-full sm:w-72 [&>input]:h-10 [&>input]:py-0"
            />
            <div className="flex h-10 w-fit items-center gap-1 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-deep)] p-0.5">
              {(['time', 'rps', 'latency'] as const).map((k) => (
                <button
                  key={k}
                  onClick={() => setSortBy(k)}
                  className={[
                    'h-8 px-3 rounded-[var(--radius-sm)] type-body-xs transition-colors',
                    sortBy === k
                      ? 'bg-[var(--accent)] text-[var(--text-on-filled)]'
                      : 'text-[var(--text-muted)] hover:text-[var(--text)]',
                  ].join(' ')}
                >
                  {t(`performance.sort_${k}`)}
                </button>
              ))}
            </div>
          </div>

          {visibleRuns.length > 0 ? (
            <div className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
              <div className="hidden grid-cols-[2.75rem_minmax(11rem,1.5fr)_minmax(10rem,1.2fr)_9.5rem_7rem_6rem_6rem_1rem] items-center gap-3 border-b border-[var(--border)] px-3 py-3 text-xs font-semibold text-[var(--text-muted)] lg:grid">
                <span />
                <span>{t('reports.columns.model')}</span>
                <span>{t('reports.columns.dataset')}</span>
                <span>{t('reports.columns.time')}</span>
                <span>{t('performance.sort_rps')}</span>
                <span>{t('performance.sort_latency')}</span>
                <span>{t('performance.successColumn')}</span>
                <span />
              </div>
              <div className="divide-y divide-[var(--border)]">
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
            </div>
          ) : (
            <EmptyStateSystem
              reason="no-match"
              context={{ view: 'performance', clearFiltersTo: '#clear-filters' }}
              onAction={handleEmptyAction}
            />
          )}

          <SelectionTray
            count={orderedSelection.length}
            capNotice={capNotice}
            canViewHtml={orderedSelection.length === 1 && !!selectedRun?.has_html}
            onViewHtml={viewSelectedHtml}
            onCompare={compareSelected}
            onClear={() => setSelected([])}
          />
        </>
      )}
    </div>
  )
}
