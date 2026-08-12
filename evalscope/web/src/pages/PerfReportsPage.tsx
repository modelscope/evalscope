import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { ChevronRight } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { useScan } from '@/contexts/ReportsContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { getPerfHistoryReportUrl, listPerfRuns, deletePerfRun } from '@/api/perf'
import type { PerfRunSummary } from '@/api/types'
import Skeleton from '@/components/ui/Skeleton'
import EmptyStateSystem, { type ResolvedEmptyStateAction } from '@/components/common/EmptyStateSystem'
import SearchInput from '@/components/ui/SearchInput'
import SelectionCheckbox from '@/components/ui/SelectionCheckbox'
import ErrorAlert from '@/components/ui/ErrorAlert'
import ConfirmDialog from '@/components/ui/ConfirmDialog'
import SelectionTray from '@/components/reports/SelectionTray'
import { formatMetric } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'
import { formatTimestamp } from '@/utils/formatUtils'
import { resolveProvider } from '@/domain/perf/providerResolution'
import { addToSelection, preserveSelectionAcrossReorder } from '@/domain/compare/selection'

/** Locale translate contract (kept minimal so cards can format metrics). */
type Translate = (key: string, vars?: Record<string, string | number>) => string

type SortKey = 'time' | 'rps' | 'latency'

/** Stable placeholders so an unresolved read keeps a single identity. */
const EMPTY_RUNS: PerfRunSummary[] = []
const EMPTY_SELECTION: string[] = []
const EMPTY_SEMANTICS: Record<string, MetricSemantics> = {}

/** Format the avg input/output token pair as e.g. `10000→300t`; `—` when absent. */
function formatIoTokens(run: PerfRunSummary): string {
  const input = run.avg_input_tokens
  const output = run.avg_output_tokens
  // Only treat missing fields as "no data"; a legitimate 0 still renders.
  if (input == null || output == null) return '—'
  return `${Math.round(input)}→${Math.round(output)}t`
}

function PerfRunCard({
  run,
  selected,
  onToggle,
  onClick,
  t,
  perfSemantics,
}: {
  run: PerfRunSummary
  selected: boolean
  onToggle: () => void
  onClick: () => void
  t: Translate
  /** Field key -> semantics, provided by the perf run list API. */
  perfSemantics?: Record<string, MetricSemantics | undefined>
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
        label={`${t('perf.archive.selectRun')}: ${run.model || run.dataset || '—'}`}
        className="shrink-0 cursor-pointer"
      />
      <button
        type="button"
        onClick={onClick}
        className="grid min-h-11 min-w-0 flex-1 grid-cols-[minmax(0,1fr)_auto] items-center gap-3 text-left lg:grid-cols-[minmax(11rem,1.5fr)_minmax(10rem,1.2fr)_9.5rem_7rem_7rem_6rem_6rem_auto]"
      >
        <div className="flex min-w-0 flex-col gap-0.5">
          {/* Model alias is the primary identity; fall back to dataset, never
              the raw path/timestamp, when the alias is absent. */}
          <span className="type-body-sm font-semibold text-[var(--text)] break-words min-w-0">{run.model || run.dataset || '—'}</span>
          <span className="type-caption-mono text-[var(--text-muted)] break-words">
            {identity.provider} · {identity.protocol}
          </span>
          <span className="type-caption-mono text-[var(--text-muted)] break-words lg:hidden">
            {t('perf.archive.runMeta', { concurrency, requests: run.total_requests, runs: run.num_runs })}
          </span>
          <span className="type-caption-mono text-[var(--text-muted)] break-words lg:hidden">
            {(run.dataset || '—')} · {formatIoTokens(run)} · {formatTimestamp(run.timestamp, 'seconds')}
          </span>
        </div>
        <div className="hidden min-w-0 flex-col gap-0.5 lg:flex">
          <span className="type-body-sm text-[var(--text)] break-words">{run.dataset || '—'}</span>
          <span className="type-caption-mono text-[var(--text-muted)] break-words">
            {t('perf.archive.runMeta', { concurrency, requests: run.total_requests, runs: run.num_runs })}
          </span>
        </div>
        <span className="type-caption-mono hidden whitespace-nowrap text-[var(--text-muted)] lg:block">
          {formatTimestamp(run.timestamp, 'seconds')}
        </span>
        <span className="type-caption-mono hidden whitespace-nowrap text-[var(--text)] lg:block">
          {formatIoTokens(run)}
        </span>
        {/* Domain metrics render through the shared formatter so the same
            value rounds identically here, in the detail view and per-run
            tables. */}
        <span className="type-caption-mono hidden whitespace-nowrap text-[var(--text)] lg:block">
          {formatMetric(run.best_rps, perfSemantics?.best_rps).primary}
        </span>
        <span className="type-caption-mono hidden whitespace-nowrap text-[var(--text)] lg:block">
          {formatMetric(run.best_latency, perfSemantics?.best_latency).primary}
        </span>
        <span className="type-caption-mono hidden whitespace-nowrap text-[var(--text)] lg:block">
          {formatMetric(run.success_rate, perfSemantics?.success_rate).primary}
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
  const { rootPath, scanToken, setRootPath } = useScan()

  // Sync root_path from URL on mount (e.g. when navigating back from a detail
  // or compare page, which carry the active root in their breadcrumbs).
  useEffect(() => {
    const urlRoot = searchParams.get('root_path')
    if (urlRoot) setRootPath(urlRoot)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const runsResource = useAsyncResource(
    (signal) => listPerfRuns(rootPath, signal),
    [rootPath, scanToken],
    { enabled: Boolean(rootPath), fallbackMessage: 'Failed to load perf runs' },
  )
  const runs = runsResource.data?.runs ?? EMPTY_RUNS
  const perfSemantics = runsResource.data?.metric_semantics ?? EMPTY_SEMANTICS
  const loading = runsResource.loading
  const hasLoaded = runsResource.data !== undefined || Boolean(runsResource.error)

  // Deleting reports its own failure; a load failure comes from the resource.
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const error = deleteError ?? (runsResource.error || null)

  // List controls (symmetric with the Evaluations page).
  const [query, setQuery] = useState('')
  const [sortBy, setSortBy] = useState<SortKey>('time')

  // Multi-select for cross-run comparison (page-local; independent of eval Compare).
  // Unbounded: the perf compare view overlays any number of runs, and the same
  // selection drives batch delete.
  // A new root or a rescan describes a different set of runs, so a selection made
  // under the old scope is dropped by comparison rather than by an effect. A plain
  // reload keeps the same scope, and with it the selection.
  const selectionScope = `${rootPath}\0${scanToken}`
  const [picked, setPicked] = useState<{ scope: string; paths: string[] }>({ scope: '', paths: [] })
  const selected = picked.scope === selectionScope ? picked.paths : EMPTY_SELECTION
  const setSelected = useCallback(
    (next: string[] | ((prev: string[]) => string[])) => setPicked((prev) => {
      const base = prev.scope === selectionScope ? prev.paths : EMPTY_SELECTION
      return { scope: selectionScope, paths: typeof next === 'function' ? next(base) : next }
    }),
    [selectionScope],
  )
  const [deleting, setDeleting] = useState(false)
  const [confirmOpen, setConfirmOpen] = useState(false)

  const toggleSelect = (path: string) => {
    if (selected.includes(path)) {
      setSelected(selected.filter((p) => p !== path))
      return
    }
    setSelected(addToSelection(selected, path))
  }

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

  const selectedRun = selected.length === 1 ? runs.find((run) => run.path === selected[0]) : undefined

  const viewSelectedHtml = () => {
    if (!selectedRun?.has_html) return
    window.open(getPerfHistoryReportUrl(rootPath, selectedRun.path), '_blank')
  }

  const requestDeleteSelected = () => {
    if (selected.length === 0 || deleting) return
    setConfirmOpen(true)
  }

  const confirmDeleteSelected = async () => {
    if (deleting || selected.length === 0) return
    setDeleting(true)
    setDeleteError(null)
    const deleted: string[] = []
    try {
      for (const path of selected) {
        await deletePerfRun(rootPath, path)
        deleted.push(path)
      }
      setSelected([])
    } catch (err) {
      setSelected((prev) => prev.filter((p) => !deleted.includes(p)))
      setDeleteError(t('reports.deleteFailed', { msg: err instanceof Error ? err.message : String(err) }))
    } finally {
      setDeleting(false)
      setConfirmOpen(false)
      runsResource.reload()
    }
  }

  const pendingDeleteItems = useMemo(
    () =>
      selected.map((path) => {
        const run = runs.find((r) => r.path === path)
        if (!run) return path
        return `${run.model || run.dataset || path} · ${formatTimestamp(run.timestamp, 'seconds')}`
      }),
    [selected, runs],
  )

  // In-view recovery: retry re-fetches, clear-filters resets the search query;
  // other empty-state actions (create task, browse benchmarks) navigate.
  const reloadRuns = runsResource.reload
  const handleEmptyAction = useCallback((action: ResolvedEmptyStateAction) => {
    if (action.navigateTo === '#retry') {
      reloadRuns()
      return true
    }
    if (action.navigateTo === '#clear-filters') {
      setQuery('')
      return true
    }
    return false
  }, [reloadRuns])

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
            hint={!error && hasLoaded ? t('perf.archive.noRunsHint') : undefined}
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
              placeholder={t('perf.archive.searchPlaceholder')}
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
                  {t(`perf.archive.sort_${k}`)}
                </button>
              ))}
            </div>
          </div>

          {visibleRuns.length > 0 ? (
            <div className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
              <div className="hidden grid-cols-[2.75rem_minmax(11rem,1.5fr)_minmax(10rem,1.2fr)_9.5rem_7rem_7rem_6rem_6rem_1rem] items-center gap-3 border-b border-[var(--border)] px-3 py-3 text-xs font-semibold text-[var(--text-muted)] lg:grid">
                <span />
                <span>{t('reports.columns.model')}</span>
                <span>{t('reports.columns.dataset')}</span>
                <span>{t('reports.columns.time')}</span>
                <span>{t('perf.archive.ioTokens')}</span>
                <span>{t('perf.archive.sort_rps')}</span>
                <span>{t('perf.archive.sort_latency')}</span>
                <span>{t('perf.archive.successColumn')}</span>
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
                    perfSemantics={perfSemantics}
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
            canViewHtml={orderedSelection.length === 1 && !!selectedRun?.has_html}
            onViewHtml={viewSelectedHtml}
            onCompare={compareSelected}
            onClear={() => setSelected([])}
            onDelete={requestDeleteSelected}
            deleting={deleting}
          />

          <ConfirmDialog
            open={confirmOpen}
            danger
            busy={deleting}
            title={t('reports.deleteConfirmTitle')}
            message={t('reports.deleteConfirm', { n: selected.length })}
            items={pendingDeleteItems}
            confirmLabel={t('reports.delete')}
            cancelLabel={t('common.cancel')}
            onConfirm={confirmDeleteSelected}
            onCancel={() => setConfirmOpen(false)}
          />
        </>
      )}
    </div>
  )
}
