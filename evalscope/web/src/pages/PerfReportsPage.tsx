import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { ChevronRight, Copy, Check } from 'lucide-react'
import { useCopy } from '@/hooks/useCopy'
import { useLocale, type Translate } from '@/contexts/LocaleContext'
import { useScan } from '@/contexts/ReportsContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { useBatchDelete } from '@/hooks/useBatchDelete'
import { useScopedState } from '@/hooks/useScopedState'
import { getPerfHistoryReportUrl, listPerfRuns, deletePerfRun } from '@/api/perf'
import type { PerfRunSummary } from '@/api/types'
import Skeleton from '@/components/ui/Skeleton'
import EmptyStateSystem, { type ResolvedEmptyStateAction } from '@/components/common/EmptyStateSystem'
import SearchInput from '@/components/ui/SearchInput'
import SelectionCheckbox from '@/components/ui/SelectionCheckbox'
import SegmentedControl from '@/components/ui/SegmentedControl'
import ErrorAlert from '@/components/ui/ErrorAlert'
import ConfirmDialog from '@/components/ui/ConfirmDialog'
import SelectionTray from '@/components/reports/SelectionTray'
import { formatMetric } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'
import { formatTimestamp } from '@/utils/formatUtils'
import { resolveProvider } from '@/domain/perf/providerResolution'
import { addToSelection, preserveSelectionAcrossReorder } from '@/domain/compare/selection'

type SortKey = 'time' | 'rps' | 'latency'

/** Sort options in display order, also the source of the segmented control. */
const SORT_KEYS: readonly SortKey[] = ['time', 'rps', 'latency']

/** Stable placeholders so an unresolved read keeps a single identity. */
const EMPTY_RUNS: PerfRunSummary[] = []
const EMPTY_SELECTION: string[] = []
const EMPTY_SEMANTICS: Record<string, MetricSemantics> = {}

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
  const { copy, copied } = useCopy()
  const displayName = run.alias || run.model || run.dataset || '—'
  const metrics = [
    { key: 'best_rps', label: t('perf.archive.sort_rps'), value: run.best_rps },
    { key: 'best_latency', label: t('perf.archive.sort_latency'), value: run.best_latency },
    { key: 'success_rate', label: t('perf.archive.successColumn'), value: run.success_rate },
  ]

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
        label={`${t('perf.archive.selectRun')}: ${displayName}`}
        className="shrink-0 cursor-pointer"
      />
      <button
        type="button"
        onClick={onClick}
        className="grid min-h-16 min-w-0 flex-1 grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-3 py-2 text-left sm:grid-cols-[minmax(0,1fr)_17rem_auto]"
      >
        <div className="min-w-0 space-y-1.5">
          <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1">
            <span title={displayName} className="type-body-sm min-w-0 truncate font-semibold text-[var(--text)]">{displayName}</span>
            <time dateTime={run.timestamp || undefined} title={formatTimestamp(run.timestamp, 'seconds')} className="shrink-0 text-xs tabular-nums text-[var(--text-muted)]">
              {formatTimestamp(run.timestamp)}
            </time>
          </div>
          <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[var(--text-muted)]">
            {run.alias && <span className="max-w-48 truncate" title={run.model}>{run.model}</span>}
            {run.api_host && <span className="max-w-full truncate font-mono text-[var(--text)]" title={run.api_host}>{run.api_host}</span>}
            <span className="max-w-32 truncate" title={run.dataset || undefined}>{run.dataset || '—'}</span>
            {run.num_runs > 1 && <span>{run.num_runs} {t('perf.archive.runs')}</span>}
          </div>
        </div>
        <div className="col-span-2 row-start-2 grid grid-cols-3 gap-3 sm:col-span-1 sm:col-start-2 sm:row-start-1">
          {metrics.map(({ key, label, value }) => (
            <div key={key} className="min-w-0 sm:text-right">
              <span className="block text-xs text-[var(--text-muted)]">{label}</span>
              <span className="mt-1 block whitespace-nowrap text-sm font-medium tabular-nums text-[var(--text)]">
                {formatMetric(value, perfSemantics?.[key]).primary}
              </span>
            </div>
          ))}
        </div>
        <ChevronRight size={16} className="col-start-2 row-start-1 shrink-0 text-[var(--text-dim)] sm:col-start-3" />
      </button>
      {run.api_host && (
        <button
          type="button"
          aria-label={`${t(copied ? 'tasks.copied' : 'tasks.copy')}: ${run.api_host}`}
          title={t(copied ? 'tasks.copied' : 'tasks.copy')}
          onClick={() => copy(run.api_host!)}
          className="flex min-h-11 min-w-11 shrink-0 items-center justify-center rounded text-[var(--text-muted)] hover:text-[var(--text)]"
        >
          {copied ? <Check size={16} /> : <Copy size={16} />}
        </button>
      )}
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
    { enabled: Boolean(rootPath), fallbackMessage: t('common.loadError') },
  )
  const runs = runsResource.data?.runs ?? EMPTY_RUNS
  const perfSemantics = runsResource.data?.metric_semantics ?? EMPTY_SEMANTICS
  const loading = runsResource.loading
  const hasLoaded = runsResource.data !== undefined || Boolean(runsResource.error)

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
  const [selected, setSelected] = useScopedState<string[]>(selectionScope, EMPTY_SELECTION)

  const reloadRuns = runsResource.reload
  const deletion = useBatchDelete<string>({
    items: selected,
    deleteItem: (path) => deletePerfRun(rootPath, path),
    onSettled: setSelected,
    reload: reloadRuns,
    formatError: (msg) => t('reports.deleteFailed', { msg }),
  })

  // Deleting reports its own failure; a load failure comes from the resource.
  const error = deletion.error ?? (runsResource.error || null)

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

  const pendingDeleteItems = useMemo(
    () =>
      selected.map((path) => {
        const run = runs.find((r) => r.path === path)
        if (!run) return path
        return `${run.alias || run.model || run.dataset || path} · ${formatTimestamp(run.timestamp, 'seconds')}`
      }),
    [selected, runs],
  )

  // In-view recovery: retry re-fetches, clear-filters resets the search query;
  // other empty-state actions (create task, browse benchmarks) navigate.
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
              (r.alias || '').toLowerCase().includes(q) ||
              (r.api_host || '').toLowerCase().includes(q) ||
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
              className="w-full sm:max-w-lg sm:flex-1 [&>input]:h-10 [&>input]:py-0"
            />
            <SegmentedControl
              options={SORT_KEYS.map((k) => ({ value: k, label: t(`perf.archive.sort_${k}`) }))}
              value={sortBy}
              onChange={setSortBy}
              ariaLabel={t('perf.archive.sortLabel')}
              size="sm"
              className="h-10 w-fit"
            />
          </div>

          {visibleRuns.length > 0 ? (
            <div className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
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
            onDelete={deletion.request}
            deleting={deletion.deleting}
          />

          <ConfirmDialog
            open={deletion.confirmOpen}
            danger
            busy={deletion.deleting}
            title={t('reports.deleteConfirmTitle')}
            message={t('reports.deleteConfirm', { n: selected.length })}
            items={pendingDeleteItems}
            confirmLabel={t('reports.delete')}
            cancelLabel={t('common.cancel')}
            onConfirm={deletion.confirm}
            onCancel={deletion.cancel}
          />
        </>
      )}
    </div>
  )
}
