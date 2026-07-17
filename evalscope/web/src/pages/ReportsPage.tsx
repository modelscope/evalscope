import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import * as reportsApi from '@/api/reports'
import { isDomainError } from '@/api/errors'
import type { ListReportsResponse, ReportSummary } from '@/api/types'
import Skeleton from '@/components/ui/Skeleton'
import SelectionCheckbox from '@/components/ui/SelectionCheckbox'
import Pagination from '@/components/ui/Pagination'
import ErrorAlert from '@/components/ui/ErrorAlert'
import EmptyStateSystem, {
  type EmptyReason,
  type ResolvedEmptyStateAction,
} from '@/components/common/EmptyStateSystem'
import ReportFiltersBar, { type ReportFilters } from '@/components/reports/ReportFilters'
import ReportCard from '@/components/reports/ReportCard'
import ReportsTable from '@/components/reports/ReportsTable'
import SelectionTray from '@/components/reports/SelectionTray'
import {
  addToSelection,
  preserveSelectionAcrossReorder,
} from '@/domain/compare/compareModel'

const PAGE_SIZE = 20

const defaultFilters: ReportFilters = {
  search: '',
  models: [],
  datasets: [],
  scoreMin: 0,
  scoreMax: 1,
  sortBy: 'time',
  sortOrder: 'desc',
}

export default function ReportsPage() {
  const { t } = useLocale()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()

  const {
    rootPath,
    scanToken,
    setRootPath,
    selectedForCompare,
    setCompareSelection,
    clearCompareSelection,
  } = useReports()

  // ---- Local state ----
  const [filters, setFilters] = useState<ReportFilters>(defaultFilters)
  const [page, setPage] = useState(1)
  const [reports, setReports] = useState<ReportSummary[]>([])
  const [total, setTotal] = useState(0)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasLoaded, setHasLoaded] = useState(false)
  // Bumped to re-trigger the fetch effect when the user retries from an empty state.
  const [reloadToken, setReloadToken] = useState(0)
  // Transient notice shown when the compare-selection cap is reached.
  const [capNotice, setCapNotice] = useState(false)
  const capTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  // Debounce search
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const searchTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    searchTimer.current = setTimeout(() => setDebouncedSearch(filters.search), 300)
    return () => clearTimeout(searchTimer.current)
  }, [filters.search])

  // Sync root_path from URL on mount (e.g. when navigating back from detail page)
  useEffect(() => {
    const urlRoot = searchParams.get('root_path')
    if (urlRoot) setRootPath(urlRoot)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // A new global scan (or root change) resets filters/pagination/compare.
  useEffect(() => {
    const reset = () => {
      setPage(1)
      setFilters(defaultFilters)
      clearCompareSelection()
    }
    reset()
  }, [rootPath, scanToken]) // eslint-disable-line react-hooks/exhaustive-deps

  // Reset to the first page whenever the user changes a filter.
  const handleFiltersChange = useCallback((next: ReportFilters) => {
    setFilters(next)
    setPage(1)
  }, [])

  // Fetch reports on root/scan/filter/page change. When any dependency changes
  // the previous in-flight request is aborted; its late/aborted
  // response is dropped so only the newest request updates the UI.
  useEffect(() => {
    if (!rootPath) return
    const controller = new AbortController()
    const load = async () => {
      setLoading(true)
      setError(null)
      try {
        const res: ListReportsResponse = await reportsApi.listReports({
          rootPath,
          search: debouncedSearch || undefined,
          models: filters.models.length ? filters.models : undefined,
          datasets: filters.datasets.length ? filters.datasets : undefined,
          scoreMin: filters.scoreMin > 0 ? filters.scoreMin : undefined,
          scoreMax: filters.scoreMax < 1 ? filters.scoreMax : undefined,
          sortBy: filters.sortBy,
          sortOrder: filters.sortOrder,
          page,
          pageSize: PAGE_SIZE,
          signal: controller.signal,
        })
        if (controller.signal.aborted) return
        setReports(res.reports)
        setTotal(res.total)
        setAvailableModels(res.filters.available_models)
        setAvailableDatasets(res.filters.available_datasets)
      } catch (err) {
        // A superseded request aborts; drop its outcome without surfacing an error.
        if (controller.signal.aborted || (isDomainError(err) && err.kind === 'aborted')) return
        setError(err instanceof Error ? err.message : 'Failed to load reports')
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false)
          setHasLoaded(true)
        }
      }
    }
    load()
    return () => controller.abort()
  }, [rootPath, scanToken, debouncedSearch, filters.models, filters.datasets, filters.scoreMin, filters.scoreMax, filters.sortBy, filters.sortOrder, page, reloadToken])

  // ---- Selection helpers ----
  const currentPageNames = useMemo(() => reports.map((r) => r.name), [reports])
  const allSelected = currentPageNames.length > 0 && currentPageNames.every((n) => selectedForCompare.includes(n))

  // Selection is stored by run name in context, so it is naturally independent
  // of the current sort/filter order. Reconcile it against the freshly ordered
  // list so the tray follows the on-screen order while never dropping a run
  // that was filtered off the current page.
  const orderedSelection = useMemo(
    () => preserveSelectionAcrossReorder(selectedForCompare, currentPageNames),
    [selectedForCompare, currentPageNames],
  )

  // Surface the selection-cap notice briefly, then let it fade.
  const flagCapReached = useCallback(() => {
    setCapNotice(true)
    clearTimeout(capTimer.current)
    capTimer.current = setTimeout(() => setCapNotice(false), 3000)
  }, [])

  // Cap-aware toggle: removing is always allowed; adding is rejected once the
  // selection is at the domain cap.
  const handleToggleSelect = useCallback(
    (name: string) => {
      if (selectedForCompare.includes(name)) {
        setCompareSelection(selectedForCompare.filter((n) => n !== name))
        return
      }
      const { next, rejected } = addToSelection(selectedForCompare, name)
      if (rejected) {
        flagCapReached()
        return
      }
      setCompareSelection(next)
    },
    [selectedForCompare, setCompareSelection, flagCapReached],
  )

  const handleSelectAll = useCallback(() => {
    if (allSelected) {
      setCompareSelection(selectedForCompare.filter((n) => !currentPageNames.includes(n)))
      return
    }
    // Add current-page runs one at a time so the cap is enforced.
    let nextSel = selectedForCompare
    let hitCap = false
    for (const name of currentPageNames) {
      const { next, rejected } = addToSelection(nextSel, name)
      if (rejected) {
        hitCap = true
        break
      }
      nextSel = next
    }
    if (hitCap) flagCapReached()
    setCompareSelection(nextSel)
  }, [allSelected, selectedForCompare, currentPageNames, setCompareSelection, flagCapReached])

  useEffect(() => () => clearTimeout(capTimer.current), [])

  const handleCardClick = useCallback(
    (name: string) => {
      navigate(`/reports/${encodeURIComponent(name)}?root_path=${encodeURIComponent(rootPath)}`)
    },
    [navigate, rootPath],
  )

  const handleCompare = useCallback(() => {
    if (selectedForCompare.length >= 2) {
      navigate(`/compare?reports=${selectedForCompare.slice(0, 3).join(';')}&root_path=${encodeURIComponent(rootPath)}`)
    }
  }, [selectedForCompare, navigate, rootPath])

  const handleViewHtml = useCallback(() => {
    if (selectedForCompare.length === 1) {
      const url = reportsApi.getHtmlReportUrl(rootPath, selectedForCompare[0])
      window.open(url, '_blank')
    }
  }, [selectedForCompare, rootPath])

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  // Distinguish the three empty-state reasons: a load failure, an
  // active-filter miss, or a genuinely empty directory.
  const hasActiveFilters = useMemo(
    () =>
      filters.search.trim() !== '' ||
      filters.models.length > 0 ||
      filters.datasets.length > 0 ||
      filters.scoreMin > 0 ||
      filters.scoreMax < 1,
    [filters],
  )
  const emptyReason: EmptyReason = error ? 'load-error' : hasActiveFilters ? 'no-match' : 'no-data'

  // In-view recovery for retry / clear-filters (routed via sentinel targets);
  // other actions fall through to real navigation.
  const handleEmptyAction = useCallback((action: ResolvedEmptyStateAction) => {
    if (action.navigateTo === '#retry') {
      setReloadToken((n) => n + 1)
      return true
    }
    if (action.navigateTo === '#clear-filters') {
      setFilters(defaultFilters)
      setPage(1)
      return true
    }
    return false
  }, [])

  return (
    <div className="page-enter mx-auto flex w-full max-w-7xl flex-col gap-5">
      {/* Filters */}
      <ReportFiltersBar
        filters={filters}
        availableModels={availableModels}
        availableDatasets={availableDatasets}
        onChange={handleFiltersChange}
      />

      {/* Error */}
      {error && (
        <ErrorAlert>{error}</ErrorAlert>
      )}

      {/* Content */}
      {loading ? (
        <div className="flex flex-col gap-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} height={64} className="rounded-[var(--radius)]" />
          ))}
        </div>
      ) : reports.length === 0 ? (
        <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
          <EmptyStateSystem
            reason={emptyReason}
            context={{
              view: 'evaluations',
              retryTo: '#retry',
              clearFiltersTo: '#clear-filters',
            }}
            hint={emptyReason === 'no-data' && hasLoaded ? t('reports.scanFirst') : undefined}
            onAction={handleEmptyAction}
          />
        </div>
      ) : (
        <>
          {/* Desktop (>=1024px): tabular view with fixed, ordered columns. */}
          <div className="hidden lg:block">
            <ReportsTable
              reports={reports}
              selected={selectedForCompare}
              allSelected={allSelected}
              onToggleSelectAll={handleSelectAll}
              onToggleSelect={handleToggleSelect}
              onRowClick={handleCardClick}
            />
          </div>

          {/* Narrow (<1024px): card view with fields consistent with the table. */}
          <div className="flex flex-col gap-2 lg:hidden">
            <SelectionCheckbox
              checked={allSelected}
              label={t('reports.selectAll')}
              onClick={handleSelectAll}
              className="w-fit text-sm text-[var(--text-muted)] transition-colors hover:text-[var(--text)]"
            >
              {t('reports.selectAll')}
            </SelectionCheckbox>
            {reports.map((report) => (
              <ReportCard
                key={report.name}
                report={report}
                selected={selectedForCompare.includes(report.name)}
                onSelect={handleToggleSelect}
                onClick={handleCardClick}
              />
            ))}
          </div>
        </>
      )}

      {/* Pagination */}
      <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />

      <SelectionTray
        count={orderedSelection.length}
        capNotice={capNotice}
        canViewHtml={orderedSelection.length === 1}
        onViewHtml={handleViewHtml}
        onCompare={handleCompare}
        onClear={clearCompareSelection}
      />
    </div>
  )
}
