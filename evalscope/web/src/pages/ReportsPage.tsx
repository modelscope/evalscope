import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import { datasetLabel } from '@/domain/report/primaryMetrics'
import { formatReportRef, parseReportRef, reportRefFromSummary } from '@/domain/report/reportRef'
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
import ConfirmDialog from '@/components/ui/ConfirmDialog'
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
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [deleting, setDeleting] = useState(false)

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
  const currentPageNames = useMemo(
    () => reports.map((r) => formatReportRef(reportRefFromSummary(r))),
    [reports],
  )
  const allSelected = currentPageNames.length > 0 && currentPageNames.every((n) => selectedForCompare.includes(n))

  // Selection is stored by run name in context, so it is naturally independent
  // of the current sort/filter order. Reconcile it against the freshly ordered
  // list so the tray follows the on-screen order while never dropping a run
  // that was filtered off the current page.
  const orderedSelection = useMemo(
    () => preserveSelectionAcrossReorder(selectedForCompare, currentPageNames),
    [selectedForCompare, currentPageNames],
  )

  // Selection is unbounded: score comparison uses the full set, while the
  // prediction tab derives its own three-report subset.
  const handleToggleSelect = useCallback(
    (name: string) => {
      if (selectedForCompare.includes(name)) {
        setCompareSelection(selectedForCompare.filter((n) => n !== name))
        return
      }
      setCompareSelection(addToSelection(selectedForCompare, name))
    },
    [selectedForCompare, setCompareSelection],
  )

  const handleSelectAll = useCallback(() => {
    if (allSelected) {
      setCompareSelection(selectedForCompare.filter((n) => !currentPageNames.includes(n)))
      return
    }
    setCompareSelection(currentPageNames.reduce(addToSelection, selectedForCompare))
  }, [allSelected, selectedForCompare, currentPageNames, setCompareSelection])

  const handleCardClick = useCallback(
    (ref: string) => {
      const { runId, modelId } = parseReportRef(ref)
      navigate(
        `/reports/${encodeURIComponent(runId)}/${encodeURIComponent(modelId)}?root_path=${encodeURIComponent(rootPath)}`,
      )
    },
    [navigate, rootPath],
  )

  const handleCompare = useCallback(() => {
    if (selectedForCompare.length >= 2) {
      const params = new URLSearchParams({ root_path: rootPath })
      for (const ref of selectedForCompare) params.append('report', ref)
      navigate(`/compare?${params.toString()}`)
    }
  }, [selectedForCompare, navigate, rootPath])

  const handleViewHtml = useCallback(() => {
    if (selectedForCompare.length === 1) {
      const url = reportsApi.getHtmlReportUrl(rootPath, selectedForCompare[0])
      window.open(url, '_blank')
    }
  }, [selectedForCompare, rootPath])

  const requestDeleteSelected = useCallback(() => {
    if (selectedForCompare.length === 0 || deleting) return
    setConfirmOpen(true)
  }, [selectedForCompare, deleting])

  const confirmDeleteSelected = useCallback(async () => {
    if (deleting || selectedForCompare.length === 0) return
    setDeleting(true)
    setError(null)
    const deleted: string[] = []
    try {
      for (const name of selectedForCompare) {
        await reportsApi.deleteReport(rootPath, name)
        deleted.push(name)
      }
      clearCompareSelection()
    } catch (err) {
      setCompareSelection(selectedForCompare.filter((n) => !deleted.includes(n)))
      setError(t('reports.deleteFailed', { msg: err instanceof Error ? err.message : String(err) }))
    } finally {
      setDeleting(false)
      setConfirmOpen(false)
      setReloadToken((n) => n + 1)
    }
  }, [deleting, selectedForCompare, rootPath, clearCompareSelection, setCompareSelection, t])

  const pendingDeleteItems = useMemo(
    () =>
      selectedForCompare.map((ref) => {
        const report = reports.find((r) => formatReportRef(reportRefFromSummary(r)) === ref)
        if (!report) return ref
        return `${report.model_name} · ${datasetLabel(report)}`
      }),
    [selectedForCompare, reports],
  )

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
            {reports.map((report) => {
              const ref = formatReportRef(reportRefFromSummary(report))
              return (
                <ReportCard
                  key={ref}
                  report={report}
                  selected={selectedForCompare.includes(ref)}
                  onSelect={handleToggleSelect}
                  onClick={handleCardClick}
                />
              )
            })}
          </div>
        </>
      )}

      {/* Pagination */}
      <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />

      <SelectionTray
        count={orderedSelection.length}
        canViewHtml={orderedSelection.length === 1}
        onViewHtml={handleViewHtml}
        onCompare={handleCompare}
        onClear={clearCompareSelection}
        onDelete={requestDeleteSelected}
        deleting={deleting}
      />

      <ConfirmDialog
        open={confirmOpen}
        danger
        busy={deleting}
        title={t('reports.deleteConfirmTitle')}
        message={t('reports.deleteConfirm', { n: selectedForCompare.length })}
        items={pendingDeleteItems}
        confirmLabel={t('reports.delete')}
        cancelLabel={t('common.cancel')}
        onConfirm={confirmDeleteSelected}
        onCancel={() => setConfirmOpen(false)}
      />
    </div>
  )
}
