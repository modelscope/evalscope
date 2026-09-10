import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import { datasetLabel } from '@/domain/report/primaryMetrics'
import { formatReportRef, parseReportRef, reportRefFromSummary } from '@/domain/report/reportRef'
import { useCompareSelection, useScan } from '@/contexts/ReportsContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { useBatchDelete } from '@/hooks/useBatchDelete'
import * as reportsApi from '@/api/reports'
import type { ListReportsGroupedResponse, ListReportsResponse, ReportGroup, ReportSummary } from '@/api/types'
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
import ReportGroupList from '@/components/reports/ReportGroupList'
import SelectionTray from '@/components/reports/SelectionTray'
import ConfirmDialog from '@/components/ui/ConfirmDialog'
import {
  addToSelection,
  preserveSelectionAcrossReorder,
} from '@/domain/compare/selection'

const PAGE_SIZE = 20

/** Stable placeholders so an unresolved listing keeps a single identity. */
const EMPTY_REPORTS: ReportSummary[] = []
const EMPTY_GROUPS: ReportGroup[] = []
const EMPTY_FACETS: string[] = []

const defaultFilters: ReportFilters = {
  search: '',
  models: [],
  datasets: [],
  sortBy: 'time',
  sortOrder: 'desc',
  groupByModel: false,
}

/**
 * `listReports`/`listReportsGrouped` are two distinct endpoints with two
 * distinct response shapes (see `api/reports.ts`) so that every existing
 * flat-list caller is unaffected by grouping. This page is the one caller
 * that needs both, so it tags whichever one it fetched with a `grouped`
 * discriminant locally, rather than pushing that union into the shared API
 * types.
 */
type ReportsListing =
  | ({ grouped: false } & ListReportsResponse)
  | ({ grouped: true } & ListReportsGroupedResponse)

export default function ReportsPage() {
  const { t } = useLocale()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()

  const { rootPath, scanToken, setRootPath } = useScan()
  const { selectedForCompare, setCompareSelection, clearCompareSelection } = useCompareSelection()

  // ---- Local state ----
  const [filters, setFilters] = useState<ReportFilters>(defaultFilters)
  const [page, setPage] = useState(1)

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
  // the previous in-flight request is aborted; its late/aborted response is
  // dropped so only the newest request updates the UI.
  const listing = useAsyncResource<ReportsListing>(
    async (signal) => {
      const params = {
        rootPath,
        search: debouncedSearch || undefined,
        models: filters.models.length ? filters.models : undefined,
        datasets: filters.datasets.length ? filters.datasets : undefined,
        sortBy: filters.sortBy,
        sortOrder: filters.sortOrder,
        page,
        pageSize: PAGE_SIZE,
        signal,
      }
      if (filters.groupByModel) {
        return { grouped: true, ...(await reportsApi.listReportsGrouped(params)) }
      }
      return { grouped: false, ...(await reportsApi.listReports(params)) }
    },
    [
      rootPath,
      scanToken,
      debouncedSearch,
      filters.models,
      filters.datasets,
      filters.sortBy,
      filters.sortOrder,
      filters.groupByModel,
      page,
    ],
    { enabled: Boolean(rootPath), fallbackMessage: t('common.loadError') },
  )

  const listingData = listing.data
  const grouped = listingData?.grouped ?? false
  const reports: ReportSummary[] = listingData && !listingData.grouped ? listingData.reports : EMPTY_REPORTS
  const groups: ReportGroup[] = listingData && listingData.grouped ? listingData.reports : EMPTY_GROUPS
  const total = listingData?.total ?? 0
  const availableModels = listingData?.filters.available_models ?? EMPTY_FACETS
  const availableDatasets = listingData?.filters.available_datasets ?? EMPTY_FACETS
  const loading = listing.loading
  const hasLoaded = listing.data !== undefined || Boolean(listing.error)
  const isEmpty = grouped ? groups.length === 0 : reports.length === 0

  // Which model rows are expanded in grouped view. Reset on every reload so
  // a new filter/sort/page doesn't carry stale expand state.
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set())
  const handleToggleExpand = useCallback((modelName: string) => {
    setExpandedModels((prev) => {
      const next = new Set(prev)
      if (next.has(modelName)) next.delete(modelName)
      else next.add(modelName)
      return next
    })
  }, [])

  // ---- Selection helpers ----
  // In grouped view, "this page" means every child report under every
  // model row shown - selection/compare/delete act on the real reports
  // underneath, whether or not their group is currently expanded.
  const currentPageNames = useMemo(
    () =>
      grouped
        ? groups.flatMap((g) => g.refs)
        : reports.map((r) => formatReportRef(reportRefFromSummary(r))),
    [grouped, groups, reports],
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

  // Select/deselect every report under one grouped model row at once.
  const handleSelectGroup = useCallback(
    (refs: string[], select: boolean) => {
      if (select) {
        setCompareSelection(refs.reduce(addToSelection, selectedForCompare))
        return
      }
      setCompareSelection(selectedForCompare.filter((n) => !refs.includes(n)))
    },
    [selectedForCompare, setCompareSelection],
  )

  const handleCardClick = useCallback(
    (ref: string) => {
      const { runId, modelId } = parseReportRef(ref)
      navigate(
        `/reports/${encodeURIComponent(runId)}/${encodeURIComponent(modelId)}?root_path=${encodeURIComponent(rootPath)}`,
      )
    },
    [navigate, rootPath],
  )

  const handleCompareRefs = useCallback(
    (refs: string[]) => {
      if (refs.length < 2) return
      const params = new URLSearchParams({ root_path: rootPath })
      for (const ref of refs) params.append('report', ref)
      navigate(`/compare?${params.toString()}`)
    },
    [navigate, rootPath],
  )

  const handleCompare = useCallback(
    () => handleCompareRefs(selectedForCompare),
    [handleCompareRefs, selectedForCompare],
  )

  const handleViewHtml = useCallback(() => {
    if (selectedForCompare.length === 1) {
      const url = reportsApi.getHtmlReportUrl(rootPath, selectedForCompare[0])
      window.open(url, '_blank')
    }
  }, [selectedForCompare, rootPath])

  const reloadListing = listing.reload
  const deletion = useBatchDelete<string>({
    items: selectedForCompare,
    deleteItem: (ref) => reportsApi.deleteReport(rootPath, ref),
    onSettled: setCompareSelection,
    reload: reloadListing,
    formatError: (msg) => t('reports.deleteFailed', { msg }),
  })

  // Deleting reports its own failure; a load failure comes from the resource.
  const error = deletion.error ?? (listing.error || null)

  // Selection is stored by ref, but the current page's reports live either flat
  // or nested under groups - flatten once so a pending delete can always show
  // a real report's label regardless of view mode or expand state.
  const allPageReports = useMemo(
    () => (grouped ? groups.flatMap((g) => g.children) : reports),
    [grouped, groups, reports],
  )

  const pendingDeleteItems = useMemo(
    () =>
      selectedForCompare.map((ref) => {
        const report = allPageReports.find((r) => formatReportRef(reportRefFromSummary(r)) === ref)
        if (!report) return ref
        return `${report.model_name} · ${datasetLabel(report)}`
      }),
    [selectedForCompare, allPageReports],
  )

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  // Distinguish the three empty-state reasons: a load failure, an
  // active-filter miss, or a genuinely empty directory.
  const hasActiveFilters = useMemo(
    () =>
      filters.search.trim() !== '' ||
      filters.models.length > 0 ||
      filters.datasets.length > 0,
    [filters],
  )
  const emptyReason: EmptyReason = error ? 'load-error' : hasActiveFilters ? 'no-match' : 'no-data'

  // In-view recovery for retry / clear-filters (routed via sentinel targets);
  // other actions fall through to real navigation.
  const handleEmptyAction = useCallback((action: ResolvedEmptyStateAction) => {
    if (action.navigateTo === '#retry') {
      reloadListing()
      return true
    }
    if (action.navigateTo === '#clear-filters') {
      setFilters(defaultFilters)
      setPage(1)
      return true
    }
    return false
  }, [reloadListing])

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
      ) : isEmpty ? (
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
      ) : grouped ? (
        <>
          {/* Desktop (>=1024px): each model row expands into the same table used flat. */}
          <div className="hidden lg:block">
            <ReportGroupList
              groups={groups}
              expandedModels={expandedModels}
              onToggleExpand={handleToggleExpand}
              selected={selectedForCompare}
              onToggleSelect={handleToggleSelect}
              onSelectGroup={handleSelectGroup}
              onRowClick={handleCardClick}
              onCompareGroup={handleCompareRefs}
              variant="table"
            />
          </div>

          {/* Narrow (<1024px): each model row expands into the same cards used flat. */}
          <div className="lg:hidden">
            <ReportGroupList
              groups={groups}
              expandedModels={expandedModels}
              onToggleExpand={handleToggleExpand}
              selected={selectedForCompare}
              onToggleSelect={handleToggleSelect}
              onSelectGroup={handleSelectGroup}
              onRowClick={handleCardClick}
              onCompareGroup={handleCompareRefs}
              variant="cards"
            />
          </div>
        </>
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
        onDelete={deletion.request}
        deleting={deletion.deleting}
      />

      <ConfirmDialog
        open={deletion.confirmOpen}
        danger
        busy={deletion.deleting}
        title={t('reports.deleteConfirmTitle')}
        message={t('reports.deleteConfirm', { n: selectedForCompare.length })}
        items={pendingDeleteItems}
        confirmLabel={t('reports.delete')}
        cancelLabel={t('common.cancel')}
        onConfirm={deletion.confirm}
        onCancel={deletion.cancel}
      />
    </div>
  )
}
