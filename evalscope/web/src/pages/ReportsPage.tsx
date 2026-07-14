import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { Eye, GitCompareArrows, Inbox } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import * as reportsApi from '@/api/reports'
import type { ListReportsResponse, ReportSummary } from '@/api/types'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Button from '@/components/ui/Button'
import Skeleton from '@/components/ui/Skeleton'
import EmptyState from '@/components/common/EmptyState'
import ReportFiltersBar, { type ReportFilters } from '@/components/reports/ReportFilters'
import ReportCard from '@/components/reports/ReportCard'

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
    toggleSelectForCompare,
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

  // Fetch reports on root/scan/filter/page change.
  useEffect(() => {
    if (!rootPath) return
    let cancelled = false
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
        })
        if (cancelled) return
        setReports(res.reports)
        setTotal(res.total)
        setAvailableModels(res.filters.available_models)
        setAvailableDatasets(res.filters.available_datasets)
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load reports')
          setReports([])
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
          setHasLoaded(true)
        }
      }
    }
    load()
    return () => { cancelled = true }
  }, [rootPath, scanToken, debouncedSearch, filters.models, filters.datasets, filters.scoreMin, filters.scoreMax, filters.sortBy, filters.sortOrder, page])

  // ---- Selection helpers ----
  const currentPageNames = useMemo(() => reports.map((r) => r.name), [reports])
  const allSelected = currentPageNames.length > 0 && currentPageNames.every((n) => selectedForCompare.includes(n))

  const handleSelectAll = useCallback(() => {
    if (allSelected) {
      setCompareSelection(selectedForCompare.filter((n) => !currentPageNames.includes(n)))
    } else {
      const merged = new Set([...selectedForCompare, ...currentPageNames])
      setCompareSelection(Array.from(merged))
    }
  }, [allSelected, selectedForCompare, currentPageNames, setCompareSelection])

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

  return (
    <div className="page-enter flex flex-col gap-5">
      {/* Breadcrumb */}
      <Breadcrumb items={[{ label: t('nav.evaluations') }]} />

      {/* Filters */}
      <ReportFiltersBar
        filters={filters}
        availableModels={availableModels}
        availableDatasets={availableDatasets}
        onChange={handleFiltersChange}
      />

      {/* Action bar */}
      {reports.length > 0 && (
        <div className="flex items-center gap-3 flex-wrap">
          <button
            type="button"
            onClick={handleSelectAll}
            className="flex items-center gap-2 text-sm text-[var(--text-muted)] cursor-pointer hover:text-[var(--text)] transition-colors"
          >
            <span
              role="checkbox"
              aria-checked={allSelected}
              className="w-4.5 h-4.5 rounded-[var(--radius-xs)] border-2 flex items-center justify-center transition-all duration-150 shrink-0"
              style={{
                borderColor: allSelected ? 'var(--accent)' : 'var(--border-strong)',
                background: allSelected ? 'var(--accent)' : 'transparent',
              }}
            >
              {allSelected && (
                <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="2,6 5,9 10,3" />
                </svg>
              )}
            </span>
            {t('reports.selectAll')}
          </button>

          {selectedForCompare.length > 0 && (
            <span className="text-xs text-[var(--text-muted)]">
              {selectedForCompare.length} {t('reports.selected')}
              {selectedForCompare.length > 3 && (
                <span className="ml-1 text-[var(--warning-color)]">{t('compare.maxThreeSelected')}</span>
              )}
            </span>
          )}

          <div className="flex items-center gap-2 ml-auto">
            <Button variant="outline" size="sm" disabled={selectedForCompare.length < 2} onClick={handleCompare}>
              <GitCompareArrows size={14} />
              {t('reports.compare')}
            </Button>
            <Button variant="outline" size="sm" disabled={selectedForCompare.length !== 1} onClick={handleViewHtml}>
              <Eye size={14} />
              {t('reports.viewHtml')}
            </Button>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="px-4 py-3 rounded-[var(--radius)] bg-[var(--danger-bg)] border border-[var(--danger-border)] text-sm text-[var(--danger)]">
          {error}
        </div>
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
          <EmptyState
            icon={<Inbox size={28} strokeWidth={1.5} />}
            title={t('reports.noReports')}
            hint={hasLoaded ? t('reports.scanFirst') : ''}
          />
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {reports.map((report) => (
            <ReportCard
              key={report.name}
              report={report}
              selected={selectedForCompare.includes(report.name)}
              onSelect={toggleSelectForCompare}
              onClick={handleCardClick}
            />
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 pt-2">
          <Button variant="ghost" size="sm" disabled={page <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>
            ←
          </Button>
          {Array.from({ length: totalPages }, (_, i) => i + 1)
            .filter((p) => p === 1 || p === totalPages || Math.abs(p - page) <= 2)
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
                  variant={item === page ? 'primary' : 'ghost'}
                  size="sm"
                  onClick={() => setPage(item as number)}
                  className="!min-w-[32px]"
                >
                  {item}
                </Button>
              ),
            )}
          <Button variant="ghost" size="sm" disabled={page >= totalPages} onClick={() => setPage((p) => Math.min(totalPages, p + 1))}>
            →
          </Button>
        </div>
      )}
    </div>
  )
}
