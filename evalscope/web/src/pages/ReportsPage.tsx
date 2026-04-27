import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Download, Eye, FolderOpen, GitCompareArrows, Loader2, ScanSearch } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import * as reportsApi from '@/api/reports'
import type { ListReportsResponse, ReportSummary } from '@/api/types'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Button from '@/components/ui/Button'
import Skeleton from '@/components/ui/Skeleton'
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
  const {
    rootPath,
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
  const [hasScanned, setHasScanned] = useState(false)

  // Debounce search
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const searchTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  useEffect(() => {
    searchTimer.current = setTimeout(() => setDebouncedSearch(filters.search), 300)
    return () => clearTimeout(searchTimer.current)
  }, [filters.search])

  // Fetch reports when filters/page change
  const fetchReports = useCallback(async () => {
    if (!hasScanned) return
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
      setReports(res.reports)
      setTotal(res.total)
      setAvailableModels(res.filters.available_models)
      setAvailableDatasets(res.filters.available_datasets)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load reports')
    } finally {
      setLoading(false)
    }
  }, [rootPath, debouncedSearch, filters.models, filters.datasets, filters.scoreMin, filters.scoreMax, filters.sortBy, filters.sortOrder, page, hasScanned])

  useEffect(() => {
    fetchReports()
  }, [fetchReports])

  // Reset page on filter change
  useEffect(() => {
    setPage(1)
  }, [debouncedSearch, filters.models, filters.datasets, filters.scoreMin, filters.scoreMax, filters.sortBy, filters.sortOrder])

  const handleScan = useCallback(async () => {
    setLoading(true)
    setError(null)
    setHasScanned(true)
    try {
      const res = await reportsApi.listReports({
        rootPath,
        page: 1,
        pageSize: PAGE_SIZE,
        sortBy: filters.sortBy,
        sortOrder: filters.sortOrder,
      })
      setReports(res.reports)
      setTotal(res.total)
      setAvailableModels(res.filters.available_models)
      setAvailableDatasets(res.filters.available_datasets)
      setPage(1)
      setFilters(defaultFilters)
      clearCompareSelection()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Scan failed')
    } finally {
      setLoading(false)
    }
  }, [rootPath, filters.sortBy, filters.sortOrder, clearCompareSelection])

  // Auto-scan on mount if rootPath is available
  const hasAutoScanned = useRef(false)
  useEffect(() => {
    if (rootPath && !hasScanned && !hasAutoScanned.current) {
      hasAutoScanned.current = true
      handleScan()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ---- Selection helpers ----
  const currentPageNames = useMemo(() => reports.map((r) => r.name), [reports])

  const allSelected = currentPageNames.length > 0 && currentPageNames.every((n) => selectedForCompare.includes(n))

  const handleSelectAll = useCallback(() => {
    if (allSelected) {
      // Deselect current page
      setCompareSelection(selectedForCompare.filter((n) => !currentPageNames.includes(n)))
    } else {
      // Select current page (merge with existing)
      const merged = new Set([...selectedForCompare, ...currentPageNames])
      setCompareSelection(Array.from(merged))
    }
  }, [allSelected, selectedForCompare, currentPageNames, setCompareSelection])

  const handleCardClick = useCallback(
    (name: string) => {
      // Navigate to detail — use the report load route or a detail page
      navigate(`/reports/${encodeURIComponent(name)}?root_path=${encodeURIComponent(rootPath)}`)
    },
    [navigate, rootPath],
  )

  const handleCompare = useCallback(() => {
    if (selectedForCompare.length >= 2) {
      navigate(`/compare?reports=${selectedForCompare.join(';')}&root_path=${encodeURIComponent(rootPath)}`)
    }
  }, [selectedForCompare, navigate, rootPath])

  const handleExport = useCallback(() => {
    const data = reports.filter((r) => selectedForCompare.includes(r.name))
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'reports-export.json'
    a.click()
    URL.revokeObjectURL(url)
  }, [reports, selectedForCompare])

  const handleViewHtml = useCallback(() => {
    if (selectedForCompare.length === 1) {
      const url = reportsApi.getHtmlReportUrl(rootPath, selectedForCompare[0])
      window.open(url, '_blank')
    }
  }, [selectedForCompare, rootPath])

  // Pagination
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <div className="page-enter flex flex-col gap-5">
      {/* Breadcrumb */}
      <Breadcrumb items={[{ label: t('reports.title') }]} />

      {/* Path bar */}
      <div className="flex items-center gap-2">
        <FolderOpen size={16} className="text-[var(--text-muted)] shrink-0" />
        <input
          type="text"
          value={rootPath}
          onChange={(e) => setRootPath(e.target.value)}
          className="flex-1 px-3 py-2 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:border-[var(--accent)] transition-all duration-[var(--transition)]"
          placeholder={t('reports.pathLabel')}
        />
        <Button onClick={handleScan} disabled={loading} size="md">
          {loading ? (
            <>
              <Loader2 size={14} className="animate-spin" />
              {t('reports.scanning')}
            </>
          ) : (
            <>
              <ScanSearch size={14} />
              {t('reports.scan')}
            </>
          )}
        </Button>
      </div>

      {/* Filters */}
      {hasScanned && (
        <ReportFiltersBar
          filters={filters}
          availableModels={availableModels}
          availableDatasets={availableDatasets}
          onChange={setFilters}
        />
      )}

      {/* Action bar */}
      {hasScanned && reports.length > 0 && (
        <div className="flex items-center gap-3 flex-wrap">
          <label className="flex items-center gap-2 text-sm text-[var(--text-muted)] cursor-pointer">
            <input
              type="checkbox"
              checked={allSelected}
              onChange={handleSelectAll}
              className="accent-[var(--accent)] w-4 h-4"
            />
            {t('reports.selectAll')}
          </label>
          {selectedForCompare.length > 0 && (
            <span className="text-xs text-[var(--text-muted)]">
              {selectedForCompare.length} {t('reports.selected')}
            </span>
          )}
          <div className="flex items-center gap-2 ml-auto">
            <Button
              variant="outline"
              size="sm"
              disabled={selectedForCompare.length < 2}
              onClick={handleCompare}
            >
              <GitCompareArrows size={14} />
              {t('reports.compare')}
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={selectedForCompare.length === 0}
              onClick={handleExport}
            >
              <Download size={14} />
              {t('reports.export')}
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={selectedForCompare.length !== 1}
              onClick={handleViewHtml}
            >
              <Eye size={14} />
              {t('reports.viewHtml')}
            </Button>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="px-4 py-3 rounded-[var(--radius)] bg-[rgba(239,68,68,0.08)] border border-[rgba(239,68,68,0.2)] text-sm text-[#ef4444]">
          {error}
        </div>
      )}

      {/* Content */}
      {loading && !hasScanned ? null : loading ? (
        <div className="flex flex-col gap-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} height={64} className="rounded-[var(--radius)]" />
          ))}
        </div>
      ) : !hasScanned ? (
        <EmptyState icon={<ScanSearch size={40} />} title={t('reports.noReports')} subtitle={t('reports.scanFirst')} />
      ) : reports.length === 0 ? (
        <EmptyState icon={<ScanSearch size={40} />} title={t('reports.noReports')} subtitle={t('reports.scanFirst')} />
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
      {hasScanned && totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 pt-2">
          <Button
            variant="ghost"
            size="sm"
            disabled={page <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
          >
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
          <Button
            variant="ghost"
            size="sm"
            disabled={page >= totalPages}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
          >
            →
          </Button>
        </div>
      )}
    </div>
  )
}

// ---- Internal EmptyState component ----
function EmptyState({
  icon,
  title,
  subtitle,
}: {
  icon: React.ReactNode
  title: string
  subtitle: string
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-3">
      <div className="text-[var(--text-dim)]">{icon}</div>
      <h3 className="text-lg font-semibold text-[var(--text)]">{title}</h3>
      <p className="text-sm text-[var(--text-muted)]">{subtitle}</p>
    </div>
  )
}
