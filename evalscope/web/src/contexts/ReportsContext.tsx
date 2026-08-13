import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import type { LoadReportResponse, ReportData } from '@/api/types'
import * as reportsApi from '@/api/reports'
import { apiValidated } from '@/api/client'
import { configResponseSchema } from '@/api/schemas'

/**
 * Report-scoped application state, split into three independent contexts.
 *
 * The three concerns below change at unrelated times, so they are published
 * separately: a compare-selection toggle on the reports list must not re-render
 * the dashboard, and a report landing in the cache must not re-render the
 * scan-path bar. `ReportsProvider` composes all three so callers still mount a
 * single provider.
 */

const INITIAL_ROOT = './outputs' // fallback; will be overridden by /api/v1/config
const REPORT_CACHE_LIMIT = 32 // bound the in-memory cache so long sessions don't grow unbounded

// ------------------------------------------------------------------ //
// Scan scope: which directory is being read, and when to re-read it   //
// ------------------------------------------------------------------ //

interface ScanCtx {
  rootPath: string
  /** Monotonically-increasing token; bumped by triggerScan to fan out a rescan. */
  scanToken: number
  setRootPath: (path: string) => void
  triggerScan: (path?: string) => void
}

const ScanContext = createContext<ScanCtx>(null!)

// ------------------------------------------------------------------ //
// Compare selection: which runs are ticked on the reports list        //
// ------------------------------------------------------------------ //

interface CompareSelectionCtx {
  /** Reports selected for compare (and for batch deletion) across pages. */
  selectedForCompare: string[]
  setCompareSelection: (names: string[]) => void
  clearCompareSelection: () => void
}

const CompareSelectionContext = createContext<CompareSelectionCtx>(null!)

// ------------------------------------------------------------------ //
// Report cache: loaded report payloads, keyed by report reference     //
// ------------------------------------------------------------------ //

interface ReportCacheCtx {
  /** Keyed by report reference (`{runId}/{modelId}`). */
  reportCache: Record<string, LoadReportResponse>
  /** True while at least one load is in flight. */
  loading: boolean
  loadMultiReports: (names: string[], signal?: AbortSignal) => Promise<ReportData[]>
}

const ReportCacheContext = createContext<ReportCacheCtx>(null!)

/**
 * Evict the oldest entries once the cache exceeds its limit.
 *
 * The just-written key is never evicted, so a cache at its limit still admits a
 * new entry rather than dropping the value the caller is about to read.
 */
/** Cached report payloads together with the scan scope they were read under. */
interface CachedReports {
  scope: string
  entries: Record<string, LoadReportResponse>
}

const EMPTY_CACHE: CachedReports = { scope: '', entries: {} }

function withCacheLimit(
  cache: Record<string, LoadReportResponse>,
  justAdded: string,
): Record<string, LoadReportResponse> {
  const keys = Object.keys(cache)
  if (keys.length <= REPORT_CACHE_LIMIT) return cache
  const next = { ...cache }
  // Evict in insertion order until the limit holds, skipping the new entry.
  for (const key of keys) {
    if (Object.keys(next).length <= REPORT_CACHE_LIMIT) break
    if (key !== justAdded) delete next[key]
  }
  return next
}

function ScanProvider({ children }: { children: ReactNode }) {
  const [rootPath, setRootPathState] = useState(INITIAL_ROOT)
  const [scanToken, setScanToken] = useState(0)

  // Mirror the latest root into a ref so the mount effect and triggerScan can
  // read a fresh value without joining a dependency array.
  const rootRef = useRef(rootPath)
  useEffect(() => { rootRef.current = rootPath }, [rootPath])

  // Fetch the server-side default outputs_root from /api/v1/config on mount.
  // Only apply it when the user has not already changed the root away from the
  // initial default (checked at resolve time via the ref).
  useEffect(() => {
    let cancelled = false
    apiValidated('/api/v1/config', configResponseSchema)
      .then((cfg) => {
        if (!cancelled && cfg.outputs_root && rootRef.current === INITIAL_ROOT) {
          setRootPathState(cfg.outputs_root)
        }
      })
      .catch(() => {/* ignore; keep default */})
    return () => { cancelled = true }
  }, [])

  const setRootPath = useCallback((path: string) => setRootPathState(path), [])

  const triggerScan = useCallback((path?: string) => {
    setRootPathState(path ?? rootRef.current)
    setScanToken((token) => token + 1)
  }, [])

  const value = useMemo<ScanCtx>(
    () => ({ rootPath, scanToken, setRootPath, triggerScan }),
    [rootPath, scanToken, setRootPath, triggerScan],
  )

  return <ScanContext.Provider value={value}>{children}</ScanContext.Provider>
}

function CompareSelectionProvider({ children }: { children: ReactNode }) {
  const [selectedForCompare, setSelected] = useState<string[]>([])

  const setCompareSelection = useCallback((names: string[]) => setSelected(names), [])
  const clearCompareSelection = useCallback(() => setSelected([]), [])

  const value = useMemo<CompareSelectionCtx>(
    () => ({ selectedForCompare, setCompareSelection, clearCompareSelection }),
    [selectedForCompare, setCompareSelection, clearCompareSelection],
  )

  return <CompareSelectionContext.Provider value={value}>{children}</CompareSelectionContext.Provider>
}

function ReportCacheProvider({ children }: { children: ReactNode }) {
  const { rootPath, scanToken } = useScan()
  // The cache carries the scope it was filled under, so a rescan or a root change
  // invalidates it by comparison at read time rather than by clearing it later.
  const scope = `${rootPath}\0${scanToken}`
  const [cached, setCached] = useState<CachedReports>(EMPTY_CACHE)
  const reportCache = cached.scope === scope ? cached.entries : EMPTY_CACHE.entries
  // A counter rather than a flag: concurrent loads must not let the first one to
  // settle report the others as finished.
  const [inFlight, setInFlight] = useState(0)

  // Read the cache through a ref so `loadMultiReports` keeps a stable identity:
  // callers put it in effect dependency arrays, and a new identity per cache
  // write would re-fire those effects in a loop.
  const cacheRef = useRef(reportCache)
  useEffect(() => { cacheRef.current = reportCache }, [reportCache])
  const scopeRef = useRef({ scope, rootPath })
  useEffect(() => { scopeRef.current = { scope, rootPath } }, [scope, rootPath])

  const loadMultiReports = useCallback(async (names: string[], signal?: AbortSignal) => {
    setInFlight((n) => n + 1)
    try {
      // Load via a cache-aware path so repeat loads in compare view don't refetch.
      // Per-report tagging preserves source mapping when reports share model_name.
      const { scope: readScope, rootPath: root } = scopeRef.current
      const results = await Promise.all(
        names.map(async (name) => {
          const cached = cacheRef.current[name]
          if (cached) return cached
          const data = await reportsApi.loadReport(root, name, signal)
          setCached((prev) => {
            // A read that started before a rescan must not repopulate the new scope.
            if (scopeRef.current.scope !== readScope) return prev
            const base = prev.scope === readScope ? prev.entries : {}
            return { scope: readScope, entries: withCacheLimit({ ...base, [name]: data }, name) }
          })
          return data
        }),
      )
      return results.flatMap((res, i) =>
        res.report_list.map((r) => ({ ...r, _reportRef: names[i] })),
      )
    } finally {
      setInFlight((n) => n - 1)
    }
  }, [])

  const value = useMemo<ReportCacheCtx>(
    () => ({ reportCache, loading: inFlight > 0, loadMultiReports }),
    [reportCache, inFlight, loadMultiReports],
  )

  return <ReportCacheContext.Provider value={value}>{children}</ReportCacheContext.Provider>
}

export function ReportsProvider({ children }: { children: ReactNode }) {
  return (
    <ScanProvider>
      <CompareSelectionProvider>
        <ReportCacheProvider>{children}</ReportCacheProvider>
      </CompareSelectionProvider>
    </ScanProvider>
  )
}

/* eslint-disable react-refresh/only-export-components */

/** Which output directory is being read, and the token that fans out a rescan. */
export function useScan(): ScanCtx {
  return useContext(ScanContext)
}

/** Runs ticked for comparison / batch deletion on the reports list. */
export function useCompareSelection(): CompareSelectionCtx {
  return useContext(CompareSelectionContext)
}

/** Cache-aware multi-report loader, shared by the compare surfaces. */
export function useReportCache(): ReportCacheCtx {
  return useContext(ReportCacheContext)
}
