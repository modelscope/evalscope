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
import type { ConfigResponse, LoadReportResponse, ReportData } from '@/api/types'
import * as reportsApi from '@/api/reports'
import { apiValidated } from '@/api/client'

/**
 * Application and report state are split into contexts with unrelated update
 * times. `ReportsProvider` composes them so callers still mount a single
 * provider while consumers subscribe only to the state they need.
 */

const INITIAL_ROOT = './outputs' // fallback; will be overridden by /api/v1/config
const REPORT_CACHE_LIMIT = 32 // bound the in-memory cache so long sessions don't grow unbounded

// ------------------------------------------------------------------ //
// Application configuration: backend metadata shared across the shell   //
// ------------------------------------------------------------------ //

interface AppConfigCtx {
  config: ConfigResponse | null
}

const AppConfigContext = createContext<AppConfigCtx>({ config: null })

function AppConfigProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<ConfigResponse | null>(null)

  useEffect(() => {
    let cancelled = false
    apiValidated<ConfigResponse>('/api/v1/config')
      .then((response) => {
        if (!cancelled) setConfig(response)
      })
      .catch(() => {/* retain the empty config when the service is unavailable */})
    return () => { cancelled = true }
  }, [])

  const value = useMemo(() => ({ config }), [config])
  return <AppConfigContext.Provider value={value}>{children}</AppConfigContext.Provider>
}

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
  const { config } = useAppConfig()
  const [rootPath, setRootPathState] = useState(INITIAL_ROOT)
  const [scanToken, setScanToken] = useState(0)

  // Mirror the latest root into a ref so the mount effect and triggerScan can
  // read a fresh value without joining a dependency array.
  const rootRef = useRef(rootPath)
  useEffect(() => { rootRef.current = rootPath }, [rootPath])

  // Apply the server-side default unless the user has already changed the root.
  useEffect(() => {
    if (config?.outputs_root && rootRef.current === INITIAL_ROOT) {
      setRootPathState(config.outputs_root)
    }
  }, [config?.outputs_root])

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
    <AppConfigProvider>
      <ScanProvider>
        <CompareSelectionProvider>
          <ReportCacheProvider>{children}</ReportCacheProvider>
        </CompareSelectionProvider>
      </ScanProvider>
    </AppConfigProvider>
  )
}

/* eslint-disable react-refresh/only-export-components */

/** Backend configuration shared by the application shell and scan controls. */
export function useAppConfig(): AppConfigCtx {
  return useContext(AppConfigContext)
}

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
