import { createContext, useCallback, useContext, useEffect, useMemo, useReducer, useRef, type ReactNode } from 'react'
import type { LoadReportResponse, ReportData } from '@/api/types'
import * as reportsApi from '@/api/reports'
import { api } from '@/api/client'

// ------------------------------------------------------------------ //
// State                                                               //
// ------------------------------------------------------------------ //

interface ReportsState {
  rootPath: string
  availableReports: string[]
  selectedReports: string[]
  /** Keyed by report_name */
  reportCache: Record<string, LoadReportResponse>
  /** Multi-report list cache (all selected) */
  multiReportList: ReportData[]
  /** Reports selected for compare across pages */
  selectedForCompare: string[]
  loading: boolean
}

type Action =
  | { type: 'SET_ROOT'; rootPath: string }
  | { type: 'SET_AVAILABLE'; reports: string[] }
  | { type: 'SET_SELECTED'; reports: string[] }
  | { type: 'CACHE_REPORT'; name: string; data: LoadReportResponse }
  | { type: 'SET_MULTI'; list: ReportData[] }
  | { type: 'SET_LOADING'; loading: boolean }
  | { type: 'CLEAR_CACHE' }
  | { type: 'TOGGLE_COMPARE'; name: string }
  | { type: 'SET_COMPARE'; names: string[] }
  | { type: 'CLEAR_COMPARE' }

const INITIAL_ROOT = './outputs' // fallback; will be overridden by /api/v1/config
const REPORT_CACHE_LIMIT = 32 // bound the in-memory cache so long sessions don't grow unbounded

const initialState: ReportsState = {
  rootPath: INITIAL_ROOT,
  availableReports: [],
  selectedReports: [],
  reportCache: {},
  multiReportList: [],
  selectedForCompare: [],
  loading: false,
}

function reducer(state: ReportsState, action: Action): ReportsState {
  switch (action.type) {
    case 'SET_ROOT':
      return { ...state, rootPath: action.rootPath }
    case 'SET_AVAILABLE':
      return { ...state, availableReports: action.reports }
    case 'SET_SELECTED':
      return { ...state, selectedReports: action.reports }
    case 'CACHE_REPORT': {
      const next = { ...state.reportCache, [action.name]: action.data }
      const keys = Object.keys(next)
      if (keys.length > REPORT_CACHE_LIMIT) {
        // Drop oldest insertion-order keys (skip the entry we just added).
        const drop = keys.length - REPORT_CACHE_LIMIT
        for (let i = 0; i < drop; i++) {
          if (keys[i] !== action.name) delete next[keys[i]]
        }
      }
      return { ...state, reportCache: next }
    }
    case 'SET_MULTI':
      return { ...state, multiReportList: action.list }
    case 'SET_LOADING':
      return { ...state, loading: action.loading }
    case 'CLEAR_CACHE':
      return { ...state, reportCache: {}, multiReportList: [] }
    case 'TOGGLE_COMPARE': {
      const set = new Set(state.selectedForCompare)
      if (set.has(action.name)) set.delete(action.name)
      else set.add(action.name)
      return { ...state, selectedForCompare: Array.from(set) }
    }
    case 'SET_COMPARE':
      return { ...state, selectedForCompare: action.names }
    case 'CLEAR_COMPARE':
      return { ...state, selectedForCompare: [] }
    default:
      return state
  }
}

// ------------------------------------------------------------------ //
// Context                                                             //
// ------------------------------------------------------------------ //

interface ReportsCtx extends ReportsState {
  setRootPath: (p: string) => void
  scanReports: () => Promise<void>
  selectReports: (r: string[]) => void
  loadReport: (name: string) => Promise<LoadReportResponse>
  loadMultiReports: (names: string[]) => Promise<ReportData[]>
  toggleSelectForCompare: (name: string) => void
  setCompareSelection: (names: string[]) => void
  clearCompareSelection: () => void
}

const ReportsContext = createContext<ReportsCtx>(null!)

export function ReportsProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState)

  // Fetch the server-side default outputs_root from /api/v1/config on mount
  useEffect(() => {
    let cancelled = false
    api<{ outputs_root: string }>('/api/v1/config')
      .then((cfg) => {
        if (!cancelled && cfg.outputs_root && !userSetRootRef.current) {
          dispatch({ type: 'SET_ROOT', rootPath: cfg.outputs_root })
        }
      })
      .catch(() => {/* ignore; keep default */})
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const userSetRootRef = useRef(false)
  const stateRef = useRef(state)
  useEffect(() => { stateRef.current = state }, [state])

  const setRootPath = useCallback((p: string) => {
    userSetRootRef.current = true
    dispatch({ type: 'SET_ROOT', rootPath: p })
    dispatch({ type: 'CLEAR_CACHE' })
  }, [])

  const scanReportsAction = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', loading: true })
    try {
      const reports = await reportsApi.scanReports(state.rootPath)
      dispatch({ type: 'SET_AVAILABLE', reports })
    } finally {
      dispatch({ type: 'SET_LOADING', loading: false })
    }
  }, [state.rootPath])

  const selectReports = useCallback((r: string[]) => dispatch({ type: 'SET_SELECTED', reports: r }), [])

  const loadReport = useCallback(
    async (name: string) => {
      if (state.reportCache[name]) return state.reportCache[name]
      dispatch({ type: 'SET_LOADING', loading: true })
      try {
        const data = await reportsApi.loadReport(state.rootPath, name)
        dispatch({ type: 'CACHE_REPORT', name, data })
        return data
      } finally {
        dispatch({ type: 'SET_LOADING', loading: false })
      }
    },
    [state.rootPath, state.reportCache],
  )

  const loadMultiReports = useCallback(
    async (names: string[]) => {
      dispatch({ type: 'SET_LOADING', loading: true })
      try {
        // Load via cache-aware path so repeat loads in compare view don't refetch.
        // Per-report tagging preserves source mapping when reports share model_name.
        const { rootPath, reportCache } = stateRef.current
        const results = await Promise.all(
          names.map(async (name) => {
            if (reportCache[name]) return reportCache[name]
            const data = await reportsApi.loadReport(rootPath, name)
            dispatch({ type: 'CACHE_REPORT', name, data })
            return data
          }),
        )
        const list = results.flatMap((res, i) =>
          res.report_list.map((r) => ({ ...r, _reportName: names[i] })),
        )
        dispatch({ type: 'SET_MULTI', list })
        return list
      } finally {
        dispatch({ type: 'SET_LOADING', loading: false })
      }
    },
    [],
  )

  const toggleSelectForCompare = useCallback(
    (name: string) => dispatch({ type: 'TOGGLE_COMPARE', name }),
    [],
  )

  const setCompareSelection = useCallback(
    (names: string[]) => dispatch({ type: 'SET_COMPARE', names }),
    [],
  )

  const clearCompareSelection = useCallback(
    () => dispatch({ type: 'CLEAR_COMPARE' }),
    [],
  )

  const value = useMemo<ReportsCtx>(
    () => ({
      ...state,
      setRootPath,
      scanReports: scanReportsAction,
      selectReports,
      loadReport,
      loadMultiReports,
      toggleSelectForCompare,
      setCompareSelection,
      clearCompareSelection,
    }),
    [state, setRootPath, scanReportsAction, selectReports, loadReport, loadMultiReports, toggleSelectForCompare, setCompareSelection, clearCompareSelection],
  )

  return <ReportsContext.Provider value={value}>{children}</ReportsContext.Provider>
}

export function useReports() {
  return useContext(ReportsContext)
}
