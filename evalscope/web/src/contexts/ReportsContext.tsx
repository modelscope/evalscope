import { createContext, useCallback, useContext, useMemo, useReducer, type ReactNode } from 'react'
import type { LoadReportResponse, ReportData } from '@/api/types'
import * as reportsApi from '@/api/reports'

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

const initialState: ReportsState = {
  rootPath: './outputs',
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
    case 'CACHE_REPORT':
      return { ...state, reportCache: { ...state.reportCache, [action.name]: action.data } }
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

  const setRootPath = useCallback((p: string) => dispatch({ type: 'SET_ROOT', rootPath: p }), [])

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
        // Load each report individually to preserve the source reportName mapping.
        // This avoids data collisions when multiple reports share the same model_name.
        const results = await Promise.all(
          names.map((name) => reportsApi.loadReport(state.rootPath, name))
        )
        const list = results.flatMap((res, i) =>
          res.report_list.map((r) => ({ ...r, _reportName: names[i] }))
        )
        dispatch({ type: 'SET_MULTI', list })
        return list
      } finally {
        dispatch({ type: 'SET_LOADING', loading: false })
      }
    },
    [state.rootPath],
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
