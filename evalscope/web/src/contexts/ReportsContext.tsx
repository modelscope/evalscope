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

const initialState: ReportsState = {
  rootPath: './outputs',
  availableReports: [],
  selectedReports: [],
  reportCache: {},
  multiReportList: [],
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
        const res = await reportsApi.loadMultiReport(state.rootPath, names)
        dispatch({ type: 'SET_MULTI', list: res.report_list })
        return res.report_list
      } finally {
        dispatch({ type: 'SET_LOADING', loading: false })
      }
    },
    [state.rootPath],
  )

  const value = useMemo<ReportsCtx>(
    () => ({
      ...state,
      setRootPath,
      scanReports: scanReportsAction,
      selectReports,
      loadReport,
      loadMultiReports,
    }),
    [state, setRootPath, scanReportsAction, selectReports, loadReport, loadMultiReports],
  )

  return <ReportsContext.Provider value={value}>{children}</ReportsContext.Provider>
}

export function useReports() {
  return useContext(ReportsContext)
}
