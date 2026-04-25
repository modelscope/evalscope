import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { ReportsProvider } from '@/contexts/ReportsContext'
import MainLayout from '@/layouts/MainLayout'
import ErrorBoundary from '@/components/common/ErrorBoundary'
import { lazy, Suspense } from 'react'
import LoadingSpinner from '@/components/common/LoadingSpinner'

const DashboardPage = lazy(() => import('@/pages/DashboardPage'))
const ReportsPage = lazy(() => import('@/pages/ReportsPage'))
const ReportDetailPage = lazy(() => import('@/pages/ReportDetailPage'))
const ComparePage = lazy(() => import('@/pages/ComparePage'))
const EvalTaskPage = lazy(() => import('@/pages/EvalTaskPage'))
const PerfTaskPage = lazy(() => import('@/pages/PerfTaskPage'))
const ReportViewerPage = lazy(() => import('@/pages/ReportViewerPage'))
const BenchmarksPage = lazy(() => import('@/pages/BenchmarksPage'))

function AppRoutes() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route element={<MainLayout />}>
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/reports/:reportId" element={<ReportDetailPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/eval" element={<EvalTaskPage />} />
          <Route path="/perf" element={<PerfTaskPage />} />
          <Route path="/benchmarks" element={<BenchmarksPage />} />
          <Route path="/viewer" element={<ReportViewerPage />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Route>
      </Routes>
    </Suspense>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <ErrorBoundary>
        <LocaleProvider>
          <ReportsProvider>
            <AppRoutes />
          </ReportsProvider>
        </LocaleProvider>
      </ErrorBoundary>
    </BrowserRouter>
  )
}
