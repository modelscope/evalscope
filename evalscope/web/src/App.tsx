import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { ReportsProvider } from '@/contexts/ReportsContext'
import { ThemeProvider } from '@/contexts/ThemeContext'
import MainLayout from '@/layouts/MainLayout'
import ErrorBoundary from '@/components/common/ErrorBoundary'
import { lazy, Suspense } from 'react'
import LoadingSpinner from '@/components/common/LoadingSpinner'

const DashboardPage = lazy(() => import('@/pages/DashboardPage'))
const ReportsPage = lazy(() => import('@/pages/ReportsPage'))
const ReportDetailPage = lazy(() => import('@/pages/ReportDetailPage'))
const ComparePage = lazy(() => import('@/pages/ComparePage'))
const TasksPage = lazy(() => import('@/pages/TasksPage'))
const PerfReportsPage = lazy(() => import('@/pages/PerfReportsPage'))
const PerfReportDetailPage = lazy(() => import('@/pages/PerfReportDetailPage'))
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
          <Route path="/performance" element={<PerfReportsPage />} />
          <Route path="/perf-report" element={<PerfReportDetailPage />} />
          <Route path="/tasks" element={<TasksPage />} />
          {/* Legacy task routes — redirect into the unified Tasks page */}
          <Route path="/eval" element={<Navigate to="/tasks?tab=eval" replace />} />
          <Route path="/perf" element={<Navigate to="/tasks?tab=perf" replace />} />
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
        <ThemeProvider>
          <LocaleProvider>
            <ReportsProvider>
              <AppRoutes />
            </ReportsProvider>
          </LocaleProvider>
        </ThemeProvider>
      </ErrorBoundary>
    </BrowserRouter>
  )
}
