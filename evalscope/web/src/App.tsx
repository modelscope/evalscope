import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { LocaleProvider } from '@/contexts/LocaleContext'
import { ThemeProvider } from '@/contexts/ThemeContext'
import { ReportsProvider } from '@/contexts/ReportsContext'
import MainLayout from '@/layouts/MainLayout'
import { lazy, Suspense } from 'react'
import LoadingSpinner from '@/components/common/LoadingSpinner'

const DashboardPage = lazy(() => import('@/pages/DashboardPage'))
const SingleModelPage = lazy(() => import('@/pages/SingleModelPage'))
const MultiModelPage = lazy(() => import('@/pages/MultiModelPage'))
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
          <Route path="/dashboard/single" element={<SingleModelPage />} />
          <Route path="/dashboard/multi" element={<MultiModelPage />} />
          <Route path="/eval" element={<EvalTaskPage />} />
          <Route path="/perf" element={<PerfTaskPage />} />
          <Route path="/report" element={<ReportViewerPage />} />
          <Route path="/benchmarks" element={<BenchmarksPage />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Route>
      </Routes>
    </Suspense>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <LocaleProvider>
          <ReportsProvider>
            <AppRoutes />
          </ReportsProvider>
        </LocaleProvider>
      </ThemeProvider>
    </BrowserRouter>
  )
}
