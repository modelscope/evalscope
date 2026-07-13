import { useEffect, useRef, useState } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import TopNav from '@/components/nav/TopNav'
import PathBar from '@/components/ui/PathBar'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'

// Result routes share a single global "scan this directory" control.
const SCAN_ROUTES = new Set(['/dashboard', '/reports', '/performance'])

export default function MainLayout() {
  const location = useLocation()
  const { t } = useLocale()
  const { rootPath, triggerScan } = useReports()
  const [visible, setVisible] = useState(true)
  const prevPath = useRef(location.pathname)

  // Local mirror of rootPath so typing does not fan out a rescan on every keystroke.
  const [pathInput, setPathInput] = useState(rootPath)
  useEffect(() => {
    const sync = () => setPathInput(rootPath)
    sync()
  }, [rootPath])

  useEffect(() => {
    if (prevPath.current !== location.pathname) {
      setVisible(false)
      const timer = setTimeout(() => {
        setVisible(true)
        prevPath.current = location.pathname
      }, 60)
      return () => clearTimeout(timer)
    }
  }, [location.pathname])

  const showPathBar = SCAN_ROUTES.has(location.pathname)

  return (
    <div className="flex flex-col min-h-screen">
      <TopNav />
      <main className="flex-1 max-w-[1600px] mx-auto w-full px-4 py-5 flex flex-col gap-4">
        {showPathBar && (
          <PathBar
            value={pathInput}
            onChange={setPathInput}
            onSubmit={() => triggerScan(pathInput.trim())}
            placeholder={t('reports.pathLabel')}
            submitLabel={t('reports.scan')}
            scanningLabel={t('reports.scanning')}
          />
        )}
        <div
          key={location.pathname}
          className="page-enter"
          style={{ opacity: visible ? undefined : 0 }}
        >
          <Outlet />
        </div>
      </main>
    </div>
  )
}
