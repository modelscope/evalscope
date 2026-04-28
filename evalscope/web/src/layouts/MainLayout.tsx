import { useEffect, useRef, useState } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import TopNav from '@/components/nav/TopNav'

export default function MainLayout() {
  const location = useLocation()
  const [visible, setVisible] = useState(true)
  const prevPath = useRef(location.pathname)

  useEffect(() => {
    if (prevPath.current !== location.pathname) {
      setVisible(false)
      const t = setTimeout(() => {
        setVisible(true)
        prevPath.current = location.pathname
      }, 60)
      return () => clearTimeout(t)
    }
  }, [location.pathname])

  return (
    <div className="flex flex-col min-h-screen">
      <TopNav />
      <main className="flex-1 max-w-[1600px] mx-auto w-full px-4 py-5">
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
