import { Outlet } from 'react-router-dom'
import TopNav from '@/components/nav/TopNav'

export default function MainLayout() {
  return (
    <div className="flex flex-col min-h-screen">
      <TopNav />
      <main className="flex-1 max-w-[1600px] mx-auto w-full px-4 py-4">
        <Outlet />
      </main>
    </div>
  )
}
