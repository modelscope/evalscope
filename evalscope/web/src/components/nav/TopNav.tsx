import { NavLink } from 'react-router-dom'
import { useLocale } from '@/contexts/LocaleContext'
import LocaleToggle from './LocaleToggle'
import { BarChart3, Gauge, FlaskConical, BookOpen } from 'lucide-react'

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm transition-colors ${
    isActive
      ? 'bg-[var(--color-primary)] text-white'
      : 'hover:bg-[var(--color-surface-hover)] text-[var(--color-ink-muted)]'
  }`

export default function TopNav() {
  const { t } = useLocale()
  return (
    <header className="sticky top-0 z-50 border-b border-[var(--color-border)] bg-[var(--color-bg-page)]/80 backdrop-blur-sm">
      <div className="flex items-center justify-between h-12 px-4 max-w-[1600px] mx-auto">
        {/* Brand */}
        <div className="flex items-center gap-4">
          <span className="font-semibold text-base tracking-tight">EvalScope</span>
          <nav className="flex items-center gap-1">
            <NavLink to="/dashboard" className={linkClass}>
              <BarChart3 size={15} /> {t('nav.dashboard')}
            </NavLink>
            <NavLink to="/eval" className={linkClass}>
              <FlaskConical size={15} /> {t('nav.eval')}
            </NavLink>
            <NavLink to="/perf" className={linkClass}>
              <Gauge size={15} /> {t('nav.perf')}
            </NavLink>
            <NavLink to="/benchmarks" className={linkClass}>
              <BookOpen size={15} /> {t('nav.benchmarks')}
            </NavLink>
          </nav>
        </div>
        {/* Right */}
        <div className="flex items-center gap-3">
          <a
            href="https://github.com/modelscope/evalscope"
            target="_blank"
            rel="noreferrer"
            className="text-[var(--color-ink-muted)] hover:text-[var(--color-ink)] transition-colors"
          >
            <svg viewBox="0 0 24 24" width={18} height={18} fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
          </a>
          <LocaleToggle />
        </div>
      </div>
    </header>
  )
}
