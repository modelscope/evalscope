import { useEffect, useState } from 'react'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import { FolderOpen, Loader2, ChevronDown, ChevronUp, CheckSquare, Square } from 'lucide-react'

export default function Sidebar() {
  const { t } = useLocale()
  const { rootPath, setRootPath, availableReports, selectedReports, selectReports, scanReports, loading } = useReports()
  const [localPath, setLocalPath] = useState(rootPath)
  const [collapsed, setCollapsed] = useState(false)

  // Scan on first mount
  useEffect(() => {
    scanReports()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleScan = () => {
    setRootPath(localPath)
    setTimeout(() => scanReports(), 0)
  }

  const handleToggle = (report: string) => {
    if (selectedReports.includes(report)) {
      selectReports(selectedReports.filter((r) => r !== report))
    } else {
      selectReports([...selectedReports, report])
    }
  }

  const handleSelectAll = () => {
    if (selectedReports.length === availableReports.length) {
      selectReports([])
    } else {
      selectReports([...availableReports])
    }
  }

  const allSelected = availableReports.length > 0 && selectedReports.length === availableReports.length

  return (
    <div className="flex flex-col gap-2">
      {/* Header with collapse toggle */}
      <button
        onClick={() => setCollapsed((v) => !v)}
        className="flex items-center justify-between w-full group"
      >
        <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--color-ink-muted)] group-hover:text-[var(--color-ink)] transition-colors">
          {t('sidebar.settings')}
        </h3>
        {collapsed
          ? <ChevronDown size={13} className="text-[var(--color-ink-faint)]" />
          : <ChevronUp size={13} className="text-[var(--color-ink-faint)]" />
        }
      </button>

      {!collapsed && (
        <div className="flex flex-col gap-2.5">
          {/* Root path */}
          <div className="flex gap-1.5">
            <input
              value={localPath}
              onChange={(e) => setLocalPath(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleScan()}
              placeholder={t('sidebar.reportRootPath')}
              className="flex-1 px-2.5 py-1.5 text-xs rounded-lg bg-[var(--color-surface-2)] border border-[var(--color-border-subtle)] focus:outline-none focus:border-[var(--color-primary)] focus:shadow-[0_0_0_3px_var(--color-primary-muted)] transition-all placeholder-[var(--color-ink-faint)] text-[var(--color-ink)]"
            />
            <button
              onClick={handleScan}
              disabled={loading}
              title="Scan reports"
              className="w-8 h-[30px] flex items-center justify-center rounded-lg bg-[var(--color-surface-2)] border border-[var(--color-border-subtle)] hover:border-[var(--color-primary)] hover:text-[var(--color-primary)] disabled:opacity-50 transition-all text-[var(--color-ink-muted)]"
            >
              {loading ? <Loader2 size={13} className="animate-spin" /> : <FolderOpen size={13} />}
            </button>
          </div>

          {/* Report list header */}
          <div className="flex items-center justify-between">
            <span className="text-xs text-[var(--color-ink-muted)]">
              {t('sidebar.selectReports')}
              {availableReports.length > 0 && (
                <span className="ml-1 px-1.5 py-0.5 rounded-full text-[10px] bg-[var(--color-primary-muted)] text-[var(--color-primary)] font-medium">
                  {selectedReports.length}/{availableReports.length}
                </span>
              )}
            </span>
            {availableReports.length > 0 && (
              <button
                onClick={handleSelectAll}
                className="flex items-center gap-1 text-[10px] text-[var(--color-ink-muted)] hover:text-[var(--color-primary)] transition-colors"
              >
                {allSelected
                  ? <CheckSquare size={11} />
                  : <Square size={11} />
                }
                {allSelected ? t('dashboard.deselectAll') : t('dashboard.selectAll')}
              </button>
            )}
          </div>

          {/* Report items */}
          <div className="max-h-[280px] overflow-y-auto flex flex-col gap-0.5 -mx-1 px-1">
            {availableReports.map((r) => {
              const isChecked = selectedReports.includes(r)
              return (
                <label
                  key={r}
                  className={`flex items-start gap-2 px-2 py-1.5 rounded-lg text-xs cursor-pointer transition-all ${
                    isChecked
                      ? 'bg-[var(--color-primary-muted)] text-[var(--color-ink)]'
                      : 'hover:bg-[var(--color-surface-hover)] text-[var(--color-ink-muted)]'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={() => handleToggle(r)}
                    className="mt-0.5 accent-[var(--color-primary)] shrink-0"
                  />
                  <span className="break-all leading-tight">{r}</span>
                </label>
              )
            })}
            {availableReports.length === 0 && !loading && (
              <div className="py-4 text-center">
                <p className="text-xs text-[var(--color-ink-faint)]">{t('sidebar.warning')}</p>
              </div>
            )}
            {loading && availableReports.length === 0 && (
              <div className="py-4 flex justify-center">
                <Loader2 size={16} className="animate-spin text-[var(--color-primary)]" />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
