import { useEffect, useState } from 'react'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import { FolderOpen, Loader2 } from 'lucide-react'

export default function Sidebar() {
  const { t } = useLocale()
  const { rootPath, setRootPath, availableReports, selectedReports, selectReports, scanReports, loading } = useReports()
  const [localPath, setLocalPath] = useState(rootPath)

  // Scan on first mount
  useEffect(() => {
    scanReports()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleScan = () => {
    setRootPath(localPath)
    // Allow state to propagate then scan
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

  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-sm font-medium text-[var(--color-ink-muted)]">{t('sidebar.settings')}</h3>

      {/* Root path */}
      <div className="flex gap-2">
        <input
          value={localPath}
          onChange={(e) => setLocalPath(e.target.value)}
          placeholder={t('sidebar.reportRootPath')}
          className="flex-1 px-2 py-1.5 text-sm rounded-md bg-[var(--color-surface)] border border-[var(--color-border)] focus:outline-none focus:border-[var(--color-primary)]"
        />
        <button
          onClick={handleScan}
          disabled={loading}
          className="px-3 py-1.5 text-sm rounded-md bg-[var(--color-surface)] border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)] disabled:opacity-50 flex items-center gap-1"
        >
          {loading ? <Loader2 size={14} className="animate-spin" /> : <FolderOpen size={14} />}
        </button>
      </div>

      {/* Report list */}
      <div className="text-xs text-[var(--color-ink-muted)] flex items-center justify-between">
        <span>{t('sidebar.selectReports')} ({availableReports.length})</span>
        {availableReports.length > 0 && (
          <button onClick={handleSelectAll} className="hover:text-[var(--color-ink)]">
            {selectedReports.length === availableReports.length ? 'Deselect all' : 'Select all'}
          </button>
        )}
      </div>
      <div className="max-h-[300px] overflow-y-auto flex flex-col gap-0.5">
        {availableReports.map((r) => (
          <label
            key={r}
            className="flex items-start gap-2 px-2 py-1.5 rounded text-xs hover:bg-[var(--color-surface-hover)] cursor-pointer"
          >
            <input
              type="checkbox"
              checked={selectedReports.includes(r)}
              onChange={() => handleToggle(r)}
              className="mt-0.5 accent-[var(--color-primary)]"
            />
            <span className="break-all leading-tight">{r}</span>
          </label>
        ))}
        {availableReports.length === 0 && !loading && (
          <span className="text-xs text-[var(--color-ink-muted)] py-2">{t('sidebar.warning')}</span>
        )}
      </div>
    </div>
  )
}
