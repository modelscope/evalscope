import { useMemo, useState } from 'react'
import { MessageCircle, LayoutList } from 'lucide-react'
import type { PredictionRow } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import FilterBar from '@/components/common/FilterBar'
import Pagination from '@/components/common/Pagination'
import ChatView from './ChatView'
import FlatView from './FlatView'

interface Props {
  predictions: PredictionRow[]
}

const ANSWER_MODES = ['All', 'Pass', 'Fail']
type ViewMode = 'chat' | 'flat'

export default function PredictionBrowser({ predictions }: Props) {
  const { t } = useLocale()
  const [mode, setMode] = useState('All')
  const [threshold, setThreshold] = useState(0.99)
  const [page, setPage] = useState(1)
  const [viewMode, setViewMode] = useState<ViewMode>('flat')

  const filtered = useMemo(() => {
    if (mode === 'Pass') return predictions.filter((p) => p.NScore >= threshold)
    if (mode === 'Fail') return predictions.filter((p) => p.NScore < threshold)
    return predictions
  }, [predictions, mode, threshold])

  const passCount = useMemo(() => predictions.filter((p) => p.NScore >= threshold).length, [predictions, threshold])
  const failCount = predictions.length - passCount
  const total = filtered.length
  const row = total > 0 ? filtered[Math.min(page - 1, total - 1)] : null

  // Reset page when filter changes
  useMemo(() => setPage(1), [mode, threshold]) // eslint-disable-line react-hooks/exhaustive-deps

  const viewButtons: { key: ViewMode; icon: React.ReactNode; label: string }[] = [
    { key: 'chat', icon: <MessageCircle size={14} />, label: t('prediction.chatView') },
    { key: 'flat', icon: <LayoutList size={14} />, label: t('prediction.flatView') },
  ]

  return (
    <div className="flex flex-col gap-3">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        <FilterBar modes={ANSWER_MODES} active={mode} onChange={setMode} />
        <div className="flex items-center gap-2">
          <label className="text-xs text-[var(--color-ink-muted)]">{t('single.scoreThreshold')}</label>
          <input
            type="number"
            value={threshold}
            step={0.01}
            min={0}
            max={1}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="w-20 px-2 py-1 text-sm rounded-md bg-[var(--color-surface)] border border-[var(--color-border)]"
          />
        </div>

        {/* View mode toggle */}
        <div
          className="flex items-center rounded-xl p-0.5 gap-0.5 ml-auto"
          style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)' }}
        >
          {viewButtons.map(({ key, icon, label }) => (
            <button
              key={key}
              onClick={() => setViewMode(key)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
              style={
                viewMode === key
                  ? {
                      background: 'var(--color-primary-muted)',
                      color: 'var(--color-primary)',
                      boxShadow: '0 1px 4px rgba(99,102,241,0.15)',
                    }
                  : {
                      color: 'var(--color-ink-muted)',
                    }
              }
            >
              {icon}
              <span>{label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Stats */}
      <div className="flex items-center justify-between bg-[var(--color-surface)] rounded-lg px-3 py-2 border border-[var(--color-border)]">
        <span className="text-sm">
          All: <b>{predictions.length}</b> | Pass: <b>{passCount}</b> | Fail: <b>{failCount}</b>
        </span>
        <Pagination page={page} total={total} onChange={setPage} />
      </div>

      {/* Content area */}
      {row && (
        <>
          {viewMode === 'chat' && <ChatView prediction={row} threshold={threshold} />}
          {viewMode === 'flat' && <FlatView prediction={row} threshold={threshold} />}
        </>
      )}
    </div>
  )
}


