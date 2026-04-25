import { useMemo, useState } from 'react'
import type { PredictionRow } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import FilterBar from '@/components/common/FilterBar'
import Pagination from '@/components/common/Pagination'
import ScoreBadge from '@/components/common/ScoreBadge'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import { prettyJson } from '@/utils/formatUtils'

interface Props {
  predictions: PredictionRow[]
}

const ANSWER_MODES = ['All', 'Pass', 'Fail']

export default function PredictionBrowser({ predictions }: Props) {
  const { t } = useLocale()
  const [mode, setMode] = useState('All')
  const [threshold, setThreshold] = useState(0.99)
  const [page, setPage] = useState(1)

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
      </div>

      {/* Stats */}
      <div className="flex items-center justify-between bg-[var(--color-surface)] rounded-lg px-3 py-2 border border-[var(--color-border)]">
        <span className="text-sm">
          All: <b>{predictions.length}</b> | Pass: <b>{passCount}</b> | Fail: <b>{failCount}</b>
        </span>
        <Pagination page={page} total={total} onChange={setPage} />
      </div>

      {/* Detail panels */}
      {row && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <Panel title="Score">
            <pre className="text-xs font-mono whitespace-pre-wrap break-all">{prettyJson(row.Score)}</pre>
          </Panel>
          <Panel title={t('common.normalizedScore')}>
            <ScoreBadge score={row.NScore} threshold={threshold} />
          </Panel>
          <Panel title={t('common.gold')}>
            <MarkdownRenderer content={formatValue(row.Gold)} />
          </Panel>
          <Panel title={t('common.pred')}>
            <MarkdownRenderer content={formatValue(row.Pred)} />
          </Panel>
          <Panel title={t('common.input')}>
            <MarkdownRenderer content={formatValue(row.Input)} />
          </Panel>
          <Panel title={t('common.generated')}>
            <MarkdownRenderer content={formatValue(row.Generated)} />
          </Panel>
        </div>
      )}
    </div>
  )
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)] p-3">
      <h5 className="text-xs font-medium text-[var(--color-ink-muted)] mb-2">{title}</h5>
      <div className="max-h-[300px] overflow-auto">{children}</div>
    </div>
  )
}

function formatValue(val: unknown): string {
  if (val === null || val === undefined) return ''
  if (typeof val === 'object') return '```json\n' + JSON.stringify(val, null, 2) + '\n```'
  return String(val)
}
