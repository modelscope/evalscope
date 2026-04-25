import { useCallback, useEffect, useMemo, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import type { PredictionRow, ReportData } from '@/api/types'
import { getPredictions, getDataFrame } from '@/api/reports'
import Select from '@/components/ui/Select'
import Card from '@/components/ui/Card'

import Pagination from '@/components/common/Pagination'
import ScoreBadge from '@/components/common/ScoreBadge'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import ChatView from '@/components/single/ChatView'
import FlatView from '@/components/single/FlatView'
import Tabs from '@/components/ui/Tabs'
import Skeleton from '@/components/ui/Skeleton'
import { prettyJson } from '@/utils/formatUtils'

interface Props {
  reportName: string
  datasetName: string
  rootPath: string
  report?: ReportData
}

type ViewMode = 'chat' | 'flat' | 'table'

export default function PredictionsTab({ reportName, datasetName, rootPath }: Props) {
  const { t } = useLocale()
  const [subsets, setSubsets] = useState<string[]>([])
  const [selectedSubset, setSelectedSubset] = useState('')
  const [predictions, setPredictions] = useState<PredictionRow[]>([])
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState('All')
  const [threshold, setThreshold] = useState(0.99)
  const [page, setPage] = useState(1)
  const [viewMode, setViewMode] = useState<ViewMode>('chat')

  // Load subsets when dataset changes
  useEffect(() => {
    if (!datasetName || !reportName) return
    let cancelled = false

    const loadSubsets = async () => {
      try {
        const dfRes = await getDataFrame(rootPath, reportName, 'dataset', datasetName)
        if (cancelled) return
        const subNames: string[] = []
        for (const row of dfRes.data) {
          const catCol = Object.keys(row).find((k) => k.startsWith('Cat.'))
          if (catCol && row[catCol] === '-') continue
          const name = String(row['Subset'] ?? '')
          if (name && !subNames.includes(name)) subNames.push(name)
        }
        setSubsets(subNames)
        setSelectedSubset(subNames[0] ?? '')
        setPredictions([])
      } catch (e) {
        console.error('Failed to load subsets:', e)
      }
    }
    loadSubsets()
    return () => { cancelled = true }
  }, [datasetName, reportName, rootPath])

  // Load predictions when subset changes
  const loadPredictions = useCallback(async () => {
    if (!selectedSubset || !reportName || !datasetName) return
    setLoading(true)
    try {
      const res = await getPredictions(rootPath, reportName, datasetName, selectedSubset)
      setPredictions(res.predictions)
    } catch (e) {
      console.error('Failed to load predictions:', e)
      setPredictions([])
    } finally {
      setLoading(false)
    }
  }, [rootPath, reportName, datasetName, selectedSubset])

  useEffect(() => {
    loadPredictions()
  }, [loadPredictions])

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
  useEffect(() => {
    setPage(1)
  }, [mode, threshold, selectedSubset])

  const viewTabs = [
    { key: 'chat', label: t('prediction.chatView') },
    { key: 'flat', label: t('prediction.flatView') },
    { key: 'table', label: t('prediction.tableView') },
  ]

  const subsetOptions = subsets.map((s) => ({ value: s, label: s }))

  return (
    <div className="flex flex-col gap-4">
      {/* Subset selector */}
      <div className="max-w-xs">
        <Select
          label={t('reportDetail.selectSubset')}
          options={subsetOptions}
          value={selectedSubset}
          onChange={setSelectedSubset}
          placeholder={`-- ${t('reportDetail.selectSubset')} --`}
        />
      </div>

      {loading && <Skeleton lines={4} />}

      {!loading && predictions.length > 0 && (
        <>
          {/* Controls row */}
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex flex-wrap gap-1">
              {([['All', predictions.length], ['Pass', passCount], ['Fail', failCount]] as const).map(([m, count]) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`px-3 py-1 text-xs rounded-full transition-colors ${
                    mode === m
                      ? 'bg-[var(--color-primary)] text-white'
                      : 'bg-[var(--color-surface)] text-[var(--color-ink-muted)] hover:bg-[var(--color-surface-hover)] border border-[var(--color-border)]'
                  }`}
                >
                  {m} ({count})
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-[var(--text-muted)]">{t('single.scoreThreshold')}</label>
              <input
                type="number"
                value={threshold}
                step={0.01}
                min={0}
                max={1}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-20 px-2 py-1 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] focus:outline-none focus:border-[var(--accent)]"
              />
            </div>
            <div className="ml-auto">
              <Tabs tabs={viewTabs} activeKey={viewMode} onChange={(k) => setViewMode(k as ViewMode)} />
            </div>
          </div>

          {/* Pagination bar */}
          <Card>
            <div className="flex items-center justify-end">
              <Pagination page={page} total={total} onChange={setPage} />
            </div>
          </Card>

          {/* Content area */}
          {row && (
            <div className="transition-all duration-200">
              {viewMode === 'chat' && <ChatView prediction={row} threshold={threshold} />}
              {viewMode === 'flat' && <FlatView prediction={row} threshold={threshold} />}
              {viewMode === 'table' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <TablePanel title="Score">
                    <pre className="text-xs font-mono whitespace-pre-wrap break-all text-[var(--text-muted)]">
                      {prettyJson(row.Score)}
                    </pre>
                  </TablePanel>
                  <TablePanel title={t('common.normalizedScore')}>
                    <ScoreBadge score={row.NScore} threshold={threshold} />
                  </TablePanel>
                  <TablePanel title={t('common.gold')}>
                    <MarkdownRenderer content={formatValue(row.Gold)} />
                  </TablePanel>
                  <TablePanel title={t('common.pred')}>
                    <MarkdownRenderer content={formatValue(row.Pred)} />
                  </TablePanel>
                  <TablePanel title={t('common.input')}>
                    <MarkdownRenderer content={formatValue(row.Input)} />
                  </TablePanel>
                  <TablePanel title={t('common.generated')}>
                    <MarkdownRenderer content={formatValue(row.Generated)} />
                  </TablePanel>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {!loading && predictions.length === 0 && selectedSubset && (
        <p className="text-sm text-[var(--text-dim)] py-4">{t('common.noData')}</p>
      )}
    </div>
  )
}

function TablePanel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-4">
      <h5 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-dim)] mb-2">{title}</h5>
      <div className="max-h-[300px] overflow-auto">{children}</div>
    </div>
  )
}

function formatValue(val: unknown): string {
  if (val === null || val === undefined) return ''
  if (typeof val === 'object') return '```json\n' + JSON.stringify(val, null, 2) + '\n```'
  return String(val)
}
