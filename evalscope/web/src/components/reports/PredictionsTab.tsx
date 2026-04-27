import { useCallback, useEffect, useMemo, useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import type { PredictionRow, ReportData } from '@/api/types'
import { getPredictions, getDataFrame } from '@/api/reports'
import Select from '@/components/ui/Select'

import ChatView from '@/components/single/ChatView'
import FlatView from '@/components/single/FlatView'
import Tabs from '@/components/ui/Tabs'
import Skeleton from '@/components/ui/Skeleton'

interface Props {
  reportName: string
  datasetName: string
  rootPath: string
  report?: ReportData
  initialSubset?: string
}

type ViewMode = 'chat' | 'flat'

export default function PredictionsTab({ reportName, datasetName, rootPath, initialSubset }: Props) {
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
        // Use initialSubset if provided and valid, otherwise fallback to first
        const target = initialSubset && subNames.includes(initialSubset) ? initialSubset : (subNames[0] ?? '')
        setSelectedSubset(target)
        setPredictions([])
      } catch (e) {
        console.error('Failed to load subsets:', e)
      }
    }
    loadSubsets()
    return () => { cancelled = true }
  }, [datasetName, reportName, rootPath, initialSubset])

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
  const totalPages = filtered.length
  const row = totalPages > 0 ? filtered[Math.min(page - 1, totalPages - 1)] : null

  const filtered_counts = {
    all: predictions.length,
    pass: passCount,
    fail: failCount,
  }

  // Reset page when filter changes
  useEffect(() => {
    setPage(1)
  }, [mode, threshold, selectedSubset])

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'ArrowLeft' && page > 1) {
        setPage(p => p - 1)
      } else if (e.key === 'ArrowRight' && page < totalPages) {
        setPage(p => p + 1)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [page, totalPages])

  const viewTabs = [
    { key: 'chat', label: t('prediction.chatView') },
    { key: 'flat', label: t('prediction.flatView') },
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
          {/* Row 1: Filters + Threshold */}
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '1rem' }}>
            {/* Segmented filter control */}
            <div style={{
              display: 'inline-flex',
              borderRadius: 'var(--radius)',
              border: '1px solid var(--border-md)',
              overflow: 'hidden',
            }}>
              {(['All', 'Pass', 'Fail'] as const).map((m, idx, arr) => {
                const count = m === 'All' ? filtered_counts.all : m === 'Pass' ? filtered_counts.pass : filtered_counts.fail
                const isActive = mode === m
                return (
                  <button
                    key={m}
                    onClick={() => { setMode(m); setPage(1) }}
                    style={{
                      padding: '0.625rem 1.25rem',
                      fontSize: '0.875rem',
                      fontWeight: 500,
                      background: isActive ? 'var(--accent)' : 'var(--bg-card2)',
                      color: isActive ? '#fff' : 'var(--text-muted)',
                      border: 'none',
                      borderRight: idx < arr.length - 1 ? '1px solid var(--border-md)' : 'none',
                      cursor: 'pointer',
                      transition: 'var(--transition)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                    }}
                  >
                    {m}
                    <span style={{ opacity: 0.7 }}>{count}</span>
                  </button>
                )
              })}
            </div>

            {/* Threshold input */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
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
          </div>

          {/* Row 2: Sample nav + View mode */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {/* Left arrow */}
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page <= 1}
              style={{
                background: 'transparent',
                border: '1px solid var(--border)',
                borderRadius: '50%',
                width: '32px',
                height: '32px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: page <= 1 ? 'not-allowed' : 'pointer',
                opacity: page <= 1 ? 0.3 : 1,
                color: 'var(--text)',
                transition: 'var(--transition)',
              }}
            >
              <ChevronLeft size={16} />
            </button>

            {/* Sample X / Y */}
            <span style={{ color: 'var(--text-muted)', fontSize: '0.875rem', fontVariantNumeric: 'tabular-nums' }}>
              Sample {page} / {totalPages}
            </span>

            {/* Right arrow */}
            <button
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              style={{
                background: 'transparent',
                border: '1px solid var(--border)',
                borderRadius: '50%',
                width: '32px',
                height: '32px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: page >= totalPages ? 'not-allowed' : 'pointer',
                opacity: page >= totalPages ? 0.3 : 1,
                color: 'var(--text)',
                transition: 'var(--transition)',
              }}
            >
              <ChevronRight size={16} />
            </button>

            {/* View mode tabs pushed right */}
            <div style={{ marginLeft: 'auto' }}>
              <Tabs tabs={viewTabs} activeKey={viewMode} onChange={(k) => setViewMode(k as ViewMode)} />
            </div>
          </div>

          {/* Content area */}
          {row && (
            <div className="transition-all duration-200">
              {viewMode === 'chat' && <ChatView prediction={row} threshold={threshold} />}
              {viewMode === 'flat' && <FlatView prediction={row} threshold={threshold} />}
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
