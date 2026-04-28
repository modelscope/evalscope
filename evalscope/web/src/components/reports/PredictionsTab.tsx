import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Hash, List, CircleCheck, CircleX, HelpCircle, Search, MessageSquare, AlertCircle } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import type { PredictionRow, ReportData } from '@/api/types'
import { getPredictions, getDataFrame } from '@/api/reports'
import Select from '@/components/ui/Select'

import ChatView from '@/components/single/ChatView'
import Skeleton from '@/components/ui/Skeleton'

interface Props {
  reportName: string
  datasetName: string
  rootPath: string
  report?: ReportData
  initialSubset?: string
}

export default function PredictionsTab({ reportName, datasetName, rootPath, initialSubset }: Props) {
  const { t } = useLocale()
  const [subsets, setSubsets] = useState<string[]>([])
  const [selectedSubset, setSelectedSubset] = useState('')
  const [predictions, setPredictions] = useState<PredictionRow[]>([])
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState('All')
  const [threshold, setThreshold] = useState(0.99)
  const [page, setPage] = useState(1)

  // Search state
  const [indexSearch, setIndexSearch] = useState('')
  const [msgIdSearch, setMsgIdSearch] = useState('')
  const [indexError, setIndexError] = useState(false)
  const [msgIdError, setMsgIdError] = useState(false)
  const [highlightMsgId, setHighlightMsgId] = useState<string | undefined>(undefined)
  const indexInputRef = useRef<HTMLInputElement>(null)
  const msgIdInputRef = useRef<HTMLInputElement>(null)

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

  // Reset page & search state when filter changes
  useEffect(() => {
    setPage(1)
    setIndexSearch('')
    setMsgIdSearch('')
    setIndexError(false)
    setMsgIdError(false)
    setHighlightMsgId(undefined)
  }, [mode, threshold, selectedSubset])

  // Keyboard navigation (skip when search inputs are focused)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'ArrowLeft' && page > 1) {
        setPage(p => p - 1)
        setHighlightMsgId(undefined)
      } else if (e.key === 'ArrowRight' && page < totalPages) {
        setPage(p => p + 1)
        setHighlightMsgId(undefined)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [page, totalPages])

  // --- Search handlers ---
  const handleIndexSearch = useCallback(() => {
    const q = indexSearch.trim()
    if (!q) return
    const idx = filtered.findIndex(p => p.Index === q)
    if (idx >= 0) {
      setPage(idx + 1)
      setIndexError(false)
      setHighlightMsgId(undefined)
    } else {
      setIndexError(true)
      setTimeout(() => setIndexError(false), 1800)
    }
  }, [indexSearch, filtered])

  const handleMsgIdSearch = useCallback(() => {
    const q = msgIdSearch.trim()
    if (!q) return
    // Search across all predictions (not just filtered) to locate the sample
    const idx = filtered.findIndex(p =>
      p.Messages?.some(m => m.id && m.id.startsWith(q))
    )
    if (idx >= 0) {
      setPage(idx + 1)
      setHighlightMsgId(q)
      setMsgIdError(false)
    } else {
      setMsgIdError(true)
      setHighlightMsgId(undefined)
      setTimeout(() => setMsgIdError(false), 1800)
    }
  }, [msgIdSearch, filtered])

  const subsetOptions = subsets.map((s) => ({ value: s, label: s }))

  // Filter button config
  const filterBtns = [
    { key: 'All', icon: <List size={13} />, count: predictions.length },
    { key: 'Pass', icon: <CircleCheck size={13} />, count: passCount },
    { key: 'Fail', icon: <CircleX size={13} />, count: failCount },
  ] as const

  return (
    <div className="flex flex-col gap-3">

      {/* ── 行 1：全局配置区 — Subset（左）+ Threshold（右） ── */}
      <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
        {/* 左：Subset 选择器 */}
        <div style={{ flex: '0 0 auto', maxWidth: '280px', minWidth: '160px' }}>
          <Select
            label={t('reportDetail.selectSubset')}
            options={subsetOptions}
            value={selectedSubset}
            onChange={setSelectedSubset}
            placeholder={`-- ${t('reportDetail.selectSubset')} --`}
          />
        </div>

        {/* 右：Score Threshold + ? 图标 */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flexShrink: 0, paddingBottom: '2px' }}>
          <label
            className="text-xs"
            style={{ color: 'var(--text-muted)', whiteSpace: 'nowrap' }}
          >
            {t('single.scoreThreshold')}
          </label>
          <input
            type="number"
            value={threshold}
            step={0.01}
            min={0}
            max={1}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="w-20 px-2 py-1 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] focus:outline-none focus:border-[var(--accent)]"
          />
          <span
            style={{ position: 'relative', display: 'inline-flex', cursor: 'help', color: 'var(--text-dim)' }}
            onMouseEnter={e => {
              const tip = document.createElement('div')
              tip.id = '__threshold_tip'
              tip.textContent = t('prediction.thresholdHint')
              Object.assign(tip.style, {
                position: 'fixed',
                background: 'var(--bg-card)',
                color: 'var(--text)',
                border: '1px solid var(--border)',
                borderRadius: '6px',
                padding: '0.35rem 0.65rem',
                fontSize: '0.72rem',
                lineHeight: '1.5',
                maxWidth: '220px',
                zIndex: '9999',
                pointerEvents: 'none',
                boxShadow: 'var(--shadow)',
                whiteSpace: 'normal',
              })
              document.body.appendChild(tip)
              const rect = e.currentTarget.getBoundingClientRect()
              tip.style.left = `${Math.min(rect.left, window.innerWidth - 240)}px`
              tip.style.top = `${rect.bottom + 6}px`
            }}
            onMouseLeave={() => document.getElementById('__threshold_tip')?.remove()}
          >
            <HelpCircle size={14} />
          </span>
        </div>
      </div>

      {/* 分隔线 */}
      <hr style={{ border: 'none', borderTop: '1px solid var(--color-border-subtle)', margin: '0' }} />

      {loading && <Skeleton lines={4} />}

      {!loading && predictions.length > 0 && (
        <>
          {/* ── 行 2：操作区 — 过滤器（左）+ 搜索框（右） ── */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', flexWrap: 'wrap' }}>
            {/* 左：All / Pass / Fail 按钮组 */}
            <div style={{
              display: 'inline-flex',
              borderRadius: 'var(--radius)',
              border: '1px solid var(--border-md)',
              overflow: 'hidden',
            }}>
              {filterBtns.map(({ key, icon, count }, idx) => {
                const isActive = mode === key
                return (
                  <button
                    key={key}
                    onClick={() => { setMode(key); setPage(1) }}
                    style={{
                      padding: '0.5rem 1rem',
                      fontSize: '0.875rem',
                      fontWeight: 500,
                      background: isActive ? 'var(--accent)' : 'var(--bg-card2)',
                      color: isActive ? 'var(--bg)' : 'var(--text-muted)',
                      border: 'none',
                      borderRight: idx < filterBtns.length - 1 ? '1px solid var(--border-md)' : 'none',
                      cursor: 'pointer',
                      transition: 'var(--transition)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.4rem',
                    }}
                  >
                    {icon}
                    <span>{key}</span>
                    <span style={{ opacity: 0.65, fontSize: '0.8rem' }}>{count}</span>
                  </button>
                )
              })}
            </div>

            {/* 右：搜索跳转框 */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              {/* Sample index search */}
              <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                <Search size={12} style={{ position: 'absolute', left: '0.5rem', color: 'var(--text-dim)', pointerEvents: 'none' }} />
                <input
                  ref={indexInputRef}
                  type="text"
                  value={indexSearch}
                  onChange={e => { setIndexSearch(e.target.value); setIndexError(false) }}
                  onKeyDown={e => e.key === 'Enter' && handleIndexSearch()}
                  placeholder={t('prediction.searchByIndex')}
                  style={{
                    paddingLeft: '1.75rem',
                    paddingRight: '0.5rem',
                    paddingTop: '0.3rem',
                    paddingBottom: '0.3rem',
                    fontSize: '0.8rem',
                    width: '120px',
                    background: 'var(--bg-deep)',
                    border: `1px solid ${indexError ? 'var(--danger)' : 'var(--border)'}`,
                    borderRadius: 'var(--radius-sm)',
                    color: 'var(--text)',
                    outline: 'none',
                    transition: 'border 0.2s',
                  }}
                />
                {indexError && (
                  <span style={{ position: 'absolute', right: '-1.2rem', color: 'var(--danger)', display: 'inline-flex' }}>
                    <AlertCircle size={13} />
                  </span>
                )}
              </div>

              {/* Message id search */}
              <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                <MessageSquare size={12} style={{ position: 'absolute', left: '0.5rem', color: 'var(--text-dim)', pointerEvents: 'none' }} />
                <input
                  ref={msgIdInputRef}
                  type="text"
                  value={msgIdSearch}
                  onChange={e => { setMsgIdSearch(e.target.value); setMsgIdError(false) }}
                  onKeyDown={e => e.key === 'Enter' && handleMsgIdSearch()}
                  placeholder={t('prediction.searchByMsgId')}
                  style={{
                    paddingLeft: '1.75rem',
                    paddingRight: '0.5rem',
                    paddingTop: '0.3rem',
                    paddingBottom: '0.3rem',
                    fontSize: '0.8rem',
                    width: '120px',
                    background: 'var(--bg-deep)',
                    border: `1px solid ${msgIdError ? 'var(--danger)' : 'var(--border)'}`,
                    borderRadius: 'var(--radius-sm)',
                    color: 'var(--text)',
                    outline: 'none',
                    transition: 'border 0.2s',
                  }}
                />
                {msgIdError && (
                  <span style={{ position: 'absolute', right: '-1.2rem', color: 'var(--danger)', display: 'inline-flex' }}>
                    <AlertCircle size={13} />
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Row 2: Sample nav */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            {/* Left arrow */}
            <button
              onClick={() => { setPage(p => Math.max(1, p - 1)); setHighlightMsgId(undefined) }}
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

            {/* Sample X / Y with hash icon */}
            <span style={{ color: 'var(--text-muted)', fontSize: '0.875rem', fontVariantNumeric: 'tabular-nums', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
              <Hash size={13} style={{ opacity: 0.5 }} />
              Sample {page} / {totalPages}
              {row && (
                <span style={{ fontSize: '0.75rem', opacity: 0.5, marginLeft: '0.25rem' }}>
                  (index: {row.Index})
                </span>
              )}
            </span>

            {/* Right arrow */}
            <button
              onClick={() => { setPage(p => Math.min(totalPages, p + 1)); setHighlightMsgId(undefined) }}
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
          </div>

          {/* Content area */}
          {row && (
            <div className="transition-all duration-200">
              <ChatView prediction={row} threshold={threshold} highlightMsgId={highlightMsgId} />
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
