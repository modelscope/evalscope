import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Hash, List, ArrowUp, ArrowDown, HelpCircle, Search, MessageSquare, AlertCircle } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import type { PredictionRow, ReportData } from '@/api/types'
import { isDomainError } from '@/api/errors'
import { getPredictions, getDataFrame } from '@/api/reports'
import Select from '@/components/ui/Select'

import ChatView from '@/components/single/ChatView'
import Skeleton from '@/components/ui/Skeleton'
import EmptyStateSystem from '@/components/common/EmptyStateSystem'

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
  const [loadError, setLoadError] = useState('')
  const [reloadToken, setReloadToken] = useState(0)
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
    const controller = new AbortController()

    const loadSubsets = async () => {
      setLoadError('')
      try {
        const dfRes = await getDataFrame(rootPath, reportName, 'dataset', datasetName, controller.signal)
        if (controller.signal.aborted) return
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
        if (controller.signal.aborted || (isDomainError(e) && e.kind === 'aborted')) return
        setLoadError(e instanceof Error ? e.message : t('common.loadError'))
        console.error('Failed to load subsets:', e)
      }
    }
    loadSubsets()
    return () => controller.abort()
  }, [datasetName, reportName, rootPath, initialSubset, t])

  // Load predictions when subset changes
  useEffect(() => {
    if (!selectedSubset || !reportName || !datasetName) return
    const controller = new AbortController()
    const loadPredictions = async () => {
      setLoading(true)
      setLoadError('')
      try {
        const res = await getPredictions(rootPath, reportName, datasetName, selectedSubset, controller.signal)
        if (!controller.signal.aborted) setPredictions(res.predictions)
      } catch (e) {
        if (controller.signal.aborted || (isDomainError(e) && e.kind === 'aborted')) return
        console.error('Failed to load predictions:', e)
        setLoadError(e instanceof Error ? e.message : t('common.loadError'))
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }
    loadPredictions()
    return () => controller.abort()
  }, [rootPath, reportName, datasetName, selectedSubset, reloadToken, t])

  // The threshold is a view-only filter (above/below), not a pass/fail verdict,
  // and it never leaves this view (Req 1.10, 1.11).
  const filtered = useMemo(() => {
    if (mode === 'Above') return predictions.filter((p) => p.NScore >= threshold)
    if (mode === 'Below') return predictions.filter((p) => p.NScore < threshold)
    return predictions
  }, [predictions, mode, threshold])

  const aboveCount = useMemo(() => predictions.filter((p) => p.NScore >= threshold).length, [predictions, threshold])
  const belowCount = predictions.length - aboveCount
  const totalPages = filtered.length
  const row = totalPages > 0 ? filtered[Math.min(page - 1, totalPages - 1)] : null

  // Reset page & search state when filter changes
  useEffect(() => {
    const reset = () => {
      setPage(1)
      setIndexSearch('')
      setMsgIdSearch('')
      setIndexError(false)
      setMsgIdError(false)
      setHighlightMsgId(undefined)
    }
    reset()
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

  // Filter button config. Labels describe the view filter (above/below the
  // threshold), not a pass/fail outcome (Req 1.11).
  const filterBtns = [
    { key: 'All', label: t('common.all'), icon: <List size={13} />, count: predictions.length },
    { key: 'Above', label: t('prediction.aboveFilter'), icon: <ArrowUp size={13} />, count: aboveCount },
    { key: 'Below', label: t('prediction.belowFilter'), icon: <ArrowDown size={13} />, count: belowCount },
  ] as const

  const navBtnBase = 'bg-transparent border border-[var(--border)] rounded-full min-w-[44px] min-h-[44px] flex items-center justify-center text-[var(--text)] transition-colors'
  const searchInputBase = 'pl-7 pr-2 py-[0.3rem] text-[0.8rem] w-[120px] bg-[var(--bg-deep)] rounded-[var(--radius-sm)] text-[var(--text)] outline-none transition-colors'

  return (
    <div className="flex flex-col gap-3">

      {/* ── 行 1：全局配置区 — Subset（左）+ Threshold（右） ── */}
      <div className="flex items-end justify-between gap-4 flex-wrap">
        {/* 左：Subset 选择器 */}
        <div className="flex-none max-w-[280px] min-w-[160px]">
          <Select
            label={t('reportDetail.selectSubset')}
            options={subsetOptions}
            value={selectedSubset}
            onChange={setSelectedSubset}
            placeholder={`-- ${t('reportDetail.selectSubset')} --`}
          />
        </div>

        {/* 右：Score Threshold + ? 图标 */}
        <div className="flex items-center gap-[0.4rem] shrink-0 pb-[2px]">
          <label htmlFor="prediction-score-threshold" className="text-xs text-[var(--text-muted)] whitespace-nowrap">
            {t('single.scoreThreshold')}
          </label>
          <input
            id="prediction-score-threshold"
            name="prediction-score-threshold"
            type="number"
            value={threshold}
            step={0.01}
            min={0}
            max={1}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="w-20 px-2 py-1 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] focus:outline-none focus:border-[var(--accent)]"
          />
          <span
            className="relative inline-flex cursor-help text-[var(--text-dim)]"
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
      <hr className="border-none border-t border-[var(--border)] m-0" />

      {loading && <Skeleton lines={4} />}

      {loadError && (
        <div
          role="alert"
          className="flex flex-wrap items-center justify-between gap-2 rounded-[var(--radius-sm)] border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-sm text-[var(--danger)]"
        >
          <span>{loadError}</span>
          {selectedSubset && (
            <button
              type="button"
              onClick={() => setReloadToken((value) => value + 1)}
              className="min-h-11 rounded-[var(--radius-sm)] border border-current px-3 font-medium"
            >
              {t('common.retry')}
            </button>
          )}
        </div>
      )}

      {!loading && predictions.length > 0 && (
        <>
          {/* ── 行 2：操作区 — 过滤器（左）+ 搜索框（右） ── */}
          <div className="flex items-center justify-between gap-3 flex-wrap">
            {/* 左：All / Pass / Fail 按钮组 */}
            <div className="inline-flex rounded-[var(--radius)] border border-[var(--border-md)] overflow-hidden">
              {filterBtns.map(({ key, label, icon, count }, idx) => {
                const isActive = mode === key
                return (
                  <button
                    key={key}
                    onClick={() => { setMode(key); setPage(1) }}
                    className={[
                      'flex items-center gap-[0.4rem] px-4 py-2 text-sm font-medium border-0 cursor-pointer transition-colors',
                      idx < filterBtns.length - 1 ? 'border-r border-[var(--border-md)]' : '',
                      isActive ? 'bg-[var(--accent)] text-[var(--bg)]' : 'bg-[var(--bg-card2)] text-[var(--text-muted)]',
                    ].join(' ')}
                  >
                    {icon}
                    <span>{label}</span>
                    <span className="opacity-65 text-[0.8rem]">{count}</span>
                  </button>
                )
              })}
            </div>

            {/* 右：搜索跳转框 */}
            <div className="flex items-center gap-2">
              {/* Sample index search */}
              <div className="relative flex items-center">
                <Search size={12} className="absolute left-2 text-[var(--text-dim)] pointer-events-none" />
                <input
                  ref={indexInputRef}
                  aria-label={t('prediction.searchByIndex')}
                  name="prediction-index-search"
                  type="text"
                  value={indexSearch}
                  onChange={e => { setIndexSearch(e.target.value); setIndexError(false) }}
                  onKeyDown={e => e.key === 'Enter' && handleIndexSearch()}
                  placeholder={t('prediction.searchByIndex')}
                  className={`${searchInputBase} border ${indexError ? 'border-[var(--danger)]' : 'border-[var(--border)]'}`}
                />
                {indexError && (
                  <span className="absolute -right-5 text-[var(--danger)] inline-flex">
                    <AlertCircle size={13} />
                  </span>
                )}
              </div>

              {/* Message id search */}
              <div className="relative flex items-center">
                <MessageSquare size={12} className="absolute left-2 text-[var(--text-dim)] pointer-events-none" />
                <input
                  ref={msgIdInputRef}
                  aria-label={t('prediction.searchByMsgId')}
                  name="prediction-message-id-search"
                  type="text"
                  value={msgIdSearch}
                  onChange={e => { setMsgIdSearch(e.target.value); setMsgIdError(false) }}
                  onKeyDown={e => e.key === 'Enter' && handleMsgIdSearch()}
                  placeholder={t('prediction.searchByMsgId')}
                  className={`${searchInputBase} border ${msgIdError ? 'border-[var(--danger)]' : 'border-[var(--border)]'}`}
                />
                {msgIdError && (
                  <span className="absolute -right-5 text-[var(--danger)] inline-flex">
                    <AlertCircle size={13} />
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Row 2: Sample nav */}
          <div className="flex items-center gap-3">
            {/* Left arrow */}
            <button
              aria-label={t('prediction.previousSample')}
              onClick={() => { setPage(p => Math.max(1, p - 1)); setHighlightMsgId(undefined) }}
              disabled={page <= 1}
              className={`${navBtnBase} ${page <= 1 ? 'opacity-30 cursor-not-allowed' : 'cursor-pointer'}`}
            >
              <ChevronLeft size={16} />
            </button>

            {/* Sample X / Y with hash icon */}
            <span className="flex items-center gap-[0.3rem] text-[var(--text-muted)] text-sm tabular-nums">
              <Hash size={13} className="opacity-50" />
              Sample {page} / {totalPages}
              {row && (
                <span className="text-xs opacity-50 ml-1">
                  (index: {row.Index})
                </span>
              )}
            </span>

            {/* Right arrow */}
            <button
              aria-label={t('prediction.nextSample')}
              onClick={() => { setPage(p => Math.min(totalPages, p + 1)); setHighlightMsgId(undefined) }}
              disabled={page >= totalPages}
              className={`${navBtnBase} ${page >= totalPages ? 'opacity-30 cursor-not-allowed' : 'cursor-pointer'}`}
            >
              <ChevronRight size={16} />
            </button>
          </div>

          {/* Content area */}
          {row && (
            <div className="transition-all duration-200">
              {highlightMsgId && (
                <p className="mb-2 type-caption text-[var(--text-muted)]" role="status">
                  {t('prediction.messageLocated', { id: highlightMsgId })}
                </p>
              )}
              <ChatView prediction={row} threshold={threshold} highlightMsgId={highlightMsgId} />
            </div>
          )}
        </>
      )}

      {!loading && predictions.length === 0 && selectedSubset && (
        <EmptyStateSystem reason="no-data" context={{ view: 'evaluations' }} />
      )}
    </div>
  )
}
