import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Hash, List, ArrowUp, ArrowDown, HelpCircle, Search, MessageSquare, AlertCircle } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { useScopedState } from '@/hooks/useScopedState'
import type { PredictionRow, ReportData } from '@/api/types'
import { getPredictions, getDataFrame } from '@/api/reports'
import Select from '@/components/ui/Select'
import SegmentedControl, { type SegmentedOption } from '@/components/ui/SegmentedControl'
import ScoreThresholdInput from '@/components/ui/ScoreThresholdInput'
import Tooltip from '@/components/ui/Tooltip'
import SampleNavigator from '@/components/reports/SampleNavigator'

import ChatView from '@/components/chat/ChatView'
import Skeleton from '@/components/ui/Skeleton'
import EmptyStateSystem from '@/components/common/EmptyStateSystem'
import ErrorAlert from '@/components/ui/ErrorAlert'

interface Props {
  reportName: string
  datasetName: string
  rootPath: string
  report?: ReportData
  initialSubset?: string
}

/** Stable placeholders so an unresolved read does not produce a new array each render. */
const EMPTY_SUBSETS: string[] = []
const EMPTY_PREDICTIONS: PredictionRow[] = []

export default function PredictionsTab({ reportName, datasetName, rootPath, initialSubset }: Props) {
  const { t } = useLocale()
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

  // Which subsets this dataset was scored on.
  const subsetsResource = useAsyncResource(
    async (signal) => {
      const frame = await getDataFrame(rootPath, reportName, 'dataset', datasetName, signal)
      const names: string[] = []
      for (const row of frame.data) {
        const catCol = Object.keys(row).find((k) => k.startsWith('Cat.'))
        if (catCol && row[catCol] === '-') continue
        const name = String(row['Subset'] ?? '')
        if (name && !names.includes(name)) names.push(name)
      }
      return names
    },
    [rootPath, reportName, datasetName],
    { enabled: Boolean(datasetName && reportName), fallbackMessage: t('common.loadError') },
  )
  const subsets = subsetsResource.data ?? EMPTY_SUBSETS

  // Follow the requested subset when it exists, otherwise open on the first one.
  const subsetScope = `${rootPath}\0${reportName}\0${datasetName}`
  const defaultSubset = initialSubset && subsets.includes(initialSubset) ? initialSubset : (subsets[0] ?? '')
  // A pick only holds while it is still one of the subsets on offer: the list can
  // come back without it after a rescan.
  const [pickedSubset, setSelectedSubset] = useScopedState<string | null>(subsetScope, null)
  const selectedSubset = pickedSubset !== null && subsets.includes(pickedSubset)
    ? pickedSubset
    : defaultSubset

  const predictionsResource = useAsyncResource(
    (signal) => getPredictions(rootPath, reportName, datasetName, selectedSubset, signal),
    [rootPath, reportName, datasetName, selectedSubset],
    {
      enabled: Boolean(selectedSubset && reportName && datasetName),
      fallbackMessage: t('common.loadError'),
    },
  )
  const predictions = predictionsResource.data?.predictions ?? EMPTY_PREDICTIONS
  const loading = predictionsResource.loading
  const loadError = subsetsResource.error || predictionsResource.error

  // The threshold is a view-only filter (above/below), not a pass/fail verdict,
  // and it never leaves this view. A sample without a usable score belongs to neither
  // side; counting it as Below would report an unscored sample as a failure.
  const filtered = useMemo(() => {
    if (mode === 'Above') return predictions.filter((p) => p.NScore !== null && p.NScore >= threshold)
    if (mode === 'Below') return predictions.filter((p) => p.NScore !== null && p.NScore < threshold)
    return predictions
  }, [predictions, mode, threshold])

  const aboveCount = useMemo(
    () => predictions.filter((p) => p.NScore !== null && p.NScore >= threshold).length,
    [predictions, threshold],
  )
  const belowCount = useMemo(
    () => predictions.filter((p) => p.NScore !== null && p.NScore < threshold).length,
    [predictions, threshold],
  )
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

  const goToSample = useCallback((next: number) => {
    setPage(next)
    setHighlightMsgId(undefined)
  }, [])

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

  // Filter options describe the view filter (above/below the threshold), not a
  // pass/fail outcome.
  const filterOptions: SegmentedOption<string>[] = [
    { value: 'All', label: t('common.all'), icon: <List size={13} />, count: predictions.length },
    { value: 'Above', label: t('prediction.aboveFilter'), icon: <ArrowUp size={13} />, count: aboveCount },
    { value: 'Below', label: t('prediction.belowFilter'), icon: <ArrowDown size={13} />, count: belowCount },
  ]

  const searchInputBase = 'pl-7 pr-2 py-[0.3rem] text-[0.8rem] w-[120px] bg-[var(--bg-deep)] rounded-[var(--radius-sm)] text-[var(--text)] outline-none transition-colors'

  return (
    <div className="flex flex-col gap-3">

      {/* ── Row 1: global config — Subset (left) + Threshold (right) ── */}
      <div className="flex items-end justify-between gap-4 flex-wrap">
        {/* Left: Subset selector */}
        <div className="flex-none max-w-[280px] min-w-[160px]">
          <Select
            label={t('reportDetail.selectSubset')}
            options={subsetOptions}
            value={selectedSubset}
            onChange={setSelectedSubset}
            placeholder={`-- ${t('reportDetail.selectSubset')} --`}
          />
        </div>

        {/* Right: Score Threshold + ? icon */}
        <div className="flex shrink-0 items-end gap-[0.4rem] pb-[2px]">
          <ScoreThresholdInput
            id="prediction-score-threshold"
            value={threshold}
            onChange={setThreshold}
            label={t('single.scoreThreshold')}
          />
          {/* text-dim allowed: non-essential help affordance per DESIGN.md §Text */}
          <Tooltip
            content={t('prediction.thresholdHint')}
            label={t('prediction.thresholdHint')}
            className="mb-2 cursor-help text-[var(--text-dim)]"
          >
            <HelpCircle size={14} />
          </Tooltip>
        </div>
      </div>

      {/* Divider */}
      <hr className="border-none border-t border-[var(--border)] m-0" />

      {loading && <Skeleton lines={4} />}

      {loadError && (
        <ErrorAlert className="flex flex-wrap items-center justify-between gap-2 rounded-[var(--radius-sm)] px-3 py-2">
          <span>{loadError}</span>
          {selectedSubset && (
            <button
              type="button"
              onClick={predictionsResource.reload}
              className="min-h-11 rounded-[var(--radius-sm)] border border-current px-3 font-medium"
            >
              {t('common.retry')}
            </button>
          )}
        </ErrorAlert>
      )}

      {!loading && predictions.length > 0 && (
        <>
          {/* ── Row 2: actions — filters (left) + search box (right) ── */}
          <div className="flex items-center justify-between gap-3 flex-wrap">
            {/* Left: All / Above / Below filter group */}
            <SegmentedControl
              options={filterOptions}
              value={mode}
              onChange={(next) => { setMode(next); setPage(1) }}
              ariaLabel={t('prediction.aboveFilter')}
            />

            {/* Right: search-jump box */}
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
          <SampleNavigator page={page} total={totalPages} onPageChange={goToSample}>
            {/* Sample X / Y with hash icon */}
            <span className="flex items-center gap-[0.3rem] text-sm tabular-nums text-[var(--text-muted)]">
              <Hash size={13} className="opacity-50" />
              Sample {page} / {totalPages}
              {row && (
                <span className="ml-1 text-xs opacity-50">
                  (index: {row.Index})
                </span>
              )}
            </span>
          </SampleNavigator>

          {/* Content area */}
          {row && (
            <div className="transition-all duration-200">
              {highlightMsgId && (
                <p className="type-body-xs mb-2 text-[var(--text-muted)]" role="status">
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
