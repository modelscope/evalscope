import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Hash, List, ArrowUp, ArrowDown, HelpCircle, Search, MessageSquare, AlertCircle } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { useScopedState } from '@/hooks/useScopedState'
import type { PredictionRow, ReportData, SampleKey } from '@/api/types'
import { getPredictions, getDataFrame, getDiagnosis, getSampleDiagnoses, type DiagnosisFilters } from '@/api/reports'
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
  initialIndex?: number
  focusStep?: number
  diagnosisFilters?: DiagnosisFilters
  onDiagnosisFiltersChange?: (filters: DiagnosisFilters, sample?: SampleKey) => void
}

/** Stable placeholders so an unresolved read does not produce a new array each render. */
const EMPTY_SUBSETS: string[] = []
const EMPTY_PREDICTIONS: PredictionRow[] = []

export default function PredictionsTab({ reportName, datasetName, rootPath, initialSubset, report, initialIndex, focusStep,
  diagnosisFilters = {}, onDiagnosisFiltersChange }: Props) {
  const { t } = useLocale()
  const [mode, setMode] = useState('All')
  const [threshold, setThreshold] = useState(0.99)

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

  const diagnosis = useAsyncResource(signal => getDiagnosis(rootPath,reportName,datasetName,signal),
    [rootPath,reportName,datasetName], {enabled:Boolean(datasetName && onDiagnosisFiltersChange)})
  // Follow the requested subset when it exists, otherwise open on the first one.
  const subsetScope = `${rootPath}\0${reportName}\0${datasetName}\0${initialSubset}`
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
  const filterKey = JSON.stringify([rootPath,reportName,datasetName,selectedSubset,diagnosisFilters])
  const diagnosticFilterActive = Boolean(diagnosisFilters.outcome || diagnosisFilters.failure_type ||
    diagnosisFilters.attribution || diagnosisFilters.finding_category)
  const matches = useAsyncResource(async signal => {
    const ids = new Set<string>()
    let page = 1
    let more: boolean
    do {
      const response = await getSampleDiagnoses(rootPath,reportName,datasetName,
        {...diagnosisFilters,subset_name:selectedSubset},page,signal)
      if (response.status !== 'ready') return {key:filterKey, ids}
      for (const sample of response.items) ids.add(String(sample.key.index))
      more = page*200 < response.total
      page++
    } while (more && !signal.aborted)
    return {key:filterKey, ids}
  }, [filterKey], {enabled:diagnosticFilterActive && Boolean(selectedSubset)})
  const loading = predictionsResource.loading || (diagnosticFilterActive && matches.loading)
  const loadError = subsetsResource.error || predictionsResource.error || (diagnosticFilterActive && matches.error)

  // The threshold is a view-only filter (above/below), not a pass/fail verdict,
  // and it never leaves this view. A sample without a usable score belongs to neither
  // side; counting it as Below would report an unscored sample as a failure.
  const candidates = useMemo(() => diagnosticFilterActive
    ? predictions.filter(p => matches.data?.key===filterKey && matches.data.ids.has(p.Index)) : predictions,
    [predictions, diagnosticFilterActive, matches.data, filterKey])
  const filtered = useMemo(() => {
    if (mode === 'Above') return candidates.filter((p) => p.NScore !== null && p.NScore >= threshold)
    if (mode === 'Below') return candidates.filter((p) => p.NScore !== null && p.NScore < threshold)
    return candidates
  }, [candidates, mode, threshold])

  const aboveCount = useMemo(
    () => candidates.filter((p) => p.NScore !== null && p.NScore >= threshold).length,
    [candidates, threshold],
  )
  const belowCount = useMemo(
    () => candidates.filter((p) => p.NScore !== null && p.NScore < threshold).length,
    [candidates, threshold],
  )
  const requestedPage = filtered.findIndex(p => p.Index === String(initialIndex)) + 1
  const [pickedPage, setPage] = useScopedState<number | null>(`${filterKey}/${mode}/${threshold}/${initialIndex}`, null)
  const page = pickedPage ?? (requestedPage || 1)
  const totalPages = filtered.length
  const row = totalPages > 0 ? filtered[Math.min(page - 1, totalPages - 1)] : null
  const rowScope = `${rootPath}/${reportName}/${datasetName}/${selectedSubset}/${row?.Index}`
  const sampleDiagnosis = useAsyncResource(async signal => ({scope:rowScope,
    response:await getSampleDiagnoses(rootPath,reportName,datasetName,{subset_name:selectedSubset,index:Number(row?.Index)},1,signal)}),
    [rowScope], {enabled:Boolean(row && onDiagnosisFiltersChange)})
  const diagnosisRow = sampleDiagnosis.data?.scope===rowScope && sampleDiagnosis.data.response.status==='ready'
    ? sampleDiagnosis.data.response.items[0] : undefined

  // Reset page & search state when filter changes
  useEffect(() => {
    const reset = () => {
      setIndexSearch('')
      setMsgIdSearch('')
      setIndexError(false)
      setMsgIdError(false)
      setHighlightMsgId(undefined)
    }
    reset()
  }, [mode, threshold, selectedSubset, filterKey])

  const goToSample = useCallback((next: number) => {
    setPage(next)
    const target = filtered[next - 1]
    if (target) onDiagnosisFiltersChange?.({ ...diagnosisFilters, subset_name: selectedSubset },
      { subset: selectedSubset, index: Number(target.Index) })
    setHighlightMsgId(undefined)
  }, [setPage, filtered, onDiagnosisFiltersChange, diagnosisFilters, selectedSubset])

  // --- Search handlers ---
  const handleIndexSearch = useCallback(() => {
    const q = indexSearch.trim()
    if (!q) return
    const idx = filtered.findIndex(p => p.Index === q)
    if (idx >= 0) {
      goToSample(idx + 1)
      setIndexError(false)
      setHighlightMsgId(undefined)
    } else {
      setIndexError(true)
      setTimeout(() => setIndexError(false), 1800)
    }
  }, [indexSearch, filtered, goToSample])

  const handleMsgIdSearch = useCallback(() => {
    const q = msgIdSearch.trim()
    if (!q) return
    // Search across all predictions (not just filtered) to locate the sample
    const idx = filtered.findIndex(p =>
      p.Messages?.some(m => m.id && m.id.startsWith(q))
    )
    if (idx >= 0) {
      goToSample(idx + 1)
      setHighlightMsgId(q)
      setMsgIdError(false)
    } else {
      setMsgIdError(true)
      setHighlightMsgId(undefined)
      setTimeout(() => setMsgIdError(false), 1800)
    }
  }, [msgIdSearch, filtered, goToSample])

  const subsetOptions = subsets.map((s) => ({ value: s, label: s }))

  // Filter options describe the view filter (above/below the threshold), not a
  // pass/fail outcome.
  const filterOptions: SegmentedOption<string>[] = [
    { value: 'All', label: t('common.all'), icon: <List size={13} />, count: candidates.length },
    { value: 'Above', label: t('prediction.aboveFilter'), icon: <ArrowUp size={13} />, count: aboveCount },
    { value: 'Below', label: t('prediction.belowFilter'), icon: <ArrowDown size={13} />, count: belowCount },
  ]

  const searchInputBase = 'pl-7 pr-2 py-[0.3rem] text-[0.8rem] w-[120px] bg-[var(--bg-deep)] rounded-[var(--radius-sm)] text-[var(--text)] outline-none transition-colors'

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-3">
        <strong className="text-sm">{report?.dataset_pretty_name || datasetName}</strong>
        <div className="w-36 shrink-0 [&_label]:sr-only">
          <Select
            label={t('reportDetail.selectSubset')}
            options={subsetOptions}
            value={selectedSubset}
            onChange={(value) => {
              setSelectedSubset(value)
              onDiagnosisFiltersChange?.({ ...diagnosisFilters, subset_name: value })
            }}
            placeholder={`-- ${t('reportDetail.selectSubset')} --`}
          />
        </div>

        <SampleNavigator page={page} total={totalPages} onPageChange={goToSample}>
          <span className="flex items-center gap-[0.3rem] text-sm tabular-nums text-[var(--text-muted)]">
            <Hash size={13} className="opacity-50" />
            {t('reportDetail.samplePosition', { page, total: totalPages })}
            {row && (
              <span className="ml-1 text-xs opacity-50">{t('reportDetail.sampleIndex', { index: row.Index })}</span>
            )}
          </span>
        </SampleNavigator>

        <div className="flex items-center gap-2">
          <div className="relative flex items-center">
            <Search size={12} className="absolute left-2 text-[var(--text-dim)] pointer-events-none" />
            <input
              ref={indexInputRef}
              aria-label={t('prediction.searchByIndex')}
              name="prediction-index-search"
              type="text"
              value={indexSearch}
              onChange={(e) => {
                setIndexSearch(e.target.value)
                setIndexError(false)
              }}
              onKeyDown={(e) => e.key === 'Enter' && handleIndexSearch()}
              placeholder={t('prediction.searchByIndex')}
              className={`${searchInputBase} border ${indexError ? 'border-[var(--danger)]' : 'border-[var(--border)]'}`}
            />
            {indexError && (
              <span className="absolute -right-5 text-[var(--danger)] inline-flex">
                <AlertCircle size={13} />
              </span>
            )}
          </div>

          <div className="relative flex items-center">
            <MessageSquare size={12} className="absolute left-2 text-[var(--text-dim)] pointer-events-none" />
            <input
              ref={msgIdInputRef}
              aria-label={t('prediction.searchByMsgId')}
              name="prediction-message-id-search"
              type="text"
              value={msgIdSearch}
              onChange={(e) => {
                setMsgIdSearch(e.target.value)
                setMsgIdError(false)
              }}
              onKeyDown={(e) => e.key === 'Enter' && handleMsgIdSearch()}
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
        <details className="relative shrink-0">
          <summary className="cursor-pointer rounded-md border border-[var(--border)] px-3 py-2 text-xs text-[var(--text-muted)]">
            {t('diagnosis.filters')}
            {diagnosticFilterActive && <span className="ml-2 text-[var(--accent)]">{t('diagnosis.filterActive')}</span>}
          </summary>
          <div className="absolute right-0 top-full z-20 mt-2 flex max-h-[65vh] w-[700px] flex-wrap items-end gap-3 overflow-auto rounded-md border border-[var(--border)] bg-[var(--bg-card)] p-3 shadow-lg">
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
            <SegmentedControl
              options={filterOptions}
              value={mode}
              onChange={(next) => {
                setMode(next)
                setPage(1)
              }}
              ariaLabel={t('prediction.aboveFilter')}
            />
            {onDiagnosisFiltersChange && (
              <>
                <Select
                  label={t('diagnosis.type')}
                  value={diagnosisFilters.failure_type ?? ''}
                  options={[
                    { value: '', label: t('diagnosis.all') },
                    ...[...new Set(diagnosis.data?.report?.patterns.map((p) => p.failure_type) ?? [])].map((value) => ({
                      value,
                      label: t(`diagnosis.issue.${value}`),
                    })),
                  ]}
                  onChange={(failure_type) =>
                    onDiagnosisFiltersChange({
                      ...diagnosisFilters,
                      failure_type,
                      subset_name: selectedSubset,
                    })
                  }
                />
                <Select
                  label={t('diagnosis.observedComponent')}
                  value={diagnosisFilters.attribution ?? ''}
                  options={[
                    { value: '', label: t('diagnosis.all') },
                    ...['model', 'guard', 'environment', 'benchmark', 'unknown'].map((value) => ({
                      value,
                      label: t(`diagnosis.${value === 'unknown' ? 'unknownComponent' : value}`),
                    })),
                  ]}
                  onChange={(attribution) =>
                    onDiagnosisFiltersChange({
                      ...diagnosisFilters,
                      attribution,
                      subset_name: selectedSubset,
                    })
                  }
                />
                <Select
                  label={t('diagnosis.category')}
                  value={diagnosisFilters.finding_category ?? ''}
                  options={[
                    { value: '', label: t('diagnosis.all') },
                    ...[
                      'tool_failure',
                      'repetition',
                      'termination',
                      'refusal',
                      'risky_command',
                      'attack_goal_action',
                      'tainted_side_effect',
                      'side_effect',
                    ].map((value) => ({ value, label: t(`diagnosis.finding.${value}`) })),
                  ]}
                  onChange={(finding_category) =>
                    onDiagnosisFiltersChange({
                      ...diagnosisFilters,
                      finding_category,
                      subset_name: selectedSubset,
                    })
                  }
                />
                {diagnosticFilterActive && (
                  <button
                    type="button"
                    className="min-h-11 text-sm text-[var(--accent)]"
                    onClick={() => onDiagnosisFiltersChange({ subset_name: selectedSubset })}
                  >
                    {t('diagnosis.reset')}
                  </button>
                )}
              </>
            )}
          </div>
        </details>
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
          {/* Content area */}
          {row && (
            <div className="transition-all duration-200">
              {highlightMsgId && (
                <p className="type-body-xs mb-2 text-[var(--text-muted)]" role="status">
                  {t('prediction.messageLocated', { id: highlightMsgId })}
                </p>
              )}
              <ChatView
                key={`${rowScope}/${focusStep ?? ''}`}
                prediction={row}
                threshold={threshold}
                highlightMsgId={highlightMsgId}
                metrics={report?.metrics}
                diagnosis={diagnosisRow}
                focusEvidence={diagnosticFilterActive || focusStep !== undefined}
                focusStep={focusStep}
              />
            </div>
          )}
        </>
      )}

      {!loading && predictions.length > 0 && filtered.length === 0 && !loadError && (
        <div role="status" className="space-y-2 py-4 text-[var(--text-muted)]">
          <p>{t('diagnosis.noMatch')}</p>
          <button
            type="button"
            className="min-h-11 text-sm text-[var(--accent)]"
            onClick={() => {
              setMode('All')
              setPage(1)
              onDiagnosisFiltersChange?.({ subset_name: selectedSubset })
            }}
          >
            {t('reportDetail.clearFilters')}
          </button>
        </div>
      )}

      {!loading && predictions.length === 0 && selectedSubset && (
        <EmptyStateSystem reason="no-data" context={{ view: 'evaluations' }} />
      )}
    </div>
  )
}
