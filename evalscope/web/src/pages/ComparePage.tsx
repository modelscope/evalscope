import { useCallback, useEffect, useMemo, useState, type CSSProperties, type ReactNode } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import { getPredictions, getCompareChartUrl } from '@/api/reports'
import type { ReportData, PredictionRow } from '@/api/types'
import { parseReportRef } from '@/domain/report/reportRef'
import {
  buildDisplayLabels,
  compatibilityReason,
  getDisplayNames,
  MAX_COMPARE_SLOTS,
  togglePredictionSelection,
} from '@/domain/compare/compareModel'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Card from '@/components/ui/Card'
import Tabs from '@/components/ui/Tabs'
import { scoreBg, scoreColor } from '@/utils/colorScale'
import { formatDifference, formatMetric, getBoundedQualityRatio } from '@/domain/metric'
import {
  RATIO_PERCENT_SEMANTICS,
  datasetLabel,
  directionHintKey,
  primaryMetricOf,
} from '@/domain/report/primaryMetrics'
import type { MetricSemantics } from '@/domain/metric'
import Button from '@/components/ui/Button'
import Select from '@/components/ui/Select'
import SelectionCheckbox from '@/components/ui/SelectionCheckbox'
import Skeleton from '@/components/ui/Skeleton'
import Badge from '@/components/ui/Badge'
import ScoreBadge from '@/components/ui/ScoreBadge'
import Eyebrow from '@/components/ui/Eyebrow'
import { cn } from '@/lib/utils'
import PlotlyChart from '@/components/charts/PlotlyChart'
import ChatView from '@/components/single/ChatView'
import EmptyStateSystem from '@/components/common/EmptyStateSystem'
import ErrorAlert from '@/components/ui/ErrorAlert'
import { Plus, ChevronLeft, ChevronRight, AlertCircle, ArrowUp, ArrowDown, Search, X } from 'lucide-react'

// ------------------------------------------------------------------ //
// Types                                                               //
// ------------------------------------------------------------------ //

interface MergedPrediction {
  Index: string
  Input: string
  Gold: string
  models: Record<string, PredictionRow>
}

type PerModelFilter = 'any' | 'above' | 'below'
type Translate = (path: string, vars?: Record<string, string | number>) => string

// Distinct accent color palette for each model column (up to MAX_COMPARE_SLOTS)
// DESIGN.md §Compare Slots: only 3 brand-color slots exist.
// Do NOT add a 4th entry to this palette — extra models must collapse to a
// numbered legend instead. Iteration paths use `MODEL_PALETTE[idx] ?? MODEL_PALETTE[0]`
// as a safety fallback in case slicing is bypassed upstream.
const MODEL_PALETTE = [
  {
    dot: 'var(--compare-0-dot)',
    border: 'var(--compare-0-border)',
    bg: 'var(--compare-0-bg)',
    headerBg: 'var(--compare-0-bg-header)',
  },
  {
    dot: 'var(--compare-1-dot)',
    border: 'var(--compare-1-border)',
    bg: 'var(--compare-1-bg)',
    headerBg: 'var(--compare-1-bg-header)',
  },
  {
    dot: 'var(--compare-2-dot)',
    border: 'var(--compare-2-border)',
    bg: 'var(--compare-2-bg)',
    headerBg: 'var(--compare-2-bg-header)',
  },
]

// ------------------------------------------------------------------ //
// Main Component                                                      //
// ------------------------------------------------------------------ //

export default function ComparePage() {
  const { t } = useLocale()
  const qp = useQueryParams()
  const { rootPath: ctxRootPath, setRootPath, loadMultiReports, loading, reportCache } = useReports()

  // Score comparison consumes the complete URL selection. Prediction comparison
  // derives its own bounded subset below.
  const rootPath = qp.get('root_path') || ctxRootPath
  // Key the memo on a stable string: `useQueryParams` returns a fresh object each render, so a raw
  // `[qp]` dependency would rebuild the array every render and re-fire the load effect in a loop.
  const reportKey = qp.getList('report').filter(Boolean).join(';')
  const reportNames = useMemo(
    () => (reportKey ? reportKey.split(';') : []),
    [reportKey],
  )

  // State
  const [reports, setReports] = useState<ReportData[]>([])
  const [activeTab, setActiveTab] = useState<'score' | 'prediction'>('score')
  const [dataLoaded, setDataLoaded] = useState(false)
  const [scoreLoadError, setScoreLoadError] = useState('')
  const [scoreReloadToken, setScoreReloadToken] = useState(0)
  const [addInput, setAddInput] = useState('')
  const [showAddInput, setShowAddInput] = useState(false)
  const [reportSearch, setReportSearch] = useState('')

  // Prediction tab state
  const [predictionSelection, setPredictionSelection] = useState<string[]>(
    () => reportNames.slice(0, MAX_COMPARE_SLOTS),
  )
  const predictionReportNames = useMemo(
    () => predictionSelection.filter((name) => reportNames.includes(name)),
    [predictionSelection, reportNames],
  )
  const [selectedDs, setSelectedDs] = useState('')
  const [selectedSubset, setSelectedSubset] = useState('')
  const [mergedPredictions, setMergedPredictions] = useState<MergedPrediction[]>([])
  const [perModelFilter, setPerModelFilter] = useState<Record<string, PerModelFilter>>({})
  const [threshold, setThreshold] = useState(0.99)
  const [page, setPage] = useState(1)
  const [predictionsLoading, setPredictionsLoading] = useState(false)
  const [predictionsError, setPredictionsError] = useState('')
  const [predictionsReloadToken, setPredictionsReloadToken] = useState(0)

  // Sync root path
  useEffect(() => {
    if (rootPath && rootPath !== ctxRootPath) setRootPath(rootPath)
  }, [rootPath, ctxRootPath, setRootPath])

  // Load score data and per-report cache (datasets/subsets) in one cache-aware pass
  useEffect(() => {
    if (reportNames.length < 2) return
    const controller = new AbortController()
    const load = async () => {
      setDataLoaded(false)
      setScoreLoadError('')
      try {
        const list = await loadMultiReports(reportNames, controller.signal)
        if (!controller.signal.aborted) {
          setReports(list)
          setDataLoaded(true)
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          setScoreLoadError(error instanceof Error ? error.message : t('common.loadError'))
          setDataLoaded(true)
        }
      }
    }
    load()
    return () => controller.abort()
  }, [reportNames, loadMultiReports, scoreReloadToken, t])

  // ------------------------------------------------------------------ //
  // Score Tab Data                                                      //
  // ------------------------------------------------------------------ //

  const { scoreTableData, scoreTableColumns, displayNames, scoreSemantics } = useMemo(() => {
    const displayNames = getDisplayNames(reportNames)
    if (!reports.length) {
      return { scoreTableData: [], scoreTableColumns: [], displayNames, scoreSemantics: {} }
    }

    const byReport: Record<string, Record<string, number>> = {}
    const labelsByDataset: Record<string, string> = {}
    //  Each dataset's primary metric decides how its row is formatted and which end is "best".
    const semanticsByDataset: Record<string, MetricSemantics | undefined> = {}
    for (const r of reports) {
      const key = (r as ReportData & { _reportRef?: string })._reportRef ?? r.model_name
      if (!byReport[key]) byReport[key] = {}
      const primary = primaryMetricOf(r)
      if (!primary) continue
      byReport[key][r.dataset_name] = primary.score
      labelsByDataset[r.dataset_name] = datasetLabel(r)
      if (primary?.semantics) {
        semanticsByDataset[r.dataset_name] = primary.semantics
      }
    }

    const reportKeys = reportNames.filter((n) => byReport[n])
    const dsLists = reportKeys.map((k) => new Set(Object.keys(byReport[k])))
    const common = dsLists.length
      ? [...dsLists.reduce((a, b) => new Set([...a].filter((x) => b.has(x))))]
      : []
    common.sort()

    const rows: Record<string, unknown>[] = common.map((ds) => {
      const row: Record<string, unknown> = { dataset: labelsByDataset[ds] || ds, dataset_id: ds }
      const scores = reportKeys.map((k) => byReport[k][ds] ?? 0)
      // "Best" follows the metric's direction: a low WER wins, a high accuracy wins.
      const lowerIsBetter = semanticsByDataset[ds]?.direction === 'lower_is_better'
      const bestScore = lowerIsBetter ? Math.min(...scores) : Math.max(...scores)
      reportKeys.forEach((k, i) => {
        row[k] = scores[i]
        row[`${k}_best`] = scores[i] === bestScore && scores.length > 1
      })
      return row
    })

    const columns = [
      { key: 'dataset', label: t('compare.dataset') },
      ...reportKeys.map((k) => ({ key: k, label: displayNames[k] })),
    ]

    return { scoreTableData: rows, scoreTableColumns: columns, displayNames, scoreSemantics: semanticsByDataset }
  }, [reports, reportNames, t])

  // ------------------------------------------------------------------ //
  // Prediction Tab Data                                                 //
  // ------------------------------------------------------------------ //

  const predCommonDatasets = useMemo(() => {
    if (predictionReportNames.length < 2) return []
    const dsLists = predictionReportNames.map((name) => {
      const cached = reportCache[name]
      return cached ? new Set(cached.datasets) : new Set<string>()
    })
    if (dsLists.some((s) => s.size === 0)) return []
    return [...dsLists.reduce((a, b) => new Set([...a].filter((x) => b.has(x))))]
  }, [predictionReportNames, reportCache])

  const predictionDatasetLabels = useMemo(() => {
    const labels: Record<string, string> = {}
    for (const cached of Object.values(reportCache)) {
      for (const report of cached.report_list) labels[report.dataset_name] = datasetLabel(report)
    }
    return labels
  }, [reportCache])

  useEffect(() => {
    const applyDefault = () => {
      if (activeTab === 'prediction' && predCommonDatasets.length > 0 && !selectedDs) {
        setSelectedDs(predCommonDatasets[0])
      }
    }
    applyDefault()
  }, [activeTab, predCommonDatasets, selectedDs])

  const subsets = useMemo(() => {
    if (!selectedDs || predictionReportNames.length < 1) return []
    const cached = reportCache[predictionReportNames[0]]
    if (!cached) return []
    const report = cached.report_list.find((r) => r.dataset_name === selectedDs)
    if (!report) return []
    const subs: string[] = []
    for (const m of report.metrics) {
      for (const c of m.categories) {
        if (c.name.length && c.name.join('/') === '-') continue
        for (const s of c.subsets) {
          if (s.name !== 'overall_score' && !subs.includes(s.name)) subs.push(s.name)
        }
      }
    }
    return subs
  }, [selectedDs, predictionReportNames, reportCache])

  useEffect(() => {
    const applyDefault = () => {
      if (subsets.length > 0 && !selectedSubset) setSelectedSubset(subsets[0])
    }
    applyDefault()
  }, [subsets, selectedSubset])

  useEffect(() => {
    if (!selectedDs || !selectedSubset || predictionReportNames.length < 2) return
    const controller = new AbortController()
    const loadPredictions = async () => {
      setPredictionsLoading(true)
      setPredictionsError('')
      try {
        const results = await Promise.all(
          predictionReportNames.map(
            (name) => getPredictions(rootPath, name, selectedDs, selectedSubset, controller.signal),
          ),
        )
        const indexMap = new Map<string, MergedPrediction>()
        results.forEach((res, i) => {
          const modelName = predictionReportNames[i]
          for (const p of res.predictions) {
            if (!indexMap.has(p.Index)) {
              indexMap.set(p.Index, { Index: p.Index, Input: p.Input, Gold: p.Gold, models: {} })
            }
            indexMap.get(p.Index)!.models[modelName] = p
          }
        })
        const merged = [...indexMap.values()].filter((row) =>
          predictionReportNames.every((m) => row.models[m]),
        )
        if (!controller.signal.aborted) {
          setMergedPredictions(merged)
          setPage(1)
        }
      } catch (e) {
        console.error('Failed to load predictions:', e)
        if (!controller.signal.aborted) {
          setPredictionsError(e instanceof Error ? e.message : t('common.loadError'))
        }
      } finally {
        if (!controller.signal.aborted) setPredictionsLoading(false)
      }
    }
    loadPredictions()
    return () => controller.abort()
  }, [rootPath, predictionReportNames, selectedDs, selectedSubset, predictionsReloadToken, t])

  // Filtered predictions using per-model constraints
  const filtered = useMemo(() => {
    return mergedPredictions.filter((row) =>
      predictionReportNames.every((name) => {
        const f = perModelFilter[name] ?? 'any'
        if (f === 'any') return true
        // The threshold is a view-only filter (above/below), not a pass/fail
        // verdict.
        const above = (row.models[name]?.NScore ?? 0) >= threshold
        return f === 'above' ? above : !above
      }),
    )
  }, [mergedPredictions, perModelFilter, threshold, predictionReportNames])

  // Fraction of samples per model that sit above the view filter (full set).
  const aboveRates = useMemo(() => {
    if (!mergedPredictions.length) return {} as Record<string, number>
    const rates: Record<string, number> = {}
    for (const name of predictionReportNames) {
      const above = mergedPredictions.filter((r) => (r.models[name]?.NScore ?? 0) >= threshold).length
      rates[name] = above / mergedPredictions.length
    }
    return rates
  }, [mergedPredictions, predictionReportNames, threshold])

  const totalPages = filtered.length
  const currentRow = filtered.length > 0 ? filtered[Math.min(page - 1, filtered.length - 1)] : null

  // Datasets each selected report covers, keyed by report reference. Sourced from the loaded
  // reports (one report per dataset) rather than from the reference, which carries no datasets.
  const datasetsByRef = useMemo(() => {
    const map: Record<string, string[]> = {}
    for (const r of reports) {
      const key = (r as ReportData & { _reportRef?: string })._reportRef ?? r.model_name
      if (!map[key]) map[key] = []
      if (r.dataset_name && !map[key].includes(r.dataset_name)) map[key].push(r.dataset_name)
    }
    return map
  }, [reports])

  // Meaningful model + dataset display label per run, used for table headers and
  // column identifiers instead of the raw reference.
  const displayLabels = useMemo(
    () => buildDisplayLabels(reportNames, datasetsByRef),
    [reportNames, datasetsByRef],
  )

  const activeReportNames = activeTab === 'score' ? reportNames : predictionReportNames

  // Incompatibility follows the active comparison mode: all reports for score,
  // and only the user-selected columns for prediction.
  const incompatibilityReason = useMemo(() => {
    if (activeReportNames.length < 2) return null
    return compatibilityReason(activeReportNames.map((ref) => datasetsByRef[ref] ?? []))
  }, [activeReportNames, datasetsByRef])

  // ------------------------------------------------------------------ //
  // URL manipulation                                                    //
  // ------------------------------------------------------------------ //

  const removeReport = useCallback((name: string) => {
    qp.setList('report', reportNames.filter((n) => n !== name))
  }, [reportNames, qp])

  const addReport = useCallback(() => {
    if (!addInput.trim() || reportNames.includes(addInput.trim())) return
    qp.setList('report', [...reportNames, addInput.trim()])
    setAddInput('')
    setShowAddInput(false)
  }, [addInput, reportNames, qp])

  const togglePredictionReport = useCallback((name: string) => {
    setPredictionSelection((current) => togglePredictionSelection(
      current.filter((selectedName) => reportNames.includes(selectedName)),
      name,
    ))
  }, [reportNames])

  // ------------------------------------------------------------------ //
  // Render                                                              //
  // ------------------------------------------------------------------ //

  if (reportNames.length < 2) {
    return (
      <div className="page-enter">
        <Breadcrumb items={[{ label: t('reports.title'), href: '/reports' }, { label: t('compare.title') }]} />
        <div className="flex flex-col items-center justify-center gap-4 py-20">
          {/* text-dim allowed: empty-state alert icon (DESIGN.md §Text) */}
          <AlertCircle size={48} className="text-[var(--text-dim)]" />
          <p className="text-[var(--text-muted)] text-lg">{t('compare.needTwo')}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="page-enter mx-auto flex w-full max-w-[1600px] flex-col gap-5">
      <Breadcrumb items={[{ label: t('reports.title'), href: '/reports' }, { label: t('compare.title') }]} />

      <div className="grid items-start gap-5 lg:grid-cols-[300px_minmax(0,1fr)]">
        <CompareReportRail
          activeTab={activeTab}
          reportNames={reportNames}
          predictionReportNames={predictionReportNames}
          displayLabels={displayLabels}
          reportSearch={reportSearch}
          setReportSearch={setReportSearch}
          onRemoveReport={removeReport}
          onTogglePredictionReport={togglePredictionReport}
          addInput={addInput}
          setAddInput={setAddInput}
          showAddInput={showAddInput}
          setShowAddInput={setShowAddInput}
          onAddReport={addReport}
          t={t}
        />

        <div className="flex min-w-0 flex-col gap-4">
          {/* Incompatible runs notice — selection is preserved */}
          {incompatibilityReason && (
            <div
              role="status"
              className="flex items-start gap-3 rounded-[var(--radius)] border border-[var(--warning-border)] bg-[var(--warning-bg)] px-4 py-3"
            >
              <AlertCircle size={18} className="mt-0.5 shrink-0 text-[var(--warning-color)]" />
              <div className="flex flex-col gap-0.5">
                <p className="text-sm font-medium text-[var(--text)]">{t('compare.incompatible')}</p>
                <p className="text-xs text-[var(--text-muted)]">
                  {t(incompatibilityReason)} · {t('compare.incompatibleHint')}
                </p>
              </div>
            </div>
          )}

          {scoreLoadError && (
            <ErrorAlert className="flex items-center justify-between gap-3">
              <span className="type-body-sm break-words">{scoreLoadError}</span>
              <Button size="sm" variant="outline" onClick={() => setScoreReloadToken((value) => value + 1)}>
                {t('common.retry')}
              </Button>
            </ErrorAlert>
          )}

          <Tabs
            className="mb-4 w-fit"
            tabs={[
              {
                key: 'score',
                label: `${t('compare.scoreComparison')} · ${t('compare.reportCount', { n: reportNames.length })}`,
                panelId: 'compare-score-panel',
              },
              {
                key: 'prediction',
                label: `${t('compare.predictionComparison')} · ${t('compare.chooseUpToThree')}`,
                panelId: 'compare-prediction-panel',
              },
            ]}
            activeKey={activeTab}
            onChange={(k) => setActiveTab(k as 'score' | 'prediction')}
            panels={{
              'compare-score-panel': loading && !dataLoaded ? (
                <div className="flex flex-col gap-4">
                  <Skeleton height={450} />
                  <Skeleton height={300} />
                </div>
              ) : (
                <ScoreTab
                  rootPath={rootPath}
                  reportNames={reportNames}
                  scoreTableColumns={scoreTableColumns}
                  scoreTableData={scoreTableData}
                  scoreSemantics={scoreSemantics}
                  displayNames={displayNames}
                  displayLabels={displayLabels}
                  t={t}
                />
              ),
              'compare-prediction-panel': loading && !dataLoaded ? (
                <div className="flex flex-col gap-4">
                  <Skeleton height={450} />
                  <Skeleton height={300} />
                </div>
              ) : predictionReportNames.length < 2 ? (
                <Card>
                  <EmptyStateSystem
                    reason="no-match"
                    context={{ view: 'compare' }}
                    hint={t('compare.chooseTwoForPrediction')}
                  />
                </Card>
              ) : (
                <PredictionTab
                  reportNames={predictionReportNames}
                  displayNames={displayNames}
                  displayLabels={displayLabels}
                  predCommonDatasets={predCommonDatasets}
                  datasetLabels={predictionDatasetLabels}
                  selectedDs={selectedDs}
                  setSelectedDs={setSelectedDs}
                  subsets={subsets}
                  selectedSubset={selectedSubset}
                  setSelectedSubset={setSelectedSubset}
                  perModelFilter={perModelFilter}
                  setPerModelFilter={setPerModelFilter}
                  threshold={threshold}
                  setThreshold={setThreshold}
                  aboveRates={aboveRates}
                  mergedPredictions={mergedPredictions}
                  filtered={filtered}
                  currentRow={currentRow}
                  page={page}
                  setPage={setPage}
                  totalPages={totalPages}
                  predictionsLoading={predictionsLoading}
                  predictionsError={predictionsError}
                  onRetryPredictions={() => setPredictionsReloadToken((value) => value + 1)}
                  t={t}
                />
              ),
            }}
          />
        </div>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ //
// Report Selection Rail                                               //
// ------------------------------------------------------------------ //

function CompareReportRail({
  activeTab,
  reportNames,
  predictionReportNames,
  displayLabels,
  reportSearch,
  setReportSearch,
  onRemoveReport,
  onTogglePredictionReport,
  addInput,
  setAddInput,
  showAddInput,
  setShowAddInput,
  onAddReport,
  t,
}: {
  activeTab: 'score' | 'prediction'
  reportNames: string[]
  predictionReportNames: string[]
  displayLabels: Record<string, string>
  reportSearch: string
  setReportSearch: (value: string) => void
  onRemoveReport: (name: string) => void
  onTogglePredictionReport: (name: string) => void
  addInput: string
  setAddInput: (value: string) => void
  showAddInput: boolean
  setShowAddInput: (value: boolean) => void
  onAddReport: () => void
  t: Translate
}) {
  const normalizedSearch = reportSearch.trim().toLowerCase()
  const visibleReports = reportNames.filter((name) => {
    if (!normalizedSearch) return true
    return (displayLabels[name] ?? name).toLowerCase().includes(normalizedSearch)
  })
  const isPrediction = activeTab === 'prediction'

  return (
    <aside className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] lg:sticky lg:top-20">
      <div className="border-b border-[var(--border)] px-4 py-3">
        <div className="flex items-center justify-between gap-3">
          <h2 className="type-label-xs">{t('compare.selectedReports')}</h2>
          <span className="text-xs font-semibold tabular-nums text-[var(--accent)]">
            {isPrediction
              ? `${predictionReportNames.length} / ${MAX_COMPARE_SLOTS}`
              : t('compare.reportCount', { n: reportNames.length })}
          </span>
        </div>
        <p className="mt-1.5 text-xs leading-5 text-[var(--text-muted)]">
          {isPrediction ? t('compare.predictionSelectionHint') : t('compare.scoreSelectionHint')}
        </p>
      </div>

      <div className="p-3">
        <label className="relative block">
          <span className="sr-only">{t('compare.searchReports')}</span>
          <Search
            size={15}
            className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-dim)]"
          />
          <input
            type="search"
            value={reportSearch}
            onChange={(event) => setReportSearch(event.target.value)}
            placeholder={t('compare.searchReports')}
            className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-deep)] py-2 pl-9 pr-3 text-sm text-[var(--text)] placeholder:text-[var(--text-dim)] focus:border-[var(--accent)] focus:outline-none"
          />
        </label>
      </div>

      <div className="max-h-[360px] overflow-y-auto border-y border-[var(--border)] lg:max-h-[calc(100vh-360px)] lg:min-h-[280px]">
        {visibleReports.map((name) => {
          const checked = isPrediction ? predictionReportNames.includes(name) : true
          const disabled = isPrediction
            ? !checked && predictionReportNames.length >= MAX_COMPARE_SLOTS
            : reportNames.length <= 2
          const label = displayLabels[name] ?? (parseReportRef(name).modelId || name)
          const reportIndex = reportNames.indexOf(name)
          const slotIndex = isPrediction ? predictionReportNames.indexOf(name) : reportIndex
          const palette = slotIndex >= 0 && slotIndex < MODEL_PALETTE.length
            ? MODEL_PALETTE[slotIndex]
            : null
          return (
            <SelectionCheckbox
              key={name}
              checked={checked}
              disabled={disabled}
              label={`${isPrediction ? t('compare.predictionComparison') : t('compare.scoreComparison')}: ${label}`}
              onClick={() => {
                if (isPrediction) onTogglePredictionReport(name)
                else onRemoveReport(name)
              }}
              className="w-full justify-start gap-2.5 border-b border-[var(--border)] px-3 py-2 text-left last:border-b-0 hover:bg-[var(--bg-card2)]"
            >
              <span className="flex min-w-0 items-start gap-2">
                {palette ? (
                  <span
                    aria-hidden="true"
                    className="mt-1.5 inline-block h-2 w-2 shrink-0 rounded-full"
                    style={{ backgroundColor: palette.dot }}
                  />
                ) : isPrediction ? (
                  <span
                    aria-hidden="true"
                    className="mt-1.5 inline-block h-2 w-2 shrink-0 rounded-full border border-[var(--border-strong)]"
                  />
                ) : (
                  <span className="mt-0.5 min-w-5 shrink-0 rounded-[var(--radius-xs)] bg-[var(--bg-deep)] px-1 py-0.5 text-center text-[10px] tabular-nums text-[var(--text-dim)]">
                    {reportIndex + 1}
                  </span>
                )}
                <span className="min-w-0 break-words text-xs font-medium leading-5 text-[var(--text)]" title={label}>
                  {label}
                </span>
              </span>
            </SelectionCheckbox>
          )
        })}
      </div>

      <div className="p-3">
        {showAddInput ? (
          <div className="flex flex-col gap-2">
            <input
              type="text"
              value={addInput}
              onChange={(event) => setAddInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') onAddReport()
                if (event.key === 'Escape') setShowAddInput(false)
              }}
              placeholder={t('compare.reportNamePlaceholder')}
              autoFocus
              className="w-full rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-deep)] px-3 py-2 text-sm text-[var(--text)] placeholder:text-[var(--text-dim)] focus:border-[var(--accent)] focus:outline-none"
            />
            <div className="flex items-center gap-2">
              <Button size="sm" onClick={onAddReport}>{t('compare.addModel')}</Button>
              <Button
                size="sm"
                variant="ghost"
                aria-label={t('compare.cancelAdd')}
                onClick={() => setShowAddInput(false)}
              >
                <X size={14} />
              </Button>
            </div>
          </div>
        ) : (
          <Button size="sm" variant="outline" className="w-full" onClick={() => setShowAddInput(true)}>
            <Plus size={14} />
            {t('compare.addModel')}
          </Button>
        )}
      </div>
    </aside>
  )
}

// ------------------------------------------------------------------ //
// Score Comparison Tab                                                //
// ------------------------------------------------------------------ //

function comparisonDeltaBackground(delta: number, maxAbsoluteDelta: number): string {
  if (delta === 0 || maxAbsoluteDelta === 0) return 'var(--bg-deep)'
  const intensity = Math.min(1, Math.abs(delta) / maxAbsoluteDelta)
  const weight = Math.round((0.06 + intensity * 0.24) * 100)
  const semanticColor = delta > 0 ? 'var(--success)' : 'var(--danger)'
  return `color-mix(in srgb, ${semanticColor} ${weight}%, var(--bg-deep))`
}

function signedDifference(delta: number, semantics: MetricSemantics | undefined): string {
  const formatted = formatDifference(delta, semantics).primary
  return delta > 0 ? `+${formatted}` : formatted
}

function ScoreTab({
  rootPath,
  reportNames,
  scoreTableColumns,
  scoreTableData,
  scoreSemantics,
  displayNames,
  displayLabels,
  t,
}: {
  rootPath: string
  reportNames: string[]
  scoreTableColumns: { key: string; label: string }[]
  scoreTableData: Record<string, unknown>[]
  /** Dataset name -> semantics of that dataset's primary metric. */
  scoreSemantics: Record<string, MetricSemantics | undefined>
  displayNames: Record<string, string>
  displayLabels: Record<string, string>
  t: Translate
}) {
  const reportKeys = useMemo(() => scoreTableColumns.slice(1).map((c) => c.key), [scoreTableColumns])
  const dataRows = scoreTableData
  const [comparisonMode, setComparisonMode] = useState<'absolute' | 'baseline'>('baseline')
  const [selectedBaselineReport, setSelectedBaselineReport] = useState(reportKeys[0] ?? '')
  const baselineReport = reportKeys.includes(selectedBaselineReport)
    ? selectedBaselineReport
    : reportKeys[0] ?? ''

  const deltaRanges = useMemo(() => {
    const ranges: Record<string, number> = {}
    for (const row of dataRows) {
      const key = String(row.dataset_id)
      const baseline = Number(row[baselineReport])
      ranges[key] = Math.max(
        0,
        ...reportKeys.map((report) => Math.abs(Number(row[report]) - baseline)),
      )
    }
    return ranges
  }, [baselineReport, dataRows, reportKeys])

  const renderScoreCell = (
    score: number | null,
    baselineScore: number | null,
    semantics: MetricSemantics | undefined,
    isBest: boolean,
    isBaseline: boolean,
  ): ReactNode => {
    if (score == null) {
      return <span className="text-[var(--text-dim)]">—</span>
    }
    const delta = baselineScore == null ? 0 : score - baselineScore

    return (
      <div
        className="flex min-h-12 flex-col items-center justify-center px-3 py-1 font-mono"
      >
        <span className="text-xs font-semibold text-[var(--text)]">
          {comparisonMode === 'absolute' && isBest && (
            <span className="mr-1 inline-block h-1.5 w-1.5 rounded-full bg-current align-middle opacity-80" />
          )}
          {formatMetric(score, semantics).primary}
        </span>
        <span
          aria-hidden={comparisonMode !== 'baseline'}
          className={cn(
            'mt-0.5 h-4 text-[10px] font-semibold transition-opacity duration-[var(--transition)]',
            comparisonMode === 'baseline' ? 'opacity-100' : 'opacity-0',
            isBaseline || delta === 0
              ? 'text-[var(--text-dim)]'
              : delta > 0
                ? 'text-[var(--success)]'
                : 'text-[var(--danger)]',
          )}
        >
          {isBaseline ? t('compare.baseline') : signedDifference(delta, semantics)}
        </span>
      </div>
    )
  }

  const scoreCellStyle = (
    score: number | null,
    baselineScore: number | null,
    semantics: MetricSemantics | undefined,
    rangeKey: string,
  ): CSSProperties => {
    if (score == null) return { backgroundColor: 'var(--bg-deep)' }
    const ratio = getBoundedQualityRatio(score, semantics)
    const delta = baselineScore == null ? 0 : score - baselineScore
    if (comparisonMode === 'baseline') {
      return { backgroundColor: comparisonDeltaBackground(delta, deltaRanges[rangeKey] ?? 0) }
    }
    if (ratio == null) return { backgroundColor: 'var(--bg-deep)', color: 'var(--text)' }
    return { backgroundColor: scoreBg(ratio, 0.18), color: scoreColor(ratio) }
  }

  return (
    <div className="flex flex-col gap-6">
      <PlotlyChart
        src={getCompareChartUrl(rootPath, reportNames, 'radar')}
        fallbackTable={{
          columns: scoreTableColumns.map((column) => column.key),
          rows: scoreTableData,
          scoreColumns: reportKeys,
          semantics: undefined,
        }}
        height={420}
        title={t('multi.modelRadar')}
      />

      <div className="overflow-hidden rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] shadow-[var(--shadow-sm)]">
        <div className="border-b border-[var(--border-strong)] px-5 py-4">
          <h3 className="type-label-xs">{t('multi.modelScores')}</h3>
          <div className="mt-3 flex flex-wrap items-end gap-x-8 gap-y-3">
            <div className="flex flex-wrap items-center gap-4">
              <span className="text-sm font-medium text-[var(--text-muted)]">
                {t('compare.baselineMode')}
              </span>
              <div className="grid min-w-[320px] grid-cols-2 overflow-hidden rounded-[var(--radius-sm)] border border-[var(--border-strong)]">
                {([
                  ['absolute', t('compare.absoluteScores')],
                  ['baseline', t('compare.vsBaseline')],
                ] as const).map(([mode, label]) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setComparisonMode(mode)}
                    className={cn(
                      'px-4 py-2 text-sm font-medium transition-colors duration-[var(--transition)]',
                      comparisonMode === mode
                        ? 'bg-[var(--accent)] text-[var(--text-on-filled)]'
                        : 'bg-[var(--bg-deep)] text-[var(--text-muted)] hover:text-[var(--text)]',
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <div
              className={cn(
                'w-full min-w-0 max-w-[460px] transition-opacity duration-[var(--transition)] sm:w-[38vw]',
                comparisonMode === 'baseline' ? 'opacity-100' : 'opacity-40',
              )}
            >
              <Select
                disabled={comparisonMode !== 'baseline'}
                label={t('compare.baseline')}
                options={reportKeys.map((report) => ({
                  value: report,
                  label: displayLabels[report] ?? displayNames[report] ?? report,
                }))}
                value={baselineReport}
                onChange={setSelectedBaselineReport}
              />
            </div>
          </div>
        </div>

        {scoreTableData.length === 0 ? (
          <EmptyStateSystem reason="no-data" context={{ view: 'compare' }} />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full table-fixed border-collapse text-sm">
              <colgroup>
                <col style={{ width: 'clamp(280px, 34%, 420px)' }} />
                {dataRows.map((row) => <col key={String(row.dataset_id)} />)}
              </colgroup>
              <thead className="bg-[var(--bg-card2)]">
                <tr className="border-b border-[var(--border-strong)]">
                  <th className="sticky left-0 z-10 min-w-[280px] max-w-[420px] border-r border-[var(--border-strong)] bg-[var(--bg-card2)] px-4 py-2.5 text-left type-table-xs">
                    {t('compare.model')}
                  </th>
                  {dataRows.map((row) => {
                    const datasetId = String(row.dataset_id)
                    const semantics = scoreSemantics[datasetId]
                    const hintKey = directionHintKey(semantics)
                    return (
                      <th
                        key={datasetId}
                        title={datasetId}
                        className="min-w-[120px] border-l border-[var(--border-strong)] px-3 py-2 text-center type-table-xs !normal-case whitespace-nowrap first:border-l-0"
                      >
                        <span className="flex flex-col items-center justify-center gap-0.5">
                          <span>{String(row.dataset)}</span>
                          {hintKey && (
                            <span
                              aria-label={t(hintKey)}
                              title={t(hintKey)}
                              className="text-[var(--text-dim)]"
                            >
                              {semantics?.direction === 'lower_is_better'
                                ? <ArrowDown aria-hidden="true" size={11} strokeWidth={2.5} />
                                : <ArrowUp aria-hidden="true" size={11} strokeWidth={2.5} />}
                            </span>
                          )}
                        </span>
                      </th>
                    )
                  })}
                </tr>
              </thead>
              <tbody>
                {reportKeys.map((rk, rkIdx) => {
                  const isBaseline = comparisonMode === 'baseline' && rk === baselineReport
                  const modelLabel = displayLabels[rk] ?? displayNames[rk] ?? rk
                  return (
                  <tr key={rk} className="border-b border-[var(--border-strong)] last:border-b-0 hover:bg-[var(--bg-card2)] transition-colors">
                    <td className="sticky left-0 z-10 min-w-[280px] max-w-[420px] border-r border-[var(--border-strong)] bg-[var(--bg-card)] px-4 py-1 text-xs font-medium">
                      <div className="flex min-h-12 items-center gap-2">
                        {rkIdx < MODEL_PALETTE.length ? (
                          <span className="mt-1.5 inline-block h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: MODEL_PALETTE[rkIdx].dot }} />
                        ) : (
                          <span className="min-w-5 shrink-0 rounded-[var(--radius-xs)] bg-[var(--bg-deep)] px-1 py-0.5 text-center text-[10px] tabular-nums text-[var(--text-dim)]">
                            {rkIdx + 1}
                          </span>
                        )}
                        <span className="min-w-0 break-words leading-5 text-[var(--text)]" title={modelLabel}>
                          {modelLabel}
                        </span>
                        {isBaseline && (
                          <Badge variant="default" className="ml-auto shrink-0">{t('compare.baseline')}</Badge>
                        )}
                      </div>
                    </td>
                    {dataRows.map((row) => {
                      const ds = String(row.dataset_id)
                      const score = row ? (row[rk] as number) : null
                      const baselineScore = row ? (row[baselineReport] as number) : null
                      const isBest = row ? !!(row[`${rk}_best`]) : false
                      const semantics = scoreSemantics[ds]
                      return (
                        <td
                          key={ds}
                          className="min-w-[120px] border-l border-[var(--border-strong)] p-0 text-center transition-colors duration-[var(--transition)]"
                          style={scoreCellStyle(score, baselineScore, semantics, ds)}
                        >
                          {renderScoreCell(score, baselineScore, semantics, isBest, isBaseline)}
                        </td>
                      )
                    })}
                  </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ //
// Prediction Comparison Tab                                           //
// ------------------------------------------------------------------ //

function PredictionTab({
  reportNames,
  displayNames,
  displayLabels,
  predCommonDatasets,
  datasetLabels,
  selectedDs,
  setSelectedDs,
  subsets,
  selectedSubset,
  setSelectedSubset,
  perModelFilter,
  setPerModelFilter,
  threshold,
  setThreshold,
  aboveRates,
  mergedPredictions,
  filtered,
  currentRow,
  page,
  setPage,
  totalPages,
  predictionsLoading,
  predictionsError,
  onRetryPredictions,
  t,
}: {
  reportNames: string[]
  displayNames: Record<string, string>
  displayLabels: Record<string, string>
  predCommonDatasets: string[]
  datasetLabels: Record<string, string>
  selectedDs: string
  setSelectedDs: (ds: string) => void
  subsets: string[]
  selectedSubset: string
  setSelectedSubset: (s: string) => void
  perModelFilter: Record<string, PerModelFilter>
  setPerModelFilter: (f: Record<string, PerModelFilter>) => void
  threshold: number
  setThreshold: (n: number) => void
  aboveRates: Record<string, number>
  mergedPredictions: MergedPrediction[]
  filtered: MergedPrediction[]
  currentRow: MergedPrediction | null
  page: number
  setPage: (p: number) => void
  totalPages: number
  predictionsLoading: boolean
  predictionsError: string
  onRetryPredictions: () => void
  t: Translate
}) {
  // ── Filter helpers ──────────────────────────────────────────────
  const setModelFilter = (name: string, f: PerModelFilter) =>
    setPerModelFilter({ ...perModelFilter, [name]: f })

  const setAllFilters = (f: PerModelFilter) => {
    const next: Record<string, PerModelFilter> = {}
    reportNames.forEach((n) => { next[n] = f })
    setPerModelFilter(next)
  }

  const isAllAny = reportNames.every((n) => (perModelFilter[n] ?? 'any') === 'any')
  const isAllAbove = reportNames.every((n) => (perModelFilter[n] ?? 'any') === 'above')
  const isAllBelow = reportNames.every((n) => (perModelFilter[n] ?? 'any') === 'below')

  // ── Keyboard navigation ─────────────────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'ArrowLeft' && page > 1) setPage(page - 1)
      else if (e.key === 'ArrowRight' && page < totalPages) setPage(page + 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [page, totalPages, setPage])

  if (predCommonDatasets.length === 0) {
    return (
      <Card>
        <EmptyStateSystem
          reason="no-match"
          context={{ view: 'compare' }}
          hint={t('compare.noCommon')}
        />
      </Card>
    )
  }

  // Preset buttons config
  const presets = [
    { label: t('common.all'), active: isAllAny, onClick: () => { setPerModelFilter({}); setPage(1) } },
    { label: t('compare.allAbove'), active: isAllAbove, onClick: () => { setAllFilters('above'); setPage(1) } },
    { label: t('compare.allBelow'), active: isAllBelow, onClick: () => { setAllFilters('below'); setPage(1) } },
  ]
  const modelFilterOptions: {
    key: PerModelFilter
    label: string
    accessibleLabel: string
    icon?: ReactNode
  }[] = [
    { key: 'any', label: t('compare.any'), accessibleLabel: t('compare.any') },
    {
      key: 'above',
      label: t('prediction.above'),
      accessibleLabel: t('prediction.aboveFilter'),
      icon: <ArrowUp size={12} />,
    },
    {
      key: 'below',
      label: t('prediction.below'),
      accessibleLabel: t('prediction.belowFilter'),
      icon: <ArrowDown size={12} />,
    },
  ]

  return (
    <div className="flex flex-col gap-4">
      {/* ── Dataset controls + per-model filters ── */}
      <div
        data-testid="prediction-controls"
        className="flex flex-col gap-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-4"
      >
        <div className="grid items-end gap-3 lg:grid-cols-[minmax(200px,1fr)_minmax(180px,1fr)_112px]">
          <div className="min-w-0">
            <Select
              label={t('compare.selectDataset')}
              options={predCommonDatasets.map((ds) => ({ value: ds, label: datasetLabels[ds] || ds }))}
              value={selectedDs}
              onChange={(v) => { setSelectedDs(v); setSelectedSubset('') }}
              placeholder={`-- ${t('compare.selectDataset')} --`}
            />
          </div>
          {subsets.length > 0 && (
            <div className="min-w-0">
              <Select
                label={t('compare.selectSubset')}
                options={subsets.map((s) => ({ value: s, label: s }))}
                value={selectedSubset}
                onChange={setSelectedSubset}
                placeholder={`-- ${t('compare.selectSubset')} --`}
              />
            </div>
          )}
          {subsets.length === 0 && <div aria-hidden="true" className="hidden lg:block" />}
          <div className="flex flex-col gap-1.5">
            <label
              htmlFor="compare-score-threshold"
              className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]"
            >
              {t('compare.scoreThreshold')}
            </label>
            <input
              id="compare-score-threshold"
              name="compare-score-threshold"
              type="number"
              value={threshold}
              step={0.01}
              min={0}
              max={1}
              onChange={(e) => { setThreshold(Number(e.target.value)); setPage(1) }}
              className="w-24 px-3 py-2 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] focus:outline-none focus:border-[var(--accent)]"
            />
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-[var(--border)] pt-3">
          <Eyebrow as="span">{t('compare.filterByModel')}</Eyebrow>
          <div
            role="group"
            aria-label={t('compare.filterByModel')}
            className="inline-flex overflow-hidden rounded-[var(--radius-sm)] border border-[var(--border)]"
          >
            {presets.map(({ label, active, onClick }, idx, arr) => (
              <button
                key={label}
                aria-pressed={active}
                onClick={onClick}
                className={cn(
                  'px-3.5 py-1.5 type-button-sm transition-colors cursor-pointer',
                  active
                    ? 'bg-[var(--accent)] text-[var(--text-on-filled)]'
                    : 'bg-transparent text-[var(--text-muted)] hover:text-[var(--text)]',
                  idx < arr.length - 1 && 'border-r border-[var(--border)]',
                )}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Per-model tri-state filters */}
        <div
          data-testid="prediction-model-filters"
          className="grid gap-2 md:grid-cols-2 xl:grid-cols-3"
        >
          {reportNames.map((name, idx) => {
            const palette = MODEL_PALETTE[idx] ?? MODEL_PALETTE[0]
            const cur = perModelFilter[name] ?? 'any'
            const rate = aboveRates[name]
            const modelLabel = displayLabels[name]
              ?? displayNames[name]
              ?? (parseReportRef(name).modelId || name)
            return (
              <div
                key={name}
                role="group"
                aria-label={`${t('compare.filterByModel')}: ${modelLabel}`}
                className="flex min-w-0 flex-col gap-2 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-deep)] p-3"
              >
                <div className="flex min-w-0 items-center gap-2">
                  <span
                    className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ backgroundColor: palette.dot }}
                  />
                  <span
                    className="min-w-0 flex-1 truncate text-xs font-medium text-[var(--text)]"
                    title={modelLabel}
                  >
                    {modelLabel}
                  </span>
                  {rate !== undefined && mergedPredictions.length > 0 && (
                    <Badge variant="default" className="shrink-0">
                      {formatMetric(rate, RATIO_PERCENT_SEMANTICS).primary}
                    </Badge>
                  )}
                </div>

                {/* Tri-state chips */}
                <div className="grid grid-cols-3 overflow-hidden rounded-[var(--radius-sm)] border border-[var(--border)]">
                  {modelFilterOptions.map(({ key, label, accessibleLabel, icon }, ci, options) => {
                    const isActive = cur === key
                    return (
                      <button
                        key={key}
                        aria-label={accessibleLabel}
                        aria-pressed={isActive}
                        onClick={() => { setModelFilter(name, key); setPage(1) }}
                        className={cn(
                          'flex min-w-0 items-center justify-center gap-1 px-2 py-1.5 type-button-sm whitespace-nowrap transition-colors cursor-pointer',
                          isActive
                            ? 'bg-[var(--accent)] text-[var(--text-on-filled)]'
                            : 'bg-transparent text-[var(--text-muted)] hover:text-[var(--text)]',
                          ci < options.length - 1 && 'border-r border-[var(--border)]',
                        )}
                      >
                        {icon}
                        {label}
                      </button>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Stats Bar + Pagination ── */}
      {!predictionsLoading && mergedPredictions.length > 0 && (
        <div className="flex items-center justify-between px-4 py-2.5 rounded-[var(--radius)] bg-[var(--bg-card)] border border-[var(--border)] gap-2 flex-wrap">
          <span className="text-sm text-[var(--text-muted)]">
            {t('compare.showing')}{' '}
            <strong className="text-[var(--text)]">{filtered.length}</strong>{' '}
            {t('compare.of')}{' '}
            <strong className="text-[var(--text)]">{mergedPredictions.length}</strong>{' '}
            {t('compare.predictions')}
            {currentRow && (
              <span className="ml-2 text-xs opacity-50">#{currentRow.Index}</span>
            )}
          </span>
          <div className="flex items-center gap-2">
            <button
              aria-label={t('prediction.previousSample')}
              disabled={page <= 1}
              onClick={() => setPage(page - 1)}
              className="p-1.5 rounded-[var(--radius-sm)] hover:bg-[var(--bg-card2)] disabled:opacity-30 transition-colors cursor-pointer disabled:cursor-not-allowed"
            >
              <ChevronLeft size={16} />
            </button>
            <span className="text-sm text-[var(--text-muted)] min-w-[5rem] text-center tabular-nums">
              {t('compare.sample')} {page} / {totalPages}
            </span>
            <button
              aria-label={t('prediction.nextSample')}
              disabled={page >= totalPages}
              onClick={() => setPage(page + 1)}
              className="p-1.5 rounded-[var(--radius-sm)] hover:bg-[var(--bg-card2)] disabled:opacity-30 transition-colors cursor-pointer disabled:cursor-not-allowed"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}

      {/* ── Loading skeleton ── */}
      {predictionsLoading && <Skeleton height={400} />}

      {predictionsError && (
        <ErrorAlert className="flex items-center justify-between gap-3">
          <span className="type-body-sm break-words">{predictionsError}</span>
          <Button size="sm" variant="outline" onClick={onRetryPredictions}>
            {t('common.retry')}
          </Button>
        </ErrorAlert>
      )}

      {/* ── ChatView Columns ── */}
      {!predictionsLoading && currentRow && (
        <div
          className="grid gap-4"
          style={{
            gridTemplateColumns: `repeat(${reportNames.length}, minmax(0, 1fr))`,
          }}
        >
          {reportNames.map((name, idx) => {
            const palette = MODEL_PALETTE[idx] ?? MODEL_PALETTE[0]
            const modelRow = currentRow.models[name]
            if (!modelRow) return null
            return (
              <div
                key={name}
                className="flex flex-col rounded-[var(--radius)] border overflow-hidden"
                style={{ borderColor: palette.border, background: palette.bg }}
              >
                {/* Column Header */}
                <div
                  className="flex items-center justify-between px-4 py-2.5 border-b shrink-0"
                  style={{ borderColor: palette.border, background: palette.headerBg }}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span
                      className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                      style={{ backgroundColor: palette.dot }}
                    />
                    <span
                      className="text-xs font-semibold truncate"
                      style={{ color: palette.dot }}
                      title={displayLabels[name] ?? displayNames[name] ?? (parseReportRef(name).modelId || name)}
                    >
                      {displayLabels[name] ?? displayNames[name] ?? (parseReportRef(name).modelId || name)}
                    </span>
                  </div>
                  {/*
                    Native normalized score with a neutral gradient, independent
                    of the view threshold. The threshold only filters
                    rows in this view and is not a pass/fail verdict.
                  */}
                  <ScoreBadge
                    score={modelRow.NScore}
                    semantics={RATIO_PERCENT_SEMANTICS}
                    className="ml-2 shrink-0 !font-mono"
                  />
                </div>

                {/* ChatView */}
                <div
                  className="overflow-y-auto p-3"
                  style={{ maxHeight: 'calc(100vh - 380px)', minHeight: '280px' }}
                >
                  <ChatView prediction={modelRow} threshold={threshold} />
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* ── Empty state ── */}
      {!predictionsLoading && !predictionsError && mergedPredictions.length === 0 && (
        <Card>
          <EmptyStateSystem reason="no-data" context={{ view: 'compare' }} />
        </Card>
      )}

      {!predictionsLoading && !predictionsError && mergedPredictions.length > 0 && filtered.length === 0 && (
        <Card>
          <EmptyStateSystem reason="no-match" context={{ view: 'compare' }} />
        </Card>
      )}
    </div>
  )
}
