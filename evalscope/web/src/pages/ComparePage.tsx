import { useCallback, useEffect, useMemo, useState } from 'react'
import { AlertCircle } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReportCache, useScan } from '@/contexts/ReportsContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { useScopedState } from '@/hooks/useScopedState'
import { useQueryParams } from '@/hooks/useQueryParams'
import { getPredictions } from '@/api/reports'
import type { ReportData } from '@/api/types'
import {
  buildDisplayLabels,
  compatibilityReason,
  getDisplayNames,
  MAX_COMPARE_SLOTS,
  togglePredictionSelection,
} from '@/domain/compare/selection'
import { metricComparisonKey } from '@/domain/compare/scoreMatrix'
import { datasetLabel, primaryMetricOf } from '@/domain/report/primaryMetrics'
import { formatMetricIdentityLabel } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Button from '@/components/ui/Button'
import Callout from '@/components/ui/Callout'
import Card from '@/components/ui/Card'
import Tabs from '@/components/ui/Tabs'
import Skeleton from '@/components/ui/Skeleton'
import ErrorAlert from '@/components/ui/ErrorAlert'
import EmptyStateSystem from '@/components/common/EmptyStateSystem'
import CompareReportRail from '@/components/compare/CompareReportRail'
import ScoreMatrixTab from '@/components/compare/ScoreMatrixTab'
import PredictionCompareTab from '@/components/compare/PredictionCompareTab'
import type { MergedPrediction, PerModelFilter } from '@/components/compare/compareSlots'

// ------------------------------------------------------------------ //
// Types                                                               //
// ------------------------------------------------------------------ //

/** Stable placeholders so an unresolved read keeps a single identity. */
const EMPTY_REPORT_LIST: ReportData[] = []
const EMPTY_MERGED: MergedPrediction[] = []

// ------------------------------------------------------------------ //
// Main Component                                                      //
// ------------------------------------------------------------------ //

export default function ComparePage() {
  const { t } = useLocale()
  const qp = useQueryParams()
  const { rootPath: ctxRootPath, setRootPath } = useScan()
  const { loadMultiReports, loading, reportCache } = useReportCache()

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
  const [activeTab, setActiveTab] = useState<'score' | 'prediction'>('score')
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
  const [perModelFilter, setPerModelFilter] = useState<Record<string, PerModelFilter>>({})
  const [threshold, setThreshold] = useState(0.99)

  // Sync root path
  useEffect(() => {
    if (rootPath && rootPath !== ctxRootPath) setRootPath(rootPath)
  }, [rootPath, ctxRootPath, setRootPath])

  // Load score data and per-report cache (datasets/subsets) in one cache-aware pass
  const scores = useAsyncResource(
    (signal) => loadMultiReports(reportNames, signal),
    [reportNames, loadMultiReports],
    { enabled: reportNames.length >= 2, fallbackMessage: t('common.loadError') },
  )
  const reports = scores.data ?? EMPTY_REPORT_LIST
  // Resolved either way: a failure is still an answer, and the view stops waiting.
  const dataLoaded = !scores.loading && (scores.data !== undefined || Boolean(scores.error))
  const scoreLoadError = scores.error

  // ------------------------------------------------------------------ //
  // Score Tab Data                                                      //
  // ------------------------------------------------------------------ //

  const { scoreTableData, scoreTableColumns, displayNames, scoreSemantics } = useMemo(() => {
    const displayNames = getDisplayNames(reportNames)
    if (!reports.length) {
      return { scoreTableData: [], scoreTableColumns: [], displayNames, scoreSemantics: {} }
    }

    const byReport: Record<string, Record<string, number>> = {}
    const datasetsByReport: Record<string, Set<string>> = {}
    const labelsByDataset: Record<string, string> = {}
    const groups: Record<string, {
      id: string
      datasetId: string
      metricLabel: string
      semantics: MetricSemantics | undefined
    }> = {}
    for (const r of reports) {
      const key = (r as ReportData & { _reportRef?: string })._reportRef ?? r.model_name
      if (!byReport[key]) byReport[key] = {}
      if (!datasetsByReport[key]) datasetsByReport[key] = new Set()
      const primary = primaryMetricOf(r)
      if (!primary) continue
      const groupId = JSON.stringify([
        r.dataset_name,
        metricComparisonKey(primary.identity, primary.semantics),
      ])
      byReport[key][groupId] = primary.score
      datasetsByReport[key].add(r.dataset_name)
      labelsByDataset[r.dataset_name] = datasetLabel(r)
      groups[groupId] = {
        id: groupId,
        datasetId: r.dataset_name,
        metricLabel: formatMetricIdentityLabel(primary.identity, primary.semantics, primary.legacy_name),
        semantics: primary.semantics,
      }
    }

    const reportKeys = reportNames.filter((n) => byReport[n])
    const dsLists = reportKeys.map((k) => datasetsByReport[k])
    const common = dsLists.length
      ? [...dsLists.reduce((a, b) => new Set([...a].filter((x) => b.has(x))))]
      : []
    const commonSet = new Set(common)
    const scoreGroups = Object.values(groups)
      .filter((group) => commonSet.has(group.datasetId))
      .sort((left, right) => (
        left.datasetId.localeCompare(right.datasetId)
        || left.metricLabel.localeCompare(right.metricLabel)
        || left.id.localeCompare(right.id)
      ))
    const semanticsByGroup: Record<string, MetricSemantics | undefined> = {}

    const rows: Record<string, unknown>[] = scoreGroups.map((group) => {
      const row: Record<string, unknown> = {
        dataset: labelsByDataset[group.datasetId] || group.datasetId,
        dataset_id: group.id,
        source_dataset_id: group.datasetId,
        metric: group.metricLabel,
      }
      const availableScores = reportKeys.flatMap((key) => {
        const score = byReport[key][group.id]
        return typeof score === 'number' && Number.isFinite(score) ? [score] : []
      })
      const lowerIsBetter = group.semantics?.direction === 'lower_is_better'
      const bestScore = availableScores.length > 0
        ? lowerIsBetter ? Math.min(...availableScores) : Math.max(...availableScores)
        : null
      reportKeys.forEach((key) => {
        const score = byReport[key][group.id]
        if (typeof score !== 'number' || !Number.isFinite(score)) return
        row[key] = score
        row[`${key}_best`] = score === bestScore && availableScores.length > 1
      })
      semanticsByGroup[group.id] = group.semantics
      return row
    })

    const columns = [
      { key: 'dataset', label: t('compare.dataset') },
      ...reportKeys.map((k) => ({ key: k, label: displayNames[k] })),
    ]

    return { scoreTableData: rows, scoreTableColumns: columns, displayNames, scoreSemantics: semanticsByGroup }
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

  const predictionsResource = useAsyncResource(
    async (signal) => {
      const results = await Promise.all(
        predictionReportNames.map(
          (name) => getPredictions(rootPath, name, selectedDs, selectedSubset, signal),
        ),
      )
      // Align the reports sample by sample; only samples every report answered
      // can be shown side by side.
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
      return [...indexMap.values()].filter((row) =>
        predictionReportNames.every((m) => row.models[m]),
      )
    },
    [rootPath, predictionReportNames, selectedDs, selectedSubset],
    {
      enabled: Boolean(selectedDs && selectedSubset) && predictionReportNames.length >= 2,
      fallbackMessage: t('common.loadError'),
    },
  )
  const mergedPredictions = predictionsResource.data ?? EMPTY_MERGED
  const predictionsLoading = predictionsResource.loading
  const predictionsError = predictionsResource.error

  // A fresh set of samples starts at the first one: the page number is scoped to
  // the sample set it was chosen in, so new data reverts to page 1 by comparison.
  const pageScope = `${rootPath}\0${selectedDs}\0${selectedSubset}\0${predictionReportNames.join(';')}`
  const [page, setPage] = useScopedState(pageScope, 1)

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
            <Callout
              variant="warning"
              icon={<AlertCircle size={18} className="text-[var(--warning-color)]" />}
              className="gap-3"
            >
              <div className="flex flex-col gap-0.5">
                <p className="text-sm font-medium text-[var(--text)]">{t('compare.incompatible')}</p>
                <p className="text-xs text-[var(--text-muted)]">
                  {t(incompatibilityReason)} · {t('compare.incompatibleHint')}
                </p>
              </div>
            </Callout>
          )}

          {scoreLoadError && (
            <ErrorAlert className="flex items-center justify-between gap-3">
              <span className="type-body-sm break-words">{scoreLoadError}</span>
              <Button size="sm" variant="outline" onClick={scores.reload}>
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
                <ScoreMatrixTab
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
                <PredictionCompareTab
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
                  onRetryPredictions={predictionsResource.reload}
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
