import { useCallback, useEffect, useMemo, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import { getPredictions, getChartUrl } from '@/api/reports'
import type { ReportData } from '@/api/types'
import { getDisplayNames, parseReportName } from '@/utils/reportParser'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Card from '@/components/ui/Card'
import Tabs from '@/components/ui/Tabs'
import { scoreColor } from '@/components/ui/Table'
import FilterChip from '@/components/ui/FilterChip'
import Button from '@/components/ui/Button'
import Select from '@/components/ui/Select'
import Skeleton from '@/components/ui/Skeleton'
import Badge from '@/components/ui/Badge'
import PlotlyChart from '@/components/charts/PlotlyChart'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import { Plus, ChevronLeft, ChevronRight, AlertCircle } from 'lucide-react'

// ------------------------------------------------------------------ //
// Types                                                               //
// ------------------------------------------------------------------ //

interface MergedPrediction {
  Index: string
  Input: string
  Gold: string
  models: Record<string, { Generated: string; Pred: string; NScore: number; Score: Record<string, unknown> }>
}

type AnswerFilter = 'all' | 'allPass' | 'allFail' | 'mixed'

// ------------------------------------------------------------------ //
// Main Component                                                      //
// ------------------------------------------------------------------ //

export default function ComparePage() {
  const { t } = useLocale()
  const qp = useQueryParams()
  const { rootPath: ctxRootPath, setRootPath, loadMultiReports, loading, loadReport, reportCache } = useReports()

  // URL params
  const rootPath = qp.get('root_path') || ctxRootPath
  const reportsParam = qp.get('reports') || ''
  const reportNames = useMemo(() => reportsParam.split(';').filter(Boolean), [reportsParam])

  // State
  const [reports, setReports] = useState<ReportData[]>([])
  const [activeTab, setActiveTab] = useState<'score' | 'prediction'>('score')
  const [dataLoaded, setDataLoaded] = useState(false)
  const [addInput, setAddInput] = useState('')
  const [showAddInput, setShowAddInput] = useState(false)

  // Prediction tab state
  const [selectedDs, setSelectedDs] = useState('')
  const [selectedSubset, setSelectedSubset] = useState('')
  const [mergedPredictions, setMergedPredictions] = useState<MergedPrediction[]>([])
  const [answerFilter, setAnswerFilter] = useState<AnswerFilter>('all')
  const [threshold, setThreshold] = useState(0.99)
  const [page, setPage] = useState(1)
  const [predictionsLoading, setPredictionsLoading] = useState(false)

  // Sync root path
  useEffect(() => {
    if (rootPath && rootPath !== ctxRootPath) setRootPath(rootPath)
  }, [rootPath, ctxRootPath, setRootPath])

  // Load reports data
  useEffect(() => {
    if (reportNames.length < 2) return
    setDataLoaded(false)
    loadMultiReports(reportNames).then((list) => {
      setReports(list)
      setDataLoaded(true)
    }).catch(() => setDataLoaded(true))
  }, [reportNames, loadMultiReports])

  // Load individual reports for predictions (need datasets info)
  useEffect(() => {
    if (reportNames.length < 2) return
    reportNames.forEach((name) => loadReport(name))
  }, [reportNames, loadReport])

  // ------------------------------------------------------------------ //
  // Score Tab Data                                                      //
  // ------------------------------------------------------------------ //

  const { scoreTableData, scoreTableColumns, displayNames } = useMemo(() => {
    const displayNames = getDisplayNames(reportNames)
    if (!reports.length) return { scoreTableData: [], scoreTableColumns: [], displayNames }

    const byReport: Record<string, Record<string, number>> = {}
    for (const r of reports) {
      const key = (r as ReportData & { _reportName?: string })._reportName ?? r.model_name
      if (!byReport[key]) byReport[key] = {}
      byReport[key][r.dataset_name] = r.score
    }

    const reportKeys = reportNames.filter((n) => byReport[n])
    const dsLists = reportKeys.map((k) => new Set(Object.keys(byReport[k])))
    const common = dsLists.length
      ? [...dsLists.reduce((a, b) => new Set([...a].filter((x) => b.has(x))))]
      : []
    common.sort()

    // Build table rows: each row = one dataset, columns = reportKeys
    const rows: Record<string, unknown>[] = common.map((ds) => {
      const row: Record<string, unknown> = { dataset: ds }
      const scores = reportKeys.map((k) => byReport[k][ds] ?? 0)
      const maxScore = Math.max(...scores)
      reportKeys.forEach((k, i) => {
        row[k] = scores[i]
        row[`${k}_best`] = scores[i] === maxScore && maxScore > 0
      })
      return row
    })

    // Average row
    if (common.length > 0) {
      const avgRow: Record<string, unknown> = { dataset: t('compare.average') }
      reportKeys.forEach((k) => {
        const scores = common.map((ds) => byReport[k][ds] ?? 0)
        const avg = scores.reduce((a, b) => a + b, 0) / scores.length
        avgRow[k] = avg
        avgRow[`${k}_best`] = false
      })
      let bestAvg = -1
      reportKeys.forEach((k) => { if ((avgRow[k] as number) > bestAvg) bestAvg = avgRow[k] as number })
      reportKeys.forEach((k) => { if ((avgRow[k] as number) === bestAvg && bestAvg > 0) avgRow[`${k}_best`] = true })
      rows.push(avgRow)
    }

    const columns = [
      { key: 'dataset', label: t('compare.dataset') },
      ...reportKeys.map((k) => ({
        key: k,
        label: displayNames[k],
      })),
    ]

    return { scoreTableData: rows, scoreTableColumns: columns, displayNames }
  }, [reports, reportNames, t])

  // ------------------------------------------------------------------ //
  // Prediction Tab Data                                                 //
  // ------------------------------------------------------------------ //

  // Common datasets for prediction tab (from loaded reports)
  const predCommonDatasets = useMemo(() => {
    if (reportNames.length < 2) return []
    const dsLists = reportNames.map((name) => {
      const cached = reportCache[name]
      return cached ? new Set(cached.datasets) : new Set<string>()
    })
    if (dsLists.some((s) => s.size === 0)) return []
    return [...dsLists.reduce((a, b) => new Set([...a].filter((x) => b.has(x))))]
  }, [reportNames, reportCache])

  // Auto-select first dataset when switching to prediction tab or when predCommonDatasets loads
  useEffect(() => {
    if (activeTab === 'prediction' && predCommonDatasets.length > 0 && !selectedDs) {
      setSelectedDs(predCommonDatasets[0])
    }
  }, [activeTab, predCommonDatasets, selectedDs])

  // Subsets for selected dataset
  const subsets = useMemo(() => {
    if (!selectedDs || reportNames.length < 1) return []
    const cached = reportCache[reportNames[0]]
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
  }, [selectedDs, reportNames, reportCache])

  // Auto-select first subset when subsets load
  useEffect(() => {
    if (subsets.length > 0 && !selectedSubset) {
      setSelectedSubset(subsets[0])
    }
  }, [subsets, selectedSubset])

  // Load predictions for all models
  const loadPredictions = useCallback(async () => {
    if (!selectedDs || !selectedSubset || reportNames.length < 2) return
    setPredictionsLoading(true)
    try {
      const results = await Promise.all(
        reportNames.map((name) => getPredictions(rootPath, name, selectedDs, selectedSubset))
      )
      // Merge by Index
      const indexMap = new Map<string, MergedPrediction>()
      results.forEach((res, modelIdx) => {
        const modelName = reportNames[modelIdx]
        for (const p of res.predictions) {
          if (!indexMap.has(p.Index)) {
            indexMap.set(p.Index, { Index: p.Index, Input: p.Input, Gold: p.Gold, models: {} })
          }
          const merged = indexMap.get(p.Index)!
          merged.models[modelName] = {
            Generated: p.Generated,
            Pred: p.Pred,
            NScore: p.NScore,
            Score: p.Score,
          }
        }
      })
      // Only keep rows present in all models
      const allModels = reportNames
      const merged = [...indexMap.values()].filter((row) =>
        allModels.every((m) => row.models[m])
      )
      setMergedPredictions(merged)
      setPage(1)
    } catch (e) {
      console.error('Failed to load predictions:', e)
    } finally {
      setPredictionsLoading(false)
    }
  }, [rootPath, reportNames, selectedDs, selectedSubset])

  useEffect(() => { loadPredictions() }, [loadPredictions])

  // Filtered predictions
  const filtered = useMemo(() => {
    return mergedPredictions.filter((row) => {
      const scores = reportNames.map((m) => row.models[m]?.NScore ?? 0)
      const allPass = scores.every((s) => s >= threshold)
      const allFail = scores.every((s) => s < threshold)
      switch (answerFilter) {
        case 'allPass': return allPass
        case 'allFail': return allFail
        case 'mixed': return !allPass && !allFail
        default: return true
      }
    })
  }, [mergedPredictions, answerFilter, threshold, reportNames])

  // Pass rates per model
  const passRates = useMemo(() => {
    if (!mergedPredictions.length) return {} as Record<string, number>
    const rates: Record<string, number> = {}
    for (const name of reportNames) {
      const pass = mergedPredictions.filter((r) => (r.models[name]?.NScore ?? 0) >= threshold).length
      rates[name] = pass / mergedPredictions.length
    }
    return rates
  }, [mergedPredictions, reportNames, threshold])

  const totalPages = filtered.length
  const currentRow = filtered.length > 0 ? filtered[Math.min(page - 1, filtered.length - 1)] : null

  // ------------------------------------------------------------------ //
  // URL manipulation                                                    //
  // ------------------------------------------------------------------ //

  const removeReport = useCallback((name: string) => {
    const next = reportNames.filter((n) => n !== name)
    qp.set('reports', next.join(';'))
  }, [reportNames, qp])

  const addReport = useCallback(() => {
    if (!addInput.trim()) return
    const next = [...reportNames, addInput.trim()]
    qp.set('reports', next.join(';'))
    setAddInput('')
    setShowAddInput(false)
  }, [addInput, reportNames, qp])

  // ------------------------------------------------------------------ //
  // Render                                                              //
  // ------------------------------------------------------------------ //

  if (reportNames.length < 2) {
    return (
      <div className="page-enter">
        <Breadcrumb items={[{ label: t('reports.title'), href: '/reports' }, { label: t('compare.title') }]} />
        <div className="flex flex-col items-center justify-center gap-4 py-20">
          <AlertCircle size={48} className="text-[var(--text-dim)]" />
          <p className="text-[var(--text-muted)] text-lg">{t('compare.needTwo')}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="page-enter flex flex-col gap-6">
      {/* Breadcrumb */}
      <Breadcrumb items={[{ label: t('reports.title'), href: '/reports' }, { label: t('compare.title') }]} />

      {/* Selected Models Header */}
      <Card title={t('compare.selectedModels')}>
        <div className="flex flex-wrap items-center gap-2">
          {reportNames.map((name) => (
            <FilterChip
              key={name}
              label={displayNames[name] ?? (parseReportName(name).model || name)}
              onRemove={reportNames.length > 2 ? () => removeReport(name) : undefined}
            />
          ))}
          {showAddInput ? (
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={addInput}
                onChange={(e) => setAddInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') addReport(); if (e.key === 'Escape') setShowAddInput(false) }}
                placeholder="Report name..."
                autoFocus
                className="px-3 py-1.5 text-sm rounded-[var(--radius-sm)] bg-[var(--bg-deep)] border border-[var(--border)] text-[var(--text)] placeholder:text-[var(--text-dim)] focus:outline-none focus:border-[var(--accent)]"
              />
              <Button size="sm" onClick={addReport}>{t('compare.addModel')}</Button>
              <Button size="sm" variant="ghost" onClick={() => setShowAddInput(false)}>✕</Button>
            </div>
          ) : (
            <Button size="sm" variant="outline" onClick={() => setShowAddInput(true)}>
              <Plus size={14} />
              {t('compare.addModel')}
            </Button>
          )}
        </div>
      </Card>

      {/* Tabs */}
      <Tabs
        tabs={[
          { key: 'score', label: t('compare.scoreComparison') },
          { key: 'prediction', label: t('compare.predictionComparison') },
        ]}
        activeKey={activeTab}
        onChange={(k) => setActiveTab(k as 'score' | 'prediction')}
      />

      {/* Tab Content */}
      {loading && !dataLoaded ? (
        <div className="flex flex-col gap-4">
          <Skeleton height={450} />
          <Skeleton height={300} />
        </div>
      ) : activeTab === 'score' ? (
        <ScoreTab
          rootPath={rootPath}
          reportNames={reportNames}
          scoreTableColumns={scoreTableColumns}
          scoreTableData={scoreTableData}
          displayNames={displayNames}
          t={t}
        />
      ) : (
        <PredictionTab
          reportNames={reportNames}
          displayNames={displayNames}
          predCommonDatasets={predCommonDatasets}
          selectedDs={selectedDs}
          setSelectedDs={setSelectedDs}
          subsets={subsets}
          selectedSubset={selectedSubset}
          setSelectedSubset={setSelectedSubset}
          answerFilter={answerFilter}
          setAnswerFilter={setAnswerFilter}
          threshold={threshold}
          setThreshold={setThreshold}
          passRates={passRates}
          mergedPredictions={mergedPredictions}
          filtered={filtered}
          currentRow={currentRow}
          page={page}
          setPage={setPage}
          totalPages={totalPages}
          predictionsLoading={predictionsLoading}
          t={t}
        />
      )}
    </div>
  )
}

// ------------------------------------------------------------------ //
// Score Comparison Tab                                                 //
// ------------------------------------------------------------------ //

function ScoreTab({
  rootPath,
  reportNames,
  scoreTableColumns,
  scoreTableData,
  displayNames,
  t,
}: {
  rootPath: string
  reportNames: string[]
  scoreTableColumns: { key: string; label: string }[]
  scoreTableData: Record<string, unknown>[]
  displayNames: Record<string, string>
  t: (p: string) => string
}) {
  // reportKeys = column keys excluding the first 'dataset' column
  const reportKeys = scoreTableColumns.slice(1).map((c) => c.key)
  const dataRows = scoreTableData.filter((r) => r.dataset !== t('compare.average'))
  const avgRow = scoreTableData.find((r) => r.dataset === t('compare.average')) ?? null
  // Collect sorted dataset names from data rows
  const datasetNames = dataRows.map((r) => r.dataset as string)

  return (
    <div className="flex flex-col gap-6">
      {/* Radar Chart */}
      <PlotlyChart
        src={getChartUrl(rootPath, 'radar', { reportNames })}
        height={450}
        title={t('multi.modelRadar')}
      />

      {/* Score Heatmap: rows = reports, columns = datasets */}
      <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden shadow-[var(--shadow-sm)]">
        {/* Title bar */}
        <div className="flex items-center justify-between border-b border-[var(--border)] px-5 py-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            {t('multi.modelScores')}
          </h3>
        </div>

        {scoreTableData.length === 0 ? (
          <div className="py-12 text-center text-sm text-[var(--text-dim)]">{t('common.noData')}</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="text-sm border-collapse w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--text-dim)] sticky left-0 bg-[var(--bg-card)] z-10 border-r border-[var(--border)] w-32">
                    Model
                  </th>
                  {datasetNames.map((ds) => (
                    <th
                      key={ds}
                      className="py-2.5 text-center text-[10px] font-semibold uppercase tracking-wider text-[var(--text-dim)] whitespace-nowrap w-[100px]"
                    >
                      {ds}
                    </th>
                  ))}
                  {avgRow && (
                    <th className="py-2.5 text-center text-[10px] font-semibold uppercase tracking-wider text-[var(--accent)] whitespace-nowrap border-l border-[var(--border)] w-[100px]">
                      {t('compare.average')}
                    </th>
                  )}
                </tr>
              </thead>
              <tbody>
                {reportKeys.map((rk) => (
                  <tr key={rk} className="hover:bg-[var(--bg-card2)] transition-colors">
                    {/* Sticky model label */}
                    <td className="px-3 py-2 text-xs font-medium text-[var(--text-muted)] whitespace-nowrap sticky left-0 bg-[var(--bg-card)] z-10 border-r border-[var(--border)] w-32">
                      {displayNames[rk] ?? rk}
                    </td>
                    {/* Score cells per dataset */}
                    {datasetNames.map((ds) => {
                      // find the row where row.dataset === ds
                      const row = dataRows.find((r) => r.dataset === ds)
                      const score = row ? (row[rk] as number) : null
                      const isBest = row ? !!(row[`${rk}_best`]) : false
                      const hasScore = score != null
                      return (
                        <td key={ds} className="px-1 py-1 w-[100px]">
                          {hasScore ? (
                            <div
                              className="w-full py-1.5 px-2 rounded-[var(--radius-xs)] text-xs font-mono font-medium text-center text-white"
                              style={{ backgroundColor: scoreColor(score) }}
                            >
                              {isBest && (
                                <span className="inline-block w-1.5 h-1.5 rounded-full bg-[var(--accent)] mr-1 align-middle" />
                              )}
                              {(score * 100).toFixed(1)}%
                            </div>
                          ) : (
                            <div className="w-full py-1.5 px-2 text-xs text-center text-[var(--text-dim)] bg-[var(--bg-deep)] rounded-[var(--radius-xs)]">
                              —
                            </div>
                          )}
                        </td>
                      )
                    })}
                    {/* Average cell */}
                    {avgRow && (() => {
                      const score = avgRow[rk] as number
                      const isBest = !!(avgRow[`${rk}_best`])
                      return (
                        <td className="px-1 py-1 border-l border-[var(--border)] w-[100px]">
                          <div
                            className="w-full py-1.5 px-2 rounded-[var(--radius-xs)] text-xs font-mono font-semibold text-center text-white"
                            style={{ backgroundColor: scoreColor(score) }}
                          >
                            {isBest && (
                              <span className="inline-block w-1.5 h-1.5 rounded-full bg-[var(--accent)] mr-1 align-middle" />
                            )}
                            {(score * 100).toFixed(1)}%
                          </div>
                        </td>
                      )
                    })()}
                  </tr>
                ))}
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
  predCommonDatasets,
  selectedDs,
  setSelectedDs,
  subsets,
  selectedSubset,
  setSelectedSubset,
  answerFilter,
  setAnswerFilter,
  threshold,
  setThreshold,
  passRates,
  mergedPredictions,
  filtered,
  currentRow,
  page,
  setPage,
  totalPages,
  predictionsLoading,
  t,
}: {
  reportNames: string[]
  displayNames: Record<string, string>
  predCommonDatasets: string[]
  selectedDs: string
  setSelectedDs: (ds: string) => void
  subsets: string[]
  selectedSubset: string
  setSelectedSubset: (s: string) => void
  answerFilter: AnswerFilter
  setAnswerFilter: (f: AnswerFilter) => void
  threshold: number
  setThreshold: (n: number) => void
  passRates: Record<string, number>
  mergedPredictions: MergedPrediction[]
  filtered: MergedPrediction[]
  currentRow: MergedPrediction | null
  page: number
  setPage: (p: number) => void
  totalPages: number
  predictionsLoading: boolean
  t: (p: string) => string
}) {
  const filterTabs = useMemo(() => [
    { key: 'all', label: t('common.all') },
    { key: 'allPass', label: t('compare.allPass') },
    { key: 'allFail', label: t('compare.allFail') },
    { key: 'mixed', label: t('compare.mixed') },
  ], [t])

  if (predCommonDatasets.length === 0) {
    return (
      <Card>
        <div className="flex flex-col items-center justify-center gap-3 py-12">
          <AlertCircle size={32} className="text-[var(--text-dim)]" />
          <p className="text-[var(--text-muted)]">{t('compare.noCommon')}</p>
        </div>
      </Card>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Controls Row */}
      <Card>
        <div className="flex flex-wrap items-end gap-4">
          <div className="min-w-[200px] flex-1">
            <Select
              label={t('compare.selectDataset')}
              options={predCommonDatasets.map((ds) => ({ value: ds, label: ds }))}
              value={selectedDs}
              onChange={(v) => { setSelectedDs(v); setSelectedSubset('') }}
              placeholder={`-- ${t('compare.selectDataset')} --`}
            />
          </div>
          {subsets.length > 0 && (
            <div className="min-w-[200px] flex-1">
              <Select
                label={t('compare.selectSubset')}
                options={subsets.map((s) => ({ value: s, label: s }))}
                value={selectedSubset}
                onChange={setSelectedSubset}
                placeholder={`-- ${t('compare.selectSubset')} --`}
              />
            </div>
          )}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">
              {t('compare.scoreThreshold')}
            </label>
            <input
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
      </Card>

      {/* Answer Filter */}
      <Tabs
        tabs={filterTabs}
        activeKey={answerFilter}
        onChange={(k) => { setAnswerFilter(k as AnswerFilter); setPage(1) }}
      />

      {/* Stats Summary */}
      {mergedPredictions.length > 0 && (
        <div className="flex flex-wrap items-center gap-3 px-4 py-3 rounded-[var(--radius)] bg-[var(--bg-card)] border border-[var(--border)]">
          <span className="text-sm text-[var(--text-muted)]">
            {t('compare.showing')} <strong className="text-[var(--text)]">{filtered.length}</strong> {t('compare.of')} <strong className="text-[var(--text)]">{mergedPredictions.length}</strong> {t('compare.predictions')}
          </span>
          <span className="text-[var(--border)]">|</span>
          {reportNames.map((name) => (
            <span key={name} className="text-sm text-[var(--text-muted)]">
              {displayNames[name] ?? (parseReportName(name).model || name)}: <Badge variant={(passRates[name] ?? 0) >= 0.5 ? 'success' : 'warning'}>{((passRates[name] ?? 0) * 100).toFixed(1)}%</Badge>
            </span>
          ))}
        </div>
      )}

      {/* Loading */}
      {predictionsLoading && (
        <div className="flex flex-col gap-3">
          <Skeleton height={200} />
          <Skeleton height={200} />
        </div>
      )}

      {/* Pagination */}
      {!predictionsLoading && filtered.length > 0 && (
        <div className="flex items-center justify-between px-4 py-2 rounded-[var(--radius)] bg-[var(--bg-card)] border border-[var(--border)]">
          <span className="text-sm text-[var(--text-muted)]">#{currentRow?.Index}</span>
          <div className="flex items-center gap-2 text-sm">
            <button
              disabled={page <= 1}
              onClick={() => setPage(page - 1)}
              className="p-1.5 rounded-[var(--radius-sm)] hover:bg-[var(--bg-card2)] disabled:opacity-30 transition-colors cursor-pointer disabled:cursor-not-allowed"
            >
              <ChevronLeft size={16} />
            </button>
            <span className="text-[var(--text-muted)] min-w-[5rem] text-center">
              {page} / {totalPages}
            </span>
            <button
              disabled={page >= totalPages}
              onClick={() => setPage(page + 1)}
              className="p-1.5 rounded-[var(--radius-sm)] hover:bg-[var(--bg-card2)] disabled:opacity-30 transition-colors cursor-pointer disabled:cursor-not-allowed"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}

      {/* Side-by-Side Predictions */}
      {!predictionsLoading && currentRow && (
        <div className="flex flex-col gap-4">
          {/* Shared: Input & Gold */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <PredictionPanel title={t('compare.input')}>
              <MarkdownRenderer content={currentRow.Input} />
            </PredictionPanel>
            <PredictionPanel title={t('compare.goldAnswer')}>
              <MarkdownRenderer content={formatVal(currentRow.Gold)} />
            </PredictionPanel>
          </div>

          {/* Per-model columns: Score */}
          <div
            className="grid grid-cols-1 sm:grid-cols-2 gap-4"
            style={{ ...(reportNames.length > 2 ? { gridTemplateColumns: `repeat(${reportNames.length}, minmax(0, 1fr))` } : {}) }}
          >
            {reportNames.map((name) => {
              const data = currentRow.models[name]
              if (!data) return null
              const pass = data.NScore >= threshold
              return (
                <PredictionPanel
                  key={name}
                  title={`${displayNames[name] ?? (parseReportName(name).model || name)} — ${t('compare.score')}`}
                >
                  <span
                    className="inline-block px-2.5 py-1 rounded text-sm font-mono font-semibold"
                    style={{
                      backgroundColor: pass ? 'rgba(15,156,126,0.15)' : 'rgba(239,68,68,0.15)',
                      color: pass ? 'var(--green)' : '#ef4444',
                    }}
                  >
                    {data.NScore.toFixed(4)}
                  </span>
                </PredictionPanel>
              )
            })}
          </div>

          {/* Per-model columns: Prediction */}
          <div
            className="grid grid-cols-1 sm:grid-cols-2 gap-4"
            style={{ ...(reportNames.length > 2 ? { gridTemplateColumns: `repeat(${reportNames.length}, minmax(0, 1fr))` } : {}) }}
          >
            {reportNames.map((name) => {
              const data = currentRow.models[name]
              if (!data) return null
              return (
                <PredictionPanel
                  key={name}
                  title={`${displayNames[name] ?? (parseReportName(name).model || name)} — ${t('compare.prediction')}`}
                >
                  <MarkdownRenderer content={formatVal(data.Pred)} />
                </PredictionPanel>
              )
            })}
          </div>

          {/* Per-model columns: Generated */}
          <div
            className="grid grid-cols-1 sm:grid-cols-2 gap-4"
            style={{ ...(reportNames.length > 2 ? { gridTemplateColumns: `repeat(${reportNames.length}, minmax(0, 1fr))` } : {}) }}
          >
            {reportNames.map((name) => {
              const data = currentRow.models[name]
              if (!data) return null
              return (
                <PredictionPanel
                  key={name}
                  title={`${displayNames[name] ?? (parseReportName(name).model || name)} — ${t('compare.generated')}`}
                >
                  <MarkdownRenderer content={formatVal(data.Generated)} />
                </PredictionPanel>
              )
            })}
          </div>
        </div>
      )}

      {/* Empty state */}
      {!predictionsLoading && mergedPredictions.length > 0 && filtered.length === 0 && (
        <Card>
          <div className="text-center py-8 text-[var(--text-dim)]">{t('common.noData')}</div>
        </Card>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ //
// Helpers                                                             //
// ------------------------------------------------------------------ //

function PredictionPanel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden">
      <div className="px-4 py-2 border-b border-[var(--border)]">
        <h5 className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-dim)]">{title}</h5>
      </div>
      <div className="p-4 max-h-[300px] overflow-auto text-sm text-[var(--text)]">{children}</div>
    </div>
  )
}

function formatVal(v: unknown): string {
  if (v === null || v === undefined) return ''
  if (typeof v === 'object') return '```json\n' + JSON.stringify(v, null, 2) + '\n```'
  return String(v)
}
