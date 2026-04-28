import { useCallback, useEffect, useMemo, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import { getPredictions, getChartUrl } from '@/api/reports'
import type { ReportData, PredictionRow } from '@/api/types'
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
import ChatView from '@/components/single/ChatView'
import { Plus, ChevronLeft, ChevronRight, AlertCircle, CircleCheck, CircleX } from 'lucide-react'

// ------------------------------------------------------------------ //
// Types                                                               //
// ------------------------------------------------------------------ //

interface MergedPrediction {
  Index: string
  Input: string
  Gold: string
  models: Record<string, PredictionRow>
}

type PerModelFilter = 'any' | 'pass' | 'fail'

// Distinct accent color palette for each model column (up to 3)
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
  const { rootPath: ctxRootPath, setRootPath, loadMultiReports, loading, loadReport, reportCache } = useReports()

  // URL params — limit to 3 models
  const rootPath = qp.get('root_path') || ctxRootPath
  const reportsParam = qp.get('reports') || ''
  const reportNames = useMemo(
    () => reportsParam.split(';').filter(Boolean).slice(0, 3),
    [reportsParam],
  )

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
  const [perModelFilter, setPerModelFilter] = useState<Record<string, PerModelFilter>>({})
  const [threshold, setThreshold] = useState(0.99)
  const [page, setPage] = useState(1)
  const [predictionsLoading, setPredictionsLoading] = useState(false)

  // Reset per-model filters when selected models change
  const reportNamesKey = reportNames.join(';')
  useEffect(() => {
    setPerModelFilter({})
  }, [reportNamesKey]) // eslint-disable-line react-hooks/exhaustive-deps

  // Sync root path
  useEffect(() => {
    if (rootPath && rootPath !== ctxRootPath) setRootPath(rootPath)
  }, [rootPath, ctxRootPath, setRootPath])

  // Load score data for all reports
  useEffect(() => {
    if (reportNames.length < 2) return
    setDataLoaded(false)
    loadMultiReports(reportNames)
      .then((list) => { setReports(list); setDataLoaded(true) })
      .catch(() => setDataLoaded(true))
  }, [reportNames, loadMultiReports])

  // Load individual reports (needed for dataset/subset lists)
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

    if (common.length > 0) {
      const avgRow: Record<string, unknown> = { dataset: t('compare.average') }
      reportKeys.forEach((k) => {
        const scores = common.map((ds) => byReport[k][ds] ?? 0)
        avgRow[k] = scores.reduce((a, b) => a + b, 0) / scores.length
        avgRow[`${k}_best`] = false
      })
      let bestAvg = -1
      reportKeys.forEach((k) => { if ((avgRow[k] as number) > bestAvg) bestAvg = avgRow[k] as number })
      reportKeys.forEach((k) => { if ((avgRow[k] as number) === bestAvg && bestAvg > 0) avgRow[`${k}_best`] = true })
      rows.push(avgRow)
    }

    const columns = [
      { key: 'dataset', label: t('compare.dataset') },
      ...reportKeys.map((k) => ({ key: k, label: displayNames[k] })),
    ]

    return { scoreTableData: rows, scoreTableColumns: columns, displayNames }
  }, [reports, reportNames, t])

  // ------------------------------------------------------------------ //
  // Prediction Tab Data                                                 //
  // ------------------------------------------------------------------ //

  const predCommonDatasets = useMemo(() => {
    if (reportNames.length < 2) return []
    const dsLists = reportNames.map((name) => {
      const cached = reportCache[name]
      return cached ? new Set(cached.datasets) : new Set<string>()
    })
    if (dsLists.some((s) => s.size === 0)) return []
    return [...dsLists.reduce((a, b) => new Set([...a].filter((x) => b.has(x))))]
  }, [reportNames, reportCache])

  useEffect(() => {
    if (activeTab === 'prediction' && predCommonDatasets.length > 0 && !selectedDs) {
      setSelectedDs(predCommonDatasets[0])
    }
  }, [activeTab, predCommonDatasets, selectedDs])

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

  useEffect(() => {
    if (subsets.length > 0 && !selectedSubset) setSelectedSubset(subsets[0])
  }, [subsets, selectedSubset])

  const loadPredictions = useCallback(async () => {
    if (!selectedDs || !selectedSubset || reportNames.length < 2) return
    setPredictionsLoading(true)
    try {
      const results = await Promise.all(
        reportNames.map((name) => getPredictions(rootPath, name, selectedDs, selectedSubset)),
      )
      const indexMap = new Map<string, MergedPrediction>()
      results.forEach((res, i) => {
        const modelName = reportNames[i]
        for (const p of res.predictions) {
          if (!indexMap.has(p.Index)) {
            indexMap.set(p.Index, { Index: p.Index, Input: p.Input, Gold: p.Gold, models: {} })
          }
          indexMap.get(p.Index)!.models[modelName] = p
        }
      })
      const merged = [...indexMap.values()].filter((row) =>
        reportNames.every((m) => row.models[m]),
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

  // Filtered predictions using per-model constraints
  const filtered = useMemo(() => {
    return mergedPredictions.filter((row) =>
      reportNames.every((name) => {
        const f = perModelFilter[name] ?? 'any'
        if (f === 'any') return true
        const pass = (row.models[name]?.NScore ?? 0) >= threshold
        return f === 'pass' ? pass : !pass
      }),
    )
  }, [mergedPredictions, perModelFilter, threshold, reportNames])

  // Pass rates per model (based on full set)
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
    qp.set('reports', reportNames.filter((n) => n !== name).join(';'))
  }, [reportNames, qp])

  const addReport = useCallback(() => {
    if (!addInput.trim() || reportNames.length >= 3) return
    qp.set('reports', [...reportNames, addInput.trim()].join(';'))
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
      <Breadcrumb items={[{ label: t('reports.title'), href: '/reports' }, { label: t('compare.title') }]} />

      {/* Selected Models */}
      <Card title={t('compare.selectedModels')}>
        <div className="flex flex-wrap items-center gap-2">
          {reportNames.map((name) => (
            <FilterChip
              key={name}
              label={displayNames[name] ?? (parseReportName(name).model || name)}
              onRemove={reportNames.length > 2 ? () => removeReport(name) : undefined}
            />
          ))}
          {reportNames.length < 3 && (showAddInput ? (
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
          ))}
          <span className="text-xs text-[var(--text-dim)] ml-auto">{t('compare.maxThreeHint')}</span>
        </div>
      </Card>

      {/* Tab Switch */}
      <Tabs
        tabs={[
          { key: 'score', label: t('compare.scoreComparison') },
          { key: 'prediction', label: t('compare.predictionComparison') },
        ]}
        activeKey={activeTab}
        onChange={(k) => setActiveTab(k as 'score' | 'prediction')}
      />

      {/* Content */}
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
          perModelFilter={perModelFilter}
          setPerModelFilter={setPerModelFilter}
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
// Score Comparison Tab                                                //
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
  const reportKeys = scoreTableColumns.slice(1).map((c) => c.key)
  const dataRows = scoreTableData.filter((r) => r.dataset !== t('compare.average'))
  const avgRow = scoreTableData.find((r) => r.dataset === t('compare.average')) ?? null
  const datasetNames = dataRows.map((r) => r.dataset as string)

  return (
    <div className="flex flex-col gap-6">
      <PlotlyChart
        src={getChartUrl(rootPath, 'radar', { reportNames })}
        height={450}
        title={t('multi.modelRadar')}
      />

      <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden shadow-[var(--shadow-sm)]">
        <div className="flex items-center border-b border-[var(--border)] px-5 py-3">
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
                    <th key={ds} className="py-2.5 text-center text-[10px] font-semibold uppercase tracking-wider text-[var(--text-dim)] whitespace-nowrap w-[100px]">
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
                {reportKeys.map((rk, rkIdx) => (
                  <tr key={rk} className="hover:bg-[var(--bg-card2)] transition-colors">
                    <td className="px-3 py-2 text-xs font-medium whitespace-nowrap sticky left-0 bg-[var(--bg-card)] z-10 border-r border-[var(--border)] w-32">
                      <div className="flex items-center gap-1.5">
                        <span className="inline-block w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: MODEL_PALETTE[rkIdx]?.dot }} />
                        <span className="text-[var(--text-muted)] truncate max-w-[110px]" title={displayNames[rk] ?? rk}>
                          {displayNames[rk] ?? rk}
                        </span>
                      </div>
                    </td>
                    {datasetNames.map((ds) => {
                      const row = dataRows.find((r) => r.dataset === ds)
                      const score = row ? (row[rk] as number) : null
                      const isBest = row ? !!(row[`${rk}_best`]) : false
                      return (
                        <td key={ds} className="px-1 py-1 w-[100px]">
                          {score != null ? (
                            <div className="w-full py-1.5 px-2 rounded-[var(--radius-xs)] text-xs font-mono font-medium text-center text-white" style={{ backgroundColor: scoreColor(score) }}>
                              {isBest && <span className="inline-block w-1.5 h-1.5 rounded-full bg-white mr-1 align-middle opacity-80" />}
                              {(score * 100).toFixed(1)}%
                            </div>
                          ) : (
                            <div className="w-full py-1.5 px-2 text-xs text-center text-[var(--text-dim)] bg-[var(--bg-deep)] rounded-[var(--radius-xs)]">—</div>
                          )}
                        </td>
                      )
                    })}
                    {avgRow && (() => {
                      const score = avgRow[rk] as number
                      const isBest = !!(avgRow[`${rk}_best`])
                      return (
                        <td className="px-1 py-1 border-l border-[var(--border)] w-[100px]">
                          <div className="w-full py-1.5 px-2 rounded-[var(--radius-xs)] text-xs font-mono font-semibold text-center text-white" style={{ backgroundColor: scoreColor(score) }}>
                            {isBest && <span className="inline-block w-1.5 h-1.5 rounded-full bg-white mr-1 align-middle opacity-80" />}
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
  perModelFilter,
  setPerModelFilter,
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
  perModelFilter: Record<string, PerModelFilter>
  setPerModelFilter: (f: Record<string, PerModelFilter>) => void
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
  // ── Filter helpers ──────────────────────────────────────────────
  const setModelFilter = (name: string, f: PerModelFilter) =>
    setPerModelFilter({ ...perModelFilter, [name]: f })

  const setAllFilters = (f: PerModelFilter) => {
    const next: Record<string, PerModelFilter> = {}
    reportNames.forEach((n) => { next[n] = f })
    setPerModelFilter(next)
  }

  const isAllAny = reportNames.every((n) => (perModelFilter[n] ?? 'any') === 'any')
  const isAllPass = reportNames.every((n) => (perModelFilter[n] ?? 'any') === 'pass')
  const isAllFail = reportNames.every((n) => (perModelFilter[n] ?? 'any') === 'fail')

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
        <div className="flex flex-col items-center justify-center gap-3 py-12">
          <AlertCircle size={32} className="text-[var(--text-dim)]" />
          <p className="text-[var(--text-muted)]">{t('compare.noCommon')}</p>
        </div>
      </Card>
    )
  }

  // Preset buttons config
  const presets = [
    { label: t('common.all'), active: isAllAny, onClick: () => { setPerModelFilter({}); setPage(1) } },
    { label: t('compare.allPass'), active: isAllPass, onClick: () => { setAllFilters('pass'); setPage(1) } },
    { label: t('compare.allFail'), active: isAllFail, onClick: () => { setAllFilters('fail'); setPage(1) } },
  ]

  return (
    <div className="flex flex-col gap-4">

      {/* ── Dataset / Subset / Threshold ── */}
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

      {/* ── Per-model Filter Section ── */}
      <div className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-4 flex flex-col gap-3">
        {/* Quick preset row */}
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-xs font-semibold uppercase tracking-wider text-[var(--text-dim)]">
            {t('compare.filterByModel')}
          </span>
          <div
            style={{
              display: 'inline-flex',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border)',
              overflow: 'hidden',
            }}
          >
            {presets.map(({ label, active, onClick }, idx, arr) => (
              <button
                key={label}
                onClick={onClick}
                style={{
                  padding: '0.3rem 0.85rem',
                  fontSize: '0.78rem',
                  fontWeight: 500,
                  background: active ? 'var(--accent)' : 'transparent',
                  color: active ? 'var(--bg)' : 'var(--text-muted)',
                  border: 'none',
                  borderRight: idx < arr.length - 1 ? '1px solid var(--border)' : 'none',
                  cursor: 'pointer',
                  transition: 'background 0.15s, color 0.15s',
                }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Per-model tri-state chips */}
        <div className="flex flex-col gap-2.5">
          {reportNames.map((name, idx) => {
            const palette = MODEL_PALETTE[idx] ?? MODEL_PALETTE[0]
            const cur = perModelFilter[name] ?? 'any'
            const rate = passRates[name]
            const chips: { key: PerModelFilter; label: string; icon?: React.ReactNode; activeBg: string }[] = [
              { key: 'any', label: t('compare.any'), activeBg: 'var(--accent)' },
              { key: 'pass', label: t('common.pass'), icon: <CircleCheck size={12} />, activeBg: 'var(--color-pass)' },
              { key: 'fail', label: t('common.fail'), icon: <CircleX size={12} />, activeBg: 'var(--color-fail)' },
            ]
            return (
              <div key={name} className="flex items-center gap-3 flex-wrap">
                {/* Model label */}
                <div className="flex items-center gap-1.5 w-36 shrink-0">
                  <span className="inline-block w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: palette.dot }} />
                  <span
                    className="text-xs font-medium truncate"
                    style={{ color: palette.dot }}
                    title={displayNames[name] ?? (parseReportName(name).model || name)}
                  >
                    {displayNames[name] ?? (parseReportName(name).model || name)}
                  </span>
                </div>

                {/* Tri-state chips */}
                <div style={{ display: 'inline-flex', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                  {chips.map(({ key, label, icon, activeBg }, ci, ca) => {
                    const isActive = cur === key
                    return (
                      <button
                        key={key}
                        onClick={() => { setModelFilter(name, key); setPage(1) }}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.25rem',
                          padding: '0.3rem 0.7rem',
                          fontSize: '0.75rem',
                          fontWeight: 500,
                          background: isActive ? activeBg : 'transparent',
                          color: isActive ? 'var(--bg)' : 'var(--text-dim)',
                          border: 'none',
                          borderRight: ci < ca.length - 1 ? '1px solid var(--border)' : 'none',
                          cursor: 'pointer',
                          transition: 'background 0.15s, color 0.15s',
                        }}
                      >
                        {icon}
                        {label}
                      </button>
                    )
                  })}
                </div>

                {/* Pass rate badge */}
                {rate !== undefined && mergedPredictions.length > 0 && (
                  <Badge variant={rate >= 0.5 ? 'success' : 'warning'}>
                    {(rate * 100).toFixed(1)}%
                  </Badge>
                )}
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
            const pass = modelRow.NScore >= threshold
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
                      title={displayNames[name] ?? (parseReportName(name).model || name)}
                    >
                      {displayNames[name] ?? (parseReportName(name).model || name)}
                    </span>
                  </div>
                  <span
                    className="text-xs font-mono font-semibold ml-2 shrink-0 px-2 py-0.5 rounded"
                    style={{
                      backgroundColor: pass ? 'var(--color-accent-muted)' : 'var(--danger-bg)',
                      color: pass ? 'var(--bubble-bot-color)' : 'var(--danger)',
                    }}
                  >
                    {modelRow.NScore.toFixed(4)}
                  </span>
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
      {!predictionsLoading && mergedPredictions.length > 0 && filtered.length === 0 && (
        <Card>
          <div className="text-center py-8 text-[var(--text-dim)]">{t('common.noData')}</div>
        </Card>
      )}
    </div>
  )
}
