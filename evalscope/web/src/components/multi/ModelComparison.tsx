import { useCallback, useEffect, useMemo, useState } from 'react'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import { getPredictions } from '@/api/reports'
import { parseReportName } from '@/utils/reportParser'
import FilterBar from '@/components/common/FilterBar'
import Pagination from '@/components/common/Pagination'
import ScoreBadge from '@/components/common/ScoreBadge'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import EmptyState from '@/components/common/EmptyState'

const ANSWER_MODES = ['All', 'Pass A & B', 'Fail A & B', 'Pass A, Fail B', 'Fail A, Pass B']

interface MergedRow {
  Index: string
  A_Input: string; A_Generated: string; A_Gold: string; A_Pred: string; A_NScore: number; A_Score: Record<string, unknown>
  B_Input: string; B_Generated: string; B_Gold: string; B_Pred: string; B_NScore: number; B_Score: Record<string, unknown>
}

export default function ModelComparison() {
  const { t } = useLocale()
  const { rootPath, selectedReports, reportCache, loadReport } = useReports()
  const [modelA, setModelA] = useState('')
  const [modelB, setModelB] = useState('')
  const [commonDs, setCommonDs] = useState<string[]>([])
  const [selectedDs, setSelectedDs] = useState('')
  const [subsets, setSubsets] = useState<string[]>([])
  const [selectedSubset, setSelectedSubset] = useState('')
  const [mergedData, setMergedData] = useState<MergedRow[]>([])
  const [mode, setMode] = useState('All')
  const [threshold, setThreshold] = useState(0.99)
  const [page, setPage] = useState(1)

  // Set initial model selections
  useEffect(() => {
    if (selectedReports.length >= 2) {
      setModelA(selectedReports[0])
      setModelB(selectedReports[1])
    }
  }, [selectedReports])

  // Find common datasets when models change
  useEffect(() => {
    if (!modelA || !modelB) return
    const load = async () => {
      const dataA = await loadReport(modelA)
      const dataB = await loadReport(modelB)
      const dsA = new Set(dataA.datasets)
      const common = dataB.datasets.filter((d) => dsA.has(d))
      setCommonDs(common)
      if (common.length) setSelectedDs(common[0])
      setSelectedSubset('')
      setMergedData([])
    }
    load()
  }, [modelA, modelB, loadReport])

  // Build subsets when dataset changes
  useEffect(() => {
    if (!selectedDs || !modelA) return
    const dataA = reportCache[modelA]
    if (!dataA) return
    const report = dataA.report_list.find((r) => r.dataset_name === selectedDs)
    if (!report) return
    const subs: string[] = []
    for (const m of report.metrics) {
      for (const c of m.categories) {
        if (c.name.length && c.name.join('/') === '-') continue
        for (const s of c.subsets) {
          if (s.name !== 'overall_score' && !subs.includes(s.name)) subs.push(s.name)
        }
      }
    }
    setSubsets(subs)
    setSelectedSubset('')
    setMergedData([])
  }, [selectedDs, modelA, reportCache])

  // Load and merge predictions
  const loadComparison = useCallback(async () => {
    if (!selectedSubset || !selectedDs || !modelA || !modelB) return
    try {
      const [resA, resB] = await Promise.all([
        getPredictions(rootPath, modelA, selectedDs, selectedSubset),
        getPredictions(rootPath, modelB, selectedDs, selectedSubset),
      ])
      const mapB = new Map(resB.predictions.map((p) => [p.Index, p]))
      const merged: MergedRow[] = []
      for (const a of resA.predictions) {
        const b = mapB.get(a.Index)
        if (!b) continue
        merged.push({
          Index: a.Index,
          A_Input: a.Input, A_Generated: a.Generated, A_Gold: a.Gold, A_Pred: a.Pred, A_NScore: a.NScore, A_Score: a.Score,
          B_Input: b.Input, B_Generated: b.Generated, B_Gold: b.Gold, B_Pred: b.Pred, B_NScore: b.NScore, B_Score: b.Score,
        })
      }
      setMergedData(merged)
      setPage(1)
    } catch (e) {
      console.error('Failed to load comparison:', e)
    }
  }, [rootPath, modelA, modelB, selectedDs, selectedSubset])

  useEffect(() => { loadComparison() }, [loadComparison])

  const filtered = useMemo(() => {
    switch (mode) {
      case 'Pass A & B': return mergedData.filter((r) => r.A_NScore >= threshold && r.B_NScore >= threshold)
      case 'Fail A & B': return mergedData.filter((r) => r.A_NScore < threshold && r.B_NScore < threshold)
      case 'Pass A, Fail B': return mergedData.filter((r) => r.A_NScore >= threshold && r.B_NScore < threshold)
      case 'Fail A, Pass B': return mergedData.filter((r) => r.A_NScore < threshold && r.B_NScore >= threshold)
      default: return mergedData
    }
  }, [mergedData, mode, threshold])

  const passA = useMemo(() => mergedData.filter((r) => r.A_NScore >= threshold).length, [mergedData, threshold])
  const passB = useMemo(() => mergedData.filter((r) => r.B_NScore >= threshold).length, [mergedData, threshold])
  const passBoth = useMemo(() => mergedData.filter((r) => r.A_NScore >= threshold && r.B_NScore >= threshold).length, [mergedData, threshold])
  const failBoth = useMemo(() => mergedData.filter((r) => r.A_NScore < threshold && r.B_NScore < threshold).length, [mergedData, threshold])
  const row = filtered.length > 0 ? filtered[Math.min(page - 1, filtered.length - 1)] : null
  const modelAName = modelA ? parseReportName(modelA).model : 'A'
  const modelBName = modelB ? parseReportName(modelB).model : 'B'

  return (
    <div className="flex flex-col gap-4">
      {/* Model selectors */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-[var(--color-ink-muted)] mb-1 block">{t('multi.selectModelA')}</label>
          <select value={modelA} onChange={(e) => setModelA(e.target.value)} className="w-full px-2 py-1.5 text-sm rounded-md bg-[var(--color-surface)] border border-[var(--color-border)]">
            {selectedReports.map((r) => <option key={r} value={r}>{parseReportName(r).model}</option>)}
          </select>
        </div>
        <div>
          <label className="text-xs text-[var(--color-ink-muted)] mb-1 block">{t('multi.selectModelB')}</label>
          <select value={modelB} onChange={(e) => setModelB(e.target.value)} className="w-full px-2 py-1.5 text-sm rounded-md bg-[var(--color-surface)] border border-[var(--color-border)]">
            {selectedReports.map((r) => <option key={r} value={r}>{parseReportName(r).model}</option>)}
          </select>
        </div>
      </div>

      {/* Dataset selector */}
      {commonDs.length > 0 && (
        <div>
          <label className="text-xs text-[var(--color-ink-muted)] mb-1 block">{t('multi.selectDataset')}</label>
          <div className="flex flex-wrap gap-1">
            {commonDs.map((ds) => (
              <button key={ds} onClick={() => setSelectedDs(ds)} className={`px-3 py-1 text-xs rounded-full transition-colors ${selectedDs === ds ? 'bg-[var(--color-primary)] text-white' : 'bg-[var(--color-surface)] text-[var(--color-ink-muted)] border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)]'}`}>{ds}</button>
            ))}
          </div>
        </div>
      )}

      {/* Subset selector */}
      {subsets.length > 0 && (
        <div>
          <label className="text-xs text-[var(--color-ink-muted)] mb-1 block">{t('multi.selectSubset')}</label>
          <select value={selectedSubset} onChange={(e) => setSelectedSubset(e.target.value)} className="px-2 py-1.5 text-sm rounded-md bg-[var(--color-surface)] border border-[var(--color-border)]">
            <option value="">-- {t('multi.selectSubset')} --</option>
            {subsets.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      )}

      {mergedData.length > 0 && (
        <>
          {/* Filter + threshold */}
          <div className="flex flex-wrap items-center gap-4">
            <FilterBar modes={ANSWER_MODES} active={mode} onChange={(m) => { setMode(m); setPage(1) }} />
            <div className="flex items-center gap-2">
              <label className="text-xs text-[var(--color-ink-muted)]">{t('multi.scoreThreshold')}</label>
              <input type="number" value={threshold} step={0.01} min={0} max={1} onChange={(e) => { setThreshold(Number(e.target.value)); setPage(1) }} className="w-20 px-2 py-1 text-sm rounded-md bg-[var(--color-surface)] border border-[var(--color-border)]" />
            </div>
          </div>

          {/* Stats */}
          <div className="flex items-center justify-between bg-[var(--color-surface)] rounded-lg px-3 py-2 border border-[var(--color-border)]">
            <span className="text-sm">All: <b>{mergedData.length}</b> | Pass A: <b>{passA}</b> | Pass B: <b>{passB}</b> | Both Pass: <b>{passBoth}</b> | Both Fail: <b>{failBoth}</b></span>
            <Pagination page={page} total={filtered.length} onChange={setPage} />
          </div>

          {/* Side-by-side display */}
          {row && (
            <div className="flex flex-col gap-3">
              <div className="grid grid-cols-2 gap-3">
                <Panel title={t('multi.input')}><MarkdownRenderer content={row.A_Input} /></Panel>
                <Panel title={t('multi.goldAnswer')}><MarkdownRenderer content={formatVal(row.A_Gold)} /></Panel>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Panel title={`${modelAName} - Score`}><ScoreBadge score={row.A_NScore} threshold={threshold} /></Panel>
                <Panel title={`${modelBName} - Score`}><ScoreBadge score={row.B_NScore} threshold={threshold} /></Panel>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Panel title={`${modelAName} - ${t('multi.prediction')}`}><MarkdownRenderer content={formatVal(row.A_Pred)} /></Panel>
                <Panel title={`${modelBName} - ${t('multi.prediction')}`}><MarkdownRenderer content={formatVal(row.B_Pred)} /></Panel>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Panel title={`${modelAName} - ${t('multi.generated')}`}><MarkdownRenderer content={formatVal(row.A_Generated)} /></Panel>
                <Panel title={`${modelBName} - ${t('multi.generated')}`}><MarkdownRenderer content={formatVal(row.B_Generated)} /></Panel>
              </div>
            </div>
          )}
          {!row && filtered.length === 0 && <EmptyState text="No matching samples" />}
        </>
      )}
    </div>
  )
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)] p-3">
      <h5 className="text-xs font-medium text-[var(--color-ink-muted)] mb-2">{title}</h5>
      <div className="max-h-[300px] overflow-auto">{children}</div>
    </div>
  )
}

function formatVal(v: unknown): string {
  if (v === null || v === undefined) return ''
  if (typeof v === 'object') return '```json\n' + JSON.stringify(v, null, 2) + '\n```'
  return String(v)
}
