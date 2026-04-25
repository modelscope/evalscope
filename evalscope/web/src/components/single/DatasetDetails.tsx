import { useCallback, useEffect, useState } from 'react'
import { useReports } from '@/contexts/ReportsContext'
import { useLocale } from '@/contexts/LocaleContext'
import type { PredictionRow, ReportData } from '@/api/types'
import { getPredictions, getAnalysis, getDataFrame, getChartUrl } from '@/api/reports'
import ChartEmbed from '@/components/charts/ChartEmbed'
import ScoreHistogram from '@/components/charts/ScoreHistogram'
import DataTable from '@/components/common/DataTable'
import PredictionBrowser from './PredictionBrowser'
import MarkdownRenderer from '@/components/common/MarkdownRenderer'
import EmptyState from '@/components/common/EmptyState'

interface Props {
  reports: ReportData[]
  reportName: string
}

export default function DatasetDetails({ reports, reportName }: Props) {
  const { t } = useLocale()
  const { rootPath } = useReports()
  const [datasets, setDatasets] = useState<string[]>([])
  const [selectedDs, setSelectedDs] = useState('')
  const [subsets, setSubsets] = useState<string[]>([])
  const [selectedSubset, setSelectedSubset] = useState('')
  const [analysis, setAnalysis] = useState('')
  const [subsetTableData, setSubsetTableData] = useState<{ columns: string[]; data: Record<string, unknown>[] }>({
    columns: [],
    data: [],
  })
  const [predictions, setPredictions] = useState<PredictionRow[]>([])

  // Build dataset list from reports
  useEffect(() => {
    const ds = reports.map((r) => r.dataset_name)
    setDatasets([...new Set(ds)])
    if (ds.length) setSelectedDs(ds[0])
  }, [reports])

  // Load dataset details when selection changes
  useEffect(() => {
    if (!selectedDs || !reportName) return
    let cancelled = false

    const load = async () => {
      try {
        // Get dataframe for the selected dataset
        const dfRes = await getDataFrame(rootPath, reportName, 'dataset', selectedDs)
        if (cancelled) return

        // Extract subsets
        const subNames: string[] = []
        const rows = dfRes.data
        for (const row of rows) {
          const catCol = Object.keys(row).find((k) => k.startsWith('Cat.'))
          if (catCol && row[catCol] === '-') continue
          const name = String(row['Subset'] ?? '')
          if (name && !subNames.includes(name)) {
            subNames.push(name)
          }
        }
        setSubsets(subNames)
        setSubsetTableData({ columns: dfRes.columns, data: dfRes.data })
        setSelectedSubset('')
        setPredictions([])

        // Load analysis
        const analysisText = await getAnalysis(rootPath, reportName, selectedDs)
        if (!cancelled) setAnalysis(analysisText)
      } catch (e) {
        console.error('Failed to load dataset details:', e)
      }
    }
    load()
    return () => { cancelled = true }
  }, [selectedDs, reportName, rootPath])

  // Load predictions when subset changes
  const loadPredictions = useCallback(async () => {
    if (!selectedSubset || !reportName || !selectedDs) return
    try {
      const res = await getPredictions(rootPath, reportName, selectedDs, selectedSubset)
      setPredictions(res.predictions)
    } catch (e) {
      console.error('Failed to load predictions:', e)
      setPredictions([])
    }
  }, [rootPath, reportName, selectedDs, selectedSubset])

  useEffect(() => {
    loadPredictions()
  }, [loadPredictions])

  if (!reports.length || !reportName) return <EmptyState />

  return (
    <div className="flex flex-col gap-4">
      {/* Dataset selector */}
      <div>
        <label className="text-xs text-[var(--color-ink-muted)] mb-1 block">{t('single.selectDataset')}</label>
        <div className="flex flex-wrap gap-1">
          {datasets.map((ds) => (
            <button
              key={ds}
              onClick={() => setSelectedDs(ds)}
              className={`px-3 py-1 text-xs rounded-full transition-colors ${
                selectedDs === ds
                  ? 'bg-[var(--color-primary)] text-white'
                  : 'bg-[var(--color-surface)] text-[var(--color-ink-muted)] border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)]'
              }`}
            >
              {ds}
            </button>
          ))}
        </div>
      </div>

      {/* Analysis */}
      {analysis && analysis !== 'N/A' && (
        <details className="bg-[var(--color-surface)] rounded-lg border border-[var(--color-border)] p-3" open>
          <summary className="text-sm font-medium cursor-pointer">{t('single.reportAnalysis')}</summary>
          <div className="mt-2">
            <MarkdownRenderer content={analysis} />
          </div>
        </details>
      )}

      {/* Dataset scores chart via backend */}
      {selectedDs && (
        <div className="bg-[var(--color-surface)] rounded-lg p-2 border border-[var(--color-border)]">
          <ChartEmbed
            src={getChartUrl(rootPath, 'dataset_scores', { reportName, datasetName: selectedDs })}
            height={350}
          />
        </div>
      )}

      {/* Subset table */}
      {subsetTableData.data.length > 0 && (
        <DataTable columns={subsetTableData.columns} data={subsetTableData.data} scoreColumns={['Score']} />
      )}

      {/* Subset selector + predictions */}
      <div>
        <label className="text-xs text-[var(--color-ink-muted)] mb-1 block">{t('single.selectSubset')}</label>
        <select
          value={selectedSubset}
          onChange={(e) => setSelectedSubset(e.target.value)}
          className="px-2 py-1.5 text-sm rounded-md bg-[var(--color-surface)] border border-[var(--color-border)] focus:outline-none focus:border-[var(--color-primary)]"
        >
          <option value="">-- {t('single.selectSubset')} --</option>
          {subsets.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      {/* Score distribution histogram */}
      {(predictions.length > 0 || selectedSubset) && (
        <div className="glass-card rounded-xl p-4">
          <ScoreHistogram predictions={predictions} />
        </div>
      )}

      {predictions.length > 0 && <PredictionBrowser predictions={predictions} />}
    </div>
  )
}
