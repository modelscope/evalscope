import { useMemo, useState, type CSSProperties, type ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { formatMetric, getBoundedQualityRatio, getComparisonVerdict } from '@/domain/metric'
import type { MetricSemantics } from '@/domain/metric'
import { scoreBg, scoreColor } from '@/utils/colorScale'
import { directionHintKey } from '@/domain/report/primaryMetrics'
import {
  comparisonDeltaBackground,
  computeDeltaRanges,
  signedDifference,
} from '@/domain/compare/scoreMatrix'
import Badge from '@/components/ui/Badge'
import Select from '@/components/ui/Select'
import SegmentedControl from '@/components/ui/SegmentedControl'
import PlotlyChart from '@/components/charts/PlotlyChart'
import EmptyStateSystem from '@/components/common/EmptyStateSystem'
import { getCompareChartUrl } from '@/api/reports'
import { MODEL_PALETTE, type Translate } from '@/components/compare/compareSlots'

export default function ScoreMatrixTab({
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
  /** Dataset/metric comparison key -> semantics of that primary metric. */
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

  const deltaRanges = useMemo(
    () => computeDeltaRanges(dataRows, reportKeys, baselineReport),
    [baselineReport, dataRows, reportKeys],
  )

  const renderScoreCell = (
    score: number | null,
    baselineScore: number | null,
    semantics: MetricSemantics | undefined,
    isBest: boolean,
    isBaseline: boolean,
  ): ReactNode => {
    if (score == null || !Number.isFinite(score)) {
      return <span className="text-[var(--text-dim)]">—</span>
    }
    const hasBaseline = baselineScore != null && Number.isFinite(baselineScore)
    const delta = hasBaseline ? score - baselineScore : null
    const verdict = delta == null ? 'incomparable' : getComparisonVerdict(delta, semantics)

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
            isBaseline || verdict === 'equal' || verdict === 'incomparable'
              ? 'text-[var(--text-dim)]'
              : verdict === 'better'
                ? 'text-[var(--success)]'
                : 'text-[var(--danger)]',
          )}
        >
          {isBaseline ? t('compare.baseline') : delta == null ? '—' : signedDifference(delta, semantics)}
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
    if (score == null || !Number.isFinite(score)) return { backgroundColor: 'var(--bg-deep)' }
    const ratio = getBoundedQualityRatio(score, semantics)
    const hasBaseline = baselineScore != null && Number.isFinite(baselineScore)
    if (comparisonMode === 'baseline') {
      if (!hasBaseline) return { backgroundColor: 'var(--bg-deep)' }
      const delta = score - baselineScore
      return { backgroundColor: comparisonDeltaBackground(delta, deltaRanges[rangeKey] ?? 0, semantics) }
    }
    if (ratio == null) return { backgroundColor: 'var(--bg-deep)', color: 'var(--text)' }
    return { backgroundColor: scoreBg(ratio, 0.18), color: scoreColor(ratio) }
  }

  return (
    <div className="flex flex-col gap-6">
      <PlotlyChart
        src={getCompareChartUrl(rootPath, reportNames, 'radar')}
        fallbackTable={{
          columns: ['dataset', 'metric', ...reportKeys],
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
              <SegmentedControl
                options={[
                  { value: 'absolute', label: t('compare.absoluteScores') },
                  { value: 'baseline', label: t('compare.vsBaseline') },
                ]}
                value={comparisonMode}
                onChange={setComparisonMode}
                ariaLabel={t('compare.baselineMode')}
                className="min-w-[320px]"
              />
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
                    const sourceDatasetId = String(row.source_dataset_id ?? row.dataset)
                    const semantics = scoreSemantics[datasetId]
                    const hintKey = directionHintKey(semantics)
                    return (
                      <th
                        key={datasetId}
                        title={`${sourceDatasetId} · ${String(row.metric)}`}
                        className="min-w-[120px] border-l border-[var(--border-strong)] px-3 py-2 text-center type-table-xs !normal-case whitespace-nowrap first:border-l-0"
                      >
                        <span className="flex flex-col items-center justify-center gap-0.5">
                          <span>{String(row.dataset)}</span>
                          <span className="text-[var(--text-dim)]">
                            {String(row.metric)}
                            {hintKey && <span className="sr-only" aria-label={t(hintKey)} />}
                          </span>
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
                      const score = typeof row[rk] === 'number' ? row[rk] as number : null
                      const baselineScore = typeof row[baselineReport] === 'number'
                        ? row[baselineReport] as number
                        : null
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
