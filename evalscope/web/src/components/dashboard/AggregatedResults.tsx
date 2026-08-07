import { Fragment, useState } from 'react'
import { ChevronDown, ChevronRight, ExternalLink, FileText, Gauge } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { formatDifference, formatMetric, getBoundedQualityRatio } from '@/domain/metric'
import { scoreColor } from '@/utils/colorScale'
import MetricTrend from '@/components/charts/MetricTrend'
import { cellKey, compareByInstability, compareByRecency } from '@/domain/report/runAggregation'
import type { AggregatedRow, CellPoint } from '@/domain/report/runAggregation'

/**
 * Results grouped by what they measure, with each group's history behind a disclosure.
 *
 * One row is a model measured by one metric on one benchmark, not a run. Repeating a benchmark
 * produces many runs describing a single thing -- this project's own output directory holds a pair
 * measured 70 times -- and listing them flat renders 70 near-identical rows. Collapsed into a row
 * with its spread and its shape, the same data answers a question the flat list could not: whether
 * the number can be trusted yet.
 *
 * Nothing here labels a result as good, bad or broken. The spread is shown as a plain quantity and
 * used as the default sort, so wide swings surface at the top without the page asserting a
 * threshold that would be its own invention.
 */

export type ResultsSort = 'instability' | 'recency'

interface AggregatedResultsProps {
  rows: AggregatedRow[]
  /** Opens the run behind a given point. */
  onOpenRun: (row: AggregatedRow, point: CellPoint) => void
}

/** Direction arrow for a metric, matching the other surfaces. */
function directionArrow(semantics: AggregatedRow['cell']['semantics']): string {
  if (semantics?.direction === 'higher_is_better') return '↑'
  if (semantics?.direction === 'lower_is_better') return '↓'
  return ''
}

function metricLabel(row: AggregatedRow): string {
  const { semantics, metricName } = row.cell
  // A resolved metric shows its display name; otherwise the raw name is what identifies it.
  return semantics ? `${semantics.metric_name} ${directionArrow(semantics)}`.trimEnd() : metricName
}

export default function AggregatedResults({ rows, onOpenRun }: AggregatedResultsProps) {
  const { t } = useLocale()
  const [sort, setSort] = useState<ResultsSort>('instability')
  const [expanded, setExpanded] = useState<string | null>(null)

  const sorted = [...rows].sort(sort === 'instability' ? compareByInstability : compareByRecency)

  return (
    <div className="overflow-x-auto rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
      <div className="flex flex-wrap items-center gap-2 border-b border-[var(--border)] px-4 py-2.5">
        <div className="flex items-center gap-1 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-deep)] p-0.5">
          {(['instability', 'recency'] as const).map((key) => (
            <button
              key={key}
              type="button"
              onClick={() => setSort(key)}
              aria-pressed={sort === key}
              className={[
                'rounded-[var(--radius-sm)] px-2.5 py-1 type-body-xs transition-colors',
                sort === key
                  ? 'bg-[var(--accent)] text-[var(--text-on-filled)]'
                  : 'text-[var(--text-muted)] hover:text-[var(--text)]',
              ].join(' ')}
            >
              {t(`dashboard.sort_${key}`)}
            </button>
          ))}
        </div>
      </div>

      <table className="w-full">
        <thead>
          <tr className="border-b border-[var(--border)] text-left type-body-xs text-[var(--text-muted)]">
            <th scope="col" className="w-10 px-2 py-2.5" />
            <th scope="col" className="px-2 py-2.5 font-semibold">{t('dashboard.model')}</th>
            <th scope="col" className="px-2 py-2.5 font-semibold">{t('dashboard.benchmark')}</th>
            <th scope="col" className="hidden px-2 py-2.5 font-semibold xl:table-cell">
              {t('reportDetail.metric')}
            </th>
            <th scope="col" className="px-2 py-2.5 text-right font-semibold">{t('dashboard.latest')}</th>
            <th scope="col" className="hidden px-2 py-2.5 font-semibold sm:table-cell">{t('dashboard.trend')}</th>
            <th scope="col" className="hidden px-2 py-2.5 text-right font-semibold lg:table-cell">
              {t('dashboard.spread')}
            </th>
            <th scope="col" className="hidden px-2 py-2.5 text-right font-semibold md:table-cell">
              {t('dashboard.runsCol')}
            </th>
            <th scope="col" className="w-8 px-2 py-2.5" />
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--border)]">
          {sorted.map((row) => {
            const key = cellKey(row.cell)
            const isOpen = expanded === key
            const { cell, stats } = row
            const quality = getBoundedQualityRatio(stats.latest, cell.semantics)
            const latest = formatMetric(stats.latest, cell.semantics)
            // A spread is a difference, so a percent metric reports it in points rather than
            // claiming a gap between 50% and 100% is "50%" of anything. When the series holds values
            // from outside the declared range the scale demonstrably does not apply to it, and a
            // point conversion would be as unfounded as a percentage: it stays a plain quantity.
            const spread = stats.outOfRange
              ? formatMetric(stats.spread, null)
              : formatDifference(stats.spread, cell.semantics)
            const label = metricLabel(row)

            return (
              <Fragment key={key}>
                <tr className={isOpen ? 'bg-[var(--bg-card2)]' : undefined}>
                  <td className="px-2 py-2">
                    <span
                      title={t(`dashboard.filter_${cell.kind}`)}
                      className="flex h-7 w-7 items-center justify-center rounded-[var(--radius-sm)] bg-[var(--bg-card2)] text-[var(--text-muted)]"
                    >
                      {cell.kind === 'eval' ? <FileText size={14} /> : <Gauge size={14} />}
                    </span>
                  </td>
                  <td className="px-2 py-2 type-body-sm break-words text-[var(--text)]">{cell.model}</td>
                  <td className="px-2 py-2">
                    <span className="type-body-sm break-words text-[var(--text)]">{cell.benchmark}</span>
                    {/* Below the width where the metric has a column, it rides with the benchmark so
                        the value is never shown without saying what it measures. */}
                    <span className="block type-caption text-[var(--text-muted)] xl:hidden">{label}</span>
                  </td>
                  <td className="hidden px-2 py-2 type-caption text-[var(--text-muted)] xl:table-cell">
                    <span title={cell.metricName}>{label}</span>
                  </td>
                  <td
                    className="px-2 py-2 text-right type-caption-mono font-semibold tabular-nums"
                    style={{ color: quality == null ? 'var(--text)' : scoreColor(quality) }}
                  >
                    {latest.primary}
                  </td>
                  <td className="hidden px-2 py-2 sm:table-cell">
                    <MetricTrend
                      history={cell.history}
                      semantics={cell.semantics}
                      label={t('dashboard.trendLabel', { metric: label, runs: stats.runs })}
                    />
                  </td>
                  <td className="hidden px-2 py-2 text-right type-caption-mono tabular-nums text-[var(--text-muted)] lg:table-cell">
                    {stats.runs > 1 ? spread.primary : '—'}
                  </td>
                  <td className="hidden px-2 py-2 text-right type-caption-mono tabular-nums text-[var(--text-muted)] md:table-cell">
                    {stats.runs}
                  </td>
                  <td className="px-2 py-2">
                    <button
                      type="button"
                      onClick={() => setExpanded(isOpen ? null : key)}
                      aria-expanded={isOpen}
                      aria-label={t(isOpen ? 'dashboard.collapseRow' : 'dashboard.expandRow')}
                      className="flex h-7 w-7 items-center justify-center rounded-[var(--radius-sm)] text-[var(--text-dim)] transition-colors hover:bg-[var(--bg-card2)] hover:text-[var(--text)] focus-visible:outline-2 focus-visible:outline-[var(--accent)]"
                    >
                      {isOpen ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                    </button>
                  </td>
                </tr>

                {isOpen && (
                  <tr className="bg-[var(--bg-card2)]">
                    <td colSpan={9} className="px-4 pb-4 pt-1">
                      <StatsPanel row={row} onOpenRun={onOpenRun} />
                    </td>
                  </tr>
                )}
              </Fragment>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/** Full statistics for one cell, shown when its row is expanded. */
function StatsPanel({
  row,
  onOpenRun,
}: {
  row: AggregatedRow
  onOpenRun: (row: AggregatedRow, point: CellPoint) => void
}) {
  const { t } = useLocale()
  const { cell, stats } = row
  // Off-scale values are shown as recorded, with no conversion applied on top of them.
  const value = (score: number) =>
    stats.outOfRange ? formatMetric(score, null).primary : formatMetric(score, cell.semantics).primary
  const difference = (score: number) =>
    stats.outOfRange ? formatMetric(score, null).primary : formatDifference(score, cell.semantics).primary
  const latestPoint = cell.history[cell.history.length - 1]

  const figures: { label: string; text: string }[] = [
    { label: t('dashboard.statLatest'), text: value(stats.latest) },
    { label: t('dashboard.statMean'), text: value(stats.mean) },
    { label: t('dashboard.statRange'), text: `${value(stats.min)} – ${value(stats.max)}` },
    { label: t('dashboard.statSpread'), text: stats.runs > 1 ? difference(stats.spread) : '—' },
    { label: t('dashboard.statStddev'), text: stats.runs > 1 ? difference(stats.stddev) : '—' },
    { label: t('dashboard.statRuns'), text: String(stats.runs) },
  ]

  return (
    <div className="flex flex-col gap-3">
      <dl className="flex flex-wrap gap-x-6 gap-y-2">
        {figures.map((figure) => (
          <div key={figure.label} className="flex flex-col">
            <dt className="type-caption text-[var(--text-muted)]">{figure.label}</dt>
            <dd className="type-caption-mono tabular-nums text-[var(--text)]">{figure.text}</dd>
          </div>
        ))}
      </dl>

      {stats.outOfRange && (
        <p className="type-body-xs text-[var(--text-muted)]">{t('dashboard.outOfRangeHint')}</p>
      )}

      {stats.runs > 1 ? (
        <MetricTrend
          history={cell.history}
          semantics={cell.semantics}
          variant="detail"
          onSelect={(point) => onOpenRun(row, point)}
          label={t('dashboard.trendDetailLabel')}
          className="max-w-2xl"
        />
      ) : (
        <p className="type-body-xs text-[var(--text-muted)]">{t('dashboard.singleRunHint')}</p>
      )}

      <button
        type="button"
        onClick={() => latestPoint && onOpenRun(row, latestPoint)}
        className="inline-flex w-fit items-center gap-1.5 type-body-xs text-[var(--accent)] hover:underline"
      >
        <ExternalLink size={12} />
        {t('dashboard.openLatest')}
      </button>
    </div>
  )
}
