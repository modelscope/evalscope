import { Fragment, useState } from 'react'
import { ChevronDown, ChevronRight, ChevronUp, FileText, Gauge } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import { formatDifference, formatMetric, getBoundedQualityRatio } from '@/domain/metric'
import { scoreColor } from '@/utils/colorScale'
import MetricTrend from '@/components/charts/MetricTrend'
import { cellKey } from '@/domain/report/runAggregation'
import type { AggregatedRow, CellPoint } from '@/domain/report/runAggregation'

/**
 * Results grouped by what they measure, with each group's history behind a disclosure.
 *
 * One row is a model measured by one metric on one benchmark, not a run. Repeating a benchmark
 * produces many runs describing a single thing -- this project's own output directory holds a pair
 * measured 70 times -- and listing them flat renders 70 near-identical rows. Collapsed into a row
 * with its shape and its statistics, the same data answers a question the flat list could not:
 * whether the number can be trusted yet.
 *
 * Nothing here labels a result as good, bad or broken. The statistics are plain quantities, and the
 * default order is the one the reader can verify at a glance -- most recent first -- rather than a
 * ranking the page invented.
 */

/**
 * Column the table can be ordered by.
 *
 * `Latest` is deliberately not among them. Every row carries its own metric and its own scale, so
 * ordering the table by raw latest value would rank a 0.12 req/s throughput against a 0.95 accuracy
 * -- a comparison this module refuses to make anywhere else (see `runAggregation`: comparison stays
 * inside a cell). The four keys here are all scale-free.
 */
type SortKey = 'model' | 'benchmark' | 'runs' | 'lastRun'

interface SortState {
  key: SortKey
  descending: boolean
}

/** Whether a column reads better descending on its first click. */
const DESCENDING_FIRST: Record<SortKey, boolean> = {
  model: false,
  benchmark: false,
  runs: true,
  lastRun: true,
}

/**
 * Widths for the columns that hold a number.
 *
 * `w-[1%]` plus `whitespace-nowrap` makes a table cell shrink to its content, which hands all the
 * slack to Model and Benchmark -- the two columns holding text of unpredictable length. Without it
 * the leftover width is shared out proportionally and opens a gulf between the metric name and its
 * value, leaving the numbers stranded at opposite ends of the row.
 */
const NUMERIC_COLUMN = 'w-[1%] whitespace-nowrap'

/** Keeps the value column off the sparkline that follows it. */
const TREND_GUTTER = 'pl-5'

/**
 * The one column that absorbs the table's leftover width.
 *
 * With nothing claiming the slack it is shared between Model and Benchmark, which opens a void after
 * each of them. Parked on Benchmark instead, the identifying text stays tight on the left, the
 * figures stay tight on the right, and the empty space collects in one place between the two.
 */
const FLEXIBLE_COLUMN = 'w-full'

/**
 * Floor width for the Model column.
 *
 * Once Benchmark claims the slack, an unconstrained Model column shrinks to its *narrowest* word
 * fragment, which breaks `qwen-plus` across two lines and `qwen-vl-plus` across three. A floor keeps
 * ordinary names on one line while leaving `break-words` free to wrap a genuinely long one.
 */
const MODEL_COLUMN = 'min-w-[8rem]'

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

function lastRunAt(row: AggregatedRow): string {
  return row.cell.history[row.cell.history.length - 1]?.timestamp ?? ''
}

function formatShortTime(timestamp: string): string {
  return timestamp ? timestamp.replace('T', ' ').slice(5, 16) : ''
}

/** Order two rows by one column, ascending; the caller flips it. */
function compareBy(key: SortKey, a: AggregatedRow, b: AggregatedRow): number {
  switch (key) {
    case 'model':
      return a.cell.model.localeCompare(b.cell.model) || a.cell.benchmark.localeCompare(b.cell.benchmark)
    case 'benchmark':
      return a.cell.benchmark.localeCompare(b.cell.benchmark) || a.cell.model.localeCompare(b.cell.model)
    case 'runs':
      return a.stats.runs - b.stats.runs || a.cell.model.localeCompare(b.cell.model)
    case 'lastRun':
      return lastRunAt(a).localeCompare(lastRunAt(b)) || a.cell.model.localeCompare(b.cell.model)
  }
}

export default function AggregatedResults({ rows, onOpenRun }: AggregatedResultsProps) {
  const { t } = useLocale()
  // Time order by default: it is the one ordering the reader can confirm from the column itself.
  const [sort, setSort] = useState<SortState>({ key: 'lastRun', descending: true })
  const [expanded, setExpanded] = useState<string | null>(null)

  const sorted = [...rows].sort((a, b) => (sort.descending ? -1 : 1) * compareBy(sort.key, a, b))

  const toggleSort = (key: SortKey) => {
    setSort((current) =>
      current.key === key
        ? { key, descending: !current.descending }
        : { key, descending: DESCENDING_FIRST[key] },
    )
  }

  const header = (key: SortKey, label: string, className: string) => (
    <th
      scope="col"
      aria-sort={sort.key === key ? (sort.descending ? 'descending' : 'ascending') : 'none'}
      className={className}
    >
      <button
        type="button"
        onClick={() => toggleSort(key)}
        title={t('dashboard.sortBy', { column: label })}
        className="inline-flex items-center gap-1 font-semibold text-inherit transition-colors hover:text-[var(--text)] focus-visible:outline-2 focus-visible:outline-[var(--accent)]"
      >
        {label}
        {sort.key === key
          ? (sort.descending ? <ChevronDown size={12} /> : <ChevronUp size={12} />)
          // A placeholder keeps the label from shifting sideways when the sort moves to it.
          : <span className="w-3" aria-hidden="true" />}
      </button>
    </th>
  )

  return (
    <div className="overflow-x-auto rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)]">
      <table className="w-full">
        <thead>
          <tr className="border-b border-[var(--border)] text-left type-body-xs text-[var(--text-muted)]">
            <th scope="col" className="w-8 px-2 py-2.5" />
            {header('model', t('dashboard.model'), `px-2 py-2.5 ${MODEL_COLUMN}`)}
            {header('benchmark', t('dashboard.benchmark'), `px-2 py-2.5 ${FLEXIBLE_COLUMN}`)}
            <th scope="col" className={`hidden px-2 py-2.5 font-semibold xl:table-cell ${NUMERIC_COLUMN}`}>
              {t('reportDetail.metric')}
            </th>
            <th scope="col" className={`px-2 py-2.5 text-right font-semibold ${NUMERIC_COLUMN}`}>
              {t('dashboard.latest')}
            </th>
            <th scope="col" className={`hidden px-2 py-2.5 font-semibold sm:table-cell ${NUMERIC_COLUMN} ${TREND_GUTTER}`}>
              {t('dashboard.trend')}
            </th>
            {header('runs', t('dashboard.runsCol'), `hidden px-2 py-2.5 text-right md:table-cell ${NUMERIC_COLUMN}`)}
            {header('lastRun', t('dashboard.lastRun'), `hidden px-2 py-2.5 text-right lg:table-cell ${NUMERIC_COLUMN}`)}
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--border)]">
          {sorted.map((row) => {
            const key = cellKey(row.cell)
            const isOpen = expanded === key
            const { cell, stats } = row
            const quality = getBoundedQualityRatio(stats.latest, cell.semantics)
            const latest = formatMetric(stats.latest, cell.semantics)
            const label = metricLabel(row)
            const toggle = () => setExpanded(isOpen ? null : key)

            return (
              <Fragment key={key}>
                {/* The whole row is clickable, not just the chevron: the row is what the reader aims
                    at, and a 28px target at the far right of a wide table is the hardest part of it
                    to hit. The chevron stays a real button so the disclosure keeps a focusable,
                    labelled control -- `role="button"` on the `tr` would have bought the same click
                    area by destroying the row/cell semantics a table is read through. */}
                <tr
                  onClick={toggle}
                  className={[
                    'cursor-pointer transition-colors',
                    isOpen ? 'bg-[var(--bg-card2)]' : 'hover:bg-[var(--bg-card2)]',
                  ].join(' ')}
                >
                  <td className="px-2 py-2">
                    <button
                      type="button"
                      onClick={(event) => {
                        // The row handler already toggles; without this the two cancel out.
                        event.stopPropagation()
                        toggle()
                      }}
                      aria-expanded={isOpen}
                      aria-label={t(isOpen ? 'dashboard.collapseRow' : 'dashboard.expandRow')}
                      className="flex h-6 w-6 items-center justify-center rounded-[var(--radius-sm)] text-[var(--text-dim)] transition-colors hover:text-[var(--text)] focus-visible:outline-2 focus-visible:outline-[var(--accent)]"
                    >
                      {isOpen ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                    </button>
                  </td>
                  <td className={`px-2 py-2 ${MODEL_COLUMN}`}>
                    <span className="flex items-center gap-1.5">
                      <span className="shrink-0 text-[var(--text-dim)]" title={t(`dashboard.filter_${cell.kind}`)}>
                        {cell.kind === 'eval' ? <FileText size={13} /> : <Gauge size={13} />}
                      </span>
                      <span className="type-body-sm break-words text-[var(--text)]">{cell.model}</span>
                    </span>
                  </td>
                  <td className={`px-2 py-2 ${FLEXIBLE_COLUMN}`}>
                    <span className="type-body-sm break-words text-[var(--text)]">{cell.benchmark}</span>
                    {/* Below the width where the metric has a column, it rides with the benchmark so
                        the value is never shown without saying what it measures. */}
                    <span className="block type-caption text-[var(--text-muted)] xl:hidden">{label}</span>
                  </td>
                  <td className={`hidden px-2 py-2 type-caption text-[var(--text-muted)] xl:table-cell ${NUMERIC_COLUMN}`}>
                    <span title={cell.metricName}>{label}</span>
                  </td>
                  <td
                    className={`px-2 py-2 text-right type-caption-mono font-semibold tabular-nums ${NUMERIC_COLUMN}`}
                    style={{ color: quality == null ? 'var(--text)' : scoreColor(quality) }}
                  >
                    {latest.primary}
                  </td>
                  <td className={`hidden px-2 py-2 sm:table-cell ${NUMERIC_COLUMN} ${TREND_GUTTER}`}>
                    <MetricTrend
                      history={cell.history}
                      semantics={cell.semantics}
                      label={t('dashboard.trendLabel', { metric: label, runs: stats.runs })}
                    />
                  </td>
                  <td className={`hidden px-2 py-2 text-right type-caption-mono tabular-nums text-[var(--text-muted)] md:table-cell ${NUMERIC_COLUMN}`}>
                    {stats.runs}
                  </td>
                  <td className={`hidden px-2 py-2 text-right type-caption-mono tabular-nums text-[var(--text-dim)] lg:table-cell ${NUMERIC_COLUMN}`}>
                    {formatShortTime(lastRunAt(row))}
                  </td>
                </tr>

                {isOpen && (
                  <tr className="bg-[var(--bg-card2)]">
                    <td colSpan={8} className="px-4 pb-4 pt-1">
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

/** Width the history panel holds, so the statistics beside it never reflow with the run count. */
const TREND_PANEL_WIDTH = 'lg:w-[420px]'

/**
 * Full statistics for one cell, shown when its row is expanded.
 *
 * The history keeps a fixed width and scrolls sideways when it outgrows it. Letting it stretch
 * instead would make the panel's whole layout a function of how many times a benchmark happened to
 * be re-run, so opening two rows would give two different-looking panels.
 */
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
  // A spread is a difference, so a percent metric reports it in points rather than claiming a gap
  // between 50% and 100% is "50%" of anything. When the series holds values from outside the
  // declared range the scale demonstrably does not apply to it, and a point conversion would be as
  // unfounded as a percentage: it stays a plain quantity.
  const difference = (score: number) =>
    stats.outOfRange ? formatMetric(score, null).primary : formatDifference(score, cell.semantics).primary

  const figures: { label: string; text: string }[] = [
    { label: t('dashboard.statLatest'), text: value(stats.latest) },
    { label: t('dashboard.statMean'), text: value(stats.mean) },
    { label: t('dashboard.statRange'), text: `${value(stats.min)} – ${value(stats.max)}` },
    { label: t('dashboard.statSpread'), text: stats.runs > 1 ? difference(stats.spread) : '—' },
    { label: t('dashboard.statStddev'), text: stats.runs > 1 ? difference(stats.stddev) : '—' },
    { label: t('dashboard.statRuns'), text: String(stats.runs) },
  ]

  return (
    <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:gap-6">
      <div className={`w-full shrink-0 ${TREND_PANEL_WIDTH}`}>
        {stats.runs > 1 ? (
          <div className="overflow-x-auto rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-card)] p-3">
            <MetricTrend
              history={cell.history}
              semantics={cell.semantics}
              variant="detail"
              onSelect={(point) => onOpenRun(row, point)}
              label={t('dashboard.trendDetailLabel')}
            />
          </div>
        ) : (
          <p className="rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-card)] p-3 type-body-xs text-[var(--text-muted)]">
            {t('dashboard.singleRunHint')}
          </p>
        )}
      </div>

      <div className="min-w-0">
        {/* Each figure names itself. An earlier pass replaced these labels with bare glyphs -- a sigma
            for the mean, a wave for the standard deviation -- which only works for a reader who
            already knows which statistic is which, and that reader did not need the panel. */}
        <dl className="grid grid-cols-2 gap-x-8 gap-y-3 sm:grid-cols-3">
          {figures.map((figure) => (
            <div key={figure.label} className="flex flex-col gap-0.5">
              <dt className="type-caption text-[var(--text-muted)]">{figure.label}</dt>
              <dd className="type-body-sm font-mono tabular-nums text-[var(--text)]">{figure.text}</dd>
            </div>
          ))}
        </dl>

        {stats.outOfRange && (
          <p className="mt-3 max-w-prose type-body-xs text-[var(--text-muted)]">{t('dashboard.outOfRangeHint')}</p>
        )}
      </div>
    </div>
  )
}
