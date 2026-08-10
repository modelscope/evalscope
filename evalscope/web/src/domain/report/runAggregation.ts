import type { MetricSemantics } from '@/domain/metric'
import type { PerfRunSummary, ReportSummary } from '@/api/types'
import { primaryMetricsOf } from '@/domain/report/primaryMetrics'

/**
 * Aggregation of evaluation and performance runs by what they measure.
 *
 * A run is not the unit of interest. Re-running one benchmark against one model produces many
 * runs that describe a single thing, and a flat time-ordered feed renders them as that many
 * near-identical rows -- this project's own output directory contains a pair measured 70 times.
 * The unit here is therefore a cell: one model, one benchmark, one metric, plus the history of
 * every score ever recorded for it.
 *
 * Comparison stays inside a cell. Every point in a cell shares one metric and one scale, so its
 * spread is a real quantity; two different cells are never subtracted from each other. Ordering
 * across cells uses the spread normalized against the metric's own declared `value_range`, and a
 * metric with no range is not given a synthesized one -- it simply cannot be ordered that way and
 * says so.
 */

/** A single recorded score, kept with what is needed to reopen its source. */
export interface CellPoint {
  /** ISO timestamp of the run. */
  timestamp: string
  /** Score in the metric's native scale. */
  score: number
  /** Identifier used to navigate to the underlying run. */
  runId: string
}

/** What kind of run produced a cell, which decides where clicking it navigates to. */
export type CellKind = 'eval' | 'perf'

/** One model measured by one metric on one benchmark, with its full history. */
export interface AggregatedCell {
  kind: CellKind
  model: string
  /** Benchmark name for an eval cell; dataset or API type for a perf cell. */
  benchmark: string
  /** Final report metric name, as produced by the backend. */
  metricName: string
  semantics: MetricSemantics | null
  /** Every score recorded for this cell, oldest first. */
  history: CellPoint[]
}

/** Descriptive statistics over a cell's history, all in the metric's native scale. */
export interface CellStats {
  /** Most recent score. */
  latest: number
  /** Arithmetic mean across all runs. */
  mean: number
  /** Population standard deviation; `0` for a single run. */
  stddev: number
  min: number
  max: number
  /** `max - min`. Zero for a single run, so a lone result never reads as stable. */
  spread: number
  /**
   * Whether any recorded value falls outside the metric's declared `value_range`.
   *
   * This means the semantics do not describe the series: the usual cause is an adapter that changed
   * the scale it emits, leaving older reports on the other one. A percentage-point conversion is
   * only meaningful when both endpoints sit on the declared scale, so callers present the spread as
   * a plain quantity in that case instead of converting it.
   */
  outOfRange: boolean
  /**
   * `spread` as a fraction of the metric's own `value_range`, or `null` without one.
   *
   * This is the only cross-cell ordering key: it puts a 50pp accuracy swing and a 50pp F1 swing on
   * the same footing without pretending a throughput in tokens/s can join them. Capped at 1, so a
   * series holding values from two different scales still sorts among the widest rather than
   * reporting a ratio no bounded metric can reach.
   */
  relativeSpread: number | null
  /** Number of recorded runs. */
  runs: number
}

/** A cell paired with its statistics, ready to render. */
export interface AggregatedRow {
  cell: AggregatedCell
  stats: CellStats
}

/** Key identifying a cell, used for React keys and for merging runs. */
export function cellKey(cell: Pick<AggregatedCell, 'kind' | 'model' | 'benchmark' | 'metricName'>): string {
  return `${cell.kind}\u0000${cell.model}\u0000${cell.benchmark}\u0000${cell.metricName}`
}

/**
 * Compute descriptive statistics for a history.
 *
 * @param history Recorded points, in any order.
 * @param semantics Metric contract, consulted only for `value_range`.
 * @returns Statistics, or `null` when the history holds no scores.
 */
export function computeCellStats(
  history: CellPoint[],
  semantics: MetricSemantics | null | undefined,
): CellStats | null {
  const scores = history.map((point) => point.score).filter((score) => Number.isFinite(score))
  if (scores.length === 0) {
    return null
  }
  const sum = scores.reduce((acc, score) => acc + score, 0)
  const mean = sum / scores.length
  const variance = scores.reduce((acc, score) => acc + (score - mean) ** 2, 0) / scores.length
  const min = Math.min(...scores)
  const max = Math.max(...scores)
  const spread = max - min

  // Normalizing needs a declared range. An unbounded metric (throughput, token counts) has none,
  // so it reports `null` rather than borrowing a denominator such as the mean, which would make a
  // near-zero mean look infinitely unstable.
  const range = semantics?.value_range
  const width = range ? range.max - range.min : 0
  // Capped: values from two different scales can differ by far more than the range is wide, and an
  // uncapped ratio would report an instability no bounded metric can actually reach.
  const relativeSpread = width > 0 ? Math.min(1, spread / width) : null
  const outOfRange = range ? scores.some((score) => score < range.min || score > range.max) : false

  return {
    latest: history[history.length - 1].score,
    mean,
    stddev: Math.sqrt(variance),
    min,
    max,
    spread,
    outOfRange,
    relativeSpread,
    runs: scores.length,
  }
}

/**
 * Aggregate evaluation reports and performance runs into cells.
 *
 * Each report contributes one point per dataset it covers, taken from that dataset's primary
 * metric, so a multi-dataset run is split into the cells it actually measured. Each performance
 * run contributes its best throughput.
 *
 * @param reports Evaluation report summaries.
 * @param perfRuns Performance run summaries.
 * @param perfSemantics Semantics map from the perf list response, keyed by API path.
 * @returns Rows ordered by decreasing spread; see {@link compareByInstability}.
 */
export function aggregateRuns(
  reports: ReportSummary[],
  perfRuns: PerfRunSummary[],
  perfSemantics: Record<string, MetricSemantics> = {},
): AggregatedRow[] {
  const cells = new Map<string, AggregatedCell>()

  const push = (cell: Omit<AggregatedCell, 'history'>, point: CellPoint) => {
    const key = cellKey(cell)
    const existing = cells.get(key)
    if (existing) {
      const latestTimestamp = existing.history.reduce(
        (latest, historyPoint) => historyPoint.timestamp > latest ? historyPoint.timestamp : latest,
        '',
      )
      existing.history.push(point)
      // A later run's semantics win: the catalog may have gained a declaration since the older run
      // was written, and the newer resolution is the more accurate one.
      if (point.timestamp >= latestTimestamp && cell.semantics) {
        existing.semantics = cell.semantics
      }
      return
    }
    cells.set(key, { ...cell, history: [point] })
  }

  for (const report of reports) {
    for (const ref of primaryMetricsOf(report)) {
      if (ref.score == null) continue
      push(
        {
          kind: 'eval',
          model: report.model_name,
          benchmark: ref.dataset_name || report.dataset_name,
          metricName: ref.metric_name,
          semantics: ref.semantics ?? null,
        },
        { timestamp: report.timestamp || '', score: ref.score, runId: report.name },
      )
    }
  }

  const rpsSemantics = perfSemantics.best_rps ?? null
  for (const run of perfRuns) {
    if (run.best_rps == null || !Number.isFinite(run.best_rps)) continue
    push(
      {
        kind: 'perf',
        model: run.model,
        benchmark: run.dataset || run.api_type || 'perf',
        metricName: rpsSemantics?.metric_name ?? 'Best RPS',
        semantics: rpsSemantics,
      },
      { timestamp: run.timestamp || '', score: run.best_rps, runId: run.path },
    )
  }

  const rows: AggregatedRow[] = []
  for (const cell of cells.values()) {
    cell.history.sort((a, b) => a.timestamp.localeCompare(b.timestamp))
    const stats = computeCellStats(cell.history, cell.semantics)
    if (stats) rows.push({ cell, stats })
  }
  return rows.sort(compareByInstability)
}

/**
 * Order rows so the least reproducible results come first.
 *
 * The user reads this to find results they should not trust yet, so a wide spread outranks a
 * narrow one. Cells whose metric declares no range cannot be compared on that scale and are placed
 * after those that can, ordered among themselves by run count -- deliberately not by raw spread,
 * since a spread in tokens/s and one in seconds are not the same quantity.
 */
export function compareByInstability(a: AggregatedRow, b: AggregatedRow): number {
  const left = a.stats.relativeSpread
  const right = b.stats.relativeSpread
  if (left != null && right != null && left !== right) return right - left
  if (left != null && right == null) return -1
  if (left == null && right != null) return 1
  if (a.stats.runs !== b.stats.runs) return b.stats.runs - a.stats.runs
  const byModel = a.cell.model.localeCompare(b.cell.model)
  return byModel !== 0 ? byModel : a.cell.benchmark.localeCompare(b.cell.benchmark)
}

/** Vertical extent a trend is drawn against, in the metric's native scale. */
export interface TrendBounds {
  low: number
  high: number
  /**
   * Whether the extent comes from the metric's declared `value_range`.
   *
   * `false` means it was derived from the series itself, so bar heights are only comparable within
   * that one trend. Callers surface this rather than letting a series-relative height be read as an
   * absolute position.
   */
  declared: boolean
}

/**
 * Decide the extent to draw a trend against.
 *
 * A declared `value_range` wins, so a series wobbling between 50% and 60% draws a small wobble
 * instead of being stretched to fill the box and looking catastrophic. Two cases fall back to an
 * extent derived from the series itself, reported through `declared: false`:
 *
 * - an unbounded metric (throughput, latency) has no ceiling to normalize against;
 * - a series holding values outside its declared range is not described by that range, and clamping
 *   to it would draw every out-of-range run at exactly full height -- three runs of 96.77, 96.51 and
 *   0.9651 would become three identical full bars, hiding the very difference worth seeing.
 *
 * The fallback extent is `[min(0, ...scores), max]`: zero is a real floor for the quantities that
 * reach this branch, which keeps heights meaningful without inventing a ceiling.
 *
 * @param scores Recorded values; may be empty.
 * @param semantics Metric contract, consulted for `value_range`.
 * @returns Extent to map values onto, guaranteed to have `high > low`.
 */
export function trendBounds(
  scores: number[],
  semantics: MetricSemantics | null | undefined,
): TrendBounds {
  const range = semantics?.value_range
  const fits = range ? scores.every((score) => score >= range.min && score <= range.max) : false
  if (range && fits && range.max > range.min) {
    return { low: range.min, high: range.max, declared: true }
  }
  const max = scores.length > 0 ? Math.max(...scores) : 0
  const min = Math.min(0, ...scores)
  // A flat series at zero would give a zero-width extent, so widen it and draw everything level.
  return max > min ? { low: min, high: max, declared: false } : { low: min, high: min + 1, declared: false }
}

/**
 * Position a value within a trend's extent, as a ratio in `[0, 1]`.
 *
 * This is magnitude, never quality: it is not inverted for a `lower_is_better` metric, so a
 * falling error rate draws falling bars. Colour carries the quality, which is why
 * `getBoundedQualityRatio` stays a separate primitive.
 */
export function trendPosition(score: number, bounds: TrendBounds): number {
  const ratio = (score - bounds.low) / (bounds.high - bounds.low)
  return Math.min(1, Math.max(0, ratio))
}
