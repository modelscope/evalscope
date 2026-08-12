/**
 * Performance comparison data model — pure logic, no rendering.
 *
 * This module turns a set of `PerfDetailResponse` records into a decision-surface
 * model for the Performance_Compare_View: per-metric deltas (baseline / candidate
 * / absolute delta / percent delta) with a direction-aware verdict, low-sample
 * classification, symmetric config diff, always-recorded sample counts, and a
 * workload-mismatch flag.
 *
 * It has no dependency on React, the DOM, the network, the system clock or
 * randomness, so it is covered by property-based tests. The
 * rendering (delta table, baseline swap, low-sample de-emphasis, mismatch hint)
 * lives in the component layer; this module only produces the data
 * contract.
 *
 * Formatting is delegated to the single metric-formatting entry point
 * `formatMetric` so the same metric rounds identically everywhere.
 * `metricFormat.ts` is a forward dependency and may not exist yet;
 * this module imports it per the design contract and the checkpoint validates
 * the wiring once both are present.
 */

import type { PerfDetailResponse } from '../../api/types'
import { formatDifference, formatMetric, getComparisonVerdict } from '../metric'
import type { FormattedMetric, MetricSemantics } from '../metric'

/**
 * Direction-aware verdict for a single metric delta.
 *
 * This is an informational direction annotation, never a hard pass/fail gate:
 * - `improvement` / `regression` — candidate moved in the better / worse
 *   direction relative to baseline given the metric's `direction`;
 * - `neutral` — candidate equals baseline;
 * - `incomputable` — a value is missing on either side.
 */
export type DeltaVerdict = 'improvement' | 'regression' | 'neutral' | 'incomputable'

/** Per-metric comparison entry between baseline and candidate. */
export interface MetricDelta {
  /** Stable implementation-level metric key. */
  metricKey: string
  /** Human-readable label supplied by the API. */
  metricLabel: string
  /** Baseline value, formatted for display. */
  baseline: FormattedMetric
  /** Candidate value, formatted for display. */
  candidate: FormattedMetric
  /** Absolute delta (`candidate - baseline`), formatted in the metric's own form. */
  absoluteDelta: FormattedMetric
  /** Percent delta (`(candidate - baseline) / |baseline| * 100`), formatted as a `%` value. */
  percentDelta: FormattedMetric
  /** Direction-aware, informational verdict. */
  verdict: DeltaVerdict
}

/**
 * Low-sample tier for percentile statistics:
 * - `critical` — `n < 30` (strong warning, de-emphasize P90/P95/P99);
 * - `warn` — `30 <= n < 100` (warn/de-emphasize P95/P99);
 * - `ok` — `n >= 100` (show normally).
 */
export type SampleTier = 'critical' | 'warn' | 'ok'

/** A single differing entry in the config diff. */
export interface ConfigDiffEntry {
  /** Config key that differs or exists on only one side. */
  key: string
  /** Baseline value, or `''` when the key is absent on the baseline side. */
  baseline: string
  /** Candidate value, or `''` when the key is absent on the candidate side. */
  candidate: string
}

/**
 * Full comparison model consumed by the Performance_Compare_View.
 *
 * The model is always between a single `baselineId` and `candidateId`; sample
 * counts are always recorded and the config diff is a symmetric
 * difference over the two runs' configs.
 */
export interface PerfCompareModel {
  /** Id (run path) of the baseline run — defaults to the oldest run. */
  baselineId: string
  /** Id (run path) of the candidate run. */
  candidateId: string
  /** Per-metric deltas across the union of both runs' metrics. */
  deltas: MetricDelta[]
  /** Sample count per run id; always recorded. */
  sampleCounts: Record<string, number>
  /** True when the two runs used different workloads. */
  workloadMismatch: boolean
  /** Symmetric config differences between the two runs. */
  configDiff: ConfigDiffEntry[]
}

/** Lower bound (exclusive) for the `warn` tier / upper bound (exclusive) for `critical`. */
const CRITICAL_SAMPLE_THRESHOLD = 30
/** Lower bound (exclusive) for the `ok` tier / upper bound (exclusive) for `warn`. */
const WARN_SAMPLE_THRESHOLD = 100

/**
 * Classify a percentile sample size into a low-sample tier.
 *
 * Boundaries are explicit: `29 → 'critical'`, `30 → 'warn'`, `99 → 'warn'`,
 * `100 → 'ok'`. Non-positive / non-finite inputs collapse to `'critical'`.
 *
 * @param n - Non-negative percentile sample count.
 * @returns The low-sample tier for `n`.
 */
export function classifySampleSize(n: number): SampleTier {
  if (!Number.isFinite(n) || n < CRITICAL_SAMPLE_THRESHOLD) return 'critical'
  if (n < WARN_SAMPLE_THRESHOLD) return 'warn'
  return 'ok'
}

/** Semantics of a percent-change value: a plain `%`-suffixed number, 2 decimals. */
const PERCENT_DELTA_SEMANTICS: MetricSemantics = {
  semantic_id: 'diagnostic.unspecified',
  metric_name: 'Change',
  role: 'diagnostic',
  direction: 'none',
  display_kind: 'number',
  display_unit: '%',
  display_precision: 2,
  contract_version: 1,
}

/**
 * Coerce a raw summary-row cell into a finite number, or `null` when it cannot
 * be interpreted as one (missing value → incomputable delta).
 */
function toNumeric(value: unknown): number | null {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null
  }
  if (typeof value === 'string') {
    const trimmed = value.trim().replace(/,/g, '').replace(/%$/, '')
    if (trimmed.length === 0) return null
    const parsed = Number(trimmed)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

interface WideMetricColumn {
  key: string
  columnIndex: number
  semantics: MetricSemantics
}

function getWideMetricColumns(run: PerfDetailResponse): WideMetricColumn[] {
  return run.summary_columns.flatMap((column, columnIndex) => {
    return column.semantics ? [{ key: column.key, columnIndex, semantics: column.semantics }] : []
  })
}

function toMetricMap(run: PerfDetailResponse, wideRow?: (string | number)[]): Map<string, number | null> {
  const map = new Map<string, number | null>()
  for (const { key, columnIndex } of getWideMetricColumns(run)) {
    map.set(key, wideRow ? toNumeric(wideRow[columnIndex]) : null)
  }
  return map
}

function wideRowConfig(run: PerfDetailResponse, row: (string | number)[]): Record<string, string> {
  const metricIndexes = new Set(getWideMetricColumns(run).map(({ columnIndex }) => columnIndex))
  const config: Record<string, string> = {}
  run.summary_columns.forEach((column, index) => {
    if (metricIndexes.has(index)) return
    config[column.key] = String(row[index] ?? '').trim()
  })
  return config
}

function configIdentity(config: Record<string, string>): string {
  return JSON.stringify(Object.entries(config).sort(([a], [b]) => a.localeCompare(b)))
}

function matchingWideRows(
  baseline: PerfDetailResponse,
  candidate: PerfDetailResponse,
): {
  baseline: (string | number)[]
  baselineIndex: number
  candidate: (string | number)[]
  candidateIndex: number
} | null {
  const candidates = new Map(
    candidate.summary_rows.map((row, index) => [
      configIdentity(wideRowConfig(candidate, row)),
      { row, index },
    ]),
  )
  for (const [baselineIndex, row] of baseline.summary_rows.entries()) {
    const match = candidates.get(configIdentity(wideRowConfig(baseline, row)))
    if (match) {
      return {
        baseline: row,
        baselineIndex,
        candidate: match.row,
        candidateIndex: match.index,
      }
    }
  }
  return null
}

/** Build a single `MetricDelta` for one metric key across both runs. */
function buildMetricDelta(
  metricKey: string,
  metricLabel: string,
  baselineValue: number | null,
  candidateValue: number | null,
  semantics: MetricSemantics | null | undefined,
): MetricDelta {
  const baseline = formatMetric(baselineValue, semantics)
  const candidate = formatMetric(candidateValue, semantics)

  const computable = baselineValue !== null && candidateValue !== null
  const absoluteValue = computable ? candidateValue - baselineValue : null
  // Percent change is undefined when the baseline is zero.
  const percentValue =
    computable && baselineValue !== 0 ? ((candidateValue - baselineValue) / Math.abs(baselineValue)) * 100 : null

  // A delta of a percent-rendered metric is expressed in percentage points, not re-scaled.
  const absoluteDelta = formatDifference(absoluteValue, semantics)
  const percentDelta = formatMetric(percentValue, PERCENT_DELTA_SEMANTICS)
  // Diagnostic fields (request counts, cache details, failures) return 'incomparable', which is
  // what keeps them out of the winner decision.
  const rawVerdict = computable ? getComparisonVerdict(candidateValue - baselineValue, semantics) : 'incomparable'
  const verdict: DeltaVerdict = !computable || rawVerdict === 'incomparable'
    ? 'incomputable'
    : rawVerdict === 'equal' ? 'neutral' : rawVerdict === 'better' ? 'improvement' : 'regression'

  return { metricKey, metricLabel, baseline, candidate, absoluteDelta, percentDelta, verdict }
}

/** Parse a run timestamp into epoch millis; unparseable timestamps sort as "oldest". */
function timestampOf(run: PerfDetailResponse): number {
  const parsed = Date.parse(run?.generated_at ?? '')
  return Number.isNaN(parsed) ? Number.NEGATIVE_INFINITY : parsed
}

/**
 * Pick the oldest run (smallest timestamp). Ties are broken by original index,
 * so the earliest-listed run wins, keeping the choice deterministic.
 */
function pickOldest(runs: PerfDetailResponse[]): PerfDetailResponse {
  return runs.reduce((oldest, run) => (timestampOf(run) < timestampOf(oldest) ? run : oldest))
}

/**
 * Pick the newest run (largest timestamp). Ties are broken by original index,
 * so the earliest-listed run wins on a tie.
 */
function pickNewest(runs: PerfDetailResponse[]): PerfDetailResponse {
  return runs.reduce((newest, run) => (timestampOf(run) > timestampOf(newest) ? run : newest))
}

/** Extract the request count for one summary row, or the run total when no row is selected. */
function getSampleCount(run: PerfDetailResponse, rowIndex?: number): number {
  if (rowIndex !== undefined) {
    const rowCount = run.summary_sample_counts[rowIndex]
    return Number.isFinite(rowCount) ? rowCount : 0
  }
  return Number.isFinite(run.total_requests) ? run.total_requests : 0
}

function wideConfig(run: PerfDetailResponse): Record<string, string> {
  const metricIndexes = new Set(getWideMetricColumns(run).map(({ columnIndex }) => columnIndex))
  const config: Record<string, string> = {}
  run.summary_columns.forEach((column, index) => {
    if (metricIndexes.has(index)) return
    const values = Array.from(new Set(run.summary_rows.map((row) => String(row[index] ?? '')).filter(Boolean)))
    config[column.key] = values.join(', ')
  })
  config.number_of_requests = String(getSampleCount(run))
  return config
}

function selectedWideConfig(
  run: PerfDetailResponse,
  row?: (string | number)[],
  rowIndex?: number,
): Record<string, string> {
  if (!row) return wideConfig(run)
  return { ...wideRowConfig(run, row), number_of_requests: String(getSampleCount(run, rowIndex)) }
}

/**
 * Compute the symmetric config difference between two runs.
 *
 * A key is included when it exists on only one side, or exists on both sides
 * with different values. Keys present on both sides with equal values are
 * excluded. The determination is based on key presence and value, not on the
 * emitted string representation.
 */
function computeConfigDiff(
  baselineConfig: Record<string, string>,
  candidateConfig: Record<string, string>,
): ConfigDiffEntry[] {
  const keys = new Set<string>([...Object.keys(baselineConfig), ...Object.keys(candidateConfig)])
  const diff: ConfigDiffEntry[] = []
  for (const key of keys) {
    const inBaseline = Object.prototype.hasOwnProperty.call(baselineConfig, key)
    const inCandidate = Object.prototype.hasOwnProperty.call(candidateConfig, key)
    const baselineValue = inBaseline ? baselineConfig[key] : ''
    const candidateValue = inCandidate ? candidateConfig[key] : ''
    if (inBaseline && inCandidate && baselineValue === candidateValue) {
      continue // identical on both sides — not a difference.
    }
    diff.push({ key, baseline: baselineValue, candidate: candidateValue })
  }
  return diff
}

/** Empty model returned when there are no runs to compare. */
function emptyModel(): PerfCompareModel {
  return {
    baselineId: '',
    candidateId: '',
    deltas: [],
    sampleCounts: {},
    workloadMismatch: false,
    configDiff: [],
  }
}

/**
 * Build the performance comparison model between a baseline and a candidate run.
 *
 * Baseline selection: the run whose `path` equals `baselineId`, or the
 * oldest run when `baselineId` is empty or does not match any run. The candidate
 * is the newest of the remaining runs (or the baseline itself when only one run
 * is supplied).
 *
 * For every metric in the union of both runs' summary rows a `MetricDelta` is
 * produced with baseline / candidate / absolute delta / percent delta and a
 * direction-aware verdict. Metrics missing on either side are
 * marked `incomputable` while metrics present on both sides still yield deltas.
 * Sample counts are always recorded, workload mismatch is
 * flagged and the config diff is a symmetric difference.
 *
 * @param runs - Performance run details participating in the comparison.
 * @param baselineId - Explicitly selected baseline run id (path); empty selects the default.
 * @returns The comparison model, or an empty model when `runs` is empty.
 */
export function buildCompareModel(runs: PerfDetailResponse[], baselineId: string): PerfCompareModel {
  if (!Array.isArray(runs) || runs.length === 0) {
    return emptyModel()
  }

  const baseline = runs.find((run) => run.path === baselineId) ?? pickOldest(runs)
  const others = runs.filter((run) => run !== baseline)
  const candidate = others.length > 0 ? pickNewest(others) : baseline

  const semanticsByField = new Map<string, MetricSemantics>()
  const labelsByField = new Map<string, string>()
  for (const run of [baseline, candidate]) {
    for (const column of run.summary_columns) {
      if (column.semantics) {
        semanticsByField.set(column.key, column.semantics)
        labelsByField.set(column.key, column.label)
      }
    }
  }

  const matchedRows = matchingWideRows(baseline, candidate)
  const oneSideMissing = baseline.summary_rows.length === 0 || candidate.summary_rows.length === 0
  const baselineRowIndex = matchedRows?.baselineIndex ?? (oneSideMissing && baseline.summary_rows.length ? 0 : undefined)
  const candidateRowIndex = matchedRows?.candidateIndex ?? (oneSideMissing && candidate.summary_rows.length ? 0 : undefined)
  const baselineMetrics = matchedRows || oneSideMissing
    ? toMetricMap(baseline, baselineRowIndex === undefined ? undefined : baseline.summary_rows[baselineRowIndex])
    : new Map<string, number | null>()
  const candidateMetrics = matchedRows || oneSideMissing
    ? toMetricMap(candidate, candidateRowIndex === undefined ? undefined : candidate.summary_rows[candidateRowIndex])
    : new Map<string, number | null>()

  // Union of metric keys, baseline order first then candidate-only keys.
  const metricKeys: string[] = [...baselineMetrics.keys()]
  for (const key of candidateMetrics.keys()) {
    if (!baselineMetrics.has(key)) metricKeys.push(key)
  }

  const deltas = metricKeys.map((key) =>
    buildMetricDelta(
      key,
      labelsByField.get(key) ?? key,
      baselineMetrics.get(key) ?? null,
      candidateMetrics.get(key) ?? null,
      semanticsByField.get(key),
    ),
  )

  const sampleCounts: Record<string, number> = {
    [baseline.path]: getSampleCount(baseline, baselineRowIndex),
    [candidate.path]: getSampleCount(candidate, candidateRowIndex),
  }

  const baselineConfig = selectedWideConfig(baseline, matchedRows?.baseline, baselineRowIndex)
  const candidateConfig = selectedWideConfig(candidate, matchedRows?.candidate, candidateRowIndex)
  const configLabels = new Map<string, string>([['number_of_requests', 'Number of requests']])
  for (const run of [baseline, candidate]) {
    for (const column of run.summary_columns) {
      if (!column.semantics) configLabels.set(column.key, column.label)
    }
  }
  const configDiff = computeConfigDiff(baselineConfig, candidateConfig).map((entry) => ({
    ...entry,
    key: configLabels.get(entry.key) ?? entry.key,
  }))
  const workloadMismatch = baseline !== candidate && (
    matchedRows === null || baseline.dataset.trim().toLowerCase() !== candidate.dataset.trim().toLowerCase()
  )

  return {
    baselineId: baseline.path,
    candidateId: candidate.path,
    deltas,
    sampleCounts,
    workloadMismatch,
    configDiff,
  }
}
