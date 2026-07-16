/**
 * Performance comparison data model — pure logic, no rendering (Req 9, task 13.1).
 *
 * This module turns a set of `PerfDetailResponse` records into a decision-surface
 * model for the Performance_Compare_View: per-metric deltas (baseline / candidate
 * / absolute delta / percent delta) with a direction-aware verdict, low-sample
 * classification, symmetric config diff, always-recorded sample counts, and a
 * workload-mismatch flag.
 *
 * It has no dependency on React, the DOM, the network, the system clock or
 * randomness, so it is the target of the property tests in tasks 13.2–13.7. The
 * rendering (delta table, baseline swap, low-sample de-emphasis, mismatch hint)
 * lives in the component layer (task 13.10); this module only produces the data
 * contract.
 *
 * Formatting is delegated to the single metric-formatting entry point
 * `formatMetric` so the same metric rounds identically everywhere (Req 8.10).
 * `metricFormat.ts` (task 3.2) is a forward dependency and may not exist yet;
 * this module imports it per the design contract and the checkpoint validates
 * the wiring once both are present.
 */

import type { PerfDetailResponse } from '../../api/types'
import type { FormattedMetric } from '../metric/metricFormat'
import { formatMetric } from '../metric/metricFormat'
import type { MetricDirection, MetricDisplaySpec } from '../metric/MetricDisplaySpec'
import { DEFAULT_METRIC_SPEC } from '../metric/MetricDisplaySpec'
import { getMetricSpec } from '../metric/registry'

/**
 * Direction-aware verdict for a single metric delta (Req 9.4, 9.5).
 *
 * This is an informational direction annotation, never a hard pass/fail gate:
 * - `improvement` / `regression` — candidate moved in the better / worse
 *   direction relative to baseline given the metric's `direction`;
 * - `neutral` — candidate equals baseline;
 * - `incomputable` — a value is missing on either side (Req 9.14).
 */
export type DeltaVerdict = 'improvement' | 'regression' | 'neutral' | 'incomputable'

/** Per-metric comparison entry between baseline and candidate (Req 9.1). */
export interface MetricDelta {
  /** Implementation-level metric name (the summary-row label). */
  metricKey: string
  /** Baseline value, formatted for display. */
  baseline: FormattedMetric
  /** Candidate value, formatted for display. */
  candidate: FormattedMetric
  /** Absolute delta (`candidate - baseline`), formatted in the metric's own form. */
  absoluteDelta: FormattedMetric
  /** Percent delta (`(candidate - baseline) / |baseline| * 100`), formatted as a `%` value. */
  percentDelta: FormattedMetric
  /** Direction-aware, informational verdict (Req 9.4, 9.5, 9.14). */
  verdict: DeltaVerdict
}

/**
 * Low-sample tier for percentile statistics (Req 9.6–9.8):
 * - `critical` — `n < 30` (strong warning, de-emphasize P90/P95/P99);
 * - `warn` — `30 <= n < 100` (warn/de-emphasize P95/P99);
 * - `ok` — `n >= 100` (show normally).
 */
export type SampleTier = 'critical' | 'warn' | 'ok'

/** A single differing entry in the config diff (Req 9.13). */
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
 * counts are always recorded (Req 9.9) and the config diff is a symmetric
 * difference over the two runs' configs (Req 9.13).
 */
export interface PerfCompareModel {
  /** Id (run path) of the baseline run — defaults to the oldest run (Req 9.2). */
  baselineId: string
  /** Id (run path) of the candidate run. */
  candidateId: string
  /** Per-metric deltas across the union of both runs' metrics (Req 9.1, 9.14). */
  deltas: MetricDelta[]
  /** Sample count per run id; always recorded (Req 9.9). */
  sampleCounts: Record<string, number>
  /** True when the two runs used different workloads (Req 9.10). */
  workloadMismatch: boolean
  /** Symmetric config differences between the two runs (Req 9.13). */
  configDiff: ConfigDiffEntry[]
}

/** Lower bound (exclusive) for the `warn` tier / upper bound (exclusive) for `critical`. */
const CRITICAL_SAMPLE_THRESHOLD = 30
/** Lower bound (exclusive) for the `ok` tier / upper bound (exclusive) for `warn`. */
const WARN_SAMPLE_THRESHOLD = 100

/**
 * Classify a percentile sample size into a low-sample tier (Req 9.6–9.8).
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

/** Identity locale resolver used for internal formatting (labels are localized at the render layer). */
const identityT = (key: string): string => key

/** Display spec for a percent-delta value: a plain `%`-suffixed number, 2 decimals. */
const PERCENT_DELTA_SPEC: MetricDisplaySpec = {
  ...DEFAULT_METRIC_SPEC,
  unit: '%',
  rawPrecision: 2,
}

/** Absolute delta for a 0-100 success rate is expressed in percentage points. */
const PERCENTAGE_POINT_SPEC: MetricDisplaySpec = {
  ...DEFAULT_METRIC_SPEC,
  unit: 'pp',
  rawPrecision: 1,
}

/**
 * Coerce a raw summary-row cell into a finite number, or `null` when it cannot
 * be interpreted as one (missing value → incomputable delta, Req 9.14).
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
}

const WIDE_METRIC_KEYS: Record<string, string> = {
  rps: 'rps',
  'avg lat s': 'latency',
  'p99 lat s': 'p99_latency_s',
  'avg ttft ms': 'ttft_ms',
  'p99 ttft ms': 'p99_ttft_ms',
  'avg tpot ms': 'tpot_ms',
  'p99 tpot ms': 'p99_tpot_ms',
  'gen tok s': 'throughput',
  'success rate': 'success_rate',
}

function normalizeColumn(column: string): string {
  return column.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim()
}

function isVerticalSummary(run: PerfDetailResponse): boolean {
  const columns = run.summary_columns.map(normalizeColumn)
  return columns[0] === 'metric' && columns[1] === 'value'
}

function getWideMetricColumns(run: PerfDetailResponse): WideMetricColumn[] {
  return run.summary_columns.flatMap((column, columnIndex) => {
    const key = WIDE_METRIC_KEYS[normalizeColumn(column)]
    return key ? [{ key, columnIndex }] : []
  })
}

function selectBestObserved(values: number[], metricKey: string): number | null {
  if (values.length === 0) return null
  const { spec } = getMetricSpec(metricKey)
  return spec.direction === 'lower-is-better' ? Math.min(...values) : Math.max(...values)
}

/**
 * Build an ordered metric map from a run's summary rows.
 *
 * Each row is `[metricName, value, ...]`; the first cell is the key and the
 * second is the (possibly non-numeric) value. Insertion order is preserved so
 * deltas follow the baseline's natural metric order.
 */
function toMetricMap(run: PerfDetailResponse): Map<string, number | null> {
  const map = new Map<string, number | null>()
  const rows = Array.isArray(run.summary_rows) ? run.summary_rows : []
  if (!isVerticalSummary(run)) {
    for (const { key, columnIndex } of getWideMetricColumns(run)) {
      const values = rows.flatMap((row) => {
        const value = toNumeric(row[columnIndex])
        return value === null ? [] : [value]
      })
      map.set(key, selectBestObserved(values, key))
    }
    return map
  }
  for (const row of rows) {
    if (!Array.isArray(row) || row.length === 0) continue
    const key = String(row[0])
    if (key.length === 0 || map.has(key)) continue
    map.set(key, toNumeric(row[1]))
  }
  return map
}

/** Compute the direction-aware verdict for a metric (Req 9.4, 9.5, 9.14). */
function computeVerdict(
  baseline: number | null,
  candidate: number | null,
  direction: MetricDirection,
): DeltaVerdict {
  if (baseline === null || candidate === null) return 'incomputable'
  if (candidate === baseline) return 'neutral'
  const candidateIsHigher = candidate > baseline
  const higherIsBetter = direction === 'higher-is-better'
  // improvement when the candidate moved in the "better" direction.
  return candidateIsHigher === higherIsBetter ? 'improvement' : 'regression'
}

/** Build a single `MetricDelta` for one metric key across both runs. */
function buildMetricDelta(
  metricKey: string,
  baselineValue: number | null,
  candidateValue: number | null,
): MetricDelta {
  const { spec } = getMetricSpec(metricKey)

  const baseline = formatMetric(baselineValue, spec, identityT)
  const candidate = formatMetric(candidateValue, spec, identityT)

  const computable = baselineValue !== null && candidateValue !== null
  const absoluteValue = computable ? candidateValue - baselineValue : null
  // Percent change is undefined when the baseline is zero.
  const percentValue =
    computable && baselineValue !== 0 ? ((candidateValue - baselineValue) / Math.abs(baselineValue)) * 100 : null

  const absoluteDelta = formatMetric(
    absoluteValue,
    metricKey === 'success_rate' ? PERCENTAGE_POINT_SPEC : spec,
    identityT,
  )
  const percentDelta = formatMetric(percentValue, PERCENT_DELTA_SPEC, identityT)
  const verdict = computeVerdict(baselineValue, candidateValue, spec.direction)

  return { metricKey, baseline, candidate, absoluteDelta, percentDelta, verdict }
}

/** Parse a run timestamp into epoch millis; unparseable timestamps sort as "oldest". */
function timestampOf(run: PerfDetailResponse): number {
  const parsed = Date.parse(run?.generated_at ?? '')
  return Number.isNaN(parsed) ? Number.NEGATIVE_INFINITY : parsed
}

/**
 * Pick the oldest run (smallest timestamp). Ties are broken by original index,
 * so the earliest-listed run wins, keeping the choice deterministic (Req 9.2).
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

/** Extract the number of requests for a run (used as its sample count, Req 9.9). */
function getSampleCount(run: PerfDetailResponse): number {
  const rows = Array.isArray(run.summary_rows) ? run.summary_rows : []
  for (const row of rows) {
    if (!Array.isArray(row) || row.length < 2) continue
    if (String(row[0]).toLowerCase() === 'number of requests') {
      const n = toNumeric(row[1])
      if (n !== null) return n
    }
  }
  for (const [key, value] of Object.entries(run.basic_info ?? {})) {
    if (normalizeColumn(key) !== 'total requests') continue
    const fromBasic = toNumeric(value)
    if (fromBasic !== null) return fromBasic
  }
  return 0
}

function wideConfig(run: PerfDetailResponse): Record<string, string> | null {
  if (isVerticalSummary(run) || run.summary_rows.length === 0) return null

  const metricIndexes = new Set(getWideMetricColumns(run).map(({ columnIndex }) => columnIndex))
  const labels: Record<string, string> = {
    'conc': 'Concurrency',
    'rate': 'Request rate',
  }
  const config: Record<string, string> = {}
  run.summary_columns.forEach((column, index) => {
    if (metricIndexes.has(index)) return
    const normalized = normalizeColumn(column)
    const label = labels[normalized]
    if (!label) return
    const values = Array.from(new Set(run.summary_rows.map((row) => String(row[index] ?? '')).filter(Boolean)))
    config[label] = values.join(', ')
  })
  config['Number of requests'] = String(getSampleCount(run))
  return config
}

function comparisonConfig(run: PerfDetailResponse): Record<string, string> {
  return wideConfig(run) ?? (run.best_config ?? {})
}

/**
 * Compute the symmetric config difference between two runs (Req 9.13).
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

/** Trimmed, case-insensitive workload identity for a run (its dataset). */
function workloadIdentity(run: PerfDetailResponse): string {
  const config = wideConfig(run)
  const workload = config
    ? Object.entries(config).sort(([a], [b]) => a.localeCompare(b))
    : []
  return JSON.stringify([(run?.dataset ?? '').trim().toLowerCase(), workload])
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
 * Baseline selection (Req 9.2): the run whose `path` equals `baselineId`, or the
 * oldest run when `baselineId` is empty or does not match any run. The candidate
 * is the newest of the remaining runs (or the baseline itself when only one run
 * is supplied).
 *
 * For every metric in the union of both runs' summary rows a `MetricDelta` is
 * produced with baseline / candidate / absolute delta / percent delta and a
 * direction-aware verdict (Req 9.1, 9.4, 9.5). Metrics missing on either side are
 * marked `incomputable` while metrics present on both sides still yield deltas
 * (Req 9.14). Sample counts are always recorded (Req 9.9), workload mismatch is
 * flagged (Req 9.10) and the config diff is a symmetric difference (Req 9.13).
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

  const baselineMetrics = toMetricMap(baseline)
  const candidateMetrics = toMetricMap(candidate)

  // Union of metric keys, baseline order first then candidate-only keys.
  const metricKeys: string[] = [...baselineMetrics.keys()]
  for (const key of candidateMetrics.keys()) {
    if (!baselineMetrics.has(key)) metricKeys.push(key)
  }

  const deltas = metricKeys.map((key) =>
    buildMetricDelta(key, baselineMetrics.get(key) ?? null, candidateMetrics.get(key) ?? null),
  )

  const sampleCounts: Record<string, number> = {
    [baseline.path]: getSampleCount(baseline),
    [candidate.path]: getSampleCount(candidate),
  }

  const configDiff = computeConfigDiff(comparisonConfig(baseline), comparisonConfig(candidate))
  const workloadMismatch = baseline !== candidate && workloadIdentity(baseline) !== workloadIdentity(candidate)

  return {
    baselineId: baseline.path,
    candidateId: candidate.path,
    deltas,
    sampleCounts,
    workloadMismatch,
    configDiff,
  }
}
