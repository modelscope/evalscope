import type { PerfDetailResponse } from '../../api/types'
import { formatDifference, formatMetric, getComparisonVerdict } from '../metric'
import type { FormattedMetric, MetricSemantics } from '../metric'

export type DeltaVerdict = 'improvement' | 'regression' | 'neutral' | 'incomputable'
export type SampleTier = 'critical' | 'warn' | 'ok'

export interface MetricCell {
  value: number | null
  formatted: FormattedMetric
  sampleCount: number
  ambiguous: boolean
}

export interface PerfMetricRow {
  key: string
  configIndex: number
  metricKey: string
  metricLabel: string
  semantics: MetricSemantics
  values: Record<string, MetricCell>
}

export interface WorkloadConfigEntry {
  key: string
  label: string
  value: string
}

export interface WorkloadGroup {
  key: string
  index: number
  config: WorkloadConfigEntry[]
  rows: PerfMetricRow[]
}

export interface ConfigDiffEntry {
  key: string
  label: string
  values: Record<string, string | null>
}

export interface PerfCompareModel {
  baselineId: string
  groups: WorkloadGroup[]
  workloadMismatch: boolean
  duplicateWorkloads: boolean
  configDiff: ConfigDiffEntry[]
}

export interface MetricComparison {
  absoluteValue: number | null
  percentValue: number | null
  absoluteDelta: FormattedMetric
  percentDelta: FormattedMetric
  verdict: DeltaVerdict
}

export type InputLengthIssueCode =
  | 'missingConfiguration'
  | 'notRandom'
  | 'variableLength'
  | 'openLoop'
  | 'missingWorkload'
  | 'duplicatePoint'

export interface InputLengthIssue {
  code: InputLengthIssueCode
  runPath?: string
}

export interface InputLengthPoint {
  runPath: string
  inputLength: number
  concurrency: number
  requests: number
}

export interface InputLengthEligibility {
  eligible: boolean
  issue: InputLengthIssue | null
  points: InputLengthPoint[]
}

type SummaryRow = PerfDetailResponse['summary_rows'][number]
type SummaryColumn = PerfDetailResponse['summary_columns'][number]
type MetricColumn = SummaryColumn & { semantics: MetricSemantics }

type PendingMetricCell = { value: number | null; sampleCount: number; ambiguous: boolean }
type PendingMetricRow = Omit<PerfMetricRow, 'values'> & { values: Record<string, PendingMetricCell> }
type PendingWorkloadGroup = Omit<WorkloadGroup, 'rows'> & {
  identity: string
  rowsByMetric: Map<string, PendingMetricRow>
}

const PERCENT_DELTA_SEMANTICS: MetricSemantics = {
  semantic_id: 'diagnostic.unspecified',
  metric_name: 'Change',
  kind: 'diagnostic',
  direction: 'none',
  display_kind: 'number',
  display_unit: '%',
  display_precision: 2,
}

const CONFIG_LABELS: Record<string, string> = {
  api: 'API',
  dataset: 'Dataset',
  parallel: 'Concurrency',
  number: 'Number of requests',
  rate: 'Request rate',
  max_tokens: 'Max tokens',
  min_tokens: 'Min tokens',
  max_prompt_length: 'Max prompt length',
  min_prompt_length: 'Min prompt length',
}

export function classifySampleSize(n: number): SampleTier {
  if (!Number.isFinite(n) || n < 30) return 'critical'
  if (n < 100) return 'warn'
  return 'ok'
}

function metricColumns(run: PerfDetailResponse): MetricColumn[] {
  return run.summary_columns.filter((column): column is MetricColumn => column.semantics !== null)
}

function workloadConfig(run: PerfDetailResponse, row: SummaryRow): WorkloadConfigEntry[] {
  const metricKeys = new Set(metricColumns(run).map((column) => column.key))
  return run.summary_columns.flatMap((column) => {
    if (metricKeys.has(column.key)) return []
    return [{ key: column.key, label: column.label, value: String(row.values[column.key] ?? '').trim() }]
  })
}

function configIdentity(config: WorkloadConfigEntry[]): string {
  return JSON.stringify(config.map(({ key, value }) => [key, value]).sort(([a], [b]) => a.localeCompare(b)))
}

function timestampOf(run: PerfDetailResponse): number {
  const parsed = Date.parse(run.generated_at ?? '')
  return Number.isNaN(parsed) ? Number.NEGATIVE_INFINITY : parsed
}

function pickBaseline(runs: PerfDetailResponse[], baselineId: string): PerfDetailResponse {
  return runs.find((run) => run.path === baselineId)
    ?? runs.reduce((oldest, run) => (timestampOf(run) < timestampOf(oldest) ? run : oldest))
}

function sampleCount(metricKey: string, row: SummaryRow): number {
  const count = row.sample_counts[metricKey]
  return typeof count === 'number' && Number.isFinite(count) ? count : 0
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function buildConfigDiff(runs: PerfDetailResponse[]): ConfigDiffEntry[] {
  const configs: Record<string, string>[] = runs.map((run) => ({
    ...run.configuration,
    number: run.configuration?.number ?? String(run.total_requests),
  }))
  const keys = [...new Set(configs.flatMap((config) => Object.keys(config)))]
  return keys.flatMap((key) => {
    const values = configs.map((config) => Object.prototype.hasOwnProperty.call(config, key) ? config[key] : null)
    const comparable = values.map((value) => value === null ? '<missing>' : value)
    if (new Set(comparable).size <= 1) return []
    return [{
      key,
      label: CONFIG_LABELS[key] ?? key,
      values: Object.fromEntries(runs.map((run, index) => [run.path, values[index]])),
    }]
  })
}

function hasWorkloadMismatch(runs: PerfDetailResponse[]): boolean {
  const datasets = new Set(runs.map((run) => run.dataset.trim().toLowerCase()))
  if (datasets.size > 1) return true
  const identities = runs.flatMap((run) => {
    if (run.summary_rows.length === 0) return []
    const configs = run.summary_rows.map((row) => configIdentity(workloadConfig(run, row))).sort()
    return [JSON.stringify(configs)]
  })
  return new Set(identities).size > 1
}

export function buildMetricComparison(
  value: number | null,
  baselineValue: number | null,
  semantics: MetricSemantics | null | undefined,
): MetricComparison {
  const computable = value !== null && baselineValue !== null
  const absoluteValue = computable ? value - baselineValue : null
  const percentValue = computable && baselineValue !== 0
    ? (absoluteValue! / Math.abs(baselineValue)) * 100
    : null
  const rawVerdict = computable ? getComparisonVerdict(absoluteValue!, semantics) : 'incomparable'
  const verdict: DeltaVerdict = !computable || rawVerdict === 'incomparable'
    ? 'incomputable'
    : rawVerdict === 'equal'
      ? 'neutral'
      : rawVerdict === 'better' ? 'improvement' : 'regression'
  return {
    absoluteValue,
    percentValue,
    absoluteDelta: formatDifference(absoluteValue, semantics),
    percentDelta: formatMetric(percentValue, PERCENT_DELTA_SEMANTICS),
    verdict,
  }
}

export function buildCompareModel(runs: PerfDetailResponse[], baselineId: string): PerfCompareModel {
  if (runs.length === 0) {
    return { baselineId: '', groups: [], workloadMismatch: false, duplicateWorkloads: false, configDiff: [] }
  }

  const groups: PendingWorkloadGroup[] = []
  const groupsByIdentity = new Map<string, PendingWorkloadGroup>()
  let duplicateWorkloads = false

  for (const run of runs) {
    const seenConfigs = new Set<string>()
    const duplicateConfigs = new Set<string>()
    for (const summaryRow of run.summary_rows) {
      const identity = configIdentity(workloadConfig(run, summaryRow))
      if (seenConfigs.has(identity)) duplicateConfigs.add(identity)
      seenConfigs.add(identity)
    }
    duplicateWorkloads ||= duplicateConfigs.size > 0

    for (const summaryRow of run.summary_rows) {
      const config = workloadConfig(run, summaryRow)
      const identity = configIdentity(config)
      let group = groupsByIdentity.get(identity)
      if (!group) {
        group = { key: `workload-${groups.length}`, index: groups.length, identity, config, rowsByMetric: new Map() }
        groupsByIdentity.set(identity, group)
        groups.push(group)
      }
      const ambiguous = duplicateConfigs.has(identity)

      for (const metric of metricColumns(run)) {
        let row = group.rowsByMetric.get(metric.key)
        if (!row) {
          row = {
            key: `${group.index}:${metric.key}`,
            configIndex: group.index,
            metricKey: metric.key,
            metricLabel: metric.label,
            semantics: metric.semantics,
            values: {},
          }
          group.rowsByMetric.set(metric.key, row)
        }
        row.values[run.path] = {
          value: ambiguous ? null : finiteNumber(summaryRow.values[metric.key]),
          sampleCount: ambiguous ? 0 : sampleCount(metric.key, summaryRow),
          ambiguous,
        }
      }
    }
  }

  const builtGroups = groups.map((group) => ({
    key: group.key,
    index: group.index,
    config: group.config,
    rows: [...group.rowsByMetric.values()].map((row) => ({
      key: row.key,
      configIndex: row.configIndex,
      metricKey: row.metricKey,
      metricLabel: row.metricLabel,
      semantics: row.semantics,
      values: Object.fromEntries(runs.map((run) => {
        const cell = row.values[run.path] ?? { value: null, sampleCount: 0, ambiguous: false }
        return [run.path, { ...cell, formatted: formatMetric(cell.value, row.semantics) }]
      })),
    })),
  }))

  return {
    baselineId: pickBaseline(runs, baselineId).path,
    groups: builtGroups,
    workloadMismatch: hasWorkloadMismatch(runs),
    duplicateWorkloads,
    configDiff: buildConfigDiff(runs),
  }
}

function parsePositiveInteger(value: string | undefined): number | null {
  if (value === undefined || value.trim() === '') return null
  const parsed = Number(value)
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null
}

export function getFixedInputLength(run: PerfDetailResponse): number | null {
  if (run.configuration?.dataset?.trim().toLowerCase() !== 'random') return null
  const min = parsePositiveInteger(run.configuration?.min_prompt_length)
  const max = parsePositiveInteger(run.configuration?.max_prompt_length)
  return min !== null && min === max ? min : null
}

export function getInputLengthEligibility(runs: PerfDetailResponse[]): InputLengthEligibility {
  const points: InputLengthPoint[] = []
  const pointKeys = new Set<string>()
  for (const run of runs) {
    if (!run.configuration) return { eligible: false, issue: { code: 'missingConfiguration', runPath: run.path }, points }
    if (run.configuration.dataset?.trim().toLowerCase() !== 'random') {
      return { eligible: false, issue: { code: 'notRandom', runPath: run.path }, points }
    }
    const inputLength = getFixedInputLength(run)
    if (inputLength === null) {
      return { eligible: false, issue: { code: 'variableLength', runPath: run.path }, points }
    }
    if (run.summary_rows.length === 0) {
      return { eligible: false, issue: { code: 'missingWorkload', runPath: run.path }, points }
    }

    for (const row of run.summary_rows) {
      const config = workloadConfig(run, row)
      const concurrency = parsePositiveInteger(config.find((entry) => entry.key === 'concurrency')?.value)
      if (concurrency === null) {
        return { eligible: false, issue: { code: 'openLoop', runPath: run.path }, points }
      }
      const key = `${inputLength}:${concurrency}`
      if (pointKeys.has(key)) return { eligible: false, issue: { code: 'duplicatePoint', runPath: run.path }, points }
      pointKeys.add(key)
      points.push({ runPath: run.path, inputLength, concurrency, requests: sampleCount('success_rate', row) })
    }
  }
  return { eligible: runs.length > 0, issue: null, points }
}
