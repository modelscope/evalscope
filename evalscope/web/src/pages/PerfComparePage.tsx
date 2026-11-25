import { useMemo, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useScan } from '@/contexts/ReportsContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { useQueryParams } from '@/hooks/useQueryParams'
import { getPerfCompareChartUrl, getPerfDetail } from '@/api/perf'
import type { PerfDetailResponse } from '@/api/types'
import {
  buildCompareModel,
  buildMetricComparison,
  classifySampleSize,
  getFixedInputLength,
  getInputLengthEligibility,
} from '@/domain/perf/deltaModel'
import type {
  DeltaVerdict,
  InputLengthIssueCode,
  SampleTier,
  WorkloadGroup,
} from '@/domain/perf/deltaModel'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Badge from '@/components/ui/Badge'
import Callout from '@/components/ui/Callout'
import Card from '@/components/ui/Card'
import Select from '@/components/ui/Select'
import SegmentedControl from '@/components/ui/SegmentedControl'
import Skeleton from '@/components/ui/Skeleton'
import PerfChartGroup from '@/components/perf/PerfChartGroup'
import ErrorAlert from '@/components/ui/ErrorAlert'
import { LATENCY_CHARTS, THROUGHPUT_CHARTS, perfRunMode } from '@/domain/perf/charts'
import { formatTimestamp } from '@/utils/formatUtils'
import { AlertTriangle, GitCompareArrows, Info } from 'lucide-react'

type ComparisonMode = 'absolute' | 'baseline'
type ChartAxis = 'workload' | 'input_length'

function tierRank(tier: SampleTier): number {
  return tier === 'critical' ? 2 : tier === 'warn' ? 1 : 0
}

function worstSampleTier(counts: number[]): SampleTier {
  return counts.map(classifySampleSize).reduce<SampleTier>(
    (worst, tier) => tierRank(tier) > tierRank(worst) ? tier : worst,
    'ok',
  )
}

function percentileLevel(metricKey: string): 90 | 95 | 99 | null {
  const match = metricKey.match(/p\s*(90|95|99)/i)
  return match ? Number(match[1]) as 90 | 95 | 99 : null
}

function percentileDeEmphasized(tier: SampleTier, level: 90 | 95 | 99): boolean {
  if (tier === 'critical') return true
  if (tier === 'warn') return level >= 95
  return false
}

function fullRunLabel(run: PerfDetailResponse): string {
  const parts = [
    run.alias || run.model,
    run.basic_info.Endpoint || run.basic_info['API Host'],
    run.dataset,
    formatTimestamp(run.generated_at, 'seconds'),
    run.path,
  ].filter(Boolean)
  return parts.join(' · ')
}

function workloadLabel(group: WorkloadGroup, t: (key: string) => string): string {
  const concurrency = group.config.find((entry) => entry.key === 'concurrency')?.value
  const openLoop = concurrency !== undefined && Number(concurrency) <= 0
  return group.config.flatMap((entry) => {
    if (entry.key === 'concurrency') {
      return openLoop ? [] : [`${t('perf.archive.concurrency')} ${entry.value || '—'}`]
    }
    if (entry.key === 'request_rate') {
      if (openLoop) {
        const rate = Number(entry.value) === -1 ? '' : ` · ${t('perf.archive.requestRate')} ${entry.value || '—'}`
        return [`${t('perf.archive.openLoop')}${rate}`]
      }
      return [Number(entry.value) === -1
        ? t('perf.archive.closedLoop')
        : `${t('perf.archive.requestRate')} ${entry.value || '—'}`]
    }
    return [`${entry.label}=${entry.value || '—'}`]
  }).join(' · ')
}

function verdictClass(verdict: DeltaVerdict): string {
  if (verdict === 'improvement') return 'text-[var(--success)]'
  if (verdict === 'regression') return 'text-[var(--danger)]'
  return 'text-[var(--text-dim)]'
}

function signed(primary: string, value: number | null): string {
  return value !== null && value > 0 ? `+${primary}` : primary
}

function configurationLabel(key: string, fallback: string, t: (key: string) => string): string {
  const labels: Record<string, string> = {
    api: 'perf.task.apiType',
    dataset: 'perf.task.dataset',
    parallel: 'perf.archive.concurrency',
    number: 'perf.archive.numberOfRequests',
    rate: 'perf.archive.requestRate',
    max_tokens: 'perf.task.maxTokens',
    min_tokens: 'perf.task.minTokens',
    max_prompt_length: 'perf.task.maxPromptLen',
    min_prompt_length: 'perf.task.minPromptLen',
  }
  return labels[key] ? t(labels[key]) : fallback
}

function chartMetricKey(run: PerfDetailResponse, chart: string): string {
  if (chart === 'latency') return 'avg_latency'
  if (chart === 'ttft') return 'avg_ttft'
  if (chart === 'tpot') return 'avg_tpot'
  if (chart === 'rps') return 'request_throughput'
  if (chart === 'success') return 'success_rate'
  return perfRunMode(run) === 'generation' ? 'output_token_throughput' : 'input_token_throughput'
}

export default function PerfComparePage() {
  const { t } = useLocale()
  const { get, set } = useQueryParams()
  const { rootPath: ctxRoot } = useScan()
  const [comparisonMode, setComparisonMode] = useState<ComparisonMode>('absolute')

  const rootPath = get('root_path') ?? ctxRoot
  const paths = useMemo(
    () => (get('paths') ?? '').split(';').map((path) => path.trim()).filter(Boolean),
    [get],
  )
  const baselineParam = get('baseline') ?? ''
  const pathsKey = paths.join(';')
  const comparison = useAsyncResource(
    async (signal) => {
      const results = await Promise.allSettled(paths.map((path) => getPerfDetail(rootPath, path, signal)))
      const runs = results
        .filter((result): result is PromiseFulfilledResult<PerfDetailResponse> => result.status === 'fulfilled')
        .map((result) => result.value)
      return { runs, missingCount: paths.length - runs.length }
    },
    [rootPath, pathsKey],
    { enabled: paths.length >= 2, fallbackMessage: t('perf.archive.compareLoadError') },
  )

  const details = comparison.loading ? null : comparison.data?.runs ?? null
  const missingCount = comparison.data?.missingCount ?? 0
  const loadError = comparison.error || (details?.length === 0 ? t('perf.archive.compareLoadError') : '')
  const model = useMemo(() => details ? buildCompareModel(details, baselineParam) : null, [details, baselineParam])
  const inputLength = useMemo(() => getInputLengthEligibility(details ?? []), [details])
  const requestedAxis = get('x_axis')
  const chartAxis: ChartAxis = requestedAxis === 'input_length' && inputLength.eligible ? 'input_length' : 'workload'

  const inputOnly = details?.some((detail) => perfRunMode(detail) !== 'generation') ?? false
  const latencyCharts = useMemo(() => inputOnly ? ['latency'] as const : LATENCY_CHARTS, [inputOnly])
  const runIndexes = useMemo(
    () => new Map((details ?? []).map((run, index) => [run.path, index + 1] as const)),
    [details],
  )
  const runCode = (path: string) => `R${runIndexes.get(path) ?? '?'}`
  const runSecondary = (run: PerfDetailResponse) => {
    const length = getFixedInputLength(run)
    if (length !== null) return t('perf.archive.inputLength', { n: length.toLocaleString('en-US') })
    return [run.alias || run.model, formatTimestamp(run.generated_at)].filter(Boolean).join(' · ')
  }

  if (paths.length < 2) {
    return (
      <div className="page-enter flex flex-col gap-4">
        <Breadcrumb
          items={[
            { label: t('nav.performance'), href: `/performance?root_path=${encodeURIComponent(rootPath)}` },
            { label: t('perf.archive.comparePageTitle') },
          ]}
        />
        <div className="py-16 text-center type-body-sm text-[var(--text-muted)]">
          {t('perf.archive.selectToCompare')}
        </div>
      </div>
    )
  }

  const percentileCounts = (model?.groups ?? []).flatMap((group) => group.rows.flatMap((row) => {
    if (percentileLevel(row.metricKey) === null) return []
    return Object.values(row.values).filter((cell) => cell.value !== null).map((cell) => cell.sampleCount)
  }))
  const sampleTier = worstSampleTier(percentileCounts)
  const hasEmptyRun = (details ?? []).some((run) => run.summary_rows.length === 0)
  const hasMissingMetric = (model?.groups ?? []).some((group) => group.rows.some((row) =>
    Object.values(row.values).some((cell) => cell.formatted.isMissing || cell.ambiguous),
  ))
  const showMissingHint = missingCount > 0 || hasEmptyRun || hasMissingMetric
  const pointCount = chartAxis === 'input_length'
    ? inputLength.points.length
    : (details ?? []).reduce((total, run) => total + run.summary_rows.length, 0)

  const issueKey: Record<InputLengthIssueCode, string> = {
    missingConfiguration: 'perf.archive.inputLengthIssueMissingConfiguration',
    notRandom: 'perf.archive.inputLengthIssueNotRandom',
    variableLength: 'perf.archive.inputLengthIssueVariableLength',
    openLoop: 'perf.archive.inputLengthIssueOpenLoop',
    missingWorkload: 'perf.archive.inputLengthIssueMissingWorkload',
    duplicatePoint: 'perf.archive.inputLengthIssueDuplicatePoint',
  }
  const inputLengthIssue = inputLength.issue
    ? `${inputLength.issue.runPath ? `${runCode(inputLength.issue.runPath)}: ` : ''}${t(issueKey[inputLength.issue.code])}`
    : ''

  const chartFallback = (chart: string) => {
    const runLabel = t('perf.archive.runCol')
    const requestsLabel = t('perf.archive.requestsCol')
    const valueLabel = t('perf.archive.valueCol')
    if (chartAxis === 'input_length') {
      const concurrencyLabel = t('perf.archive.concurrency')
      const lengthLabel = t('perf.archive.configuredInputLength')
      const rows = [...inputLength.points]
        .sort((a, b) => a.concurrency - b.concurrency || a.inputLength - b.inputLength)
        .flatMap((point) => {
          const run = details?.find((item) => item.path === point.runPath)
          if (!run) return []
          const metricKey = chartMetricKey(run, chart)
          for (const group of model?.groups ?? []) {
            const concurrency = group.config.find((entry) => entry.key === 'concurrency')?.value
            if (Number(concurrency) !== point.concurrency) continue
            const row = group.rows.find((item) => item.metricKey === metricKey)
            const cell = row?.values[run.path]
            if (!cell || cell.value === null || cell.ambiguous) continue
            return [{
              [concurrencyLabel]: point.concurrency,
              [lengthLabel]: point.inputLength,
              [runLabel]: runCode(run.path),
              [requestsLabel]: cell.sampleCount,
              [valueLabel]: cell.formatted.primary,
            }]
          }
          return []
        })
      return { columns: [concurrencyLabel, lengthLabel, runLabel, requestsLabel, valueLabel], rows }
    }

    const workload = t('perf.archive.workloadCol')
    const rows = (details ?? []).flatMap((run) => {
      const metricKey = chartMetricKey(run, chart)
      return (model?.groups ?? []).flatMap((group) => {
        const row = group.rows.find((item) => item.metricKey === metricKey)
        const cell = row?.values[run.path]
        if (!cell || cell.value === null || cell.ambiguous) return []
        return [{
          [runLabel]: runCode(run.path),
          [workload]: workloadLabel(group, t),
          [requestsLabel]: cell.sampleCount,
          [valueLabel]: cell.formatted.primary,
        }]
      })
    })
    return { columns: [runLabel, workload, requestsLabel, valueLabel], rows }
  }

  return (
    <div className="page-enter flex flex-col gap-4">
      <Breadcrumb
        items={[
          { label: t('nav.performance'), href: `/performance?root_path=${encodeURIComponent(rootPath)}` },
          { label: t('perf.archive.comparePageTitle') },
        ]}
      />

      <div className="flex items-start gap-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-5">
        <span className="mt-0.5 shrink-0 text-[var(--accent)]"><GitCompareArrows size={20} /></span>
        <h1 className="type-title-md min-w-0 text-[var(--text)]">{t('perf.archive.comparing', { n: paths.length })}</h1>
      </div>

      {details === null ? (
        <Skeleton width="100%" height={220} />
      ) : loadError ? (
        <ErrorAlert className="p-6 type-body-sm">{loadError}</ErrorAlert>
      ) : model && (
        <>
          {model.workloadMismatch && (
            <Callout
              variant="warning"
              icon={<AlertTriangle size={15} className="text-[var(--yellow)]" />}
              className="rounded-[var(--radius-sm)]"
              data-testid="workload-mismatch"
            >
              {t('perf.archive.workloadMismatch')}
            </Callout>
          )}
          {model.duplicateWorkloads && (
            <Callout
              variant="warning"
              icon={<AlertTriangle size={15} className="text-[var(--yellow)]" />}
              className="rounded-[var(--radius-sm)]"
              data-testid="duplicate-workloads"
            >
              {t('perf.archive.duplicateWorkloads')}
            </Callout>
          )}
          {sampleTier !== 'ok' && (
            <Callout
              variant="warning"
              icon={<AlertTriangle size={15} className={sampleTier === 'critical' ? 'text-[var(--danger)]' : 'text-[var(--yellow)]'} />}
              className="rounded-[var(--radius-sm)]"
              data-testid={sampleTier === 'critical' ? 'low-sample-critical' : 'low-sample-warn'}
            >
              {sampleTier === 'critical' ? t('perf.archive.lowSampleCritical') : t('perf.archive.lowSampleWarn')}
            </Callout>
          )}
          {showMissingHint && (
            <Callout
              variant="warning"
              icon={<Info size={15} className="text-[var(--yellow)]" />}
              className="rounded-[var(--radius-sm)]"
              data-testid="missing-perf-data"
            >
              {t('perf.archive.missingPerfData')}
            </Callout>
          )}

          {model.groups.length > 0 && (
            <Card title={t('perf.archive.performanceSummary')}>
              <div className="mb-4 flex flex-wrap items-end gap-x-6 gap-y-3">
                <div>
                  <div className="mb-1.5 text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">
                    {t('perf.archive.comparisonMode')}
                  </div>
                  <SegmentedControl
                    options={[
                      { value: 'absolute', label: t('perf.archive.absoluteValues') },
                      { value: 'baseline', label: t('perf.archive.vsBaseline') },
                    ]}
                    value={comparisonMode}
                    onChange={setComparisonMode}
                    ariaLabel={t('perf.archive.comparisonMode')}
                  />
                </div>
                <Select
                  disabled={comparisonMode !== 'baseline'}
                  label={t('perf.archive.selectBaseline')}
                  options={details.map((run, index) => ({ value: run.path, label: `R${index + 1} · ${runSecondary(run)}` }))}
                  value={model.baselineId}
                  onChange={(value) => set('baseline', value)}
                  className="min-w-56"
                />
              </div>
              <p className="type-body-xs mb-3 text-[var(--text-muted)]">{t('perf.archive.performanceSummaryHint')}</p>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse" data-testid="performance-summary-table">
                  <thead className="bg-[var(--bg-card2)]">
                    <tr>
                      <th className="sticky left-0 z-10 min-w-52 border-b border-r border-[var(--border)] bg-[var(--bg-card2)] px-3 py-2 text-left type-table-xs">
                        {t('perf.archive.metricCol')}
                      </th>
                      {details.map((run, index) => (
                        <th
                          key={run.path}
                          title={fullRunLabel(run)}
                          className="min-w-40 border-b border-[var(--border)] px-3 py-2 text-center type-table-xs"
                          data-testid={`run-heading-${index + 1}`}
                        >
                          <span className="flex items-center justify-center gap-1.5">
                            <span>R{index + 1}</span>
                            {comparisonMode === 'baseline' && run.path === model.baselineId && (
                              <Badge>{t('perf.archive.baselineBadge')}</Badge>
                            )}
                          </span>
                          <span className="mt-0.5 block max-w-48 truncate text-[10px] font-normal normal-case text-[var(--text-muted)]">
                            {runSecondary(run)}
                          </span>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {model.groups.map((group) => (
                      <FragmentGroup
                        key={group.key}
                        group={group}
                        details={details}
                        modelBaseline={model.baselineId}
                        comparisonMode={comparisonMode}
                        t={t}
                      />
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="type-body-xs mt-3 text-[var(--text-muted)]">{t('perf.archive.deltaInfoNote')}</p>
            </Card>
          )}

          <Card title={t('perf.archive.configDiffTitle')}>
            {model.configDiff.length === 0 ? (
              <div className="type-body-sm text-[var(--text-muted)]">{t('perf.archive.noConfigDiff')}</div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full border-collapse" data-testid="config-diff-table">
                  <thead>
                    <tr>
                      <th className="border-b border-[var(--border)] px-3 py-2 text-left type-table-xs">
                        {t('perf.archive.configKeyCol')}
                      </th>
                      {details.map((run, index) => (
                        <th key={run.path} className="border-b border-[var(--border)] px-3 py-2 text-right type-table-xs">
                          R{index + 1}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {model.configDiff.map((entry) => (
                      <tr key={entry.key} className="border-b border-[var(--border)] last:border-b-0">
                        <td className="whitespace-nowrap px-3 py-2 text-left type-body-sm text-[var(--text)]">
                          {configurationLabel(entry.key, entry.label, t)}
                        </td>
                        {details.map((run) => (
                          <td key={run.path} className="whitespace-nowrap px-3 py-2 text-right type-body-sm tabular-nums">
                            {entry.values[run.path] ?? '—'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </>
      )}

      {details !== null && !loadError && (
        <div className="flex flex-wrap items-end gap-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-4">
          <div>
            <div className="mb-1.5 text-xs font-medium uppercase tracking-wider text-[var(--text-muted)]">
              {t('perf.archive.chartAxis')}
            </div>
            <SegmentedControl
              options={[
                { value: 'workload', label: t('perf.archive.axisWorkload') },
                {
                  value: 'input_length',
                  label: t('perf.archive.axisInputLength'),
                  disabled: !inputLength.eligible,
                  title: inputLengthIssue || undefined,
                },
              ]}
              value={chartAxis}
              onChange={(value) => set('x_axis', value === 'workload' ? undefined : value)}
              ariaLabel={t('perf.archive.chartAxis')}
            />
          </div>
          {inputLengthIssue && (
            <p className="max-w-2xl pb-1 text-xs text-[var(--text-muted)]" data-testid="input-length-unavailable">
              {t('perf.archive.inputLengthUnavailable', { reason: inputLengthIssue })}
            </p>
          )}
        </div>
      )}

      {details !== null && pointCount <= 2 && !loadError && (
        <Callout
          variant="info"
          icon={<Info size={15} className="text-[var(--accent)]" />}
          className="rounded-[var(--radius-sm)]"
          data-testid="sparse-hint"
        >
          {t('perf.archive.sparseCompareHint')}
        </Callout>
      )}

      <PerfChartGroup
        title={t('perf.archive.latencyGroup')}
        charts={latencyCharts}
        fallbackTable={chartFallback}
        getChartUrl={(chart) => getPerfCompareChartUrl(rootPath, paths, chart, undefined, chartAxis === 'workload' ? undefined : chartAxis)}
        loading={details === null}
      />
      <PerfChartGroup
        title={t('perf.archive.throughputGroup')}
        charts={THROUGHPUT_CHARTS}
        fallbackTable={chartFallback}
        getChartUrl={(chart) => getPerfCompareChartUrl(rootPath, paths, chart, undefined, chartAxis === 'workload' ? undefined : chartAxis)}
        loading={details === null}
      />
    </div>
  )
}

function FragmentGroup({
  group,
  details,
  modelBaseline,
  comparisonMode,
  t,
}: {
  group: WorkloadGroup
  details: PerfDetailResponse[]
  modelBaseline: string
  comparisonMode: ComparisonMode
  t: (key: string) => string
}) {
  return (
    <>
      <tr className="bg-[var(--bg-deep)]" data-testid={`workload-group-${group.index}`}>
        <th colSpan={details.length + 1} className="border-b border-[var(--border)] px-3 py-2 text-left type-label-xs">
          {workloadLabel(group, t)}
        </th>
      </tr>
      {group.rows.map((row) => {
        const level = percentileLevel(row.metricKey)
        const baselineCell = row.values[modelBaseline]
        return (
          <tr
            key={row.key}
            className="border-b border-[var(--border)] last:border-b-0"
            data-testid={`summary-row-${group.index}-${row.metricKey}`}
          >
            <td className="sticky left-0 z-10 border-r border-[var(--border)] bg-[var(--bg-card)] px-3 py-2 text-left type-body-sm font-medium">
              {row.metricLabel}
            </td>
            {details.map((run, runIndex) => {
              const cell = row.values[run.path]
              const isBaseline = run.path === modelBaseline
              const comparison = buildMetricComparison(cell.value, baselineCell?.value ?? null, row.semantics)
              const counts = comparisonMode === 'baseline' && !isBaseline
                ? [cell.sampleCount, baselineCell?.sampleCount ?? 0]
                : [cell.sampleCount]
              const deEmphasized = cell.value === null || (
                level !== null && percentileDeEmphasized(worstSampleTier(counts), level)
              )
              return (
                <td
                  key={run.path}
                  className={`px-3 py-2 text-right type-body-sm tabular-nums ${deEmphasized ? 'opacity-45' : ''}`}
                  title={cell.ambiguous ? t('perf.archive.duplicateWorkloads') : cell.formatted.raw}
                  data-testid={`summary-cell-${group.index}-${row.metricKey}-${runIndex + 1}`}
                  data-deemphasized={deEmphasized ? 'true' : 'false'}
                >
                  <div className={cell.formatted.isMissing ? 'text-[var(--text-dim)]' : 'text-[var(--text)]'}>
                    {cell.formatted.primary}
                  </div>
                  {comparisonMode === 'baseline' && (
                    <div className={`mt-0.5 min-h-4 text-[10px] ${isBaseline ? 'text-[var(--text-dim)]' : verdictClass(comparison.verdict)}`}>
                      {isBaseline
                        ? t('perf.archive.baselineBadge')
                        : `${signed(comparison.absoluteDelta.primary, comparison.absoluteValue)} · ${signed(comparison.percentDelta.primary, comparison.percentValue)}`}
                    </div>
                  )}
                </td>
              )
            })}
          </tr>
        )
      })}
    </>
  )
}
