import { useEffect, useMemo, useState } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useReports } from '@/contexts/ReportsContext'
import { useQueryParams } from '@/hooks/useQueryParams'
import { getPerfCompareChartUrl, getPerfDetail } from '@/api/perf'
import type { PerfDetailResponse } from '@/api/types'
import { buildCompareModel, classifySampleSize } from '@/domain/perf/compareModel'
import type { DeltaVerdict, PerfCompareModel, SampleTier } from '@/domain/perf/compareModel'
import { getMetricSpec } from '@/domain/metric/registry'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Badge from '@/components/ui/Badge'
import Card from '@/components/ui/Card'
import Skeleton from '@/components/ui/Skeleton'
import PlotlyChart from '@/components/charts/PlotlyChart'
import { LATENCY_CHARTS, THROUGHPUT_CHARTS, CHART_TITLES, formatFull } from '@/utils/perf'
import { AlertTriangle, ArrowLeftRight, GitCompareArrows, Info } from 'lucide-react'

type CompareVisualization = 'sparse' | 'trend'

function selectCompareVisualization(n: number): CompareVisualization {
  return Number.isFinite(n) && n > 2 ? 'trend' : 'sparse'
}

// ------------------------------------------------------------------ //
// Low-sample de-emphasis helpers                        //
// ------------------------------------------------------------------ //

/** Rank a sample tier so the worst (lowest-sample) tier can be selected. */
function tierRank(tier: SampleTier): number {
  return tier === 'critical' ? 2 : tier === 'warn' ? 1 : 0
}

/** Worst (lowest-sample) tier across the baseline and candidate sample counts. */
function worstSampleTier(counts: Record<string, number>): SampleTier {
  const tiers = Object.values(counts).map(classifySampleSize)
  return tiers.reduce<SampleTier>((worst, tier) => (tierRank(tier) > tierRank(worst) ? tier : worst), 'ok')
}

/** Percentile level referenced by a metric label, or `null` when it is not a P90/P95/P99 metric. */
function percentileLevel(metricKey: string): 90 | 95 | 99 | null {
  const match = metricKey.match(/p\s*(90|95|99)/i)
  return match ? (Number(match[1]) as 90 | 95 | 99) : null
}

/**
 * Whether a percentile metric should be de-emphasized at the given sample tier.
 * `critical` de-emphasizes P90/P95/P99, `warn` de-emphasizes
 * P95/P99, `ok` de-emphasizes nothing.
 */
function percentileDeEmphasized(tier: SampleTier, level: 90 | 95 | 99): boolean {
  if (tier === 'critical') return true
  if (tier === 'warn') return level >= 95
  return false
}

// ------------------------------------------------------------------ //
// Presentational helpers                                              //
// ------------------------------------------------------------------ //

const VERDICT_LABEL_KEY: Record<DeltaVerdict, string> = {
  improvement: 'performance.verdictImprovement',
  regression: 'performance.verdictRegression',
  neutral: 'performance.verdictNeutral',
  incomputable: 'performance.verdictIncomputable',
}

const VERDICT_VARIANT: Record<DeltaVerdict, 'success' | 'danger' | 'default' | 'warning'> = {
  improvement: 'success',
  regression: 'danger',
  neutral: 'default',
  incomputable: 'warning',
}

/** Short display label for a run: model, dataset and a compact timestamp. */
function runLabel(run: PerfDetailResponse | undefined): string {
  if (!run) return ''
  const parts = [run.model, run.dataset].filter(Boolean)
  const ts = formatFull(run.generated_at)
  return ts ? `${parts.join(' · ')} · ${ts}` : parts.join(' · ')
}

export default function PerfComparePage() {
  const { t } = useLocale()
  const { get, set } = useQueryParams()
  const { rootPath: ctxRoot } = useReports()

  const rootPath = get('root_path') ?? ctxRoot
  const paths = useMemo(
    () => (get('paths') ?? '').split(';').map((p) => p.trim()).filter(Boolean),
    [get],
  )

  // Persisted baseline selection: the effective baseline id lives in
  // the `baseline` query param so a swap survives subsequent loads of this view.
  const baselineParam = get('baseline') ?? ''

  const [details, setDetails] = useState<PerfDetailResponse[] | null>(null)
  const [missingCount, setMissingCount] = useState(0)
  const [loadError, setLoadError] = useState('')

  const pathsKey = paths.join(';')
  useEffect(() => {
    if (paths.length < 2) return
    const controller = new AbortController()
    const load = async () => {
      setDetails(null)
      setLoadError('')
      setMissingCount(0)
      const results = await Promise.allSettled(paths.map((p) => getPerfDetail(rootPath, p, controller.signal)))
      if (controller.signal.aborted) return
      const ok = results.filter((r): r is PromiseFulfilledResult<PerfDetailResponse> => r.status === 'fulfilled')
      const runs = ok.map((r) => r.value)
      setMissingCount(paths.length - runs.length)
      setDetails(runs)
      if (runs.length === 0) setLoadError(t('performance.compareLoadError'))
    }
    load()
    return () => {
      controller.abort()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rootPath, pathsKey])

  const model: PerfCompareModel | null = useMemo(
    () => (details ? buildCompareModel(details, baselineParam) : null),
    [details, baselineParam],
  )

  // Embedding/rerank runs omit TTFT/TPOT charts (mirrors PerfReportDetailPage).
  const isEmbedding = details?.[0]?.is_embedding ?? false
  const latencyCharts = useMemo(
    () => (isEmbedding ? (['latency'] as const) : LATENCY_CHARTS),
    [isEmbedding],
  )

  if (paths.length < 2) {
    return (
      <div className="page-enter flex flex-col gap-4">
        <Breadcrumb
          items={[
            { label: t('nav.performance'), href: `/performance?root_path=${encodeURIComponent(rootPath)}` },
            { label: t('performance.comparePageTitle') },
          ]}
        />
        <div className="py-16 text-center type-body-sm text-[var(--text-muted)]">
          {t('performance.selectToCompare')}
        </div>
      </div>
    )
  }

  const byPath = new Map((details ?? []).map((d) => [d.path, d]))
  const baselineRun = model ? byPath.get(model.baselineId) : undefined
  const candidateRun = model ? byPath.get(model.candidateId) : undefined
  const canSwap = Boolean(model && model.candidateId && model.candidateId !== model.baselineId)

  const sampleTier: SampleTier = model ? worstSampleTier(model.sampleCounts) : 'ok'
  // A run missing performance data has no summary rows.
  const hasEmptyRun = (details ?? []).some((d) => !Array.isArray(d.summary_rows) || d.summary_rows.length === 0)
  const showMissingHint = missingCount > 0 || hasEmptyRun || Boolean(model?.deltas.some((d) => d.verdict === 'incomputable'))
  const vizMode = selectCompareVisualization((details ?? []).length)

  return (
    <div className="page-enter flex flex-col gap-4">
      <Breadcrumb
        items={[
          { label: t('nav.performance'), href: `/performance?root_path=${encodeURIComponent(rootPath)}` },
          { label: t('performance.comparePageTitle') },
        ]}
      />

      {/* Header */}
      <div className="flex items-start gap-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-5">
        <span className="text-[var(--accent)] shrink-0 mt-0.5">
          <GitCompareArrows size={20} />
        </span>
        <div className="flex flex-col gap-1 min-w-0">
          <h1 className="type-title-md text-[var(--text)]">
            {t('performance.comparing', { n: paths.length })}
          </h1>
          <div
            className="type-caption-mono text-[var(--text-muted)] break-words"
            title={paths.join('\n')}
            data-testid="compare-run-labels"
          >
            {details ? details.map((run) => runLabel(run)).join('  ·  ') : t('common.loading')}
          </div>
        </div>
      </div>

      {details === null ? (
        <Skeleton width="100%" height={220} />
      ) : loadError ? (
        <div className="p-6 rounded-[var(--radius)] border border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger)] type-body-sm">
          {loadError}
        </div>
      ) : (
        model && (
          <>
            {/* Baseline / candidate selector with effective-baseline marker */}
            <div
              className="flex flex-wrap items-stretch gap-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-4"
              data-testid="baseline-selector"
            >
              <div className="flex flex-col gap-1 min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <Badge>{t('performance.baselineBadge')}</Badge>
                  <span className="type-table-xs uppercase tracking-wider text-[var(--text-muted)]">
                    {t('performance.effectiveBaseline')}
                  </span>
                </div>
                <div className="type-body-sm text-[var(--text)] break-all" data-testid="baseline-label">
                  {runLabel(baselineRun)}
                </div>
                <div className="type-caption text-[var(--text-muted)] tabular-nums">
                  {t('performance.sampleCount', { n: model.sampleCounts[model.baselineId] ?? 0 })}
                </div>
              </div>

              <button
                type="button"
                onClick={() => canSwap && set('baseline', model.candidateId)}
                disabled={!canSwap}
                className="flex items-center gap-1.5 self-center px-3 py-1.5 rounded-[var(--radius-sm)] border border-[var(--border-md)] type-body-sm text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--bg-card2)] transition-colors disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
                data-testid="swap-baseline"
              >
                <ArrowLeftRight size={14} />
                {t('performance.swapBaseline')}
              </button>

              <div className="flex flex-col gap-1 min-w-0 flex-1 md:text-right">
                <div className="flex items-center gap-2 md:justify-end">
                  <Badge variant="success">{t('performance.candidateBadge')}</Badge>
                </div>
                <div className="type-body-sm text-[var(--text)] break-all" data-testid="candidate-label">
                  {runLabel(candidateRun)}
                </div>
                <div className="type-caption text-[var(--text-muted)] tabular-nums">
                  {t('performance.sampleCount', { n: model.sampleCounts[model.candidateId] ?? 0 })}
                </div>
              </div>
            </div>

            {/* Warnings — informational, never blocking */}
            {model.workloadMismatch && (
              <div
                className="flex items-start gap-2 px-4 py-3 rounded-[var(--radius-sm)] border border-[var(--warning-border)] bg-[var(--warning-bg)] type-body-sm text-[var(--text)]"
                data-testid="workload-mismatch"
              >
                <AlertTriangle size={15} className="text-[var(--yellow)] shrink-0 mt-0.5" />
                <span>{t('performance.workloadMismatch')}</span>
              </div>
            )}

            {sampleTier !== 'ok' && (
              <div
                className={
                  sampleTier === 'critical'
                    ? 'flex items-start gap-2 px-4 py-3 rounded-[var(--radius-sm)] border border-[var(--danger-border)] bg-[var(--danger-bg)] type-body-sm text-[var(--text)]'
                    : 'flex items-start gap-2 px-4 py-3 rounded-[var(--radius-sm)] border border-[var(--warning-border)] bg-[var(--warning-bg)] type-body-sm text-[var(--text)]'
                }
                data-testid={sampleTier === 'critical' ? 'low-sample-critical' : 'low-sample-warn'}
              >
                <AlertTriangle
                  size={15}
                  className={sampleTier === 'critical' ? 'text-[var(--danger)] shrink-0 mt-0.5' : 'text-[var(--yellow)] shrink-0 mt-0.5'}
                />
                <span>
                  {sampleTier === 'critical' ? t('performance.lowSampleCritical') : t('performance.lowSampleWarn')}
                </span>
              </div>
            )}

            {showMissingHint && (
              <div
                className="flex items-start gap-2 px-4 py-3 rounded-[var(--radius-sm)] border border-[var(--warning-border)] bg-[var(--warning-bg)] type-body-sm text-[var(--text)]"
                data-testid="missing-perf-data"
              >
                <Info size={15} className="text-[var(--yellow)] shrink-0 mt-0.5" />
                <span>{t('performance.missingPerfData')}</span>
              </div>
            )}

            {/* Delta summary table */}
            <Card title={t('performance.deltaSummary')}>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse" data-testid="delta-table">
                  <thead>
                    <tr>
                      {[
                        t('performance.metricCol'),
                        t('performance.baselineCol'),
                        t('performance.candidateCol'),
                        t('performance.absDeltaCol'),
                        t('performance.pctDeltaCol'),
                        t('performance.directionCol'),
                      ].map((label, i) => (
                        <th
                          key={label}
                          className={`type-table-xs uppercase tracking-wider px-3 py-2 whitespace-nowrap border-b border-[var(--border)] text-[var(--text-muted)] ${i === 0 ? 'text-left' : 'text-right'}`}
                        >
                          {label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {model.deltas.map((delta) => {
                      const resolvedMetric = getMetricSpec(delta.metricKey)
                      const metricLabel = resolvedMetric.isFallback
                        ? delta.metricKey
                        : t(resolvedMetric.spec.labelKey)
                      const level = percentileLevel(delta.metricKey)
                      const lowSample = level !== null && percentileDeEmphasized(sampleTier, level)
                      const incomputable = delta.verdict === 'incomputable'
                      // De-emphasize incomputable deltas and low-sample percentiles,
                      // but keep raw values available via the cell tooltip.
                      const deEmphasized = lowSample || incomputable
                      return (
                        <tr
                          key={delta.metricKey}
                          className={`border-b border-[var(--border)] last:border-b-0 ${deEmphasized ? 'opacity-50' : ''}`}
                          data-testid={`delta-row-${delta.metricKey}`}
                          data-deemphasized={deEmphasized ? 'true' : 'false'}
                        >
                          <td className="type-body-sm px-3 py-2 text-left text-[var(--text)]">
                            <span className="block font-medium">{metricLabel}</span>
                            {metricLabel !== delta.metricKey && (
                              <span className="block type-caption-mono text-[var(--text-muted)]">{delta.metricKey}</span>
                            )}
                            {resolvedMetric.isFallback && (
                              <span className="block type-caption text-[var(--warning)]">
                                {t('metrics.undefined_display')}
                              </span>
                            )}
                          </td>
                          <td
                            className="type-body-sm tabular-nums px-3 py-2 text-right whitespace-nowrap text-[var(--text)]"
                            title={delta.baseline.raw}
                          >
                            {delta.baseline.primary}
                          </td>
                          <td
                            className="type-body-sm tabular-nums px-3 py-2 text-right whitespace-nowrap text-[var(--text)]"
                            title={delta.candidate.raw}
                          >
                            {delta.candidate.primary}
                          </td>
                          <td
                            className="type-body-sm tabular-nums px-3 py-2 text-right whitespace-nowrap text-[var(--text)]"
                            title={delta.absoluteDelta.raw}
                          >
                            {delta.absoluteDelta.primary}
                          </td>
                          <td
                            className="type-body-sm tabular-nums px-3 py-2 text-right whitespace-nowrap text-[var(--text)]"
                            title={delta.percentDelta.raw}
                          >
                            {delta.percentDelta.primary}
                          </td>
                          <td className="px-3 py-2 text-right whitespace-nowrap">
                            <Badge variant={VERDICT_VARIANT[delta.verdict]}>
                              {t(VERDICT_LABEL_KEY[delta.verdict])}
                            </Badge>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
              <p className="mt-3 type-caption text-[var(--text-muted)]">{t('performance.deltaInfoNote')}</p>
            </Card>

            {/* Configuration differences */}
            <Card title={t('performance.configDiffTitle')}>
              {model.configDiff.length === 0 ? (
                <div className="type-body-sm text-[var(--text-muted)]">{t('performance.noConfigDiff')}</div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse" data-testid="config-diff-table">
                    <thead>
                      <tr>
                        {[
                          t('performance.configKeyCol'),
                          t('performance.baselineCol'),
                          t('performance.candidateCol'),
                        ].map((label, i) => (
                          <th
                            key={label}
                            className={`type-table-xs uppercase tracking-wider px-3 py-2 whitespace-nowrap border-b border-[var(--border)] text-[var(--text-muted)] ${i === 0 ? 'text-left' : 'text-right'}`}
                          >
                            {label}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {model.configDiff.map((entry) => (
                        <tr key={entry.key} className="border-b border-[var(--border)] last:border-b-0">
                          <td className="type-body-sm px-3 py-2 text-left whitespace-nowrap text-[var(--text)]">
                            {entry.key}
                          </td>
                          <td className="type-body-sm tabular-nums px-3 py-2 text-right whitespace-nowrap text-[var(--text)]">
                            {entry.baseline || '—'}
                          </td>
                          <td className="type-body-sm tabular-nums px-3 py-2 text-right whitespace-nowrap text-[var(--text)]">
                            {entry.candidate || '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          </>
        )
      )}

      {/* Sparse-vs-trend hint for the visualization */}
      {details !== null && vizMode === 'sparse' && (
        <div
          className="flex items-start gap-2 px-4 py-3 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-card2)] type-body-sm text-[var(--text-muted)]"
          data-testid="sparse-hint"
        >
          <Info size={15} className="text-[var(--accent)] shrink-0 mt-0.5" />
          <span>{t('performance.sparseCompareHint')}</span>
        </div>
      )}

      {/* Latency group */}
      <Card title={t('performance.latencyGroup')}>
        {details === null ? (
          <Skeleton width="100%" height={340} />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {latencyCharts.map((ct) => (
              <PlotlyChart
                key={ct}
                src={getPerfCompareChartUrl(rootPath, paths, ct)}
                fallbackTable={{
                  columns: ['Metric', 'Baseline', 'Candidate', 'Absolute delta', 'Percent delta'],
                  rows: (model?.deltas ?? []).map((delta) => ({
                    Metric: delta.metricKey,
                    Baseline: delta.baseline.primary,
                    Candidate: delta.candidate.primary,
                    'Absolute delta': delta.absoluteDelta.primary,
                    'Percent delta': delta.percentDelta.primary,
                  })),
                }}
                title={CHART_TITLES[ct]}
                height={340}
              />
            ))}
          </div>
        )}
      </Card>

      {/* Throughput group */}
      <Card title={t('performance.throughputGroup')}>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {THROUGHPUT_CHARTS.map((ct) => (
            <PlotlyChart
              key={ct}
              src={getPerfCompareChartUrl(rootPath, paths, ct)}
              fallbackTable={{
                columns: ['Metric', 'Baseline', 'Candidate', 'Absolute delta', 'Percent delta'],
                rows: (model?.deltas ?? []).map((delta) => ({
                  Metric: delta.metricKey,
                  Baseline: delta.baseline.primary,
                  Candidate: delta.candidate.primary,
                  'Absolute delta': delta.absoluteDelta.primary,
                  'Percent delta': delta.percentDelta.primary,
                })),
              }}
              title={CHART_TITLES[ct]}
              height={340}
            />
          ))}
        </div>
      </Card>
    </div>
  )
}
