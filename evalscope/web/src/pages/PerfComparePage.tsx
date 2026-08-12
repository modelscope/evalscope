import { useMemo } from 'react'
import { useLocale } from '@/contexts/LocaleContext'
import { useScan } from '@/contexts/ReportsContext'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { useQueryParams } from '@/hooks/useQueryParams'
import { getPerfCompareChartUrl, getPerfDetail } from '@/api/perf'
import type { PerfDetailResponse } from '@/api/types'
import { buildCompareModel, classifySampleSize } from '@/domain/perf/deltaModel'
import type { DeltaVerdict, PerfCompareModel, SampleTier } from '@/domain/perf/deltaModel'
import Breadcrumb from '@/components/ui/Breadcrumb'
import Badge from '@/components/ui/Badge'
import Callout from '@/components/ui/Callout'
import Card from '@/components/ui/Card'
import Skeleton from '@/components/ui/Skeleton'
import PerfChartGroup from '@/components/perf/PerfChartGroup'
import ErrorAlert from '@/components/ui/ErrorAlert'
import { LATENCY_CHARTS, THROUGHPUT_CHARTS } from '@/domain/perf/charts'
import { formatTimestamp } from '@/utils/formatUtils'
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
  improvement: 'perf.archive.verdictImprovement',
  regression: 'perf.archive.verdictRegression',
  neutral: 'perf.archive.verdictNeutral',
  incomputable: 'perf.archive.verdictIncomputable',
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
  const ts = formatTimestamp(run.generated_at, 'seconds')
  return ts ? `${parts.join(' · ')} · ${ts}` : parts.join(' · ')
}

export default function PerfComparePage() {
  const { t } = useLocale()
  const { get, set } = useQueryParams()
  const { rootPath: ctxRoot } = useScan()

  const rootPath = get('root_path') ?? ctxRoot
  const paths = useMemo(
    () => (get('paths') ?? '').split(';').map((p) => p.trim()).filter(Boolean),
    [get],
  )

  // Persisted baseline selection: the effective baseline id lives in
  // the `baseline` query param so a swap survives subsequent loads of this view.
  const baselineParam = get('baseline') ?? ''

  const pathsKey = paths.join(';')
  const comparison = useAsyncResource(
    async (signal) => {
      // Settled, not all-or-nothing: a run that has since been deleted must not
      // hide the ones that are still there.
      const results = await Promise.allSettled(paths.map((p) => getPerfDetail(rootPath, p, signal)))
      const runs = results
        .filter((r): r is PromiseFulfilledResult<PerfDetailResponse> => r.status === 'fulfilled')
        .map((r) => r.value)
      return { runs, missingCount: paths.length - runs.length }
    },
    [rootPath, pathsKey],
    { enabled: paths.length >= 2, fallbackMessage: t('perf.archive.compareLoadError') },
  )

  // `null` is this view's "not resolved yet" signal, driving every skeleton below.
  const details = comparison.loading ? null : (comparison.data?.runs ?? null)
  const missingCount = comparison.data?.missingCount ?? 0
  // Nothing loaded at all: there is no comparison to show, only the failure.
  const loadError = comparison.error || (details?.length === 0 ? t('perf.archive.compareLoadError') : '')

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
            { label: t('perf.archive.comparePageTitle') },
          ]}
        />
        <div className="py-16 text-center type-body-sm text-[var(--text-muted)]">
          {t('perf.archive.selectToCompare')}
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
  const chartFallback = {
    columns: ['Metric', 'Baseline', 'Candidate', 'Absolute delta', 'Percent delta'],
    rows: (model?.deltas ?? []).map((delta) => ({
      Metric: delta.metricLabel,
      Baseline: delta.baseline.primary,
      Candidate: delta.candidate.primary,
      'Absolute delta': delta.absoluteDelta.primary,
      'Percent delta': delta.percentDelta.primary,
    })),
  }

  return (
    <div className="page-enter flex flex-col gap-4">
      <Breadcrumb
        items={[
          { label: t('nav.performance'), href: `/performance?root_path=${encodeURIComponent(rootPath)}` },
          { label: t('perf.archive.comparePageTitle') },
        ]}
      />

      {/* Header */}
      <div className="flex items-start gap-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-5">
        <span className="text-[var(--accent)] shrink-0 mt-0.5">
          <GitCompareArrows size={20} />
        </span>
        <div className="flex flex-col gap-1 min-w-0">
          <h1 className="type-title-md text-[var(--text)]">
            {t('perf.archive.comparing', { n: paths.length })}
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
        <ErrorAlert className="p-6 type-body-sm">{loadError}</ErrorAlert>
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
                  <Badge>{t('perf.archive.baselineBadge')}</Badge>
                  <span className="type-table-xs">
                    {t('perf.archive.effectiveBaseline')}
                  </span>
                </div>
                <div className="type-body-sm text-[var(--text)] break-all" data-testid="baseline-label">
                  {runLabel(baselineRun)}
                </div>
                <div className="type-body-xs text-[var(--text-muted)] tabular-nums">
                  {t('perf.archive.sampleCount', { n: model.sampleCounts[model.baselineId] ?? 0 })}
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
                {t('perf.archive.swapBaseline')}
              </button>

              <div className="flex flex-col gap-1 min-w-0 flex-1 md:text-right">
                <div className="flex items-center gap-2 md:justify-end">
                  <Badge variant="success">{t('perf.archive.candidateBadge')}</Badge>
                </div>
                <div className="type-body-sm text-[var(--text)] break-all" data-testid="candidate-label">
                  {runLabel(candidateRun)}
                </div>
                <div className="type-body-xs text-[var(--text-muted)] tabular-nums">
                  {t('perf.archive.sampleCount', { n: model.sampleCounts[model.candidateId] ?? 0 })}
                </div>
              </div>
            </div>

            {/* Warnings — informational, never blocking */}
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

            {sampleTier !== 'ok' && (
              <Callout
                variant="warning"
                icon={
                  <AlertTriangle
                    size={15}
                    className={sampleTier === 'critical' ? 'text-[var(--danger)]' : 'text-[var(--yellow)]'}
                  />
                }
                className={
                  sampleTier === 'critical'
                    ? 'rounded-[var(--radius-sm)] border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--text)]'
                    : 'rounded-[var(--radius-sm)]'
                }
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

            {/* Delta summary table */}
            <Card title={t('perf.archive.deltaSummary')}>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse" data-testid="delta-table">
                  <thead>
                    <tr>
                      {[
                        t('perf.archive.metricCol'),
                        t('perf.archive.baselineCol'),
                        t('perf.archive.candidateCol'),
                        t('perf.archive.absDeltaCol'),
                        t('perf.archive.pctDeltaCol'),
                        t('perf.archive.directionCol'),
                      ].map((label, i) => (
                        <th
                          key={label}
                          className={`type-table-xs px-3 py-2 whitespace-nowrap border-b border-[var(--border)] ${i === 0 ? 'text-left' : 'text-right'}`}
                        >
                          {label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {model.deltas.map((delta) => {
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
                            {/* The field key is the label: the backend already names the field in
                                the form the perf tables use, so no spec lookup is needed. */}
                            <span className="block font-medium">{delta.metricLabel}</span>
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
              <p className="type-body-xs mt-3 text-[var(--text-muted)]">{t('perf.archive.deltaInfoNote')}</p>
            </Card>

            {/* Configuration differences */}
            <Card title={t('perf.archive.configDiffTitle')}>
              {model.configDiff.length === 0 ? (
                <div className="type-body-sm text-[var(--text-muted)]">{t('perf.archive.noConfigDiff')}</div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse" data-testid="config-diff-table">
                    <thead>
                      <tr>
                        {[
                          t('perf.archive.configKeyCol'),
                          t('perf.archive.baselineCol'),
                          t('perf.archive.candidateCol'),
                        ].map((label, i) => (
                          <th
                            key={label}
                            className={`type-table-xs px-3 py-2 whitespace-nowrap border-b border-[var(--border)] ${i === 0 ? 'text-left' : 'text-right'}`}
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
        getChartUrl={(chart) => getPerfCompareChartUrl(rootPath, paths, chart)}
        loading={details === null}
      />
      <PerfChartGroup
        title={t('perf.archive.throughputGroup')}
        charts={THROUGHPUT_CHARTS}
        fallbackTable={chartFallback}
        getChartUrl={(chart) => getPerfCompareChartUrl(rootPath, paths, chart)}
        loading={details === null}
      />
    </div>
  )
}
