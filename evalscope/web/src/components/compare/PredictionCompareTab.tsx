import { ArrowDown, ArrowUp } from 'lucide-react'
import { parseReportRef } from '@/domain/report/reportRef'
import { formatMetric } from '@/domain/metric'
import { RATIO_PERCENT_SEMANTICS } from '@/domain/report/primaryMetrics'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import Card from '@/components/ui/Card'
import Eyebrow from '@/components/ui/Eyebrow'
import ErrorAlert from '@/components/ui/ErrorAlert'
import ScoreBadge from '@/components/ui/ScoreBadge'
import Select from '@/components/ui/Select'
import Skeleton from '@/components/ui/Skeleton'
import SegmentedControl, { type SegmentedOption } from '@/components/ui/SegmentedControl'
import ScoreThresholdInput from '@/components/ui/ScoreThresholdInput'
import SampleNavigator from '@/components/reports/SampleNavigator'
import ChatView from '@/components/chat/ChatView'
import EmptyStateSystem from '@/components/common/EmptyStateSystem'
import {
  MODEL_PALETTE,
  type MergedPrediction,
  type PerModelFilter,
  type Translate,
} from '@/components/compare/compareSlots'

export default function PredictionCompareTab({
  reportNames,
  displayNames,
  displayLabels,
  predCommonDatasets,
  datasetLabels,
  selectedDs,
  setSelectedDs,
  subsets,
  selectedSubset,
  setSelectedSubset,
  perModelFilter,
  setPerModelFilter,
  threshold,
  setThreshold,
  aboveRates,
  mergedPredictions,
  filtered,
  currentRow,
  page,
  setPage,
  totalPages,
  predictionsLoading,
  predictionsError,
  onRetryPredictions,
  t,
}: {
  reportNames: string[]
  displayNames: Record<string, string>
  displayLabels: Record<string, string>
  predCommonDatasets: string[]
  datasetLabels: Record<string, string>
  selectedDs: string
  setSelectedDs: (ds: string) => void
  subsets: string[]
  selectedSubset: string
  setSelectedSubset: (s: string) => void
  perModelFilter: Record<string, PerModelFilter>
  setPerModelFilter: (f: Record<string, PerModelFilter>) => void
  threshold: number
  setThreshold: (n: number) => void
  aboveRates: Record<string, number>
  mergedPredictions: MergedPrediction[]
  filtered: MergedPrediction[]
  currentRow: MergedPrediction | null
  page: number
  setPage: (p: number) => void
  totalPages: number
  predictionsLoading: boolean
  predictionsError: string
  onRetryPredictions: () => void
  t: Translate
}) {
  // ── Filter helpers ──────────────────────────────────────────────
  const setModelFilter = (name: string, f: PerModelFilter) =>
    setPerModelFilter({ ...perModelFilter, [name]: f })

  const setAllFilters = (f: PerModelFilter) => {
    const next: Record<string, PerModelFilter> = {}
    reportNames.forEach((n) => { next[n] = f })
    setPerModelFilter(next)
  }

  const isAllAny = reportNames.every((n) => (perModelFilter[n] ?? 'any') === 'any')
  const isAllAbove = reportNames.every((n) => (perModelFilter[n] ?? 'any') === 'above')
  const isAllBelow = reportNames.every((n) => (perModelFilter[n] ?? 'any') === 'below')

  // Empty when the per-model filters are mixed, so no preset reads as active.
  const presetValue: PerModelFilter | '' = isAllAny ? 'any' : isAllAbove ? 'above' : isAllBelow ? 'below' : ''

  if (predCommonDatasets.length === 0) {
    return (
      <Card>
        <EmptyStateSystem
          reason="no-match"
          context={{ view: 'compare' }}
          hint={t('compare.noCommon')}
        />
      </Card>
    )
  }

  // Presets set every model's filter at once; the tri-state options are per model.
  const presetOptions: SegmentedOption<PerModelFilter>[] = [
    { value: 'any', label: t('common.all') },
    { value: 'above', label: t('compare.allAbove') },
    { value: 'below', label: t('compare.allBelow') },
  ]
  const applyPreset = (next: PerModelFilter) => {
    if (next === 'any') setPerModelFilter({})
    else setAllFilters(next)
    setPage(1)
  }
  const modelFilterOptions: SegmentedOption<PerModelFilter>[] = [
    { value: 'any', label: t('compare.any') },
    {
      value: 'above',
      label: t('prediction.above'),
      accessibleLabel: t('prediction.aboveFilter'),
      icon: <ArrowUp size={12} />,
    },
    {
      value: 'below',
      label: t('prediction.below'),
      accessibleLabel: t('prediction.belowFilter'),
      icon: <ArrowDown size={12} />,
    },
  ]

  return (
    <div className="flex flex-col gap-4">
      {/* ── Dataset controls + per-model filters ── */}
      <div
        data-testid="prediction-controls"
        className="flex flex-col gap-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-card)] p-4"
      >
        <div className="grid items-end gap-3 lg:grid-cols-[minmax(200px,1fr)_minmax(180px,1fr)_112px]">
          <div className="min-w-0">
            <Select
              label={t('compare.selectDataset')}
              options={predCommonDatasets.map((ds) => ({ value: ds, label: datasetLabels[ds] || ds }))}
              value={selectedDs}
              onChange={(v) => { setSelectedDs(v); setSelectedSubset('') }}
              placeholder={`-- ${t('compare.selectDataset')} --`}
            />
          </div>
          {subsets.length > 0 && (
            <div className="min-w-0">
              <Select
                label={t('compare.selectSubset')}
                options={subsets.map((s) => ({ value: s, label: s }))}
                value={selectedSubset}
                onChange={setSelectedSubset}
                placeholder={`-- ${t('compare.selectSubset')} --`}
              />
            </div>
          )}
          {subsets.length === 0 && <div aria-hidden="true" className="hidden lg:block" />}
          <ScoreThresholdInput
            id="compare-score-threshold"
            value={threshold}
            onChange={(next) => { setThreshold(next); setPage(1) }}
            label={t('compare.scoreThreshold')}
          />
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-[var(--border)] pt-3">
          <Eyebrow as="span">{t('compare.filterByModel')}</Eyebrow>
          <SegmentedControl
            options={presetOptions}
            value={presetValue as PerModelFilter}
            onChange={applyPreset}
            ariaLabel={t('compare.filterByModel')}
            size="sm"
          />
        </div>

        {/* Per-model tri-state filters */}
        <div
          data-testid="prediction-model-filters"
          className="grid gap-2 md:grid-cols-2 xl:grid-cols-3"
        >
          {reportNames.map((name, idx) => {
            const palette = MODEL_PALETTE[idx] ?? MODEL_PALETTE[0]
            const cur = perModelFilter[name] ?? 'any'
            const rate = aboveRates[name]
            const modelLabel = displayLabels[name]
              ?? displayNames[name]
              ?? (parseReportRef(name).modelId || name)
            return (
              <div
                key={name}
                className="flex min-w-0 flex-col gap-2 rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-deep)] p-3"
              >
                <div className="flex min-w-0 items-center gap-2">
                  <span
                    className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ backgroundColor: palette.dot }}
                  />
                  <span
                    className="min-w-0 flex-1 truncate text-xs font-medium text-[var(--text)]"
                    title={modelLabel}
                  >
                    {modelLabel}
                  </span>
                  {rate !== undefined && mergedPredictions.length > 0 && (
                    <Badge variant="default" className="shrink-0">
                      {formatMetric(rate, RATIO_PERCENT_SEMANTICS).primary}
                    </Badge>
                  )}
                </div>

                {/* Tri-state chips. This is the model's filter group, so it
                    carries the per-model label. */}
                <SegmentedControl
                  options={modelFilterOptions}
                  value={cur}
                  onChange={(next) => { setModelFilter(name, next); setPage(1) }}
                  ariaLabel={`${t('compare.filterByModel')}: ${modelLabel}`}
                  size="sm"
                  fullWidth
                />
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Stats Bar + Pagination ── */}
      {!predictionsLoading && mergedPredictions.length > 0 && (
        <div className="flex items-center justify-between px-4 py-2.5 rounded-[var(--radius)] bg-[var(--bg-card)] border border-[var(--border)] gap-2 flex-wrap">
          <span className="text-sm text-[var(--text-muted)]">
            {t('compare.showing')}{' '}
            <strong className="text-[var(--text)]">{filtered.length}</strong>{' '}
            {t('compare.of')}{' '}
            <strong className="text-[var(--text)]">{mergedPredictions.length}</strong>{' '}
            {t('compare.predictions')}
            {currentRow && (
              <span className="ml-2 text-xs opacity-50">#{currentRow.Index}</span>
            )}
          </span>
          <SampleNavigator page={page} total={totalPages} onPageChange={setPage}>
            <span className="min-w-[5rem] text-center text-sm tabular-nums text-[var(--text-muted)]">
              {t('compare.sample')} {page} / {totalPages}
            </span>
          </SampleNavigator>
        </div>
      )}

      {/* ── Loading skeleton ── */}
      {predictionsLoading && <Skeleton height={400} />}

      {predictionsError && (
        <ErrorAlert className="flex items-center justify-between gap-3">
          <span className="type-body-sm break-words">{predictionsError}</span>
          <Button size="sm" variant="outline" onClick={onRetryPredictions}>
            {t('common.retry')}
          </Button>
        </ErrorAlert>
      )}

      {/* ── ChatView Columns ── */}
      {!predictionsLoading && currentRow && (
        <div
          className="grid gap-4"
          style={{
            gridTemplateColumns: `repeat(${reportNames.length}, minmax(0, 1fr))`,
          }}
        >
          {reportNames.map((name, idx) => {
            const palette = MODEL_PALETTE[idx] ?? MODEL_PALETTE[0]
            const modelRow = currentRow.models[name]
            if (!modelRow) return null
            return (
              <div
                key={name}
                className="flex flex-col rounded-[var(--radius)] border overflow-hidden"
                style={{ borderColor: palette.border, background: palette.bg }}
              >
                {/* Column Header */}
                <div
                  className="flex items-center justify-between px-4 py-2.5 border-b shrink-0"
                  style={{ borderColor: palette.border, background: palette.headerBg }}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span
                      className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                      style={{ backgroundColor: palette.dot }}
                    />
                    <span
                      className="text-xs font-semibold truncate"
                      style={{ color: palette.dot }}
                      title={displayLabels[name] ?? displayNames[name] ?? (parseReportRef(name).modelId || name)}
                    >
                      {displayLabels[name] ?? displayNames[name] ?? (parseReportRef(name).modelId || name)}
                    </span>
                  </div>
                  {/*
                    Native normalized score with a neutral gradient, independent
                    of the view threshold. The threshold only filters
                    rows in this view and is not a pass/fail verdict.
                  */}
                  <ScoreBadge
                    score={modelRow.NScore}
                    semantics={RATIO_PERCENT_SEMANTICS}
                    className="ml-2 shrink-0 !font-mono"
                  />
                </div>

                {/* ChatView */}
                <div
                  className="overflow-y-auto p-3"
                  style={{ maxHeight: 'calc(100vh - 380px)', minHeight: '280px' }}
                >
                  <ChatView prediction={modelRow} threshold={threshold} />
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* ── Empty state ── */}
      {!predictionsLoading && !predictionsError && mergedPredictions.length === 0 && (
        <Card>
          <EmptyStateSystem reason="no-data" context={{ view: 'compare' }} />
        </Card>
      )}

      {!predictionsLoading && !predictionsError && mergedPredictions.length > 0 && filtered.length === 0 && (
        <Card>
          <EmptyStateSystem reason="no-match" context={{ view: 'compare' }} />
        </Card>
      )}
    </div>
  )
}
