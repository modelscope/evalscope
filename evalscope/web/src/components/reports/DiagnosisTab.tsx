import { formatMetric, formatProportion } from '@/domain/metric'
import { getDiagnosis, type DiagnosisFilters } from '@/api/reports'
import type { Fact, SampleKey } from '@/api/types'
import { parseReportRef } from '@/domain/report/reportRef'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { useLocale } from '@/contexts/LocaleContext'
import Card from '@/components/ui/Card'
import ScoreBar from '@/components/ui/ScoreBar'
import Badge from '@/components/ui/Badge'
import Tooltip from '@/components/ui/Tooltip'
import Skeleton from '@/components/ui/Skeleton'
import ErrorAlert from '@/components/ui/ErrorAlert'

interface Props {
  rootPath: string
  reportName: string
  datasetName: string
  onSamples: (filters: DiagnosisFilters, sample?: SampleKey) => void
}

function numericEntries(value: unknown): [string, number][] {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return []
  return Object.entries(value).filter((entry): entry is [string, number] => typeof entry[1] === 'number')
}

function FactData({ fact }: { fact: Fact }) {
  const { t } = useLocale()
  if (fact.id === 'F-mcq')
    return (
      <div className="grid gap-3 sm:grid-cols-2">
        {(['gold', 'prediction'] as const).map((key) => (
          <div key={key}>
            <h5>{t(`diagnosis.${key}`)}</h5>
            {numericEntries(fact.data[key]).map(([label, count]) => (
              <div key={label} className="flex items-center gap-2 text-sm">
                <span>{label}</span>
                <ScoreBar
                  score={count / (Number(fact.data.total) || 1)}
                  variant="proportion"
                  ariaLabel={`${label}: ${count}`}
                  track="fixed"
                />
                {count}
              </div>
            ))}
          </div>
        ))}
        <p className="text-sm">
          {t('diagnosis.choiceCoverage')}: {String(fact.data.covered)}/{String(fact.data.total)}
        </p>
        <ul className="text-sm">
          {(fact.data.confusions as { gold: string; prediction: string; count: number }[]).map((row) => (
            <li key={`${row.gold}:${row.prediction}`}>
              {row.gold} → {row.prediction}: {row.count}
            </li>
          ))}
        </ul>
      </div>
    )
  if (fact.id === 'F-scores' && Array.isArray(fact.data.metrics))
    return (
      <ul className="text-sm">
        {fact.data.metrics.map((entry: Record<string, unknown>) => (
          <li key={String(entry.metric)}>
            {String(entry.metric)}: {t('diagnosis.mean')} {formatMetric(Number(entry.mean), null).primary} ·{' '}
            {t('diagnosis.sampleCount', { count: Number(entry.n) })}
            {Array.isArray(entry.bins) && (
              <div className="mt-2 space-y-1">
                {(entry.bins as { bin: number; count: number }[]).map((bin) => {
                  const min = Number(entry.min),
                    max = Number(entry.max)
                  const low = min + ((max - min) * bin.bin) / 10,
                    high = min + ((max - min) * (bin.bin + 1)) / 10
                  return (
                    <div key={bin.bin} className="flex items-center gap-3 text-xs">
                      <span className="min-w-20 font-mono">
                        {formatMetric(low, null).primary}
                        {min !== max && `–${formatMetric(high, null).primary}`}
                      </span>
                      <ScoreBar
                        score={bin.count / Number(entry.n)}
                        variant="proportion"
                        ariaLabel={t('diagnosis.scoreRange')}
                        className="flex-1"
                      />
                      <span>{bin.count}</span>
                    </div>
                  )
                })}
              </div>
            )}
          </li>
        ))}
      </ul>
    )
  const rows = (key: string) =>
    Array.isArray(fact.data[key]) ? (fact.data[key] as Record<string, unknown>[]) : []
  if (fact.id === 'F-weak-subsets')
    return (
      <ul className="text-sm">
        {rows('items').length === 0 && <li>{t('diagnosis.noWeakSubsets')}</li>}
        {rows('items').map((row) => (
          <li key={String(row.subset)}>
            {String(row.subset)}: {formatMetric(Number(row.score), null).primary} ·{' '}
            {t('diagnosis.sampleCount', { count: Number(row.n) })}
          </li>
        ))}
      </ul>
    )
  if (fact.id === 'F-length')
    return (
      <ul className="text-sm">
        {rows('bins').map((row) => (
          <li key={String(row.length)}>
            {t(`diagnosis.${row.length}`)}: {formatProportion(Number(row.success_rate)).primary} ·{' '}
            {t('diagnosis.sampleCount', { count: Number(row.n) })}
          </li>
        ))}
      </ul>
    )
  if (fact.id === 'F-score-status')
    return (
      <ul className="text-sm">
        {numericEntries(fact.data.statuses).map(([key, value]) => (
          <li key={key}>
            {t(`diagnosis.status.${key}`) === `diagnosis.status.${key}`
              ? t('diagnosis.otherStatus')
              : t(`diagnosis.status.${key}`)}
            : {value}
          </li>
        ))}
        {numericEntries(fact.data.reasons).map(([key, value]) => (
          <li key={key}>
            {key}: {value}
          </li>
        ))}
      </ul>
    )
  if (fact.id === 'F-truncation')
    return (
      <p className="text-xs">
        {t('diagnosis.exactStops')}: {String(fact.data.exact_denominator)} · {t('diagnosis.inferredStops')}:{' '}
        {String(fact.data.heuristic_denominator)}
      </p>
    )
  if (fact.id === 'F-judge')
    return (
      <ul>
        {numericEntries(fact.data.failures).map(([key, count]) => (
          <li key={key}>
            {t(`diagnosis.status.${key}`) === `diagnosis.status.${key}`
              ? t('diagnosis.otherStatus')
              : t(`diagnosis.status.${key}`)}
            : {count}
          </li>
        ))}
      </ul>
    )
  if (fact.id === 'F-repeats')
    return (
      <p className="text-sm">
        {t('diagnosis.unstable')}: {String(fact.data.unstable_groups)} · {t('diagnosis.incomplete')}:{' '}
        {String(fact.data.incomplete_groups)}
      </p>
    )
  if (fact.id === 'F-agent')
    return (
      <ul className="text-sm">
        <li>
          {t('diagnosis.toolCalls')}: {String(fact.data.tool_calls)}
        </li>
        {numericEntries(fact.data.sample_categories).map(([key, value]) => (
          <li key={key}>
            {t(`diagnosis.finding.${key}`)}: {value}
          </li>
        ))}
      </ul>
    )
  if (fact.id === 'F-perf')
    return (
      <dl className="flex flex-wrap gap-4 text-sm">
        {[
          ['requests', 'requests'],
          ['input_tokens', 'inputTokens'],
          ['output_tokens', 'outputTokens'],
          ['latency_mean_s', 'latency'],
        ].map(
          ([key, label]) =>
            typeof fact.data[key] === 'number' && (
              <div key={key}>
                <dt>{t(`diagnosis.${label}`)}</dt>
                <dd className="font-mono">{Number(fact.data[key]).toLocaleString()}</dd>
              </div>
            ),
        )}
      </dl>
    )
  return null
}

export default function DiagnosisTab({ rootPath, reportName, datasetName, onSamples }: Props) {
  const { t } = useLocale()
  const resource = useAsyncResource(
    (signal) => getDiagnosis(rootPath, reportName, datasetName, signal),
    [rootPath, reportName, datasetName],
    { enabled: Boolean(datasetName), fallbackMessage: t('common.loadError') },
  )
  if (resource.loading) return <Skeleton lines={6} />
  if (resource.error) return <ErrorAlert>{resource.error}</ErrorAlert>
  const state = resource.data
  const ref = parseReportRef(reportName)
  const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`
  const command = `evalscope analyze --run-dir ${quote(`${rootPath}/${ref.runId}`)} --model ${quote(ref.modelId)} --benchmark ${quote(datasetName)} --level facts`
  if (!state || state.status !== 'ready' || !state.report)
    return (
      <div className="space-y-3" role="status">
        <p>
          {t(
            `diagnosis.${state?.status === 'stale' ? 'stale' : state?.status === 'error' ? 'error' : 'missing'}`,
          )}
        </p>
        <code className="block select-all overflow-x-auto whitespace-pre-wrap rounded bg-[var(--bg-deep)] p-3 text-sm">
          {command}
        </code>
      </div>
    )
  const report = state.report
  const c = report.coverage
  const subsets = [...new Set(report.patterns.flatMap((p) => Object.keys(p.by_subset)))].sort()
  const types = [...new Set(report.patterns.map((p) => p.failure_type))]
  const facts = [...report.facts].sort(
    (a, b) => ({ critical: 0, warn: 1, info: 2 })[a.severity] - { critical: 0, warn: 1, info: 2 }[b.severity],
  )
  const issueName = (type: string) => t(`diagnosis.issue.${type}`)
  const componentName = (value: string) => t(`diagnosis.${value === 'unknown' ? 'unknownComponent' : value}`)
  const sampleLabel = (sample: SampleKey) =>
    `${sample.subset === 'default' ? '' : `${sample.subset} · `}${t('diagnosis.sampleLink', { index: sample.index })}`
  const factTitle = (fact: Fact) => {
    const key = `diagnosis.fact.${fact.id}`
    return t(key) === key ? fact.title : t(key)
  }
  const factValue = (fact: Fact) =>
    fact.value === null
      ? ''
      : fact.unit === 'fraction'
        ? formatProportion(fact.value).primary
        : formatMetric(fact.value, null).primary
  const clues = Math.max(0, c.task_failed - c.unexplained_failures)
  const attributions = ['model', 'guard', 'environment', 'benchmark', 'unknown']
    .map((key) => ({
      label: componentName(key),
      tone: key,
      count: report.patterns.filter((p) => p.attribution === key).reduce((sum, p) => sum + p.count, 0),
    }))
    .filter((item) => item.count > 0)
  const cellCount = (subset: string, type: string) => report.patterns
    .filter(p => p.failure_type === type)
    .reduce((sum, p) => sum + (p.by_subset[subset] ?? 0), 0)
  const maxCell = Math.max(1, ...subsets.flatMap(subset => types.map(type => cellCount(subset, type))))
  const leading = report.patterns.find((p) => p.failure_type !== 'unexplained')
  const basis = (fact: Fact) => {
    const key = `diagnosis.description.${fact.id}`
    return t(key) === key ? fact.definition : t(key)
  }
  const renderFact = (fact: Fact, expanded = false) => (
    <li id={`${expanded ? 'detail-' : ''}${fact.id}`} key={fact.id} className="diagnosis-fact">
      <div className="flex items-start gap-2">
        <Badge
          variant={fact.severity === 'critical' ? 'danger' : fact.severity === 'warn' ? 'warning' : 'default'}
          className="shrink-0 rounded-md"
        >
          {t(`diagnosis.severity.${fact.severity}`)}
        </Badge>
        <div className="min-w-0 flex-1">
          <Tooltip content={basis(fact)} maxWidth={340}>
            <span className="text-sm">
              {factTitle(fact)}
              {fact.value !== null && (
                <>
                  {' '}
                  · <strong className="font-mono">{factValue(fact)}</strong>
                </>
              )}
            </span>
          </Tooltip>
          {expanded && fact.samples.length > 0 && (
            <div className="mt-1 flex flex-wrap gap-x-3">
              {fact.samples.map((sample) => (
                <button
                  key={`${sample.subset}:${sample.index}`}
                  className="diagnosis-link text-xs"
                  onClick={() => onSamples({}, sample)}
                >
                  {sampleLabel(sample)}
                </button>
              ))}
            </div>
          )}
          {expanded && Object.keys(fact.data).length > 0 && (
            <div className="mt-2 text-xs text-[var(--text-muted)]">
              <FactData fact={fact} />
            </div>
          )}
        </div>
        {!expanded && fact.samples[0] && (
          <button className="diagnosis-link shrink-0 text-xs" onClick={() => onSamples({}, fact.samples[0])}>
            {sampleLabel(fact.samples[0])}
          </button>
        )}
      </div>
    </li>
  )
  return (
    <div className="space-y-4">
      <div className="text-xs text-[var(--text-muted)]">
        {t(report.level === 'facts' ? 'diagnosis.ruleAnalysis' : 'diagnosis.modelAnalysis')} · {t('diagnosis.generated')}{' '}
        {new Date(report.generated_at).toLocaleString()}
      </div>
      {report.level !== 'facts' && <Card title={t('diagnosis.modelCoverage')} className="min-w-0">
        <p className="text-sm">{t('diagnosis.proposalCoverage', { ok: c.diagnosed_ok, selected: c.selected, total: c.diagnosis_population })}</p>
        <p className="mt-2 text-xs text-[var(--text-muted)]">{t('diagnosis.proposalNotice')}</p>
        {c.failed_unexplained_estimate && <p className="mt-2 text-sm">
          {t('diagnosis.weightedUnresolved')}: {formatProportion(c.failed_unexplained_estimate.proportion).primary}
          <span className="ml-2 text-xs text-[var(--text-muted)]">{t('diagnosis.stratifiedInterval')} {formatProportion(c.failed_unexplained_estimate.lower).primary}–{formatProportion(c.failed_unexplained_estimate.upper).primary}</span>
        </p>}
        {report.self_evaluation && <p className="mt-2 text-xs text-[var(--warning-text)]">{t('diagnosis.selfEvaluation')}</p>}
      </Card>}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="min-w-0">
          <h3 className="text-sm text-[var(--text-muted)]">{t('diagnosis.failureRate')}</h3>
          <div className="mt-3 flex flex-wrap items-baseline gap-3">
            <strong className="text-2xl font-semibold">
              {c.task_success_known ? formatProportion(c.task_failed / c.task_success_known).primary : '—'}
            </strong>
            <span className="text-xs text-[var(--text-muted)]">
              {c.task_success_known ? `${c.task_failed} / ${c.task_success_known}` : t('diagnosis.noTaskOutcome')}
            </span>
          </div>
        </Card>
        <Card className="min-w-0">
          <h3 className="text-sm text-[var(--text-muted)]">{t('diagnosis.ruleCoverage')}</h3>
          <Segments
            emptyLabel={t(c.task_success_known ? 'diagnosis.noFailures' : 'diagnosis.noTaskOutcome')}
            items={[
              { label: t('diagnosis.hasClues'), count: clues, tone: 'rule' },
              { label: t('diagnosis.pending'), count: c.unexplained_failures, tone: 'unknown' },
            ]}
          />
        </Card>
        <Card className="min-w-0">
          <h3 className="text-sm text-[var(--text-muted)]">{t('diagnosis.attribution')}</h3>
          <Segments items={attributions} />
        </Card>
        <Card className="min-w-0">
          <h3 className="text-sm text-[var(--text-muted)]">{t('diagnosis.analysisUsage')}</h3>
          <strong className="mt-3 block text-2xl font-semibold">
            {t('diagnosis.callCount', { count: report.cost.calls })}
          </strong>
          <span className="text-xs text-[var(--text-muted)]">
            {report.cost.calls
              ? t('diagnosis.tokenCount', { count: report.cost.input_tokens + report.cost.output_tokens })
              : !report.cost.cached_calls && t('diagnosis.noModelCalls')}
          </span>
          {report.cost.cached_calls > 0 && (
            <p className="text-xs text-[var(--text-muted)]">
              {t('diagnosis.cachedCalls', { count: report.cost.cached_calls })}
            </p>
          )}
        </Card>
      </div>
      <Card
        className="min-w-0"
        title={t('diagnosis.patterns', { count: c.needs_attention })}
      >
        {report.patterns.length ? (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[800px] text-left text-sm">
              <thead>
                <tr>
                  {['patternId', 'type', 'attribution', 'count', 'share', 'source', 'examples'].map((key) => (
                    <th key={key} className="whitespace-nowrap px-2 pb-2 text-xs font-medium text-[var(--text-muted)]">
                      {t(`diagnosis.${key}`)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {report.patterns.map((pattern, index) => (
                  <tr
                    key={pattern.id}
                    className="cursor-pointer border-t border-[var(--border)] hover:bg-[var(--bg-card2)]"
                    onClick={() => onSamples({ pattern_id: pattern.id })}
                  >
                    <td className="whitespace-nowrap px-2 py-1 font-mono text-xs">P{index + 1}</td>
                    <td className="px-2 py-1">
                      <button className="min-h-9 whitespace-nowrap text-left">{issueName(pattern.failure_type)}</button>
                      {pattern.detail.includes(' @ ') && (
                        <span className="block text-xs text-[var(--text-muted)]">
                          {pattern.detail.split(' @ ').slice(1).join(' @ ')}
                        </span>
                      )}
                    </td>
                    <td className="whitespace-nowrap px-2 py-1 text-xs">{componentName(pattern.attribution)}</td>
                    <td className="px-2 py-1 tabular-nums">{pattern.count}</td>
                    <td className="px-2 py-1">
                      <ScoreBar
                        score={pattern.share}
                        variant="proportion"
                        track="fill"
                        ariaLabel={issueName(pattern.failure_type)}
                        className="diagnosis-pattern-bar"
                      />
                      {pattern.estimated_share && <div className="mt-1 text-xs text-[var(--text-muted)]">
                        {t('diagnosis.populationEstimate')} {formatProportion(pattern.estimated_share.proportion).primary}
                        <span className="block">{t('diagnosis.stratifiedInterval')} {formatProportion(pattern.estimated_share.lower).primary}–{formatProportion(pattern.estimated_share.upper).primary}</span>
                      </div>}
                    </td>
                    <td className="whitespace-nowrap px-2 py-1 text-xs text-[var(--text-muted)]">
                      {pattern.source === 'none' ? '—' : t(`diagnosis.sources.${pattern.source}`)}
                    </td>
                    <td className="px-2 py-1">
                      <div className="flex flex-wrap gap-x-3">
                        {pattern.examples.map((sample) => (
                          <button
                            key={`${sample.subset}:${sample.index}`}
                            className="diagnosis-link min-h-9 whitespace-nowrap text-xs"
                            onClick={(event) => {
                              event.stopPropagation()
                              onSamples({ pattern_id: pattern.id }, sample)
                            }}
                          >
                            {sampleLabel(sample)}
                          </button>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-[var(--text-muted)]">{t('diagnosis.noPatterns')}</p>
        )}
      </Card>
      <div className="grid items-stretch gap-4 xl:grid-cols-2">
        <Card className="min-w-0" title={t('diagnosis.heatmap')}>
          {subsets.length ? (
            <div className="overflow-x-auto">
              <table className="w-full border-separate border-spacing-2 text-xs">
                <thead>
                  <tr>
                    <th />
                    <>
                      {types.map((type) => (
                        <th key={type} className="max-w-24 pb-1 font-medium text-[var(--text-muted)]">
                          {issueName(type)}
                        </th>
                      ))}
                    </>
                  </tr>
                </thead>
                <tbody>
                  {subsets.map((subset) => (
                    <tr key={subset}>
                      <th className="pr-2 text-left font-normal text-[var(--text-muted)]">
                        {subset === 'default' ? t('diagnosis.defaultSubset') : subset}
                      </th>
                      {types.map((type) => {
                        const count = cellCount(subset, type)
                        return (
                          <td key={type}>
                            <button
                              disabled={!count}
                              className={`min-h-10 w-full min-w-10 rounded text-sm tabular-nums disabled:text-[var(--text-muted)] ${count / maxCell < 2 / 3 ? 'diagnosis-heat-low' : ''}`}
                              style={{
                                background: count
                                  ? `color-mix(in srgb, var(--accent-dark) ${25 + (75 * count) / maxCell}%, var(--bg-card2))`
                                  : 'var(--bg-card2)',
                                color: count ? 'var(--diagnosis-heat-text, var(--text-on-filled))' : undefined,
                              }}
                              aria-label={`${subset} · ${issueName(type)}: ${count}`}
                              onClick={() => onSamples({ subset_name: subset, failure_type: type })}
                            >
                              {count || '—'}
                            </button>
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-sm text-[var(--text-muted)]">{t('diagnosis.noPatterns')}</p>
          )}
        </Card>
        <Card className="min-w-0" title={t('diagnosis.facts')}>
          <ul className="space-y-2">{facts.slice(0, 5).map((fact) => renderFact(fact))}</ul>
          {facts.length > 0 && (
            <details className="mt-3 border-t border-[var(--border)] pt-2">
              <summary className="cursor-pointer py-2 text-xs text-[var(--accent)]">
                {t('diagnosis.allFacts', { count: facts.length })}
              </summary>
              <ul className="mt-2 space-y-3">{facts.map((fact) => renderFact(fact, true))}</ul>
              {Object.keys(c.skipped_facts).length > 0 && (
                <details className="mt-2 text-xs text-[var(--text-muted)]">
                  <summary className="cursor-pointer py-2">
                    {t('diagnosis.skipped')} ({Object.keys(c.skipped_facts).length})
                  </summary>
                  <ul className="space-y-1">
                    {Object.keys(c.skipped_facts).map((key) => (
                      <li key={key}>
                        {t(`diagnosis.unavailable.${key}`) === `diagnosis.unavailable.${key}`
                          ? t('diagnosis.noData')
                          : t(`diagnosis.unavailable.${key}`)}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </details>
          )}
        </Card>
      </div>
      {report.narrative && <Card className="min-w-0" title={t('diagnosis.modelSummary')}>
        <ul className="space-y-3 text-sm">{[...report.narrative.findings, ...report.narrative.recommendations].map((item, index) => <li key={index}>
          {item.text} {item.refs.map(ref => {
            const pattern = report.patterns.find(p => p.id === ref)
            const fact = report.facts.find(f => f.id === ref)
            return <button key={ref} className="diagnosis-link ml-1" onClick={() => {
              if (pattern) onSamples({ pattern_id: pattern.id })
              else if (fact?.samples[0]) onSamples({}, fact.samples[0])
              else {
                const element = document.getElementById(`detail-${ref}`)
                const details = element?.closest('details')
                if (details) details.open = true
                element?.scrollIntoView({ block: 'center' })
              }
            }}>[{pattern ? `P${report.patterns.indexOf(pattern) + 1}` : fact ? t(`diagnosis.fact.${fact.id}`) : ref}]</button>
          })}
        </li>)}</ul>
      </Card>}
      <Card className="min-w-0" title={t('diagnosis.summary')}>
        <ul className="space-y-2 text-sm">
          <li>
            {t(c.task_success_known ? 'diagnosis.summaryResults' : 'diagnosis.summaryNoOutcome', {
              samples: c.samples,
              failed: c.task_failed,
              known: c.task_success_known,
            })}
          </li>
          {(c.excluded > 0 || c.outcome_unknown > 0 || c.errored_without_record > 0 || c.unreadable > 0) && (
            <li className="text-[var(--text-muted)]">
              {[
                ['unscored', c.excluded],
                ['unknown', c.outcome_unknown],
                ['executionErrors', c.errored_without_record],
                ['unreadable', c.unreadable],
              ]
                .filter(([, count]) => Number(count) > 0)
                .map(([key, count]) => `${t(`diagnosis.${key}`)} ${count}`)
                .join(' · ')}
            </li>
          )}
          {leading && (
            <li>
              {t('diagnosis.summaryPattern', { type: issueName(leading.failure_type), count: leading.count })}{' '}
              <button className="diagnosis-link" onClick={() => onSamples({ pattern_id: leading.id })}>
                [P{report.patterns.indexOf(leading) + 1}]
              </button>
            </li>
          )}
          {report.level === 'facts' && c.task_failed > 0 && (
            <li>{t('diagnosis.summaryPending', { count: c.unexplained_failures, total: c.task_failed })}</li>
          )}
        </ul>
      </Card>
    </div>
  )
}

function Segments({ items, emptyLabel }: { items: { label: string; count: number; tone: string }[]; emptyLabel?: string }) {
  const total = items.reduce((sum, item) => sum + item.count, 0)
  const { t } = useLocale()
  if (!total) return <p className="mt-3 text-xs text-[var(--text-muted)]">{emptyLabel ?? t('diagnosis.noData')}</p>
  return (
    <div className="mt-3">
      <div
        role="img"
        aria-label={items.map((item) => `${item.label}: ${item.count}`).join(' · ')}
        className="flex h-3 overflow-hidden rounded bg-[var(--bg-card2)]"
      >
        {items
          .filter((item) => item.count > 0)
          .map((item) => (
            <span
              key={item.tone}
              className={`diagnosis-segment-${item.tone}`}
              style={{ width: `${(100 * item.count) / total}%` }}
            />
          ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-2 gap-y-1 text-xs text-[var(--text-muted)]">
        {items.map((item) => (
          <span key={item.tone}>
            {item.label} {total ? formatProportion(item.count / total).primary : '—'}
          </span>
        ))}
      </div>
    </div>
  )
}
