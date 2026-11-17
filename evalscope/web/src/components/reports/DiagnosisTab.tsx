import { useState } from 'react'
import { AlertTriangle, ArrowRight, FileText } from 'lucide-react'
import DiagnosisSamplePanel from './DiagnosisSamplePanel'
import type { ReportMetric } from '@/domain/report/primaryMetrics'
import { formatMetric, formatProportion } from '@/domain/metric'
import { getDiagnosis, type DiagnosisFilters } from '@/api/reports'
import type { Fact, SampleKey } from '@/api/types'
import { parseReportRef } from '@/domain/report/reportRef'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { useLocale } from '@/contexts/LocaleContext'
import Card from '@/components/ui/Card'
import ScoreBar from '@/components/ui/ScoreBar'
import Skeleton from '@/components/ui/Skeleton'
import MarkdownRenderer from '@/components/ui/MarkdownRenderer'
import ErrorAlert from '@/components/ui/ErrorAlert'

interface Props {
  metrics?: ReportMetric[]
  legacyAnalysis?: string
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

interface IssueSelection {
  id: string
  title: string
  count: number
  filters: DiagnosisFilters
  description: string
  suggestion: string
}

export default function DiagnosisTab({ rootPath, reportName, datasetName, onSamples, legacyAnalysis, metrics }: Props) {
  const { t } = useLocale()
  const [selection, setSelection] = useState<IssueSelection | null>(null)
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
        <p>{t(`diagnosis.${state?.status === 'stale' ? 'stale' : state?.status === 'error' ? 'error' : 'missing'}`)}</p>
        {state?.status === 'missing' && legacyAnalysis && legacyAnalysis !== 'N/A' && (
          <Card title={t('diagnosis.summary')}><MarkdownRenderer content={legacyAnalysis} /></Card>
        )}
        <code className="block select-all overflow-x-auto whitespace-pre-wrap rounded bg-[var(--bg-deep)] p-3 text-sm">
          {command}
        </code>
      </div>
    )

  const report = state.report
  const c = report.coverage
  const translated = (group: string, key: string, fallback: string) => {
    const path = `diagnosis.${group}.${key}`
    return t(path) === path ? t(`diagnosis.${group}.${fallback}`) : t(path)
  }
  const issueTitle = (type: string) => translated('issue', type, 'other')
  const makeEntry = (id: string, title: string, count: number, filters: DiagnosisFilters): IssueSelection => ({
    id, title, count, filters,
    description: translated('issueDescription', id, 'other'),
    suggestion: translated('issueSuggestion', id, 'other'),
  })
  const issues = (state.issues ?? []).filter(issue => (issue.failed ?? 0) > 0 && issue.failure_type !== 'unexplained')
    .sort((a, b) => (b.failed ?? 0) - (a.failed ?? 0) || a.failure_type.localeCompare(b.failure_type))
    .map(issue => makeEntry(issue.failure_type, issueTitle(issue.failure_type), issue.failed ?? 0,
      { failure_type: issue.failure_type, outcome: 'failed' }))
  const pending = c.unexplained_failures > 0
    ? makeEntry('unexplained', t('diagnosis.issue.unexplained'), c.unexplained_failures,
      { failure_type: 'unexplained', outcome: 'failed' }) : null
  const unscored = c.excluded > 0
    ? makeEntry('unscored', t('diagnosis.unscored'), c.excluded, { outcome: 'unscored' }) : null
  const risk = state.risk_samples
    ? makeEntry('risk', t('diagnosis.riskTitle'), state.risk_samples, { outcome: 'risk' }) : null
  const passed = state.passed_observations
    ? makeEntry('passed', t('diagnosis.passedObservations'), state.passed_observations, { outcome: 'passed' }) : null
  const selected = selection ?? issues[0] ?? pending ?? unscored ?? risk ?? passed
  const tools = selected?.filters.failure_type
    ? Object.keys(state.issues?.find(issue => issue.failure_type === selected.filters.failure_type)?.tools ?? {}) : []
  const factTitle = (fact: Fact) => {
    const key = `diagnosis.fact.${fact.id}`
    return t(key) === key ? fact.title : t(key)
  }
  const openSamples = (filters: DiagnosisFilters, sample?: SampleKey, title = t('diagnosis.samplePanel')) => {
    setSelection(makeEntry('other', title, 0, sample ? { ...filters, subset_name: sample.subset, index: sample.index } : filters))
  }
  const issueButton = (entry: IssueSelection) => (
    <button
      key={entry.id}
      type="button"
      aria-pressed={selected?.id === entry.id}
      onClick={() => setSelection(entry)}
      className={`w-full rounded-md border-l-4 px-3 py-3 text-left transition-colors ${selected?.id === entry.id
        ? 'border-[var(--accent)] bg-[var(--accent-dim)]'
        : 'border-transparent hover:bg-[var(--bg-card2)]'}`}
    >
      <span className="flex items-start justify-between gap-3 text-sm font-semibold">
        <span>{entry.title}</span>
        <span className="shrink-0 font-mono text-[var(--text-muted)]">{t('diagnosis.issueCount', { count: entry.count })}</span>
      </span>
      <span className="mt-2 block text-xs leading-relaxed text-[var(--text-muted)]">{entry.description}</span>
    </button>
  )

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-[var(--border)] pb-4">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <p className="text-base font-semibold">{t(c.task_success_known ? 'diagnosis.overview' : 'diagnosis.summaryNoOutcome', {
            samples: c.samples, failed: c.task_failed, known: c.task_success_known,
          })}</p>
          {c.task_failed > 0 && <p className="text-sm text-[var(--text-muted)]">{t('diagnosis.clueSummary', {
            clues: Math.max(0, c.task_failed - c.unexplained_failures), pending: c.unexplained_failures,
          })}</p>}
        </div>
        {risk && <button type="button" className="flex items-center gap-2 text-sm text-[var(--yellow)]"
          aria-pressed={selected?.id === 'risk'} onClick={() => setSelection(risk)}>
          <AlertTriangle size={16} />{t('diagnosis.riskCount', { count: risk.count })}<ArrowRight size={14} />
        </button>}
        {(c.outcome_unknown > 0 || c.errored_without_record > 0 || c.unreadable > 0) && <p className="w-full text-xs text-[var(--text-muted)]">{[
          ['unknown', c.outcome_unknown], ['executionErrors', c.errored_without_record], ['unreadable', c.unreadable],
        ].filter(([, count]) => Number(count) > 0).map(([key, count]) => `${t(`diagnosis.${key}`)} ${count}`).join(' · ')}</p>}
      </div>
      {issues.length > 1 && <p className="text-xs text-[var(--text-muted)]">{t('diagnosis.overlappingIssues')}</p>}
      <div className="grid grid-cols-[250px_minmax(0,1fr)] items-start gap-5">
        <section aria-label={t('diagnosis.majorIssues')} className="min-w-0 space-y-2 border-r border-[var(--border)] pr-4">
          <h3 className="border-b border-[var(--border)] pb-3 text-base font-semibold">{t('diagnosis.majorIssues')}</h3>
          {issues.map(issueButton)}
          {pending && <div className="border-t border-[var(--border)] pt-2">{issueButton(pending)}</div>}
          {!issues.length && !pending && <p className="py-3 text-sm text-[var(--text-muted)]">
            {t(c.task_failed ? 'diagnosis.noIdentifiedIssues' : c.task_success_known ? 'diagnosis.noFailures' : 'diagnosis.noTaskOutcome')}
          </p>}
          {unscored && issueButton(unscored)}
          {passed && <button type="button" className="diagnosis-link mt-3 block text-left text-xs leading-relaxed"
            aria-pressed={selected?.id === 'passed'} onClick={() => setSelection(passed)}>
            {t('diagnosis.passedCount', { count: passed.count })}
          </button>}
          <a href="#diagnosis-analysis-record" className="diagnosis-link mt-6 flex items-center gap-2 border-t border-[var(--border)] pt-4 text-xs"
            onClick={() => { const record = document.getElementById('diagnosis-analysis-record'); if (record instanceof HTMLDetailsElement) record.open = true }}>
            <FileText size={14} />{t('diagnosis.analysisRecord')}
          </a>
        </section>
        <div className="min-w-0 space-y-4">
          {selected ? <>
            <header className="space-y-1">
              <h3 className="text-lg font-semibold">{selected.title}</h3>
              <p className="text-xs text-[var(--text-muted)]">{selected.count > 0 && `${t(selected.filters.outcome === 'failed' ? 'diagnosis.failedCount' : 'diagnosis.sampleCount', { count: selected.count })} · `}
                {t('diagnosis.ruleObservation')}</p>
              <p className="text-sm leading-relaxed">{selected.description}</p>
              <p className="border-l-2 border-[var(--accent)] pl-3 text-sm leading-relaxed">{selected.suggestion}</p>
            </header>
            <DiagnosisSamplePanel key={JSON.stringify(selected.filters)} rootPath={rootPath} reportName={reportName}
              datasetName={datasetName} filters={selected.filters} metrics={metrics} onSamples={onSamples} />
          </> : <p className="py-6 text-sm text-[var(--text-muted)]">{t(c.task_success_known ? 'diagnosis.noFailures' : 'diagnosis.noTaskOutcome')}</p>}
        </div>
      </div>
      {report.narrative && <section aria-label={t('diagnosis.modelSummary')} className="space-y-3 border-t border-[var(--border)] pt-4">
        <h3 className="text-base font-semibold">{t('diagnosis.modelSummary')}</h3>
        <p className="text-xs text-[var(--text-muted)]">{t('diagnosis.proposalNotice')}</p>
        <ul className="space-y-3 text-sm">{[...report.narrative.findings, ...report.narrative.recommendations].map((item, index) => <li key={index}>
          {item.text} {item.refs.map(reference => {
            const pattern = report.patterns.find(p => p.id === reference)
            const fact = report.facts.find(f => f.id === reference)
            return <button type="button" key={reference} className="diagnosis-link ml-1" onClick={() => {
              if (pattern) openSamples({ pattern_id: pattern.id }, undefined, issueTitle(pattern.failure_type))
              else if (fact?.samples[0]) openSamples({}, fact.samples[0], factTitle(fact))
              else {
                const element = document.getElementById(`detail-${reference}`)
                const record = document.getElementById('diagnosis-analysis-record')
                if (record instanceof HTMLDetailsElement) record.open = true
                element?.scrollIntoView({ block: 'center' })
              }
            }}>[{pattern ? `P${report.patterns.indexOf(pattern) + 1}` : fact ? factTitle(fact) : reference}]</button>
          })}
        </li>)}</ul>
      </section>}
      <details id="diagnosis-analysis-record" className="border-t border-[var(--border)] pt-3">
        <summary className="cursor-pointer text-sm text-[var(--text-muted)]">{t('diagnosis.analysisRecord')}</summary>
        <div className="mt-4 space-y-4 text-sm">
          <p className="text-xs text-[var(--text-muted)]">{t(report.level === 'facts' ? 'diagnosis.ruleAnalysis' : 'diagnosis.modelAnalysis')} · {t('diagnosis.generated')} {new Date(report.generated_at).toLocaleString()}</p>
          {report.level !== 'facts' && <div className="space-y-2">
            <p>{t('diagnosis.proposalCoverage', { ok: c.diagnosed_ok, selected: c.selected, total: c.diagnosis_population })}</p>
            {report.self_evaluation && <p>{t('diagnosis.selfEvaluation')}</p>}
          </div>}
          <p>{t('diagnosis.callCount', { count: report.cost.calls })}{report.cost.cached_calls > 0 && ` · ${t('diagnosis.cachedCalls', { count: report.cost.cached_calls })}`}
            {report.cost.calls > 0 && ` · ${t('diagnosis.tokenCount', { count: report.cost.input_tokens + report.cost.output_tokens })}`}
          </p>
          <p className="text-xs text-[var(--text-muted)]">{t('diagnosis.recordScope')}</p>
              {tools.length > 0 && <p className="text-xs text-[var(--text-muted)]">{t('diagnosis.relatedTools')}: <span className="font-mono">{tools.join(' / ')}</span></p>}

          <ul className="divide-y divide-[var(--border)]">{report.facts.map(fact => <li id={`detail-${fact.id}`} key={fact.id} className="space-y-2 py-3">
            <h4 className="font-medium">{factTitle(fact)}{fact.value !== null && ` · ${fact.unit === 'fraction' ? formatProportion(fact.value).primary : formatMetric(fact.value, null).primary}`}</h4>
            <FactData fact={fact} />
            {fact.samples.length > 0 && <div className="flex flex-wrap gap-x-3">{fact.samples.map(sample => <button type="button" key={`${sample.subset}:${sample.index}`} className="diagnosis-link text-xs"
              onClick={() => onSamples({}, sample)}>{sample.subset} · {t('diagnosis.sampleLink', { index: sample.index })}</button>)}</div>}
          </li>)}</ul>
          {Object.keys(c.skipped_facts).length > 0 && <details>
            <summary className="cursor-pointer text-xs text-[var(--text-muted)]">{t('diagnosis.skipped')} ({Object.keys(c.skipped_facts).length})</summary>
            <ul className="mt-2 space-y-1 text-xs text-[var(--text-muted)]">{Object.keys(c.skipped_facts).map(key => <li key={key}>{translated('unavailable', key, 'predictions')}</li>)}</ul>
          </details>}
        </div>
      </details>
    </div>
  )
}
