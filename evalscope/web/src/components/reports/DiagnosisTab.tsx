import { useState, type ReactNode } from 'react'
import { AlertTriangle, ArrowRight, ChevronDown, FileText } from 'lucide-react'
import DiagnosisSamplePanel from './DiagnosisSamplePanel'
import DiagnosisActions from './DiagnosisActions'
import type { ReportMetric } from '@/domain/report/primaryMetrics'
import { formatMetric, formatProportion } from '@/domain/metric'
import { getDiagnosis, type DiagnosisFilters } from '@/api/reports'
import type { DiagnosisReport, Fact, SampleKey } from '@/api/types'
import { parseReportRef } from '@/domain/report/reportRef'
import { useAsyncResource } from '@/hooks/useAsyncResource'
import { useLocale } from '@/contexts/LocaleContext'
import Card from '@/components/ui/Card'
import Badge from '@/components/ui/Badge'
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
  location: { filters: DiagnosisFilters; sample?: SampleKey; step?: number; position?: number }
  onLocate: (filters: DiagnosisFilters, sample?: SampleKey, step?: number, position?: number) => void
  onSamples: Props['onLocate']
}

function FactData({ fact }: { fact: Fact }) {
  const { t } = useLocale()
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
  return null
}

function AnalysisInformation({ report }: { report: DiagnosisReport }) {
  const { t } = useLocale()
  const { coverage, cost } = report
  const modelAnalysis = report.level !== 'facts'
  const mode = t(`diagnosis.analysisMode.${report.level}`)
  const tokenFormat = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 2 })
  const narrativeItems = (report.narrative?.findings.length ?? 0) + (report.narrative?.recommendations.length ?? 0)
  const summaryStatus = report.narrative_status === 'ok' && !narrativeItems ? 'no_findings' : report.narrative_status
  const summaryFailed = ['parse_error', 'transport_error', 'invalid_references'].includes(summaryStatus)
  const tokenUsage = cost.calls > 0
    ? cost.usage_missing_calls > 0 ? t('diagnosis.usageIncomplete')
      : t('diagnosis.tokenBreakdown', { input: tokenFormat.format(cost.input_tokens), output: tokenFormat.format(cost.output_tokens) })
    : t('diagnosis.noNewTokenUsage')
  const fields: { label: string; value: ReactNode; title?: string }[] = [
    { label: t('diagnosis.statisticsScope'), value: t('diagnosis.sampleCount', { count: coverage.samples }) },
    ...(modelAnalysis ? [
      { label: t('diagnosis.selectedScope'), value: t(`diagnosis.reviewScope.${coverage.selection_scope ?? 'unknown'}`) },
      { label: t('diagnosis.analysisModel'), value: report.diagnosis_model ?? '—' },
      { label: t('diagnosis.samplesReviewed'), value: coverage.diagnosis_population > 0
        ? `${coverage.selected} / ${coverage.diagnosis_population}` : t('diagnosis.noEligibleSamples') },
      { label: t('diagnosis.citationCheck'), value: coverage.citations_total > 0
        ? `${coverage.citations_valid} / ${coverage.citations_total}` : t('diagnosis.noCitations') },
      { label: t('diagnosis.hypotheses'), value: t('diagnosis.sampleCount', { count: coverage.diagnosed_ok }) },
      { label: t('diagnosis.newRequests'), value: String(cost.calls) },
      { label: t('diagnosis.cacheReuse'), value: String(cost.cached_calls) },
      { label: t('diagnosis.tokenUsage'), value: tokenUsage,
        title: cost.usage_missing_calls ? undefined : `${t('diagnosis.inputTokens')}: ${cost.input_tokens.toLocaleString()} · ${t('diagnosis.outputTokens')}: ${cost.output_tokens.toLocaleString()}` },
    ] : [{ label: t('diagnosis.modelCalls'), value: t('diagnosis.noModelCalls') }]),
    ...(report.level === 'diagnose' ? [{ label: t('diagnosis.summaryStatus'), value:
      <Badge variant={summaryFailed || summaryStatus === 'partial_references' ? 'warning' : 'default'}>
        {t(`diagnosis.summaryState.${summaryStatus}`)}
      </Badge> }] : []),
    { label: t('diagnosis.generated'), value: new Date(report.generated_at).toLocaleString() },
  ]
  const statuses = Object.entries(coverage.diagnosis_statuses).filter(([status, count]) => status !== 'ok' && count > 0)
  return (
    <details id="diagnosis-analysis-info" className="group rounded-lg border border-[var(--border)] bg-[var(--bg-card)]">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-4 py-3 text-sm [&::-webkit-details-marker]:hidden">
        <span className="flex items-center gap-2 font-medium"><FileText size={15} />{t('diagnosis.analysisInformation')}</span>
        <span className="flex items-center gap-3"><Badge>{mode}</Badge>{report.level === 'diagnose' && <span className={summaryFailed ? 'text-[var(--warning-text)]' : 'text-[var(--text-muted)]'}>{t(`diagnosis.summaryState.${summaryStatus}`)}</span>}<ChevronDown size={15} className="transition-transform group-open:rotate-180" /></span>
      </summary>
      <div className="space-y-4 border-t border-[var(--border)] p-4">
        <dl className="grid grid-cols-3 gap-x-8 gap-y-4">
          {fields.map(field => <div key={field.label} className="min-w-0">
            <dt className="text-xs text-[var(--text-muted)]">{field.label}</dt>
            <dd title={field.title} className="mt-1 break-words text-sm font-medium tabular-nums">{field.value}</dd>
          </div>)}
        </dl>
        {modelAnalysis && <p className="text-xs leading-relaxed text-[var(--text-muted)]">{t('diagnosis.reviewScopeNotice')}</p>}
        {modelAnalysis && coverage.selection_scope === 'failed' && coverage.outcome_unknown > 0 && <p className="text-xs leading-relaxed text-[var(--text-muted)]">{t('diagnosis.unknownOutcomeScopeNotice')}</p>}
        {statuses.length > 0 && modelAnalysis && <p className="text-xs text-[var(--text-muted)]">{statuses.map(([status, count]) => `${t(`diagnosis.llmStatus.${status}`)} ${count}`).join(' · ')}</p>}
        {report.self_evaluation && modelAnalysis && <p className="text-xs text-[var(--warning-text)]">{t('diagnosis.selfEvaluation')}</p>}
      </div>
    </details>
  )
}

interface IssueSelection {
  id: string
  title: string
  count: number
  filters: DiagnosisFilters
  description: string
  suggestion: string
}

export default function DiagnosisTab({ rootPath, reportName, datasetName, location, onLocate, onSamples, legacyAnalysis, metrics }: Props) {
  const { t } = useLocale()
  const [factReference, setFactReference] = useState<string | null>(null)
  const resource = useAsyncResource(
    (signal) => getDiagnosis(rootPath, reportName, datasetName, signal),
    [rootPath, reportName, datasetName],
    { enabled: Boolean(datasetName), fallbackMessage: t('common.loadError') },
  )
  if (resource.loading) return <Skeleton lines={6} />
  if (resource.error) return <ErrorAlert>{resource.error}</ErrorAlert>
  const state = resource.data
  const ready = state?.status === 'ready' && Boolean(state.report)
  const actions = <DiagnosisActions rootPath={rootPath} reportName={reportName} datasetName={datasetName}
    ready={ready} onFinished={resource.setData} />
  const ref = parseReportRef(reportName)
  const quote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`
  const command = `evalscope analyze --run-dir ${quote(`${rootPath}/${ref.runId}`)} --model ${quote(ref.modelId)} --benchmark ${quote(datasetName)} --level facts`
  if (!state || state.status !== 'ready' || !state.report)
    return (
      <div className="space-y-3" role="status">
        <p>{t(`diagnosis.${state?.status === 'stale' ? 'stale' : state?.status === 'error' ? 'error' : 'missing'}`)}</p>
        {actions}
        <details className="text-sm">
          <summary className="diagnosis-link w-fit cursor-pointer">{t('diagnosis.commandLine')}</summary>
          <code className="mt-2 block select-all overflow-x-auto whitespace-pre-wrap rounded bg-[var(--bg-deep)] p-3">
            {command}
          </code>
        </details>
        {state?.status === 'missing' && legacyAnalysis && legacyAnalysis !== 'N/A' && (
          <details className="text-sm">
            <summary className="diagnosis-link w-fit cursor-pointer">{t('diagnosis.summary')}</summary>
            <Card><MarkdownRenderer content={legacyAnalysis} /></Card>
          </details>
        )}
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
  const issues = report.patterns.filter(issue => (issue.failed ?? 0) > 0 && issue.failure_type !== 'unexplained')
    .sort((a, b) => (b.failed ?? 0) - (a.failed ?? 0) || a.failure_type.localeCompare(b.failure_type))
    .map(issue => makeEntry(issue.failure_type, issueTitle(issue.failure_type), issue.failed ?? 0,
      { failure_type: issue.failure_type, outcome: 'failed' }))
  const pending = c.unexplained_failures > 0
    ? makeEntry('unexplained', t('diagnosis.issue.unexplained'), c.unexplained_failures,
      { failure_type: 'unexplained', outcome: 'failed' }) : null
  const unscored = c.excluded > 0
    ? makeEntry('unscored', t('diagnosis.unscored'), c.excluded, { outcome: 'unscored' }) : null
  const risk = c.risk_samples
    ? makeEntry('risk', t('diagnosis.riskTitle'), c.risk_samples, { outcome: 'risk' }) : null
  const selectedGroup = [...issues, pending, unscored, risk].find(entry =>
    entry && entry.filters.failure_type === location.filters.failure_type && entry.filters.outcome === location.filters.outcome)
  const selected: Omit<IssueSelection, 'count'> | null = location.sample && (location.position === undefined || !selectedGroup)
    ? { id: 'sample', filters: { subset_name: location.sample.subset, index: location.sample.index },
      title: '', description: '', suggestion: '' }
    : selectedGroup ?? issues[0] ?? pending ?? unscored ?? risk
  const focusStep = location.step
  const position = selectedGroup === selected ? location.position ?? 0 : 0
  const tools = selected?.filters.failure_type
    ? Object.keys(report.patterns.find(issue => issue.failure_type === selected.filters.failure_type)?.tools ?? {}) : []
  const scoreFact = c.outcome_unknown > 0 ? report.facts.find(fact => fact.id === 'F-scores') : undefined
  const referencedFact = report.facts.find(fact => fact.id === factReference)
  const narrativeGroups = report.narrative ? [
    { key: 'summaryFindings', items: report.narrative.findings },
    { key: 'summaryRecommendations', items: report.narrative.recommendations },
  ].filter(group => group.items.length) : []
  const hasNarrative = report.level === 'diagnose' && ['ok', 'partial_references'].includes(report.narrative_status)
    && Boolean(report.narrative && (report.narrative.findings.length || report.narrative.recommendations.length))
  const factTitle = (fact: Fact) => {
    const key = `diagnosis.fact.${fact.id}`
    return t(key) === key ? fact.title : t(key)
  }
  const openSample = (sample: SampleKey, step?: number) => {
    onLocate({}, sample, step)
    document.getElementById('diagnosis-issues')?.scrollIntoView?.({ block: 'start', behavior: 'smooth' })
  }
  const issueButton = (entry: IssueSelection) => (
    <button
      key={entry.id}
      type="button"
      aria-pressed={selected?.id === entry.id}
      onClick={() => onLocate(entry.filters)}
      className={`w-full rounded-md border-l-4 px-3 py-3 text-left transition-colors ${selected?.id === entry.id
        ? 'border-[var(--accent)] bg-[var(--accent-dim)]'
        : 'border-transparent hover:bg-[var(--bg-card2)]'}`}
    >
      <span className="flex items-start justify-between gap-3 text-sm font-semibold">
        <span>{entry.title}</span>
        <span className="shrink-0 font-mono text-[var(--text-muted)]">{t('diagnosis.issueCount', { count: entry.count })}</span>
      </span>
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
          aria-pressed={selected?.id === 'risk'} onClick={() => onLocate(risk.filters)}>
          <AlertTriangle size={16} />{t('diagnosis.riskCount', { count: risk.count })}<ArrowRight size={14} />
        </button>}
        {(c.outcome_unknown > 0 || c.errored_without_record > 0 || c.unreadable > 0) && <p className="w-full text-xs text-[var(--text-muted)]">{[
          ['unknown', c.outcome_unknown], ['executionErrors', c.errored_without_record], ['unreadable', c.unreadable],
        ].filter(([, count]) => Number(count) > 0).map(([key, count]) => `${t(`diagnosis.${key}`)} ${count}`).join(' · ')}</p>}
      </div>
      {actions}
      {issues.length > 1 && <p className="text-xs text-[var(--text-muted)]">{t('diagnosis.overlappingIssues')}</p>}
      <div id="diagnosis-issues" className="grid scroll-mt-20 grid-cols-[220px_minmax(0,1fr)] items-start gap-5">
        <section aria-label={t('diagnosis.majorIssues')} className="min-w-0 space-y-2 border-r border-[var(--border)] pr-4">
          <h3 className="border-b border-[var(--border)] pb-3 text-base font-semibold">{t('diagnosis.majorIssues')}</h3>
          {issues.map(issueButton)}
          {pending && <div className="border-t border-[var(--border)] pt-2">
            {issueButton(pending)}
            {c.proposals_without_rule_clues > 0 && <p className="px-3 pt-1 text-xs text-[var(--text-muted)]">
              {t('diagnosis.proposalsWithoutRuleClues', { count: c.proposals_without_rule_clues })}
            </p>}
          </div>}
          {!issues.length && !pending && <p className="py-3 text-sm text-[var(--text-muted)]">
            {t(c.task_failed ? 'diagnosis.noIdentifiedIssues' : c.task_success_known ? 'diagnosis.noFailures' : 'diagnosis.noTaskOutcome')}
          </p>}
          {unscored && issueButton(unscored)}
          {scoreFact && <a href="#diagnosis-scores" className="diagnosis-link inline-block px-3 py-2 text-sm"
            onClick={() => { const details = document.getElementById('diagnosis-scores'); if (details instanceof HTMLDetailsElement) details.open = true }}>
            {t('diagnosis.scoreEvidence')}
          </a>}
          <a href="#diagnosis-analysis-info" className="diagnosis-link mt-6 flex items-center gap-2 border-t border-[var(--border)] pt-4 text-xs"
            onClick={() => { const record = document.getElementById('diagnosis-analysis-info'); if (record instanceof HTMLDetailsElement) record.open = true }}>
            <FileText size={14} />{t('diagnosis.analysisInformation')}
          </a>
        </section>
        <div className="min-w-0 space-y-4">
          {selected ? <>
            {selected.filters.index === undefined && <header className="space-y-1">
              <h3 className="text-lg font-semibold">{selected.title}</h3>
              {tools.length > 0 && <p className="text-xs text-[var(--text-muted)]">{t('diagnosis.relatedTools')}: <span className="font-mono">{tools.join(' / ')}</span></p>}
              <p className="text-sm leading-relaxed">{selected.description}</p>
              {selected.suggestion && <p className="text-sm leading-relaxed text-[var(--text-muted)]">{selected.suggestion}</p>}
            </header>}
            <DiagnosisSamplePanel key={report.generated_at} rootPath={rootPath} reportName={reportName}
              datasetName={datasetName} filters={selected.filters} metrics={metrics} focusStep={focusStep} onSamples={onSamples}
              position={position} expectedSample={location.sample} onPositionChange={next => onLocate(selected.filters, undefined, undefined, next)} />
          </> : <p className="py-6 text-sm text-[var(--text-muted)]">{t(c.task_success_known ? 'diagnosis.noFailures' : 'diagnosis.noTaskOutcome')}</p>}
        </div>
      </div>
      {hasNarrative && report.narrative && <section aria-label={t('diagnosis.modelSummary')} className="space-y-3 border-t border-[var(--border)] pt-4">
        <h3 className="text-base font-semibold">{t('diagnosis.modelSummary')}</h3>
        <p className="text-xs text-[var(--text-muted)]">{t('diagnosis.proposalNotice')}</p>
        <div className={narrativeGroups.length > 1 ? 'grid grid-cols-2 gap-6' : 'space-y-3'}>{narrativeGroups.map(group => <section key={group.key} className="space-y-2"><h4 className="text-sm font-medium">{t(`diagnosis.${group.key}`)}</h4><ul className="space-y-3 text-sm">{group.items.map((item, index) => <li key={index}>
          {item.text} {item.refs.map(reference => {
            const sample = report.narrative_samples[reference]
            const fact = report.facts.find(f => f.id === reference)
            if (!sample && !fact) return null
            const label = sample
              ? `${sample.sample.subset} · ${t('diagnosis.sampleLink', { index: sample.sample.index })}${sample.step != null ? ` · ${t('trace.step')} ${sample.step}` : ''}`
              : fact!.id === 'F-task-failure' && fact!.numerator !== null
                ? t('diagnosis.failedSampleReference', { count: fact!.numerator }) : factTitle(fact!)
            return <button type="button" key={reference} className="diagnosis-link ml-1" onClick={() => {
              if (sample) {
                setFactReference(null)
                openSample(sample.sample, sample.step ?? undefined)
              } else if (fact) setFactReference(fact.id)
            }}>[{label}]</button>
          })}
        </li>)}</ul></section>)}</div>
        {referencedFact && <aside aria-label={t('diagnosis.referenceBasis')} className="space-y-2 rounded-md border border-[var(--border)] bg-[var(--bg-card)] p-4">
          <h4 className="text-sm font-medium">{factTitle(referencedFact)}{referencedFact.value !== null && ` · ${referencedFact.unit === 'fraction' ? formatProportion(referencedFact.value).primary : formatMetric(referencedFact.value, null).primary}`}</h4>
          <p className="text-xs text-[var(--text-muted)]">{t(`diagnosis.description.${referencedFact.id}`) === `diagnosis.description.${referencedFact.id}` ? referencedFact.definition : t(`diagnosis.description.${referencedFact.id}`)}</p>
          {referencedFact.denominator !== null && <p className="text-xs">{referencedFact.numerator} / {referencedFact.denominator}</p>}
          <FactData fact={referencedFact} />
          {referencedFact.samples.length > 0 && <div className="flex flex-wrap gap-3">{referencedFact.samples.map(sample => <button type="button" key={`${sample.subset}:${sample.index}`} className="diagnosis-link text-xs" onClick={() => openSample(sample)}>{sample.subset} · {t('diagnosis.sampleLink', { index: sample.index })}</button>)}</div>}
        </aside>}
      </section>}
      {scoreFact && <details id="diagnosis-scores" className="scroll-mt-20 space-y-2 rounded-md border border-[var(--border)] p-3">
        <summary className="cursor-pointer text-sm font-semibold">{factTitle(scoreFact)}</summary>
        {scoreFact.samples.length > 0 && <div className="flex flex-wrap gap-3">
          {scoreFact.samples.map(sample => <button type="button" key={`${sample.subset}:${sample.index}`} className="diagnosis-link text-xs"
            onClick={() => openSample(sample)}>{sample.subset} · {t('diagnosis.sampleLink', { index: sample.index })}</button>)}
        </div>}
        <FactData fact={scoreFact} />
      </details>}
      <AnalysisInformation report={report} />
    </div>
  )
}
