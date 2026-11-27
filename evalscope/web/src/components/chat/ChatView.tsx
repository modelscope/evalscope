import { scrollToContent } from './scrollToContent'
import { useState, useCallback, useMemo, useRef, useId, useEffect } from 'react'
import { AlertCircle, Flag, List, User, Star } from 'lucide-react'
import { useLocale } from '@/contexts/LocaleContext'
import type { PredictionRow, SampleDiagnosis } from '@/api/types'
import { EvidenceByMessage } from '@/contexts/EvidenceContext'
import SampleDiagnosisCard from '@/components/reports/SampleDiagnosisCard'
import { hasSystemPrompt, parseSystemUser, contentToText } from '@/domain/chat/messageText'
import { MessageRow, SystemPromptRow, HeaderPerfChip } from './MessageComponents'
import { StructuredMessages, TracedTimeline } from './AgentTraceView'
import { buildStepGroups } from '@/domain/trace/stepGroups'
import { CollapsibleJson, EvalResultPanel } from './EvalResultPanel'
import { JudgeReviewPanel } from './JudgeReviewPanel'
import { selectJudgeReview, scoreWithoutJudgeAttempts, isUsableStatus } from '@/domain/chat/judgeReview'
import DecisionPanel from './DecisionPanel'
import type { ReportMetric } from '@/domain/report/primaryMetrics'

type ChatPresentation = 'traced' | 'structured' | 'legacy'

function selectChatPresentation(prediction: Pick<PredictionRow, 'Messages' | 'AgentTrace'>): ChatPresentation {
  const hasMessages = Boolean(prediction.Messages?.length)
  const hasTrace = Boolean(prediction.AgentTrace?.events?.length)
  if (hasMessages && hasTrace) return 'traced'
  if (hasMessages) return 'structured'
  return 'legacy'
}

interface Props {
  prediction: PredictionRow
  threshold?: number
  highlightMsgId?: string
  metrics?: ReportMetric[]
  diagnosis?: SampleDiagnosis
  focusEvidence?: boolean
  focusStep?: number
  navigation?: 'sidebar' | 'toolbar'
}

/** Plain input/output rendering when no structured Messages or AgentTrace are available. */
function LegacyMessages({ prediction }: { prediction: PredictionRow }) {
  const isSystemMsg = hasSystemPrompt(prediction.Input)
  const { system, user } = isSystemMsg ? parseSystemUser(prediction.Input) : { system: '', user: prediction.Input }
  const headerPerf = prediction.PerfMetrics ? (
    <HeaderPerfChip
      latency={prediction.PerfMetrics.latency != null ? prediction.PerfMetrics.latency * 1000 : null}
      ttft={prediction.PerfMetrics.ttft}
      tpot={prediction.PerfMetrics.tpot}
      inTok={prediction.PerfMetrics.input_tokens}
      outTok={prediction.PerfMetrics.output_tokens}
    />
  ) : undefined
  return (
    <div className="flex flex-col gap-2">
      {system && <SystemPromptRow content={system} />}
      {/* When a system prompt was extracted but there is no user turn, `user` is
          empty; rendering the raw input here would repeat the system text. */}
      {(user || !system) && <MessageRow role="user" content={user || prediction.Input} />}
      {prediction.Generated && <MessageRow role="assistant" content={prediction.Generated} headerExtra={headerPerf} />}
    </div>
  )
}

export default function ChatView({
  prediction,
  threshold = 0.99,
  highlightMsgId,
  metrics,
  diagnosis,
  focusEvidence = false,
  focusStep,
  navigation = 'sidebar',
}: Props) {
  const { t } = useLocale()
  const showPred =
    prediction.Pred &&
    prediction.Pred !== '*Same as Generated*' &&
    prediction.Generated &&
    prediction.Pred.trim() !== prediction.Generated.trim()

  const messages = prediction.Messages
  const agentTrace = prediction.AgentTrace
  const presentation = selectChatPresentation(prediction)
  const readerRef = useRef<HTMLDivElement>(null)
  const prefix = useId()
  const hasProposal = diagnosis?.llm_status === 'ok' && Boolean(diagnosis.llm)
  const openProposal = () => {
    const details = readerRef.current?.querySelector<HTMLDetailsElement>('[data-model-proposal]')
    if (!details) return
    details.open = true
    const firstEvidence = details.querySelector('details')
    if (firstEvidence) firstEvidence.open = true
    scrollToContent(details)
  }
  const findingStep =
    agentTrace?.events.find(
      (e) => e.type === 'tool_call' && diagnosis?.findings?.some((f) => f.call_ids?.includes(String(e.payload.id))),
    )?.step ?? null
  const rootStep = focusStep ?? (
    diagnosis?.llm_status === 'ok' ? (diagnosis.llm?.root_cause_step ?? (focusEvidence ? findingStep : null)) : focusEvidence ? findingStep : null
  )
  useEffect(() => {
    if (!focusEvidence || presentation === 'traced') return
    const evidence = readerRef.current?.querySelector<HTMLElement>('[data-evidence-source]')
    scrollToContent(evidence ?? document.getElementById(`${prefix}-scoring`))
  }, [focusEvidence, presentation, diagnosis?.llm_status, prefix])
  const [pickedStep, setHighlightedStep] = useState<number | null | undefined>(undefined)
  const highlightedStep = pickedStep === undefined ? rootStep : pickedStep
  const activeMessageId = pickedStep === undefined ? highlightMsgId : undefined
  const evidenceByMessage = useMemo(() => {
    const result: Record<string, string[]> = {}
    if (diagnosis?.llm_status !== 'ok') return result
    for (const ref of diagnosis.llm?.evidence ?? []) {
      const source = diagnosis.sources?.find((s) => s.id === ref.source_id)
      const id = source?.message_index != null ? messages?.[source.message_index]?.id : null
      if (id && ref.source_id.startsWith('m:') && source?.message_index != null) {
        const original = contentToText(messages?.[source.message_index]?.content ?? '')
        if (original.split(ref.quote).length === 2) (result[id] ??= []).push(ref.quote)
      }
    }
    return result
  }, [diagnosis, messages])

  const judgeReview = useMemo(() => selectJudgeReview(prediction.Score), [prediction.Score])
  // The raw judge replies are rendered by the review panel below; drop them from the Score Detail
  // JSON so a rubric sample's 100 KB of attempts is not shown twice.
  const scoreForDetail = useMemo(() => scoreWithoutJudgeAttempts(prediction.Score), [prediction.Score])

  const stepGroups = useMemo(() => {
    if (presentation !== 'traced' || !messages || !agentTrace) return null
    return buildStepGroups(messages, agentTrace)
  }, [presentation, messages, agentTrace])

  const handleStepClick = useCallback(
    (step: number) => {
      setHighlightedStep(highlightedStep === step ? null : step)
    },
    [highlightedStep],
  )

  const goTo = (section: 'question' | 'answer' | 'scoring') => {
    if (section === 'scoring') scrollToContent(document.getElementById(`${prefix}-scoring`))
    else {
      const nodes = readerRef.current?.querySelectorAll(
        `[data-message-role="${section === 'question' ? 'user' : 'assistant'}"]`,
      )
      const node =
        section === 'question'
          ? (nodes?.[0] ?? readerRef.current?.querySelector('[data-message-role="environment"]'))
          : nodes?.[nodes.length - 1]
      scrollToContent(node as HTMLElement | undefined)
    }
  }
  const steps = stepGroups?.filter((g) => g.step >= 0) ?? []
  const answerStep = steps.filter((g) => g.assistant && contentToText(g.assistant.content).trim()).at(-1)?.step

  const toolbar = navigation === 'toolbar'
  const navButtonClass = `flex items-center gap-2 rounded p-2 text-left hover:bg-[var(--accent-dim)] ${toolbar ? 'shrink-0' : 'w-full'}`

  return (
    <EvidenceByMessage.Provider value={evidenceByMessage}>
      <div data-sample-scroll
        className={`grid max-h-[var(--sample-reader-height,calc(100vh-360px))] ${toolbar ? 'grid-cols-1 gap-3' : 'grid-cols-[150px_minmax(0,1fr)] gap-4'} items-start overflow-auto ${toolbar ? 'pb-2' : 'py-2'}`} ref={readerRef}>
        <nav
          aria-label={t('trace.readingOutline')}
          data-reader-toolbar={toolbar || undefined}
          className={toolbar
            ? 'sticky top-0 z-10 flex items-center gap-2 border-b border-[var(--border)] bg-[var(--bg-card)] py-2 text-xs'
            : 'sticky top-0 max-h-[var(--sample-reader-height,calc(100vh-380px))] space-y-1 overflow-auto border-r border-[var(--border)] pr-3 text-xs'}
        >
          {diagnosis && (diagnosis.task_success !== null || !isUsableStatus(diagnosis.score_status)) && (
            <div
              className={`text-sm font-medium ${diagnosis.task_success === false || !isUsableStatus(diagnosis.score_status) ? 'text-[var(--danger)]' : 'text-[var(--success)]'}`}
            >
              {t(
                !isUsableStatus(diagnosis.score_status)
                  ? 'trace.unscored'
                  : diagnosis.task_success
                    ? 'trace.passed'
                    : 'trace.failed',
              )}
            </div>
          )}
          <button
            onClick={() => goTo('question')}
            className={navButtonClass}
          >
            <User size={14} />
            {t('trace.question')}
          </button>
          {toolbar && steps.length > 0 && <select
            aria-label={t('trace.locateStep')}
            className="ml-auto max-w-64 rounded border border-[var(--border)] bg-[var(--bg-deep)] px-3 py-2 text-sm"
            value={highlightedStep == null ? '' : String(highlightedStep)}
            onChange={event => {
              const step = Number(event.target.value)
              setHighlightedStep(step)
              scrollToContent(document.getElementById(`${prefix}-step-${step}`))
            }}
          >
            <option value="" disabled>{t('trace.locateStep')}</option>
            {steps.map(group => <option key={group.step} value={group.step}>
              {t('trace.step')} {group.step} · {group.assistant?.tool_calls?.map(call => call.function).join(', ')
                || group.traceEvents.filter(event => event.type === 'tool_call').map(event => String(event.payload.name ?? '')).join(', ')
                || t('trace.assistant')}
            </option>)}
          </select>}
          {!toolbar && steps.map((g) => {
            const calls =
              g.assistant?.tool_calls?.map((c) => c.function) ??
              g.traceEvents.filter((e) => e.type === 'tool_call').map((e) => String(e.payload.name ?? ''))
            const flagged = diagnosis?.findings?.some((f) =>
              f.call_ids?.some(
                (id) =>
                  g.assistant?.tool_calls?.some((c) => c.id === id) ||
                  g.traceEvents.some((e) => e.type === 'tool_call' && e.payload.id === id),
              ),
            )
            return (
              <button
                key={g.step}
                onClick={() => {
                  setHighlightedStep(g.step)
                  scrollToContent(document.getElementById(`${prefix}-step-${g.step}`))
                }}
                aria-current={highlightedStep === g.step ? 'step' : undefined}
                className={`block w-full rounded p-2 text-left hover:bg-[var(--accent-dim)] ${highlightedStep === g.step ? 'bg-[var(--accent-dim)] text-[var(--accent)]' : 'text-[var(--text-muted)]'}`}
              >
                <span className="flex items-center gap-2">
                  {flagged ? <AlertCircle size={13} /> : <List size={13} />}
                  {t('trace.step')} {g.step}
                </span>
                <span className="mt-1 block truncate pl-5 font-mono" title={calls.join(', ')}>
                  {calls.join(', ') || t('trace.assistant')}
                </span>
              </button>
            )
          })}
          <button
            onClick={() => {
              if (answerStep !== undefined) {
                setHighlightedStep(answerStep)
                scrollToContent(document.getElementById(`${prefix}-step-${answerStep}`))
              } else goTo('answer')
            }}
            className={navButtonClass}
          >
            <Flag size={14} />
            {t('trace.answer')}
          </button>
          <button
            onClick={() => goTo('scoring')}
            className={navButtonClass}
          >
            <Star size={14} />
            {t('trace.scoring')}
          </button>
          {toolbar && hasProposal && (
            <button onClick={openProposal} className={`${navButtonClass} diagnosis-link`}>
              {t('diagnosis.reviewModelProposal')}
            </button>
          )}
        </nav>
        <div className="min-w-0 space-y-4">
          {presentation === 'traced' && stepGroups && messages && agentTrace ? (
            <TracedTimeline
              groups={stepGroups}
              anchorPrefix={prefix}
              messages={messages}
              trace={agentTrace}
              highlightStep={highlightedStep}
              highlightId={activeMessageId}
              onStepClick={handleStepClick}
              findings={diagnosis?.findings}
            />
          ) : presentation === 'structured' && messages ? (
            <div className="flex flex-col gap-2">
              <StructuredMessages messages={messages} highlightId={activeMessageId} />
            </div>
          ) : (
            <LegacyMessages prediction={prediction} />
          )}

          {agentTrace && <CollapsibleJson label={t('trace.rawTrace')} value={agentTrace} maxHeight={320} />}

          <div id={`${prefix}-scoring`} className="scroll-mt-24 border-t border-[var(--border)]" />

          <DecisionPanel metadata={prediction.Score.metadata} />

          <EvalResultPanel
            pred={prediction.Pred}
            gold={prediction.Gold}
            nScore={prediction.NScore}
            status={prediction.Status ?? undefined}
            score={scoreForDetail}
            metadata={prediction.Metadata}
            threshold={threshold}
            showPred={!!showPred}
            metrics={metrics}
            compact={toolbar}
            recordedError={diagnosis?.recorded_error}
          />

          {diagnosis && <SampleDiagnosisCard diagnosis={diagnosis} showFindings={!toolbar || presentation !== 'traced'} />}

          {judgeReview && <JudgeReviewPanel review={judgeReview} />}
        </div>
      </div>
    </EvidenceByMessage.Provider>
  )
}
