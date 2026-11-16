import { useState, useCallback, useMemo } from 'react'
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
import { selectJudgeReview, scoreWithoutJudgeAttempts } from '@/domain/chat/judgeReview'
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
}

/** Plain input/output rendering when no structured Messages or AgentTrace are available. */
function LegacyMessages({ prediction }: { prediction: PredictionRow }) {
  const isSystemMsg = hasSystemPrompt(prediction.Input)
  const { system, user } = isSystemMsg
    ? parseSystemUser(prediction.Input)
    : { system: '', user: prediction.Input }
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
      {prediction.Generated && (
        <MessageRow role="assistant" content={prediction.Generated} headerExtra={headerPerf} />
      )}
    </div>
  )
}

export default function ChatView({ prediction, threshold = 0.99, highlightMsgId, metrics, diagnosis }: Props) {
  const { t } = useLocale()
  const showPred =
    prediction.Pred &&
    prediction.Pred !== '*Same as Generated*' &&
    prediction.Generated &&
    prediction.Pred.trim() !== prediction.Generated.trim()

  const messages = prediction.Messages
  const agentTrace = prediction.AgentTrace
  const presentation = selectChatPresentation(prediction)
  const rootStep = diagnosis?.llm_status === 'ok' ? diagnosis.llm?.root_cause_step ?? null : null
  const [pickedStep, setHighlightedStep] = useState<number | null | undefined>(undefined)
  const highlightedStep = pickedStep === undefined ? rootStep : pickedStep
  const evidenceByMessage = useMemo(() => {
    const result: Record<string, string[]> = {}
    if (diagnosis?.llm_status !== 'ok') return result
    for (const ref of diagnosis.llm?.evidence ?? []) {
      const source = diagnosis.sources?.find(s => s.id === ref.source_id)
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

  const handleStepClick = useCallback((step: number) => {
    setHighlightedStep(highlightedStep === step ? null : step)
  }, [highlightedStep])

  return (
    <EvidenceByMessage.Provider value={evidenceByMessage}>
    <div className="flex flex-col gap-4 py-2">
      {presentation === 'traced' && stepGroups && messages && agentTrace ? (
        <TracedTimeline
          groups={stepGroups}
          messages={messages}
          trace={agentTrace}
          highlightStep={highlightedStep}
          highlightId={highlightMsgId}
          onStepClick={handleStepClick}
          findings={diagnosis?.findings}
        />
      ) : presentation === 'structured' && messages ? (
        <div className="flex flex-col gap-2">
          <StructuredMessages messages={messages} highlightId={highlightMsgId} />
        </div>
      ) : (
        <LegacyMessages prediction={prediction} />
      )}

      {agentTrace && <CollapsibleJson label={t('trace.rawTrace')} value={agentTrace} maxHeight={320} />}

      <div className="border-t border-[var(--border)]" />

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
      />

      {diagnosis && <SampleDiagnosisCard diagnosis={diagnosis} />}

      {judgeReview && <JudgeReviewPanel review={judgeReview} />}
    </div>
    </EvidenceByMessage.Provider>
  )
}
