import { useState, useCallback, useMemo } from 'react'
import type { PredictionRow } from '@/api/types'
import { hasSystemPrompt, parseSystemUser } from '@/domain/chat/messageText'
import { MessageRow, SystemPromptRow, HeaderPerfChip } from './MessageComponents'
import { StructuredMessages, TracedTimeline } from './AgentTraceView'
import { buildStepGroups } from '@/domain/trace/stepGroups'
import { EvalResultPanel } from './EvalResultPanel'
import { JudgeReviewPanel } from './JudgeReviewPanel'
import { selectJudgeReview, scoreWithoutJudgeAttempts } from '@/domain/chat/judgeReview'

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

export default function ChatView({ prediction, threshold = 0.99, highlightMsgId }: Props) {
  const showPred =
    prediction.Pred &&
    prediction.Pred !== '*Same as Generated*' &&
    prediction.Generated &&
    prediction.Pred.trim() !== prediction.Generated.trim()

  const messages = prediction.Messages
  const agentTrace = prediction.AgentTrace
  const presentation = selectChatPresentation(prediction)
  const [highlightedStep, setHighlightedStep] = useState<number | null>(null)

  const judgeReview = useMemo(() => selectJudgeReview(prediction.Score), [prediction.Score])
  // The raw judge replies are rendered by the review panel below; drop them from the Score Detail
  // JSON so a rubric sample's 100 KB of attempts is not shown twice.
  const scoreForDetail = useMemo(() => scoreWithoutJudgeAttempts(prediction.Score), [prediction.Score])

  const stepGroups = useMemo(() => {
    if (presentation !== 'traced' || !messages || !agentTrace) return null
    return buildStepGroups(messages, agentTrace)
  }, [presentation, messages, agentTrace])

  const handleStepClick = useCallback((step: number) => {
    setHighlightedStep(prev => (prev === step ? null : step))
  }, [])

  return (
    <div className="flex flex-col gap-4 py-2">
      {presentation === 'traced' && stepGroups && messages && agentTrace ? (
        <TracedTimeline
          groups={stepGroups}
          messages={messages}
          trace={agentTrace}
          highlightStep={highlightedStep}
          highlightId={highlightMsgId}
          onStepClick={handleStepClick}
        />
      ) : presentation === 'structured' && messages ? (
        <div className="flex flex-col gap-2">
          <StructuredMessages messages={messages} highlightId={highlightMsgId} />
        </div>
      ) : (
        <LegacyMessages prediction={prediction} />
      )}

      <div className="border-t border-[var(--border)]" />

      <EvalResultPanel
        pred={prediction.Pred}
        gold={prediction.Gold}
        nScore={prediction.NScore}
        status={prediction.Status ?? undefined}
        score={scoreForDetail}
        metadata={prediction.Metadata}
        threshold={threshold}
        showPred={!!showPred}
      />

      {judgeReview && <JudgeReviewPanel review={judgeReview} />}
    </div>
  )
}
