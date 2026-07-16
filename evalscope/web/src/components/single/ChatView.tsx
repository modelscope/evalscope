import { useState, useCallback, useMemo } from 'react'
import type { PredictionRow } from '@/api/types'
import { hasSystemPrompt, parseSystemUser } from './chat/chatHelpers'
import { MessageRow, SystemPromptRow, HeaderPerfChip } from './chat/MessageComponents'
import { StructuredMessages, TracedTimeline, buildStepGroups } from './chat/AgentTraceView'
import { EvalResultPanel } from './chat/EvalResultPanel'
import { selectChatPresentation } from '@/domain/predictions/chatPresentation'

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
      <MessageRow role="user" content={user || prediction.Input} />
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
        score={prediction.Score}
        metadata={prediction.Metadata}
        threshold={threshold}
        showPred={!!showPred}
      />
    </div>
  )
}
