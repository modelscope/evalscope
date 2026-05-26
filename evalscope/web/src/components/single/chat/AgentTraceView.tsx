import React from 'react'
import {
  AlertTriangle,
  Cpu,
  Sparkles,
  Wrench,
  Check,
  Play,
  Square,
} from 'lucide-react'
import type { ChatMessage, AgentTrace, AgentTraceEvent, ToolCall } from '@/api/types'
import { useLocale } from '@/contexts/LocaleContext'
import { fmtMs } from '@/utils/formatUtils'
import { contentToText } from './chatHelpers'
import { type Role } from './roleConfig'
import { MessageRow, SystemPromptRow, HeaderPerfChip } from './MessageComponents'
import { type ToolCallEntry, ToolCallsGroup } from './ToolCallComponents'
import { bubbleAccent } from '@/components/ui/ChatBubble'

/* ─── EnvExecRow ───────────────────────────────────────────── */

export function EnvExecRow({ event }: { event: AgentTraceEvent }) {
  const cmd = typeof event.payload.command === 'string' ? event.payload.command : ''
  if (!cmd) return null
  return (
    <div className="flex items-start gap-2 px-[0.6rem] py-[0.35rem] bg-[var(--bg-deep)] border border-[var(--border)] rounded-[0.4rem] text-[0.72rem] font-mono text-[var(--text-muted)] mt-[0.4rem]">
      <Cpu size={12} className="text-[var(--text-muted)] mt-[2px] shrink-0" />
      <span className="whitespace-pre-wrap break-all flex-1">$ {cmd}</span>
      {event.latency_ms != null && (
        <span className="opacity-60 whitespace-nowrap">{fmtMs(event.latency_ms)}</span>
      )}
    </div>
  )
}

/* ─── LoopErrorRow ─────────────────────────────────────────── */

export function LoopErrorRow({ event }: { event: AgentTraceEvent }) {
  const { t } = useLocale()
  const msg = event.payload.message != null ? String(event.payload.message) : ''
  if (!msg) return null
  // Look up a human-readable label for known loop messages
  // (e.g. ``model_context_overflow``); fall back to the raw identifier
  // when no translation exists.
  const label = t(`trace.loopMessage.${msg}`)
  const display = label === `trace.loopMessage.${msg}` ? msg : label
  return (
    <div className="flex items-start gap-2 px-[0.6rem] py-[0.4rem] bg-[var(--danger-bg)] border border-[var(--danger-border)] rounded-[0.4rem] text-[0.72rem] font-mono text-[var(--danger)] mt-[0.4rem]">
      <AlertTriangle size={12} className="mt-[2px] shrink-0" />
      <span className="whitespace-pre-wrap break-all flex-1">{display}</span>
    </div>
  )
}

/* ─── NudgeRow ─────────────────────────────────────────────── */

/** Compact nudge reminder row (system-injected when model didn't call tools). */
export function NudgeRow({ msg }: { msg: ChatMessage }) {
  const text = contentToText(msg.content)
  return (
    <div className="flex items-center gap-[0.4rem] px-[0.6rem] py-[0.35rem] bg-[var(--warning-bg)] border border-[var(--warning-border)] rounded-[0.4rem] text-[0.72rem] font-mono text-[var(--warning-text)] mt-1">
      <AlertTriangle size={12} className="shrink-0" />
      <span className="whitespace-pre-wrap break-all flex-1">{text}</span>
    </div>
  )
}

/* ─── StructuredMessages ───────────────────────────────────── */

export function StructuredMessages({
  messages,
  highlightId,
}: {
  messages: ChatMessage[]
  highlightId?: string
}) {
  // Build id->message for tool_call_id resolution
  const byToolCallId = new Map<string, ChatMessage>()
  for (const m of messages) {
    if (m.role === 'tool' && m.tool_call_id) byToolCallId.set(m.tool_call_id, m)
  }

  const rendered: React.ReactNode[] = []
  const consumedToolIds = new Set<string>()

  messages.forEach((msg, idx) => {
    if (msg.role === 'system') {
      rendered.push(
        <SystemPromptRow
          key={idx}
          content={msg.content}
          msgId={msg.id}
          highlightId={highlightId}
        />
      )
      return
    }

    if (msg.role === 'user') {
      rendered.push(
        <MessageRow
          key={idx}
          role="user"
          content={msg.content}
          msgId={msg.id}
          highlightId={highlightId}
        />
      )
      return
    }

    if (msg.role === 'assistant') {
      const pm = msg.perf_metrics
      const headerPerf = pm ? (
        <HeaderPerfChip
          latency={pm.latency != null ? pm.latency * 1000 : null}
          ttft={pm.ttft}
          tpot={pm.tpot}
          inTok={pm.input_tokens}
          outTok={pm.output_tokens}
        />
      ) : undefined

      // Build tool call entries (resolve results via tool_call_id)
      const entries: ToolCallEntry[] = (msg.tool_calls ?? []).map(tc => {
        const result = byToolCallId.get(tc.id)
        if (result?.id) consumedToolIds.add(result.id)
        return {
          id: tc.id,
          function: tc.function,
          arguments: tc.arguments,
          result,
        }
      })

      rendered.push(
        <MessageRow
          key={idx}
          role="assistant"
          content={msg.content}
          msgId={msg.id}
          model={msg.model}
          highlightId={highlightId}
          headerExtra={headerPerf}
        >
          {entries.length > 0 && <ToolCallsGroup calls={entries} />}
        </MessageRow>
      )
      return
    }

    if (msg.role === 'tool') {
      // Skip tool messages already consumed by a ToolCallsGroup above
      if (msg.id && consumedToolIds.has(msg.id)) return
      rendered.push(
        <MessageRow
          key={idx}
          role="tool"
          content={msg.content}
          msgId={msg.id}
          highlightId={highlightId}
          toolError={msg.error ?? null}
          toolFunction={msg.function}
        />
      )
      return
    }
  })

  return <>{rendered}</>
}

/* ─── StepGroup / buildStepGroups ─────────────────────────── */

export interface StepGroup {
  step: number
  /** Pre-agent messages (system/user) — only for step -1. */
  preAgentMessages: ChatMessage[]
  assistant: ChatMessage | null
  tools: ChatMessage[]
  traceEvents: AgentTraceEvent[]
  totalLatencyMs: number | null
}

/** Cross-step linkage built once from the full message list and trace.
 *
 * Some recorders (e.g. the Claude Code external bridge) emit a tool_call on
 * step N but the matching tool_result on step N+1 (when observed in the next
 * request). Per-step lookups would then fail to inline the result under the
 * call. These globals let StepBlock resolve results regardless of the step
 * the result event landed on.
 */
export interface TraceContext {
  /** All tool messages indexed by their tool_call_id. */
  toolMsgByCallId: Map<string, ChatMessage>
  /** All tool_result trace events indexed by payload.id (= tool_call id). */
  toolResultEvByCallId: Map<string, AgentTraceEvent>
  /** Tool message ids already consumed as a result inside some assistant's
   *  tool_calls — should be excluded from any step's residualTools. */
  consumedToolMsgIds: Set<string>
}

export function buildTraceContext(
  messages: ChatMessage[],
  trace: AgentTrace,
  groups: StepGroup[]
): TraceContext {
  const toolMsgByCallId = new Map<string, ChatMessage>()
  for (const m of messages) {
    if (m.role === 'tool' && m.tool_call_id) {
      toolMsgByCallId.set(m.tool_call_id, m)
    }
  }

  const toolResultEvByCallId = new Map<string, AgentTraceEvent>()
  for (const ev of trace.events) {
    if (ev.type !== 'tool_result') continue
    const id = typeof ev.payload?.id === 'string' ? ev.payload.id : null
    if (id) toolResultEvByCallId.set(id, ev)
  }

  const consumedToolMsgIds = new Set<string>()
  for (const g of groups) {
    if (!g.assistant?.tool_calls) continue
    for (const tc of g.assistant.tool_calls) {
      const tm = toolMsgByCallId.get(tc.id)
      if (tm?.id) consumedToolMsgIds.add(tm.id)
    }
  }

  return { toolMsgByCallId, toolResultEvByCallId, consumedToolMsgIds }
}

export function buildStepGroups(messages: ChatMessage[], trace: AgentTrace): StepGroup[] {
  const messageById = new Map<string, ChatMessage>()
  for (const m of messages) if (m.id) messageById.set(m.id, m)

  const stepEvents = new Map<number, AgentTraceEvent[]>()
  for (const ev of trace.events) {
    if (!stepEvents.has(ev.step)) stepEvents.set(ev.step, [])
    stepEvents.get(ev.step)!.push(ev)
  }

  const referencedIds = new Set<string>()
  for (const ev of trace.events) if (ev.message_id) referencedIds.add(ev.message_id)

  const preAgent: ChatMessage[] = []
  for (const m of messages) {
    if (m.id && referencedIds.has(m.id)) break
    preAgent.push(m)
  }

  const groups: StepGroup[] = []
  if (preAgent.length > 0) {
    groups.push({
      step: -1,
      preAgentMessages: preAgent,
      assistant: null,
      tools: [],
      traceEvents: [],
      totalLatencyMs: null,
    })
  }

  const sortedSteps = Array.from(stepEvents.keys()).sort((a, b) => a - b)
  for (const step of sortedSteps) {
    const events = stepEvents.get(step)!
    let assistant: ChatMessage | null = null
    const tools: ChatMessage[] = []
    const seenToolIds = new Set<string>()
    for (const ev of events) {
      if (!ev.message_id) continue
      const msg = messageById.get(ev.message_id)
      if (!msg) continue
      if (msg.role === 'assistant') {
        if (!assistant) assistant = msg
      } else if (msg.role === 'tool' || msg.role === 'user') {
        if (msg.id && !seenToolIds.has(msg.id)) {
          seenToolIds.add(msg.id)
          tools.push(msg)
        }
      }
    }
    let totalLatency: number | null = null
    for (const e of events) {
      if (e.latency_ms != null) totalLatency = (totalLatency ?? 0) + e.latency_ms
    }
    groups.push({
      step,
      preAgentMessages: [],
      assistant,
      tools,
      traceEvents: events,
      totalLatencyMs: totalLatency,
    })
  }

  return groups
}

/* ─── TraceEventPill ───────────────────────────────────────── */

export function TraceEventPill({ event }: { event: AgentTraceEvent }) {
  const { t } = useLocale()
  const cfg = (() => {
    switch (event.type) {
      case 'model_generate':
        return { Icon: Sparkles, color: bubbleAccent('bot'), labelKey: 'trace.modelGenerate' }
      case 'tool_call':
        return { Icon: Wrench, color: bubbleAccent('user'), labelKey: 'trace.toolCall' }
      case 'tool_result':
        return { Icon: Wrench, color: bubbleAccent('tool'), labelKey: 'trace.toolResult' }
      case 'env_exec':
        return { Icon: Cpu, color: 'var(--text-muted)', labelKey: 'trace.envExec' }
      case 'error':
        return { Icon: AlertTriangle, color: 'var(--danger)', labelKey: 'trace.error' }
      case 'nudge':
        return { Icon: AlertTriangle, color: 'var(--warning-color)', labelKey: 'trace.nudge' }
      case 'submit':
        return { Icon: Check, color: 'var(--success)', labelKey: 'trace.submit' }
      case 'run_start':
        return { Icon: Play, color: 'var(--text-muted)', labelKey: 'trace.runStart' }
      case 'run_end':
        return { Icon: Square, color: 'var(--text-muted)', labelKey: 'trace.runEnd' }
      default:
        return { Icon: Wrench, color: 'var(--text-muted)', labelKey: event.type }
    }
  })()
  const Icon = cfg.Icon
  const label = t(cfg.labelKey)
  return (
    <span
      className="inline-flex items-center gap-[3px] px-[6px] py-[1px] rounded-[3px] bg-transparent border text-[0.58rem] font-mono font-medium opacity-85 whitespace-nowrap"
      style={{ borderColor: cfg.color, color: cfg.color }}
    >
      <Icon size={9} />
      {label === cfg.labelKey ? event.type : label}
    </span>
  )
}

/* ─── StepBlock ────────────────────────────────────────────── */

export function StepBlock({
  group,
  highlightId,
  highlighted,
  onStepClick,
  ctx,
}: {
  group: StepGroup
  highlightId?: string
  highlighted: boolean
  onStepClick: (step: number) => void
  ctx?: TraceContext
}) {
  const { t } = useLocale()

  // Pre-agent messages (system/user) — render as-is, no wrapper
  if (group.step === -1) {
    return (
      <div className="flex flex-col gap-2">
        {group.preAgentMessages.map((msg, idx) =>
          msg.role === 'system' ? (
            <SystemPromptRow
              key={idx}
              content={msg.content}
              msgId={msg.id}
              highlightId={highlightId}
            />
          ) : (
            <MessageRow
              key={idx}
              role={msg.role as Role}
              content={msg.content}
              msgId={msg.id}
              highlightId={highlightId}
            />
          )
        )}
      </div>
    )
  }

  const modelGen = group.traceEvents.find(e => e.type === 'model_generate')
  const toolCallEvents = group.traceEvents.filter(e => e.type === 'tool_call')
  const toolResultEvents = group.traceEvents.filter(e => e.type === 'tool_result')
  const envExecs = group.traceEvents.filter(e => e.type === 'env_exec')
  const loopErrors = group.traceEvents.filter(e => e.type === 'error')

  // Build assistant header perf info from model_generate event (preferred)
  const mg = modelGen
  const stopReason = mg?.payload?.stop_reason ? String(mg.payload.stop_reason) : undefined
  const modelGenLatMs = mg?.latency_ms ?? null
  const inTok = mg?.token_usage?.input ?? null
  const outTok = mg?.token_usage?.output ?? null

  // Fallback to assistant.perf_metrics
  const ap = group.assistant?.perf_metrics
  const headerLat = modelGenLatMs ?? (ap?.latency != null ? ap.latency * 1000 : null)
  const headerIn = inTok ?? ap?.input_tokens ?? null
  const headerOut = outTok ?? ap?.output_tokens ?? null

  const headerPerf = (
    <HeaderPerfChip
      latency={headerLat}
      ttft={ap?.ttft}
      tpot={ap?.tpot}
      inTok={headerIn}
      outTok={headerOut}
      stopReason={stopReason}
    />
  )

  // Build ToolCallEntry[] — prefer assistant.tool_calls, fallback to tool_call events.
  // Use global ctx maps when available so results emitted on a different step
  // (e.g. Claude Code bridge emits tool_result on step+1) still get linked.
  const toolResultByCallId = ctx?.toolResultEvByCallId ?? (() => {
    const m = new Map<string, AgentTraceEvent>()
    for (const ev of toolResultEvents) {
      const id = typeof ev.payload.id === 'string' ? ev.payload.id : null
      if (id) m.set(id, ev)
    }
    return m
  })()

  const toolMsgByCallId = ctx?.toolMsgByCallId ?? (() => {
    const m = new Map<string, ChatMessage>()
    for (const tm of group.tools) {
      if (tm.tool_call_id) m.set(tm.tool_call_id, tm)
    }
    // Also link via trace tool_result events (for mini-swe where observations
    // are ChatMessageUser without tool_call_id).
    for (const ev of toolResultEvents) {
      const callId = typeof ev.payload.id === 'string' ? ev.payload.id : null
      if (callId && !m.has(callId) && ev.message_id) {
        const msg = group.tools.find(t => t.id === ev.message_id)
        if (msg) m.set(callId, msg)
      }
    }
    return m
  })()

  let entries: ToolCallEntry[] = []
  if (group.assistant?.tool_calls && group.assistant.tool_calls.length > 0) {
    entries = group.assistant.tool_calls.map((tc: ToolCall) => {
      const resultEv = toolResultByCallId.get(tc.id)
      return {
        id: tc.id,
        function: tc.function,
        arguments: tc.arguments,
        result: toolMsgByCallId.get(tc.id),
        latencyMs: resultEv?.latency_ms ?? null,
      }
    })
  } else if (toolCallEvents.length > 0) {
    // Fallback: reconstruct from tool_call events
    entries = toolCallEvents.map(ev => {
      const id = typeof ev.payload.id === 'string' ? ev.payload.id : ''
      const name = typeof ev.payload.name === 'string' ? ev.payload.name : ''
      const args = ev.payload.arguments
      const resultEv = toolResultByCallId.get(id)
      return {
        id,
        function: name,
        arguments: args,
        result: id ? toolMsgByCallId.get(id) : undefined,
        latencyMs: resultEv?.latency_ms ?? null,
      }
    })
  }

  // Residual tool messages not linked to any call (rare; textual_block mode).
  // Exclude this step's own linked results AND any tool message consumed by
  // another step's assistant.tool_calls (cross-step tool_result placement).
  const linkedToolIds = new Set<string>()
  for (const e of entries) if (e.result?.id) linkedToolIds.add(e.result.id)
  const residualTools = group.tools.filter(m => {
    if (!m.id) return true
    if (linkedToolIds.has(m.id)) return false
    if (ctx?.consumedToolMsgIds.has(m.id)) return false
    return true
  })

  return (
    <div
      className={[
        'rounded-[0.6rem] border-l-2 transition-[background-color,border-color] duration-300',
        highlighted
          ? 'bg-[var(--accent-dim)] border-[var(--accent)] py-[0.2rem] pl-[0.4rem] pr-0'
          : 'bg-transparent border-transparent py-[0.2rem] px-0',
      ].join(' ')}
    >
      {/* Step header strip */}
      <button
        onClick={() => onStepClick(group.step)}
        className="flex items-center gap-2 w-full bg-transparent border-0 border-b border-dashed border-[var(--border)] pt-[0.3rem] pb-[0.4rem] mb-2 cursor-pointer"
      >
        <span
          className={[
            'text-[0.7rem] font-bold font-mono tracking-[0.04em]',
            highlighted ? 'text-[var(--accent)]' : 'text-[var(--text-muted)]',
          ].join(' ')}
        >
          {t('trace.step')} {group.step}
        </span>
        {group.totalLatencyMs != null && (
          <span className="text-[0.65rem] font-mono text-[var(--text-muted)] opacity-70">
            {fmtMs(group.totalLatencyMs)}
          </span>
        )}
        {/* Event pills */}
        <div className="flex flex-wrap flex-1 gap-1">
          {group.traceEvents.map((ev, i) => (
            <TraceEventPill key={i} event={ev} />
          ))}
        </div>
      </button>

      <div className="flex flex-col gap-2">
        {/* Assistant row */}
        {group.assistant && (
          <MessageRow
            role="assistant"
            content={group.assistant.content}
            msgId={group.assistant.id}
            model={group.assistant.model}
            highlightId={highlightId}
            headerExtra={headerPerf}
          >
            {entries.length > 0 && <ToolCallsGroup calls={entries} />}
          </MessageRow>
        )}

        {/* env_exec entries */}
        {envExecs.map((ev, i) => (
          <EnvExecRow key={`env-${i}`} event={ev} />
        ))}

        {/* Loop-level errors (not tied to a tool_call) */}
        {loopErrors.map((ev, i) => (
          <LoopErrorRow key={`err-${i}`} event={ev} />
        ))}

        {/* Any residual tool messages not linked via tool_call_id */}
        {residualTools.map((m, i) => {
          // Nudge messages: role='user' AND confirmed by a 'nudge' trace event
          const isNudge = m.role === 'user' && group.traceEvents.some(
            ev => ev.type === 'nudge' && ev.message_id === m.id
          )
          if (isNudge) {
            return <NudgeRow key={`nudge-${m.id ?? i}`} msg={m} />
          }
          return (
            <MessageRow
              key={`residual-${m.id ?? i}`}
              role={m.role === 'tool' ? 'tool' : 'user'}
              content={m.content}
              msgId={m.id}
              highlightId={highlightId}
              toolError={m.error ?? null}
              toolFunction={m.function}
            />
          )
        })}
      </div>
    </div>
  )
}

/* ─── TracedTimeline ───────────────────────────────────────── */

export function TracedTimeline({
  groups,
  messages,
  trace,
  highlightStep,
  highlightId,
  onStepClick,
}: {
  groups: StepGroup[]
  messages: ChatMessage[]
  trace: AgentTrace
  highlightStep: number | null
  highlightId?: string
  onStepClick: (step: number) => void
}) {
  const preGroup = groups.find(g => g.step === -1)
  const agentGroups = groups.filter(g => g.step >= 0)
  const ctx = React.useMemo(
    () => buildTraceContext(messages, trace, groups),
    [messages, trace, groups]
  )

  return (
    <div className="flex flex-col gap-[0.6rem]">
      {preGroup && (
        <StepBlock
          group={preGroup}
          highlightId={highlightId}
          highlighted={false}
          onStepClick={onStepClick}
          ctx={ctx}
        />
      )}
      {agentGroups.map((g) => {
        const isActive = highlightStep === g.step
        return (
          <StepBlock
            key={g.step}
            group={g}
            highlightId={highlightId}
            highlighted={isActive}
            onStepClick={onStepClick}
            ctx={ctx}
          />
        )
      })}
    </div>
  )
}
