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
    <div
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.5rem',
        padding: '0.35rem 0.6rem',
        background: 'var(--bg-deep)',
        border: '1px solid var(--border)',
        borderRadius: '0.4rem',
        fontSize: '0.72rem',
        fontFamily: 'var(--font-mono, monospace)',
        color: 'var(--text-muted)',
        marginTop: '0.4rem',
      }}
    >
      <Cpu size={12} style={{ color: 'var(--text-muted)', marginTop: 2, flexShrink: 0 }} />
      <span style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', flex: 1 }}>$ {cmd}</span>
      {event.latency_ms != null && (
        <span style={{ opacity: 0.6, whiteSpace: 'nowrap' }}>{fmtMs(event.latency_ms)}</span>
      )}
    </div>
  )
}

/* ─── LoopErrorRow ─────────────────────────────────────────── */

export function LoopErrorRow({ event }: { event: AgentTraceEvent }) {
  const msg = event.payload.message != null ? String(event.payload.message) : ''
  if (!msg) return null
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.5rem',
        padding: '0.4rem 0.6rem',
        background: 'var(--danger-bg)',
        border: '1px solid var(--danger-border, var(--danger))',
        borderRadius: '0.4rem',
        fontSize: '0.72rem',
        fontFamily: 'var(--font-mono, monospace)',
        color: 'var(--danger)',
        marginTop: '0.4rem',
      }}
    >
      <AlertTriangle size={12} style={{ marginTop: 2, flexShrink: 0 }} />
      <span style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', flex: 1 }}>{msg}</span>
    </div>
  )
}

/* ─── NudgeRow ─────────────────────────────────────────────── */

/** Compact nudge reminder row (system-injected when model didn't call tools). */
export function NudgeRow({ msg }: { msg: ChatMessage }) {
  const text = contentToText(msg.content)
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.4rem',
        padding: '0.35rem 0.6rem',
        background: 'var(--warning-bg)',
        border: '1px solid var(--warning-border)',
        borderRadius: '0.4rem',
        fontSize: '0.72rem',
        fontFamily: 'var(--font-mono, monospace)',
        color: 'var(--warning-text)',
        marginTop: '0.25rem',
      }}
    >
      <AlertTriangle size={12} style={{ flexShrink: 0 }} />
      <span style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', flex: 1 }}>{text}</span>
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
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 3,
        padding: '1px 6px',
        borderRadius: 3,
        background: 'transparent',
        border: `1px solid ${cfg.color}`,
        color: cfg.color,
        fontSize: '0.58rem',
        fontFamily: 'var(--font-mono, monospace)',
        fontWeight: 500,
        opacity: 0.85,
        whiteSpace: 'nowrap',
      }}
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
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
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
      style={{
        borderRadius: '0.6rem',
        background: highlighted ? 'var(--accent-dim)' : 'transparent',
        borderLeft: highlighted ? '2px solid var(--accent)' : '2px solid transparent',
        padding: highlighted ? '0.2rem 0 0.2rem 0.4rem' : '0.2rem 0',
        transition: 'background 0.3s, border-color 0.3s',
      }}
    >
      {/* Step header strip */}
      <button
        onClick={() => onStepClick(group.step)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          width: '100%',
          background: 'none',
          border: 'none',
          borderBottom: '1px dashed var(--border)',
          padding: '0.3rem 0 0.4rem 0',
          marginBottom: '0.5rem',
          cursor: 'pointer',
        }}
      >
        <span
          style={{
            fontSize: '0.7rem',
            fontWeight: 700,
            fontFamily: 'var(--font-mono, monospace)',
            color: highlighted ? 'var(--accent)' : 'var(--text-muted)',
            letterSpacing: '0.04em',
          }}
        >
          {t('trace.step')} {group.step}
        </span>
        {group.totalLatencyMs != null && (
          <span
            style={{
              fontSize: '0.65rem',
              fontFamily: 'var(--font-mono, monospace)',
              color: 'var(--text-muted)',
              opacity: 0.7,
            }}
          >
            {fmtMs(group.totalLatencyMs)}
          </span>
        )}
        {/* Event pills */}
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', flex: 1 }}>
          {group.traceEvents.map((ev, i) => (
            <TraceEventPill key={i} event={ev} />
          ))}
        </div>
      </button>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
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
