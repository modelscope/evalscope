/**
 * Agent-trace grouping (pure logic, no rendering).
 *
 * A trace arrives as a flat event list plus the messages those events refer to.
 * Turning that into the per-step structure the timeline renders — which
 * assistant turn opened a step, which tool results belong to it, and which
 * messages preceded the agent entirely — is pure data work, so it lives here
 * and is exercised directly rather than through the timeline component.
 */

import type { AgentTrace, AgentTraceEvent, ChatMessage } from '@/api/types'

/** Tool-call ids a message answers, normalised to an array. */
export function toolCallIds(message: ChatMessage): string[] {
  if (Array.isArray(message.tool_call_id)) return message.tool_call_id
  return message.tool_call_id ? [message.tool_call_id] : []
}

/**
 * Whether a message is an environment attachment rather than a user turn.
 *
 * Some recorders deliver a tool observation on the `user` role while still
 * tagging it with the call it answers; treating it as a user turn would break
 * the step it belongs to out of the timeline.
 */
export function isEnvironmentAttachment(message: ChatMessage): boolean {
  return message.role === 'user' && toolCallIds(message).length > 0
}

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
    if (m.role === 'tool' && typeof m.tool_call_id === 'string') {
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

  const groupByStep = new Map(groups.filter(group => group.step >= 0).map(group => [group.step, group]))
  const stepByToolCallId = new Map<string, number>()
  for (const [step, events] of stepEvents) {
    for (const event of events) {
      const callId = event.type === 'tool_call' && typeof event.payload.id === 'string'
        ? event.payload.id
        : null
      if (callId) stepByToolCallId.set(callId, step)
    }
  }
  for (const message of messages) {
    if (!isEnvironmentAttachment(message)) continue
    const step = toolCallIds(message)
      .map(callId => stepByToolCallId.get(callId))
      .find(candidate => candidate !== undefined)
    if (step === undefined) continue
    const group = groupByStep.get(step)
    if (!group || group.tools.some(tool => tool.id === message.id)) continue
    group.tools.push(message)
  }

  return groups
}
