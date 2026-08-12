import { describe, expect, it } from 'vitest'

import type { AgentTraceEvent, ChatMessage } from '@/api/types'
import {
  buildToolMessageIndex,
  resetMessageIdsOf,
  resolveResidualTools,
  resolveToolCalls,
  type StepGroup,
  type ToolCallEntry,
} from './stepGroups'

function assistant(calls: { id: string; function: string; arguments?: unknown }[]): ChatMessage {
  return {
    role: 'assistant',
    content: '',
    tool_calls: calls.map((call) => ({ arguments: {}, ...call })),
  } as ChatMessage
}

function toolMessage(id: string, callId: string | string[]): ChatMessage {
  return { id, role: 'tool', content: 'observed', tool_call_id: callId } as ChatMessage
}

function event(partial: Partial<AgentTraceEvent>): AgentTraceEvent {
  return { type: 'tool_call', step: 0, payload: {}, ...partial } as AgentTraceEvent
}

function group(partial: Partial<StepGroup>): StepGroup {
  return {
    step: 0,
    preAgentMessages: [],
    assistant: null,
    tools: [],
    traceEvents: [],
    totalLatencyMs: null,
    ...partial,
  }
}

describe('buildToolMessageIndex', () => {
  it('indexes a tool message by the call it answers', () => {
    const message = toolMessage('m1', 'call-1')
    expect(buildToolMessageIndex([message]).get('call-1')).toBe(message)
  })

  it('indexes a message that answers several calls under each of them', () => {
    const message = toolMessage('m1', ['call-1', 'call-2'])
    const index = buildToolMessageIndex([message])

    expect(index.get('call-1')).toBe(message)
    expect(index.get('call-2')).toBe(message)
  })

  it('ignores messages that are not tool results', () => {
    const user = { id: 'u1', role: 'user', content: 'hi' } as ChatMessage
    expect(buildToolMessageIndex([user]).size).toBe(0)
  })
})

describe('resolveToolCalls', () => {
  it("treats the assistant's own tool_calls as authoritative", () => {
    const result = toolMessage('m1', 'call-1')
    const entries = resolveToolCalls(
      assistant([{ id: 'call-1', function: 'search' }]),
      [event({ payload: { id: 'other', name: 'ignored' } })],
      { toolMsgByCallId: new Map([['call-1', result]]) },
    )

    expect(entries).toHaveLength(1)
    expect(entries[0].function).toBe('search')
    expect(entries[0].result).toBe(result)
  })

  it('reconstructs calls from trace events when the turn declares none', () => {
    const entries = resolveToolCalls(
      assistant([]),
      [event({ payload: { id: 'call-1', name: 'bash', arguments: { cmd: 'ls' } } })],
      { toolMsgByCallId: new Map() },
    )

    expect(entries).toEqual<ToolCallEntry[]>([
      { id: 'call-1', function: 'bash', arguments: { cmd: 'ls' }, result: undefined, latencyMs: null },
    ])
  })

  it('links a result recorded on a different step than its call', () => {
    // The Claude Code bridge emits tool_result on step+1; resolution goes through
    // the cross-step index precisely so this still pairs up.
    const result = toolMessage('m1', 'call-1')
    const entries = resolveToolCalls(
      assistant([{ id: 'call-1', function: 'search' }]),
      [],
      {
        toolMsgByCallId: new Map([['call-1', result]]),
        toolResultEvByCallId: new Map([['call-1', event({ type: 'tool_result', step: 1, latency_ms: 42 })]]),
      },
    )

    expect(entries[0].result).toBe(result)
    expect(entries[0].latencyMs).toBe(42)
  })

  it('reports no latency when no result event was recorded', () => {
    const entries = resolveToolCalls(assistant([{ id: 'call-1', function: 'f' }]), [], {
      toolMsgByCallId: new Map(),
    })

    expect(entries[0].latencyMs).toBeNull()
    expect(entries[0].result).toBeUndefined()
  })

  it('returns nothing for a turn with neither calls nor events', () => {
    expect(resolveToolCalls(null, [], { toolMsgByCallId: new Map() })).toEqual([])
  })
})

describe('resolveResidualTools', () => {
  it('drops a message this step already inlined under a call', () => {
    const result = toolMessage('m1', 'call-1')
    const entries: ToolCallEntry[] = [
      { id: 'call-1', function: 'f', arguments: {}, result },
    ]

    const residual = resolveResidualTools(group({ tools: [result] }), entries, new Set())

    expect(residual).toEqual([])
  })

  it('drops a message another step claimed', () => {
    // Cross-step consumption has to be known here, or the same observation would
    // be rendered twice: once inlined on its own step, once loose on this one.
    const claimed = toolMessage('m1', 'call-1')

    const residual = resolveResidualTools(
      group({ tools: [claimed] }),
      [],
      new Set(),
      new Set(['m1']),
    )

    expect(residual).toEqual([])
  })

  it('drops an environment reset message, which renders separately', () => {
    const reset = toolMessage('m1', 'call-1')
    const residual = resolveResidualTools(group({ tools: [reset] }), [], new Set(['m1']))

    expect(residual).toEqual([])
  })

  it('keeps an unclaimed message so it is never silently lost', () => {
    const loose = toolMessage('m9', 'call-9')
    const residual = resolveResidualTools(group({ tools: [loose] }), [], new Set())

    expect(residual).toEqual([loose])
  })

  it('keeps a message with no id, which cannot be matched to any call', () => {
    const anonymous = { role: 'tool', content: 'stdout' } as ChatMessage
    const residual = resolveResidualTools(group({ tools: [anonymous] }), [], new Set())

    expect(residual).toEqual([anonymous])
  })
})

describe('resetMessageIdsOf', () => {
  it('collects the message ids of this step\'s env_reset events', () => {
    const ids = resetMessageIdsOf(group({
      traceEvents: [
        event({ type: 'env_reset', message_id: 'm1' }),
        event({ type: 'tool_call', message_id: 'm2' }),
        event({ type: 'env_reset' }),
      ],
    }))

    expect([...ids]).toEqual(['m1'])
  })
})
