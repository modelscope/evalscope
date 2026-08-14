/**
 * Regression coverage for `textual_block` trajectories.
 *
 * A `textual_block` strategy (e.g. `swe_bench_backticks`, mirroring
 * mini-swe-agent) returns its observation as a plain **user** message, because the
 * models it targets do not speak the OpenAI `tool` role. The assistant turn also
 * carries no `tool_calls`: the call was parsed out of a fenced markdown block, so
 * it exists only as a `tool_call` trace event.
 *
 * That leaves the `tool_result` event as the sole link between a call and its
 * output. Before this was handled, the bash output resolved to no result at all
 * and fell through to the residual list, where it rendered as a *user turn* — the
 * command's output attributed to the person, not the tool.
 *
 * The fixture below is verbatim output from a real `AgentLoop` +
 * `SweBenchBackticksStrategy` run, not a hand-authored shape.
 */
import { describe, expect, it } from 'vitest'

import type { AgentTrace, ChatMessage } from '@/api/types'
import {
  buildStepGroups,
  buildTraceContext,
  resetMessageIdsOf,
  resolveResidualTools,
  resolveToolCalls,
} from './stepGroups'

/** Ids as emitted by the strategy: `sweb_*` for calls, 8-hex for messages. */
const CALL_ID = 'sweb_f6f8ace3'
const OBSERVATION_ID = 'd8d8b5c7'
const ASSISTANT_ID = '6d9ff106'

const messages = [
  { id: 'e6a265b3', role: 'system', content: 'You are a helpful assistant...' },
  { id: 'bc2cb69e', role: 'user', content: 'Fix the bug in src/.' },
  // No `tool_calls`: the call lives only in the trace.
  { id: ASSISTANT_ID, role: 'assistant', content: 'THOUGHT: list the repo.\n\n```mswea_bash_command\nls\n```' },
  // The observation: `user` role, and it names no call.
  {
    id: OBSERVATION_ID,
    role: 'user',
    content: '<returncode>0</returncode>\n<output>\nREADME.md\nsetup.py\nsrc/\n</output>',
  },
] as unknown as ChatMessage[]

const trace = {
  framework: 'native',
  strategy: 'swe_bench_backticks',
  events: [
    { step: 0, type: 'model_generate', message_id: ASSISTANT_ID, payload: {} },
    {
      step: 0,
      type: 'tool_call',
      message_id: ASSISTANT_ID,
      payload: { name: 'bash', arguments: { command: 'ls' }, id: CALL_ID },
    },
    {
      step: 0,
      type: 'tool_result',
      message_id: OBSERVATION_ID,
      latency_ms: 3.2,
      payload: { name: 'bash', id: CALL_ID, preview: 'README.md' },
    },
  ],
} as unknown as AgentTrace

/** Resolve step 0 exactly the way the timeline does. */
function resolveStepZero() {
  const groups = buildStepGroups(messages, trace)
  const ctx = buildTraceContext(messages, trace, groups)
  const step = groups.find((g) => g.step === 0)!
  const entries = resolveToolCalls(
    step.assistant,
    step.traceEvents.filter((e) => e.type === 'tool_call'),
    ctx,
  )
  const residual = resolveResidualTools(step, entries, resetMessageIdsOf(step), ctx.consumedToolMsgIds)
  return { step, entries, residual }
}

describe('textual_block trajectory', () => {
  it('reconstructs the call from the trace when the assistant declares none', () => {
    const { step, entries } = resolveStepZero()

    expect(step.assistant?.tool_calls).toBeUndefined()
    expect(entries).toHaveLength(1)
    expect(entries[0]).toMatchObject({ id: CALL_ID, function: 'bash', latencyMs: 3.2 })
  })

  it('inlines the observation under its call even though it names no call', () => {
    const { entries } = resolveStepZero()

    // The link comes from the tool_result event alone.
    expect(entries[0].result?.id).toBe(OBSERVATION_ID)
  })

  it('does not also leave the observation loose, where it would read as a user turn', () => {
    const { residual } = resolveStepZero()

    expect(residual).toEqual([])
  })
})

describe('function_calling trajectory', () => {
  // The ordinary path must keep resolving through the message's own tool_call_id;
  // the back-fill above is a fallback, not a replacement.
  const fcMessages = [
    { id: 'a1', role: 'assistant', content: '', tool_calls: [{ id: 'c1', function: 'bash', arguments: {} }] },
    { id: 't1', role: 'tool', content: 'ok', tool_call_id: 'c1' },
  ] as unknown as ChatMessage[]

  const fcTrace = {
    framework: 'native',
    events: [
      { step: 0, type: 'model_generate', message_id: 'a1', payload: {} },
      { step: 0, type: 'tool_call', message_id: 'a1', payload: { id: 'c1', name: 'bash' } },
      { step: 0, type: 'tool_result', message_id: 't1', latency_ms: 1, payload: { id: 'c1' } },
    ],
  } as unknown as AgentTrace

  it('resolves the result through the tool message itself', () => {
    const groups = buildStepGroups(fcMessages, fcTrace)
    const ctx = buildTraceContext(fcMessages, fcTrace, groups)
    const step = groups.find((g) => g.step === 0)!
    const entries = resolveToolCalls(step.assistant, [], ctx)

    expect(entries[0].result?.id).toBe('t1')
    expect(resolveResidualTools(step, entries, new Set(), ctx.consumedToolMsgIds)).toEqual([])
  })
})
