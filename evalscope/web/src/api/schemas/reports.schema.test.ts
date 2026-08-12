import { describe, expect, it } from 'vitest'

import { loadFixture } from '@/test/loadFixture'
import { agentTraceSchema, chatMessageSchema, loadReportResponseSchema } from './reports.schema'

describe('loadReportResponseSchema real report compatibility', () => {
  const fixture = loadFixture<unknown>('report-real-single-sample')

  it('accepts single-sample null standard deviations and reports without perf metrics', () => {
    const result = loadReportResponseSchema.safeParse(fixture)

    expect(result.success).toBe(true)
    if (!result.success) return
    expect(result.data.report_list[0].perf_metrics?.summary.latency.std).toBeNull()
    expect(result.data.report_list[1].perf_metrics).toBeNull()
  })

  it('continues to reject null for defined statistics such as the mean', () => {
    const invalid = structuredClone(fixture) as {
      report_list: Array<{ perf_metrics?: { summary: { latency: { mean: number | null } } } }>
    }
    invalid.report_list[0].perf_metrics!.summary.latency.mean = null

    expect(loadReportResponseSchema.safeParse(invalid).success).toBe(false)
  })

  it('accepts the BrowserGym environment reset event emitted by the backend', () => {
    const result = agentTraceSchema.safeParse({
      strategy: 'function_calling',
      environment: 'browsergym',
      max_steps: 10,
      events: [
        {
          step: 0,
          timestamp: 1_700_000_000,
          type: 'env_reset',
          message_id: 'browser-observation-0',
          latency_ms: 120,
          payload: {
            backend: 'browsergym',
            reward: 0,
            done: false,
            screenshot_path: '/tmp/miniwob/step-000.png',
          },
        },
      ],
    })

    expect(result.success).toBe(true)
  })

  it('accepts tool-linked environment attachments emitted as user messages', () => {
    const result = chatMessageSchema.safeParse({
      id: 'browser-observation-1',
      role: 'user',
      content: [{ type: 'image', image: '/tmp/miniwob/step-001.png' }],
      tool_call_id: ['browser-call-0'],
      metadata: {
        reward: 1,
        done: true,
        screenshot_path: '/tmp/miniwob/step-001.png',
      },
    })

    expect(result.success).toBe(true)
  })

  it('accepts a reasoning block with a null reasoning_tokens, as emitted when the backend has no token count', () => {
    // ContentReasoning.reasoning_tokens is `Optional[int] = None` on the Python side, which
    // serializes as an explicit JSON `null` rather than an absent key. Reasoning models that
    // don't report a token count (e.g. via completion_tokens_details) hit this on every turn.
    const result = chatMessageSchema.safeParse({
      role: 'assistant',
      content: [
        { type: 'reasoning', reasoning: 'thinking it through...', reasoning_tokens: null },
        { type: 'text', text: 'final answer' },
      ],
    })

    expect(result.success).toBe(true)
  })
})
